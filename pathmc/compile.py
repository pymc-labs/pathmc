#   Copyright 2025 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Structural equation compiler: Spec + data -> pm.Model.

Builds a **generative** PyMC model where all endogenous variables are
free random variables (not observed). Exogenous inputs use ``pm.Data``,
linear predictors are tracked as ``pm.Deterministic("mu_{var}", ...)``,
and each endogenous variable is emitted as ``pm.Normal("{var}", ...)``.

The caller uses ``pm.observe()`` to condition the free RVs on observed
data for estimation, and ``pm.do()`` on the generative model for
interventional simulation.

Regressions are compiled in topological order so downstream equations
wire through upstream free RVs, enabling PyMC-native do() interventions
via graph surgery.

Panel models with temporal dependencies (adstock transforms or lag
terms) are compiled using ``pytensor.scan`` so that the generative model
encodes the full temporal structure. This allows ``pm.do()`` to handle
panel interventions natively — no separate simulation engine needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import narwhals.stable.v1 as nw
import networkx as nx
import numpy as np
import patsy
import pymc as pm

from pathmc.graph import GraphInfo
from pathmc.panel import PanelInfo
from pathmc.parse import Regression, Spec, Term, TransformCall
from pathmc.transforms import get_transform

__all__: list[str] = []


@dataclass
class PanelScanInfo:
    """Metadata stored on a scan-compiled panel model.

    Allows the caller to reshape flat observation arrays into the
    ``(n_times, n_units)`` layout expected by the scan model.
    """

    sort_idx: np.ndarray
    reverse_idx: np.ndarray
    n_units: int
    n_times: int
    unit_labels: list[str] = field(default_factory=list)
    time_values: list = field(default_factory=list)


@dataclass(frozen=True)
class PredictorSlot:
    """One slot in a linear predictor (one column of the design matrix).

    Classifies each predictor term by its structural kind so that the
    shared ``build_mu`` loop can delegate tensor resolution to a
    path-specific resolver.
    """

    name: str
    coeff_type: Literal["free", "fixed"]
    coeff_value: float | None = None
    kind: Literal["intercept", "plain", "interaction", "transform", "lag"] = "plain"
    lag_of: str | None = None
    interaction_parts: tuple[str, ...] | None = None
    transform: TransformCall | None = None


@dataclass
class MuSpec:
    """Symbolic description of a linear predictor for one equation.

    A pure representation of the linear-predictor structure that
    decouples equation logic from tensor resolution.  Both the
    cross-sectional and scan compilation paths consume ``MuSpec``
    via ``build_mu``, each providing their own resolver.
    """

    lhs: str
    slots: list[PredictorSlot]


def get_predictor_columns(reg: Regression) -> list[str]:
    """Return *all* predictor column names for a regression equation.

    Includes both free and fixed-coefficient predictors.

    Parameters
    ----------
    reg : Regression
        Parsed regression with terms and intercept flag.

    Returns
    -------
    list[str]
        Column names including ``"Intercept"`` when applicable.
    """
    cols: list[str] = []
    if reg.has_intercept:
        cols.append("Intercept")
    cols.extend(t.variable for t in reg.terms)
    return cols


def get_free_predictor_columns(reg: Regression) -> list[str]:
    """Return predictor column names that have free (estimated) coefficients.

    Fixed-value terms (e.g. ``1*X``) are excluded.
    """
    cols: list[str] = []
    if reg.has_intercept:
        cols.append("Intercept")
    cols.extend(t.variable for t in reg.terms if t.fixed_value is None)
    return cols


def get_fixed_coefficients(reg: Regression) -> dict[str, float]:
    """Return a mapping of predictor name -> fixed coefficient value."""
    return {t.variable: t.fixed_value for t in reg.terms if t.fixed_value is not None}


def _term_base_vars(term: Term) -> list[str]:
    """Return the base variable names a term depends on.

    For interaction terms, returns the constituent variables.
    For plain terms, returns a single-element list.
    """
    if term.interaction_of is not None:
        return list(term.interaction_of)
    return [term.variable]


# ---------------------------------------------------------------------------
# MuSpec construction
# ---------------------------------------------------------------------------


def build_mu_specs(spec: Spec) -> dict[str, MuSpec]:
    """Convert Spec regressions into MuSpec intermediate representations.

    Pure transformation with no PyMC dependency.  Each ``Regression``
    becomes a ``MuSpec`` whose ``PredictorSlot`` list classifies every
    predictor column by structural kind (intercept, plain variable,
    interaction, transform, or lag).

    Parameters
    ----------
    spec : Spec
        Parsed model specification.

    Returns
    -------
    dict[str, MuSpec]
        Mapping from endogenous variable name to its ``MuSpec``.
    """
    result: dict[str, MuSpec] = {}

    for reg in spec.regressions:
        slots: list[PredictorSlot] = []

        if reg.has_intercept:
            slots.append(
                PredictorSlot(
                    name="Intercept",
                    coeff_type="free",
                    kind="intercept",
                )
            )

        for term in reg.terms:
            coeff_type: Literal["free", "fixed"] = (
                "fixed" if term.fixed_value is not None else "free"
            )

            if term.transform is not None:
                kind: Literal[
                    "intercept", "plain", "interaction", "transform", "lag"
                ] = "transform"
            elif term.interaction_of is not None:
                kind = "interaction"
            elif term.lag_of is not None:
                kind = "lag"
            else:
                kind = "plain"

            slots.append(
                PredictorSlot(
                    name=term.variable,
                    coeff_type=coeff_type,
                    coeff_value=term.fixed_value,
                    kind=kind,
                    lag_of=term.lag_of,
                    interaction_parts=term.interaction_of,
                    transform=term.transform,
                )
            )

        result[reg.lhs] = MuSpec(lhs=reg.lhs, slots=slots)

    return result


def build_design_matrix(reg: Regression, data: nw.DataFrame) -> nw.DataFrame:
    """Build a patsy design matrix for a single regression equation.

    Parameters
    ----------
    reg : Regression
        Parsed regression with terms and intercept flag.
    data : nw.DataFrame
        Observed data containing the predictor columns.

    Returns
    -------
    nw.DataFrame
        Design matrix with named columns (including ``Intercept`` when
        applicable), backed by the same native backend as *data*. For
        equations with latent (unobserved) parents, returns a frame with
        correct column names but NaN for the latent columns.

    Notes
    -----
    The patsy path materialises *data* to pandas via ``data.to_pandas()``
    because ``patsy.dmatrix`` only understands pandas frames; the result is
    rewrapped to the input backend with ``nw.from_dict(...)``. Do not remove
    the ``.to_pandas()`` call — polars (and other non-pandas) inputs rely on it.
    """
    rhs_parts = [t.variable for t in reg.terms]
    n = len(data)

    missing: list[str] = []
    for term in reg.terms:
        for v in _term_base_vars(term):
            if v not in data.columns and v not in missing:
                missing.append(v)

    if missing:
        columns: dict[str, np.ndarray] = {}
        if reg.has_intercept:
            columns["Intercept"] = np.ones(n)
        for term in reg.terms:
            v = term.variable
            if term.interaction_of is not None:
                product = np.ones(n)
                for part in term.interaction_of:
                    if part in data.columns:
                        product = product * data[part].to_numpy().astype(float)
                    else:
                        product = product * np.nan
                columns[v] = product
            elif v in data.columns:
                columns[v] = data[v].to_numpy().astype(float)
            else:
                columns[v] = np.full(n, np.nan)
        return nw.from_dict(
            {c: columns[c] for c in get_predictor_columns(reg)},
            backend=data.implementation,
        )

    if reg.has_intercept:
        formula_str = " + ".join(rhs_parts)
    else:
        formula_str = "0 + " + " + ".join(rhs_parts)

    dm = patsy.dmatrix(formula_str, data=data.to_pandas(), return_type="dataframe")
    return nw.from_dict(
        {str(col): dm[col].to_numpy() for col in dm.columns},
        backend=data.implementation,
    )


def compile_to_pymc(
    spec: Spec,
    data: nw.DataFrame,
    design_matrices: dict[str, nw.DataFrame],
    families: dict[str, str] | None = None,
    panel_info: PanelInfo | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    graph_info: GraphInfo | None = None,
    priors: dict[str, Any] | None = None,
) -> pm.Model:
    """Compile a structural specification into a generative PyMC model.

    All endogenous variables are emitted as **free random variables**.
    The caller should use ``pm.observe()`` to condition on observed data
    for estimation, and ``pm.do()`` on this generative model for
    interventional simulation.

    Downstream equations wire through the upstream free RV (not data
    columns), so ``pm.do()`` naturally propagates interventions through
    the causal chain.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    data : nw.DataFrame
        Observed data.
    design_matrices : dict[str, nw.DataFrame]
        Pre-built design matrices keyed by endogenous variable name.
    families : dict[str, str] | None
        Per-variable distribution families. Defaults to ``"gaussian"``
        for all variables.
    panel_info : PanelInfo | None
        Panel metadata for hierarchical models.
    pooling : str | dict | None
        ``"partial"`` for random intercepts. Dict for random slopes.
    latent : set[str] | None
        Endogenous variables with no observed data (deterministic mediators).
    graph_info : GraphInfo | None
        Pre-built graph info. If ``None``, built from *spec*.
    priors : dict[str, Any] | None
        Custom prior configuration mapping parameter names to ``Prior``
        objects from ``pymc_extras``. If ``None``, sensible defaults are
        used. See :func:`pathmc.priors.default_priors` for the full
        list of parameter keys.

    Returns
    -------
    pm.Model
        Generative PyMC model (all endogenous vars are free RVs).
        Use ``pm.observe()`` to condition on data before sampling.

    Raises
    ------
    ValueError
        If ``~~`` is used between non-Gaussian variables.
    """
    from pathmc.priors import _ensure_dims, default_priors

    if families is None:
        families = {}
    if latent is None:
        latent = set()

    if priors is None:
        priors = default_priors(spec, families, pooling, latent)

    if graph_info is None:
        from pathmc.graph import build_graph

        graph_info = build_graph(spec, latent=latent)

    _validate_residual_cov_families(spec, families)
    _warn_partial_pooling_intercept(spec, pooling)

    if panel_info is not None and _has_temporal_deps(spec, graph_info):
        _validate_scan_non_gaussian_intermediaries(spec, families, latent)
        return _compile_scan_panel(
            spec=spec,
            data=data,
            design_matrices=design_matrices,
            families=families,
            panel_info=panel_info,
            pooling=pooling,
            latent=latent,
            graph_info=graph_info,
            priors=priors,
        )

    block_vars, blocks = _identify_residual_blocks(spec)

    has_random_intercepts = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    unit_idx: np.ndarray | None = None

    reg_by_lhs = {r.lhs: r for r in spec.regressions}

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        free_cols = get_free_predictor_columns(reg)
        if free_cols:
            coords[f"{reg.lhs}_predictors"] = free_cols

    if has_random_intercepts and panel_info is not None:
        coords["unit"] = panel_info.unit_labels
        unit_idx = _build_unit_index(data, panel_info)

    transform_map = _build_transform_map(spec)
    mu_specs = build_mu_specs(spec)

    sparse_data: dict[str, np.ma.MaskedArray] = {}
    for reg in spec.regressions:
        v = reg.lhs
        if v not in latent and v in data.columns:
            vals = np.asarray(data[v].to_numpy(), dtype=float)
            if np.isnan(vals).any():
                sparse_data[v] = np.ma.masked_invalid(vals)

    with pm.Model(coords=coords) as pymc_model:
        import pytensor.tensor as pt

        transform_param_rvs = _emit_transform_priors(spec, transform_map, priors)

        data_vars: dict[str, Any] = {}
        for var in graph_info.topological_order:
            if var in graph_info.exogenous and var in data.columns:
                data_vars[var] = pm.Data(var, data[var].to_numpy().astype(float))

        endogenous_rvs: dict[str, Any] = {}

        var_to_block_idx: dict[str, int] = {}
        for idx, block in enumerate(blocks):
            for v in block:
                var_to_block_idx[v] = idx
        block_members_seen: dict[int, set[str]] = {i: set() for i in range(len(blocks))}
        compiled_blocks: set[int] = set()

        for var in graph_info.topological_order:
            if var not in reg_by_lhs:
                continue

            if var in block_vars:
                bidx = var_to_block_idx[var]
                block_members_seen[bidx].add(var)
                if (
                    block_members_seen[bidx] == blocks[bidx]
                    and bidx not in compiled_blocks
                ):
                    block_topo = [
                        v for v in graph_info.topological_order if v in blocks[bidx]
                    ]
                    _compile_residual_block(
                        blocks[bidx],
                        data,
                        mu_specs,
                        data_vars,
                        endogenous_rvs,
                        transform_map,
                        transform_param_rvs,
                        panel_info,
                        priors,
                        topological_order=block_topo,
                    )
                    compiled_blocks.add(bidx)
                continue

            reg = reg_by_lhs[var]
            family = families.get(var, "gaussian")
            free_cols = get_free_predictor_columns(reg)

            beta = None
            if free_cols:
                beta_prior = _ensure_dims(priors[f"beta_{var}"], f"{var}_predictors")
                beta = beta_prior.create_variable(f"beta_{var}")

            resolver = _make_cross_sectional_resolver(
                data,
                data_vars,
                endogenous_rvs,
                transform_map,
                transform_param_rvs,
                panel_info,
            )
            mu = build_mu(mu_specs[var], resolver, beta, pt.zeros(len(data)))

            if (
                has_random_intercepts
                and panel_info is not None
                and unit_idx is not None
            ):
                mu = mu + _compile_random_intercept(var, unit_idx, priors)

            if slope_vars and panel_info is not None and unit_idx is not None:
                mu = mu + _compile_random_slopes(
                    reg, slope_vars, data_vars, unit_idx, priors
                )

            mu_det = pm.Deterministic(f"mu_{var}", mu)

            rv = _emit_free_rv(var, mu_det, family, latent, sparse_data, priors)
            if family in ("bernoulli", "poisson", "negbinomial"):
                endogenous_rvs[var] = pt.cast(rv, "float64")
            else:
                endogenous_rvs[var] = rv

    return pymc_model


# ---------------------------------------------------------------------------
# Helpers: unified mu construction (shared by cross-sectional and scan paths)
# ---------------------------------------------------------------------------


def build_mu(
    mu_spec: MuSpec,
    resolver: Callable[[PredictorSlot], Any],
    beta: Any | None,
    zero: Any,
) -> Any:
    """Build a linear predictor from a ``MuSpec`` and a resolver.

    The shared coefficient loop handles free/fixed indexing and
    intercept logic.  The path-specific *resolver* maps each
    non-intercept ``PredictorSlot`` to its tensor value.

    Parameters
    ----------
    mu_spec : MuSpec
        Symbolic linear predictor description.
    resolver : Callable[[PredictorSlot], Any]
        Maps a ``PredictorSlot`` to a tensor variable.
    beta : Any | None
        Free coefficient vector (indexed for free slots).  ``None``
        when all coefficients are fixed.
    zero : Any
        Initial accumulator (e.g. ``pt.zeros(n_obs)``).

    Returns
    -------
    Any
        PyTensor expression for the linear predictor.
    """
    mu = zero
    free_idx = 0

    for slot in mu_spec.slots:
        if slot.coeff_type == "fixed":
            coef: Any = slot.coeff_value
        else:
            assert beta is not None
            coef = beta[free_idx]
            free_idx += 1

        if slot.kind == "intercept":
            mu = mu + coef
        else:
            tensor = resolver(slot)
            mu = mu + coef * tensor

    return mu


def _make_cross_sectional_resolver(
    data: nw.DataFrame,
    data_vars: dict[str, Any],
    endogenous_rvs: dict[str, Any],
    transform_map: dict[str, TransformCall],
    transform_param_rvs: dict[str, Any],
    panel_info: PanelInfo | None,
) -> Callable[[PredictorSlot], Any]:
    """Create a resolver for cross-sectional mu construction.

    Resolves tensors through ``pm.Data`` for exogenous inputs and
    upstream free RVs for endogenous inputs, enabling ``pm.do()``
    propagation.
    """
    import pytensor.tensor as pt

    def _resolve_var(name: str) -> Any:
        if name in endogenous_rvs:
            return endogenous_rvs[name]
        if name in data_vars:
            return data_vars[name]
        return pt.as_tensor_variable(data[name].to_numpy().astype(float))

    def resolve(slot: PredictorSlot) -> Any:
        if slot.kind == "transform":
            tc = transform_map[slot.name]
            return _apply_transform_chain(
                tc,
                data,
                transform_param_rvs,
                panel_info=panel_info,
                data_vars=data_vars,
                endogenous_rvs=endogenous_rvs,
            )
        if slot.kind == "interaction":
            assert slot.interaction_parts is not None
            product = _resolve_var(slot.interaction_parts[0])
            for part in slot.interaction_parts[1:]:
                product = product * _resolve_var(part)
            return product
        return _resolve_var(slot.name)

    return resolve


def _make_scan_resolver(
    exog_t: dict[str, Any],
    new_endo: dict[str, Any],
    prev_endo: dict[str, Any],
    prev_exog: dict[str, Any],
    transform_map: dict[str, TransformCall],
    ns_map: dict[str, Any],
    adstock_state: dict[str, Any],
    n_units: int,
) -> Callable[[PredictorSlot], Any]:
    """Create a resolver for scan-panel mu construction.

    Resolves tensors from per-step scan arguments: current-step
    exogenous slices, already-computed endogenous values, and
    previous-step carry state for lags.

    Side effect: transform slots update *adstock_state* in-place.
    """
    import pytensor.tensor as pt

    def _resolve_var(name: str) -> Any:
        if name in new_endo:
            return new_endo[name]
        if name in exog_t:
            return exog_t[name]
        return pt.zeros(n_units)

    def resolve(slot: PredictorSlot) -> Any:
        if slot.kind == "transform":
            tc = transform_map[slot.name]
            inp_name = _get_adstock_input(tc)
            raw = _resolve_var(inp_name)
            transformed, updated = _apply_step_transform(
                tc, raw, adstock_state, ns_map, slot.name
            )
            adstock_state.update(updated)
            return transformed
        if slot.kind == "interaction":
            assert slot.interaction_parts is not None
            product = _resolve_var(slot.interaction_parts[0])
            for part in slot.interaction_parts[1:]:
                product = product * _resolve_var(part)
            return product
        if slot.kind == "lag":
            base_var = slot.lag_of
            if base_var is not None:
                if base_var in prev_endo:
                    return prev_endo[base_var]
                if base_var in prev_exog:
                    return prev_exog[base_var]
            return pt.zeros(n_units)
        return _resolve_var(slot.name)

    return resolve


# ---------------------------------------------------------------------------
# Helpers: pooling / random effects
# ---------------------------------------------------------------------------


def _has_random_intercepts(pooling: str | dict | None) -> bool:
    """Whether pooling spec requests random intercepts."""
    if pooling == "partial":
        return True
    if isinstance(pooling, dict):
        return pooling.get("intercept", False)
    return False


def _get_slope_vars(pooling: str | dict | None) -> list[str]:
    """Extract variables that should get random slopes."""
    if isinstance(pooling, dict):
        return list(pooling.get("slopes", []))
    return []


def _build_unit_index(data: nw.DataFrame, panel_info: PanelInfo) -> np.ndarray:
    """Map each row to an integer unit index."""
    label_to_idx = {label: i for i, label in enumerate(panel_info.unit_labels)}
    units = data[panel_info.unit].to_numpy()
    try:
        return np.array([label_to_idx[u] for u in units])
    except KeyError as exc:
        raise ValueError(
            f"Panel unit column '{panel_info.unit}' contains a value "
            f"({exc.args[0]!r}) that is not among the known unit labels. "
            f"This usually means a null/NaN unit id. Drop rows with missing "
            f"unit identifiers before fitting."
        ) from exc


def _compile_random_intercept(
    var: str,
    unit_idx: np.ndarray,
    priors: dict[str, Any],
) -> Any:
    """Emit hierarchical random intercept for *var*, return alpha[unit_idx]."""
    mu_alpha = priors[f"mu_alpha_{var}"].create_variable(f"mu_alpha_{var}")
    sigma_alpha = priors[f"sigma_alpha_{var}"].create_variable(f"sigma_alpha_{var}")
    alpha = pm.Normal(f"alpha_{var}", mu=mu_alpha, sigma=sigma_alpha, dims="unit")
    return alpha[unit_idx]


def _compile_random_slopes(
    reg: Regression,
    slope_vars: list[str],
    data_vars: dict[str, Any],
    unit_idx: np.ndarray,
    priors: dict[str, Any],
) -> Any:
    """Emit hierarchical random slopes for specified predictors.

    Uses the symbolic ``pm.Data`` variables so that ``pm.do()``
    interventions propagate through the random slope terms.
    """
    import pymc as pm

    contribution = 0
    term_variables = {t.variable for t in reg.terms}
    for svar in slope_vars:
        if svar not in term_variables:
            continue
        mu_slope = priors[f"mu_slope_{reg.lhs}_{svar}"].create_variable(
            f"mu_slope_{reg.lhs}_{svar}"
        )
        sigma_slope = priors[f"sigma_slope_{reg.lhs}_{svar}"].create_variable(
            f"sigma_slope_{reg.lhs}_{svar}"
        )
        slope = pm.Normal(
            f"slope_{reg.lhs}_{svar}",
            mu=mu_slope,
            sigma=sigma_slope,
            dims="unit",
        )
        x_symbolic = data_vars[svar]
        contribution = contribution + slope[unit_idx] * x_symbolic
    return contribution


# ---------------------------------------------------------------------------
# Helpers: residual blocks (MvNormal with LKJ)
# ---------------------------------------------------------------------------


def _compile_residual_block(
    block: set[str],
    data: nw.DataFrame,
    mu_specs: dict[str, MuSpec],
    data_vars: dict[str, Any],
    endogenous_rvs: dict[str, Any],
    transform_map: dict[str, TransformCall],
    transform_param_rvs: dict[str, Any],
    panel_info: PanelInfo | None = None,
    priors: dict[str, Any] | None = None,
    topological_order: list[str] | None = None,
) -> None:
    """Compile a residual-covariance block.

    Uses ``MuSpec`` + resolver so that endogenous predictors wire
    through upstream free RVs, enabling ``pm.do()`` propagation.
    Creates ``mu_{var}`` deterministics and registers block variables
    in *endogenous_rvs* so downstream equations wire through the model
    graph (enabling ``pm.do()`` propagation through block variables).

    Delegates the covariance parameterization and likelihood emission
    to an :class:`~pathmc.residuals.LKJResidual`.

    Parameters
    ----------
    topological_order : list[str] | None
        Block members in topological order.  When provided, variables
        are processed in this order to ensure correct wiring when one
        block member depends on another.
    """
    import pytensor.tensor as pt

    from pathmc.priors import _ensure_dims
    from pathmc.residuals import LKJResidual

    process_order = (
        topological_order if topological_order is not None else sorted(block)
    )
    block_sorted = sorted(block)

    mu_dict: dict[str, Any] = {}
    data_dict: dict[str, np.ndarray] = {}

    for var in process_order:
        ms = mu_specs[var]
        has_free = any(s.coeff_type == "free" for s in ms.slots)
        beta = None
        if has_free:
            if priors and f"beta_{var}" in priors:
                beta_prior = _ensure_dims(priors[f"beta_{var}"], f"{var}_predictors")
                beta = beta_prior.create_variable(f"beta_{var}")
            else:
                beta = pm.Normal(
                    f"beta_{var}",
                    mu=0,
                    sigma=10,
                    dims=f"{var}_predictors",
                )
        resolver = _make_cross_sectional_resolver(
            data,
            data_vars,
            endogenous_rvs,
            transform_map,
            transform_param_rvs,
            panel_info,
        )
        mu = build_mu(ms, resolver, beta, pt.zeros(len(data)))

        mu_det = pm.Deterministic(f"mu_{var}", mu)
        endogenous_rvs[var] = mu_det
        mu_dict[var] = mu
        data_dict[var] = data[var].to_numpy()

    structure = LKJResidual()
    structure.emit(block_sorted, mu_dict, data_dict, priors)


def _identify_residual_blocks(spec: Spec) -> tuple[set[str], list[set[str]]]:
    """Return variables in residual blocks and the blocks themselves."""
    if not spec.residual_covs:
        return set(), []

    ug = nx.Graph()
    for rc in spec.residual_covs:
        ug.add_edge(rc.var1, rc.var2)

    blocks = list(nx.connected_components(ug))
    block_vars = set().union(*blocks)
    return block_vars, blocks


# ---------------------------------------------------------------------------
# Helpers: likelihoods
# ---------------------------------------------------------------------------


def _emit_free_rv(
    var: str,
    mu: Any,
    family: str,
    latent: set[str],
    sparse_data: dict[str, np.ma.MaskedArray] | None = None,
    priors: dict[str, Any] | None = None,
) -> Any:
    """Emit a free random variable for an endogenous variable.

    Latent variables are emitted as ``pm.Deterministic`` (no noise)
    unless ``family="latent_normal"``, in which case they get a
    ``pm.Normal`` with process noise.

    Sparse measurement variables (present in *sparse_data*) are emitted
    as observed ``pm.Normal`` with a masked array so PyMC automatically
    handles missing positions.

    Returns the RV tensor so downstream equations can wire through it.
    """
    if var in latent:
        if family == "latent_normal":
            sigma = _create_prior_var(priors, f"sigma_{var}")
            return pm.Normal(var, mu=mu, sigma=sigma)
        return mu

    if sparse_data is not None and var in sparse_data:
        sigma = _create_prior_var(priors, f"sigma_{var}")
        pm.Normal(var, mu=mu, sigma=sigma, observed=sparse_data[var])
        return mu

    if family == "bernoulli":
        return pm.Bernoulli(var, logit_p=mu)
    if family == "poisson":
        return pm.Poisson(var, mu=pm.math.exp(mu))
    if family == "negbinomial":
        alpha_disp = _create_prior_var(priors, f"alpha_disp_{var}")
        return pm.NegativeBinomial(var, mu=pm.math.exp(mu), alpha=alpha_disp)
    if family == "studentt":
        sigma = _create_prior_var(priors, f"sigma_{var}")
        nu = _create_prior_var(priors, f"nu_{var}")
        return pm.StudentT(var, nu=nu, mu=mu, sigma=sigma)

    sigma = _create_prior_var(priors, f"sigma_{var}")
    return pm.Normal(var, mu=mu, sigma=sigma)


def _create_prior_var(priors: dict[str, Any] | None, name: str) -> Any:
    """Create a PyMC variable from a Prior in the config, or fall back to default."""
    if priors and name in priors:
        return priors[name].create_variable(name)
    return pm.HalfNormal(name, sigma=1)


# ---------------------------------------------------------------------------
# Helpers: transforms
# ---------------------------------------------------------------------------


def _build_transform_map(spec: Spec) -> dict[str, TransformCall]:
    """Map variable names to their TransformCall for all transform terms."""
    tmap: dict[str, TransformCall] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.transform is not None:
                tmap[term.variable] = term.transform
    return tmap


def _emit_transform_priors(
    spec: Spec,
    transform_map: dict[str, TransformCall],
    priors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Emit PyMC priors for all transform parameters. Returns name->RV mapping."""
    emitted: dict[str, Any] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.transform is not None:
                _emit_transform_call_priors(term.transform, emitted, priors)
    return emitted


def _emit_transform_call_priors(
    tc: TransformCall,
    emitted: dict[str, Any],
    priors: dict[str, Any] | None = None,
) -> None:
    """Recursively emit priors for a (possibly nested) TransformCall."""
    if isinstance(tc.input_expr, TransformCall):
        _emit_transform_call_priors(tc.input_expr, emitted, priors)

    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in emitted:
            if priors and param_name in priors:
                emitted[param_name] = priors[param_name].create_variable(param_name)
            else:
                pspec = transform.param_specs[param_key]
                emitted[param_name] = transform.emit_prior(param_name, pspec)


def _apply_transform_chain(
    tc: TransformCall,
    data: nw.DataFrame,
    param_rvs: dict[str, Any],
    panel_info: PanelInfo | None = None,
    data_vars: dict[str, Any] | None = None,
    endogenous_rvs: dict[str, Any] | None = None,
) -> Any:
    """Recursively apply a (possibly nested) transform chain.

    Uses ``pm.Data`` for exogenous inputs, upstream free RVs for
    endogenous inputs, or raw data tensors as fallback.
    """
    import pytensor.tensor as pt

    if isinstance(tc.input_expr, TransformCall):
        input_tensor = _apply_transform_chain(
            tc.input_expr,
            data,
            param_rvs,
            panel_info=panel_info,
            data_vars=data_vars,
            endogenous_rvs=endogenous_rvs,
        )
    else:
        if data_vars and tc.input_expr in data_vars:
            input_tensor = data_vars[tc.input_expr]
        elif endogenous_rvs and tc.input_expr in endogenous_rvs:
            input_tensor = endogenous_rvs[tc.input_expr]
        else:
            input_tensor = pt.as_tensor_variable(
                data[tc.input_expr].to_numpy().astype(float)
            )

    transform = get_transform(tc.name)
    params = {key: param_rvs[name] for key, name in tc.params.items()}
    return transform.apply_pymc(input_tensor, params, panel_info=panel_info, data=data)


def _validate_residual_cov_families(spec: Spec, families: dict[str, str]) -> None:
    """Raise if any variable in a ``~~`` pair is non-Gaussian."""
    allowed = {"gaussian"}
    for rc in spec.residual_covs:
        for var in (rc.var1, rc.var2):
            family = families.get(var, "gaussian")
            if family not in allowed:
                raise ValueError(
                    f"Residual covariance (~~) requires Gaussian family, "
                    f"but '{var}' has family '{family}'. "
                    f"Covariance modeling is only supported for continuous "
                    f"Gaussian outcomes."
                )


def _warn_partial_pooling_intercept(spec: Spec, pooling: str | dict | None) -> None:
    """Warn if partial pooling is combined with formula intercepts.

    Partial pooling models include both a fixed intercept (beta[Intercept])
    and a hierarchical mean (mu_alpha). Only their sum is identified by the
    data, creating a non-identifiable parameterization that causes divergences.

    Raises
    ------
    UserWarning
        When any equation has an intercept and pooling requests random intercepts.
    """
    import warnings

    if not _has_random_intercepts(pooling):
        return

    equations_with_intercepts = [
        reg.lhs for reg in spec.regressions if reg.has_intercept
    ]

    if not equations_with_intercepts:
        return

    var_list = ", ".join(f"'{v}'" for v in equations_with_intercepts)
    formulas = "\n".join(
        f"  {reg.lhs} ~ 0 + {' + '.join(t.variable for t in reg.terms)}"
        for reg in spec.regressions
        if reg.has_intercept
    )

    warnings.warn(
        f"\n{'=' * 78}\n"
        f"PARTIAL POOLING WITH REDUNDANT INTERCEPT\n"
        f"{'=' * 78}\n"
        f"\n"
        f"Your model uses pooling='partial' (random intercepts) but the following\n"
        f"equations include a formula intercept: {var_list}.\n"
        f"\n"
        f"This creates a NON-IDENTIFIABLE parameterization:\n"
        f"  • beta[Intercept] (fixed global intercept)\n"
        f"  • mu_alpha (mean of random intercepts)\n"
        f"Only their sum is identified by the data. This causes sampling divergences.\n"
        f"\n"
        f"SOLUTION: Remove the intercept from your formula(s):\n"
        f"\n"
        f"{formulas}\n"
        f"\n"
        f"The hierarchical mean mu_alpha will serve as the effective intercept.\n"
        f"{'=' * 78}\n",
        UserWarning,
        stacklevel=4,
    )


# ---------------------------------------------------------------------------
# Temporal dependency detection
# ---------------------------------------------------------------------------


def _build_lag_map(spec: Spec) -> dict[str, str]:
    """Map lag term variable names to their base variables.

    Returns a dict like ``{"lag(sales)": "sales"}`` built from
    ``Term.lag_of`` fields produced by the ``lag()`` DSL syntax.
    """
    lag_map: dict[str, str] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.lag_of is not None:
                lag_map[term.variable] = term.lag_of
    return lag_map


def _has_temporal_deps(spec: Spec, graph_info: GraphInfo) -> bool:
    """Return True if the model has adstock transforms or any lag terms.

    Detects temporal dependencies from:
    - ``lag()`` DSL syntax (``Term.lag_of``)
    - ``adstock()`` transforms
    """
    for reg in spec.regressions:
        for term in reg.terms:
            if term.lag_of is not None:
                return True
            if term.transform is not None:
                tc: TransformCall | None = term.transform
                while tc is not None:
                    if tc.name == "adstock":
                        return True
                    tc = (
                        tc.input_expr
                        if isinstance(tc.input_expr, TransformCall)
                        else None
                    )
    return False


def _transform_base_vars(tc: TransformCall) -> list[str]:
    """Return leaf variable names used by a transform chain."""
    if isinstance(tc.input_expr, TransformCall):
        return _transform_base_vars(tc.input_expr)
    return [tc.input_expr]


def _scan_term_base_vars(term: Term) -> list[str]:
    """Return base variables a scan predictor term reads from."""
    if term.lag_of is not None:
        return [term.lag_of]

    if term.transform is not None:
        return _transform_base_vars(term.transform)

    return _term_base_vars(term)


def _validate_scan_non_gaussian_intermediaries(
    spec: Spec,
    families: dict[str, str],
    latent: set[str],
) -> None:
    """Reject scan models that would pass response means downstream."""
    discrete_families = {"bernoulli", "poisson", "negbinomial"}
    non_gaussian = {
        reg.lhs
        for reg in spec.regressions
        if reg.lhs not in latent
        and families.get(reg.lhs, "gaussian") in discrete_families
    }
    if not non_gaussian:
        return

    downstream: dict[str, set[str]] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            for base_var in _scan_term_base_vars(term):
                if base_var in non_gaussian:
                    downstream.setdefault(base_var, set()).add(reg.lhs)

    if not downstream:
        return

    details = "; ".join(
        f"'{var}' ({families.get(var, 'gaussian')}) -> {', '.join(sorted(targets))}"
        for var, targets in sorted(downstream.items())
    )
    raise ValueError(
        "Scan-compiled panel models do not support non-Gaussian endogenous "
        "variables as predictors in downstream equations. The current scan "
        "compiler would propagate response means (probabilities or rates) "
        f"instead of sampled Bernoulli/count values: {details}. Make these "
        "variables terminal outcomes, remove temporal terms so the "
        "cross-sectional compiler can be used, or follow #191 for the "
        "split-scan stochastic compiler work."
    )


def _get_adstock_input(tc: TransformCall) -> str:
    """Return the leaf input variable name of a (possibly nested) transform chain."""
    current = tc
    while isinstance(current.input_expr, TransformCall):
        current = current.input_expr
    return current.input_expr


def _has_adstock(tc: TransformCall) -> bool:
    """Return True if the transform chain includes adstock."""
    current: Any = tc
    while current is not None:
        if isinstance(current, TransformCall) and current.name == "adstock":
            return True
        current = current.input_expr if isinstance(current, TransformCall) else None
    return False


# ---------------------------------------------------------------------------
# Scan-based panel compilation
# ---------------------------------------------------------------------------


def _reshape_to_panel(
    data_sorted: nw.DataFrame,
    column: str,
    n_units: int,
    n_times: int,
) -> np.ndarray:
    """Reshape a column from flat sorted data to ``(n_times, n_units)``."""
    return data_sorted[column].to_numpy().reshape(n_units, n_times).T


def _apply_step_transform(
    tc: TransformCall,
    x_t: Any,
    prev_adstock: dict[str, Any],
    param_rvs: dict[str, Any],
    col_name: str,
) -> tuple[Any, dict[str, Any]]:
    """Apply a (possibly nested) transform chain for one scan time step.

    Returns ``(transformed_value, updated_adstock_dict)``.
    """
    transform = get_transform(tc.name)
    params = {key: param_rvs[name] for key, name in tc.params.items()}

    if isinstance(tc.input_expr, TransformCall):
        x_t, prev_adstock = _apply_step_transform(
            tc.input_expr, x_t, prev_adstock, param_rvs, col_name
        )

    state = prev_adstock.get(col_name) if transform.has_state else None
    import pytensor.tensor as pt

    if state is None:
        state = pt.zeros_like(x_t)

    out, new_state = transform.step(x_t, state, params)

    if transform.has_state:
        prev_adstock = {**prev_adstock, col_name: new_state}

    return out, prev_adstock


def _compile_scan_panel(
    spec: Spec,
    data: nw.DataFrame,
    design_matrices: dict[str, nw.DataFrame],
    families: dict[str, str],
    panel_info: PanelInfo,
    pooling: str | dict | None,
    latent: set[str],
    graph_info: GraphInfo,
    priors: dict[str, Any] | None = None,
) -> pm.Model:
    """Compile a panel model with temporal deps using ``pytensor.scan``.

    The generative model encodes the full temporal structure so that
    ``pm.do()`` handles interventions natively.  Free RVs have shape
    ``(n_times, n_units)`` in unit-major sorted order.
    """
    import pytensor
    import pytensor.tensor as pt

    from pathmc.priors import _ensure_dims, default_priors

    if priors is None:
        priors = default_priors(spec, families, pooling, latent)

    _warn_partial_pooling_intercept(spec, pooling)

    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    transform_map = _build_transform_map(spec)
    mu_specs = build_mu_specs(spec)
    has_ri = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    # --- sort data ---
    unit_col = panel_info.unit
    time_col = panel_info.time
    sorted_with_pos = data.with_row_index("__nw_row_pos__").sort([unit_col, time_col])
    sort_idx = sorted_with_pos["__nw_row_pos__"].to_numpy()
    data_sorted = sorted_with_pos.drop("__nw_row_pos__")
    reverse_idx = np.argsort(sort_idx)
    units = panel_info.unit_labels
    n_units = len(units)
    n_times = len(data) // n_units
    time_values = sorted(data_sorted[time_col].unique().to_list())

    # --- classify columns ---
    lag_map = _build_lag_map(spec)

    pure_exog = [
        v
        for v in graph_info.topological_order
        if v in graph_info.exogenous and v not in lag_map and v in data_sorted.columns
    ]
    lag_cols: dict[str, tuple[str, int]] = {
        col_name: (base_var, 1) for col_name, base_var in lag_map.items()
    }

    adstock_cols = [col for col, tc in transform_map.items() if _has_adstock(tc)]

    stochastic_latent = sorted(
        v for v in latent if families.get(v, "gaussian") == "latent_normal"
    )
    stochastic_latent_set = set(stochastic_latent)

    sparse_panel_data: dict[str, np.ma.MaskedArray] = {}
    for reg in spec.regressions:
        v = reg.lhs
        if v not in latent and v in data.columns:
            raw = np.asarray(data_sorted[v].to_numpy(), dtype=float)
            if np.isnan(raw).any():
                panel_vals = raw.reshape(n_units, n_times).T
                sparse_panel_data[v] = np.ma.masked_invalid(panel_vals)

    # --- coords ---
    coords: dict[str, Any] = {}
    fixed_coeffs_by_var: dict[str, dict[str, float]] = {}
    for reg in spec.regressions:
        free_cols = get_free_predictor_columns(reg)
        if free_cols:
            coords[f"{reg.lhs}_predictors"] = free_cols
        fixed_coeffs_by_var[reg.lhs] = get_fixed_coefficients(reg)
    if has_ri:
        coords["unit"] = units

    with pm.Model(coords=coords) as scan_model:
        # --- transform parameter priors ---
        tparam_rvs: dict[str, Any] = {}
        for reg in spec.regressions:
            for term in reg.terms:
                if term.transform is not None:
                    _emit_transform_call_priors(term.transform, tparam_rvs, priors)

        # --- regression parameter priors ---
        beta_rvs: dict[str, Any] = {}
        sigma_rvs: dict[str, Any] = {}
        for var in endogenous_order:
            family = families.get(var, "gaussian")
            free_cols = get_free_predictor_columns(reg_by_lhs[var])
            if free_cols:
                beta_prior = _ensure_dims(priors[f"beta_{var}"], f"{var}_predictors")
                beta_rvs[var] = beta_prior.create_variable(f"beta_{var}")
            else:
                beta_rvs[var] = None
            if var not in latent:
                if family in ("gaussian", "studentt"):
                    sigma_rvs[var] = _create_prior_var(priors, f"sigma_{var}")
                if family == "studentt":
                    _create_prior_var(priors, f"nu_{var}")
                if family == "negbinomial":
                    _create_prior_var(priors, f"alpha_disp_{var}")
            elif family == "latent_normal":
                sigma_rvs[var] = _create_prior_var(priors, f"sigma_{var}")

        # --- random effects ---
        alpha_rvs: dict[str, Any] = {}
        slope_rvs: dict[str, dict[str, Any]] = {}
        if has_ri:
            for var in endogenous_order:
                mu_a = priors[f"mu_alpha_{var}"].create_variable(f"mu_alpha_{var}")
                sig_a = priors[f"sigma_alpha_{var}"].create_variable(
                    f"sigma_alpha_{var}"
                )
                alpha_rvs[var] = pm.Normal(
                    f"alpha_{var}", mu=mu_a, sigma=sig_a, dims="unit"
                )
        for var in endogenous_order:
            reg = reg_by_lhs[var]
            term_variables = {t.variable for t in reg.terms}
            slope_rvs[var] = {}
            for svar in slope_vars:
                if svar in term_variables:
                    mu_s = priors[f"mu_slope_{var}_{svar}"].create_variable(
                        f"mu_slope_{var}_{svar}"
                    )
                    sig_s = priors[f"sigma_slope_{var}_{svar}"].create_variable(
                        f"sigma_slope_{var}_{svar}"
                    )
                    slope_rvs[var][svar] = pm.Normal(
                        f"slope_{var}_{svar}",
                        mu=mu_s,
                        sigma=sig_s,
                        dims="unit",
                    )

        # --- exogenous data as pm.Data (n_times, n_units) ---
        # Include both direct exogenous vars and base vars of lag columns
        endo_set = frozenset(endogenous_order)
        exog_data_vars = {v for v in pure_exog}
        for _col, (base, _k) in lag_cols.items():
            if base not in endo_set and base in data_sorted.columns:
                exog_data_vars.add(base)

        exog_data_nodes: dict[str, Any] = {}
        for var in sorted(exog_data_vars):
            mat = _reshape_to_panel(data_sorted, var, n_units, n_times)
            exog_data_nodes[var] = pm.Data(var, mat.astype(float))

        # --- scan setup ---
        endo_keys = list(endogenous_order)
        adstock_keys = sorted(adstock_cols)
        exog_keys = sorted(exog_data_nodes.keys())

        # Exogenous variables referenced by lag columns need carry state
        exog_lag_bases = sorted({
            base for _col, (base, _k) in lag_cols.items() if base not in endo_set
        })

        init_endo: dict[str, np.ndarray] = {
            var: np.zeros(n_units, dtype="float64") for var in endo_keys
        }
        for lag_col, (base_var, _lag_k) in lag_cols.items():
            if base_var in init_endo and lag_col in data_sorted.columns:
                mat = _reshape_to_panel(data_sorted, lag_col, n_units, n_times)
                init_endo[base_var] = mat[0].astype("float64")
            elif base_var in init_endo and base_var in data_sorted.columns:
                mat = _reshape_to_panel(data_sorted, base_var, n_units, n_times)
                init_endo[base_var] = mat[0].astype("float64")

        init_exog_lag: dict[str, np.ndarray] = {}
        for base in exog_lag_bases:
            lag_col_name = f"{base}_lag1"
            if lag_col_name in data_sorted.columns:
                mat = _reshape_to_panel(data_sorted, lag_col_name, n_units, n_times)
                init_exog_lag[base] = mat[0].astype("float64")
            elif base in data_sorted.columns:
                mat = _reshape_to_panel(data_sorted, base, n_units, n_times)
                init_exog_lag[base] = mat[0].astype("float64")
            else:
                init_exog_lag[base] = np.zeros(n_units, dtype="float64")

        init_adstock: dict[str, np.ndarray] = {
            col: np.zeros(n_units, dtype="float64") for col in adstock_keys
        }

        # Flag toggled by caller: 0 for generative recursion, 1 for observed carry.
        use_observed_carry = pm.Data("_use_observed_carry", np.array(0, dtype="int8"))

        endo_lag_bases = sorted({
            base for _col, (base, _k) in lag_cols.items() if base in endo_set
        })
        stochastic_carry_vars = sorted(
            v
            for v in endo_lag_bases
            if v not in latent
            and families.get(v, "gaussian") in ("gaussian", "studentt")
        )

        observed_carry_nodes: dict[str, Any] = {}
        for var in stochastic_carry_vars:
            if var in data_sorted.columns:
                observed_panel = _reshape_to_panel(data_sorted, var, n_units, n_times)
            else:
                observed_panel = np.full((n_times, n_units), np.nan, dtype="float64")
            observed_carry_nodes[var] = pm.Data(
                f"_obs_carry_{var}", observed_panel.astype("float64")
            )

        latent_innovation_nodes: dict[str, Any] = {}
        for var in stochastic_latent:
            latent_innovation_nodes[var] = pm.Normal(
                f"innovations_{var}", mu=0, sigma=1, shape=(n_times, n_units)
            )

        carry_innovation_nodes: dict[str, Any] = {}
        for var in stochastic_carry_vars:
            carry_innovation_nodes[var] = pm.Normal(
                f"carry_innovations_{var}", mu=0, sigma=1, shape=(n_times, n_units)
            )

        # Pre-compute lagged exogenous sequences from pm.Data nodes.
        #
        # PyTensor's scan-merge optimizer has a bug that fires when a sit_sot
        # carry update is trivially ``inner_out = current_seq_slice`` (i.e., the
        # carry merely echoes the input sequence one step behind). That structure
        # appeared in the original exog-lag carry: ``out[i] = exog_t[base]``.
        # When two scan computations sharing the same inner function are compiled
        # together (as happens when ``pytensor.function`` receives both
        # ``mu_valued`` and ``logp``), the optimizer merges the two scans but
        # incorrectly permutes the carry channels, producing a wrong logp graph
        # that ``pm.sample`` then optimizes — causing the zeroed-out lag-effect
        # posteriors reported in issue #316.
        # Upstream bug: https://github.com/pymc-devs/pytensor/issues/2252
        # TODO: once pytensor/issues/2252 is fixed and released, revert to a
        # scan carry here and remove this workaround (see pathmc issue #333).
        #
        # Fix: build the lagged tensor directly from the existing pm.Data nodes
        # (so pm.set_data / do() interventions still propagate automatically)
        # and pass it as a plain scan *sequence* rather than carry state.  This
        # eliminates the trivial-echo carry that triggered the merge bug.
        lagged_exog_sequences: dict[str, Any] = {}
        for base in exog_lag_bases:
            init_row = pt.as_tensor_variable(
                init_exog_lag[base][None, :]
            )  # (1, n_units)
            if base in exog_data_nodes:
                lagged_exog_sequences[base] = pt.concatenate(
                    [init_row, exog_data_nodes[base][:-1]], axis=0
                )  # (n_times, n_units)
            else:
                # No contemporaneous exog data node for this lag base — e.g. a
                # ``lag(x)`` term whose base column is absent from the data (the
                # same case ``init_exog_lag`` handles with its zeros/lag1
                # fallback above).  ``exog_lag_bases`` filters only on
                # ``base not in endo_set`` while ``exog_data_nodes`` additionally
                # requires ``base in data_sorted.columns``, so the two key sets
                # can diverge.  The old carry path resolved such bases to the
                # init row at t=0 and zeros for t>=1 (via
                # ``exog_t.get(k, pt.zeros(n_units))``); reproduce that here
                # rather than raising KeyError on a direct index.
                zeros_tail = pt.zeros((n_times - 1, n_units))
                lagged_exog_sequences[base] = pt.concatenate(
                    [init_row, zeros_tail], axis=0
                )  # (n_times, n_units)

        sequences = (
            [exog_data_nodes[k] for k in exog_keys]
            + [lagged_exog_sequences[k] for k in exog_lag_bases]
            + [observed_carry_nodes[k] for k in stochastic_carry_vars]
            + [latent_innovation_nodes[k] for k in stochastic_latent]
            + [carry_innovation_nodes[k] for k in stochastic_carry_vars]
        )

        def _init_carry(arr: np.ndarray) -> Any:
            """Convert init array to tensor for scan carry state.

            Uses ``pytensor.shared`` when n_units=1 to prevent PyTensor
            from marking the unit dimension as broadcastable (static
            shape 1), which would cause shape mismatches in the
            gradient scan.
            """
            if arr.shape[0] == 1:
                return pytensor.shared(arr, broadcastable=(False,))
            return pt.as_tensor_variable(arr)

        outputs_info = (
            [_init_carry(init_endo[k]) for k in endo_keys]
            + [_init_carry(init_adstock[k]) for k in adstock_keys]
            + [None for _ in stochastic_carry_vars]
        )

        # Non-sequences: all parameters
        non_seq_list: list[Any] = []
        non_seq_names: list[str] = []
        beta_component_names: dict[str, list[str]] = {}
        non_seq_list.append(use_observed_carry)
        non_seq_names.append("_use_observed_carry")
        for var in endo_keys:
            if beta_rvs[var] is not None:
                beta_component_names[var] = []
                for idx in range(len(get_free_predictor_columns(reg_by_lhs[var]))):
                    component_name = f"beta_{var}__{idx}"
                    non_seq_list.append(beta_rvs[var][idx])
                    non_seq_names.append(component_name)
                    beta_component_names[var].append(component_name)
        for name, rv in tparam_rvs.items():
            non_seq_list.append(rv)
            non_seq_names.append(name)
        for var in endo_keys:
            if var in alpha_rvs:
                non_seq_list.append(alpha_rvs[var])
                non_seq_names.append(f"alpha_{var}")
        for var in endo_keys:
            for svar, srv in slope_rvs.get(var, {}).items():
                non_seq_list.append(srv)
                non_seq_names.append(f"slope_{var}_{svar}")
        for var in stochastic_latent:
            non_seq_list.append(sigma_rvs[var])
            non_seq_names.append(f"sigma_{var}")
        for var in stochastic_carry_vars:
            non_seq_list.append(sigma_rvs[var])
            non_seq_names.append(f"sigma_{var}")

        n_seq = len(sequences)
        n_exog_seq = len(exog_keys)
        n_exog_lag_seq = len(exog_lag_bases)
        n_obs_carry_seq = len(stochastic_carry_vars)
        n_latent_innov_seq = len(stochastic_latent)
        n_endo = len(endo_keys)
        n_adstock = len(adstock_keys)
        n_carry = n_endo + n_adstock

        def step_fn(*args: Any) -> list[Any]:
            seq_args = args[:n_seq]
            carry_args = args[n_seq : n_seq + n_carry]
            ns_args = args[n_seq + n_carry :]

            exog_t = {k: seq_args[i] for i, k in enumerate(exog_keys)}
            lagged_exog_t = {
                k: seq_args[n_exog_seq + i] for i, k in enumerate(exog_lag_bases)
            }
            obs_carry_t = {
                k: seq_args[n_exog_seq + n_exog_lag_seq + i]
                for i, k in enumerate(stochastic_carry_vars)
            }
            latent_innov_t = {
                k: seq_args[n_exog_seq + n_exog_lag_seq + n_obs_carry_seq + i]
                for i, k in enumerate(stochastic_latent)
            }
            carry_innov_t = {
                k: seq_args[
                    n_exog_seq
                    + n_exog_lag_seq
                    + n_obs_carry_seq
                    + n_latent_innov_seq
                    + i
                ]
                for i, k in enumerate(stochastic_carry_vars)
            }
            prev_endo = {k: carry_args[i] for i, k in enumerate(endo_keys)}
            prev_adstock_state = {
                k: carry_args[n_endo + i] for i, k in enumerate(adstock_keys)
            }
            prev_exog = lagged_exog_t

            ns_map: dict[str, Any] = {
                name: ns_args[i] for i, name in enumerate(non_seq_names)
            }
            use_observed_carry_t = ns_map["_use_observed_carry"]

            new_endo: dict[str, Any] = {}
            new_adstock = dict(prev_adstock_state)
            carry_mu: dict[str, Any] = {}

            for var in endo_keys:
                beta_names = beta_component_names.get(var)
                beta = (
                    [ns_map[name] for name in beta_names]
                    if beta_names is not None
                    else None
                )

                resolver = _make_scan_resolver(
                    exog_t,
                    new_endo,
                    prev_endo,
                    prev_exog,
                    transform_map,
                    ns_map,
                    new_adstock,
                    n_units,
                )
                mu = build_mu(mu_specs[var], resolver, beta, pt.zeros(n_units))

                if f"alpha_{var}" in ns_map:
                    mu = mu + ns_map[f"alpha_{var}"]
                for svar in slope_vars:
                    skey = f"slope_{var}_{svar}"
                    if skey in ns_map:
                        x_val = exog_t.get(svar, new_endo.get(svar, pt.zeros(n_units)))
                        mu = mu + ns_map[skey] * x_val

                family = families.get(var, "gaussian")
                if var in latent:
                    if var in stochastic_latent_set:
                        sigma_val = ns_map[f"sigma_{var}"]
                        new_endo[var] = mu + sigma_val * latent_innov_t[var]
                    else:
                        new_endo[var] = mu
                elif var in stochastic_carry_vars:
                    carry_mu[var] = mu
                    sigma_val = ns_map[f"sigma_{var}"]
                    sampled_state = mu + sigma_val * carry_innov_t[var]
                    obs_state = obs_carry_t[var]
                    observed_or_sampled = pt.switch(
                        pt.isnan(obs_state), sampled_state, obs_state
                    )
                    new_endo[var] = pt.switch(
                        use_observed_carry_t, observed_or_sampled, sampled_state
                    )
                elif family == "bernoulli":
                    new_endo[var] = 1.0 / (1.0 + pt.exp(-mu))
                elif family in ("poisson", "negbinomial"):
                    new_endo[var] = pt.exp(pt.clip(mu, -20, 20))
                else:
                    new_endo[var] = mu

            out = [new_endo[k] for k in endo_keys]
            out += [new_adstock[k] for k in adstock_keys]
            out += [carry_mu[k] for k in stochastic_carry_vars]
            return out

        results = pytensor.scan(
            fn=step_fn,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_seq_list,
            strict=True,
            return_updates=False,
        )

        if not isinstance(results, list):
            results = [results]

        carry_mu_start = n_carry
        carry_mu_results: dict[str, Any] = {
            var: results[carry_mu_start + i]
            for i, var in enumerate(stochastic_carry_vars)
        }

        # --- emit deterministics and free RVs ---
        for i, var in enumerate(endo_keys):
            carry_all = results[i]  # (n_times, n_units)
            mu_all = carry_mu_results.get(var, carry_all)

            if var in latent:
                if var in stochastic_latent_set:
                    pm.Deterministic(var, mu_all)
                else:
                    pm.Deterministic(f"mu_{var}", mu_all)
                continue

            pm.Deterministic(f"mu_{var}", mu_all)

            family = families.get(var, "gaussian")

            if var in sparse_panel_data:
                sigma = sigma_rvs[var]
                pm.Normal(
                    var,
                    mu=mu_all,
                    sigma=sigma,
                    observed=sparse_panel_data[var],
                )
            elif family == "bernoulli":
                pm.Bernoulli(var, p=mu_all, shape=(n_times, n_units))
            elif family == "poisson":
                pm.Poisson(var, mu=mu_all, shape=(n_times, n_units))
            elif family == "negbinomial":
                alpha_disp = scan_model[f"alpha_disp_{var}"]
                pm.NegativeBinomial(
                    var, mu=mu_all, alpha=alpha_disp, shape=(n_times, n_units)
                )
            elif family == "studentt":
                sigma = sigma_rvs[var]
                nu = scan_model[f"nu_{var}"]
                pm.StudentT(
                    var, nu=nu, mu=mu_all, sigma=sigma, shape=(n_times, n_units)
                )
            else:
                sigma = sigma_rvs[var]
                pm.Normal(var, mu=mu_all, sigma=sigma, shape=(n_times, n_units))

    scan_model._pathmc_panel_scan = PanelScanInfo(
        sort_idx=sort_idx,
        reverse_idx=reverse_idx,
        n_units=n_units,
        n_times=n_times,
        unit_labels=units,
        time_values=time_values,
    )
    return scan_model
