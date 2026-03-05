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

import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import patsy
import pymc as pm

from pathmc.graph import GraphInfo
from pathmc.panel import PanelInfo
from pathmc.parse import Regression, Spec, TransformCall
from pathmc.transforms import get_transform


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


def get_predictor_columns(reg: Regression) -> list[str]:
    """Return predictor column names for a regression equation.

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


def _term_base_vars(term: Any) -> list[str]:
    """Return the base variable names a term depends on.

    For interaction terms, returns the constituent variables.
    For plain terms, returns a single-element list.
    """
    if getattr(term, "interaction_of", None) is not None:
        return list(term.interaction_of)
    return [term.variable]


def build_design_matrix(reg: Regression, data: pd.DataFrame) -> pd.DataFrame:
    """Build a patsy design matrix for a single regression equation.

    Parameters
    ----------
    reg : Regression
        Parsed regression with terms and intercept flag.
    data : pd.DataFrame
        Observed data containing the predictor columns.

    Returns
    -------
    pd.DataFrame
        Design matrix with named columns (including ``Intercept`` when applicable).
        For equations with latent (unobserved) parents, returns a DataFrame with
        correct column names but NaN for the latent columns.
    """
    rhs_parts = [t.variable for t in reg.terms]

    missing: list[str] = []
    for term in reg.terms:
        for v in _term_base_vars(term):
            if v not in data.columns and v not in missing:
                missing.append(v)

    if missing:
        cols = get_predictor_columns(reg)
        dm = pd.DataFrame(index=range(len(data)), columns=cols, dtype=float)
        if reg.has_intercept:
            dm["Intercept"] = 1.0
        for term in reg.terms:
            v = term.variable
            if term.interaction_of is not None:
                product = np.ones(len(data))
                for part in term.interaction_of:
                    if part in data.columns:
                        product = product * data[part].to_numpy(dtype=float)
                    else:
                        product = product * np.nan
                dm[v] = product
            elif v in data.columns:
                dm[v] = data[v].values
            else:
                dm[v] = np.nan
        return dm

    if reg.has_intercept:
        formula_str = " + ".join(rhs_parts)
    else:
        formula_str = "0 + " + " + ".join(rhs_parts)

    dm = patsy.dmatrix(formula_str, data=data, return_type="dataframe")
    return dm


def compile_to_pymc(
    spec: Spec,
    data: pd.DataFrame,
    design_matrices: dict[str, pd.DataFrame],
    families: dict[str, str] | None = None,
    panel_info: PanelInfo | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    graph_info: GraphInfo | None = None,
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
    data : pd.DataFrame
        Observed data.
    design_matrices : dict[str, pd.DataFrame]
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
    if families is None:
        families = {}
    if latent is None:
        latent = set()

    if graph_info is None:
        from pathmc.graph import build_graph

        graph_info = build_graph(spec, latent=latent)

    _validate_residual_cov_families(spec, families)

    if panel_info is not None and _has_temporal_deps(spec, graph_info):
        return _compile_scan_panel(
            spec=spec,
            data=data,
            design_matrices=design_matrices,
            families=families,
            panel_info=panel_info,
            pooling=pooling,
            latent=latent,
            graph_info=graph_info,
        )

    block_vars, blocks = _identify_residual_blocks(spec)

    has_random_intercepts = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    unit_idx: np.ndarray | None = None

    reg_by_lhs = {r.lhs: r for r in spec.regressions}

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        coords[f"{reg.lhs}_predictors"] = get_predictor_columns(reg)

    if has_random_intercepts and panel_info is not None:
        coords["unit"] = panel_info.unit_labels
        unit_idx = _build_unit_index(data, panel_info)

    transform_map = _build_transform_map(spec)

    with pm.Model(coords=coords) as pymc_model:
        transform_param_rvs = _emit_transform_priors(spec, transform_map)

        data_vars: dict[str, Any] = {}
        for var in graph_info.topological_order:
            if var in graph_info.exogenous and var in data.columns:
                data_vars[var] = pm.Data(var, data[var].values.astype(float))

        endogenous_rvs: dict[str, Any] = {}

        for var in graph_info.topological_order:
            if var not in reg_by_lhs:
                continue
            if var in block_vars:
                continue

            reg = reg_by_lhs[var]
            family = families.get(var, "gaussian")
            cols = get_predictor_columns(reg)

            beta = pm.Normal(
                f"beta_{var}",
                mu=0,
                sigma=10,
                dims=f"{var}_predictors",
            )

            mu = _build_mu_symbolic(
                reg,
                cols,
                beta,
                data,
                data_vars,
                endogenous_rvs,
                transform_map,
                transform_param_rvs,
                panel_info,
            )

            if (
                has_random_intercepts
                and panel_info is not None
                and unit_idx is not None
            ):
                mu = mu + _compile_random_intercept(var, unit_idx)

            if slope_vars and panel_info is not None and unit_idx is not None:
                mu = mu + _compile_random_slopes(reg, slope_vars, data_vars, unit_idx)

            mu_det = pm.Deterministic(f"mu_{var}", mu)

            rv = _emit_free_rv(var, mu_det, family, latent)
            endogenous_rvs[var] = rv

        for block in blocks:
            _compile_residual_block(
                block, spec, data, design_matrices, pymc_model, endogenous_rvs
            )

    return pymc_model


# ---------------------------------------------------------------------------
# Helpers: mu computation
# ---------------------------------------------------------------------------


def _build_mu_symbolic(
    reg: Regression,
    cols: list[str],
    beta: Any,
    data: pd.DataFrame,
    data_vars: dict[str, Any],
    endogenous_rvs: dict[str, Any],
    transform_map: dict[str, TransformCall],
    transform_param_rvs: dict[str, Any],
    panel_info: PanelInfo | None,
) -> Any:
    """Build linear predictor wired through the generative graph.

    Exogenous parents use ``pm.Data``. Endogenous parents use the
    upstream free RV, so ``pm.do()`` on any ancestor naturally
    propagates through the causal chain.
    """
    import pytensor.tensor as pt

    n_obs = len(data)
    mu = pt.zeros(n_obs)

    for i, col in enumerate(cols):
        coef = beta[i]
        if col == "Intercept":
            mu = mu + coef
        elif col in transform_map:
            tc = transform_map[col]
            transformed = _apply_transform_chain(
                tc,
                data,
                transform_param_rvs,
                panel_info=panel_info,
                data_vars=data_vars,
                endogenous_rvs=endogenous_rvs,
            )
            mu = mu + coef * transformed
        elif ":" in col:
            product = _resolve_interaction_symbolic(
                col, data, data_vars, endogenous_rvs
            )
            mu = mu + coef * product
        elif col in endogenous_rvs:
            mu = mu + coef * endogenous_rvs[col]
        elif col in data_vars:
            mu = mu + coef * data_vars[col]
        else:
            mu = mu + coef * pt.as_tensor_variable(data[col].values.astype(float))

    return mu


def _resolve_interaction_symbolic(
    col: str,
    data: pd.DataFrame,
    data_vars: dict[str, Any],
    endogenous_rvs: dict[str, Any],
) -> Any:
    """Compute the symbolic product for an interaction column like ``X:Z``.

    Resolves each constituent variable through the generative graph
    (``pm.Data`` for exogenous, upstream free RV for endogenous) so
    that ``pm.do()`` interventions propagate correctly.
    """
    import pytensor.tensor as pt

    parts = col.split(":")
    values = []
    for part in parts:
        if part in endogenous_rvs:
            values.append(endogenous_rvs[part])
        elif part in data_vars:
            values.append(data_vars[part])
        else:
            values.append(pt.as_tensor_variable(data[part].values.astype(float)))
    product = values[0]
    for v in values[1:]:
        product = product * v
    return product


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


def _build_unit_index(data: pd.DataFrame, panel_info: PanelInfo) -> np.ndarray:
    """Map each row to an integer unit index."""
    label_to_idx = {label: i for i, label in enumerate(panel_info.unit_labels)}
    return data[panel_info.unit].map(label_to_idx).to_numpy()


def _compile_random_intercept(var: str, unit_idx: np.ndarray) -> Any:
    """Emit hierarchical random intercept for *var*, return alpha[unit_idx]."""
    mu_alpha = pm.Normal(f"mu_alpha_{var}", mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal(f"sigma_alpha_{var}", sigma=1)
    alpha = pm.Normal(f"alpha_{var}", mu=mu_alpha, sigma=sigma_alpha, dims="unit")
    return alpha[unit_idx]


def _compile_random_slopes(
    reg: Regression,
    slope_vars: list[str],
    data_vars: dict[str, Any],
    unit_idx: np.ndarray,
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
        mu_slope = pm.Normal(f"mu_slope_{reg.lhs}_{svar}", mu=0, sigma=10)
        sigma_slope = pm.HalfNormal(f"sigma_slope_{reg.lhs}_{svar}", sigma=1)
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
    spec: Spec,
    data: pd.DataFrame,
    design_matrices: dict[str, pd.DataFrame],
    pymc_model: pm.Model,
    endogenous_rvs: dict[str, Any] | None = None,
) -> None:
    """Compile a residual-covariance block as a single MvNormal.

    Residual blocks use data-based design matrices (not symbolic wiring)
    and are emitted as observed MvNormal. This is kept as observed because
    MvNormal blocks cannot easily be split into separate free RVs for
    ``pm.observe``.
    """
    if endogenous_rvs is None:
        endogenous_rvs = {}

    block_sorted = sorted(block)
    k = len(block_sorted)

    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    block_regs = [reg_by_lhs[v] for v in block_sorted]

    mus = []
    for reg in block_regs:
        dm = design_matrices[reg.lhs]
        X = dm.values
        beta = pm.Normal(
            f"beta_{reg.lhs}",
            mu=0,
            sigma=10,
            dims=f"{reg.lhs}_predictors",
        )
        mus.append(pm.math.dot(X, beta))

    mu_stacked = pm.math.stack(mus, axis=1)
    y_stacked = np.column_stack([data[v].to_numpy() for v in block_sorted])

    block_name = "_".join(block_sorted)
    chol, _, _ = pm.LKJCholeskyCov(
        f"chol_{block_name}",
        n=k,
        eta=2.0,
        sd_dist=pm.HalfNormal.dist(1.0),
        compute_corr=True,
    )
    pm.MvNormal(f"{block_name}_obs", mu=mu_stacked, chol=chol, observed=y_stacked)


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


def _emit_free_rv(var: str, mu: Any, family: str, latent: set[str]) -> Any:
    """Emit a free random variable for an endogenous variable.

    Latent variables are emitted as ``pm.Deterministic`` (no noise).
    Observed variables are emitted as free RVs (conditioned on data
    externally via ``pm.observe()``).

    Returns the RV tensor so downstream equations can wire through it.
    """
    if var in latent:
        return mu

    if family == "bernoulli":
        return pm.Bernoulli(var, logit_p=mu)
    if family == "poisson":
        return pm.Poisson(var, mu=pm.math.exp(mu))
    if family == "negbinomial":
        alpha_disp = pm.HalfNormal(f"alpha_disp_{var}", sigma=1)
        return pm.NegativeBinomial(var, mu=pm.math.exp(mu), alpha=alpha_disp)
    if family == "studentt":
        sigma = pm.HalfNormal(f"sigma_{var}", sigma=1)
        nu = pm.Gamma(f"nu_{var}", alpha=2, beta=0.1)
        return pm.StudentT(var, nu=nu, mu=mu, sigma=sigma)

    sigma = pm.HalfNormal(f"sigma_{var}", sigma=1)
    return pm.Normal(var, mu=mu, sigma=sigma)


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
) -> dict[str, Any]:
    """Emit PyMC priors for all transform parameters. Returns name->RV mapping."""
    emitted: dict[str, Any] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.transform is not None:
                _emit_transform_call_priors(term.transform, emitted)
    return emitted


def _emit_transform_call_priors(
    tc: TransformCall,
    emitted: dict[str, Any],
) -> None:
    """Recursively emit priors for a (possibly nested) TransformCall."""
    if isinstance(tc.input_expr, TransformCall):
        _emit_transform_call_priors(tc.input_expr, emitted)

    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in emitted:
            spec = transform.param_specs[param_key]
            emitted[param_name] = transform.emit_prior(param_name, spec)


def _apply_transform_chain(
    tc: TransformCall,
    data: pd.DataFrame,
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
                data[tc.input_expr].values.astype(float)
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


# ---------------------------------------------------------------------------
# Temporal dependency detection
# ---------------------------------------------------------------------------


_LAG_RE = re.compile(r"^(.+)_lag(\d+)$")


def _parse_lag(col: str) -> tuple[str, int] | None:
    """Parse ``"var_lag1"`` → ``("var", 1)``, or ``None``."""
    m = _LAG_RE.match(col)
    if m:
        return m.group(1), int(m.group(2))
    return None


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
    - Legacy ``_lag\\d+$`` column names (backward compat)
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
            if _parse_lag(term.variable) is not None:
                return True
    return False


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
    data_sorted: pd.DataFrame,
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
    data: pd.DataFrame,
    design_matrices: dict[str, pd.DataFrame],
    families: dict[str, str],
    panel_info: PanelInfo,
    pooling: str | dict | None,
    latent: set[str],
    graph_info: GraphInfo,
) -> pm.Model:
    """Compile a panel model with temporal deps using ``pytensor.scan``.

    The generative model encodes the full temporal structure so that
    ``pm.do()`` handles interventions natively.  Free RVs have shape
    ``(n_times, n_units)`` in unit-major sorted order.
    """
    import pytensor
    import pytensor.tensor as pt

    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    transform_map = _build_transform_map(spec)
    has_ri = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    # --- sort data ---
    unit_col = panel_info.unit
    time_col = panel_info.time
    data_sorted = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    sort_idx = np.array(data.sort_values([unit_col, time_col]).index)
    reverse_idx = np.argsort(sort_idx)
    units = panel_info.unit_labels
    n_units = len(units)
    n_times = len(data) // n_units
    time_values = sorted(data_sorted[time_col].unique())

    # --- classify columns ---
    lag_map = _build_lag_map(spec)

    pure_exog = [
        v
        for v in graph_info.topological_order
        if v in graph_info.exogenous
        and _parse_lag(v) is None
        and v not in lag_map
        and v in data_sorted.columns
    ]
    lag_cols: dict[str, tuple[str, int]] = {}
    # Regex-based lag detection (backward compat with _lag\d+$ columns)
    for v in graph_info.topological_order:
        if v in graph_info.exogenous:
            parsed = _parse_lag(v)
            if parsed is not None:
                lag_cols[v] = parsed
    # AST-driven lag detection (from lag() DSL syntax)
    for col_name, base_var in lag_map.items():
        if col_name not in lag_cols:
            lag_cols[col_name] = (base_var, 1)

    for reg in spec.regressions:
        for term in reg.terms:
            lag = _parse_lag(term.variable)
            if lag is not None and lag[1] > 1:
                raise NotImplementedError(
                    f"Scan-compiled panel models only support lag order 1, "
                    f"but '{term.variable}' has lag order {lag[1]}. "
                    f"Use lag1 terms or file a feature request for higher-order lags."
                )

    adstock_cols = [col for col, tc in transform_map.items() if _has_adstock(tc)]

    # --- coords ---
    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        coords[f"{reg.lhs}_predictors"] = get_predictor_columns(reg)
    if has_ri:
        coords["unit"] = units

    with pm.Model(coords=coords) as scan_model:
        # --- transform parameter priors ---
        tparam_rvs: dict[str, Any] = {}
        for reg in spec.regressions:
            for term in reg.terms:
                if term.transform is not None:
                    _emit_transform_call_priors(term.transform, tparam_rvs)

        # --- regression parameter priors ---
        beta_rvs: dict[str, Any] = {}
        sigma_rvs: dict[str, Any] = {}
        for var in endogenous_order:
            family = families.get(var, "gaussian")
            beta_rvs[var] = pm.Normal(
                f"beta_{var}", mu=0, sigma=10, dims=f"{var}_predictors"
            )
            if var not in latent:
                if family in ("gaussian", "studentt"):
                    sigma_rvs[var] = pm.HalfNormal(f"sigma_{var}", sigma=1)
                if family == "studentt":
                    pm.Gamma(f"nu_{var}", alpha=2, beta=0.1)
                if family == "negbinomial":
                    pm.HalfNormal(f"alpha_disp_{var}", sigma=1)

        # --- random effects ---
        alpha_rvs: dict[str, Any] = {}
        slope_rvs: dict[str, dict[str, Any]] = {}
        if has_ri:
            for var in endogenous_order:
                mu_a = pm.Normal(f"mu_alpha_{var}", mu=0, sigma=10)
                sig_a = pm.HalfNormal(f"sigma_alpha_{var}", sigma=1)
                alpha_rvs[var] = pm.Normal(
                    f"alpha_{var}", mu=mu_a, sigma=sig_a, dims="unit"
                )
        for var in endogenous_order:
            reg = reg_by_lhs[var]
            term_variables = {t.variable for t in reg.terms}
            slope_rvs[var] = {}
            for svar in slope_vars:
                if svar in term_variables:
                    mu_s = pm.Normal(f"mu_slope_{var}_{svar}", mu=0, sigma=10)
                    sig_s = pm.HalfNormal(f"sigma_slope_{var}_{svar}", sigma=1)
                    slope_rvs[var][svar] = pm.Normal(
                        f"slope_{var}_{svar}", mu=mu_s, sigma=sig_s, dims="unit"
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
        exog_lag_bases = sorted(
            {base for _col, (base, _k) in lag_cols.items() if base not in endo_set}
        )

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

        sequences = [exog_data_nodes[k] for k in exog_keys]

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
            + [_init_carry(init_exog_lag[k]) for k in exog_lag_bases]
        )

        # Non-sequences: all parameters
        non_seq_list: list[Any] = []
        non_seq_names: list[str] = []
        for var in endo_keys:
            non_seq_list.append(beta_rvs[var])
            non_seq_names.append(f"beta_{var}")
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

        n_seq = len(sequences)
        n_endo = len(endo_keys)
        n_adstock = len(adstock_keys)
        n_exog_lag = len(exog_lag_bases)
        n_carry = n_endo + n_adstock + n_exog_lag

        def step_fn(*args: Any) -> list[Any]:
            seq_args = args[:n_seq]
            carry_args = args[n_seq : n_seq + n_carry]
            ns_args = args[n_seq + n_carry :]

            exog_t = {k: seq_args[i] for i, k in enumerate(exog_keys)}
            prev_endo = {k: carry_args[i] for i, k in enumerate(endo_keys)}
            prev_adstock_state = {
                k: carry_args[n_endo + i] for i, k in enumerate(adstock_keys)
            }
            prev_exog = {
                k: carry_args[n_endo + n_adstock + i]
                for i, k in enumerate(exog_lag_bases)
            }

            ns_map: dict[str, Any] = {
                name: ns_args[i] for i, name in enumerate(non_seq_names)
            }

            new_endo: dict[str, Any] = {}
            new_adstock = dict(prev_adstock_state)

            for var in endo_keys:
                cols = get_predictor_columns(reg_by_lhs[var])
                beta = ns_map[f"beta_{var}"]
                mu = pt.zeros(n_units)

                for ci, col in enumerate(cols):
                    coef = beta[ci]
                    if col == "Intercept":
                        mu = mu + coef
                    elif col in transform_map:
                        tc = transform_map[col]
                        inp_name = _get_adstock_input(tc)
                        if inp_name in new_endo:
                            raw = new_endo[inp_name]
                        elif inp_name in exog_t:
                            raw = exog_t[inp_name]
                        else:
                            raw = pt.zeros(n_units)
                        transformed, new_adstock = _apply_step_transform(
                            tc, raw, new_adstock, ns_map, col
                        )
                        mu = mu + coef * transformed
                    elif ":" in col:
                        parts = col.split(":")
                        vals = []
                        for part in parts:
                            if part in new_endo:
                                vals.append(new_endo[part])
                            elif part in exog_t:
                                vals.append(exog_t[part])
                            else:
                                vals.append(pt.zeros(n_units))
                        product = vals[0]
                        for v in vals[1:]:
                            product = product * v
                        mu = mu + coef * product
                    elif col in new_endo:
                        mu = mu + coef * new_endo[col]
                    elif col in exog_t:
                        mu = mu + coef * exog_t[col]
                    else:
                        lag = _parse_lag(col)
                        if lag is None and col in lag_map:
                            lag = (lag_map[col], 1)
                        if lag is not None:
                            base_var, _lag_k = lag
                            if base_var in prev_endo:
                                mu = mu + coef * prev_endo[base_var]
                            elif base_var in prev_exog:
                                mu = mu + coef * prev_exog[base_var]

                if f"alpha_{var}" in ns_map:
                    mu = mu + ns_map[f"alpha_{var}"]
                for svar in slope_vars:
                    skey = f"slope_{var}_{svar}"
                    if skey in ns_map:
                        x_val = exog_t.get(svar, new_endo.get(svar, pt.zeros(n_units)))
                        mu = mu + ns_map[skey] * x_val

                family = families.get(var, "gaussian")
                if var in latent:
                    new_endo[var] = mu
                elif family == "bernoulli":
                    new_endo[var] = 1.0 / (1.0 + pt.exp(-mu))
                elif family in ("poisson", "negbinomial"):
                    new_endo[var] = pt.exp(pt.clip(mu, -20, 20))
                else:
                    new_endo[var] = mu

            out = [new_endo[k] for k in endo_keys]
            out += [new_adstock[k] for k in adstock_keys]
            out += [exog_t.get(k, pt.zeros(n_units)) for k in exog_lag_bases]
            return out

        results, _updates = pytensor.scan(
            fn=step_fn,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_seq_list,
            strict=True,
        )

        if not isinstance(results, list):
            results = [results]

        # --- emit deterministics and free RVs ---
        for i, var in enumerate(endo_keys):
            mu_all = results[i]  # (n_times, n_units)
            pm.Deterministic(f"mu_{var}", mu_all)

            if var in latent:
                continue

            family = families.get(var, "gaussian")
            if family == "bernoulli":
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
