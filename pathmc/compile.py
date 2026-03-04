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
"""

from __future__ import annotations

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
    missing = [v for v in rhs_parts if v not in data.columns]

    if missing:
        cols = get_predictor_columns(reg)
        dm = pd.DataFrame(index=range(len(data)), columns=cols, dtype=float)
        if reg.has_intercept:
            dm["Intercept"] = 1.0
        for v in rhs_parts:
            if v in data.columns:
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

    _validate_residual_cov_families(spec, families)

    block_vars, blocks = _identify_residual_blocks(spec)

    has_random_intercepts = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    unit_idx: np.ndarray | None = None

    if graph_info is None:
        from pathmc.graph import build_graph

        graph_info = build_graph(spec, latent=latent)

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
                mu = mu + _compile_random_slopes(reg, slope_vars, data, unit_idx)

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
        elif col in endogenous_rvs:
            mu = mu + coef * endogenous_rvs[col]
        elif col in data_vars:
            mu = mu + coef * data_vars[col]
        else:
            mu = mu + coef * pt.as_tensor_variable(data[col].values.astype(float))

    return mu


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
    return data[panel_info.unit].map(label_to_idx).values


def _compile_random_intercept(var: str, unit_idx: np.ndarray) -> Any:
    """Emit hierarchical random intercept for *var*, return alpha[unit_idx]."""
    mu_alpha = pm.Normal(f"mu_alpha_{var}", mu=0, sigma=10)
    sigma_alpha = pm.HalfNormal(f"sigma_alpha_{var}", sigma=1)
    alpha = pm.Normal(f"alpha_{var}", mu=mu_alpha, sigma=sigma_alpha, dims="unit")
    return alpha[unit_idx]


def _compile_random_slopes(
    reg: Regression,
    slope_vars: list[str],
    data: pd.DataFrame,
    unit_idx: np.ndarray,
) -> Any:
    """Emit hierarchical random slopes for specified predictors."""
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
        x_vals = data[svar].values
        contribution = contribution + slope[unit_idx] * x_vals
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
    y_stacked = np.column_stack([data[v].values for v in block_sorted])

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
