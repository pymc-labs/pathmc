"""Structural equation compiler: Spec + data -> pm.Model.

Builds a PyMC model with Normal or Bernoulli likelihoods for independent
equations and MvNormal likelihoods (with LKJ-correlated residuals) for
Gaussian variables connected by ``~~``.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import patsy
import pymc as pm

from pathmc.panel import PanelInfo
from pathmc.parse import Regression, Spec


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
    """
    rhs_parts = [t.variable for t in reg.terms]
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
) -> pm.Model:
    """Compile a structural specification into a PyMC model.

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

    Returns
    -------
    pm.Model
        Compiled PyMC model ready for sampling.

    Raises
    ------
    ValueError
        If ``~~`` is used between non-Gaussian variables.
    """
    if families is None:
        families = {}

    _validate_residual_cov_families(spec, families)

    block_vars, blocks = _identify_residual_blocks(spec)

    has_random_intercepts = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)

    unit_idx: np.ndarray | None = None

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        dm = design_matrices[reg.lhs]
        coords[f"{reg.lhs}_predictors"] = list(dm.columns)

    if has_random_intercepts and panel_info is not None:
        coords["unit"] = panel_info.unit_labels
        unit_idx = _build_unit_index(data, panel_info)

    with pm.Model(coords=coords) as pymc_model:
        for reg in spec.regressions:
            if reg.lhs in block_vars:
                continue

            family = families.get(reg.lhs, "gaussian")
            dm = design_matrices[reg.lhs]
            X = dm.values
            y = data[reg.lhs].values

            beta = pm.Normal(
                f"beta_{reg.lhs}",
                mu=0,
                sigma=10,
                dims=f"{reg.lhs}_predictors",
            )
            mu = pm.math.dot(X, beta)

            if (
                has_random_intercepts
                and panel_info is not None
                and unit_idx is not None
            ):
                mu = mu + _compile_random_intercept(reg.lhs, unit_idx)

            if slope_vars and panel_info is not None and unit_idx is not None:
                mu = mu + _compile_random_slopes(reg, slope_vars, data, unit_idx)

            if family == "bernoulli":
                pm.Bernoulli(f"{reg.lhs}_obs", logit_p=mu, observed=y)
            else:
                sigma = pm.HalfNormal(f"sigma_{reg.lhs}", sigma=1)
                pm.Normal(f"{reg.lhs}_obs", mu=mu, sigma=sigma, observed=y)

        for block in blocks:
            _compile_residual_block(block, spec, data, design_matrices, pymc_model)

    return pymc_model


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


def _compile_residual_block(
    block: set[str],
    spec: Spec,
    data: pd.DataFrame,
    design_matrices: dict[str, pd.DataFrame],
    pymc_model: pm.Model,
) -> None:
    """Compile a residual-covariance block as a single MvNormal likelihood."""
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


def _validate_residual_cov_families(spec: Spec, families: dict[str, str]) -> None:
    """Raise if any variable in a ``~~`` pair is non-Gaussian."""
    for rc in spec.residual_covs:
        for var in (rc.var1, rc.var2):
            family = families.get(var, "gaussian")
            if family != "gaussian":
                raise ValueError(
                    f"Residual covariance (~~) requires Gaussian family, "
                    f"but '{var}' has family '{family}'. "
                    f"Covariance modeling is only supported for continuous "
                    f"Gaussian outcomes."
                )
