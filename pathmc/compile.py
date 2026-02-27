"""Gaussian compiler: Spec + data -> pm.Model.

Builds a PyMC model with Normal likelihoods for independent equations
and MvNormal likelihoods (with LKJ-correlated residuals) for variables
connected by ``~~``.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import patsy
import pymc as pm

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
        for all variables. Currently only ``"gaussian"`` is supported.

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

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        dm = design_matrices[reg.lhs]
        coords[f"{reg.lhs}_predictors"] = list(dm.columns)

    with pm.Model(coords=coords) as pymc_model:
        for reg in spec.regressions:
            if reg.lhs in block_vars:
                continue

            dm = design_matrices[reg.lhs]
            X = dm.values
            y = data[reg.lhs].values

            beta = pm.Normal(
                f"beta_{reg.lhs}",
                mu=0,
                sigma=10,
                dims=f"{reg.lhs}_predictors",
            )
            sigma = pm.HalfNormal(f"sigma_{reg.lhs}", sigma=1)
            mu = pm.math.dot(X, beta)
            pm.Normal(f"{reg.lhs}_obs", mu=mu, sigma=sigma, observed=y)

        for block in blocks:
            _compile_residual_block(block, spec, data, design_matrices, pymc_model)

    return pymc_model


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
