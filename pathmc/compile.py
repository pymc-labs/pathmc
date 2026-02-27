"""Gaussian compiler: Spec + data -> pm.Model.

Builds a PyMC model with one Normal likelihood per endogenous variable,
weakly informative coefficient priors, and HalfNormal scale priors.
"""

from __future__ import annotations

from typing import Any

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

    Returns
    -------
    pm.Model
        Compiled PyMC model ready for sampling.
    """
    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        dm = design_matrices[reg.lhs]
        coords[f"{reg.lhs}_predictors"] = list(dm.columns)

    with pm.Model(coords=coords) as pymc_model:
        for reg in spec.regressions:
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

    return pymc_model
