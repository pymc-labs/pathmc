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
"""Regression test for continuous-outcome instrumental-variable estimation.

The IV model is written with zero API change using the ``~~`` correlated-residual
operator (issue #346, epic #343):

    treatment ~ instrument          # first stage
    outcome   ~ treatment           # structural equation (instrument excluded)
    treatment ~~ outcome            # unobserved confounding endogeneity

The ``~~`` block compiles the two equations into a joint ``MvNormal`` whose
``LKJCholeskyCov`` residual correlation soaks up the unobserved confounder. The
structural coefficient on ``treatment`` then recovers the true effect, whereas a
naive ``outcome ~ treatment`` regression is biased by the confounder. This is the
Bayesian limited-information analogue of two-stage least squares (2SLS).

On a fixed-seed simulated dataset with a *known* true effect we assert that the
naive OLS estimate is biased, and that the IV posterior recovers the true effect
and is markedly closer to it than the naive estimate.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc

B_TRUE = 1.5

IV_SPEC = """
T ~ g*Z
Y ~ b*T
T ~~ Y
"""


def _simulate_iv(n=2000, b_true=B_TRUE, seed=20260626):
    """Simulate a confounded system with a valid instrument.

    Structure (U is unobserved):

        Z ~ N(0, 1)                          instrument (exogenous)
        U ~ N(0, 1)                          unobserved confounder
        T = 0.8 Z + 1.0 U + noise            endogenous treatment
        Y = b_true T + 1.5 U + noise         outcome (Z excluded => exclusion)

    Because U drives both T and Y, OLS of Y on T is biased upward. Z moves T but
    affects Y only through T, so it is a valid instrument that identifies b_true.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U = rng.normal(size=n)
    T = 0.8 * Z + 1.0 * U + rng.normal(scale=0.5, size=n)
    Y = b_true * T + 1.5 * U + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"Z": Z, "T": T, "Y": Y})


def _ols_slope(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


@pytest.mark.slow
def test_iv_recovers_true_effect_naive_is_biased():
    df = _simulate_iv()

    # Naive OLS of Y on T is biased upward by the unobserved confounder U.
    naive = _ols_slope(df["Y"].values, df["T"].values)
    assert naive > B_TRUE + 0.2, (
        f"naive OLS should be biased upward by confounding, got {naive:.3f}"
    )

    model = pathmc.model(IV_SPEC, data=df)
    model.fit(random_seed=0, progressbar=False)
    b_post = model._idata.posterior["beta_Y"].sel(Y_predictors="T").values.flatten()

    # IV posterior recovers the true structural effect ...
    assert abs(b_post.mean() - B_TRUE) < 0.3, (
        f"IV posterior mean {b_post.mean():.3f} should recover true effect {B_TRUE}"
    )
    # ... and is markedly closer to the truth than naive OLS.
    assert abs(b_post.mean() - B_TRUE) < abs(naive - B_TRUE)


@pytest.mark.slow
def test_iv_recovers_positive_residual_correlation():
    """Residual correlation should be positive: U inflates both T and Y."""
    df = _simulate_iv()
    model = pathmc.model(IV_SPEC, data=df)
    model.fit(random_seed=0, progressbar=False)
    corr = (
        model._idata
        .posterior["chol_T_Y_corr"]
        .isel(chol_T_Y_corr_dim_0=0, chol_T_Y_corr_dim_1=1)
        .values.flatten()
    )
    assert corr.mean() > 0.1, (
        f"expected positive endogeneity correlation, got {corr.mean():.3f}"
    )
