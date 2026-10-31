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


@pytest.fixture(scope="module")
def iv_data():
    return _simulate_iv()


@pytest.fixture(scope="module")
def fitted_iv(iv_data):
    """Fit the IV model once for the whole module (MCMC is expensive)."""
    model = pathmc.model(IV_SPEC, data=iv_data)
    model.fit(random_seed=0, progressbar=False)
    return model


def _late_draws(model):
    return model._idata.posterior["beta_Y"].sel(Y_predictors="T").values.flatten()


@pytest.mark.slow
def test_iv_recovers_true_effect_naive_is_biased(iv_data, fitted_iv):
    # Naive OLS of Y on T is biased upward by the unobserved confounder U.
    naive = _ols_slope(iv_data["Y"].values, iv_data["T"].values)
    assert naive > B_TRUE + 0.2, (
        f"naive OLS should be biased upward by confounding, got {naive:.3f}"
    )

    b_post = _late_draws(fitted_iv)

    # IV posterior recovers the true structural effect ...
    assert abs(b_post.mean() - B_TRUE) < 0.3, (
        f"IV posterior mean {b_post.mean():.3f} should recover true effect {B_TRUE}"
    )
    # ... and is markedly closer to the truth than naive OLS.
    assert abs(b_post.mean() - B_TRUE) < abs(naive - B_TRUE)


@pytest.mark.slow
def test_naive_estimate_falls_outside_the_iv_interval(iv_data, fitted_iv):
    """The bias is not just a shift in the mean: the confounded estimate is
    excluded by the IV posterior interval."""
    naive = _ols_slope(iv_data["Y"].values, iv_data["T"].values)
    lo, hi = np.percentile(_late_draws(fitted_iv), [3, 97])
    assert not (lo <= naive <= hi), (
        f"naive OLS {naive:.3f} should fall outside the IV 94% interval "
        f"[{lo:.3f}, {hi:.3f}]"
    )


@pytest.mark.slow
def test_first_stage_recovers_instrument_relevance(fitted_iv):
    """Relevance is the one IV condition that *is* testable from data."""
    g_post = fitted_iv._idata.posterior["beta_T"].sel(T_predictors="Z").values.flatten()
    assert abs(g_post.mean() - 0.8) < 0.15, (
        f"first stage should recover 0.8, got {g_post.mean():.3f}"
    )


@pytest.mark.slow
def test_iv_recovers_positive_residual_correlation(fitted_iv):
    """Residual correlation should be positive: U inflates both T and Y."""
    corr = (
        fitted_iv._idata
        .posterior["chol_T_Y_corr"]
        .isel(chol_T_Y_corr_dim_0=0, chol_T_Y_corr_dim_1=1)
        .values.flatten()
    )
    assert corr.mean() > 0.1, (
        f"expected positive endogeneity correlation, got {corr.mean():.3f}"
    )


@pytest.mark.slow
def test_iv_model_converged(fitted_iv):
    """Guard the recovery assertions above against a silently bad fit."""
    import arviz as az

    idata = fitted_iv._idata
    summary = az.summary(idata, var_names=["beta_T", "beta_Y"], round_to="none")
    assert summary["r_hat"].max() < 1.05, (
        f"max r_hat {summary['r_hat'].max():.3f} indicates non-convergence"
    )
    n_div = int(idata.sample_stats["diverging"].sum())
    assert n_div == 0, f"expected no divergences, got {n_div}"


@pytest.mark.slow
def test_residual_block_is_not_backdoor_identifiable(iv_data):
    """``~~`` declares unobserved confounding, so the backdoor criterion
    must report the effect as unidentifiable (#344)."""
    model = pathmc.model(IV_SPEC, data=iv_data)
    assert not model.is_identifiable("T", "Y")
