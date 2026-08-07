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
"""Effect-math oracle tests (issue #327, item 5).

For a linear-Gaussian SCM the direct/indirect/total effect decomposition
has a closed-form analytic answer (Wright's tracing rule): the effect
along a path is the product of the edge coefficients, and the total
effect of X on Y is the sum over all X -> Y paths. These tests check
``PathModel.effect()`` / ``PathModel.effects_summary()`` (path-coefficient
multiplication, see ``pathmc/effects.py``) against:

1.  The known ground-truth coefficients used to generate the data
    (recovery-style, with sampling tolerance).
2.  An exact algebraic identity obtained by pinning the posterior to
    known coefficient values (no sampling noise at all) and checking
    ``compute_path_effect`` reproduces the hand-computed product/sum
    exactly.
3.  An independent computation route already in the codebase --
    ``PathModel.do()``, which propagates interventions through the actual
    PyMC computation graph (graph surgery) rather than by multiplying
    labeled coefficients. For a linear-Gaussian model the do()-based
    slope of Y in X must equal the path-sum total effect exactly, for
    every posterior draw. This is the "chain rule" invariant the #327
    triage flagged as untested.
4.  A negative/limitation case: when an interaction term sits on a path
    edge, the model is no longer linear and the simple product-of-
    coefficients rule no longer captures the true (state-dependent)
    marginal effect. We confirm pathmc now warns about this rather than
    silently returning a wrong number, and we quantify the discrepancy
    against the true do()-based effect at two different reference points.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import pathmc

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# 1. Recovery-style oracle: fitted posterior means close to the true DGP
#    coefficients used to simulate the data.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def oracle_mediation_data():
    """X -> M -> Y chain with known analytic effects.

    True DGP: M = 0.5*X + eps_M, Y = 0.8*M + 0.3*X + eps_Y
    => direct = 0.3, indirect = a*b = 0.5*0.8 = 0.4, total = 0.7
    """
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.3, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.fixture(scope="module")
def oracle_mediation_model(oracle_mediation_data):
    spec = "M ~ a*X\nY ~ b*M + c*X\nindirect := a*b\ntotal := a*b + c\n"
    model = pathmc.model(spec, data=oracle_mediation_data)
    model.fit(draws=200, tune=200, chains=2, random_seed=0, progressbar=False)
    return model


class TestRecoversAnalyticTruth:
    """effect()/effects_summary() recover the known DGP coefficients."""

    def test_direct_effect_recovers_truth(self, oracle_mediation_model):
        direct = oracle_mediation_model.effect("X -> Y")
        assert direct.mean == pytest.approx(0.3, abs=0.1)

    def test_indirect_effect_recovers_truth(self, oracle_mediation_model):
        indirect = oracle_mediation_model.effect("X -> M -> Y")
        assert indirect.mean == pytest.approx(0.4, abs=0.1)

    def test_total_via_chain_rule_recovers_truth(self, oracle_mediation_model):
        direct = oracle_mediation_model.effect("X -> Y")
        indirect = oracle_mediation_model.effect("X -> M -> Y")
        total = direct.mean + indirect.mean
        assert total == pytest.approx(0.7, abs=0.1)

    def test_defined_total_param_matches_analytic_truth(self, oracle_mediation_model):
        summary = oracle_mediation_model.effects_summary()
        assert summary.loc["total", "mean"] == pytest.approx(0.7, abs=0.1)


# ---------------------------------------------------------------------------
# 2 & 3. Exact-identity oracle: pin the posterior to known coefficients
#    (zero sampling noise) and check the path-coefficient math is an exact
#    algebraic identity, cross-validated against the independent do()
#    (graph-surgery) computation route.
# ---------------------------------------------------------------------------


@pytest.fixture
def pinned_mediation_model(oracle_mediation_data, mock_pymc_sample):
    """Mediation model with posterior pinned to exact, noise-free coefficients."""
    spec = "M ~ a*X\nY ~ b*M + c*X\nindirect := a*b\ntotal := a*b + c\n"
    model = pathmc.model(spec, data=oracle_mediation_data)
    model.fit(draws=25, tune=25, chains=2, random_seed=0)

    posterior = model._idata.posterior.copy(deep=True)
    posterior["beta_M"].loc[{"M_predictors": "Intercept"}] = 0.0
    posterior["beta_M"].loc[{"M_predictors": "X"}] = 0.5
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    posterior["beta_Y"].loc[{"Y_predictors": "M"}] = 0.8
    posterior["beta_Y"].loc[{"Y_predictors": "X"}] = 0.3
    posterior["sigma_M"] = posterior["sigma_M"] * 0 + 1e-6
    posterior["sigma_Y"] = posterior["sigma_Y"] * 0 + 1e-6
    model._idata["posterior"] = posterior
    return model


class TestExactChainRuleIdentity:
    """With pinned coefficients a=0.5, b=0.8, c=0.3 every quantity below is
    an exact deterministic function of the same numbers -- no sampling
    noise is involved, so equality should hold to floating-point precision.
    """

    def test_indirect_is_exact_product(self, pinned_mediation_model):
        indirect = pinned_mediation_model.effect("X -> M -> Y")
        assert indirect.mean == pytest.approx(0.5 * 0.8, abs=1e-10)

    def test_direct_is_exact(self, pinned_mediation_model):
        direct = pinned_mediation_model.effect("X -> Y")
        assert direct.mean == pytest.approx(0.3, abs=1e-10)

    def test_total_equals_direct_plus_indirect_exactly(self, pinned_mediation_model):
        direct = pinned_mediation_model.effect("X -> Y")
        indirect = pinned_mediation_model.effect("X -> M -> Y")
        summary = pinned_mediation_model.effects_summary()
        assert summary.loc["total", "mean"] == pytest.approx(
            direct.mean + indirect.mean, abs=1e-10
        )
        assert summary.loc["total", "mean"] == pytest.approx(0.7, abs=1e-10)

    def test_total_matches_independent_do_based_slope(self, pinned_mediation_model):
        """Cross-check the path-multiplication total against do()-based graph
        surgery -- a completely independent code path (pm.do + deterministic
        mean propagation) that does not multiply labeled coefficients at all.
        For a linear-Gaussian SCM these must agree exactly, per posterior draw.
        """
        model = pinned_mediation_model
        r_lo = model.do(set={"X": 0.0}, kind="mean")
        r_hi = model.do(set={"X": 1.0}, kind="mean")
        do_slope = (r_hi - r_lo).draws("Y")  # dY/dX per posterior draw

        direct = model.effect("X -> Y").draws
        indirect = model.effect("X -> M -> Y").draws
        path_total = direct + indirect

        np.testing.assert_allclose(do_slope, path_total, atol=1e-8)
        np.testing.assert_allclose(do_slope, 0.7, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Parallel mediators oracle (multiple indirect paths).
# ---------------------------------------------------------------------------


@pytest.fixture
def pinned_parallel_model(mock_pymc_sample):
    """T -> M1 -> Y, T -> M2 -> Y, T -> Y with pinned exact coefficients.

    True: a1=0.6, a2=0.4, b1=0.5, b2=0.3, c=0.2
    total = c + a1*b1 + a2*b2 = 0.2 + 0.30 + 0.12 = 0.62
    """
    rng = np.random.default_rng(7)
    n = 200
    T = rng.normal(size=n)
    M1 = 0.6 * T + rng.normal(scale=0.1, size=n)
    M2 = 0.4 * T + rng.normal(scale=0.1, size=n)
    Y = 0.5 * M1 + 0.3 * M2 + 0.2 * T + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"T": T, "M1": M1, "M2": M2, "Y": Y})

    spec = (
        "M1 ~ a1*T\nM2 ~ a2*T\nY ~ b1*M1 + b2*M2 + c*T\n"
        "indirect1 := a1*b1\nindirect2 := a2*b2\ntotal := c + a1*b1 + a2*b2\n"
    )
    model = pathmc.model(spec, data=df)
    model.fit(draws=25, tune=25, chains=2, random_seed=0)

    posterior = model._idata.posterior.copy(deep=True)
    posterior["beta_M1"].loc[{"M1_predictors": "Intercept"}] = 0.0
    posterior["beta_M1"].loc[{"M1_predictors": "T"}] = 0.6
    posterior["beta_M2"].loc[{"M2_predictors": "Intercept"}] = 0.0
    posterior["beta_M2"].loc[{"M2_predictors": "T"}] = 0.4
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    posterior["beta_Y"].loc[{"Y_predictors": "M1"}] = 0.5
    posterior["beta_Y"].loc[{"Y_predictors": "M2"}] = 0.3
    posterior["beta_Y"].loc[{"Y_predictors": "T"}] = 0.2
    for v in ["sigma_M1", "sigma_M2", "sigma_Y"]:
        posterior[v] = posterior[v] * 0 + 1e-6
    model._idata["posterior"] = posterior
    return model


class TestParallelMediatorsExactIdentity:
    def test_total_sums_all_paths_exactly(self, pinned_parallel_model):
        summary = pinned_parallel_model.effects_summary()
        assert summary.loc["total", "mean"] == pytest.approx(0.62, abs=1e-10)

    def test_total_matches_do_based_slope(self, pinned_parallel_model):
        model = pinned_parallel_model
        r_lo = model.do(set={"T": 0.0}, kind="mean")
        r_hi = model.do(set={"T": 1.0}, kind="mean")
        do_slope = (r_hi - r_lo).draws("Y")
        np.testing.assert_allclose(do_slope, 0.62, atol=1e-8)


# ---------------------------------------------------------------------------
# 5. Limitation / negative oracle: interaction terms break the linear
#    chain-rule assumption. compute_path_effect() must warn rather than
#    silently return an incomplete number.
# ---------------------------------------------------------------------------


@pytest.fixture
def pinned_interaction_model(mock_pymc_sample):
    """X -> M -> Y with an interaction M:X on Y's regression.

    Y = b*M + g*(M*X) + c*X, M = a*X (deterministic, no noise), so
    Y = c*X + b*a*X + g*a*X**2 (+ noise).
    The true marginal effect dY/dX = c + b*a + 2*g*a*X depends on X:
    it is state-dependent, unlike the constant path-sum a*b + c.
    """
    rng = np.random.default_rng(3)
    n = 200
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.05, size=n)
    Y = 0.8 * M + 0.3 * X + 0.4 * M * X + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame({"X": X, "M": M, "Y": Y})

    spec = "M ~ a*X\nY ~ b*M + g*M:X + c*X\n"
    model = pathmc.model(spec, data=df)
    model.fit(draws=25, tune=25, chains=2, random_seed=0)

    posterior = model._idata.posterior.copy(deep=True)
    posterior["beta_M"].loc[{"M_predictors": "Intercept"}] = 0.0
    posterior["beta_M"].loc[{"M_predictors": "X"}] = 0.5
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    posterior["beta_Y"].loc[{"Y_predictors": "M"}] = 0.8
    posterior["beta_Y"].loc[{"Y_predictors": "M:X"}] = 0.4
    posterior["beta_Y"].loc[{"Y_predictors": "X"}] = 0.3
    posterior["sigma_M"] = posterior["sigma_M"] * 0 + 1e-6
    posterior["sigma_Y"] = posterior["sigma_Y"] * 0 + 1e-6
    model._idata["posterior"] = posterior
    return model


class TestInteractionBreaksLinearChainRule:
    def test_path_effect_warns_about_omitted_interaction(
        self, pinned_interaction_model
    ):
        with pytest.warns(UserWarning, match="interaction term"):
            pinned_interaction_model.effect("X -> M -> Y")

    def test_path_sum_does_not_equal_do_based_effect_away_from_zero(
        self, pinned_interaction_model
    ):
        """The path-coefficient total (a*b + c = 0.5*0.8 + 0.3 = 0.7) only
        matches the true derivative at X=0. Away from X=0 the interaction
        term makes the true effect state-dependent and larger in magnitude,
        which the naive path-sum misses entirely -- this is exactly the
        gap the #327 triage flagged ("no oracle test ... for nonlinear
        chains").
        """
        model = pinned_interaction_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            direct = model.effect("X -> Y").mean
            indirect = model.effect("X -> M -> Y").mean
        path_total = direct + indirect
        assert path_total == pytest.approx(0.7, abs=1e-10)

        # True local slope at X=0: dY/dX = c + b*a + 2*g*a*0 = 0.3 + 0.4 = 0.7
        r_lo = model.do(set={"X": -0.001}, kind="mean")
        r_hi = model.do(set={"X": 0.001}, kind="mean")
        slope_at_zero = (r_hi - r_lo).mean("Y") / 0.002
        assert slope_at_zero == pytest.approx(path_total, abs=1e-3)

        # True local slope at X=2: dY/dX = 0.3 + 0.4 + 2*0.4*0.5*2 = 1.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_lo = model.do(set={"X": 1.999}, kind="mean")
            r_hi = model.do(set={"X": 2.001}, kind="mean")
        slope_at_two = (r_hi - r_lo).mean("Y") / 0.002

        # The naive path-sum is constant and wrong away from X=0: it
        # understates the true (state-dependent) effect by roughly the
        # interaction contribution 2*g*a*X = 0.8.
        assert slope_at_two == pytest.approx(1.5, abs=1e-2)
        assert abs(slope_at_two - path_total) > 0.5
