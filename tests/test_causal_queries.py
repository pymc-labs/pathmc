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
"""Gate tests for M28: Causal query sugar (ate, cate, prob)."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.idata import posterior as _posterior


def _n_posterior_samples(model) -> int:
    """Number of posterior draws (chains * draws) for a fitted model."""
    ds = _posterior(model._idata)
    return ds.sizes["chain"] * ds.sizes["draw"]


@pytest.fixture(scope="module")
def fork_model(mock_pymc_sample_module):
    """A simple fork model: Z -> X, Z -> Y, X -> Y."""
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.normal(size=n)
    X = 0.5 * Z + rng.normal(scale=0.5, size=n)
    Y = 0.4 * X + 0.6 * Z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
    model.fit(draws=50, tune=50, chains=2, random_seed=42)
    posterior = model._idata.posterior.copy(deep=True)
    posterior["beta_X"].loc[{"X_predictors": "Intercept"}] = 0.0
    posterior["beta_X"].loc[{"X_predictors": "Z"}] = 0.5
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    posterior["beta_Y"].loc[{"Y_predictors": "X"}] = 0.4
    posterior["beta_Y"].loc[{"Y_predictors": "Z"}] = 0.6
    posterior["sigma_X"] = posterior["sigma_X"] * 0 + 0.1
    posterior["sigma_Y"] = posterior["sigma_Y"] * 0 + 0.1
    model._idata.posterior = posterior
    return model


class TestATE:
    def test_ate_returns_do_result(self, fork_model):
        ate = fork_model.ate("Y", "X")
        assert hasattr(ate, "mean")
        assert hasattr(ate, "hdi")

    def test_ate_matches_manual_do(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        r0 = fork_model.do(set={"X": 0.0})
        r1 = fork_model.do(set={"X": 1.0})
        manual = r1 - r0
        assert abs(ate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_ate_positive(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        assert ate.mean("Y") > 0

    def test_ate_custom_values(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(-1.0, 2.0))
        # Should be ~3 * 0.4 = 1.2
        assert ate.mean("Y") > 0.5

    def test_ate_hdi(self, fork_model):
        ate = fork_model.ate("Y", "X")
        hdi = ate.hdi("Y", prob=0.94)
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]


class TestCATE:
    def test_cate_returns_do_result(self, fork_model):
        cate = fork_model.cate("Y", "X", condition={"Z": 1.0})
        assert hasattr(cate, "mean")

    def test_cate_matches_manual_do(self, fork_model):
        cate = fork_model.cate("Y", "X", values=(0.0, 1.0), condition={"Z": 1.0})
        r0 = fork_model.do(set={"X": 0.0, "Z": 1.0})
        r1 = fork_model.do(set={"X": 1.0, "Z": 1.0})
        manual = r1 - r0
        assert abs(cate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_cate_without_condition_equals_ate(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        cate = fork_model.cate("Y", "X", values=(0.0, 1.0))
        assert abs(ate.mean("Y") - cate.mean("Y")) < 1e-10


class TestProb:
    def test_prob_returns_float(self, fork_model):
        p = fork_model.prob("Y > 0", set={"X": 1.0})
        assert isinstance(p, float)

    def test_prob_between_zero_and_one(self, fork_model):
        p = fork_model.prob("Y > 0", set={"X": 1.0})
        assert 0.0 <= p <= 1.0

    def test_prob_higher_x_higher_prob(self, fork_model):
        p_lo = fork_model.prob("Y > 0", set={"X": -2.0})
        p_hi = fork_model.prob("Y > 0", set={"X": 2.0})
        assert p_hi > p_lo

    def test_prob_always_true(self, fork_model):
        p = fork_model.prob("Y > -1000", set={"X": 0.0})
        assert p == 1.0

    def test_prob_always_false(self, fork_model):
        p = fork_model.prob("Y > 1000", set={"X": 0.0})
        assert p == 0.0


class TestDrawCount:
    """Guard the reported draw count against per-unit inflation.

    For ``kind="mean"`` results the draws are a posterior over an expectation:
    there must be exactly ``chains * draws`` of them (one per posterior sample,
    averaged over units), never ``chains * draws * n_units``. This invariant
    must survive the planned migration to an xarray-backed internal store.
    """

    def test_ate_draws_count_equals_posterior_samples(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        ate = fork_model.ate("Y", "X")
        assert ate.draws().shape == (n_samples,)

    def test_ate_draws_independent_of_n_rows(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        n_rows = len(fork_model._data)
        ate = fork_model.ate("Y", "X")
        # The bug returned one value per (draw, unit); guard against that.
        assert ate.draws().shape[0] == n_samples
        assert ate.draws().shape[0] != n_samples * n_rows

    def test_cate_draws_count_equals_posterior_samples(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        cate = fork_model.cate("Y", "X", condition={"Z": 1.0})
        assert cate.draws().shape == (n_samples,)

    def test_do_mean_draws_count_for_every_variable(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        result = fork_model.do(set={"X": 1.0}, kind="mean")
        for var in ("X", "Y", "Z"):
            assert result.draws(var).shape == (n_samples,)

    def test_estimand_repr_is_compact_one_liner(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        ate = fork_model.ate("Y", "X")
        r = repr(ate)
        assert "\n" not in r
        assert "ATE" in r
        assert str(n_samples) not in r  # draws count lives in _repr_html_, not __repr__

    def test_estimand_repr_html_reports_posterior_sample_count(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        ate = fork_model.ate("Y", "X")
        html = ate._repr_html_()
        assert str(n_samples) in html
        assert "Draws" in html

    def test_doresult_repr_reports_posterior_sample_count(self, fork_model):
        n_samples = _n_posterior_samples(fork_model)
        result = fork_model.do(set={"X": 1.0}, kind="mean")
        assert f"{n_samples} draws" in repr(result)
