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
"""Characterization tests for panel ``by_time()`` / ``time_index`` on
:class:`DoResult` and :class:`EstimandResult`.

These lock in the *current* behavior of the per-time-step accessors, which
were previously untested despite being the distinguishing feature of panel
``do(simulate_over="time")`` results. They must continue to pass after the
planned migration to an xarray-backed internal store (see issue #319).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.idata import posterior as _posterior
from pathmc.simulate import DoResult, EstimandResult


def _n_posterior_samples(model) -> int:
    ds = _posterior(model._idata)
    return ds.sizes["chain"] * ds.sizes["draw"]


@pytest.fixture(scope="module")
def panel_lag_data():
    """Panel with lagged structure: sales ~ lag(spend), 3 regions, 15 weeks."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 15
    rows = []
    for region in regions:
        spend_prev = 0.0
        for week in range(1, n_weeks + 1):
            spend = rng.uniform(5, 15)
            sales = 5.0 + 0.5 * spend_prev + rng.normal(scale=0.5)
            rows.append({
                "region": region,
                "week": week,
                "sales": sales,
                "spend": spend,
            })
            spend_prev = spend
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_lag_model(panel_lag_data, mock_pymc_sample_module):
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=panel_lag_data,
        panel={"unit": "region", "time": "week"},
        pooling="partial",
    )
    model.fit(draws=50, tune=50, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="module")
def panel_do_result(panel_lag_model):
    """A panel do(simulate_over='time') result with per-time data."""
    return panel_lag_model.do(
        set={"spend": 10.0},
        simulate_over="time",
        kind="mean",
    )


@pytest.fixture(scope="module")
def panel_contrast(panel_lag_model):
    """A panel DoResult contrast (high - low spend)."""
    r0 = panel_lag_model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
    r1 = panel_lag_model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")
    return r1 - r0


@pytest.fixture(scope="module")
def panel_estimand(panel_lag_model):
    """A panel ATE returning an EstimandResult with per-time data."""
    return panel_lag_model.ate(
        "sales",
        "spend",
        values=(5.0, 15.0),
        simulate_over="time",
    )


# ---------------------------------------------------------------------------
# DoResult.by_time() / time_index
# ---------------------------------------------------------------------------


class TestDoResultByTime:
    """DoResult.by_time() exposes per-time-step draws."""

    @pytest.mark.slow
    def test_by_time_shape(self, panel_do_result, panel_lag_model):
        by_time = panel_do_result.by_time("sales")
        n_samples = _n_posterior_samples(panel_lag_model)
        assert by_time.shape == (15, n_samples)

    @pytest.mark.slow
    def test_by_time_first_axis_matches_n_times(self, panel_do_result):
        by_time = panel_do_result.by_time("sales")
        assert by_time.shape[0] == 15

    @pytest.mark.slow
    def test_by_time_second_axis_is_posterior_samples(
        self, panel_do_result, panel_lag_model
    ):
        by_time = panel_do_result.by_time("sales")
        n_samples = _n_posterior_samples(panel_lag_model)
        assert by_time.shape[1] == n_samples

    @pytest.mark.slow
    def test_time_index_is_not_none(self, panel_do_result):
        assert panel_do_result.time_index is not None

    @pytest.mark.slow
    def test_time_index_length_matches_n_times(self, panel_do_result):
        assert len(panel_do_result.time_index) == 15

    @pytest.mark.slow
    def test_time_index_values_match_panel(self, panel_do_result, panel_lag_model):
        expected = np.sort(panel_lag_model._data["week"].unique())
        np.testing.assert_array_equal(np.sort(panel_do_result.time_index), expected)

    @pytest.mark.slow
    def test_by_time_finite(self, panel_do_result):
        by_time = panel_do_result.by_time("sales")
        assert np.all(np.isfinite(by_time))

    @pytest.mark.slow
    def test_by_time_mean_matches_overall_mean_axis(self, panel_do_result):
        # mean over (time, samples) of by_time should equal mean of draws()
        by_time = panel_do_result.by_time("sales")
        draws = panel_do_result.draws("sales")
        assert abs(by_time.mean() - draws.mean()) < 1e-9


class TestByTimeRaisesOnCrossSectional:
    """by_time() raises when per-time data is unavailable."""

    def test_do_result_raises(self):
        rng = np.random.default_rng(0)
        result = DoResult(
            values={"Y": rng.normal(size=100)},
            n_chains=1,
            n_draws=100,
        )
        with pytest.raises(ValueError, match="Per-time data not available"):
            result.by_time("Y")

    def test_estimand_result_raises(self):
        rng = np.random.default_rng(0)
        result = EstimandResult(
            values={"Y": rng.normal(size=100)},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=1,
            n_draws=100,
        )
        with pytest.raises(ValueError, match="Per-time data not available"):
            result.by_time()


# ---------------------------------------------------------------------------
# Contrast by_time propagation in __sub__
# ---------------------------------------------------------------------------


class TestContrastByTime:
    """__sub__ propagates per-time data to the contrast DoResult."""

    @pytest.mark.slow
    def test_contrast_has_by_time(self, panel_contrast):
        by_time = panel_contrast.by_time("sales")
        assert by_time.ndim == 2

    @pytest.mark.slow
    def test_contrast_by_time_shape(self, panel_contrast):
        by_time = panel_contrast.by_time("sales")
        assert by_time.shape[0] == 15

    @pytest.mark.slow
    def test_contrast_time_index_preserved(self, panel_contrast):
        assert panel_contrast.time_index is not None
        assert len(panel_contrast.time_index) == 15

    @pytest.mark.slow
    def test_contrast_by_time_finite(self, panel_contrast):
        by_time = panel_contrast.by_time("sales")
        assert np.all(np.isfinite(by_time))


# ---------------------------------------------------------------------------
# EstimandResult.by_time() / time_index (panel estimands)
# ---------------------------------------------------------------------------


class TestEstimandByTime:
    """EstimandResult from a panel estimand exposes per-time contrasts."""

    @pytest.mark.slow
    def test_by_time_shape(self, panel_estimand):
        by_time = panel_estimand.by_time()
        assert by_time.ndim == 2
        assert by_time.shape[0] == 15

    @pytest.mark.slow
    def test_by_time_defaults_to_outcome(self, panel_estimand):
        # by_time() with no arg should equal by_time(outcome)
        default = panel_estimand.by_time()
        explicit = panel_estimand.by_time("sales")
        np.testing.assert_array_equal(default, explicit)

    @pytest.mark.slow
    def test_time_index_not_none(self, panel_estimand):
        assert panel_estimand.time_index is not None
        assert len(panel_estimand.time_index) == 15


# ---------------------------------------------------------------------------
# Per-time draw-count invariant (mirrors TestDrawCount for the time axis)
# ---------------------------------------------------------------------------


class TestTimeDrawCount:
    """Guard per-time draw counts against per-unit inflation.

    For ``kind="mean"`` panel results, each time step must have exactly
    ``chains * draws`` posterior samples (averaged over units), never
    ``chains * draws * n_units``. This invariant must survive the planned
    migration to an xarray-backed internal store.
    """

    @pytest.mark.slow
    def test_do_by_time_samples_equals_posterior(
        self, panel_do_result, panel_lag_model
    ):
        n_samples = _n_posterior_samples(panel_lag_model)
        by_time = panel_do_result.by_time("sales")
        assert by_time.shape[1] == n_samples

    @pytest.mark.slow
    def test_do_by_time_independent_of_n_units(self, panel_do_result, panel_lag_model):
        n_samples = _n_posterior_samples(panel_lag_model)
        n_units = panel_lag_model._data["region"].n_unique()
        by_time = panel_do_result.by_time("sales")
        assert by_time.shape[1] == n_samples
        assert by_time.shape[1] != n_samples * n_units

    @pytest.mark.slow
    def test_contrast_by_time_samples_equals_posterior(self, panel_contrast):
        # contrast doesn't carry _model; infer from shape consistency
        by_time = panel_contrast.by_time("sales")
        draws = panel_contrast.draws("sales")
        assert by_time.shape[1] == draws.shape[0]

    @pytest.mark.slow
    def test_estimand_by_time_samples_equals_draws(self, panel_estimand):
        by_time = panel_estimand.by_time()
        draws = panel_estimand.draws()
        assert by_time.shape[1] == draws.shape[0]
