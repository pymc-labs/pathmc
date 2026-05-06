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
"""Gate tests for M16: Time-forward do(simulate_over='time')."""

import numpy as np
import pandas as pd
import pytest

import pathmc


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
def panel_lag_model(panel_lag_data):
    """Fitted panel model with lag structure."""
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=panel_lag_data,
        panel={"unit": "region", "time": "week"},
        pooling="partial",
    )
    model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


class TestPanelDoAPI:
    """do(simulate_over='time') returns DoResult."""

    @pytest.mark.slow
    def test_returns_do_result(self, panel_lag_model):
        result = panel_lag_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert hasattr(result, "mean")
        assert hasattr(result, "hdi")

    @pytest.mark.slow
    def test_mean_is_finite(self, panel_lag_model):
        result = panel_lag_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert np.isfinite(result.mean("sales"))

    def test_error_without_panel(self, mock_pymc_sample):
        """simulate_over='time' without panel= raises ValueError."""
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n)})
        model = pathmc.model("Y ~ X", data=df)
        model.fit(draws=5, tune=5, chains=1, cores=1, random_seed=42)
        with pytest.raises(ValueError, match="panel"):
            model.do(set={"X": 1.0}, simulate_over="time")


class TestTemporalPropagation:
    """Interventions propagate through time via lag structure."""

    @pytest.mark.slow
    def test_higher_spend_higher_sales(self, panel_lag_model):
        """Higher spend -> higher sales via lag."""
        r_low = panel_lag_model.do(
            set={"spend": 5.0}, simulate_over="time", kind="mean"
        )
        r_high = panel_lag_model.do(
            set={"spend": 15.0}, simulate_over="time", kind="mean"
        )
        assert r_high.mean("sales") > r_low.mean("sales")


class TestContrastArithmetic:
    """Contrast arithmetic works with panel DoResults."""

    @pytest.mark.slow
    def test_contrast_subtraction(self, panel_lag_model):
        r0 = panel_lag_model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r1 = panel_lag_model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")
        contrast = r1 - r0
        assert np.isfinite(contrast.mean("sales"))
        assert contrast.mean("sales") > 0

    @pytest.mark.slow
    def test_hdi_is_valid(self, panel_lag_model):
        r0 = panel_lag_model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r1 = panel_lag_model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")
        contrast = r1 - r0
        hdi = contrast.hdi("sales")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]
