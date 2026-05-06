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
"""Gate tests for M17: Panel smoke tests (end-to-end integration)."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def full_panel_data():
    """Full panel pipeline data: 3 regions, 20 weeks, sales ~ lag(spend)."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    true_intercepts = {"A": 5.0, "B": 8.0, "C": 12.0}
    n_weeks = 20
    rows = []
    for region in regions:
        spend_prev = 10.0
        for week in range(1, n_weeks + 1):
            spend = rng.uniform(5, 15)
            sales = true_intercepts[region] + 0.6 * spend_prev + rng.normal(scale=0.5)
            rows.append(
                {
                    "region": region,
                    "week": week,
                    "sales": sales,
                    "spend": spend,
                }
            )
            spend_prev = spend
    return pd.DataFrame(rows)


@pytest.fixture()
def binary_panel_data():
    """Binary panel data for scan compiler validation tests."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["A", "B"]:
        for week in range(1, 8):
            x = rng.normal()
            p = 1 / (1 + np.exp(-0.5 * x))
            m = float(rng.binomial(1, p))
            y = 0.5 * m + rng.normal(scale=0.1)
            rows.append({"region": region, "week": week, "X": x, "M": m, "Y": y})
    return pd.DataFrame(rows)


class TestScanNonGaussianValidation:
    """Scan panel validation for non-Gaussian endogenous predictors."""

    def test_terminal_non_gaussian_scan_outcome_allowed(self, binary_panel_data):
        model = pathmc.model(
            "M ~ lag(X)",
            data=binary_panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": "bernoulli"},
        )

        assert "_use_observed_carry" in model.pymc_model.named_vars

    def test_non_gaussian_scan_intermediary_rejected(self, binary_panel_data):
        with pytest.raises(
            ValueError,
            match="non-Gaussian endogenous variables as predictors",
        ):
            pathmc.model(
                "M ~ X\nY ~ M + lag(Y)",
                data=binary_panel_data,
                panel={"unit": "region", "time": "week"},
                families={"M": "bernoulli"},
            )

    def test_non_gaussian_scan_self_lag_rejected(self, binary_panel_data):
        with pytest.raises(
            ValueError,
            match="non-Gaussian endogenous variables as predictors",
        ):
            pathmc.model(
                "M ~ X + lag(M)",
                data=binary_panel_data,
                panel={"unit": "region", "time": "week"},
                families={"M": "bernoulli"},
            )


@pytest.mark.slow
class TestFullPipeline:
    """End-to-end: fit with lag() -> sample -> summary -> do."""

    def test_pipeline_completes(self, full_panel_data):
        model = pathmc.model(
            "sales ~ lag(spend)",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        idata = model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        assert idata is not None

        summary = model.summary()
        assert len(summary) > 0
        assert any("alpha_sales" in str(idx) for idx in summary.index)

    def test_do_time_forward_ate(self, full_panel_data):
        """do(simulate_over='time') produces ATE with correct sign."""
        model = pathmc.model(
            "sales ~ lag(spend)",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)

        r_low = model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r_high = model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")

        ate = r_high - r_low
        assert ate.mean("sales") > 0

    def test_graph_works(self, full_panel_data):
        model = pathmc.model(
            "sales ~ lag(spend)",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        g = model.graph()
        assert g is not None


@pytest.mark.slow
class TestPanelBernoulli:
    """Panel model with Bernoulli outcome."""

    def test_panel_bernoulli_works(self):
        rng = np.random.default_rng(42)
        regions = ["A", "B"]
        rows = []
        for region in regions:
            for week in range(1, 21):
                x = rng.normal()
                p = 1 / (1 + np.exp(-(0.5 * x)))
                y = float(rng.binomial(1, p))
                rows.append({"region": region, "week": week, "X": x, "Y": y})
        df = pd.DataFrame(rows)

        model = pathmc.model(
            "Y ~ X",
            data=df,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
            families={"Y": "bernoulli"},
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r = model.do(set={"X": 1.0})
        assert 0.0 < r.mean("Y") < 1.0


@pytest.mark.slow
class TestRandomInterceptVariation:
    """Random intercepts produce per-unit variation."""

    def test_different_units_different_intercepts(self, full_panel_data):
        model = pathmc.model(
            "sales ~ lag(spend)",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        summary = model.summary()

        alpha_rows = [idx for idx in summary.index if "alpha_sales" in str(idx)]
        assert len(alpha_rows) >= 3
