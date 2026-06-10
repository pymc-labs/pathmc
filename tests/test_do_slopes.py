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
"""Tests for do() propagation of random slopes.

Separated into:
- Mechanical tests (no MCMC): verify do() runs without error on models
  with random slopes, returns expected keys and shapes.
- Recovery tests (slow, MCMC): verify the model recovers correct causal
  effects from a well-identified DGP with enough groups.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def panel_df():
    """Panel data with 3 regions, 20 weeks, and known spend patterns."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 20
    rows = []
    for region in regions:
        for week in range(1, n_weeks + 1):
            spend = rng.uniform(5, 30)
            slopes = {"A": 1.0, "B": 2.0, "C": 3.0}
            sales = 50 + slopes[region] * spend + 0.1 * week + rng.normal(scale=1.0)
            rows.append({
                "region": region,
                "week": week,
                "spend": spend,
                "trend": week,
                "sales": sales,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def slope_model(panel_df):
    """Fit a model with random slopes on spend."""
    model = pathmc.model(
        "sales ~ spend + trend",
        data=panel_df,
        panel={"unit": "region", "time": "week"},
        pooling={"intercept": True, "slopes": ["spend"]},
    )
    model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="class")
def recovery_df():
    """Well-identified DGP: 8 groups, strong signal, for recovery tests."""
    rng = np.random.default_rng(99)
    groups = [f"g{i}" for i in range(8)]
    n_weeks = 30
    group_slopes = {g: 1.0 + 0.5 * i for i, g in enumerate(groups)}
    rows = []
    for group in groups:
        for week in range(1, n_weeks + 1):
            spend = rng.uniform(5, 30)
            sales = 50 + group_slopes[group] * spend + rng.normal(scale=1.0)
            rows.append({"group": group, "week": week, "spend": spend, "sales": sales})
    return pd.DataFrame(rows)


@pytest.fixture(scope="class")
def recovery_model(recovery_df):
    """Fit a random-slopes model on a well-identified DGP."""
    model = pathmc.model(
        "sales ~ spend",
        data=recovery_df,
        panel={"unit": "group", "time": "week"},
        pooling={"intercept": True, "slopes": ["spend"]},
    )
    model.fit(
        draws=500,
        tune=500,
        chains=2,
        cores=1,
        random_seed=99,
        target_accept=0.95,
    )
    return model


# ---------------------------------------------------------------------------
# Mechanical tests — do() works with random slopes (no magnitude checks)
# ---------------------------------------------------------------------------


class TestCrossSectionalDoSlopes:
    """Cross-sectional do() runs correctly on models with random slopes."""

    def test_do_returns_result(self, slope_model):
        r = slope_model.do(set={"spend": 10.0})
        assert r.mean("sales") is not None

    def test_do_returns_finite(self, slope_model):
        r = slope_model.do(set={"spend": 10.0})
        assert np.isfinite(r.mean("sales"))

    def test_contrast_returns_result(self, slope_model):
        r_lo = slope_model.do(set={"spend": 5.0})
        r_hi = slope_model.do(set={"spend": 15.0})
        ate = r_hi - r_lo
        assert np.isfinite(ate.mean("sales"))

    def test_slopes_in_posterior(self, slope_model):
        """Random slope parameters should exist in the posterior."""
        idata = slope_model._idata
        stacked = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
        assert "slope_sales_spend" in stacked


class TestPanelDoSlopes:
    """Panel do(simulate_over='time') runs correctly with random slopes."""

    def test_panel_do_returns_result(self, slope_model):
        r = slope_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert r.mean("sales") is not None

    def test_panel_do_returns_finite(self, slope_model):
        r = slope_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert np.isfinite(r.mean("sales"))

    def test_panel_contrast_returns_result(self, slope_model):
        r_lo = slope_model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r_hi = slope_model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")
        ate = r_hi - r_lo
        assert np.isfinite(ate.mean("sales"))


# ---------------------------------------------------------------------------
# Recovery tests — correct magnitude from well-identified DGP
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSlopeRecovery:
    """With enough groups and draws, the model recovers the correct ATE.

    DGP: 8 groups with slopes [1.0, 1.5, 2.0, ..., 4.5], mean slope = 2.75.
    ATE for spend 5→15 (10-unit change) should be ~27.5.
    """

    def test_cross_sectional_ate_positive(self, recovery_model):
        r_lo = recovery_model.do(set={"spend": 5.0})
        r_hi = recovery_model.do(set={"spend": 15.0})
        ate = r_hi - r_lo
        assert ate.mean("sales") > 5.0

    def test_panel_ate_positive(self, recovery_model):
        r_lo = recovery_model.do(
            set={"spend": 5.0},
            simulate_over="time",
            kind="mean",
        )
        r_hi = recovery_model.do(
            set={"spend": 15.0},
            simulate_over="time",
            kind="mean",
        )
        ate = r_hi - r_lo
        assert ate.mean("sales") > 5.0

    def test_ate_in_ballpark(self, recovery_model):
        """ATE should be in the right ballpark (~27.5 ± wide tolerance)."""
        r_lo = recovery_model.do(set={"spend": 5.0})
        r_hi = recovery_model.do(set={"spend": 15.0})
        ate = r_hi - r_lo
        estimated = ate.mean("sales")
        assert estimated == pytest.approx(27.5, abs=15.0), (
            f"ATE={estimated:.2f}, expected ~27.5"
        )
