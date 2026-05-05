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
"""M21 gate tests: Transforms under do() — cross-sectional and panel.

Tests verify that transforms are recomputed during interventional simulation
and that adstock accumulates correctly in time-forward panel do().
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def adstock_data():
    """Time series with adstocked effect: true decay=0.7, coefficient=0.5."""
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 10, size=n)
    adstocked = np.zeros(n)
    for t in range(n):
        adstocked[t] = x[t] + (0.7 * adstocked[t - 1] if t > 0 else 0)
    y = 2.0 + 0.5 * adstocked + rng.normal(scale=1, size=n)
    return pd.DataFrame({"X": x, "Y": y})


@pytest.fixture()
def saturation_data():
    """Data with saturated effect: true lam=0.3, coefficient=3.0."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(0, 10, size=n)
    saturated = 1 - np.exp(-0.3 * x)
    y = 1.0 + 3.0 * saturated + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Y": y})


@pytest.fixture()
def panel_data_for_do():
    """Panel data for adstock do(): 3 regions, 15 weeks."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 15
    rows = []
    for region in regions:
        adstocked = 0.0
        for week in range(1, n_weeks + 1):
            x = rng.uniform(5, 15)
            adstocked = x + 0.6 * adstocked
            y = 5.0 + 0.4 * adstocked + rng.normal(scale=0.5)
            rows.append({"region": region, "week": week, "X": x, "Y": y})
    return pd.DataFrame(rows)


@pytest.mark.slow
class TestCrossSectionalTransformDo:
    """do() recomputes transforms in cross-sectional mode."""

    def test_adstock_do_returns_result(self, adstock_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"X": 5.0})
        assert np.isfinite(result.mean("Y"))

    def test_different_interventions_different_means(self, adstock_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r_low = model.do(set={"X": 1.0})
        r_high = model.do(set={"X": 10.0})
        assert r_high.mean("Y") > r_low.mean("Y")

    def test_saturation_do_returns_result(self, saturation_data):
        model = pathmc.model(
            "Y ~ logistic_saturation(X, lam=lam_x)", data=saturation_data
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"X": 5.0})
        assert np.isfinite(result.mean("Y"))

    def test_contrast_arithmetic_with_transforms(self, adstock_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r0 = model.do(set={"X": 0.0})
        r1 = model.do(set={"X": 5.0})
        contrast = r1 - r0
        assert np.isfinite(contrast.mean("Y"))
        hdi = contrast.hdi("Y")
        assert len(hdi) == 2


@pytest.mark.slow
class TestComposedTransformDo:
    """do() with nested/composed transforms."""

    def test_nested_do_returns_result(self, adstock_data):
        spec = "Y ~ logistic_saturation(adstock(X, decay=theta), lam=lam_x)"
        model = pathmc.model(spec, data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"X": 5.0})
        assert np.isfinite(result.mean("Y"))

    def test_nested_higher_x_different_mean(self, adstock_data):
        spec = "Y ~ logistic_saturation(adstock(X, decay=theta), lam=lam_x)"
        model = pathmc.model(spec, data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r_low = model.do(set={"X": 1.0})
        r_high = model.do(set={"X": 10.0})
        assert r_high.mean("Y") != r_low.mean("Y")


@pytest.mark.slow
class TestPanelTransformDo:
    """Panel do() with adstock: time-forward accumulation."""

    def test_panel_adstock_do_returns(self, panel_data_for_do):
        model = pathmc.model(
            "Y ~ adstock(X, decay=theta)",
            data=panel_data_for_do,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"X": 10.0}, simulate_over="time", kind="mean")
        assert np.isfinite(result.mean("Y"))

    def test_panel_higher_spend_higher_outcome(self, panel_data_for_do):
        model = pathmc.model(
            "Y ~ adstock(X, decay=theta)",
            data=panel_data_for_do,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r_low = model.do(set={"X": 5.0}, simulate_over="time", kind="mean")
        r_high = model.do(set={"X": 15.0}, simulate_over="time", kind="mean")
        assert r_high.mean("Y") > r_low.mean("Y")


@pytest.mark.slow
class TestPredictiveWithTransforms:
    """kind='predictive' works with transforms."""

    def test_predictive_do_with_adstock(self, adstock_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=adstock_data)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"X": 5.0}, kind="predictive")
        assert np.isfinite(result.mean("Y"))
        hdi = result.hdi("Y")
        assert hdi[0] < hdi[1]
