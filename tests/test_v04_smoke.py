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
"""Smoke tests for v0.4: Full pipeline integration tests."""

import numpy as np
import pandas as pd
import pytest

import pathmc

pytestmark = pytest.mark.slow


class TestFullPipeline:
    """fit -> adjustment_sets -> ate -> standardized."""

    @pytest.fixture(scope="class")
    def pipeline_model(self):
        rng = np.random.default_rng(42)
        n = 300
        Z = rng.normal(size=n)
        X = 0.5 * Z + rng.normal(scale=0.5, size=n)
        Y = 0.4 * X + 0.6 * Z + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        model = pathmc.model("X ~ a*Z\nY ~ b*X + c*Z", data=df)
        model.fit(draws=300, tune=300, chains=2, random_seed=42)
        return model

    def test_adjustment_sets(self, pipeline_model):
        sets = pipeline_model.adjustment_sets("X", "Y")
        assert {"Z"} in sets

    def test_is_identifiable(self, pipeline_model):
        assert pipeline_model.is_identifiable("X", "Y")

    def test_ate(self, pipeline_model):
        ate = pipeline_model.ate("Y", "X")
        # True coefficient ~0.4
        assert 0.1 < ate.mean("Y") < 0.8

    def test_cate(self, pipeline_model):
        cate = pipeline_model.cate("Y", "X", condition={"Z": 1.0})
        assert cate.mean("Y") is not None

    def test_prob(self, pipeline_model):
        p = pipeline_model.prob("Y > 0", set={"X": 2.0})
        assert 0.0 <= p <= 1.0

    def test_standardized(self, pipeline_model):
        std_df = pipeline_model.standardized()
        assert "b" in std_df.index
        assert std_df.loc["b", "mean"] > 0

    def test_effects_summary(self, pipeline_model):
        eff = pipeline_model.effects_summary()
        assert "b" in eff.index


class TestPanelRandomSlopesPipeline:
    """Panel model with random slopes -> do() -> verify slopes contribute."""

    @pytest.fixture(scope="class")
    def panel_model(self):
        rng = np.random.default_rng(42)
        regions = ["A", "B", "C"]
        n_weeks = 20
        slopes = {"A": 1.0, "B": 2.5, "C": 4.0}
        rows = []
        for region in regions:
            for week in range(1, n_weeks + 1):
                spend = rng.uniform(5, 25)
                sales = 50 + slopes[region] * spend + rng.normal(scale=1.0)
                rows.append({
                    "region": region,
                    "week": week,
                    "spend": spend,
                    "sales": sales,
                })
        df = pd.DataFrame(rows)

        model = pathmc.model(
            "sales ~ spend",
            data=df,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["spend"]},
        )
        model.fit(draws=200, tune=200, chains=2, random_seed=42)
        return model

    def test_do_works(self, panel_model):
        r = panel_model.do(set={"spend": 10.0})
        assert r.mean("sales") is not None

    def test_ate_positive(self, panel_model):
        ate = panel_model.ate("sales", "spend", values=(5.0, 15.0))
        assert ate.mean("sales") > 5.0

    def test_panel_do_works(self, panel_model):
        r = panel_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert r.mean("sales") is not None

    def test_slopes_contribute(self, panel_model):
        """Verify random slopes are present in the posterior."""
        idata = panel_model._idata
        stacked = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
        assert "slope_sales_spend" in stacked


class TestMediationStandardized:
    """Mediation model: standardized indirect effect."""

    @pytest.fixture(scope="class")
    def med_model(self):
        rng = np.random.default_rng(42)
        n = 400
        X = rng.normal(loc=10, scale=3, size=n)
        M = 0.6 * X + rng.normal(scale=1, size=n)
        Y = 0.7 * M + 0.2 * X + rng.normal(scale=1, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        model = pathmc.model(
            "M ~ a*X\nY ~ b*M + c*X\nindirect := a*b",
            data=df,
        )
        model.fit(draws=300, tune=300, chains=2, random_seed=42)
        return model

    def test_standardized_indirect(self, med_model):
        std_df = med_model.standardized()
        # 'a' and 'b' should be present
        assert "a" in std_df.index
        assert "b" in std_df.index

    def test_effects_summary_has_indirect(self, med_model):
        eff = med_model.effects_summary()
        assert "indirect" in eff.index
        assert eff.loc["indirect", "mean"] > 0.2

    def test_ate_total(self, med_model):
        ate = med_model.ate("Y", "X", values=(8.0, 12.0))
        # Total effect = direct + indirect ≈ 0.2 + 0.6*0.7 = 0.62 per unit
        # For 4-unit change: ~2.5
        assert ate.mean("Y") > 1.0
