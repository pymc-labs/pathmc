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
"""Gate tests for AdjustmentModel interpret API (#438)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc import InterpretResult
from pathmc.simulate import EstimandResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _fork_df(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(scale=0.5, size=n)
    y = 0.5 * x + 0.3 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Z": z, "Y": y})


@pytest.fixture
def fitted_adjusted(mock_pymc_sample, rng):
    df = _fork_df(rng)
    model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
    adjusted = model.adjustment_model("X -> Y")
    adjusted.fit(draws=50, chains=2, random_seed=42)
    return adjusted


class TestDesignatedTreatmentComparisons:
    def test_comparisons_agrees_with_ate(self, fitted_adjusted):
        comp = fitted_adjusted.comparisons(
            variable="X", contrast=(0.0, 1.0), average_by="all"
        )
        ate = fitted_adjusted.ate(values=(0.0, 1.0))
        xr.testing.assert_allclose(comp.dataset["Y"], ate.dataset["Y"])

    def test_defaults_match_explicit_query(self, fitted_adjusted):
        comp_default = fitted_adjusted.comparisons(average_by="all")
        comp_explicit = fitted_adjusted.comparisons(
            outcome="Y", variable="X", average_by="all"
        )
        xr.testing.assert_allclose(
            comp_default.dataset["Y"], comp_explicit.dataset["Y"]
        )

    def test_designated_treatment_causal_metadata(self, fitted_adjusted):
        comp = fitted_adjusted.comparisons(variable="X", average_by="all")
        assert isinstance(comp, EstimandResult)
        assert comp.estimator == "regression_adjustment"
        assert comp.causal is True
        assert comp.interventional is True
        assert comp.identifiable is True
        assert comp.dataset.attrs["estimator"] == "regression_adjustment"
        assert comp.dataset.attrs["adjustment_set"] == ("Z",)


class TestConfounderQueries:
    def test_confounder_slope_not_causal(self, fitted_adjusted):
        slope = fitted_adjusted.slopes(wrt="Z", average_by="all")
        assert isinstance(slope, EstimandResult)
        assert slope.estimator == "regression_adjustment"
        assert slope.causal is False
        assert slope.interventional is True

    def test_confounder_comparison_not_causal(self, fitted_adjusted):
        comp = fitted_adjusted.comparisons(
            variable="Z", contrast=(0.0, 1.0), average_by="all"
        )
        assert comp.causal is False
        assert comp.estimator == "regression_adjustment"
        assert comp.interventional is True


class TestPredictionsMetadata:
    def test_associational_without_set(self, fitted_adjusted):
        pred = fitted_adjusted.predictions()
        assert isinstance(pred, InterpretResult)
        assert pred.interventional is False
        assert pred.causal is False
        assert pred.estimator == "regression_adjustment"

    def test_interventional_on_treatment_is_causal(self, fitted_adjusted):
        pred = fitted_adjusted.predictions(set={"X": 1.0})
        assert pred.interventional is True
        assert pred.causal is True
        assert pred.estimator == "regression_adjustment"

    def test_interventional_on_confounder_not_causal(self, fitted_adjusted):
        pred = fitted_adjusted.predictions(set={"Z": 1.0})
        assert pred.interventional is True
        assert pred.causal is False


class TestAteStillCausal:
    def test_ate_causal_after_stamp_refactor(self, fitted_adjusted):
        ate = fitted_adjusted.ate()
        assert ate.causal is True
        assert ate.estimator == "regression_adjustment"
        assert ate.interventional is True
        assert ate.identifiable is True


class TestFormulaValidation:
    def test_variable_not_in_formula_raises(self, fitted_adjusted):
        with pytest.raises(ValueError, match="not in the reduced outcome formula"):
            fitted_adjusted.comparisons(variable="W")

    def test_wrt_not_in_formula_raises(self, fitted_adjusted):
        with pytest.raises(ValueError, match="not in the reduced outcome formula"):
            fitted_adjusted.slopes(wrt="W")

    def test_wrong_outcome_raises(self, fitted_adjusted):
        with pytest.raises(ValueError, match="This AdjustmentModel"):
            fitted_adjusted.comparisons(outcome="Z", variable="X")


class TestSlopesDefaults:
    def test_slopes_default_wrt_is_treatment(self, fitted_adjusted):
        slope_default = fitted_adjusted.slopes(average_by="all")
        slope_explicit = fitted_adjusted.slopes(wrt="X", average_by="all")
        xr.testing.assert_allclose(
            slope_default.dataset["Y"], slope_explicit.dataset["Y"]
        )
        assert slope_default.causal is True


class TestDatagrid:
    def test_datagrid_delegates_to_inner(self, fitted_adjusted):
        grid = fitted_adjusted.datagrid(X=[0.0, 1.0], Z=[0.0])
        assert len(grid) == 2
        assert list(grid.columns) == ["X", "Z", "Y"]
