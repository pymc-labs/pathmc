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
"""Gate tests for the interpret query layer (issue #358)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc import InterpretResult, datagrid
from pathmc.simulate import EstimandResult, run_do_pymc


@pytest.fixture(scope="module")
def fork_model(mock_pymc_sample_module):
    """Fork DAG: Z -> X, Z -> Y, X -> Y."""
    rng = np.random.default_rng(42)
    n = 300
    z = rng.normal(size=n)
    x = 0.5 * z + rng.normal(scale=0.5, size=n)
    y = 0.4 * x + 0.6 * z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    m = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
    m.fit(draws=50, tune=50, chains=2, random_seed=42)
    posterior = m._idata.posterior.copy(deep=True)
    posterior["beta_X"].loc[{"X_predictors": "Intercept"}] = 0.0
    posterior["beta_X"].loc[{"X_predictors": "Z"}] = 0.5
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    posterior["beta_Y"].loc[{"Y_predictors": "X"}] = 0.4
    posterior["beta_Y"].loc[{"Y_predictors": "Z"}] = 0.6
    posterior["sigma_X"] = posterior["sigma_X"] * 0 + 0.1
    posterior["sigma_Y"] = posterior["sigma_Y"] * 0 + 0.1
    m._idata.posterior = posterior
    return m


@pytest.fixture(scope="module")
def bernoulli_model(mock_pymc_sample_module):
    """Bernoulli outcome: logit(Y) ~ X."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    logits = -0.5 + 0.8 * x
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob).astype(float)
    df = pd.DataFrame({"X": x, "Y": y})

    m = pathmc.model("Y ~ X", data=df, families={"Y": "bernoulli"})
    m.fit(draws=50, tune=50, chains=2, random_seed=7)
    posterior = m._idata.posterior.copy(deep=True)
    posterior["beta_Y"].loc[{"Y_predictors": "Intercept"}] = -0.5
    posterior["beta_Y"].loc[{"Y_predictors": "X"}] = 0.8
    m._idata.posterior = posterior
    return m


class TestUnitLevelDraws:
    def test_prediction_dims_match_data_length(self, fork_model):
        pred = fork_model.predictions("Y", set={"X": 1.0})
        da = pred.dataset["Y"]
        assert da.dims == ("chain", "draw", "unit")
        assert da.sizes["unit"] == len(fork_model._data)

    def test_run_do_pymc_array_intervention_per_row(self, fork_model):
        assert fork_model._data is not None
        assert fork_model._gen_model is not None
        assert fork_model._idata is not None
        n = len(fork_model._data)
        x_vals = np.asarray(fork_model._data["X"].to_numpy(), dtype=float)
        result = run_do_pymc(
            gen_model=fork_model._gen_model,
            graph_info=fork_model._graph_info,
            idata=fork_model._idata,
            data=fork_model._data,
            set={"X": x_vals},
            kind="mean",
            families=fork_model._families,
            average_units=False,
        )
        x_draws = result.dataset["X"]
        assert "unit" in x_draws.dims
        unit_means = x_draws.mean(dim=("chain", "draw")).values
        np.testing.assert_allclose(unit_means, x_vals, rtol=1e-10)


class TestComparisonsATE:
    def test_average_by_all_returns_estimand_result(self, fork_model):
        comp = fork_model.comparisons("Y", "X", contrast=(0.0, 1.0))
        assert isinstance(comp, EstimandResult)

    def test_comparisons_diff_agrees_with_ate(self, fork_model):
        comp = fork_model.comparisons(
            "Y", "X", contrast=(0.0, 1.0), comparison="diff", average_by="all"
        )
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        xr.testing.assert_allclose(comp.dataset["Y"], ate.dataset["Y"])

    def test_average_by_none_returns_interpret_result_with_unit(self, fork_model):
        comp = fork_model.comparisons("Y", "X", contrast=(0.0, 1.0), average_by=None)
        assert isinstance(comp, InterpretResult)
        assert "unit" in comp.dataset["Y"].dims


class TestPredictions:
    def test_returns_interpret_result(self, fork_model):
        pred = fork_model.predictions("Y")
        assert isinstance(pred, InterpretResult)

    def test_interventional_metadata_with_set(self, fork_model):
        pred = fork_model.predictions("Y", set={"X": 1.0})
        assert pred.interventional is True
        assert pred.causal is True

    def test_associational_metadata_without_set(self, fork_model):
        pred = fork_model.predictions("Y")
        assert pred.interventional is False
        assert pred.causal is False

    def test_newdata_datagrid(self, fork_model):
        assert fork_model._data is not None
        grid = datagrid(fork_model._data.to_pandas(), X=[0, 1], Z=[0, 1, 2])
        pred = fork_model.predictions("Y", newdata=grid)
        assert isinstance(pred, InterpretResult)
        assert pred.dataset["Y"].sizes["unit"] == len(grid)


class TestRatioLift:
    def test_ratio_and_lift_finite(self, fork_model):
        ratio = fork_model.comparisons("Y", "X", comparison="ratio", average_by="all")
        lift = fork_model.comparisons("Y", "X", comparison="lift", average_by="all")
        assert isinstance(ratio, EstimandResult)
        assert isinstance(lift, EstimandResult)
        assert np.all(np.isfinite(ratio.draws()))
        assert np.all(np.isfinite(lift.draws()))

    def test_ratio_and_lift_contrast_before_average(self, fork_model):
        from pathmc.interpret import _apply_comparison

        lo = fork_model.predictions("Y", set={"X": 0.0}).dataset["Y"]
        hi = fork_model.predictions("Y", set={"X": 1.0}).dataset["Y"]
        ratio_manual = _apply_comparison(hi, lo, "ratio").mean("unit")
        lift_manual = _apply_comparison(hi, lo, "lift").mean("unit")
        ratio = fork_model.comparisons("Y", "X", comparison="ratio", average_by="all")
        lift = fork_model.comparisons("Y", "X", comparison="lift", average_by="all")
        xr.testing.assert_allclose(ratio.dataset["Y"], ratio_manual)
        xr.testing.assert_allclose(lift.dataset["Y"], lift_manual)


class TestBernoulliContrast:
    def test_response_scale_diff_not_coefficient(self, bernoulli_model):
        comp = bernoulli_model.comparisons(
            "Y", "X", contrast=(0.0, 1.0), average_by="all"
        )
        ate = bernoulli_model.ate("Y", "X", values=(0.0, 1.0))
        assert np.isclose(comp.mean(), ate.mean())
        assert not np.isclose(comp.mean(), 0.8, atol=0.05)
        assert -1.0 <= comp.mean() <= 1.0


class TestSlopes:
    def test_dydx_smoke(self, fork_model):
        result = fork_model.slopes("Y", "X", slope="dydx", average_by="all")
        assert isinstance(result, EstimandResult)
        assert np.all(np.isfinite(result.draws()))

    def test_dydx_unit_level(self, fork_model):
        result = fork_model.slopes("Y", "X", slope="dydx", average_by=None)
        assert isinstance(result, InterpretResult)
        assert "unit" in result.dataset["Y"].dims


class TestIdentifiable:
    def test_records_identifiable_without_raising(self, fork_model):
        comp = fork_model.comparisons("Y", "X", average_by="all")
        assert comp.identifiable is True

    def test_non_identifiable_pair(self, mock_pymc_sample):
        rng = np.random.default_rng(0)
        n = 80
        z = rng.normal(size=n)
        t = 0.5 * z + rng.normal(scale=0.1, size=n)
        y = 0.4 * t + rng.normal(scale=0.1, size=n)
        df = pd.DataFrame({"Z": z, "T": t, "Y": y})
        m = pathmc.model("T ~ Z\nY ~ T\nT ~~ Y", data=df)
        m.fit(draws=20, tune=20, chains=1, random_seed=0)
        comp = m.comparisons("Y", "T", average_by=None)
        assert comp.identifiable is False


class TestDatagrid:
    def test_crossed_shape(self):
        df = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Z": [1.0, 2.0, 3.0]})
        grid = datagrid(df, X=[0, 1], Z=[0, 1, 2])
        assert len(grid) == 6
        assert set(grid["X"]) == {0, 1}
        assert set(grid["Z"]) == {0, 1, 2}

    def test_unspecified_numeric_is_mean(self):
        df = pd.DataFrame({"X": [0.0, 2.0], "Z": [10.0, 20.0]})
        grid = datagrid(df, X=[0, 1])
        assert np.allclose(grid["Z"], 15.0)

    def test_pathmodel_datagrid(self, fork_model):
        grid = fork_model.datagrid(X=[0, 1], Z=[0, 1])
        assert len(grid) == 4


class TestConditionalValidation:
    def test_list_conditional_raises_type_error(self, fork_model):
        with pytest.raises(TypeError, match="datagrid"):
            fork_model.comparisons("Y", "X", conditional={"Z": [0, 1]})


class TestPanelNotImplemented:
    def test_predictions_raises(self, mock_pymc_sample):
        m = _fit_panel_model()
        with pytest.raises(NotImplementedError, match="cross-sectional"):
            m.predictions("sales")

    def test_comparisons_raises(self, mock_pymc_sample):
        m = _fit_panel_model()
        with pytest.raises(NotImplementedError, match="cross-sectional"):
            m.comparisons("sales", "spend")

    def test_slopes_raises(self, mock_pymc_sample):
        m = _fit_panel_model()
        with pytest.raises(NotImplementedError, match="cross-sectional"):
            m.slopes("sales", "spend")


def _fit_panel_model():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B"],
        "week": [1, 2, 1, 2],
        "spend": [1.0, 2.0, 3.0, 4.0],
        "sales": [10.0, 11.0, 12.0, 13.0],
    })
    m = pathmc.model(
        "sales ~ spend",
        data=df,
        panel={"unit": "region", "time": "week"},
    )
    m.fit(draws=20, tune=20, chains=1, random_seed=0)
    return m


class TestPlot:
    def test_plot_returns_figure(self, fork_model):
        import matplotlib

        matplotlib.use("Agg")
        pred = fork_model.predictions("Y", set={"X": 1.0})
        fig = pred.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_on_supplied_axis(self, fork_model):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pred = fork_model.predictions("Y", set={"X": 1.0})
        fig, ax = plt.subplots()
        out = pred.plot(ax=ax)
        assert out is fig


class TestPublicAPI:
    def test_no_polars_return_type(self, fork_model):
        pred = fork_model.predictions("Y", set={"X": 1.0})
        assert isinstance(pred, InterpretResult)
        assert isinstance(pred.dataset, xr.Dataset)

    def test_no_extra_result_classes(self):
        import inspect

        import pathmc.interpret as interpret_mod

        result_classes = [
            name
            for name, obj in inspect.getmembers(interpret_mod, inspect.isclass)
            if name.endswith("Result") and obj.__module__ == interpret_mod.__name__
        ]
        assert result_classes == ["InterpretResult"]
