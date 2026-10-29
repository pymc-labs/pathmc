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

import arviz as az
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

    @pytest.mark.parametrize(
        ("m_obs", "beta_m", "beta_y"),
        [
            ([0.0, 1.0, 100.0, 3.0], [0.0, 5.0], [0.0, 2.0, 3.0]),
            ([4.0, -2.0, 0.5, 8.0], [1.0, -3.0], [0.5, 4.0, -1.0]),
        ],
    )
    def test_omitted_set_uses_observed_mediator(self, m_obs, beta_m, beta_y):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        m_obs = np.asarray(m_obs, dtype=float)
        beta_m = np.asarray(beta_m, dtype=float)
        beta_y = np.asarray(beta_y, dtype=float)
        df = pd.DataFrame({"X": x, "M": m_obs, "Y": np.zeros(4)})
        model = pathmc.model("M ~ X\nY ~ M + X", data=df)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": beta_m.reshape(1, 1, -1),
                    "sigma_M": np.array([[1.0]]),
                    "beta_Y": beta_y.reshape(1, 1, -1),
                    "sigma_Y": np.array([[1.0]]),
                }
            },
            coords={
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M", "X"],
            },
            dims={"beta_M": ["M_predictors"], "beta_Y": ["Y_predictors"]},
        )
        pred = model.predictions("Y")
        expected = beta_y[0] + beta_y[1] * m_obs + beta_y[2] * x
        np.testing.assert_allclose(np.asarray(pred.dataset["Y"]).reshape(-1), expected)
        assert pred.interventional is False
        assert pred.causal is False

        mediator = model.predictions("M")
        np.testing.assert_allclose(
            np.asarray(mediator.dataset["M"]).reshape(-1),
            beta_m[0] + beta_m[1] * x,
        )

        reduced = beta_y[0] + beta_y[1] * (beta_m[0] + beta_m[1] * x) + beta_y[2] * x
        intervened = model.predictions("Y", set={"X": x})
        np.testing.assert_allclose(
            np.asarray(intervened.dataset["Y"]).reshape(-1), reduced
        )
        assert intervened.interventional is True

    def test_omitted_set_uses_observed_mediator_through_a_chain(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        m_obs = np.array([0.0, 1.0, 100.0, 3.0])
        y_obs = np.array([1.0, 0.0, 7.0, 2.0])
        df = pd.DataFrame({"X": x, "M": m_obs, "Y": y_obs, "Z": np.zeros(4)})
        model = pathmc.model("M ~ X\nY ~ M\nZ ~ Y", data=df)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": np.array([[[0.0, 5.0]]]),
                    "sigma_M": np.array([[1.0]]),
                    "beta_Y": np.array([[[0.0, 2.0]]]),
                    "sigma_Y": np.array([[1.0]]),
                    "beta_Z": np.array([[[0.0, 4.0]]]),
                    "sigma_Z": np.array([[1.0]]),
                }
            },
            coords={
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M"],
                "Z_predictors": ["Intercept", "Y"],
            },
            dims={
                "beta_M": ["M_predictors"],
                "beta_Y": ["Y_predictors"],
                "beta_Z": ["Z_predictors"],
            },
        )
        pred = model.predictions("Z")
        np.testing.assert_allclose(
            np.asarray(pred.dataset["Z"]).reshape(-1), 4.0 * y_obs
        )
        assert pred.interventional is False
        assert pred.causal is False
        mid = model.predictions("Y")
        np.testing.assert_allclose(
            np.asarray(mid.dataset["Y"]).reshape(-1), 2.0 * m_obs
        )

    def test_omitted_set_rejects_query_frame_missing_mediator(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        df = pd.DataFrame({"X": x, "M": [0.0, 1.0, 100.0, 3.0], "Y": np.zeros(4)})
        model = pathmc.model("M ~ X\nY ~ M + X", data=df)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": np.array([[[0.0, 5.0]]]),
                    "sigma_M": np.array([[1.0]]),
                    "beta_Y": np.array([[[0.0, 2.0, 3.0]]]),
                    "sigma_Y": np.array([[1.0]]),
                }
            },
            coords={
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M", "X"],
            },
            dims={"beta_M": ["M_predictors"], "beta_Y": ["Y_predictors"]},
        )
        with pytest.raises(ValueError, match="column 'M'"):
            model.predictions(
                "Y", newdata=pd.DataFrame({"X": [1.0, 2.0], "Y": [0.0, 0.0]})
            )
        chain = pathmc.model(
            "M ~ X\nY ~ M\nZ ~ Y",
            data=pd.DataFrame({
                "X": x,
                "M": [0.0, 1.0, 100.0, 3.0],
                "Y": [1.0, 0.0, 7.0, 2.0],
                "Z": np.zeros(4),
            }),
        )
        chain._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": np.array([[[0.0, 5.0]]]),
                    "sigma_M": np.array([[1.0]]),
                    "beta_Y": np.array([[[0.0, 2.0]]]),
                    "sigma_Y": np.array([[1.0]]),
                    "beta_Z": np.array([[[0.0, 4.0]]]),
                    "sigma_Z": np.array([[1.0]]),
                }
            },
            coords={
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M"],
                "Z_predictors": ["Intercept", "Y"],
            },
            dims={
                "beta_M": ["M_predictors"],
                "beta_Y": ["Y_predictors"],
                "beta_Z": ["Z_predictors"],
            },
        )
        with pytest.raises(ValueError, match="column 'Y'"):
            chain.predictions("Z", newdata=pd.DataFrame({"X": x, "M": x, "Z": x}))

    def test_omitted_set_rejects_partially_observed_mediator(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        m_obs = np.array([0.0, np.nan, 100.0, np.nan])
        df = pd.DataFrame({"X": x, "M": m_obs, "Y": np.zeros(4)})
        model = pathmc.model("M ~ X\nY ~ M + X", data=df)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": np.array([[[0.0, 5.0]]]),
                    "sigma_M": np.array([[1.0]]),
                    "beta_Y": np.array([[[0.0, 2.0, 3.0]]]),
                    "sigma_Y": np.array([[1.0]]),
                }
            },
            coords={
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M", "X"],
            },
            dims={"beta_M": ["M_predictors"], "beta_Y": ["Y_predictors"]},
        )
        with pytest.raises(ValueError, match="column 'M'"):
            model.predictions("Y")
        complete = pathmc.model(
            "M ~ X\nY ~ M + X",
            data=pd.DataFrame({"X": x, "M": [0.0, 1.0, 100.0, 3.0], "Y": np.zeros(4)}),
        )
        complete._idata = model._idata
        with pytest.raises(ValueError, match="column 'M'"):
            complete.predictions(
                "Y",
                newdata=pd.DataFrame({
                    "X": x,
                    "M": [0.0, 1.0, np.nan, 3.0],
                    "Y": np.zeros(4),
                }),
            )


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

    def test_pathmodel_datagrid_unscaled_matches_standalone(self, fork_model):
        assert fork_model._data is not None
        got = fork_model.datagrid(X=[0, 1])
        expected = datagrid(fork_model._data.to_pandas(), X=[0, 1])
        pd.testing.assert_frame_equal(got, expected)

    def test_pathmodel_datagrid_scaled_returns_business_units(self, mock_pymc_sample):
        df = pd.DataFrame({
            "tv": [100.0, 200.0, 300.0, 400.0],
            "radio": [20.0, 40.0, 60.0, 80.0],
            "sales": [31.0, 62.0, 93.0, 124.0],
        })
        model = pathmc.model(
            "sales ~ tv + radio",
            data=df,
            scaling=pathmc.Scaling(target={"method": "max"}, channel={"method": "max"}),
        )
        model.fit()
        assert model._idata is not None
        posterior = model._idata.posterior.copy(deep=True)
        posterior["beta_sales"].loc[{"sales_predictors": "Intercept"}] = 0.0
        posterior["beta_sales"].loc[{"sales_predictors": "tv"}] = 1.0
        posterior["beta_sales"].loc[{"sales_predictors": "radio"}] = 1.0
        model._idata.posterior = posterior

        raw_tv = 250.0
        grid = model.datagrid(tv=[raw_tv])

        assert grid["tv"].tolist() == [raw_tv]
        assert grid["radio"].tolist() == [df["radio"].mean()]
        assert grid["sales"].tolist() == [df["sales"].mean()]

        predicted = model.predictions("sales", newdata=grid)
        intervened = model.do(set={"tv": raw_tv}, kind="mean")
        assert float(predicted.dataset["sales"].mean()) == pytest.approx(
            intervened.mean("sales")
        )

    def test_pathmodel_datagrid_unscales_per_unit_held_column(self):
        df = pd.DataFrame({
            "geo": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "tv": [10.0, 30.0, 100.0, 300.0],
            "radio": [20.0, 40.0, 200.0, 400.0],
            "sales": [1.0, 2.0, 3.0, 4.0],
        })
        model = pathmc.model(
            "sales ~ tv + radio",
            data=df,
            panel={"unit": "geo", "time": "time"},
            scaling=pathmc.Scaling(channel={"method": "max", "dims": ("geo",)}),
        )

        grid = model.datagrid(tv=[50.0])

        assert grid["geo"].tolist() == [1.5]
        assert grid["tv"].tolist() == [50.0]
        assert grid["radio"].tolist() == [df["radio"].mean()]


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

    @pytest.fixture
    def pred(self, fork_model):
        import matplotlib

        matplotlib.use("Agg")
        return fork_model.predictions("Y", set={"X": 1.0})

    @pytest.mark.parametrize("bins", [-1, 0])
    def test_plot_non_positive_bin_count_rejected(self, pred, bins):
        with pytest.raises(
            ValueError, match=rf"^bins must be a positive integer, got {bins}\.$"
        ):
            pred.plot(bins=bins)

    @pytest.mark.parametrize("bins", [2.5, True, {1, 2}])
    def test_plot_invalid_bins_type_rejected(self, pred, bins):
        # bool subclasses int, so it needs an explicit reject.
        with pytest.raises(
            ValueError,
            match=r"^bins must be a positive integer, a binning strategy name, "
            r"a sequence of bin edges, or None, got ",
        ):
            pred.plot(bins=bins)

    @pytest.mark.parametrize("bins", [None, 10, "auto"])
    def test_plot_valid_bins_accepted(self, pred, bins):
        assert pred.plot(bins=bins) is not None


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


def _inject_categorical_posterior(
    model, outcome: str, predictor: str, intercept: float, contrast: float
):
    import arviz as az

    model._idata = az.from_dict(
        {
            "posterior": {
                f"beta_{outcome}": np.array([[[intercept]]]),
                f"beta_{outcome}_{predictor}": np.array([[[contrast]]]),
                f"sigma_{outcome}": np.array([[0.1]]),
            }
        },
        coords={
            f"{outcome}_predictors": ["Intercept"],
            f"{outcome}_{predictor}_levels": ["contrast"],
        },
        dims={
            f"beta_{outcome}": [f"{outcome}_predictors"],
            f"beta_{outcome}_{predictor}": [f"{outcome}_{predictor}_levels"],
        },
    )


class TestStringCategoricalPredictions:
    """Unit-level means for a string categorical predictor (issue #583)."""

    def test_predictions_return_unit_means_not_label_cast(self):
        region = np.array(["north", "south", "north", "south"])
        df = pd.DataFrame({
            "region": region,
            "Y": np.where(region == "north", 1.0, 4.0),
        })
        model = pathmc.model("Y ~ region", data=df)
        _inject_categorical_posterior(model, "Y", "region", intercept=1.0, contrast=3.0)

        pred = model.predictions("Y")
        means = pred.dataset["Y"].mean(("chain", "draw")).values
        np.testing.assert_allclose(means, [1.0, 4.0, 1.0, 4.0])

        south = model.predictions("Y", set={"region": "south"})
        south_means = south.dataset["Y"].mean(("chain", "draw")).values
        np.testing.assert_allclose(south_means, [4.0, 4.0, 4.0, 4.0])

    def test_invalid_level_names_column_and_level(self):
        market = np.array(["east", "west", "east"])
        df = pd.DataFrame({"market": market, "sales": [2.0, 7.0, 2.0]})
        model = pathmc.model("sales ~ market", data=df)
        _inject_categorical_posterior(
            model, "sales", "market", intercept=2.0, contrast=5.0
        )

        means = (
            model.predictions("sales").dataset["sales"].mean(("chain", "draw")).values
        )
        np.testing.assert_allclose(means, [2.0, 7.0, 2.0])

        with pytest.raises(ValueError, match="market") as exc_info:
            model.predictions("sales", set={"market": "central"})
        message = str(exc_info.value)
        assert "central" in message
        assert "market" in message

    def test_conditional_label_reaches_encoder(self):
        import arviz as az

        df = pd.DataFrame({
            "region": ["north", "south", "north"],
            "x": [0.0, 1.0, 2.0],
            "Y": [1.0, 4.5, 2.0],
        })
        model = pathmc.model("Y ~ x + region", data=df)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_Y": np.array([[[1.0, 0.5]]]),
                    "beta_Y_region": np.array([[[3.0]]]),
                    "sigma_Y": np.array([[0.1]]),
                }
            },
            coords={
                "Y_predictors": ["Intercept", "x"],
                "Y_region_levels": ["south"],
            },
            dims={
                "beta_Y": ["Y_predictors"],
                "beta_Y_region": ["Y_region_levels"],
            },
        )

        held = model.slopes("Y", "x", conditional={"region": "south"})
        assert float(held.dataset["Y"].mean()) == pytest.approx(0.5)
        contrast = model.comparisons(
            "Y", "x", contrast=(0.0, 1.0), conditional={"region": "south"}
        )
        assert float(contrast.dataset["Y"].mean()) == pytest.approx(0.5)

        with pytest.raises(ValueError, match="region") as exc_info:
            model.comparisons("Y", "x", conditional={"region": "delta"})
        message = str(exc_info.value)
        assert "region" in message
        assert "delta" in message
