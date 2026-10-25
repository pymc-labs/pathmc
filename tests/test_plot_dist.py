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
"""Tests for the shared ``plot_dist`` posterior-density API (#323)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from pathmc.effects import EffectResult
from pathmc.plotting import plot_density
from pathmc.simulate import DoResult, EstimandResult
from _draw_fixtures import do_result_from_flat, estimand_result_from_flat


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _draws(n: int = 500) -> np.ndarray:
    """Deterministic 1-D draws for fixture construction."""
    return np.random.default_rng(42).normal(size=n)


def _has_dashed_line(fig) -> bool:
    return any(line.get_linestyle() == "--" for line in fig.axes[0].lines)


class TestPlotDensity:
    def test_returns_figure(self):
        fig = plot_density(_draws())
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_on_supplied_axis(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = plot_density(_draws(), ax=ax)
        assert out is fig

    def test_ref_line_drawn(self):
        assert _has_dashed_line(plot_density(_draws(), ref=0.0))

    def test_no_ref_line_by_default(self):
        assert not _has_dashed_line(plot_density(_draws()))

    def test_title_set(self):
        fig = plot_density(_draws(), label="a")
        assert any(t.get_text() == "a" for t in fig.axes[0].get_legend().get_texts())


class TestEffectResultPlotDist:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        result = EffectResult("a", _draws())
        result.plot_dist()
        fig = plt.gcf()
        assert any(t.get_text() == "a" for t in fig.axes[0].get_legend().get_texts())

    def test_on_supplied_axis(self):
        import matplotlib.pyplot as plt

        result = EffectResult("a", _draws())
        fig, ax = plt.subplots()
        result.plot_dist(ax=ax)
        assert ax.get_figure() is fig

    def test_ref_line(self):
        import matplotlib.pyplot as plt

        result = EffectResult("a", _draws())
        result.plot_dist(ref=0.0)
        assert _has_dashed_line(plt.gcf())


def _estimand(values: dict[str, np.ndarray] | None = None) -> object:
    return estimand_result_from_flat(
        values or {"Y": _draws()},
        outcome="Y",
        treatment="X",
        estimand="ATE",
        n_chains=2,
        n_draws=250,
    )


class TestEstimandResultPlotDist:
    def test_defaults_to_outcome(self):
        import matplotlib.pyplot as plt

        _estimand().plot_dist()
        assert any(
            t.get_text() == "Y" for t in plt.gcf().axes[0].get_legend().get_texts()
        )

    def test_explicit_var(self):
        import matplotlib.pyplot as plt

        result = _estimand({"Y": _draws(), "M": _draws()})
        result.plot_dist(var="M")
        assert any(
            t.get_text() == "M" for t in plt.gcf().axes[0].get_legend().get_texts()
        )

    def test_unknown_var_raises(self):
        with pytest.raises(KeyError, match="Unknown variable"):
            _estimand().plot_dist(var="zzz")

    def test_on_supplied_axis(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _estimand().plot_dist(ax=ax)
        assert ax.get_figure() is fig

    def test_ref_line(self):
        import matplotlib.pyplot as plt

        _estimand().plot_dist(ref=0.0)
        assert _has_dashed_line(plt.gcf())


def _do_result(multi: bool = False) -> object:
    values: dict[str, np.ndarray] = {"Y": _draws()}
    if multi:
        values["M"] = _draws()
    return do_result_from_flat(values, n_chains=2, n_draws=250)


class TestDoResultPlotDist:
    def test_single_var_defaults(self):
        import matplotlib.pyplot as plt

        _do_result().plot_dist()
        assert any(
            t.get_text() == "Y" for t in plt.gcf().axes[0].get_legend().get_texts()
        )

    def test_multi_var_requires_selection(self):
        with pytest.raises(ValueError, match="Multiple variables"):
            _do_result(multi=True).plot_dist()

    def test_explicit_var(self):
        import matplotlib.pyplot as plt

        _do_result(multi=True).plot_dist(var="M")
        assert any(
            t.get_text() == "M" for t in plt.gcf().axes[0].get_legend().get_texts()
        )

    def test_unknown_var_raises(self):
        with pytest.raises(KeyError, match="Unknown variable"):
            _do_result().plot_dist(var="zzz")

    def test_on_supplied_axis(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _do_result().plot_dist(ax=ax)
        assert ax.get_figure() is fig

    def test_ref_line(self):
        import matplotlib.pyplot as plt

        _do_result().plot_dist(ref=0.0)
        assert _has_dashed_line(plt.gcf())


@pytest.mark.slow
class TestPathModelPlotDist:
    def test_labeled_coefficients(self, fitted_mediation):
        import matplotlib.pyplot as plt

        fitted_mediation.plot_dist(var=["a", "b", "c"])
        ax = plt.gcf().axes[0]
        assert len(ax.lines) == 3
        assert ax.get_legend() is not None

    def test_defined_params(self, fitted_mediation):
        import matplotlib.pyplot as plt

        fitted_mediation.plot_dist(var=["indirect"])
        assert len(plt.gcf().axes[0].lines) == 1

    def test_raw_posterior_var(self, fitted_mediation):
        import matplotlib.pyplot as plt

        fitted_mediation.plot_dist(var=["sigma_Y"])
        assert len(plt.gcf().axes[0].lines) == 1

    def test_raw_var_with_coords(self, fitted_mediation):
        fitted_mediation.plot_dist(var=["beta_Y"], coords={"Y_predictors": "X"})

    def test_raw_var_without_coords_raises(self, fitted_mediation):
        with pytest.raises(ValueError, match="multiple coordinates"):
            fitted_mediation.plot_dist(var=["beta_Y"])

    def test_ref_list(self, fitted_mediation):
        fitted_mediation.plot_dist(var=["a", "b"], ref=[0.5, 0.8])

    def test_ref_list_wrong_length_raises(self, fitted_mediation):
        with pytest.raises(ValueError, match=r"len\(var\)"):
            fitted_mediation.plot_dist(var=["a", "b", "c"], ref=[0.5, 0.8])

    def test_unknown_name_raises(self, fitted_mediation):
        with pytest.raises(KeyError, match="Unknown name"):
            fitted_mediation.plot_dist(var=["zzz"])

    def test_unfitted_raises(self, mediation_data):
        import pathmc

        spec = "M ~ a*X\nY ~ b*M + c*X\nindirect := a*b"
        model = pathmc.model(spec, data=mediation_data)
        with pytest.raises(RuntimeError, match="plot_dist"):
            model.plot_dist(var=["a"])


def _legend_text(fig) -> str:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return " ".join(text.get_text() for text in legend.get_texts())


def _kde_mode(line) -> float:
    x = np.asarray(line.get_xdata(), dtype=float)
    y = np.asarray(line.get_ydata(), dtype=float)
    return float(x[np.argmax(y)])


def _has_open_interval_density(ax, lo: float, hi: float) -> bool:
    """True when a non-vertical line puts positive density strictly inside (lo, hi)."""
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size == 0 or np.nanmax(x) - np.nanmin(x) < 1e-9:
            continue
        interior = (x > lo) & (x < hi) & np.isfinite(y) & (y > 0)
        if interior.any():
            return True
    return False


class TestPlotDensityDrawTypes:
    def test_constant_draws_are_a_finite_point_mass(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = plot_density(np.ones(100), label="X")
        assert not any("single value" in str(item.message) for item in caught)
        ax = fig.axes[0]
        assert any("point mass at 1" in text.get_text() for text in ax.texts)
        ys = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float) for line in ax.lines
        ])
        assert np.isfinite(ys).all()
        assert any(
            np.all(np.asarray(line.get_xdata(), dtype=float) == 1.0)
            for line in ax.lines
        )

    def test_integer_draws_have_no_density_between_support(self):
        draws = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=float)
        fig = plot_density(draws, label="Y")
        ax = fig.axes[0]
        assert not _has_open_interval_density(ax, 0.0, 1.0)
        centers = sorted(patch.get_x() + patch.get_width() / 2 for patch in ax.patches)
        assert centers == pytest.approx([0.0, 1.0])
        assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(1.0)
        assert ax.get_ylabel() == "Probability"


def _time_split_result(cls):
    noise = np.random.default_rng(7).normal(0, 0.25, (2, 100))
    ds = xr.Dataset(
        {"Y": (("time", "chain", "draw"), np.stack([-10 + noise, 10 + noise]))},
        coords={"time": ["before", "after"]},
    )
    if cls is DoResult:
        return DoResult(ds=ds)
    return EstimandResult(ds=ds, outcome="Y", treatment="X", estimand="ATE")


class TestTimeIndexedPlotDist:
    @pytest.mark.parametrize("cls", [DoResult, EstimandResult])
    def test_default_refuses_time_average(self, cls):
        result = _time_split_result(cls)
        with pytest.raises(ValueError, match="time_mean") as exc:
            result.plot_dist(var="Y")
        message = str(exc.value)
        assert "by_time" in message
        assert "average over time" in message
        if cls is DoResult:
            assert ".plot(" in message
        else:
            assert ".plot(" not in message

    @pytest.mark.parametrize("cls", [DoResult, EstimandResult])
    def test_time_mean_is_labeled_and_collapsed(self, cls):
        fig = _time_split_result(cls).plot_dist(var="Y", aggregate="time_mean")
        assert "time mean" in _legend_text(fig)
        assert abs(_kde_mode(fig.axes[0].lines[0])) < 2

    def test_time_coord_selects_one_period(self):
        fig = _time_split_result(DoResult).plot_dist(var="Y", coords={"time": "before"})
        assert "before" in _legend_text(fig)
        assert _kde_mode(fig.axes[0].lines[0]) == pytest.approx(-10, abs=2)

    def test_single_time_point_needs_no_aggregate(self):
        draws = np.random.default_rng(1).normal(size=(1, 2, 20))
        ds = xr.Dataset(
            {"Y": (("time", "chain", "draw"), draws)},
            coords={"time": ["only"]},
        )
        fig = DoResult(ds=ds).plot_dist()
        assert _legend_text(fig) == "Y"


def _two_unit_result() -> DoResult:
    rng = np.random.default_rng(0)
    data = np.stack(
        [rng.normal(0, 0.25, (2, 40)), rng.normal(100, 0.25, (2, 40))],
        axis=-1,
    )
    return DoResult(
        ds=xr.Dataset(
            {"Y": (("chain", "draw", "unit"), data)},
            coords={"unit": ["A", "B"]},
        )
    )


class TestUnitPlotDist:
    def test_default_refuses_pooled_units(self):
        with pytest.raises(ValueError, match="unit_pool") as exc:
            _two_unit_result().plot_dist(var="Y")
        assert "coords" in str(exc.value)

    def test_unit_selector_plots_that_unit_only(self):
        fig = _two_unit_result().plot_dist(var="Y", coords={"unit": "A"})
        assert "unit=A" in _legend_text(fig)
        x = np.asarray(fig.axes[0].lines[0].get_xdata(), dtype=float)
        assert np.nanmax(np.abs(x)) < 5

    def test_unit_pool_is_labeled_and_spans_both_groups(self):
        fig = _two_unit_result().plot_dist(var="Y", aggregate="unit_pool")
        assert "pooled over unit" in _legend_text(fig)
        x = np.asarray(fig.axes[0].lines[0].get_xdata(), dtype=float)
        assert x.min() < 5
        assert x.max() > 90

    def test_single_unit_needs_no_selector(self):
        draws = np.random.default_rng(2).normal(size=(2, 10, 1))
        ds = xr.Dataset(
            {"Y": (("chain", "draw", "unit"), draws)},
            coords={"unit": ["only"]},
        )
        fig = DoResult(ds=ds).plot_dist()
        assert _legend_text(fig) == "Y"


def _coord_model():
    import arviz as az
    import pandas as pd

    import pathmc

    frame = pd.DataFrame({
        "X": [0.0, 1.0, 0.0, 1.0],
        "M": [0.1, 0.2, 0.3, 0.4],
        "Y": [1.0, 1.1, 0.9, 1.2],
    })
    model = pathmc.model("M ~ X\nY ~ M + X", data=frame)
    beta_m = np.linspace(0.0, 1.0, 8).reshape(1, 8, 1)
    beta_y = np.column_stack([
        np.linspace(0.0, 1.0, 8),
        np.linspace(10.0, 11.0, 8),
    ]).reshape(1, 8, 2)
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_M": beta_m,
                "beta_Y": beta_y,
                "gamma": np.zeros((1, 4, 2, 3)),
            }
        },
        coords={
            "M_predictors": ["X"],
            "Y_predictors": ["M", "X"],
            "group": ["g0", "g1"],
            "level": ["l0", "l1", "l2"],
        },
        dims={
            "beta_M": ["M_predictors"],
            "beta_Y": ["Y_predictors"],
            "gamma": ["group", "level"],
        },
    )
    return model


class TestCoordinateSelection:
    def test_size_one_dimension_needs_no_coords(self):
        fig = _coord_model().plot_dist(var="beta_M")
        assert "beta_M" in _legend_text(fig)

    def test_empty_coords_does_not_bypass_multidim(self):
        with pytest.raises(ValueError, match="multiple coordinates"):
            _coord_model().plot_dist(var="beta_Y", coords={})

    def test_partial_selector_does_not_pool_remaining_dim(self):
        with pytest.raises(ValueError, match="multiple coordinates"):
            _coord_model().plot_dist(var="gamma", coords={"group": "g0"})

    def test_overlay_applies_only_matching_selectors(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        out = _coord_model().plot_dist(
            var=["beta_M", "beta_Y"],
            coords={"M_predictors": "X", "Y_predictors": "X"},
            ax=ax,
        )
        assert out is fig
        assert len(ax.lines) == 2
        legend = _legend_text(fig)
        assert "beta_M" in legend
        assert "beta_Y" in legend
        assert "Y_predictors=X" in legend

    def test_unused_selector_raises(self):
        with pytest.raises(ValueError, match="did not match"):
            _coord_model().plot_dist(var=["beta_M"], coords={"Y_predictors": "X"})

    def test_effect_plot_dist_returns_figure(self):
        import matplotlib.figure

        fig = EffectResult("a", np.random.default_rng(3).normal(size=50)).plot_dist()
        assert isinstance(fig, matplotlib.figure.Figure)


class TestCriticRegressions:
    def test_mixed_density_and_point_mass_raises(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        plot_density(np.linspace(0.0, 1.0, 30), ax=ax)
        with pytest.raises(ValueError, match="do not share a scale"):
            plot_density(np.ones(20), ax=ax)

    def test_point_mass_then_density_raises(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        plot_density(np.ones(20), ax=ax)
        with pytest.raises(ValueError, match="do not share a scale"):
            plot_density(np.linspace(0.0, 1.0, 30), ax=ax)

    def test_pathmodel_refuses_mixed_overlay(self):
        model = _coord_model()
        model._idata.posterior["sigma_M"] = xr.DataArray(
            np.ones((1, 8)),
            dims=("chain", "draw"),
        )
        with pytest.raises(ValueError, match="do not share a scale"):
            model.plot_dist(["beta_M", "sigma_M"])

    def test_time_mean_rejects_nonfinite(self):
        values = np.zeros((2, 1, 40))
        values[0] = -10.0
        values[1] = np.nan
        result = DoResult(
            ds=xr.Dataset(
                {"Y": (("time", "chain", "draw"), values)},
                coords={"time": [-10.0, 10.0]},
            )
        )
        with pytest.raises(ValueError, match="non-finite"):
            result.plot_dist(aggregate="time_mean")

    def test_supplied_ax_does_not_move_sibling(self):
        import matplotlib.pyplot as plt

        _, axes = plt.subplots(1, 2)
        before = axes[1].get_position().bounds
        plot_density(np.linspace(0.0, 1.0, 40), ax=axes[0])
        assert axes[1].get_position().bounds == pytest.approx(before)

    def test_detached_axes_raises(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.delaxes(ax)
        with pytest.raises(ValueError, match="not attached"):
            plot_density(np.linspace(0.0, 1.0, 20), ax=ax)

    def test_scan_panel_raw_posterior_uses_time_policy(self):
        import arviz as az
        import pandas as pd

        import pathmc

        frame = pd.DataFrame([
            {"unit": unit, "time": time, "x": 0.1 * time, "y": 0.2 * time}
            for unit in ("A", "B")
            for time in range(3)
        ])
        model = pathmc.model(
            "y ~ a*lag(y) + b*x",
            data=frame,
            panel={"unit": "unit", "time": "time"},
        )
        mu = np.zeros((1, 40, 3, 2))
        mu[:, :, 0, :] = -10.0
        mu[:, :, 2, :] = 10.0
        model._idata = az.from_dict(
            {
                "posterior": {
                    "mu_y": mu,
                    "beta_y": np.zeros((1, 40, 3)),
                    "sigma_y": np.ones((1, 40)),
                }
            },
            coords={"y_predictors": ["Intercept", "lag(y)", "x"]},
            dims={
                "mu_y": ["mu_y_dim_0", "mu_y_dim_1"],
                "beta_y": ["y_predictors"],
            },
        )
        with pytest.raises(ValueError, match="time points") as exc:
            model.plot_dist("mu_y")
        assert "more than one time point" not in str(exc.value)
        with pytest.raises(ValueError, match="units"):
            model.plot_dist("mu_y", aggregate="time_mean")
        fig = model.plot_dist(
            "mu_y",
            aggregate="time_mean",
            coords={"unit": "A"},
        )
        legend = _legend_text(fig)
        assert "time mean" in legend
        assert "unit=A" in legend
        x = np.asarray(fig.axes[0].lines[0].get_xdata(), dtype=float)
        y = np.asarray(fig.axes[0].lines[0].get_ydata(), dtype=float)
        assert abs(float(x[int(np.argmax(y))])) < 2
        with pytest.raises(KeyError) as exc:
            model.plot_dist("mu_y", coords={"time": "0", "unit": "A"})
        available = str(exc.value).split("Available values:", 1)[1]
        assert "[0, 1, 2]" in available
        assert "'0'" not in available
        selected = model.plot_dist("mu_y", coords={"time": 0, "unit": "A"})
        assert "time=0" in _legend_text(selected)
        with pytest.raises(ValueError, match="y_predictors"):
            model.plot_dist("beta_y")

    def test_point_mass_overlay_stays_inside_supplied_axes(self):
        import matplotlib.pyplot as plt

        for bars_first in (True, False):
            fig, ax = plt.subplots()
            bars = np.array([0, 1, 0, 1])
            mass = np.full(8, 5.0)
            if bars_first:
                plot_density(bars, ax=ax)
                plot_density(mass, ax=ax)
            else:
                plot_density(mass, ax=ax)
                plot_density(bars, ax=ax)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax_box = ax.get_window_extent(renderer)
            text = ax.texts[0]
            box = text.get_window_extent(renderer)
            assert box.x0 >= ax_box.x0 - 1
            assert box.x1 <= ax_box.x1 + 1
            assert "point mass at 5" in text.get_text()
            ticks = [float(tick) for tick in ax.get_xticks()]
            assert any(abs(tick - 5) < 1e-6 for tick in ticks)
            assert any(abs(tick) < 1e-6 for tick in ticks)
            assert any(abs(tick - 1) < 1e-6 for tick in ticks)
            plt.close(fig)

    def test_point_mass_label_follows_later_artists(self):
        import matplotlib
        import matplotlib.pyplot as plt

        def assert_inside(fig: matplotlib.figure.Figure) -> None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            ax = fig.axes[0]
            ax_box = ax.get_window_extent(renderer)
            labels = [text for text in ax.texts if "point mass" in text.get_text()]
            assert labels
            for text in labels:
                box = text.get_window_extent(renderer)
                assert box.x0 >= ax_box.x0 - 1, text.get_text()
                assert box.x1 <= ax_box.x1 + 1, text.get_text()

        for mass, bars in (
            (np.full(8, -2.0), np.array([0, 1, 0, 1])),
            (np.full(8, 5.0), np.array([10, 11, 10, 11])),
        ):
            fig, ax = plt.subplots()
            plot_density(mass, ax=ax)
            plot_density(bars, ax=ax)
            assert_inside(fig)
            plt.close(fig)

        fig, ax = plt.subplots()
        plot_density(np.full(8, -2.0), ref=0, ax=ax)
        assert_inside(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        EffectResult(name="const", draws=np.full(8, -2.0)).plot_dist(ref=0, ax=ax)
        assert_inside(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        plot_density(np.zeros(8), ax=ax)
        plot_density(np.full(8, 5.0), ax=ax)
        assert_inside(fig)
        plt.close(fig)

        model = _coord_model()
        model._idata.posterior["const"] = xr.DataArray(
            np.full((1, 8), -2.0),
            dims=("chain", "draw"),
        )
        model._idata.posterior["counts"] = xr.DataArray(
            np.tile([0.0, 1.0], 4).reshape(1, 8),
            dims=("chain", "draw"),
        )
        assert_inside(model.plot_dist(["const", "counts"]))

    def test_point_mass_returns_figure_without_a_canvas_renderer(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import FigureCanvasPdf
        from matplotlib.backends.backend_ps import FigureCanvasPS
        from matplotlib.backends.backend_svg import FigureCanvasSVG
        from matplotlib.figure import Figure

        for canvas_cls in (FigureCanvasPdf, FigureCanvasSVG, FigureCanvasPS):
            fig = Figure()
            canvas_cls(fig)
            ax = fig.add_subplot(111)
            returned = plot_density(np.full(8, -2.0), ref=0, ax=ax)
            assert returned is fig
            assert any("point mass at -2" in text.get_text() for text in ax.texts)

        closed, closed_ax = plt.subplots()
        plt.close(closed)
        returned = plot_density(np.full(8, -2.0), ax=closed_ax)
        assert returned is closed

    def test_point_mass_does_not_move_a_pinned_sibling(self):
        import matplotlib.pyplot as plt

        pin = (0.7, 0.2, 0.2, 0.6)
        for layout in ("tight", "constrained"):
            fig, axes = plt.subplots(1, 2, layout=layout)
            fig.canvas.draw()
            axes[1].set_position(pin)
            plot_density(np.full(8, -2.0), ax=axes[0])
            assert axes[1].get_position().bounds == pytest.approx(pin)
            plt.close(fig)

            fig, axes = plt.subplots(1, 2, layout=layout)
            before = axes[1].get_position().bounds
            plot_density(np.full(8, -2.0), ax=axes[0])
            assert axes[1].get_position().bounds == pytest.approx(before)
            plt.close(fig)

    def test_cleared_axes_accepts_a_different_kind(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        plot_density(np.linspace(0.0, 1.0, 30), ax=ax)
        ax.clear()
        plot_density(np.ones(10), ax=ax)
        assert any("point mass" in text.get_text() for text in ax.texts)
