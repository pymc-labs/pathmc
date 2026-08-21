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

import numpy as np
import pytest

from pathmc.effects import EffectResult
from pathmc.plotting import plot_density

from _draw_fixtures import do_result_from_flat, estimand_result_from_flat


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg")


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
