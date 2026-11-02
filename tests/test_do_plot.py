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
"""Tests for DoResult.plot() panel trajectory visualization (#111)."""

from __future__ import annotations

import numpy as np
import pytest

from _draw_fixtures import do_result_from_flat


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib

    matplotlib.use("Agg")


def _panel_result(
    n_times: int = 4,
    n_chains: int = 2,
    n_draws: int = 3,
    base: float = 1.0,
) -> tuple[object, np.ndarray]:
    """Build a panel DoResult with deterministic per-time means."""
    time_index = np.arange(1, n_times + 1)
    arr = np.zeros((n_times, n_chains, n_draws))
    for t in range(n_times):
        arr[t, :, :] = base + t
    values_by_time = {"Y": arr}
    result = do_result_from_flat(
        values_by_time=values_by_time,
        time_index=time_index,
        n_chains=n_chains,
        n_draws=n_draws,
    )
    return result, time_index


class TestDoResultPlot:
    def test_cross_sectional_raises(self):
        result = do_result_from_flat(
            values={"Y": np.array([1.0, 2.0, 3.0, 4.0])},
            n_chains=2,
            n_draws=2,
        )
        with pytest.raises(ValueError, match="plot_dist"):
            result.plot("Y")

    def test_plot_returns_figure(self):
        result, _ = _panel_result()
        fig = result.plot("Y")
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_on_supplied_axis(self):
        import matplotlib.pyplot as plt

        result, _ = _panel_result()
        fig, ax = plt.subplots()
        out = result.plot("Y", ax=ax)
        assert out is fig

    def test_plot_with_observed_kwarg(self):
        result, time_index = _panel_result()
        observed = np.linspace(0.5, 2.0, len(time_index))
        fig = result.plot("Y", observed=observed)
        ax = fig.axes[0]
        assert len(ax.lines) >= 2

    def test_vs_observed_without_data_raises(self):
        result, _ = _panel_result()
        with pytest.raises(ValueError, match="observed="):
            result.plot("Y", vs="observed")

    def test_vs_observed_uses_attached_metadata(self):
        n_times = 4
        time_index = np.arange(1, n_times + 1)
        arr = np.ones((n_times, 2, 3))
        result = do_result_from_flat(
            values_by_time={"Y": arr},
            time_index=time_index,
            n_chains=2,
            n_draws=3,
            observed_by_time={"Y": np.ones(n_times)},
        )
        fig = result.plot("Y", vs="observed")
        ax = fig.axes[0]
        assert any(line.get_label() == "Observed" for line in ax.lines)

    def test_observed_wrong_length_raises(self):
        result, _ = _panel_result()
        with pytest.raises(ValueError, match="length"):
            result.plot("Y", observed=np.array([1.0, 2.0]))

    def test_contrast_plot(self):
        low, _ = _panel_result(base=1.0)
        high, _ = _panel_result(base=3.0)
        contrast = high - low
        fig = contrast.plot("Y")
        assert fig is not None


def _line(fig, label: str) -> np.ndarray:
    matches = [
        line.get_ydata()
        for line in fig.axes[0].get_lines()
        if line.get_label() == label
    ]
    assert matches, f"missing {label} line"
    return np.asarray(matches[0], dtype=float)


def _pin_scan_posterior(model, *, beta_y: float, n_times: int, n_units: int) -> None:
    import arviz as az

    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.array([[[beta_y]]]),
                "beta_W": np.array([[[0.0]]]),
                "sigma_Y": np.array([[0.0]]),
                "sigma_W": np.array([[0.0]]),
                "carry_innovations_W": np.zeros((1, 1, n_times, n_units)),
            }
        },
        coords={"Y_predictors": ["X"], "W_predictors": ["lag(W)"]},
        dims={
            "beta_Y": ["Y_predictors"],
            "beta_W": ["W_predictors"],
            "carry_innovations_W": ["t", "u"],
        },
    )


class TestObservedOverlayBusinessUnits:
    """``plot(vs='observed')`` must not draw the internal column."""

    def test_fixed_target_overlay_matches_business_column(self):
        import pandas as pd

        import pathmc

        # Not the triage probe (factor 10, Y=[10, 30, 70]): a special case
        # for those inputs would still fail here.
        observed = [8.0, 16.0, 28.0]
        factor = 4.0
        df = pd.DataFrame([
            {"unit": "A", "time": t, "Y": y, "X": 2.0, "W": 0.0}
            for t, y in enumerate(observed)
        ])
        model = pathmc.model(
            "Y ~ 0 + X\nW ~ 0 + lag(W)",
            data=df,
            panel={"unit": "unit", "time": "time"},
            scaling=pathmc.Scaling(target={"method": "fixed", "value": factor}),
        )
        _pin_scan_posterior(model, beta_y=3.0, n_times=3, n_units=1)
        result = model.do(
            set={"X": np.array([1.0, 2.0, 4.0])},
            simulate_over="time",
            kind="mean",
        )
        fig = result.plot("Y", vs="observed")
        np.testing.assert_allclose(_line(fig, "Counterfactual"), [12.0, 24.0, 48.0])
        np.testing.assert_allclose(_line(fig, "Observed"), observed)
        assert not np.allclose(_line(fig, "Observed"), np.asarray(observed) / factor)

    def test_heterogeneous_target_uses_row_business_units(self):
        import pandas as pd

        import pathmc

        # Unit means of the internal column times the mean factor are
        # [22.5, 26.25], not the business-unit unit means [25, 22.5].
        rows = [
            {"unit": "A", "time": 0, "Y": 10.0, "X": 1.0, "W": 0.0},
            {"unit": "A", "time": 1, "Y": 25.0, "X": 1.0, "W": 0.0},
            {"unit": "B", "time": 0, "Y": 40.0, "X": 1.0, "W": 0.0},
            {"unit": "B", "time": 1, "Y": 20.0, "X": 1.0, "W": 0.0},
        ]
        model = pathmc.model(
            "Y ~ 0 + X\nW ~ 0 + lag(W)",
            data=pd.DataFrame(rows),
            panel={"unit": "unit", "time": "time"},
            scaling=pathmc.Scaling(
                target={
                    "method": "divide",
                    "by": {"A": 5.0, "B": 10.0},
                    "dims": ("unit",),
                }
            ),
        )
        _pin_scan_posterior(model, beta_y=2.0, n_times=2, n_units=2)
        result = model.do(
            set={"X": np.array([1.0, 2.0])},
            simulate_over="time",
            kind="mean",
        )
        observed = _line(result.plot("Y", vs="observed"), "Observed")
        np.testing.assert_allclose(observed, [25.0, 22.5])

    def test_channel_scaled_predictor_overlay_stays_business_units(self):
        import pandas as pd

        import pathmc

        raw_x = [6.0, 12.0, 18.0]
        df = pd.DataFrame([
            {"unit": "A", "time": t, "Y": 1.0, "X": x, "W": 0.0}
            for t, x in enumerate(raw_x)
        ])
        model = pathmc.model(
            "Y ~ 0 + X\nW ~ 0 + lag(W)",
            data=df,
            panel={"unit": "unit", "time": "time"},
            scaling=pathmc.Scaling(channel={"method": "fixed", "value": 3.0}),
        )
        _pin_scan_posterior(model, beta_y=1.0, n_times=3, n_units=1)
        result = model.do(
            set={"X": np.array([3.0, 6.0, 9.0])},
            simulate_over="time",
            kind="mean",
        )
        observed = _line(result.plot("X", vs="observed"), "Observed")
        np.testing.assert_allclose(observed, raw_x)
        assert not np.allclose(observed, np.asarray(raw_x) / 3.0)
