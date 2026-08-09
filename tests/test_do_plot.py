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
        result, time_index = _panel_result()
        observed = np.ones(len(time_index))
        result._observed_by_time["Y"] = observed
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
