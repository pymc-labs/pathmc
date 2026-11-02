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
"""Scan Poisson/NegBinomial clip warnings follow the linear predictor.

The scan keeps ``exp(clip(mu, -20, 20))`` as a numerical guard. A silent
``exp(20)`` is not the contract: when the linear predictor approaches the
bound, pathmc warns. That includes ``simulate()`` with no outcome column and
a large intercept with small observed counts, not only an observed count
already near the bound.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import pathmc

_PANEL = {"unit": "geo", "time": "week"}


def _panel(
    *,
    n_units: int = 2,
    n_times: int = 4,
    x: float = 0.0,
    z: float = 0.0,
    y: list[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for unit in range(n_units):
        for week in range(n_times):
            row = {
                "geo": f"g{unit}",
                "week": week,
                "X": x,
                "Z": z,
            }
            if y is not None:
                row["Y"] = y[week]
            rows.append(row)
    return pd.DataFrame(rows)


def _linear_predictor_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    messages = []
    for item in caught:
        text = str(item.message)
        if "Linear predictor" in text and "clip bound" in text:
            messages.append(text)
    return messages


def _simulate(spec: str, data: pd.DataFrame, params: dict, families: dict) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pathmc.simulate(
            spec,
            data=data,
            params=params,
            families=families,
            panel=_PANEL,
            random_seed=1,
        )
    return _linear_predictor_warnings(caught)


class TestScanRateClipFollowsLinearPredictor:
    def test_simulate_without_outcome_warns(self):
        """No outcome column: observed counts cannot disclose the clip."""
        warned = _simulate(
            "Y ~ lag(Z)",
            _panel(),
            {"beta_Y": [22.0, 0.0]},
            {"Y": "poisson"},
        )
        assert warned
        assert "Y" in warned[0]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cross = pathmc.simulate(
                "Y ~ X",
                data=_panel(x=1.0),
                params={"beta_Y": [22.0, 0.0]},
                families={"Y": "poisson"},
                random_seed=1,
            )
        assert not any("clip bound" in str(item.message) for item in caught)
        assert cross["Y"].mean() == pytest.approx(float(np.exp(22.0)), rel=0.05)

    def test_large_intercept_with_small_counts_warns(self):
        warned = _simulate(
            "Y ~ X + lag(Z)",
            _panel(y=[1.0, 0.0, 2.0, 1.0]),
            {"beta_Y": [25.0, 0.0, 0.0]},
            {"Y": "poisson"},
        )
        assert warned

    def test_negbinomial_slope_warns(self):
        """A slope, not the cited intercept, still drives the log-rate."""
        warned = _simulate(
            "Y ~ X + lag(Z)",
            _panel(x=1.0),
            {"beta_Y": [0.0, 21.0, 0.0], "alpha_disp_Y": 1.0},
            {"Y": "negbinomial"},
        )
        assert warned

    def test_log_rate_inside_margin_stays_quiet(self):
        warned = _simulate(
            "Y ~ lag(Z)",
            _panel(),
            {"beta_Y": [17.0, 0.0]},
            {"Y": "poisson"},
        )
        assert not warned

    def test_log_rate_inside_warning_margin_warns(self):
        """19 is bound-minus-one, so a special case for the cited 22 fails."""
        warned = _simulate(
            "Y ~ lag(Z)",
            _panel(),
            {"beta_Y": [19.0, 0.0]},
            {"Y": "poisson"},
        )
        assert warned

    def test_free_coefficients_do_not_warn_at_compile(self):
        """A free intercept is not a known linear predictor."""
        data = _panel(y=[1.0, 0.0, 2.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pathmc.model(
                "Y ~ X + lag(Y)",
                data=data,
                families={"Y": "poisson"},
                panel=_PANEL,
            )
        assert not _linear_predictor_warnings(caught)
