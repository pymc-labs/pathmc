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
"""Adjustment models keep the parent's business-unit scale contract (#600)."""

from __future__ import annotations

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pytest

import pathmc


def _pin(model, posterior, coords, dims) -> None:
    model._idata = az.from_dict({"posterior": posterior}, coords=coords, dims=dims)


def _scaled_parent(channel: float, outcome: float, beta_x: float):
    df = pd.DataFrame({
        "X": [0.0, channel, 2 * channel, 3 * channel],
        "C": [1.0, 2.0, 3.0, 4.0],
        "Y": [0.0, 0.0, 0.0, 0.0],
    })
    parent = pathmc.model(
        "Y ~ X + C",
        data=df,
        scaling=pathmc.Scaling(
            target={"method": "fixed", "value": outcome},
            channel={"method": "fixed", "value": channel},
        ),
    )
    _pin(
        parent,
        {
            "beta_Y": np.array([[[0.0, beta_x, 0.0]]]),
            "sigma_Y": np.array([[0.1]]),
        },
        coords={"Y_predictors": ["Intercept", "X", "C"]},
        dims={"beta_Y": ["Y_predictors"]},
    )
    return parent, df


def _pin_reduced(adjusted, beta_x: float) -> None:
    _pin(
        adjusted.outcome_model,
        {
            "beta_Y": np.array([[[0.0, beta_x]]]),
            "sigma_Y": np.array([[0.1]]),
        },
        coords={"Y_predictors": ["Intercept", "X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )


def _prediction(model, x: float, *, outcome: str | None = None) -> float:
    grid = pd.DataFrame({"X": [x], "C": [1.0], "Y": [0.0]})
    if outcome is None:
        result = model.predictions(newdata=grid)
    else:
        result = model.predictions(outcome, newdata=grid)
    return float(np.asarray(result.dataset["Y"]).reshape(-1)[0])


class TestAdjustmentKeepsParentScale:
    def test_cited_probe_agrees_with_parent(self):
        parent, _df = _scaled_parent(channel=100.0, outcome=10.0, beta_x=2.0)
        adjusted = parent.adjustment_model("X -> Y")
        _pin_reduced(adjusted, beta_x=2.0)

        factors = adjusted.outcome_model.fitted_scaling
        assert factors is not None
        assert factors.mean_factor("X", adjusted.outcome_model._data) == pytest.approx(
            100.0
        )
        assert factors.mean_factor("Y", adjusted.outcome_model._data) == pytest.approx(
            10.0
        )

        assert _prediction(parent, 500.0, outcome="Y") == pytest.approx(100.0)
        assert _prediction(adjusted, 500.0) == pytest.approx(100.0)
        assert float(parent.ate("Y", "X", values=(0.0, 100.0)).mean()) == pytest.approx(
            20.0
        )
        assert float(adjusted.ate(values=(0.0, 100.0)).mean()) == pytest.approx(20.0)
        assert float(
            np.asarray(adjusted.comparisons(contrast=(0.0, 100.0)).dataset["Y"]).mean()
        ) == pytest.approx(20.0)

    def test_nearby_scale_and_query_not_only_the_cited_pair(self):
        parent, df = _scaled_parent(channel=50.0, outcome=4.0, beta_x=3.0)
        adjusted = parent.adjustment_model("X -> Y", data=df)
        _pin_reduced(adjusted, beta_x=3.0)

        # 4 * 3 * (150/50) = 36; 0 → 200 is 4 * 3 * (200/50) = 48.
        assert _prediction(adjusted, 150.0) == pytest.approx(36.0)
        assert float(adjusted.ate(values=(0.0, 200.0)).mean()) == pytest.approx(48.0)
        assert float(adjusted.slopes(wrt="X").mean()) == pytest.approx(4.0 * 3.0 / 50.0)
        held = adjusted.datagrid(X=[0.0, 50.0])
        assert float(held["C"].iloc[0]) == pytest.approx(2.5)

    def test_extrapolation_warning_uses_business_units(self):
        parent, _df = _scaled_parent(channel=100.0, outcome=10.0, beta_x=2.0)
        adjusted = parent.adjustment_model("X -> Y")
        _pin_reduced(adjusted, beta_x=2.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adjusted.ate(values=(0.0, 100.0))
        assert not any("0.00, 3.00" in str(item.message) for item in caught)

        with pytest.warns(UserWarning, match=r"\[0\.00, 300\.00\]") as record:
            adjusted.ate(values=(0.0, 400.0))
        assert any("400.00" in str(item.message) for item in record)
        assert not any("0.00, 3.00" in str(item.message) for item in record)
