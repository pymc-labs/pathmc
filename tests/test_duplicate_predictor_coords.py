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
"""Labeled coefficients that share a predictor name must not share a column."""

import arviz as az
import numpy as np
import pandas as pd

import pathmc


def _pin(spec: str, columns: list[float]):
    x = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    data = pd.DataFrame({"X": x, "Y": np.linspace(0, 1, len(x))})
    model = pathmc.model(spec, data=data)
    order = [str(v) for v in model.pymc_model.coords["Y_predictors"]]
    model._idata = az.from_dict(
        {"posterior": {"beta_Y": np.tile(columns, (1, 4, 1))}},
        coords={"Y_predictors": order},
        dims={"beta_Y": ["Y_predictors"]},
    )
    return model, order


def test_linear_and_adstock_labels_keep_their_columns():
    model, order = _pin(
        "Y ~ a*X + b*adstock(X, decay=theta)",
        [0.0, 4.0, 9.0],
    )
    assert order == ["Intercept", "X", "adstock(X, decay=theta)"]
    assert len(order) == len(set(order))

    summary = model.effects_summary()
    assert summary.loc["a", "mean"] == 4.0
    assert summary.loc["b", "mean"] == 9.0
    assert summary.loc["a", "sd"] == 0.0
    assert summary.loc["b", "sd"] == 0.0
    # First matching edge is the linear term, not the mixture of both.
    assert model.effect("X -> Y").mean == 4.0


def test_two_transforms_of_one_variable_are_not_averaged():
    model, order = _pin(
        "Y ~ a*logistic_saturation(X, lam=lam) + b*adstock(X, decay=theta)",
        [0.0, 3.0, 8.0],
    )
    assert "X" not in order
    assert len(order) == len(set(order))
    summary = model.effects_summary()
    assert summary.loc["a", "mean"] == 3.0
    assert summary.loc["b", "mean"] == 8.0
    assert summary.loc["a", "sd"] == 0.0
    assert model.effect("X -> Y").mean == 3.0


def test_effect_skips_a_leading_fixed_term_without_mixing():
    model, order = _pin(
        "Y ~ 1*X + b*adstock(X, decay=theta)",
        [0.0, 9.0],
    )
    assert order == ["Intercept", "adstock(X, decay=theta)"]
    summary = model.effects_summary()
    assert summary.loc["b", "mean"] == 9.0
    assert summary.loc["b", "sd"] == 0.0
    assert model.effect("X -> Y").mean == 9.0
