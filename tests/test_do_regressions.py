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
"""Regression tests for previously fixed do() propagation bugs."""

import arviz as az
import numpy as np
import pandas as pd

import pathmc


def test_mean_do_propagates_bernoulli_mediator_on_response_scale() -> None:
    """Bernoulli mediators should feed E[M], not dummy zeros, into Y."""
    data = pd.DataFrame({
        "X": np.array([0.0, 1.0, 0.0, 1.0]),
        "M": np.array([0.0, 1.0, 0.0, 1.0]),
        "Y": np.array([0.0, 2.0, 0.0, 2.0]),
    })
    model = pathmc.model("M ~ X\nY ~ M", data=data, families={"M": "bernoulli"})
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_M": np.array([[[-2.0, 4.0]]]),
                "beta_Y": np.array([[[0.0, 2.0]]]),
                "sigma_Y": np.array([[1.0]]),
            }
        },
        coords={
            "M_predictors": ["Intercept", "X"],
            "Y_predictors": ["Intercept", "M"],
        },
        dims={"beta_M": ["M_predictors"], "beta_Y": ["Y_predictors"]},
    )

    r0 = model.do(set={"X": 0.0}, kind="mean")
    r1 = model.do(set={"X": 1.0}, kind="mean")

    expected_m0 = 1.0 / (1.0 + np.exp(2.0))
    expected_m1 = 1.0 / (1.0 + np.exp(-2.0))
    assert np.isclose(r0.mean("M"), expected_m0)
    assert np.isclose(r1.mean("M"), expected_m1)
    assert np.isclose(r0.mean("Y"), 2.0 * expected_m0)
    assert np.isclose(r1.mean("Y"), 2.0 * expected_m1)


def test_mean_do_propagates_poisson_mediator_on_response_scale() -> None:
    """Poisson mediators should feed E[M], not dummy zeros, into Y."""
    data = pd.DataFrame({
        "X": np.array([0.0, 1.0, 0.0, 1.0]),
        "M": np.array([1.0, 3.0, 1.0, 3.0]),
        "Y": np.array([2.0, 6.0, 2.0, 6.0]),
    })
    model = pathmc.model("M ~ X\nY ~ M", data=data, families={"M": "poisson"})
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_M": np.array([[[0.0, np.log(3.0)]]]),
                "beta_Y": np.array([[[0.0, 2.0]]]),
                "sigma_Y": np.array([[1.0]]),
            }
        },
        coords={
            "M_predictors": ["Intercept", "X"],
            "Y_predictors": ["Intercept", "M"],
        },
        dims={"beta_M": ["M_predictors"], "beta_Y": ["Y_predictors"]},
    )

    r0 = model.do(set={"X": 0.0}, kind="mean")
    r1 = model.do(set={"X": 1.0}, kind="mean")

    assert np.isclose(r0.mean("M"), 1.0)
    assert np.isclose(r1.mean("M"), 3.0)
    assert np.isclose(r0.mean("Y"), 2.0)
    assert np.isclose(r1.mean("Y"), 6.0)
