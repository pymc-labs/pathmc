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
import pytest

import pathmc
from pathmc.simulate import run_do_panel_unified


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


def test_mean_do_averages_interaction_rows_within_each_draw() -> None:
    """Interaction heterogeneity must not be mistaken for posterior uncertainty."""
    data = pd.DataFrame({
        "T": np.array([0.0, 1.0, 0.0, 1.0]),
        "X": np.arange(4, dtype=float),
        "Y": np.zeros(4),
    })
    model = pathmc.model("Y ~ T + X + T:X", data=data)
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.array([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]]),
                "sigma_Y": np.ones((1, 2)),
            }
        },
        coords={"Y_predictors": ["Intercept", "T", "X", "T:X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )

    result = model.do(set={"T": 1.0}, kind="mean")

    assert result.dataset["Y"].dims == ("chain", "draw")
    np.testing.assert_allclose(result.draws("Y"), [13.5, 27.0])


@pytest.mark.parametrize("kind", ["time-forward", "forecast", "MEAN", ""])
def test_unrecognized_do_kind_raises_and_names_legal_values(kind: str) -> None:
    """A typo must not silently take the posterior-predictive branch."""
    data = pd.DataFrame({"X": np.array([0.0, 1.0]), "Y": np.array([0.0, 2.0])})
    model = pathmc.model("Y ~ X", data=data)
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.array([[[0.0, 2.0]]]),
                "sigma_Y": np.array([[0.3]]),
            }
        },
        coords={"Y_predictors": ["Intercept", "X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )

    with pytest.raises(ValueError, match="predictive") as exc_info:
        model.do(set={"X": 0.0}, kind=kind)

    message = str(exc_info.value)
    assert "mean" in message
    assert repr(kind) in message


def test_panel_do_rejects_unrecognized_kind_before_predictive() -> None:
    """Scan-panel do() shares the same kind contract as cross-sectional do()."""
    with pytest.raises(ValueError, match="mean") as exc_info:
        run_do_panel_unified(
            gen_model=None,  # type: ignore[arg-type]
            graph_info=None,  # type: ignore[arg-type]
            idata=None,  # type: ignore[arg-type]
            panel_info=None,  # type: ignore[arg-type]
            scan_info=None,  # type: ignore[arg-type]
            kind="time-forward",
        )

    assert "predictive" in str(exc_info.value)


def _scan_discrete_panel(family: str):
    return pathmc.model(
        "sales ~ tv + lag(sales)",
        data=pd.DataFrame([
            {"unit": unit, "time": time, "tv": float(time), "sales": 1}
            for unit in ("A", "B")
            for time in (0, 1, 2)
        ]),
        panel={"unit": "unit", "time": "time"},
        families={"sales": family},
    )


@pytest.mark.parametrize(
    ("family", "link"),
    [
        ("poisson", np.exp),
        ("bernoulli", lambda mu: 1.0 / (1.0 + np.exp(-mu))),
        ("negbinomial", np.exp),
    ],
)
def test_scan_discrete_mean_do_returns_response_scale(family: str, link) -> None:
    """kind='mean' on a scan discrete panel is the response-scale mean.

    A valid ``(n_times,)`` intervention must not be rejected because the
    mean path substituted a float ``mu`` for the integer outcome.
    """
    model = _scan_discrete_panel(family)
    pred = "sales_predictors"
    names = list(model.pymc_model.coords[pred])
    # Intercept 0, tv slope 1, lag coefficient 0: E[sales_t | do(tv)] = link(tv_t).
    beta = np.zeros((1, 1, len(names)))
    beta[0, 0, names.index("tv")] = 1.0
    posterior = {"beta_sales": beta}
    if family == "negbinomial":
        posterior["alpha_disp_sales"] = np.array([[1.0]])
    model._idata = az.from_dict(
        {"posterior": posterior},
        coords={pred: names},
        dims={"beta_sales": [pred]},
    )

    tv = np.array([0.0, 1.0, 2.0])
    result = model.do(set={"tv": tv}, simulate_over="time", kind="mean")
    got = np.asarray(result.by_time("sales")).reshape(3, -1)
    expected = link(tv)
    np.testing.assert_allclose(got, expected[:, None], rtol=1e-6, atol=1e-6)


def test_scan_poisson_mean_do_propagates_lag_without_noise() -> None:
    """A non-zero lag coefficient must carry the rate, not a Poisson draw."""
    model = _scan_discrete_panel("poisson")
    pred = "sales_predictors"
    names = list(model.pymc_model.coords[pred])
    beta = np.zeros((1, 1, len(names)))
    beta[0, 0, names.index("lag(sales)")] = 0.5
    model._idata = az.from_dict(
        {"posterior": {"beta_sales": beta}},
        coords={pred: names},
        dims={"beta_sales": [pred]},
    )

    result = model.do(set={"tv": np.zeros(3)}, simulate_over="time", kind="mean")
    got = np.asarray(result.by_time("sales")).reshape(3)
    # Pre-sample carry is the observed first row (1). Each later step lags the rate.
    rate0 = np.exp(0.5 * 1.0)
    rate1 = np.exp(0.5 * rate0)
    rate2 = np.exp(0.5 * rate1)
    np.testing.assert_allclose(got, [rate0, rate1, rate2], rtol=1e-6, atol=1e-6)
