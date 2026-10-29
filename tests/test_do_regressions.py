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


def _panel_exog_frame() -> pd.DataFrame:
    """X and Z are never zero; W is a real zero so a zero fill can be legitimate."""
    rows = []
    for unit, x, z in (("B", 4.0, 1.0), ("A", 8.0, 3.0)):
        for time in (2, 0, 1):
            rows.append({
                "unit": unit,
                "time": time,
                "X": x,
                "Z": z,
                "W": 0.0,
                "T": 0.0,
                "Y": 0.0,
            })
    return pd.DataFrame(rows)


def _pin_panel_posterior(model: pathmc.PathModel, *, x_coef: float) -> None:
    pred = "Y_predictors"
    names = list(model.pymc_model.coords[pred])
    beta = np.zeros(len(names))
    beta[names.index("X")] = x_coef
    n_times = len(model._data["time"].unique())
    n_units = len(model._data["unit"].unique())
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": beta.reshape(1, 1, -1),
                "sigma_Y": np.array([[1e-8]]),
                "carry_innovations_Y": np.zeros((1, 1, n_times, n_units)),
            }
        },
        coords={pred: names},
        dims={"beta_Y": [pred]},
    )


def test_panel_do_reports_unintervened_exogenous_fill() -> None:
    """Unspecified exogenous columns are the business-unit values the mean used.

    Writing 0 for every non-intervened exogenous column is wrong even when the
    cited column is not the constant 5 from the report, and even when another
    covariate's mean is a legitimate 0.
    """
    frame = _panel_exog_frame()
    model = pathmc.model(
        "Y ~ X + Z + W + T + lag(Y)",
        data=frame,
        panel={"unit": "unit", "time": "time"},
    )
    _pin_panel_posterior(model, x_coef=3.0)

    result = model.do(set={"T": 1.0}, simulate_over="time", kind="mean")
    cross = pathmc.model("Y ~ X + Z + W + T", data=frame)
    pred = "Y_predictors"
    names = list(cross.pymc_model.coords[pred])
    beta = np.zeros(len(names))
    beta[names.index("X")] = 3.0
    cross._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": beta.reshape(1, 1, -1),
                "sigma_Y": np.array([[1e-8]]),
            }
        },
        coords={pred: names},
        dims={"beta_Y": [pred]},
    )
    cross_result = cross.do(set={"T": 1.0}, kind="mean")

    assert result.dataset["Y"].mean() == pytest.approx(18.0)
    for var, expected in (("X", 6.0), ("Z", 2.0), ("W", 0.0)):
        assert var in result.dataset
        assert result.dataset[var].dims == cross_result.dataset[var].dims
        assert float(result.dataset[var].mean()) == pytest.approx(expected)
        assert float(cross_result.dataset[var].mean()) == pytest.approx(expected)
    assert "lag(Y)" not in result.dataset


def test_panel_do_exogenous_fill_is_business_units() -> None:
    """A scaled covariate is reported after converting rows back, not as 0."""
    frame = _panel_exog_frame()
    model = pathmc.model(
        "Y ~ X + Z + W + T + lag(Y)",
        data=frame,
        panel={"unit": "unit", "time": "time"},
        scaling=pathmc.Scaling(channel={"method": "fixed", "value": 10.0}),
    )
    _pin_panel_posterior(model, x_coef=3.0)

    result = model.do(set={"T": 1.0}, simulate_over="time", kind="mean")

    assert float(np.asarray(model._data["X"].to_numpy()).mean()) == pytest.approx(0.6)
    assert float(result.dataset["X"].mean()) == pytest.approx(6.0)
    assert float(result.dataset["Z"].mean()) == pytest.approx(2.0)
    assert result.dataset["Y"].mean() == pytest.approx(1.8)


def test_panel_do_fills_bool_and_object_numeric_exogenous() -> None:
    """Columns the scan consumes as numbers are reported, not dropped or zeroed."""
    rows = []
    for unit, b, o in (("B", True, 1.5), ("A", False, 4.5)):
        for time in (0, 1):
            rows.append({
                "unit": unit,
                "time": time,
                "B": b,
                "O": o,
                "T": 0.0,
                "Y": 0.0,
            })
    frame = pd.DataFrame(rows)
    frame["B"] = frame["B"].astype(bool)
    frame["O"] = frame["O"].astype(object)
    model = pathmc.model(
        "Y ~ B + O + T + lag(Y)",
        data=frame,
        panel={"unit": "unit", "time": "time"},
    )
    pred = "Y_predictors"
    names = list(model.pymc_model.coords[pred])
    beta = np.zeros(len(names))
    beta[names.index("B")] = 3.0
    beta[names.index("O")] = 2.0
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": beta.reshape(1, 1, -1),
                "sigma_Y": np.array([[1e-8]]),
                "carry_innovations_Y": np.zeros((1, 1, 2, 2)),
            }
        },
        coords={pred: names},
        dims={"beta_Y": [pred]},
    )

    result = model.do(set={"T": 1.0}, simulate_over="time", kind="mean")

    assert float(result.dataset["B"].mean()) == pytest.approx(0.5)
    assert float(result.dataset["O"].mean()) == pytest.approx(3.0)
    assert result.dataset["B"].dims == ("chain", "draw")
    assert result.dataset["O"].dims == ("chain", "draw")


def _static_panel_model(frame: pd.DataFrame) -> pathmc.PathModel:
    """Panel ``Y ~ X`` with an injected intercept-1, slope-2 posterior."""
    model = pathmc.model(
        "Y ~ X",
        data=frame,
        panel={"unit": "unit", "time": "week"},
    )
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.array([[[1.0, 2.0]]]),
                "sigma_Y": np.array([[0.1]]),
            }
        },
        coords={"Y_predictors": ["Intercept", "X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )
    return model


def test_static_panel_time_do_returns_time_aligned_trajectory() -> None:
    """A shuffled panel must not treat a length-n_times intervention as row order.

    Weeks are stored as [3, 1, 4, 2]. The intervention [0, 20, 10, 30] is in
    time order, so week 1 gets 0 and week 3 gets 10, not the reverse.
    """
    week = np.array([3, 1, 4, 2])
    observed_x = np.array([10.0, 0.0, 30.0, 20.0])
    frame = pd.DataFrame({
        "unit": ["A"] * 4,
        "week": week,
        "X": observed_x,
        "Y": 1.0 + 2.0 * observed_x,
    })
    model = _static_panel_model(frame)
    traj = [0.0, 20.0, 10.0, 30.0]
    result = model.do(set={"X": traj}, simulate_over="time", kind="mean")

    assert list(result.time_index) == [1, 2, 3, 4]
    np.testing.assert_allclose(result.by_time("X").mean(axis=1), traj)
    np.testing.assert_allclose(
        result.by_time("Y").mean(axis=1), 1.0 + 2.0 * np.array(traj)
    )
    assert result.dataset["Y"].dims == ("time", "chain", "draw")


def test_static_panel_time_do_averages_units_not_rows() -> None:
    """Two units with different row orders share one time-aligned trajectory."""
    rows = []
    for unit, weeks in (("A", [3, 1, 4, 2]), ("B", [2, 4, 1, 3])):
        for week in weeks:
            rows.append({"unit": unit, "week": week, "X": float(week), "Y": 1.0})
    model = _static_panel_model(pd.DataFrame(rows))
    traj = np.array([0.0, 20.0, 10.0, 30.0])
    result = model.do(set={"X": traj}, simulate_over="time", kind="mean")

    np.testing.assert_allclose(result.by_time("Y").mean(axis=1), 1.0 + 2.0 * traj)
    assert result.by_time("Y").shape[0] == 4


def test_static_panel_scalar_time_do_is_flat_trajectory() -> None:
    """A scalar time simulation is the same intervention at every time, not one average."""
    week = np.array([4, 1, 3, 2])
    frame = pd.DataFrame({
        "unit": ["A"] * 4,
        "week": week,
        "X": week.astype(float),
        "Y": np.ones(4),
    })
    model = _static_panel_model(frame)
    result = model.do(set={"X": 5.0}, simulate_over="time", kind="mean")

    np.testing.assert_allclose(result.by_time("Y").mean(axis=1), np.full(4, 11.0))
    plain = model.do(set={"X": 5.0}, kind="mean")
    assert plain.dataset["Y"].dims == ("chain", "draw")


def test_static_panel_time_do_rejects_row_length_vector() -> None:
    """A multi-unit vector must be one value per time, not one per row."""
    rows = []
    for unit in ("A", "B"):
        for week in (1, 2):
            rows.append({"unit": unit, "week": week, "X": 0.0, "Y": 1.0})
    model = _static_panel_model(pd.DataFrame(rows))
    with pytest.raises(ValueError, match="expected 2"):
        model.do(set={"X": [0.0, 1.0, 2.0, 3.0]}, simulate_over="time", kind="mean")
