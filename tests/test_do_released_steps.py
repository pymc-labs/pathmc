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
"""A NaN-padded panel ``do(set=)`` reports the path its released steps took.

NaN steps of an endogenous intervention release it: the scan runs the
structural equation there, and downstream variables already follow that path.
The reported trajectory of the intervened variable must follow it too, instead
of echoing the NaN padding. The oracles are NumPy recurrences, independent of
the scan.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc

PANEL = {"unit": "unit", "time": "time"}
UNITS = ("A", "B")
N_TIMES = 5
HOLD = np.array([np.nan, 4.0, 4.0, np.nan, np.nan])


def _frame(columns: dict[str, float], n_times: int = N_TIMES) -> pd.DataFrame:
    rows = [
        {"unit": unit, "time": time, **columns}
        for unit in UNITS
        for time in range(n_times)
    ]
    return pd.DataFrame(rows)


def _pin_posterior(model: pathmc.PathModel, **values: float | list[float]) -> None:
    """Zero every free RV of the fitted graph, then pin the named ones.

    Zero carry and latent innovations keep ``kind="mean"`` on its structural
    path. A sigma of 1e-8 makes ``kind="predictive"`` match the same oracle.
    The ``observed_data`` group mirrors a real fit: posterior-predictive
    sampling on the generative graph takes its output names from it.
    """
    post = xr.Dataset(coords={"chain": [0], "draw": [0, 1]})
    for rv in model.pymc_model.free_RVs:
        shape = tuple(int(s) for s in rv.shape.eval())
        dims = ("chain", "draw", *(f"{rv.name}_dim_{i}" for i in range(len(shape))))
        array = np.zeros((1, 2, *shape))
        if rv.name in values:
            array[...] = values[rv.name]
        post[rv.name] = (dims, array)
    observed = xr.Dataset({
        rv.name: (
            tuple(f"{rv.name}_dim_{i}" for i in range(rv.ndim)),
            np.zeros(tuple(int(s) for s in rv.shape.eval())),
        )
        for rv in model.pymc_model.observed_RVs
    })
    model._idata = xr.DataTree.from_dict({
        "posterior": post,
        "observed_data": observed,
    })


def _recurrence(
    start: np.ndarray, step, hold: np.ndarray = HOLD, n_times: int = N_TIMES
) -> np.ndarray:
    """Per-unit ``(time, unit)`` path; non-NaN ``hold`` steps replace the state."""
    path = np.empty((n_times, len(start)))
    previous = np.asarray(start, dtype=float)
    for t in range(n_times):
        current = (
            step(previous) if np.isnan(hold[t]) else np.full_like(previous, hold[t])
        )
        path[t] = current
        previous = current
    return path


def _by_time(result, var: str) -> np.ndarray:
    da = result.dataset[var]
    assert set(da.dims) == {"time", "chain", "draw"}
    return da.transpose("time", "chain", "draw").values


@pytest.mark.parametrize("kind", ["mean", "predictive"])
def test_gaussian_lag_panel_reports_released_steps(kind: str) -> None:
    model = pathmc.model(
        "x ~ 0 + 0.5*lag(x)\ny ~ 0 + 2*x",
        data=_frame({"x": 1.0, "y": 2.0}),
        panel=PANEL,
        initial={"x": {"A": 2.0, "B": 6.0}},
    )
    _pin_posterior(model, sigma_x=1e-8, sigma_y=1e-8)

    result = model.do(set={"x": HOLD}, kind=kind, simulate_over="time", random_seed=3)

    x = _recurrence(np.array([2.0, 6.0]), lambda prev: 0.5 * prev).mean(axis=1)
    np.testing.assert_allclose(
        _by_time(result, "x"), x[:, None, None] * np.ones((1, 1, 2)), atol=1e-5
    )
    np.testing.assert_allclose(
        _by_time(result, "y"), 2 * x[:, None, None] * np.ones((1, 1, 2)), atol=1e-5
    )


@pytest.mark.parametrize("kind", ["mean", "predictive"])
def test_scaled_outcome_reports_released_steps_in_business_units(kind: str) -> None:
    factor = np.array([2.0, 10.0])
    model = pathmc.model(
        "y ~ 0 + 1*x + 0.5*lag(y)",
        data=_frame({"x": 1.0, "y": 3.0}),
        panel=PANEL,
        scaling=pathmc.Scaling(
            target={
                "method": "fixed",
                "value": dict(zip(UNITS, factor)),
                "dims": ("unit",),
            }
        ),
        initial={"y": {"A": 4.0, "B": 30.0}},
    )
    _pin_posterior(model, sigma_y=1e-8)
    hold = np.array([np.nan, 6.0, np.nan, np.nan, np.nan])

    result = model.do(set={"y": hold}, kind=kind, simulate_over="time", random_seed=3)

    # Model units: ys = x + 0.5 * lag(ys). Business units: y = f*x + 0.5 * lag(y).
    y = _recurrence(np.array([4.0, 30.0]), lambda prev: factor + 0.5 * prev, hold)
    np.testing.assert_allclose(
        _by_time(result, "y")[:, 0, 0], y.mean(axis=1), atol=1e-5
    )


@pytest.mark.parametrize("kind", ["mean", "predictive"])
def test_latent_state_reports_released_steps(kind: str) -> None:
    model = pathmc.model(
        "x ~ 0 + 0.5*lag(x)\ny ~ 0 + 2*x",
        data=_frame({"y": 2.0}),
        panel=PANEL,
        latent=["x"],
    )
    _pin_posterior(model, init_x=[2.0, 6.0], sigma_y=1e-8)

    result = model.do(set={"x": HOLD}, kind=kind, simulate_over="time", random_seed=3)

    x = _recurrence(np.array([2.0, 6.0]), lambda prev: 0.5 * prev).mean(axis=1)
    np.testing.assert_allclose(_by_time(result, "x")[:, 0, 0], x, atol=1e-5)
    np.testing.assert_allclose(_by_time(result, "y")[:, 0, 0], 2 * x, atol=1e-5)


@pytest.mark.parametrize("kind", ["mean", "predictive"])
def test_released_parent_of_a_count_outcome_reports_released_steps(kind: str) -> None:
    # A Poisson child keeps the model off the coherent Gaussian executor, so
    # kind="predictive" reads the released Gaussian parent from the PPC output.
    model = pathmc.model(
        "x ~ 0 + 0.5*lag(x)\ny ~ 0 + 1*x",
        data=_frame({"x": 1.0, "y": 2}),
        panel=PANEL,
        families={"y": "poisson"},
        initial={"x": {"A": 2.0, "B": 6.0}},
    )
    _pin_posterior(model, sigma_x=1e-8)

    result = model.do(set={"x": HOLD}, kind=kind, simulate_over="time", random_seed=3)

    x_path = _recurrence(np.array([2.0, 6.0]), lambda prev: 0.5 * prev)
    x = x_path.mean(axis=1)
    np.testing.assert_allclose(_by_time(result, "x")[:, 0, 0], x, atol=1e-5)
    y = _by_time(result, "y")[:, 0, 0]
    if kind == "mean":
        # Log link, then the same unit average the result reports.
        # exp of the averaged state would hide a per-unit rate bug.
        np.testing.assert_allclose(y, np.exp(x_path).mean(axis=1), atol=1e-5)
    else:
        assert np.isfinite(y).all()


def test_latent_predictive_with_fully_observed_outcome() -> None:
    # The fitted posterior has no draws for a fully observed outcome, which is
    # a free RV of the generative graph. Recovering an unintervened latent
    # state must not require them.
    model = pathmc.model(
        "x ~ 0 + 0.5*lag(x) + 1*z\ny ~ 0 + 2*x",
        data=_frame({"z": 0.0, "y": 2.0}),
        panel=PANEL,
        latent=["x"],
    )
    _pin_posterior(model, init_x=[2.0, 6.0], sigma_y=1e-8)

    result = model.do(
        set={"z": 1.0}, kind="predictive", simulate_over="time", random_seed=3
    )

    no_hold = np.full(N_TIMES, np.nan)
    x = _recurrence(np.array([2.0, 6.0]), lambda prev: 0.5 * prev + 1.0, no_hold)
    np.testing.assert_allclose(
        _by_time(result, "x")[:, 0, 0], x.mean(axis=1), atol=1e-5
    )
    np.testing.assert_allclose(
        _by_time(result, "y")[:, 0, 0], 2 * x.mean(axis=1), atol=1e-5
    )


@pytest.mark.parametrize("kind", ["mean", "predictive"])
def test_latent_normal_reports_released_steps(kind: str) -> None:
    # Stochastic latent recovery reads ``var``, not ``mu_var``.
    model = pathmc.model(
        "x ~ 0 + 0.5*lag(x)\ny ~ 0 + 2*x",
        data=_frame({"y": 2.0}),
        panel=PANEL,
        latent=["x"],
        families={"x": "latent_normal"},
    )
    _pin_posterior(model, init_x=[2.0, 6.0], sigma_x=1e-8, sigma_y=1e-8)

    result = model.do(set={"x": HOLD}, kind=kind, simulate_over="time", random_seed=3)

    # Unit mean of the 0.5*lag recurrence from init [2, 6], held at 4 on steps 1-2.
    x = np.array([2.0, 4.0, 4.0, 2.0, 1.0])
    np.testing.assert_allclose(_by_time(result, "x")[:, 0, 0], x, atol=1e-5)
    np.testing.assert_allclose(_by_time(result, "y")[:, 0, 0], 2 * x, atol=1e-5)


def test_scaled_predictive_overlays_held_values_in_business_units() -> None:
    # A Poisson child keeps predictive off the coherent executor, so the held
    # business-unit value is laid over the PPC path rather than the seam output.
    factor = np.array([2.0, 10.0])
    model = pathmc.model(
        "x ~ 0 + 1*z + 0.5*lag(x)\nc ~ 0 + 0.1*x",
        data=_frame({"z": 1.0, "x": 1.0, "c": 1}),
        panel=PANEL,
        families={"c": "poisson"},
        scaling=pathmc.Scaling(
            target={
                "method": "fixed",
                "value": dict(zip(UNITS, factor, strict=True)),
                "dims": ("unit",),
            }
        ),
        initial={"x": {"A": 4.0, "B": 30.0}},
    )
    _pin_posterior(model, sigma_x=1e-8)
    hold = np.array([np.nan, 6.0, np.nan, np.nan, np.nan])

    result = model.do(
        set={"x": hold}, kind="predictive", simulate_over="time", random_seed=3
    )

    x = _recurrence(np.array([4.0, 30.0]), lambda prev: factor + 0.5 * prev, hold)
    np.testing.assert_allclose(
        _by_time(result, "x")[:, 0, 0], x.mean(axis=1), atol=1e-5
    )
    np.testing.assert_allclose(
        _by_time(result, "x")[:, 0, 0],
        [14.5, 6.0, 9.0, 10.5, 11.25],
        atol=1e-5,
    )


def test_released_poisson_outcome_follows_the_rate_and_full_release() -> None:
    model = pathmc.model(
        "y ~ 0 + 0.1*lag(y)",
        data=_frame({"y": 1}),
        panel=PANEL,
        families={"y": "poisson"},
        initial={"y": 1.0},
    )
    _pin_posterior(model)
    # Stay far below the scan's mu clip so the oracle is the rate, not the clip.
    hold = np.array([np.nan, 2.0, np.nan, np.nan, np.nan])

    mean = model.do(set={"y": hold}, kind="mean", simulate_over="time")
    expected = _recurrence(np.ones(2), lambda prev: np.exp(0.1 * prev), hold)
    np.testing.assert_allclose(
        _by_time(mean, "y")[:, 0, 0], expected.mean(axis=1), atol=1e-5
    )

    seed = 11
    for kind in ("mean", "predictive"):
        baseline = model.do(kind=kind, simulate_over="time", random_seed=seed)
        scalar = model.do(
            set={"y": np.nan}, kind=kind, simulate_over="time", random_seed=seed
        )
        all_nan = model.do(
            set={"y": np.full(N_TIMES, np.nan)},
            kind=kind,
            simulate_over="time",
            random_seed=seed,
        )
        xr.testing.assert_allclose(scalar.dataset, baseline.dataset)
        xr.testing.assert_allclose(all_nan.dataset, baseline.dataset)

    predictive = model.do(
        set={"y": hold}, kind="predictive", simulate_over="time", random_seed=seed
    )
    got = _by_time(predictive, "y")[:, 0, 0]
    assert got[1] == 2.0
    assert np.isfinite(got).all()
