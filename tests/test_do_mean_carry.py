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
"""Panel ``do(kind="mean")`` carries a lagged observed outcome at its mean.

A lagged Gaussian outcome carries ``mu + sigma * carry_innovations`` through
the scan. Where the outcome was observed, fitting reads the observed lag, so
the posterior of those innovations is their N(0, 1) prior. ``kind="mean"``
must not feed them into the next step. The oracles are NumPy recurrences.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc.simulate import DoResult

PANEL = {"unit": "unit", "time": "time"}
N_TIMES = 4
#: One draw per stored innovation value. A noise-free mean path is the same
#: for every draw.
INNOVATIONS = np.array([0.0, 1.0, -2.5])


def _frame(columns: dict[str, Any], units: tuple[str, ...] = ("u",)) -> pd.DataFrame:
    frame = pd.DataFrame({
        "unit": np.repeat(units, N_TIMES),
        "time": np.tile(np.arange(N_TIMES), len(units)),
    })
    for name, value in columns.items():
        frame[name] = np.resize(value, len(frame))
    return frame


def _pin_posterior(model: pathmc.PathModel, **values: Any) -> None:
    """Zero every free RV of the fitted graph, then pin the named ones.

    Every ``carry_innovations_*`` RV holds ``INNOVATIONS[d]`` on draw ``d``,
    at every step and unit.
    """
    draws = len(INNOVATIONS)
    post = xr.Dataset(coords={"chain": [0], "draw": np.arange(draws)})
    for rv in model.pymc_model.free_RVs:
        shape = tuple(int(s) for s in rv.shape.eval())
        array = np.zeros((1, draws, *shape))
        if rv.name.startswith("carry_innovations_"):
            array[...] = INNOVATIONS.reshape(1, draws, *(1,) * len(shape))
        if rv.name in values:
            array[...] = values[rv.name]
        dims = ("chain", "draw", *(f"{rv.name}_dim_{i}" for i in range(len(shape))))
        post[rv.name] = (dims, array)
    coords = {name: list(c) for name, c in model.pymc_model.coords.items()}
    for name in values:
        if name.startswith("beta_"):
            dim = f"{name.removeprefix('beta_')}_predictors"
            post[name] = post[name].rename({f"{name}_dim_0": dim})
            post = post.assign_coords({dim: coords[dim]})
    model._idata = xr.DataTree.from_dict({"posterior": post})


def _per_draw(result: DoResult, var: str) -> np.ndarray:
    return result.dataset[var].transpose("chain", "draw", "time").values[0]


def _recurrence(start: np.ndarray, coef: float) -> np.ndarray:
    """Unit-mean path of ``x[t] = coef * x[t-1]`` from per-unit ``start``."""
    path, previous = [], np.asarray(start, dtype=float)
    for _ in range(N_TIMES):
        previous = coef * previous
        path.append(previous.mean())
    return np.array(path)


def test_mean_ignores_stored_carry_innovations() -> None:
    # The #649 repro: estimated beta_y = 0.5, sigma_y = 2, pre-sample y = 1.
    model = pathmc.model(
        "y ~ 0 + lag(y)",
        data=_frame({"y": 1.0}),
        panel=PANEL,
        initial={"y": 1.0},
    )
    _pin_posterior(model, beta_y=0.5, sigma_y=2.0)

    result = model.do(kind="mean", simulate_over="time")

    expected = _recurrence(np.array([1.0]), 0.5)  # 0.5, 0.25, 0.125, 0.0625
    draws = len(INNOVATIONS)
    np.testing.assert_allclose(_per_draw(result, "y"), np.tile(expected, (draws, 1)))


@pytest.mark.parametrize(
    ("case", "kwargs", "pinned", "start"),
    [
        ("student_t", {"families": {"y": "studentt"}}, {"nu_y": 5.0}, [1.0]),
        # Missing rows: the stored innovations there were imputations.
        ("sparse", {}, {}, [1.0]),
        ("two_units", {}, {}, [1.0, 4.0]),
    ],
)
def test_mean_ignores_stored_carry_innovations_across_outcomes(
    case: str, kwargs: dict[str, Any], pinned: dict[str, float], start: list[float]
) -> None:
    units = ("u", "v")[: len(start)]
    y = [1.0, np.nan, np.nan, 1.0] if case == "sparse" else 1.0
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)",
        data=_frame({"y": y}, units),
        panel=PANEL,
        initial={"y": dict(zip(units, start, strict=True))},
        **kwargs,
    )
    _pin_posterior(model, sigma_y=2.0, **pinned)

    result = model.do(kind="mean", simulate_over="time")

    expected = _recurrence(np.array(start), 0.5)
    draws = len(INNOVATIONS)
    np.testing.assert_allclose(_per_draw(result, "y"), np.tile(expected, (draws, 1)))


def test_mean_through_a_count_child_ignores_stored_carry_innovations() -> None:
    # A Poisson child keeps the model off the coherent Gaussian graph, and its
    # mean is exp(y): carried noise biases it instead of only widening it.
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)\nc ~ 0 + 1*y",
        data=_frame({"y": 1.0, "c": 1}),
        panel=PANEL,
        families={"c": "poisson"},
        initial={"y": 2.0},
    )
    _pin_posterior(model, sigma_y=2.0)

    result = model.do(kind="mean", simulate_over="time")

    y = _recurrence(np.array([2.0]), 0.5)  # 1, 0.5, 0.25, 0.125
    draws = len(INNOVATIONS)
    np.testing.assert_allclose(_per_draw(result, "y"), np.tile(y, (draws, 1)))
    np.testing.assert_allclose(
        _per_draw(result, "c"), np.tile(np.exp(y), (draws, 1)), rtol=1e-6
    )


def test_mean_with_a_held_parent_ignores_the_child_carry_innovations() -> None:
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)\nz ~ 0 + 1*y + 0.5*lag(z)",
        data=_frame({"y": 1.0, "z": 1.0}),
        panel=PANEL,
        initial={"y": 0.0, "z": 0.0},
    )
    _pin_posterior(model, sigma_y=2.0, sigma_z=2.0)
    held = np.array([2.0, 2.0, 0.0, 4.0])

    result = model.do(set={"y": held}, kind="mean", simulate_over="time")

    z, previous = [], 0.0
    for value in held:
        previous = value + 0.5 * previous
        z.append(previous)
    draws = len(INNOVATIONS)
    np.testing.assert_allclose(_per_draw(result, "y"), np.tile(held, (draws, 1)))
    np.testing.assert_allclose(_per_draw(result, "z"), np.tile(z, (draws, 1)))
