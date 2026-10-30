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

import numpy as np
import pandas as pd
import xarray as xr

import pathmc

PANEL = {"unit": "unit", "time": "time"}
N_TIMES = 4
#: One draw per stored innovation value. A noise-free mean path is the same
#: for every draw.
INNOVATIONS = np.array([0.0, 1.0, -2.5])


def _frame(columns: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"unit": "u", "time": np.arange(N_TIMES), **columns})


def _posterior(**arrays: tuple[tuple[str, ...], np.ndarray]) -> xr.DataTree:
    post = xr.Dataset(
        dict(arrays), coords={"chain": [0], "draw": np.arange(len(INNOVATIONS))}
    )
    return xr.DataTree.from_dict({"posterior": post})


def _stored_innovations() -> tuple[tuple[str, ...], np.ndarray]:
    values = np.broadcast_to(
        INNOVATIONS[None, :, None, None], (1, len(INNOVATIONS), N_TIMES, 1)
    )
    return ("chain", "draw", "t", "u"), values.copy()


def _per_draw(result, var: str) -> np.ndarray:
    return result.dataset[var].transpose("chain", "draw", "time").values[0]


def _recurrence(start: float, coef: float) -> np.ndarray:
    path, previous = [], start
    for _ in range(N_TIMES):
        previous = coef * previous
        path.append(previous)
    return np.array(path)


def test_mean_ignores_stored_carry_innovations() -> None:
    model = pathmc.model(
        "y ~ 0 + lag(y)",
        data=_frame({"y": 1.0}),
        panel=PANEL,
        initial={"y": 1.0},
    )
    draws = len(INNOVATIONS)
    model._idata = _posterior(
        beta_y=(("chain", "draw", "y_predictors"), np.full((1, draws, 1), 0.5)),
        sigma_y=(("chain", "draw"), np.full((1, draws), 2.0)),
        carry_innovations_y=_stored_innovations(),
    )
    model._idata["posterior"] = model._idata.posterior.dataset.assign_coords(
        y_predictors=list(model.pymc_model.coords["y_predictors"])
    )

    result = model.do(kind="mean", simulate_over="time")

    expected = _recurrence(1.0, 0.5)  # 0.5, 0.25, 0.125, 0.0625
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
    draws = len(INNOVATIONS)
    model._idata = _posterior(
        sigma_y=(("chain", "draw"), np.full((1, draws), 2.0)),
        carry_innovations_y=_stored_innovations(),
    )

    result = model.do(kind="mean", simulate_over="time")

    y = _recurrence(2.0, 0.5)  # 1, 0.5, 0.25, 0.125
    np.testing.assert_allclose(_per_draw(result, "y"), np.tile(y, (draws, 1)))
    np.testing.assert_allclose(
        _per_draw(result, "c"), np.tile(np.exp(y), (draws, 1)), rtol=1e-6
    )
