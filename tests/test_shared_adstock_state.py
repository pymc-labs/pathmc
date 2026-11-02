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
"""Shared scan adstock advances once per timestep (issue #566).

The oracle is a hand-written recurrence, not pathmc's kernel:

    stock[t] = x[t] + decay * stock[t - 1], stock before t=0 is 0.

A second equation that names the same transform must see that stock, not
a state already stepped by an earlier equation in the same timestep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc

pytestmark = pytest.mark.filterwarnings(
    "ignore:Intervention value.*outside the observed"
)

PANEL = {"unit": "unit", "time": "week"}
DECAY = 0.5
LAM = 0.4


def _adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Independent geometric recurrence. Initial stock is zero."""
    stock = 0.0
    out = np.empty(len(x), dtype=float)
    for i, value in enumerate(np.asarray(x, dtype=float)):
        stock = float(value) + decay * stock
        out[i] = stock
    return out


def _saturate(x: np.ndarray, lam: float) -> np.ndarray:
    """pathmc logistic saturation, written out so the test does not call it."""
    z = np.asarray(x, dtype=float)
    return (1.0 - np.exp(-lam * z)) / (1.0 + np.exp(-lam * z))


def _panel(rows: list[dict[str, float | str | int]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _series(frame: pd.DataFrame, unit: str, column: str) -> np.ndarray:
    sub = frame[frame["unit"] == unit].sort_values("week")
    return sub[column].to_numpy(dtype=float)


def _simulate(spec: str, data: pd.DataFrame, params: dict) -> pd.DataFrame:
    return pathmc.simulate(spec, data=data, params=params, panel=PANEL, random_seed=1)


class TestSharedAdstockSimulate:
    """simulate() oracles. sigma is zero so the scan mean is the recurrence."""

    def test_issue_reproduction_constant_input(self):
        data = _panel([
            {"unit": "A", "week": 1, "x": 1.0},
            {"unit": "A", "week": 2, "x": 1.0},
            {"unit": "A", "week": 3, "x": 1.0},
        ])
        params = {"beta_m": [1.0], "sigma_m": 0.0, "theta": DECAY}
        single = _simulate("m ~ 0 + adstock(x, decay=theta)", data, params)
        shared = _simulate(
            "m ~ 0 + adstock(x, decay=theta)\ny ~ 0 + adstock(x, decay=theta)",
            data,
            {**params, "beta_y": [1.0], "sigma_y": 0.0},
        )
        expected = np.array([1.0, 1.5, 1.75])
        np.testing.assert_allclose(_adstock(np.ones(3), DECAY), expected)
        np.testing.assert_allclose(_series(single, "A", "m"), expected, atol=1e-8)
        np.testing.assert_allclose(_series(shared, "A", "m"), expected, atol=1e-8)
        np.testing.assert_allclose(_series(shared, "A", "y"), expected, atol=1e-8)

    def test_adding_an_equation_does_not_change_m(self):
        x = np.array([1.0, 0.0, 2.0, 0.5])
        data = _panel([
            {"unit": "A", "week": t + 1, "x": float(value)} for t, value in enumerate(x)
        ])
        params = {"beta_m": [1.0], "sigma_m": 0.0, "theta": DECAY}
        single = _simulate("m ~ 0 + adstock(x, decay=theta)", data, params)
        shared = _simulate(
            "m ~ 0 + adstock(x, decay=theta)\ny ~ 0 + adstock(x, decay=theta)",
            data,
            {**params, "beta_y": [3.0], "sigma_y": 0.0},
        )
        expected = _adstock(x, DECAY)
        np.testing.assert_allclose(_series(single, "A", "m"), expected, atol=1e-8)
        np.testing.assert_allclose(_series(shared, "A", "m"), expected, atol=1e-8)
        np.testing.assert_allclose(_series(shared, "A", "y"), 3.0 * expected, atol=1e-8)

    def test_three_equations_and_reversed_order(self):
        x = np.array([2.0, 1.0, 0.0, 4.0])
        data = _panel([
            {"unit": "A", "week": t + 1, "x": float(value)} for t, value in enumerate(x)
        ])
        expected = _adstock(x, DECAY)
        specs = {
            "m-first": (
                "m ~ 0 + adstock(x, decay=theta)\n"
                "y ~ 0 + adstock(x, decay=theta)\n"
                "z ~ 0 + adstock(x, decay=theta)"
            ),
            "y-first": (
                "y ~ 0 + adstock(x, decay=theta)\n"
                "z ~ 0 + adstock(x, decay=theta)\n"
                "m ~ 0 + adstock(x, decay=theta)"
            ),
        }
        params = {
            "beta_m": [1.0],
            "beta_y": [2.0],
            "beta_z": [0.5],
            "sigma_m": 0.0,
            "sigma_y": 0.0,
            "sigma_z": 0.0,
            "theta": DECAY,
        }
        for spec in specs.values():
            out = _simulate(spec, data, params)
            np.testing.assert_allclose(_series(out, "A", "m"), expected, atol=1e-8)
            np.testing.assert_allclose(
                _series(out, "A", "y"), 2.0 * expected, atol=1e-8
            )
            np.testing.assert_allclose(
                _series(out, "A", "z"), 0.5 * expected, atol=1e-8
            )

    def test_units_do_not_share_state(self):
        series = {"A": np.array([1.0, 1.0, 0.0]), "B": np.array([0.0, 3.0, 1.0])}
        rows = []
        for unit, values in series.items():
            for week, value in enumerate(values, start=1):
                rows.append({"unit": unit, "week": week, "x": float(value)})
        out = _simulate(
            "m ~ 0 + adstock(x, decay=theta)\ny ~ 0 + adstock(x, decay=theta)",
            _panel(rows),
            {
                "beta_m": [1.0],
                "beta_y": [1.0],
                "sigma_m": 0.0,
                "sigma_y": 0.0,
                "theta": 0.25,
            },
        )
        for unit, values in series.items():
            expected = _adstock(values, 0.25)
            np.testing.assert_allclose(_series(out, unit, "m"), expected, atol=1e-8)
            np.testing.assert_allclose(_series(out, unit, "y"), expected, atol=1e-8)

    def test_shared_adstock_of_endogenous_input(self):
        x = np.array([1.0, 2.0, 0.0, 1.0])
        data = _panel([
            {"unit": "A", "week": t + 1, "x": float(value)} for t, value in enumerate(x)
        ])
        out = _simulate(
            "m ~ 0 + x\ny ~ 0 + adstock(m, decay=theta)\n"
            "z ~ 0 + adstock(m, decay=theta)",
            data,
            {
                "beta_m": [2.0],
                "beta_y": [1.0],
                "beta_z": [4.0],
                "sigma_m": 0.0,
                "sigma_y": 0.0,
                "sigma_z": 0.0,
                "theta": DECAY,
            },
        )
        expected_m = 2.0 * x
        expected_stock = _adstock(expected_m, DECAY)
        np.testing.assert_allclose(_series(out, "A", "m"), expected_m, atol=1e-8)
        np.testing.assert_allclose(_series(out, "A", "y"), expected_stock, atol=1e-8)
        np.testing.assert_allclose(
            _series(out, "A", "z"), 4.0 * expected_stock, atol=1e-8
        )

    def test_lag_carry_stays_independent_of_adstock(self):
        x = np.array([1.0, 0.0, 2.0, 1.0])
        data = _panel([
            {"unit": "A", "week": t + 1, "x": float(value)} for t, value in enumerate(x)
        ])
        out = _simulate(
            "m ~ 0 + adstock(x, decay=theta)\ny ~ 0 + adstock(x, decay=theta) + lag(y)",
            data,
            {
                "beta_m": [1.0],
                "beta_y": [1.0, 0.3],
                "sigma_m": 0.0,
                "sigma_y": 0.0,
                "theta": DECAY,
            },
        )
        stock = _adstock(x, DECAY)
        expected_y = np.empty_like(stock)
        prev = 0.0
        for i, value in enumerate(stock):
            expected_y[i] = value + 0.3 * prev
            prev = expected_y[i]
        np.testing.assert_allclose(_series(out, "A", "m"), stock, atol=1e-8)
        np.testing.assert_allclose(_series(out, "A", "y"), expected_y, atol=1e-8)

    def test_shared_transform_plus_plain_regressor(self):
        x = np.array([1.0, 0.0, 2.0])
        z = np.array([0.5, 1.0, -1.0])
        data = _panel([
            {"unit": "A", "week": t + 1, "x": float(xv), "z": float(zv)}
            for t, (xv, zv) in enumerate(zip(x, z, strict=True))
        ])
        out = _simulate(
            "m ~ 0 + adstock(x, decay=theta)\ny ~ 0 + adstock(x, decay=theta) + z",
            data,
            {
                "beta_m": [1.0],
                "beta_y": [2.0, 0.5],
                "sigma_m": 0.0,
                "sigma_y": 0.0,
                "theta": DECAY,
            },
        )
        stock = _adstock(x, DECAY)
        np.testing.assert_allclose(_series(out, "A", "m"), stock, atol=1e-8)
        np.testing.assert_allclose(
            _series(out, "A", "y"), 2.0 * stock + 0.5 * z, atol=1e-8
        )

    def test_shared_saturated_adstock(self):
        tv = np.array([1.0, 0.0, 3.0, 1.0])
        data = _panel([
            {"unit": "A", "week": t + 1, "tv": float(value)}
            for t, value in enumerate(tv)
        ])
        spec = (
            "search ~ 0 + logistic_saturation(adstock(tv, decay=theta), lam=lam)\n"
            "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta), lam=lam)"
        )
        out = _simulate(
            spec,
            data,
            {
                "beta_search": [1.5],
                "beta_sales": [2.0],
                "sigma_search": 0.0,
                "sigma_sales": 0.0,
                "theta": DECAY,
                "lam": LAM,
            },
        )
        saturated = _saturate(_adstock(tv, DECAY), LAM)
        np.testing.assert_allclose(
            _series(out, "A", "search"), 1.5 * saturated, atol=1e-8
        )
        np.testing.assert_allclose(
            _series(out, "A", "sales"), 2.0 * saturated, atol=1e-8
        )


def _pin(posterior: xr.Dataset, name: str, value: float, coord: dict | None = None):
    if coord is None:
        posterior[name] = posterior[name] * 0.0 + value
        return
    posterior[name].loc[coord] = value


@pytest.fixture(scope="module")
def saturated_shared_model(mock_pymc_sample_module):
    """Pinned posterior so do() is the structural mean, not an MCMC estimate."""
    tv = np.array([1.0, 0.0, 3.0, 1.0])
    data = _panel([
        {"unit": "A", "week": t + 1, "tv": float(value), "search": 0.0, "sales": 0.0}
        for t, value in enumerate(tv)
    ])
    model = pathmc.model(
        "search ~ 0 + logistic_saturation(adstock(tv, decay=theta), lam=lam)\n"
        "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta), lam=lam)",
        data=data,
        panel=PANEL,
    )
    model.fit(draws=1, tune=1, chains=1, random_seed=0)
    posterior = model._idata["posterior"].dataset.copy(deep=True)
    assert list(posterior["beta_search"].coords["search_predictors"].values) == ["tv"]
    assert list(posterior["beta_sales"].coords["sales_predictors"].values) == ["tv"]
    _pin(posterior, "beta_search", 1.5, {"search_predictors": "tv"})
    _pin(posterior, "beta_sales", 2.0, {"sales_predictors": "tv"})
    _pin(posterior, "theta", DECAY)
    _pin(posterior, "lam", LAM)
    _pin(posterior, "sigma_search", 0.0)
    _pin(posterior, "sigma_sales", 0.0)
    model._idata["posterior"] = xr.DataTree(posterior)
    return model, tv


class TestSharedAdstockTimeForwardDo:
    """Native do(simulate_over='time', kind='mean') uses the same recurrence."""

    def test_natural_and_intervened_match_numpy(self, saturated_shared_model):
        model, tv = saturated_shared_model
        intervened = np.array([0.0, 2.0, 0.0, 4.0])
        natural = model.do(set=None, simulate_over="time", kind="mean")
        forced = model.do(set={"tv": intervened}, simulate_over="time", kind="mean")
        natural_sat = _saturate(_adstock(tv, DECAY), LAM)
        forced_sat = _saturate(_adstock(intervened, DECAY), LAM)
        np.testing.assert_allclose(
            natural.by_time("search").mean(axis=1), 1.5 * natural_sat, atol=1e-8
        )
        np.testing.assert_allclose(
            natural.by_time("sales").mean(axis=1), 2.0 * natural_sat, atol=1e-8
        )
        np.testing.assert_allclose(
            forced.by_time("search").mean(axis=1), 1.5 * forced_sat, atol=1e-8
        )
        np.testing.assert_allclose(
            forced.by_time("sales").mean(axis=1), 2.0 * forced_sat, atol=1e-8
        )

    def test_contrast_matches_independent_propagation(self, saturated_shared_model):
        model, _tv = saturated_shared_model
        low = np.array([0.0, 1.0, 0.0, 0.0])
        high = np.array([2.0, 0.0, 1.0, 3.0])
        r_low = model.do(set={"tv": low}, simulate_over="time", kind="mean")
        r_high = model.do(set={"tv": high}, simulate_over="time", kind="mean")
        contrast = r_high - r_low
        delta = _saturate(_adstock(high, DECAY), LAM) - _saturate(
            _adstock(low, DECAY), LAM
        )
        np.testing.assert_allclose(
            contrast.by_time("search").mean(axis=1), 1.5 * delta, atol=1e-8
        )
        np.testing.assert_allclose(
            contrast.by_time("sales").mean(axis=1), 2.0 * delta, atol=1e-8
        )
