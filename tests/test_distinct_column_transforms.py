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
"""Distinct transforms of one column both enter the model (issue #592).

The oracle is a hand-written recurrence, not pathmc's kernel. A later call
on the same source column must not replace an earlier one, on either the
vectorized compiler or the panel scan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc

pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of `ignore_unspecified`"
)


def _geometric(x: np.ndarray, decay: float) -> np.ndarray:
    """Unbounded geometric adstock. Initial stock is zero."""
    out = np.empty(len(x), dtype=float)
    stock = 0.0
    for i, value in enumerate(np.asarray(x, dtype=float)):
        stock = float(value) + decay * stock
        out[i] = stock
    return out


def _logistic(x: np.ndarray, lam: float) -> np.ndarray:
    """Pointwise logistic saturation, written out so the test does not call it."""
    z = np.asarray(x, dtype=float)
    return (1.0 - np.exp(-lam * z)) / (1.0 + np.exp(-lam * z))


def _cross(spec: str, data: pd.DataFrame, params: dict) -> np.ndarray:
    out = pathmc.simulate(spec, data=data, params=params, random_seed=0)
    return out["Y"].to_numpy(dtype=float)


def _panel_frame(series: dict[str, np.ndarray], column: str = "X") -> pd.DataFrame:
    rows = []
    for unit, values in series.items():
        for t, value in enumerate(values):
            rows.append({"unit": unit, "time": t, column: float(value)})
    return pd.DataFrame(rows)


def _panel_series(frame: pd.DataFrame, unit: str, column: str = "Y") -> np.ndarray:
    sub = frame[frame["unit"] == unit].sort_values("time")
    return sub[column].to_numpy(dtype=float)


X = np.array([1.0, 0.0, 0.0, 0.0])
D1 = 0.0
D2 = 0.5
INDEPENDENT = _geometric(X, D1) + _geometric(X, D2)


class TestDistinctTransformsOfOneColumn:
    def test_cross_section_sums_both_adstocks(self):
        data = pd.DataFrame({"X": X})
        params = {"beta_Y": [1.0, 1.0], "sigma_Y": 0.0, "d1": D1, "d2": D2}
        got = _cross(
            "Y ~ 0 + adstock(X, decay=d1) + adstock(X, decay=d2)",
            data,
            params,
        )
        np.testing.assert_allclose(INDEPENDENT, [2.0, 0.5, 0.25, 0.125])
        np.testing.assert_allclose(got, INDEPENDENT, atol=1e-8)
        # Last-call-wins is 2 * geometric(d2). d1 must change the result.
        np.testing.assert_allclose(2.0 * _geometric(X, D2), [2.0, 1.0, 0.5, 0.25])
        assert not np.allclose(got, 2.0 * _geometric(X, D2))

    def test_reversed_parameter_order_still_sums(self):
        data = pd.DataFrame({"X": X})
        got = _cross(
            "Y ~ 0 + adstock(X, decay=d2) + adstock(X, decay=d1)",
            data,
            {"beta_Y": [1.0, 1.0], "sigma_Y": 0.0, "d1": D1, "d2": D2},
        )
        np.testing.assert_allclose(got, INDEPENDENT, atol=1e-8)

    def test_two_equations_keep_separate_decays(self):
        data = pd.DataFrame({"X": X})
        out = pathmc.simulate(
            "M ~ 0 + adstock(X, decay=d1)\nY ~ 0 + adstock(X, decay=d2)",
            data=data,
            params={
                "beta_M": [1.0],
                "beta_Y": [1.0],
                "sigma_M": 0.0,
                "sigma_Y": 0.0,
                "d1": D1,
                "d2": D2,
            },
            random_seed=0,
        )
        np.testing.assert_allclose(out["M"].to_numpy(dtype=float), _geometric(X, D1))
        np.testing.assert_allclose(
            out["Y"].to_numpy(dtype=float), _geometric(X, D2), atol=1e-8
        )

    def test_vectorized_saturations_of_one_column_both_enter(self):
        x = np.array([1.0, 2.0, 0.0, 0.5])
        lam_a, lam_b = 0.2, 1.5
        expected = _logistic(x, lam_a) + _logistic(x, lam_b)
        got = _cross(
            "Y ~ 0 + logistic_saturation(X, lam=lam_a) + logistic_saturation(X, lam=lam_b)",
            pd.DataFrame({"X": x}),
            {"beta_Y": [1.0, 1.0], "sigma_Y": 0.0, "lam_a": lam_a, "lam_b": lam_b},
        )
        np.testing.assert_allclose(got, expected, atol=1e-8)
        assert not np.allclose(got, 2.0 * _logistic(x, lam_b))

    def test_panel_scan_sums_both_adstocks(self):
        series = {
            "A": X,
            "B": np.array([0.0, 1.0, 0.0, 0.0]),
        }
        out = pathmc.simulate(
            "Y ~ 0 + adstock(X, decay=d1) + adstock(X, decay=d2)",
            data=_panel_frame(series),
            params={"beta_Y": [1.0, 1.0], "sigma_Y": 0.0, "d1": D1, "d2": D2},
            panel={"unit": "unit", "time": "time"},
            random_seed=0,
        )
        for unit, values in series.items():
            expected = _geometric(values, D1) + _geometric(values, D2)
            np.testing.assert_allclose(_panel_series(out, unit), expected, atol=1e-8)
