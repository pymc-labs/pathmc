# Copyright 2025 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Seed forwarding through the static-panel intervention adapter."""

import numpy as np
import pandas as pd
import xarray as xr

import pathmc


def test_static_panel_predictive_do_reuses_seed_without_mutating_posterior():
    frame = pd.DataFrame({
        "unit": ["A"] * 3 + ["B"] * 3,
        "time": [0, 1, 2] * 2,
        "x": [1.0, 2.0, 3.0] * 2,
        "y": [1.0, 2.0, 3.0] * 2,
    })
    fitted = pathmc.model(
        "y ~ 0 + 1*x", data=frame, panel={"unit": "unit", "time": "time"}
    )
    posterior = xr.Dataset(
        {"sigma_y": (("chain", "draw"), np.ones((1, 10)))},
        coords={"chain": [0], "draw": range(10)},
    )
    fitted._idata = xr.DataTree.from_dict({
        "posterior": posterior,
        "observed_data": xr.Dataset({"y": ("obs", frame.y.values)}),
    })
    original = fitted._idata.copy(deep=True)
    common = dict(kind="predictive", simulate_over="time", set={"x": [2.0, 3.0, 4.0]})
    first = fitted.do(random_seed=123, **common)
    second = fitted.do(random_seed=123, **common)
    third = fitted.do(random_seed=124, **common)
    xr.testing.assert_equal(first.dataset, second.dataset)
    assert not first.dataset.equals(third.dataset)
    xr.testing.assert_equal(fitted._idata, original)
