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
"""Business-unit oracles for the scaling review on PR #546."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc


def _frame():
    return pd.DataFrame({
        "unit": ["B", "A", "B", "A"],
        "time": [1, 0, 0, 1],
        "growth": [0.0] * 4,
        "water": [0.0] * 4,
        "rain": [40.0, 5.0, 30.0, 10.0],
    })


def _zero_noise(model):
    post = xr.Dataset(coords={"chain": [0], "draw": [0]})
    for var in model._gen_model._pathmc_panel_scan.gaussian_states:
        post[f"sigma_{var}"] = (("chain", "draw"), [[0.0]])
    model._idata = xr.DataTree.from_dict({"posterior": post})
    return model


@pytest.mark.parametrize("endogenous_rain", [False, True])
@pytest.mark.parametrize("rain", [10.0, np.array([10.0, 20.0])])
def test_disjoint_set_is_scaled_without_being_read_by_the_policy(endogenous_rain, rain):
    spec = "growth ~ 0 + 0*lag(growth) + 1*water + 1*rain"
    if endogenous_rain:
        spec += "\nrain ~ 0 + 1*water"
    fitted = _zero_noise(
        pathmc.model(
            spec,
            data=_frame(),
            panel={"unit": "unit", "time": "time"},
            initial={"growth": 0.0},
            scaling=pathmc.Scaling(
                channel={"method": "fixed", "value": 10.0},
                target={"method": "fixed", "value": 10.0} if endogenous_rain else None,
            ),
        )
    )
    result = fitted.do(
        policy=pathmc.TreatmentPolicy("dry", "water", pathmc.TimeSchedule((0, 0))),
        set={"rain": rain},
        horizon=2,
        simulate_over="time",
        n_simulations=2,
        keep_trajectories=True,
        random_seed=1,
    )
    # A fixed coefficient multiplies model-unit rain; the target divisor
    # converts growth back to business units only when targets are scaled.
    expected = np.broadcast_to(rain, (2,)) / 10 * (10 if endogenous_rain else 1)
    np.testing.assert_allclose(result.by_time("growth")[:, 0], expected)
    for unit in ("A", "B"):
        np.testing.assert_allclose(
            result.trajectories.growth.sel(unit=unit).values,
            np.broadcast_to(expected, (1, 1, 2, 2)),
        )
    if endogenous_rain:
        np.testing.assert_allclose(
            result.by_time("rain")[:, 0], np.broadcast_to(rain, (2,))
        )


@pytest.mark.parametrize("endogenous_water", [False, True])
def test_returned_states_use_their_own_label_aligned_business_scales(
    endogenous_water,
):
    spec = "growth ~ 0 + 0*lag(growth) + 1*water"
    if endogenous_water:
        spec += "\nwater ~ 0 + 1*rain"
    fitted = _zero_noise(
        pathmc.model(
            spec,
            data=_frame(),
            panel={"unit": "unit", "time": "time"},
            initial={"growth": 0.0},
            scaling=pathmc.Scaling(
                target={
                    "method": "fixed",
                    "value": {"A": 10.0, "B": 20.0},
                    "dims": ("unit",),
                },
                channel={"method": "fixed", "value": 10.0},
            ),
        )
    )
    result = fitted.do(
        policy=pathmc.TreatmentPolicy("wet", "water", pathmc.TimeSchedule((10, 10))),
        horizon=2,
        simulate_over="time",
        n_simulations=2,
        keep_trajectories=True,
        random_seed=1,
    )
    for unit, growth in (("A", 10.0), ("B", 10.0 if endogenous_water else 20.0)):
        np.testing.assert_allclose(result.trajectories.growth.sel(unit=unit), growth)
        np.testing.assert_allclose(result.trajectories.water.sel(unit=unit), 10.0)
    np.testing.assert_allclose(
        result.by_time("growth"), 10.0 if endogenous_water else 15.0
    )


def test_lag_only_exogenous_initial_and_query_override_use_channel_scales():
    fitted = _zero_noise(
        pathmc.model(
            "growth ~ 0 + 2*lag(rain)",
            data=_frame(),
            panel={"unit": "unit", "time": "time"},
            initial={"rain": {"B": 100.0, "A": 20.0}},
            scaling=pathmc.Scaling(
                channel={
                    "method": "fixed",
                    "value": {"A": 5.0, "B": 10.0},
                    "dims": ("unit",),
                }
            ),
        )
    )
    assert fitted.fitted_scaling.columns == ("rain",)
    baseline = fitted.initial_baseline.values["rain"]
    assert list(baseline.unit.values) == ["A", "B"]
    np.testing.assert_allclose(baseline, [4.0, 10.0])
    result = fitted.do(simulate_over="time", kind="mean")
    # First mean: 2 * mean([20/5, 100/10]); next: 2 * mean([5/5, 30/10]).
    np.testing.assert_allclose(result.by_time("growth")[:, 0], [14.0, 4.0])
    override = fitted.do(
        simulate_over="time", kind="mean", initial={"rain": {"B": 40, "A": 10}}
    )
    np.testing.assert_allclose(override.by_time("growth")[:, 0], [6.0, 4.0])
    np.testing.assert_allclose(fitted.initial_baseline.values["rain"], [4.0, 10.0])


def test_simulate_lag_only_exogenous_input_uses_the_same_channel_scales():
    result = pathmc.simulate(
        "growth ~ 0 + 2*lag(rain)",
        data=_frame().drop(columns=["growth", "water"]),
        panel={"unit": "unit", "time": "time"},
        params={"sigma_growth": 0.0},
        scaling=pathmc.Scaling(
            channel={
                "method": "fixed",
                "value": {"A": 5.0, "B": 10.0},
                "dims": ("unit",),
            }
        ),
        random_seed=1,
    ).sort_values(["unit", "time"])
    # The legacy baseline uses each unit's first observed rain value.
    np.testing.assert_allclose(result.growth, [2.0, 2.0, 6.0, 6.0])
    np.testing.assert_allclose(result.rain, [5.0, 10.0, 30.0, 40.0])
