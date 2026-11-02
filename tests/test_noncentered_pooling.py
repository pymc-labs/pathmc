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
"""Partial pooling samples unit effects outside the centered funnel."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc


def _panel(n_units: int = 8, n_times: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        for time in range(n_times):
            rows.append({
                "unit": f"u{unit}",
                "time": time,
                "X": float(rng.normal()),
                "Y": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_unit_intercept_is_offset_plus_location():
    data = _panel(n_units=4, n_times=5)
    model = pathmc.model(
        "Y ~ X",
        data=data,
        panel={"unit": "unit", "time": "time"},
        pooling="partial",
    )
    pm_model = model.pymc_model
    free = {rv.name for rv in pm_model.free_RVs}
    assert "alpha_Y" not in free
    assert "alpha_Y_offset" in free
    offset = np.array([1.0, -1.0, 0.5, 0.0])
    got = pm_model["alpha_Y"].eval({
        pm_model["alpha_Y_offset"]: offset,
        pm_model["mu_alpha_Y"]: np.array(2.0),
        pm_model["sigma_alpha_Y"]: np.array(3.0),
    })
    np.testing.assert_allclose(got, 2.0 + 3.0 * offset)


def test_simulate_uses_the_centered_unit_intercept():
    data = _panel(n_units=4, n_times=6, seed=1)
    data["X"] = 0.0
    alphas = np.array([10.0, 20.0, 30.0, 40.0])
    sim = pathmc.simulate(
        "Y ~ X",
        data=data,
        params={
            "beta_Y": [0.0],
            "sigma_Y": 1e-6,
            "alpha_Y": alphas,
            "mu_alpha_Y": 25.0,
            "sigma_alpha_Y": 5.0,
        },
        panel={"unit": "unit", "time": "time"},
        pooling="partial",
        random_seed=0,
    )
    means = sim.groupby("unit", observed=True)["Y"].mean().sort_index()
    np.testing.assert_allclose(means.to_numpy(), alphas, atol=1e-4)


def test_simulate_keeps_centered_intercept_when_scale_is_zero():
    data = _panel(n_units=4, n_times=6, seed=1)
    data["X"] = 0.0
    alphas = np.array([10.0, 20.0, 30.0, 40.0])
    sim = pathmc.simulate(
        "Y ~ X",
        data=data,
        params={
            "beta_Y": [0.0],
            "sigma_Y": 1e-6,
            "alpha_Y": alphas,
            "mu_alpha_Y": 0.0,
            "sigma_alpha_Y": 0.0,
        },
        panel={"unit": "unit", "time": "time"},
        pooling="partial",
        random_seed=0,
    )
    means = sim.groupby("unit", observed=True)["Y"].mean().sort_index()
    np.testing.assert_allclose(means.to_numpy(), alphas, atol=1e-4)


def test_random_slope_is_offset_plus_location():
    data = _panel(n_units=4, n_times=5)
    model = pathmc.model(
        "Y ~ X",
        data=data,
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": False, "slopes": ["X"]},
    )
    pm_model = model.pymc_model
    free = {rv.name for rv in pm_model.free_RVs}
    assert "slope_Y_X" not in free
    assert "slope_Y_X_offset" in free
    offset = np.array([0.0, 1.0, -2.0, 0.5])
    got = pm_model["slope_Y_X"].eval({
        pm_model["slope_Y_X_offset"]: offset,
        pm_model["mu_slope_Y_X"]: np.array(-1.0),
        pm_model["sigma_slope_Y_X"]: np.array(2.0),
    })
    np.testing.assert_allclose(got, -1.0 + 2.0 * offset)


@pytest.mark.slow
def test_partial_pooling_fit_does_not_diverge():
    """The centered unit intercept diverges on this panel; the offset must not."""
    data = _panel()
    model = pathmc.model(
        "Y ~ X",
        data=data,
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": True, "slopes": ["X"]},
    )
    with model.pymc_model:
        idata = pm.sample(
            draws=50,
            tune=100,
            chains=2,
            cores=1,
            random_seed=0,
            nuts_sampler="pymc",
            progressbar=False,
            compute_convergence_checks=False,
        )
    diverging = int(idata.sample_stats["diverging"].sum())
    assert diverging == 0
    assert "alpha_Y" in idata.posterior
    assert "slope_Y_X" in idata.posterior
