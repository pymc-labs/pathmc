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
"""Policy memory, posterior-summary, and baseline contracts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
import pathmc.policy_exec as execution
from pathmc.simulate import DoResult


@pytest.fixture
def fitted():
    frame = pd.DataFrame({
        "unit": ["A", "A", "B", "B"],
        "time": [0, 1, 0, 1],
        "growth": [0.0] * 4,
        "stress": [0.0] * 4,
        "rain": [0.0] * 4,
        "water": [0.0] * 4,
    })
    model = pathmc.model(
        "stress ~ 0 + 1*rain\ngrowth ~ 0 + 0.2*lag(growth) + 1*water + 0.3*stress",
        data=frame,
        panel={"unit": "unit", "time": "time"},
        initial={"growth": 0.0},
    )
    posterior = xr.Dataset(coords={"chain": [2, 4], "draw": [5, 8, 13]})
    for var in ("growth", "stress"):
        posterior[f"sigma_{var}"] = (("chain", "draw"), np.ones((2, 3)))
    model._idata = xr.DataTree.from_dict({"posterior": posterior})
    return model


def _run(fitted, *, keep=False, **kwargs):
    return execution.run_policy_evaluation(
        fitted,
        pathmc.TreatmentPolicy(
            "responsive", "water", pathmc.where(pathmc.current("stress") > 0, 1, 0)
        ),
        set_values=None,
        initial=None,
        horizon=2,
        n_simulations=5,
        keep_trajectories=keep,
        random_seed=7,
        kind="mean",
        simulate_over="time",
        **kwargs,
    )


def test_default_execution_generates_bounded_noise_and_discards_unit_paths(
    fitted, monkeypatch
):
    allocations = []
    generated = []
    empty = np.empty
    innovations = execution.policy_innovations

    def record_empty(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(shape)
        return empty(shape, *args, **kwargs)

    def record_noise(seed, states, chains, draws, n_sim, n_unit, n_time, **kwargs):
        generated.append((tuple(chains), tuple(draws), n_sim))
        return innovations(seed, states, chains, draws, n_sim, n_unit, n_time, **kwargs)

    monkeypatch.setattr(execution.np, "empty", record_empty)
    monkeypatch.setattr(execution, "policy_innovations", record_noise)
    bounded = _run(fitted, batch_draws=1, batch_sims=2)
    assert generated
    assert all(
        len(chains) == len(draws) == 1 and sims <= 2
        for chains, draws, sims in generated
    )
    assert (2, 3, 5, 2, 2) not in allocations
    with pytest.raises(ValueError, match="keep_trajectories"):
        _ = bounded.trajectories

    retained = _run(fitted, keep=True, batch_draws=3, batch_sims=5)
    xr.testing.assert_equal(bounded.dataset, retained.dataset)
    xr.testing.assert_equal(bounded.simulation_draws, retained.simulation_draws)
    for var in bounded.dataset:
        direct = (
            retained
            .trajectories[var]
            .mean("unit")
            .transpose("time", "chain", "draw", "simulation")
        )
        xr.testing.assert_equal(bounded.simulation_draws[var], direct)
        xr.testing.assert_equal(
            bounded.simulation_mcse(var), direct.std("simulation", ddof=1) / np.sqrt(5)
        )


def test_policy_hdi_and_plot_follow_shared_posterior_summary_conventions(fitted):
    result = _run(fitted)
    total = result.aggregate_time("growth")
    expected = DoResult(ds=total.dataset).hdi("growth", prob=0.8)
    np.testing.assert_allclose(total.hdi("growth", prob=0.8), expected)
    figure, ax = plt.subplots()
    try:
        returned = result.plot("growth", ax=ax, prob=0.8)
        assert returned is figure
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1])
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), result.by_time("growth").mean(axis=1)
        )
        assert len(ax.collections) == 1
    finally:
        plt.close(figure)


def test_policy_initial_rejects_fitted_variables_that_are_never_carried(fitted):
    with pytest.raises(ValueError, match="rain.*unused.*Remove"):
        fitted.do(
            policy=pathmc.TreatmentPolicy("constant", "water", 0),
            initial={"growth": 0.0, "rain": 100.0},
            horizon=2,
            simulate_over="time",
            n_simulations=2,
            random_seed=7,
        )
