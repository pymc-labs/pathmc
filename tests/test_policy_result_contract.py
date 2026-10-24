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

import operator
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
import pathmc.policy_exec as execution
from pathmc.simulate import DoResult
from pathmc.policy import Literal


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
    innovations = execution.policy_innovations

    class AllocationProbe:
        def __getattr__(self, name):
            return getattr(np, name)

        def empty(self, shape, *args, **kwargs):
            if isinstance(shape, tuple):
                allocations.append(shape)
            return np.empty(shape, *args, **kwargs)

    def record_noise(seed, states, chains, draws, n_sim, n_unit, n_time, **kwargs):
        generated.append((tuple(chains), tuple(draws), n_sim))
        return innovations(seed, states, chains, draws, n_sim, n_unit, n_time, **kwargs)

    monkeypatch.setattr(execution, "np", AllocationProbe())
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


def test_schedule_name_is_excluded_from_compiled_exogenous(fitted):
    from pathmc.compile import compile_to_pymc
    from pathmc.policy import plan_policy

    policy = pathmc.TreatmentPolicy("dose", "water", pathmc.TimeSchedule([0.0, 1.0]))
    names = set(fitted._graph_info.endogenous) | set(fitted._graph_info.exogenous)
    scan = fitted._gen_model._pathmc_panel_scan
    plan = plan_policy(
        policy,
        fitted._graph_info,
        names=names,
        time_values=tuple(scan.time_values),
    )
    compiled = compile_to_pymc(
        fitted._spec,
        fitted._data,
        fitted._design_matrices,
        families=fitted._families,
        panel_info=fitted._panel_info,
        pooling=fitted._pooling,
        latent=fitted._latent,
        graph_info=fitted._graph_info,
        priors=fitted._priors,
        generative=True,
        initial={"growth": np.zeros(scan.n_units)},
        policy_plan=plan,
    )
    exogenous = compiled._pathmc_panel_scan.exogenous
    assert "policy_schedule" not in exogenous
    assert "water" in exogenous


def test_previous_reference_outside_lag_bases_is_required(fitted):
    policy = pathmc.TreatmentPolicy(
        "rain-gate",
        "water",
        pathmc.where(pathmc.previous("rain") > 0, 1, 0),
    )
    with pytest.raises(ValueError, match=r"initial= must cover exactly.*rain"):
        fitted.do(
            policy=policy,
            horizon=2,
            simulate_over="time",
            n_simulations=2,
            random_seed=7,
        )


def test_reduced_policy_by_time_does_not_mention_simulate_over(fitted):
    result = _run(fitted).aggregate_time("growth")
    with pytest.raises(ValueError, match="aggregate_time") as exc:
        result.by_time("growth")
    assert "simulate_over" not in str(exc.value)


@pytest.mark.parametrize("compare,expected", [(operator.eq, 1.0), (operator.ne, 0.0)])
@pytest.mark.parametrize(
    "expression,boundary",
    [
        (Literal(0.0), 0),
        (pathmc.current("rain"), 0),
        (pathmc.current("rain") + 1, 1),
        (pathmc.current("rain") > 0, 0),
        ((pathmc.current("rain") > 0) | (pathmc.current("rain") < 0), 0),
        (~(pathmc.current("rain") > 0), 1),
        (pathmc.where(pathmc.current("rain") >= 0, 1, 0), 1),
    ],
)
def test_equality_operators_execute_for_every_expression_node(
    fitted, expression, boundary, compare, expected
):
    condition = compare(expression, boundary)
    with pytest.raises(TypeError, match="truth"):
        bool(condition)
    result = fitted.do(
        policy=pathmc.TreatmentPolicy(
            "equality", "water", pathmc.where(condition, 1, 0)
        ),
        horizon=2,
        simulate_over="time",
        n_simulations=2,
        random_seed=7,
    )
    np.testing.assert_allclose(result.by_time("water"), expected)


def test_contrasts_reject_relabelled_posteriors_even_with_identical_parameters(fitted):
    original = _run(fitted)
    posterior = (
        fitted
        ._idata["posterior"]
        .to_dataset()
        .assign_coords(chain=[20, 40], draw=[50, 80, 130])
    )
    fitted._idata = xr.DataTree.from_dict({"posterior": posterior})
    relabelled = _run(fitted)
    with pytest.raises(ValueError, match="posterior_id"):
        _ = original - relabelled


@pytest.mark.parametrize("different", ["fixed coefficient", "scaling"])
def test_contrasts_reject_different_fixed_structural_context(fitted, different):
    original = _run(fitted)
    coefficient = 2 if different == "fixed coefficient" else 1
    scaling = (
        pathmc.Scaling(target={"method": "fixed", "value": 2.0})
        if different == "scaling"
        else None
    )
    other = pathmc.model(
        "stress ~ 0 + 1*rain\n"
        f"growth ~ 0 + 0.2*lag(growth) + {coefficient}*water + 0.3*stress",
        data=fitted._data.to_native(),
        panel={"unit": "unit", "time": "time"},
        initial={"growth": 0.0},
        scaling=scaling,
    )
    other._idata = fitted._idata
    with pytest.raises(ValueError, match="posterior_id"):
        _ = original - _run(other)


@pytest.mark.parametrize("layer", ["population", "simulation_draws", "_trajectories"])
def test_contrasts_require_exact_result_coordinates(fitted, layer):
    original = _run(fitted, keep=True)
    misaligned = replace(
        original, **{layer: getattr(original, layer).assign_coords(chain=[20, 40])}
    )
    with pytest.raises(ValueError, match="coordinates.*same posterior"):
        _ = original - misaligned
