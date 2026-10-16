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
"""History-dependent treatment policies for panel do() (#529)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc.policy import TimeSchedule, TreatmentPolicy, current, previous, where
from pathmc.policy_exec import run_policy_evaluation
from pathmc.scaling import Scaling

PANEL = {"unit": "unit", "time": "time"}
TIMES = (1, 2, 3, 4)
SPEC = (
    "growth ~ 0 + 0.4*lag(growth) + 1*water\n"
    "stress ~ 0 + 0.5*lag(stress) + 0.5*lag(water)"
)
INITIAL = {"growth": 0.0, "water": 0.0, "stress": 0.6}


def _frame(times=TIMES, columns=("growth", "water", "stress")):
    rows = []
    for unit in ("A", "B"):
        for time in times:
            row = {"unit": unit, "time": time}
            row.update({name: 0.2 for name in columns})
            rows.append(row)
    return pd.DataFrame(rows)


def _pin(model, sigma=0.0, n_draw=1, sigmas=None):
    info = model._gen_model._pathmc_panel_scan
    post = xr.Dataset(coords={"chain": [0], "draw": list(range(n_draw))})
    for var in info.gaussian_states:
        if sigmas is not None and var in sigmas:
            values = np.asarray(sigmas[var], dtype=float)
        else:
            values = np.full(n_draw, sigma)
        post[f"sigma_{var}"] = (("chain", "draw"), values[None, :])
        post[f"carry_innovations_{var}"] = (
            ("chain", "draw", f"{var}_t", f"{var}_u"),
            np.zeros((1, n_draw, info.n_times, info.n_units)),
        )
    model._idata = xr.DataTree.from_dict({"posterior": post})
    return model


def _model(spec=SPEC, initial=INITIAL, sigma=0.0, n_draw=1, sigmas=None, **kwargs):
    fitted = pathmc.model(
        spec,
        data=_frame(columns=tuple(initial) if initial else ("growth", "water")),
        panel=PANEL,
        initial=initial,
        **kwargs,
    )
    return _pin(fitted, sigma=sigma, n_draw=n_draw, sigmas=sigmas)


def _roll(action_at, g0=0.0, s0=0.6, w0=0.0):
    """Independent NumPy recursion. ``action_at`` sees the previous state."""
    growth, water, stress = [], [], []
    g, s, w = g0, s0, w0
    for time in TIMES:
        action = float(action_at(time, g, s, w))
        g_now = 0.4 * g + action
        s_now = 0.5 * s + 0.5 * w
        growth.append(g_now)
        water.append(action)
        stress.append(s_now)
        g, s, w = g_now, s_now, action
    return {
        "growth": np.array(growth),
        "water": np.array(water),
        "stress": np.array(stress),
    }


def _path(result, var):
    return np.asarray(result.by_time(var)[:, 0], dtype=float)


def _assert_path(result, expected):
    for var, values in expected.items():
        np.testing.assert_allclose(_path(result, var), values, atol=1e-8)


def _policy(name, rule):
    return TreatmentPolicy(name, "water", rule)


def _do(model, policy, **kwargs):
    args = {
        "simulate_over": "time",
        "n_simulations": 2,
        "random_seed": 1,
        "initial": INITIAL,
        "horizon": len(TIMES),
    }
    args.update(kwargs)
    return model.do(policy=policy, **args)


def test_fixed_schedule_matches_numpy_and_static_do():
    fitted = _model()
    schedule = (1.0, 0.0, 0.0, 1.0)
    policy = _policy("every_third", TimeSchedule(schedule))
    fixed = _do(fitted, policy)
    expected = _roll(lambda time, g, s, w: schedule[TIMES.index(time)])
    _assert_path(fixed, expected)

    static = fitted.do(
        set={"water": np.array(schedule)},
        simulate_over="time",
        kind="mean",
        initial=INITIAL,
    )
    np.testing.assert_allclose(fixed.by_time("growth"), static.by_time("growth"))
    np.testing.assert_allclose(fixed.by_time("water"), np.array(schedule)[:, None])


def test_threshold_policies_match_numpy_and_contrast_is_not_a_winner():
    fitted = _model()
    open_rule = where(previous("stress") > 0.0, 1.0, 0.0)
    tight_rule = where(previous("stress") > 0.5, 1.0, 0.0)
    opened = _do(fitted, _policy("thresh_0", open_rule))
    tight = _do(fitted, _policy("thresh_0_5", tight_rule))
    open_expected = _roll(lambda time, g, s, w: 1.0 if s > 0.0 else 0.0)
    tight_expected = _roll(lambda time, g, s, w: 1.0 if s > 0.5 else 0.0)
    _assert_path(opened, open_expected)
    _assert_path(tight, tight_expected)

    contrast = opened - tight
    np.testing.assert_allclose(
        _path(contrast, "growth"),
        open_expected["growth"] - tight_expected["growth"],
    )
    np.testing.assert_allclose(
        _path(contrast, "water"),
        open_expected["water"] - tight_expected["water"],
    )
    assert not np.allclose(opened.by_time("water"), tight.by_time("water"))
    # The contrast is the difference. Neither path is declared better.
    assert opened.mean("growth") != pytest.approx(tight.mean("growth"))


def test_growth_and_water_are_reported_separately():
    fitted = _model()
    rule = where(previous("stress") > 0.5, 1.0, 0.0)
    result = _do(fitted, _policy("thresh_0_5", rule))
    expected = _roll(lambda time, g, s, w: 1.0 if s > 0.5 else 0.0)
    growth = result.aggregate_time("growth", reduction="sum")
    water = result.aggregate_time("water", reduction="sum")
    assert growth.mean("growth") == pytest.approx(expected["growth"].sum())
    assert water.mean("water") == pytest.approx(expected["water"].sum())
    assert growth.mean("growth") != pytest.approx(water.mean("water"))


def test_simulation_info_records_population_horizon_and_seed():
    fitted = _model()
    result = _do(fitted, _policy("fixed", 1.0), n_simulations=4, random_seed=11)
    info = result.simulation_info()
    assert info.policy_name == "fixed"
    assert info.target == "water"
    assert info.horizon == len(TIMES)
    assert info.time_values == TIMES
    assert info.unit_labels == ("A", "B")
    assert info.n_simulations == 4
    assert info.seed == 11
    mcse = result.simulation_mcse("water", by="time")
    assert mcse.dims == ("time", "chain", "draw")
    assert mcse.shape == (len(TIMES), 1, 1)
    with pytest.raises(ValueError, match="keep_trajectories"):
        _ = result.trajectories
    kept = _do(fitted, _policy("fixed", 1.0), keep_trajectories=True)
    assert kept.trajectories["water"].dims == (
        "chain",
        "draw",
        "simulation",
        "unit",
        "time",
    )


def test_same_seed_shares_noise_and_a_new_seed_does_not():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*lag(growth) + 1*water"
    fitted = _model(spec, {"stress": 0.0, "growth": 0.0}, sigma=1.0)
    watered = _do(
        fitted,
        _policy("on", 1.0),
        initial={"stress": 0.0, "growth": 0.0},
        n_simulations=4,
        random_seed=3,
    )
    dry = _do(
        fitted,
        _policy("off", 0.0),
        initial={"stress": 0.0, "growth": 0.0},
        n_simulations=4,
        random_seed=3,
    )
    other = _do(
        fitted,
        _policy("on", 1.0),
        initial={"stress": 0.0, "growth": 0.0},
        n_simulations=4,
        random_seed=9,
    )
    np.testing.assert_allclose(watered.by_time("stress"), dry.by_time("stress"))
    assert not np.allclose(watered.by_time("growth"), dry.by_time("growth"))
    assert not np.allclose(watered.by_time("stress"), other.by_time("stress"))


def test_noisy_threshold_matches_numpy_and_not_the_noise_free_path():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*lag(growth) + 1*water"
    initial = {"stress": 0.0, "growth": 0.0}
    fitted = _model(spec, initial, sigma=1.0)
    policy = _policy("thresh", where(current("stress") > 0.0, 1.0, 0.0))
    shock = np.zeros((1, 1, 2, 2, 4))
    shock[0, 0, :, :, 0] = [[0.4, -0.2], [-0.3, 0.8]]
    shock[0, 0, :, :, 1] = [[-0.1, 0.5], [0.2, -0.7]]
    shock[0, 0, :, :, 2] = [[0.3, 0.3], [-0.4, 0.1]]
    shock[0, 0, :, :, 3] = [[0.2, -0.4], [0.1, 0.6]]
    innovations = {"stress": shock, "growth": np.zeros_like(shock)}
    held = run_policy_evaluation(
        fitted,
        policy,
        set_values=None,
        initial=initial,
        horizon=len(TIMES),
        n_simulations=2,
        keep_trajectories=True,
        random_seed=4,
        kind="mean",
        simulate_over="time",
        batch_draws=1,
        batch_sims=1,
        innovations=innovations,
    )
    prev = np.zeros((2, 2))
    expected = []
    for time in range(3):
        now = prev + shock[0, 0, :, :, time]
        expected.append((now > 0.0).astype(float))
        prev = now
    water0 = held.trajectories["water"].sel(chain=0, draw=0).isel(time=0)
    np.testing.assert_allclose(np.asarray(water0), expected[0])
    assert not np.allclose(expected[0], 0.0)

    quiet = _model(spec, initial, sigma=0.0)
    noise_free = _do(quiet, policy, initial=initial, n_simulations=2)
    assert noise_free.mean("water") == pytest.approx(0.0)
    assert abs(held.mean("water") - noise_free.mean("water")) > 0.05


def test_chunk_size_does_not_change_results():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*lag(growth) + 1*water"
    initial = {"stress": 0.0, "growth": 0.0}
    fitted = _model(spec, initial, sigma=1.0, n_draw=2)
    policy = _policy("thresh", where(current("stress") > 0.0, 1.0, 0.0))
    kwargs = {
        "set_values": None,
        "initial": initial,
        "horizon": len(TIMES),
        "n_simulations": 4,
        "keep_trajectories": False,
        "random_seed": 5,
        "kind": "mean",
        "simulate_over": "time",
    }
    fine = run_policy_evaluation(fitted, policy, batch_draws=1, batch_sims=1, **kwargs)
    coarse = run_policy_evaluation(
        fitted, policy, batch_draws=2, batch_sims=4, **kwargs
    )
    np.testing.assert_allclose(fine.by_time("stress"), coarse.by_time("stress"))
    np.testing.assert_allclose(fine.by_time("water"), coarse.by_time("water"))


def test_posterior_draws_are_not_collapsed():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*lag(growth) + 1*water"
    initial = {"stress": 0.0, "growth": 0.0}
    fitted = _model(
        spec,
        initial,
        n_draw=2,
        sigmas={"stress": [0.0, 1.0], "growth": [0.0, 0.0]},
    )
    result = _do(
        fitted,
        _policy("thresh", where(current("stress") > 0.0, 1.0, 0.0)),
        initial=initial,
        n_simulations=8,
        random_seed=2,
    )
    quiet = result.by_time("water")[:, 0]
    noisy = result.by_time("water")[:, 1]
    np.testing.assert_allclose(quiet, 0.0, atol=1e-8)
    assert abs(float(np.mean(noisy))) > 0.05
    assert result.mean("water") == pytest.approx(float(np.mean(noisy)) / 2)


def test_business_unit_schedule_is_scaled_only_inside_the_scan():
    spec = "growth ~ 0 + 0.4*lag(growth) + 1*water"
    fitted = pathmc.model(
        spec,
        data=_frame(columns=("growth", "water")),
        panel=PANEL,
        initial={"growth": 0.0},
        scaling=Scaling(channel={"method": "fixed", "value": 10.0}),
    )
    _pin(fitted)
    schedule = (10.0, 0.0, 0.0, 10.0)
    result = fitted.do(
        policy=_policy("scaled", TimeSchedule(schedule)),
        simulate_over="time",
        horizon=len(TIMES),
        n_simulations=2,
        random_seed=1,
        initial={"growth": 0.0},
    )
    # Model-unit water is the schedule divided by 10, so growth starts at 1.
    np.testing.assert_allclose(_path(result, "water"), schedule)
    np.testing.assert_allclose(_path(result, "growth")[0], 1.0)
    np.testing.assert_allclose(_path(result, "growth")[1], 0.4)


def test_python_truth_tests_are_rejected():
    with pytest.raises(TypeError, match="truth"):
        if current("stress") > 0.0:
            pass
    with pytest.raises(TypeError, match="truth"):
        bool(previous("stress") > 0.5 and current("growth") > 0.0)


def test_policy_rejections_name_the_fix():
    fitted = _model()
    policy = _policy("bad", where(current("stress") > 0.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="predictive"):
        _do(fitted, policy, kind="predictive")
    with pytest.raises(ValueError, match="simulate_over"):
        fitted.do(policy=policy, initial=INITIAL)
    with pytest.raises(ValueError, match="Do not omit it"):
        fitted.do(
            policy=policy,
            simulate_over="time",
            n_simulations=2,
            initial=INITIAL,
        )
    with pytest.raises(ValueError, match="horizon"):
        _do(fitted, policy, horizon=9)
    with pytest.raises(ValueError, match="n_simulations"):
        _do(fitted, policy, n_simulations=1)
    with pytest.raises(ValueError, match="self-read"):
        _do(fitted, _policy("self", current("water")))
    with pytest.raises(ValueError, match="cyclic"):
        _do(fitted, _policy("cycle", current("growth")))
    with pytest.raises(ValueError, match="conflicts with policy target"):
        _do(fitted, policy, set={"water": 1.0})
    with pytest.raises(ValueError, match="explicit pre-sample"):
        bare = pathmc.model(SPEC, data=_frame(), panel=PANEL)
        _pin(bare)
        bare.do(
            policy=policy,
            simulate_over="time",
            horizon=len(TIMES),
            n_simulations=2,
        )

    cross = pathmc.model("y ~ x", data=pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
    post = xr.Dataset(coords={"chain": [0], "draw": [0]})
    post["sigma_y"] = (("chain", "draw"), np.zeros((1, 1)))
    cross._idata = xr.DataTree.from_dict({"posterior": post})
    with pytest.raises(ValueError, match="scan-compiled"):
        cross.do(policy=policy, simulate_over="time")


def test_non_scan_policy_points_at_cross_sectional_do_not_a_new_lag():
    policy = _policy("thresh", where(current("stress") > 0.5, 1.0, 0.0))
    cross = pathmc.model("y ~ x", data=pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
    post = xr.Dataset(coords={"chain": [0], "draw": [0]})
    post["sigma_y"] = (("chain", "draw"), np.zeros((1, 1)))
    cross._idata = xr.DataTree.from_dict({"posterior": post})
    with pytest.raises(ValueError, match=r"do\(set=") as caught:
        cross.do(policy=policy, simulate_over="time")
    assert "Add a lag" not in str(caught.value)


def _mark_fitted(model):
    model._idata = xr.DataTree.from_dict({
        "posterior": xr.Dataset(coords={"chain": [0], "draw": [0]})
    })
    return model


def test_unsupported_structure_is_rejected():
    policy = _policy("fixed", 1.0)
    poisson = pathmc.model(
        "growth ~ 0 + lag(growth) + water",
        data=_frame(columns=("growth", "water")),
        panel=PANEL,
        families={"growth": "poisson"},
        initial={"growth": 0.0},
    )
    with pytest.raises(ValueError, match="non-Gaussian"):
        _mark_fitted(poisson).do(
            policy=policy, simulate_over="time", initial={"growth": 0.0}
        )

    spend = _frame(columns=("growth", "spend"))
    spend["spend"] = 1.0
    transformed = pathmc.model(
        "growth ~ 0 + lag(growth) + adstock(spend, decay=0.5)",
        data=spend,
        panel=PANEL,
        initial={"growth": 0.0},
    )
    with pytest.raises(ValueError, match="transform/adstock"):
        _mark_fitted(transformed).do(
            policy=_policy("spend", 1.0),
            simulate_over="time",
            initial={"growth": 0.0},
        )

    latent_rows = []
    for unit in ("A", "B"):
        for time in TIMES:
            latent_rows.append({
                "unit": unit,
                "time": time,
                "survey": 0.2,
                "water": 0.2,
            })
    latent = pathmc.model(
        "survey ~ 0 + 1*awareness\nawareness ~ lag(awareness) + water",
        data=pd.DataFrame(latent_rows),
        panel=PANEL,
        latent=["awareness"],
        families={"awareness": "latent_normal"},
    )
    with pytest.raises(ValueError, match="latent"):
        _mark_fitted(latent).do(
            policy=policy,
            simulate_over="time",
            initial={"awareness": 0.0},
        )


def _roll_supplied(innov_s, innov_g, *, sigma_s, sigma_g, g0, s0, rule):
    """Hand recursion. ``rule`` sees this step's stress, then the previous one."""
    n_sim, n_unit, n_time = innov_s.shape
    growth = np.empty((n_sim, n_unit, n_time))
    stress = np.empty((n_sim, n_unit, n_time))
    water = np.empty((n_sim, n_unit, n_time))
    g = np.full((n_sim, n_unit), g0)
    s = np.full((n_sim, n_unit), s0)
    w = np.zeros((n_sim, n_unit))
    for time in range(n_time):
        s_now = 0.5 * s + 0.5 * w + sigma_s * innov_s[:, :, time]
        action = rule(s_now, s)
        g_now = 0.4 * g + action + sigma_g * innov_g[:, :, time]
        growth[:, :, time] = g_now
        stress[:, :, time] = s_now
        water[:, :, time] = action
        g, s, w = g_now, s_now, action
    return growth, stress, water


def test_numpy_oracle_matches_every_replicate_and_ignores_future_rows():
    initial = {"growth": 0.1, "water": 0.0, "stress": 0.2}
    frame = _frame()
    frame.loc[frame["time"] > TIMES[0], ["growth", "stress"]] = 999.0
    fitted = pathmc.model(SPEC, data=frame, panel=PANEL, initial=initial)
    _pin(fitted, sigmas={"growth": [0.3], "stress": [0.8]})
    n_sim, n_unit, n_time = 3, 2, len(TIMES)
    rng = np.random.default_rng(19)
    innov_s = rng.normal(size=(1, 1, n_sim, n_unit, n_time))
    innov_g = rng.normal(size=(1, 1, n_sim, n_unit, n_time))
    policy = _policy("now", where(current("stress") > 0.5, 1.0, 0.0))
    held = run_policy_evaluation(
        fitted,
        policy,
        set_values=None,
        initial=initial,
        horizon=len(TIMES),
        n_simulations=n_sim,
        keep_trajectories=True,
        random_seed=19,
        kind="mean",
        simulate_over="time",
        batch_draws=1,
        batch_sims=1,
        innovations={"stress": innov_s, "growth": innov_g},
    )
    growth, stress, water = _roll_supplied(
        innov_s[0, 0],
        innov_g[0, 0],
        sigma_s=0.8,
        sigma_g=0.3,
        g0=0.1,
        s0=0.2,
        rule=lambda now, prev: (now > 0.5).astype(float),
    )
    traj = held.trajectories.sel(chain=0, draw=0).transpose(
        "simulation", "unit", "time"
    )
    np.testing.assert_allclose(traj["growth"], growth)
    np.testing.assert_allclose(traj["stress"], stress)
    np.testing.assert_allclose(traj["water"], water)
    # Same-step outcome uses the action just assigned. Next-step stress, and
    # therefore the next decision, uses that action through lag(water).
    assert not np.allclose(water[:, :, 1], (stress[:, :, 0] > 0.5).astype(float))
    substituted = _roll_supplied(
        np.zeros_like(innov_s[0, 0]),
        np.zeros_like(innov_g[0, 0]),
        sigma_s=0.8,
        sigma_g=0.3,
        g0=0.1,
        s0=0.2,
        rule=lambda now, prev: (now > 0.5).astype(float),
    )
    assert not np.allclose(water, substituted[2])


def test_analytic_threshold_rejects_mean_substitution():
    from scipy.stats import norm

    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*water"
    initial = {"stress": 0.0}
    times = (1, 2)
    rows = [
        {"unit": "A", "time": time, "stress": 0.0, "growth": 0.0, "water": 0.0}
        for time in times
    ]
    fitted = pathmc.model(spec, data=pd.DataFrame(rows), panel=PANEL, initial=initial)
    _pin(fitted, sigmas={"stress": [1.0], "growth": [0.0]})
    n_sim = 800
    innov = np.random.default_rng(3).normal(size=(1, 1, n_sim, 1, len(times)))
    policy = _policy("thresh", where(current("stress") > 0.5, 1.0, 0.0))
    held = run_policy_evaluation(
        fitted,
        policy,
        set_values=None,
        initial=initial,
        horizon=len(times),
        n_simulations=n_sim,
        keep_trajectories=True,
        random_seed=3,
        kind="mean",
        simulate_over="time",
        innovations={"stress": innov, "growth": np.zeros_like(innov)},
    )
    rate = float(held.trajectories["water"].sel(chain=0, draw=0).isel(time=0).mean())
    expected = float(norm.sf(0.5))
    assert rate == pytest.approx(expected, abs=0.08)
    quiet = pathmc.model(spec, data=pd.DataFrame(rows), panel=PANEL, initial=initial)
    _pin(quiet, sigma=0.0)
    noise_free = quiet.do(
        policy=policy,
        simulate_over="time",
        horizon=len(times),
        n_simulations=2,
        random_seed=3,
        initial=initial,
    )
    assert noise_free.mean("water") == pytest.approx(0.0)
    assert abs(rate - noise_free.mean("water")) > 0.2


def test_threshold_equality_follows_the_comparison_operator():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*water"
    initial = {"stress": 0.5}
    rows = [
        {"unit": unit, "time": time, "stress": 0.5, "growth": 0.0, "water": 0.0}
        for unit in ("A", "B")
        for time in TIMES
    ]
    fitted = pathmc.model(spec, data=pd.DataFrame(rows), panel=PANEL, initial=initial)
    _pin(fitted, sigma=0.0)
    strict = fitted.do(
        policy=_policy("gt", where(current("stress") > 0.5, 1.0, 0.0)),
        simulate_over="time",
        horizon=len(TIMES),
        n_simulations=2,
        random_seed=1,
        initial=initial,
    )
    closed = fitted.do(
        policy=_policy("ge", where(current("stress") >= 0.5, 1.0, 0.0)),
        simulate_over="time",
        horizon=len(TIMES),
        n_simulations=2,
        random_seed=1,
        initial=initial,
    )
    assert strict.mean("water") == pytest.approx(0.0)
    assert closed.mean("water") == pytest.approx(1.0)


def test_results_follow_labels_and_contrasts_reject_context_mismatch():
    times = (10, 20, 30)
    rows = []
    for unit in ("B", "A"):
        for time in times:
            rows.append({
                "unit": unit,
                "time": time,
                "growth": 0.2,
                "water": 0.2,
                "stress": 0.2,
            })
    initial = {
        "growth": {"A": 1.0, "B": 0.0},
        "water": 0.0,
        "stress": {"A": 0.2, "B": 0.8},
    }
    fitted = pathmc.model(SPEC, data=pd.DataFrame(rows), panel=PANEL, initial=initial)
    _pin(fitted, sigma=0.0)
    opened = fitted.do(
        policy=_policy("open", where(previous("stress") > 0.0, 1.0, 0.0)),
        simulate_over="time",
        horizon=len(times),
        n_simulations=2,
        random_seed=1,
        keep_trajectories=True,
        initial=initial,
    )
    assert list(opened.trajectories.coords["unit"].values) == ["A", "B"]
    assert list(opened.time_index) == list(times)
    # A's larger initial growth must stay on label A, not on the first frame row.
    first = opened.trajectories["growth"].sel(chain=0, draw=0, simulation=0, time=10)
    assert float(first.sel(unit="A")) > float(first.sel(unit="B"))
    other_seed = fitted.do(
        policy=_policy("tight", where(previous("stress") > 0.5, 1.0, 0.0)),
        simulate_over="time",
        horizon=len(times),
        n_simulations=2,
        random_seed=2,
        initial=initial,
    )
    with pytest.raises(ValueError, match="seed"):
        _ = opened - other_seed
    other_count = fitted.do(
        policy=_policy("tight", where(previous("stress") > 0.5, 1.0, 0.0)),
        simulate_over="time",
        horizon=len(times),
        n_simulations=4,
        random_seed=1,
        initial=initial,
    )
    with pytest.raises(ValueError, match="n_simulations"):
        _ = opened - other_count


def test_repeat_calls_do_not_leak_and_paired_mcse_uses_simulation_differences():
    spec = "stress ~ 0 + 1*lag(stress)\ngrowth ~ 0 + 1*lag(growth) + 1*water"
    initial = {"stress": 0.0, "growth": 0.0}
    fitted = _model(spec, initial, sigma=1.0)
    opened = _policy("open", where(current("stress") > 0.0, 1.0, 0.0))
    tight = _policy("tight", where(current("stress") > 1.0, 1.0, 0.0))
    first = _do(
        fitted, opened, initial=initial, n_simulations=6, keep_trajectories=True
    )
    _do(fitted, tight, initial=initial, n_simulations=6, keep_trajectories=True)
    again = _do(
        fitted, opened, initial=initial, n_simulations=6, keep_trajectories=True
    )
    np.testing.assert_allclose(
        again.trajectories["stress"], first.trajectories["stress"]
    )
    np.testing.assert_allclose(again.trajectories["water"], first.trajectories["water"])
    other = _do(fitted, tight, initial=initial, n_simulations=6, keep_trajectories=True)
    contrast = first - other
    paired = first.simulation_draws["growth"] - other.simulation_draws["growth"]
    expected = paired.std(dim="simulation", ddof=1) / np.sqrt(6)
    np.testing.assert_allclose(contrast.simulation_mcse("growth"), expected)
    cumulative = contrast.aggregate_time("growth", reduction="sum")
    terminal = contrast.aggregate_time("growth", reduction="last")
    retained = (first.trajectories["growth"] - other.trajectories["growth"]).mean(
        "unit"
    )
    np.testing.assert_allclose(
        cumulative.dataset["growth"],
        retained.sum("time").mean("simulation"),
    )
    np.testing.assert_allclose(
        terminal.dataset["growth"],
        retained.isel(time=-1).mean("simulation"),
    )
    # Averaging the per-draw SE into one posterior number is not the contract.
    assert contrast.simulation_mcse("growth").dims == ("time", "chain", "draw")


@pytest.mark.slow
def test_mcmc_policy_value_is_recovered_from_a_real_fit():
    rng = np.random.default_rng(5)
    times = list(range(6))
    rows = []
    for unit in ("A", "B", "C", "D"):
        growth = 0.0
        for time in times:
            water = float(rng.uniform(0.0, 1.0))
            growth = 0.4 * growth + water + float(rng.normal(0.0, 0.05))
            rows.append({
                "unit": unit,
                "time": time,
                "growth": growth,
                "water": water,
            })
    data = pd.DataFrame(rows)
    fitted = pathmc.model(
        "growth ~ 0 + 0.4*lag(growth) + water",
        data=data,
        panel=PANEL,
        initial={"growth": 0.0},
    )
    fitted.fit(
        draws=40,
        tune=40,
        chains=1,
        progressbar=False,
        compute_log_likelihood=False,
        random_seed=5,
    )
    schedule = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
    result = fitted.do(
        policy=_policy("alternate", TimeSchedule(schedule)),
        simulate_over="time",
        horizon=len(times),
        n_simulations=2,
        random_seed=5,
        initial={"growth": 0.0},
    )
    expected = []
    growth = 0.0
    for action in schedule:
        growth = 0.4 * growth + action
        expected.append(growth)
    np.testing.assert_allclose(
        result.by_time("growth").mean(axis=1), expected, atol=0.15
    )
