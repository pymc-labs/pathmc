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
"""Independent likelihood and trajectory oracles for scan covariance (#520)."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

import pathmc
from pathmc.simulate import _execute_gaussian_panel

PANEL = {"unit": "unit", "time": "time"}
SPEC = "a ~ 0 + 0.4*lag(a)\nb ~ 0 + 0.2*lag(b)\nchild ~ 0 + 1*a + 1*b\na ~~ b"
L = np.array([[1.2, 0], [0.5, 0.8]])


def frame(n=4, units=2):
    return pd.DataFrame({
        "unit": np.repeat([f"u{i}" for i in range(units)], n),
        "time": np.tile(np.arange(n), units),
        "x": np.arange(n * units, dtype=float) + 1,
        **{v: np.zeros(n * units) for v in ["a", "b", "c", "d", "z", "child"]},
    })


def model(spec=SPEC, n=4, units=2, **kwargs):
    return pathmc.model(spec, data=frame(n, units), panel=PANEL, **kwargs)


def posterior(m, params=None, draws=3):
    params = params or {"chol_a_b": L[np.tril_indices(2)], "sigma_child": 0.0}
    ds = xr.Dataset(coords={"chain": [7], "draw": np.arange(draws) + 11})
    for name, value in params.items():
        value = np.asarray(value)
        dims = ("chain", "draw", *(f"{name}_dim_{i}" for i in range(value.ndim)))
        ds[name] = (dims, np.broadcast_to(value, (1, draws, *value.shape)).copy())
    m._idata = xr.DataTree.from_dict({"posterior": ds})
    return ds


def noise(m, post):
    info = m._gen_model._pathmc_panel_scan
    coords = {
        "chain": post.chain,
        "draw": post.draw,
        "unit": info.unit_labels,
        "time": info.time_values,
    }
    rng = np.random.default_rng(18)
    return xr.Dataset({
        v: xr.DataArray(
            rng.normal(size=(1, post.sizes["draw"], info.n_units, info.n_times)),
            dims=("chain", "draw", "unit", "time"),
            coords=coords,
        )
        for v in info.gaussian_states
    })


def test_template_and_compilation():
    spec = (
        "online ~ adstock(x, decay=theta_tv) + lag(online)\n"
        "store ~ adstock(x, decay=theta_tv) + lag(store)\nonline ~~ store"
    )
    template = pathmc.simulate_params_template(spec, frame(), panel=PANEL)
    assert {k: v["shape"] for k, v in template.items()} == {
        "beta_online": (3,),
        "beta_store": (3,),
        "theta_tv": (),
        "chol_online_store": (3,),
    }
    m = model()
    assert {v.name for v in m.pymc_model.observed_RVs} == {"a_b_obs", "child"}
    assert {v.name for v in m.pymc_model.free_RVs} == {"chol_a_b", "sigma_child"}


def test_conditional_likelihood_and_gradients():
    df = frame().assign(
        a=[2.0, -1.0, 3.0, 4.0, 5.0, -2.0, 1.0, 6.0],
        b=[-2.0, 3.0, 1.0, 5.0, 2.0, 4.0, -3.0, 1.0],
        child=[3.0, 2.0, 5.0, 1.0, -1.0, 7.0, 2.0, 4.0],
    )
    spec = "a ~ x + lag(a)\nb ~ x + lag(b)\nchild ~ a + b\na ~~ b"
    m = pathmc.model(
        spec,
        df.sample(frac=1, random_state=8),
        panel=PANEL,
        initial={"a": {"u0": 1.0, "u1": -1.0}, "b": 2.0},
    ).pymc_model
    point = m.initial_point()
    point.update(
        beta_a=np.array([0.3, 0.4, 0.2]),
        beta_b=np.array([-0.2, 0.1, 0.5]),
        beta_child=np.array([0.1, 0.8, -0.3]),
        sigma_child_log__=np.array(np.log(0.7)),
        **{"chol_a_b_cholesky-cov-packed__": np.array([np.log(1.2), 0.5, np.log(0.8)])},
    )
    x = df.x.to_numpy().reshape(2, 4)
    a, b, child = [df[v].to_numpy().reshape(2, 4) for v in ["a", "b", "child"]]
    xa = np.stack([np.ones_like(x), x, np.column_stack([[1.0, -1.0], a[:, :-1]])], -1)
    xb = np.stack([np.ones_like(x), x, np.column_stack([[2.0, 2.0], b[:, :-1]])], -1)
    xc = np.stack([np.ones_like(x), a, b], -1)

    def oracle(p):
        packed = p["chol_a_b_cholesky-cov-packed__"]
        chol = np.array([[np.exp(packed[0]), 0], [packed[1], np.exp(packed[2])]])
        residual = np.stack([a - xa @ p["beta_a"], b - xb @ p["beta_b"]], -1)
        white = np.linalg.solve(chol, residual.reshape(-1, 2).T).T
        joint = -0.5 * np.sum(white**2) - 8 * (
            np.log(2 * np.pi) + np.log(np.diag(chol)).sum()
        )
        sd = np.exp(p["sigma_child_log__"])
        return joint + np.sum(
            -0.5 * np.log(2 * np.pi * sd**2)
            - 0.5 * ((child - xc @ p["beta_child"]) / sd) ** 2
        )

    logp = m.compile_logp(vars=m.observed_RVs)
    assert float(logp(point)) == pytest.approx(oracle(point), abs=1e-9)
    names = ["beta_a", "beta_b", "beta_child", "chol_a_b_cholesky-cov-packed__"]
    grad = pt.grad(
        m.logp(vars=m.observed_RVs),
        [next(v for v in m.value_vars if v.name == name) for name in names],
    )
    evaluate = pytensor.function(m.value_vars, grad, on_unused_input="ignore")
    actual = evaluate(*[point[v.name] for v in m.value_vars])
    for name, got in zip(names, actual, strict=True):
        expected = []
        for j in range(len(point[name])):
            plus = {k: v.copy() for k, v in point.items()}
            minus = {k: v.copy() for k, v in point.items()}
            plus[name][j] += 1e-5
            minus[name][j] -= 1e-5
            expected.append((oracle(plus) - oracle(minus)) / 2e-5)
        np.testing.assert_allclose(got, expected, atol=1e-7)


def test_fixed_white_innovations_drive_same_realized_states():
    m = model(initial={"a": {"u0": 2.0, "u1": -1.0}, "b": 3.0})
    post = posterior(m)
    eps = noise(m, post)
    info = m._gen_model._pathmc_panel_scan
    result = _execute_gaussian_panel(m._gen_model, post, info, innovations=eps)
    expected = np.empty((1, 3, 2, 4, 2))
    previous = np.array([[2.0, 3.0], [-1.0, 3.0]])
    for t in range(4):
        e = np.stack([eps.a.values[..., t], eps.b.values[..., t]], -1)
        expected[..., t, :] = previous * [0.4, 0.2] + e @ L.T
        previous = expected[..., t, :]
    np.testing.assert_allclose(result.a, expected[..., 0])
    np.testing.assert_allclose(result.b, expected[..., 1])
    np.testing.assert_allclose(result.child, expected.sum(-1))


@pytest.mark.parametrize("n,units", [(1, 1), (3, 1), (1, 2)])
@pytest.mark.parametrize(
    "terms",
    ["lag(a)", "adstock(x, decay=theta)", "lag(a) + adstock(x, decay=theta)", "x"],
)
def test_scan_triggers_and_small_layouts(n, units, terms):
    spec = f"a ~ {terms}\nb ~ x\nchild ~ a + lag(child)\na ~~ b"
    m = model(spec, n=n, units=units)
    assert {v.name for v in m.pymc_model.observed_RVs} == {"a_b_obs", "child"}
    assert not {"sigma_a", "sigma_b"} & {v.name for v in m.pymc_model.free_RVs}
    assert np.isfinite(m.pymc_model.compile_logp()(m.pymc_model.initial_point()))


@pytest.mark.parametrize("held", ["a", "b"])
def test_interventions_retain_marginal_peer_and_release(held):
    m = model(initial={"a": 1.0, "b": -2.0})
    post = posterior(m)
    eps = noise(m, post)
    info = m._gen_model._pathmc_panel_scan
    baseline = _execute_gaussian_panel(m._gen_model, post, info, innovations=eps)
    held_values = xr.DataArray(
        [[8.0, np.nan, 4.0, np.nan], [2.0, np.nan, -3.0, np.nan]],
        dims=("unit", "time"),
        coords={"unit": info.unit_labels, "time": info.time_values},
    )
    got = _execute_gaussian_panel(
        m._gen_model, post, info, innovations=eps, interventions={held: held_values}
    )
    peer = "b" if held == "a" else "a"
    xr.testing.assert_equal(got[peer], baseline[peer])
    np.testing.assert_allclose(got.child.values, got.a.values + got.b.values)
    phi = 0.4 if held == "a" else 0.2
    residual = 1.2 * eps.a if held == "a" else 0.5 * eps.a + 0.8 * eps.b
    expected = np.empty(got[held].shape)
    prev = 1.0 if held == "a" else -2.0
    for t in range(4):
        expected[..., t] = np.where(
            np.isnan(held_values[:, t]),
            phi * prev + residual.values[..., t],
            held_values[:, t],
        )
        prev = expected[..., t]
    np.testing.assert_allclose(got[held], expected)
    both = _execute_gaussian_panel(
        m._gen_model,
        post,
        info,
        innovations=eps,
        interventions={"a": held_values.fillna(2), "b": held_values.fillna(-1)},
    )
    np.testing.assert_allclose(both.child, both.a + both.b)


def test_fresh_noise_ar_cross_time_covariance_and_independence():
    m = model(units=2, initial={"a": 0.0, "b": 0.0})
    post = posterior(m, draws=7000)
    info = m._gen_model._pathmc_panel_scan
    got = _execute_gaussian_panel(m._gen_model, post, info, random_seed=123)
    again = _execute_gaussian_panel(m._gen_model, post, info, random_seed=123)
    xr.testing.assert_equal(got, again)
    y = np.stack([got.a.values[0], got.b.values[0]], -1)
    samples = y[:, 0].reshape(7000, -1)
    A = np.diag([0.4, 0.2])
    sigma = L @ L.T
    P = []
    previous = np.zeros((2, 2))
    for _ in range(4):
        previous = A @ previous @ A.T + sigma
        P.append(previous)
    expected = np.block([
        [
            np.linalg.matrix_power(A, t - s) @ P[s]
            if t >= s
            else (np.linalg.matrix_power(A, s - t) @ P[t]).T
            for s in range(4)
        ]
        for t in range(4)
    ])
    se = np.sqrt((expected**2 + np.diag(expected)[:, None] * np.diag(expected)) / 6999)
    assert np.all(np.abs(np.cov(samples.T) - expected) < 5 * se)
    residual = y.copy()
    residual[:, :, 1:] -= y[:, :, :-1] * [0.4, 0.2]
    all_noise = residual.reshape(7000, -1)
    target = np.kron(np.eye(8), sigma)
    se = np.sqrt((target**2 + np.diag(target)[:, None] * np.diag(target)) / 6999)
    assert np.all(np.abs(np.cov(all_noise.T) - target) < 5 * se)
    np.testing.assert_allclose(got.child.values, got.a.values + got.b.values)


@pytest.mark.parametrize("order", [False, True])
def test_three_member_full_block_interleaving_and_multiple_blocks(order):
    equations = [
        "a ~ 0 + 0.4*lag(a)",
        "child ~ 0 + 2*a",
        "b ~ 0 + 1*child",
        "c ~ 0 + 0.2*lag(c)",
        "d ~ 0 + 1*x",
        "z ~ 0 + 1*x",
    ]
    # b's parent child also reads a; use c/d/z for the three-member block instead.
    cov = ["c ~~ d", "d ~~ z", "a ~~ b"]
    # Cross-member ancestry is permitted only through an outside child; direct edges remain refused.
    if order:
        equations = equations[::-1]
        cov = ["z ~~ d", "d ~~ c", "c ~~ d", "b ~~ a"]
    spec = "\n".join(equations + cov)
    m = model(spec)
    factor = np.array([[0.7, 0, 0], [-0.4, 1.3, 0], [0.2, 0.6, 0.8]])
    params = {
        "chol_a_b": L[np.tril_indices(2)],
        "chol_c_d_z": factor[np.tril_indices(3)],
        "sigma_child": 0.0,
    }
    post = posterior(m, params)
    eps = noise(m, post)
    info = m._gen_model._pathmc_panel_scan
    got = _execute_gaussian_panel(m._gen_model, post, info, innovations=eps)
    eta = np.stack([eps[v].values for v in ["c", "d", "z"]], -1) @ factor.T
    prev_c = 0.0
    for t in range(4):
        np.testing.assert_allclose(got.c.values[..., t], 0.2 * prev_c + eta[..., t, 0])
        prev_c = got.c.values[..., t]
    x = frame().x.to_numpy().reshape(2, 4)
    np.testing.assert_allclose(got.d, x + eta[..., 1])
    np.testing.assert_allclose(got.z, x + eta[..., 2])
    np.testing.assert_allclose(got.child, 2 * got.a)
    np.testing.assert_allclose(got.b, got.child + 0.5 * eps.a + 0.8 * eps.b)
    assert {v.name for v in m.pymc_model.observed_RVs} == {
        "a_b_obs",
        "c_d_z_obs",
        "child",
    }


def test_named_graph_draws_use_colored_states():
    m = model(units=1)
    g = pm.do(
        m._gen_model, {"chol_a_b": L[np.tril_indices(2)], "sigma_child": np.array(0.0)}
    )
    a, b, child = pm.draw([g[v] for v in ["a", "b", "child"]], draws=20, random_seed=5)
    np.testing.assert_allclose(child, a + b)


@pytest.mark.parametrize(
    "bad",
    [
        1.0,
        np.ones((2, 2)),
        [1.0, 0.0],
        [1.0, 0.5, 0.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, np.nan],
        [1.0, np.inf, 1.0],
    ],
)
def test_invalid_fixed_factor(bad):
    with pytest.raises(ValueError, match="chol_a_b.*packed vector.*template"):
        pathmc.simulate(
            SPEC, frame(), panel=PANEL, params={"chol_a_b": bad, "sigma_child": 0.0}
        )


@pytest.mark.parametrize(
    "bad",
    [
        "latent",
        "mixed",
        "member",
        "missing_a",
        "missing_child",
        "infinite",
        "exogenous",
        "singleton",
        "crosslag",
        "transform",
    ],
)
def test_capability_guards(bad):
    df = frame()
    spec = SPEC
    kwargs = {}
    error = NotImplementedError
    match = "Gaussian|gaussian"
    if bad == "latent":
        spec += "\nhidden ~ x"
        kwargs["latent"] = ["hidden"]
    elif bad == "mixed":
        kwargs["families"] = {"child": "poisson"}
    elif bad == "member":
        kwargs["families"] = {"a": "studentt"}
        error = ValueError
    elif bad.startswith("missing") or bad == "infinite":
        df.loc[2, "child" if bad == "missing_child" else "a"] = (
            np.inf if bad == "infinite" else np.nan
        )
    elif bad == "exogenous":
        spec = spec.replace("child ~ 0 + 1*a + 1*b", "child ~ a + b + x") + "\na ~~ x"
        error, match = ValueError, "x.*equation"
    elif bad == "singleton":
        spec = "a ~ lag(a)\na ~~ a"
        error, match = ValueError, "one outcome|distinct"
    elif bad == "crosslag":
        spec = "a ~ lag(a)\nb ~ lag(a)\na ~~ b"
        match = "within|directed"
    else:
        spec = "a ~ lag(a)\nb ~ adstock(a, decay=theta)\na ~~ b"
        match = "within|directed"
    with pytest.raises(error, match=match):
        pathmc.model(spec, df, panel=PANEL, **kwargs)


@pytest.mark.parametrize("one_step", [False, True])
def test_prediction_fresh_noise_selection_extension_and_no_mutation(one_step):
    m = model(units=1)
    post = posterior(m, draws=1200)
    info = m._gen_model._pathmc_panel_scan
    for v in info.gaussian_states:
        post[f"carry_innovations_{v}"] = (
            ("chain", "draw", f"{v}_time", f"{v}_unit"),
            np.full((1, 1200, 4, 1), 999.0),
        )
        post[v] = (
            ("chain", "draw", f"{v}_time", f"{v}_unit"),
            np.full((1, 1200, 4, 1), 999.0),
        )
        post[f"mu_{v}"] = post[v]
    m._idata = xr.DataTree.from_dict({"posterior": post})
    before = m._idata.posterior.to_dataset().copy(deep=True)
    kwargs = dict(
        one_step_ahead=one_step,
        extend_inferencedata=False,
        random_seed=69,
        progressbar=False,
    )
    got = m.predict(**kwargs).posterior_predictive.dataset
    assert set(got.data_vars) == {"a", "b", "child"}
    selected = m.predict(var_names=["a"], **kwargs).posterior_predictive.dataset
    assert set(selected.data_vars) == {"a"}
    replay = m.predict(var_names=["a"], **kwargs).posterior_predictive.dataset
    xr.testing.assert_equal(selected.a, replay.a)
    xr.testing.assert_equal(m._idata.posterior.to_dataset(), before)
    assert np.max(np.abs(got.a)) < 10
    if not one_step:
        np.testing.assert_allclose(got.child.values, got.a.values + got.b.values)
    a = got.a.values[0, :, 0, 0]
    b = got.b.values[0, :, 0, 0]
    np.testing.assert_allclose(np.cov([a, b]), L @ L.T, atol=0.15)
    other = m.predict(**{**kwargs, "random_seed": 70}).posterior_predictive.dataset
    assert not np.array_equal(other.a, got.a)
    extended = m.predict(**{**kwargs, "extend_inferencedata": True})
    xr.testing.assert_equal(extended.posterior_predictive.dataset, got)
    assert int(m.pymc_model["_use_observed_carry"].get_value()) == 1


def test_prior_predictive_and_noise_free_mean_with_stored_noise():
    m = model(units=1, initial={"a": 2.0, "b": -3.0})
    post = posterior(m, draws=4)
    info = m._gen_model._pathmc_panel_scan
    for v in info.gaussian_states:
        post[f"carry_innovations_{v}"] = (
            ("chain", "draw", f"{v}_t", f"{v}_u"),
            np.full((1, 4, 4, 1), 300.0),
        )
    m._idata = xr.DataTree.from_dict({"posterior": post})
    got = m.do(kind="mean", simulate_over="time").dataset
    expected_a = 2 * 0.4 ** np.arange(1, 5)
    expected_b = -3 * 0.2 ** np.arange(1, 5)
    np.testing.assert_allclose(
        got.a.transpose("chain", "draw", "time"), np.broadcast_to(expected_a, (1, 4, 4))
    )
    np.testing.assert_allclose(got.child.values, got.a.values + got.b.values)
    np.testing.assert_allclose(
        got.b.transpose("chain", "draw", "time"), np.broadcast_to(expected_b, (1, 4, 4))
    )
    m._idata.posterior["chol_a_b"] *= 7
    xr.testing.assert_equal(got, m.do(kind="mean", simulate_over="time").dataset)
    prior = m.sample_prior_predictive(draws=10, random_seed=7).prior.dataset
    # Child has a sampled scale, so reconstruct its noise rather than assuming zero.
    np.testing.assert_allclose(
        prior.child.values - prior.mu_child.values,
        prior.sigma_child.values[..., None, None]
        * prior.carry_innovations_child.values,
    )
    np.testing.assert_allclose(prior.mu_child.values, prior.a.values + prior.b.values)


@pytest.mark.parametrize("nested", [False, True])
def test_shared_adstock_initial_and_ignored_simulation_outcomes(nested):
    transform = "adstock(x, decay=theta)"
    if nested:
        transform = f"logistic_saturation({transform}, lam=lam)"
    spec = f"a ~ 0 + {transform} + 0.4*lag(a)\nb ~ 0 + {transform}\na ~~ b"
    m = model(spec, initial={"a": {"u0": 2.0, "u1": -1.0}})
    params = {
        "beta_a": [1.0],
        "beta_b": [2.0],
        "theta": 0.6,
        "chol_a_b": L[np.tril_indices(2)],
    }
    if nested:
        params["lam"] = 0.2
    post = posterior(m, params)
    eps = noise(m, post)
    got = _execute_gaussian_panel(
        m._gen_model, post, m._gen_model._pathmc_panel_scan, innovations=eps
    )
    x = frame().x.to_numpy().reshape(2, 4)
    previous_a = np.array([2.0, -1.0])
    stock = np.zeros(2)
    for t in range(4):
        stock = x[:, t] + 0.6 * stock
        transformed = np.tanh(0.1 * stock) if nested else stock
        expected_a = transformed + 0.4 * previous_a + 1.2 * eps.a.values[..., t]
        np.testing.assert_allclose(got.a.values[..., t], expected_a)
        np.testing.assert_allclose(
            got.b.values[..., t],
            2 * transformed + 0.5 * eps.a.values[..., t] + 0.8 * eps.b.values[..., t],
        )
        previous_a = expected_a
    df = frame().sample(frac=1, random_state=3)
    clean = df.drop(columns=["a", "b"])
    expected = pathmc.simulate(spec, clean, panel=PANEL, params=params, random_seed=21)
    bogus = pathmc.simulate(
        spec, df.assign(a=np.nan, b=999.0), panel=PANEL, params=params, random_seed=21
    )
    pd.testing.assert_frame_equal(expected[bogus.columns], bogus)
    np.testing.assert_array_equal(bogus.unit, df.unit)
    np.testing.assert_array_equal(bogus.time, df.time)


def test_exceptions_restore_shared_values(monkeypatch):
    m = model()
    posterior(m)
    flag = m._gen_model["_use_observed_carry"]
    flag.set_value(np.array(1, dtype=flag.dtype))
    before = m._gen_model["_do_intervene_a"].get_value().copy()

    def fail(*args, **kwargs):
        raise RuntimeError("test exception")

    monkeypatch.setattr(pm, "sample_posterior_predictive", fail)
    with pytest.raises(RuntimeError, match="test exception"):
        m.do(set={"a": 2.0}, kind="predictive", simulate_over="time")
    assert int(flag.get_value()) == 1
    np.testing.assert_array_equal(m._gen_model["_do_intervene_a"].get_value(), before)
    with pytest.raises(RuntimeError, match="test exception"):
        m.predict(one_step_ahead=True, progressbar=False)
    assert int(m.pymc_model["_use_observed_carry"].get_value()) == 1


@pytest.mark.slow
def test_real_fit_numpy_ar_covariance_and_coherent_prediction():
    rng = np.random.default_rng(431)
    rows = []
    for unit in ["u0", "u1"]:
        previous = np.zeros(2)
        for t in range(250):
            y = previous * [0.4, 0.2] + L @ rng.normal(size=2)
            rows.append(
                dict(
                    unit=unit,
                    time=t,
                    a=y[0],
                    b=y[1],
                    child=y.sum() + rng.normal(scale=0.02),
                )
            )
            previous = y
    m = pathmc.model(
        SPEC, pd.DataFrame(rows), panel=PANEL, initial={"a": 0.0, "b": 0.0}
    )
    m.fit(
        random_seed=24,
        compute_log_likelihood=False,
        progressbar=False,
        target_accept=0.95,
    )
    post = m._idata.posterior.dataset
    packed = post.chol_a_b.values.reshape(-1, 3)
    corr = packed[:, 1] / np.sqrt(packed[:, 1] ** 2 + packed[:, 2] ** 2)
    truth = (L @ L.T)[0, 1] / np.sqrt(np.prod(np.diag(L @ L.T)))
    assert (corr > 0).mean() > 0.98
    lo, hi = np.quantile(corr, [0.01, 0.99])
    assert lo < truth < hi
    assert not m._idata.sample_stats.diverging.values.any()
    logp = m.pymc_model.compile_logp()
    for factor, sd in zip(packed, post.sigma_child.values.ravel(), strict=True):
        assert np.isfinite(
            logp({
                "chol_a_b_cholesky-cov-packed__": np.array([
                    np.log(factor[0]),
                    factor[1],
                    np.log(factor[2]),
                ]),
                "sigma_child_log__": np.array(np.log(sd)),
            })
        )
    assert np.isfinite(m.pymc_model.compile_dlogp()(m.pymc_model.initial_point())).all()
    generated = m.predict(
        one_step_ahead=False,
        random_seed=51,
        extend_inferencedata=False,
        progressbar=False,
    ).posterior_predictive.dataset
    error = (
        generated.child.values - generated.a.values - generated.b.values
    ) / post.sigma_child.values[..., None, None]
    assert abs(error.mean()) < 0.04
    assert error.std() == pytest.approx(1.0, abs=0.04)


@pytest.mark.parametrize("mixed", [False, True])
def test_marginal_scale_priors_recompile_and_nested_parameters(mixed):
    priors = {
        "sigma_a": pathmc.Prior("HalfNormal", sigma=0.3),
        "sigma_b": pathmc.Prior("Exponential", lam=2.0)
        if mixed
        else pathmc.Prior("HalfNormal", sigma=1.7),
    }
    m = model(priors=priors)
    old_gen = m._gen_model
    posterior(m)
    for g in [m.pymc_model, m._gen_model]:
        sd = g["chol_a_b"].owner.inputs[3]
        actual = pm.logp(sd, [0.5, 0.8]).eval().sum()
        expected = sum(
            pm.logp(p.pymc_distribution.dist(**p.parameters), value).eval()
            for p, value in zip(priors.values(), [0.5, 0.8], strict=True)
        )
        assert actual == pytest.approx(expected)
    assert "sigma_b: " in str(m.priors())
    m.set_priors({"sigma_b": pathmc.Prior("HalfNormal", sigma=0.9)})
    assert m._gen_model is not old_gen
    assert m._idata is None
    assert m._predictive_gen_model() is m._gen_model
    for g in [m.pymc_model, m._gen_model]:
        np.testing.assert_allclose(
            g["chol_a_b"].owner.inputs[3].owner.inputs[3].eval(), [0.3, 0.9]
        )
    nested = model(
        priors={
            "sigma_a": pathmc.Prior(
                "HalfNormal", sigma=pathmc.Prior("HalfNormal", sigma=0.4)
            )
        }
    )
    for g in [nested.pymc_model, nested._gen_model]:
        assert "a_sigma" in {v.name for v in g.free_RVs}
        assert np.isfinite(g.compile_logp()(g.initial_point()))


def test_partial_intercepts_and_random_slopes_do_not_pool_covariance():
    spec = "a ~ x + 0.4*lag(a)\nb ~ x\na ~~ b"
    m = model(spec, pooling={"intercept": True, "slopes": ["x"]}, initial={"a": 0.0})
    params = {"chol_a_b": L[np.tril_indices(2)], "beta_a": [0.0], "beta_b": [0.0]}
    for v in ["a", "b"]:
        params.update({
            f"mu_alpha_{v}": 0.0,
            f"sigma_alpha_{v}": 2.0,
            f"alpha_{v}_offset": [1.0, -0.5],
            f"mu_slope_{v}_x": 0.3,
            f"sigma_slope_{v}_x": 0.2,
            f"slope_{v}_x_offset": [-1.0, 2.0],
        })
    post = posterior(m, params)
    eps = noise(m, post)
    got = _execute_gaussian_panel(
        m._gen_model, post, m._gen_model._pathmc_panel_scan, innovations=eps
    )
    x = frame().x.to_numpy().reshape(2, 4)
    mu = np.array([2.0, -1.0])[:, None] + np.array([0.1, 0.7])[:, None] * x
    np.testing.assert_allclose(got.b, mu + 0.5 * eps.a + 0.8 * eps.b)
    previous = 0.0
    for t in range(4):
        expected = mu[:, t] + 0.4 * previous + 1.2 * eps.a.values[..., t]
        np.testing.assert_allclose(got.a.values[..., t], expected)
        previous = expected
    assert {v.name for v in m.pymc_model.free_RVs if v.name.startswith("chol_")} == {
        "chol_a_b"
    }
    assert m._gen_model["chol_a_b"].shape.eval().tolist() == [3]


def test_composite_units_and_by_variable_coefficients():
    df = frame().assign(brand="p")
    spec = "a ~ x + 0.4*lag(a)\nb ~ 0 + 0.2*lag(b)\na ~~ b"
    panel = {"unit": ["unit", "brand"], "time": "time"}
    pooling = {"by_var": {"x": "none"}}
    m = pathmc.model(
        spec,
        df.sample(frac=1, random_state=8),
        panel=panel,
        pooling=pooling,
        initial={"a": 0.0, "b": 0.0},
    )
    post = posterior(
        m,
        {
            "beta_a": [0.0],
            "beta_x": [2.0, -1.0],
            "chol_a_b": L[np.tril_indices(2)],
        },
    )
    eps = noise(m, post)
    info = m._gen_model._pathmc_panel_scan
    got = _execute_gaussian_panel(m._gen_model, post, info, innovations=eps)
    assert info.unit_labels == ["u0|p", "u1|p"]
    x = df.x.to_numpy().reshape(2, 4)
    previous_a = previous_b = 0.0
    for t in range(4):
        expected_a = (
            np.array([2.0, -1.0]) * x[:, t]
            + 0.4 * previous_a
            + 1.2 * eps.a.values[..., t]
        )
        expected_b = (
            0.2 * previous_b + 0.5 * eps.a.values[..., t] + 0.8 * eps.b.values[..., t]
        )
        np.testing.assert_allclose(got.a.values[..., t], expected_a)
        np.testing.assert_allclose(got.b.values[..., t], expected_b)
        previous_a, previous_b = expected_a, expected_b


def test_scaled_shuffled_panel_covariance_is_in_business_units():
    df = frame().sample(frac=1, random_state=4)
    spec = "a ~ 0 + 0.4*lag(a)\nb ~ 0 + 0.2*lag(b)\na ~~ b"
    scaling = pathmc.ScalingFactors(
        factors={
            "a": (("unit",), {("u0",): 10.0, ("u1",): 30.0}),
            "b": (("unit",), {("u0",): 3.0, ("u1",): 8.0}),
        },
        roles={v: frozenset({"outcome", "regressor"}) for v in ["a", "b"]},
    )
    m = pathmc.model(
        spec, df, panel=PANEL, scaling=scaling, initial={"a": 0.0, "b": 0.0}
    )
    post = posterior(m, {"chol_a_b": L[np.tril_indices(2)]}, draws=5000)
    got = m.predict(
        one_step_ahead=False,
        extend_inferencedata=False,
        random_seed=34,
        progressbar=False,
    ).posterior_predictive.dataset
    for i, divisors in enumerate([[10.0, 3.0], [30.0, 8.0]]):
        samples = np.stack([got.a.values[0, :, 0, i], got.b.values[0, :, 0, i]])
        expected = np.diag(divisors) @ L @ L.T @ np.diag(divisors)
        se = np.sqrt(
            (expected**2 + np.diag(expected)[:, None] * np.diag(expected)) / 4999
        )
        assert np.all(np.abs(np.cov(samples) - expected) < 5 * se)
    # Identical base innovations also give identical paths before/after row reordering and unscaling.
    params = {"chol_a_b": L[np.tril_indices(2)]}
    raw = pathmc.simulate(spec, df, panel=PANEL, params=params, random_seed=35)
    scaled = pathmc.simulate(
        spec, df, panel=PANEL, params=params, random_seed=35, scaling=scaling
    )
    for v in ["a", "b"]:
        np.testing.assert_allclose(
            scaled[v],
            raw[v]
            * df.unit.map(
                {"u0": 10.0, "u1": 30.0} if v == "a" else {"u0": 3.0, "u1": 8.0}
            ),
        )
    pd.testing.assert_frame_equal(scaled[["unit", "time"]], df[["unit", "time"]])


def test_policy_residual_covariance_stays_rejected():
    m = model(initial={"a": 0.0, "b": 0.0})
    posterior(m)
    with pytest.raises(ValueError, match="residual"):
        m.do(
            policy=pathmc.TreatmentPolicy(
                "hold", "a", pathmc.TimeSchedule([1.0, 1.0, 1.0, 1.0])
            ),
            simulate_over="time",
            horizon=4,
        )


def test_fitting_uses_observed_nonblock_parent():
    df = frame(units=1).assign(
        parent=[7.0, -3.0, 2.0, 5.0], a=[2.0, 1.0, 4.0, -2.0], b=[1.0, -1.0, 3.0, 2.0]
    )
    spec = "parent ~ 0 + 1*x\na ~ 0 + 2*parent + 0.4*lag(a)\nb ~ 0 + 1*x\na ~~ b"
    m = pathmc.model(spec, df, panel=PANEL, initial={"a": 1.0}).pymc_model
    point = m.initial_point()
    point["chol_a_b_cholesky-cov-packed__"] = np.array([np.log(1.2), 0.5, np.log(0.8)])
    point["sigma_parent_log__"] = np.array(np.log(0.7))
    residual = np.column_stack([
        df.a - 2 * df.parent - 0.4 * np.r_[1.0, df.a[:-1]],
        df.b - df.x,
    ])
    white = np.linalg.solve(L, residual.T)
    expected = -0.5 * np.sum(white**2) - 4 * (
        np.log(2 * np.pi) + np.log(np.diag(L)).sum()
    )
    expected += np.sum(
        -0.5 * np.log(2 * np.pi * 0.7**2) - 0.5 * ((df.parent - df.x) / 0.7) ** 2
    )
    assert m.compile_logp(vars=m.observed_RVs)(point) == pytest.approx(expected)


@pytest.mark.parametrize("held,variance", [("a", 0.89), ("b", 1.44)])
def test_random_hard_intervention_preserves_marginal_variance(held, variance):
    m = model(units=1, n=1, initial={"a": 0.0, "b": 0.0})
    post = posterior(m, draws=6000)
    info = m._gen_model._pathmc_panel_scan
    trajectory = xr.DataArray(
        [[4.0]],
        dims=("unit", "time"),
        coords={"unit": info.unit_labels, "time": info.time_values},
    )
    got = _execute_gaussian_panel(
        m._gen_model, post, info, interventions={held: trajectory}, random_seed=29
    )
    peer = "b" if held == "a" else "a"
    assert abs(got[peer].values.var(ddof=1) - variance) < 5 * variance * np.sqrt(
        2 / 5999
    )


def test_supplied_noise_simulation_axis_and_validation():
    m = model(units=1)
    post = posterior(m)
    eps = noise(m, post).expand_dims(simulation=[2, 4], axis=2).copy(deep=True)
    eps.a.loc[{"simulation": 4}] += 1
    info = m._gen_model._pathmc_panel_scan
    got = _execute_gaussian_panel(m._gen_model, post, info, innovations=eps)
    assert got.a.dims == ("chain", "draw", "simulation", "unit", "time")
    assert np.all(got.b.sel(simulation=4) > got.b.sel(simulation=2))
    with pytest.raises(ValueError, match="cover exactly"):
        _execute_gaussian_panel(
            m._gen_model, post, info, innovations=eps.drop_vars("b")
        )
    with pytest.raises(ValueError, match="coordinates"):
        _execute_gaussian_panel(
            m._gen_model, post, info, innovations=eps.assign_coords(draw=[1, 2, 3])
        )


def test_one_step_observed_history_and_public_predictive_do():
    df = frame(units=1).assign(a=[2.0, -1.0, 4.0, 3.0], b=[1.0, 5.0, -2.0, 4.0])
    m = pathmc.model(SPEC, df, panel=PANEL, initial={"a": 1.0, "b": -2.0})
    post = posterior(m, draws=4000)
    conditional = m.predict(
        random_seed=83, extend_inferencedata=False, progressbar=False
    ).posterior_predictive.dataset
    expected_a = 0.4 * np.r_[1.0, df.a[:-1]]
    expected_b = 0.2 * np.r_[-2.0, df.b[:-1]]
    residual = np.stack(
        [
            conditional.a.values[0, :, :, 0] - expected_a,
            conditional.b.values[0, :, :, 0] - expected_b,
        ],
        -1,
    )
    assert np.all(np.abs(residual.mean(0)) < 5 * np.sqrt(np.diag(L @ L.T) / 4000))
    np.testing.assert_allclose(
        conditional.child.values,
        np.broadcast_to((df.a + df.b).to_numpy()[None, None, :, None], (1, 4000, 4, 1)),
    )
    before = m._idata.posterior.to_dataset().copy(deep=True)
    public = m.do(kind="predictive", simulate_over="time", random_seed=61).dataset
    again = m.do(kind="predictive", simulate_over="time", random_seed=61).dataset
    xr.testing.assert_equal(public, again)
    np.testing.assert_allclose(public.child, public.a + public.b)
    xr.testing.assert_equal(m._idata.posterior.to_dataset(), before)
    released = m.do(
        set={"a": np.array([4.0, np.nan, 2.0, np.nan])},
        kind="mean",
        simulate_over="time",
        initial={"a": 1.0, "b": -2.0},
    ).dataset
    np.testing.assert_allclose(
        released.a.transpose("chain", "draw", "time"),
        np.broadcast_to([4.0, 1.6, 2.0, 0.8], (1, 4000, 4)),
    )
    np.testing.assert_allclose(released.child, released.a + released.b)


def test_distinct_blocks_and_nonblock_residuals_are_independent():
    spec = "a ~ 0 + 0.4*lag(a)\nb ~ 0 + 1*x\nc ~ 0 + 1*x\nd ~ 0 + 1*x\nchild ~ 0 + 1*a + 1*b\na ~~ b\nc ~~ d"
    m = model(spec, n=1, units=1, initial={"a": 0.0})
    other = np.array([[0.7, 0], [-0.2, 1.1]])
    post = posterior(
        m,
        {
            "chol_a_b": L[np.tril_indices(2)],
            "chol_c_d": other[np.tril_indices(2)],
            "sigma_child": 0.4,
        },
        draws=6000,
    )
    got = _execute_gaussian_panel(
        m._gen_model, post, m._gen_model._pathmc_panel_scan, random_seed=438
    )
    residual = np.column_stack([
        got.a.values.ravel(),
        got.b.values.ravel() - 1.0,
        got.c.values.ravel() - 1.0,
        got.d.values.ravel() - 1.0,
        (got.child - got.a - got.b).values.ravel(),
    ])
    target = np.zeros((5, 5))
    target[:2, :2] = L @ L.T
    target[2:4, 2:4] = other @ other.T
    target[4, 4] = 0.4**2
    se = np.sqrt((target**2 + np.diag(target)[:, None] * np.diag(target)) / 5999)
    assert np.all(np.abs(np.cov(residual.T) - target) < 5 * se)


@pytest.mark.parametrize("caller", [pathmc.simulate, pathmc.simulate_params_template])
@pytest.mark.parametrize(
    "kind", ["latent", "mixed", "member", "singleton", "exogenous"]
)
def test_simulation_and_template_share_capability_validation(caller, kind):
    spec = SPEC
    kwargs = {}
    error = NotImplementedError
    match = "Gaussian|gaussian"
    if kind == "latent":
        spec += "\nhidden ~ x"
        kwargs["latent"] = ["hidden"]
    elif kind == "mixed":
        kwargs["families"] = {"child": "poisson"}
    elif kind == "member":
        kwargs["families"] = {"a": "studentt"}
        error = ValueError
    elif kind == "singleton":
        spec = "a ~ lag(a)\na ~~ a"
        error, match = ValueError, "one outcome|distinct"
    else:
        spec = spec.replace("child ~ 0 + 1*a + 1*b", "child ~ a + b + x") + "\na ~~ x"
        error, match = ValueError, "x.*equation"
    if caller is pathmc.simulate:
        kwargs["params"] = {}
    with pytest.raises(error, match=match):
        caller(spec, frame(), panel=PANEL, **kwargs)
