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
"""Scan shape contracts survive shared-data freezing in alternative samplers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pymc as pm

import pathmc
from pathmc.priors import Prior


def _frame(n_units=1, n_times=8, sparse=True):
    rng = np.random.default_rng(85)
    rows = []
    for unit in range(n_units):
        previous = 0.2 * (unit + 1)
        for time in range(n_times):
            x = rng.normal()
            previous = 0.1 + 0.6 * previous + 0.25 * x + rng.normal(scale=0.2)
            y = previous + rng.normal(scale=0.1)
            rows.append({
                "unit": f"u{unit}",
                "time": time,
                "x": x,
                "y": y if not sparse or time in (1, 4, 7) else np.nan,
            })
    return pd.DataFrame(rows)


def _latent(n_units=1, sparse=True, **kwargs):
    return pathmc.model(
        "a ~ lag(a) + x\ny ~ 0 + 1*a",
        data=_frame(n_units, sparse=sparse),
        panel={"unit": "unit", "time": "time"},
        latent=["a"],
        families={"a": "latent_normal"},
        **kwargs,
    )


@pytest.mark.parametrize("n_units", [1, 3])
def test_sparse_latent_jax_logp(n_units):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    model = _latent(n_units).pymc_model
    point = model.initial_point()
    actual = get_jaxified_logp(model)(list(point.values()))
    np.testing.assert_allclose(actual, model.compile_logp()(point), rtol=1e-8)


def _compare_compilers(model):
    """Exercise both shared-replacement paths, including native gradients."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.model.transform.optimization import freeze_dims_and_data
    from pymc.sampling.jax import get_jaxified_logp

    point = model.initial_point()
    values = [point[v.name] for v in model.value_vars]
    native = model.compile_logp()(point)
    native_grad = model.compile_dlogp()(point)
    function = get_jaxified_logp(model)
    np.testing.assert_allclose(function(values), native, rtol=1e-8, atol=1e-8)
    jax_grad = np.concatenate([np.ravel(v) for v in jax.grad(function)(values)])
    np.testing.assert_allclose(jax_grad, native_grad, rtol=1e-8, atol=1e-8)
    # nutpie freezes dimensions/data before its separate raveled-input path.
    frozen = freeze_dims_and_data(model)
    np.testing.assert_allclose(frozen.compile_logp()(point), native, atol=1e-8)
    frozen_gradient = frozen.compile_dlogp(
        vars=[frozen[rv.name] for rv in model.free_RVs]
    )
    np.testing.assert_allclose(frozen_gradient(point), native_grad, atol=1e-8)


@pytest.mark.parametrize("n_units", [1, 3])
@pytest.mark.parametrize(
    "case,pooling",
    [
        ("dense_latent", None),
        ("ar", None),
        ("ar", "partial"),
        ("exogenous_lag", None),
        ("exogenous_lag", "partial"),
        ("adstock", None),
        ("adstock", "partial"),
        ("plain", None),
        ("sparse", None),
    ],
)
def test_isolation_controls(n_units, case, pooling):
    if case == "dense_latent":
        fitted = _latent(n_units, sparse=False)
    else:
        spec = {
            "ar": "y ~ 0 + lag(y)",
            "exogenous_lag": "y ~ 0 + lag(x)",
            "adstock": "y ~ 0 + adstock(x, decay=d)",
            "plain": "y ~ x",
            "sparse": "y ~ x",
        }[case]
        fitted = pathmc.model(
            spec,
            data=_frame(n_units, sparse=case == "sparse"),
            panel={"unit": "unit", "time": "time"},
            pooling=pooling,
        )
        if case in {"ar", "exogenous_lag", "adstock"}:
            assert len(fitted.pymc_model.coords["y_predictors"]) == 1
    _compare_compilers(fitted.pymc_model)


def _normal_logp(value, mean=0.0, sd=1.0):
    return np.sum(
        -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((value - mean) / sd) ** 2
    )


def _trajectory(point, x):
    beta = point["beta_a"]
    sigma = np.exp(point["sigma_a_log__"])
    previous = point["init_a"].copy()
    result = []
    for t, x_t in enumerate(x):
        previous = (
            beta[0]
            + beta[1] * previous
            + beta[2] * x_t
            + sigma * point["innovations_a"][t]
        )
        result.append(previous.copy())
    return np.array(result)


@pytest.mark.parametrize("n_units", [1, 3])
def test_nonzero_joint_density_and_gradient_oracle(n_units):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from pymc.sampling.jax import get_jaxified_logp

    fitted = _latent(
        n_units,
        priors={
            "beta_a": Prior("Normal", mu=0, sigma=1),
            "init_a": Prior("Normal", mu=0, sigma=1),
            "sigma_a": Prior("HalfNormal", sigma=1),
            "sigma_y": Prior("HalfNormal", sigma=1),
        },
    )
    model = fitted.pymc_model
    data = _frame(n_units)
    x = data.x.to_numpy().reshape(n_units, 8).T
    y = data.y.to_numpy().reshape(n_units, 8).T
    missing = np.isnan(y)
    function = get_jaxified_logp(model)
    native = model.compile_logp()
    gradient = model.compile_dlogp()
    states = model.compile_fn(
        model.replace_rvs_by_values([model["a"]]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )

    def oracle(point):
        completed = y.copy()
        completed[missing] = point["y_unobserved"]
        result = _normal_logp(
            completed, _trajectory(point, x), np.exp(point["sigma_y_log__"])
        )
        for name in ("beta_a", "init_a", "innovations_a"):
            result += _normal_logp(point[name])
        for name in ("sigma_a_log__", "sigma_y_log__"):
            sigma = np.exp(point[name])
            result += np.log(2) + _normal_logp(sigma) + point[name]
        return result

    for shift in (0.0, 0.15):
        point = model.initial_point()
        point.update(
            beta_a=np.array([0.1, 0.6, 0.25]) + shift,
            init_a=np.linspace(0.3, 0.9, n_units) + shift,
            sigma_a_log__=np.array(np.log(0.3 + shift)),
            sigma_y_log__=np.array(np.log(0.2 + shift)),
            innovations_a=np.linspace(-0.4, 0.5, 8 * n_units).reshape(8, n_units)
            + shift,
            y_unobserved=np.linspace(-0.1, 0.8, missing.sum()) + shift,
        )
        assert list(model.coords["a_predictors"]) == ["Intercept", "lag(a)", "x"]
        expected = oracle(point)
        values = [point[v.name] for v in model.value_vars]
        np.testing.assert_allclose(native(point), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(function(values), expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(states(point)[0], _trajectory(point, x), atol=1e-12)
        finite_difference = []
        for var in model.value_vars:
            for index in np.ndindex(point[var.name].shape):
                plus = {k: v.copy() for k, v in point.items()}
                minus = {k: v.copy() for k, v in point.items()}
                plus[var.name][index] += 1e-6
                minus[var.name][index] -= 1e-6
                finite_difference.append((oracle(plus) - oracle(minus)) / 2e-6)
        np.testing.assert_allclose(
            gradient(point), finite_difference, rtol=2e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            np.concatenate([np.ravel(v) for v in jax.grad(function)(values)]),
            finite_difference,
            rtol=2e-6,
            atol=2e-6,
        )


@pytest.mark.parametrize("n_units,n_times", [(1, 1), (1, 12), (3, 12)])
def test_multiple_latents_indicators_masks_and_row_order(n_units, n_times):
    frame = _frame(n_units, n_times, sparse=False)
    frame["z"] = 0.7 + 0.5 * frame.y
    if n_times > 1:
        frame.loc[~frame.time.isin([1, 9]), "y"] = np.nan
        frame.loc[~frame.time.isin([2, 10]), "z"] = np.nan
    options = dict(
        panel={"unit": "unit", "time": "time"},
        latent=["a", "b"],
        families={"a": "latent_normal", "b": "latent_normal"},
    )
    spec = "a ~ lag(a) + x\nb ~ lag(b) + a\ny ~ 0 + 1*a\nz ~ 0 + 1*b"
    ordered = pathmc.model(spec, data=frame, **options)
    shuffled = pathmc.model(spec, data=frame.sample(frac=1, random_state=4), **options)
    _compare_compilers(shuffled.pymc_model)
    point = ordered.pymc_model.initial_point()
    for var in ("a", "b"):
        point[f"init_{var}"] = np.arange(n_units, dtype=float) + 0.4
        point[f"innovations_{var}"] = np.full((n_times, n_units), 0.2)
    np.testing.assert_allclose(
        shuffled.pymc_model.compile_logp()(point),
        ordered.pymc_model.compile_logp()(point),
        atol=1e-10,
    )


@pytest.mark.parametrize("n_units", [1, 3])
@pytest.mark.parametrize("parameter", ["coefficient", "decay", "slope"])
def test_per_unit_non_sequences(n_units, parameter):
    pooling = {
        "coefficient": {"intercept": True, "by_var": {"x": "none"}},
        "decay": {"intercept": True, "by_var": {"d": "none"}},
        "slope": {"intercept": True, "slopes": ["x"]},
    }[parameter]
    spec = (
        "y ~ lag(y) + adstock(x, decay=d)" if parameter == "decay" else "y ~ lag(y) + x"
    )
    fitted = pathmc.model(
        spec,
        data=_frame(n_units, sparse=False),
        panel={"unit": "unit", "time": "time"},
        pooling=pooling,
    )
    _compare_compilers(fitted.pymc_model)


@pytest.mark.parametrize("n_units", [1, 3])
def test_explicit_initial_shared_states(n_units):
    fitted = pathmc.model(
        "y ~ 0 + 0.5*lag(y) + 1*x",
        data=_frame(n_units, sparse=False),
        panel={"unit": "unit", "time": "time"},
        initial={"y": {f"u{i}": i + 0.7 for i in range(n_units)}},
    )
    _compare_compilers(fitted.pymc_model)


def _shared_snapshot(model):
    return {
        name: np.array(var.get_value(), copy=True)
        for name, var in model.named_vars.items()
        if hasattr(var, "get_value")
    }


def _assert_shared(model, snapshot):
    for name, value in snapshot.items():
        np.testing.assert_array_equal(model[name].get_value(), value, err_msg=name)


@pytest.mark.slow
@pytest.mark.parametrize("n_units", [1, 3])
@pytest.mark.parametrize("sampler", ["pymc", "nutpie-numba", "numpyro", "nutpie-jax"])
@pytest.mark.parametrize("compute_log_likelihood", [True, False])
def test_backend_fit_and_subsequent_queries(
    n_units, sampler, compute_log_likelihood, monkeypatch
):
    if sampler.startswith("nutpie"):
        pytest.importorskip("nutpie")
    if sampler in {"numpyro", "nutpie-jax"}:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
    if sampler == "numpyro":
        pytest.importorskip("numpyro")
    fitted = _latent(n_units)
    before = _shared_snapshot(fitted._gen_model)
    options = {"backend": sampler.split("-")[1]} if sampler.startswith("nutpie") else {}
    idata = fitted.fit(
        nuts_sampler="nutpie" if sampler.startswith("nutpie") else sampler,
        random_seed=85,
        progressbar=False,
        compute_convergence_checks=False,
        compute_log_likelihood=compute_log_likelihood,
        **options,
    )
    _assert_shared(fitted._gen_model, before)
    post = idata.posterior.to_dataset()
    for name in (
        "beta_a",
        "sigma_a",
        "sigma_y",
        "init_a",
        "innovations_a",
        "a",
        "y_unobserved",
    ):
        assert name in post
        assert np.isfinite(post[name]).all()
    assert ("log_likelihood" in idata) == compute_log_likelihood
    if compute_log_likelihood:
        loglik = idata.log_likelihood.to_dataset()["y_observed"]
        assert loglik.shape[-1] == 3 * n_units
        assert np.isfinite(loglik).all()
    trajectory = fitted.latent_trajectory("a")
    assert trajectory.dims == ("chain", "draw", "time", "unit")
    assert trajectory.shape == (1, 50, 8, n_units)
    np.testing.assert_array_equal(trajectory.unit, [f"u{i}" for i in range(n_units)])
    np.testing.assert_array_equal(trajectory.time, np.arange(8))

    # Keep the exact fitted draws across both query routes. With zero measurement
    # noise, a hard latent intervention has an exact oracle in both routes.
    idata.posterior["sigma_y"] = idata.posterior["sigma_y"] * 0.0
    intervention = np.linspace(0.2, 0.8, 8)
    for kind in ("mean", "predictive"):
        result = fitted.do(
            set={"a": intervention}, simulate_over="time", kind=kind, random_seed=19
        )
        actual = result.by_time("y")
        np.testing.assert_allclose(
            actual, np.broadcast_to(intervention[:, None], actual.shape), atol=1e-12
        )
        _assert_shared(fitted._gen_model, before)
    predicted = fitted.predict(
        var_names=["y"], random_seed=19, progressbar=False, extend_inferencedata=False
    )
    assert np.isfinite(predicted.posterior_predictive["y"]).all()
    _assert_shared(fitted._gen_model, before)

    # Released steps must resume the recurrence using the same parameter and
    # process-innovation draws, rather than conditioning on fitted latent states.
    idata.posterior["beta_a"] = idata.posterior["beta_a"] * 0 + np.array([
        0.1,
        0.6,
        0.25,
    ])
    idata.posterior["sigma_a"] = idata.posterior["sigma_a"] * 0 + 0.3
    idata.posterior["init_a"] = idata.posterior["init_a"] * 0 + 0.4
    idata.posterior["innovations_a"] = idata.posterior["innovations_a"] * 0 + 0.2
    released = intervention.copy()
    released[2:6] = np.nan
    expected, previous = [], np.full(n_units, 0.4)
    for t in range(8):
        previous = 0.1 + 0.6 * previous + 0.25 * before["x"][t] + 0.3 * 0.2
        if np.isfinite(released[t]):
            previous = np.full(n_units, released[t])
        expected.append(previous.mean())
    for kind in ("mean", "predictive"):
        result = fitted.do(
            set={"a": released}, simulate_over="time", kind=kind, random_seed=19
        )
        actual = result.by_time("y")
        np.testing.assert_allclose(
            actual,
            np.broadcast_to(np.array(expected)[:, None], actual.shape),
            atol=1e-10,
        )
        _assert_shared(fitted._gen_model, before)

    # Same-shape updates must still reach the retained shared input after fitting.
    model = fitted.pymc_model
    point = model.initial_point()
    point["beta_a"] = np.array([0.1, 0.6, 0.25])
    state_function = model.compile_fn(
        model.replace_rvs_by_values([model["a"]]),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    original_states = state_function(point)[0]
    with fitted._gen_model:
        pm.set_data({"x": before["x"] + 0.3})
    assert np.array_equal(fitted._gen_model["x"].get_value(), before["x"] + 0.3)
    updated = _shared_snapshot(fitted._gen_model)
    expected_states = _trajectory(point, before["x"] + 0.3)
    np.testing.assert_allclose(state_function(point)[0], expected_states, atol=1e-12)
    assert not np.allclose(expected_states, original_states)
    fitted.do(set={"a": 0.4}, simulate_over="time", kind="mean")
    _assert_shared(fitted._gen_model, updated)

    def fail(*args, **kwargs):
        raise RuntimeError("query probe")

    monkeypatch.setattr(pm, "sample_posterior_predictive", fail)
    with pytest.raises(RuntimeError, match="query probe"):
        fitted.predict(
            one_step_ahead=False, progressbar=False, extend_inferencedata=False
        )
    _assert_shared(fitted._gen_model, updated)
