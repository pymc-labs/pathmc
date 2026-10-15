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
"""Independent joint-path and conditional-likelihood oracles for #543."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

import pathmc


def _model(spec="y ~ 0 + 0.5*lag(y)\nz ~ 0 + 1*y", n=4, units=1):
    data = pd.DataFrame({
        "unit": np.repeat([f"u{i}" for i in range(units)], n),
        "time": np.tile(np.arange(n), units),
        "x": np.zeros(n * units),
        "y": np.zeros(n * units),
        "z": np.zeros(n * units),
    })
    return pathmc.model(spec, data=data, panel={"unit": "unit", "time": "time"})


def _posterior(model, draws=1000):
    """Execution oracle: deliberate zero innovations, not a fitted posterior."""
    info = model._gen_model._pathmc_panel_scan
    post = xr.Dataset(coords={"chain": [7], "draw": np.arange(draws) + 11})
    for var, sigma in [("y", 1.0), ("z", 0.0)]:
        post[f"sigma_{var}"] = (("chain", "draw"), np.full((1, draws), sigma))
        post[f"carry_innovations_{var}"] = (
            ("chain", "draw", f"{var}_dim_0", f"{var}_dim_1"),
            np.zeros((1, draws, info.n_times, info.n_units)),
        )
    model._idata = xr.DataTree.from_dict({"posterior": post})
    return post


def test_named_generative_states_are_the_propagated_realization():
    model = _model()
    graph = pm.do(
        model._gen_model, {"sigma_y": np.array(1.0), "sigma_z": np.array(0.0)}
    )
    y, z = pm.draw([graph["y"], graph["z"]], draws=100, random_seed=123)
    np.testing.assert_array_equal(y, z)


def test_fresh_noise_joint_paths_covariance_and_seed():
    model = _model()
    before = _posterior(model, draws=6000).copy(deep=True)
    first = model.do(kind="predictive", simulate_over="time", random_seed=123).dataset
    again = model.do(kind="predictive", simulate_over="time", random_seed=123).dataset
    other = model.do(kind="predictive", simulate_over="time", random_seed=124).dataset
    xr.testing.assert_equal(first, again)
    xr.testing.assert_equal(model._idata.posterior.to_dataset(), before)
    np.testing.assert_array_equal(first.y, first.z)
    assert not np.array_equal(first.y, other.y)
    samples = first.y.transpose("chain", "draw", "time").values.reshape(6000, 4)
    variance = np.array([1, 1.25, 1.3125, 1.328125])
    covariance = np.array([
        [0.5 ** abs(i - j) * variance[min(i, j)] for j in range(4)] for i in range(4)
    ])
    # Gaussian sample covariance SE: sqrt((Cij**2 + Cii*Cjj)/(N-1)).
    se = np.sqrt((covariance**2 + variance[:, None] * variance) / 5999)
    assert np.all(np.abs(np.cov(samples.T) - covariance) < 5 * se)


def test_current_intermediary_noise_propagates():
    model = _model("y ~ 0 + 1*x\nz ~ 0 + 1*y + 0.5*lag(z)")
    _posterior(model, draws=2000)
    result = model.do(kind="predictive", simulate_over="time", random_seed=52).dataset
    y = result.y.transpose("chain", "draw", "time").values
    z = result.z.transpose("chain", "draw", "time").values
    np.testing.assert_allclose(z[..., 0], y[..., 0])
    np.testing.assert_allclose(z[..., 1:], y[..., 1:] + 0.5 * z[..., :-1])
    assert np.std(z) > 0.9


@pytest.mark.parametrize("var_names", [None, ["z"]])
def test_free_running_predict_preserves_variable_selection(var_names):
    model = _model()
    _posterior(model, draws=4)
    kwargs = {
        "one_step_ahead": False,
        "extend_inferencedata": False,
        "random_seed": 19,
        "progressbar": False,
    }
    default = model.predict(**kwargs).posterior_predictive.to_dataset()
    selected = model.predict(
        var_names=var_names, **kwargs
    ).posterior_predictive.to_dataset()
    expected_names = ["y", "z"] if var_names is None else var_names
    assert set(selected.data_vars) == set(expected_names)
    for name in expected_names:
        xr.testing.assert_equal(selected[name], default[name])


def test_observational_current_predictor_logp_and_gradient():
    data = pd.DataFrame({
        "unit": ["u"] * 4,
        "time": range(4),
        "x": [1.0, 2.0, -1.0, 3.0],
        "y": [4.0, -2.0, 5.0, 1.0],
        "z": [2.0, 1.0, -3.0, 4.0],
    })
    model = pathmc.model(
        "y ~ x\nz ~ y + lag(z)", data=data, panel={"unit": "unit", "time": "time"}
    ).pymc_model
    point = model.initial_point()
    point["beta_y"] = np.array([0.2, 0.7])
    point["beta_z"] = np.array([-0.3, 0.8, 0.4])
    point["sigma_y_log__"] = np.array(np.log(0.9))
    point["sigma_z_log__"] = np.array(np.log(1.2))
    means = [
        0.2 + 0.7 * data.x.to_numpy(),
        -0.3 + 0.8 * data.y.to_numpy() + 0.4 * np.r_[data.z.iloc[0], data.z[:-1]],
    ]
    residuals = [data.y.to_numpy() - means[0], data.z.to_numpy() - means[1]]
    expected = sum(
        np.sum(-0.5 * np.log(2 * np.pi * s * s) - 0.5 * (r / s) ** 2)
        for r, s in zip(residuals, [0.9, 1.2])
    )
    logp = model.compile_logp(vars=model.observed_RVs)
    assert float(logp(point)) == pytest.approx(expected)
    gradient = pt.grad(
        model.logp(vars=model.observed_RVs),
        [model.rvs_to_values[model[name]] for name in ["beta_y", "beta_z"]],
    )
    evaluate = pytensor.function(model.value_vars, gradient, on_unused_input="ignore")
    actual = evaluate(*[point[v.name] for v in model.value_vars])
    designs = [
        np.column_stack([np.ones(4), data.x]),
        np.column_stack([np.ones(4), data.y, np.r_[data.z.iloc[0], data.z[:-1]]]),
    ]
    for got, design, residual, sigma in zip(actual, designs, residuals, [0.9, 1.2]):
        np.testing.assert_allclose(got, design.T @ residual / sigma**2, atol=1e-10)


def _noise(model, post):
    info = model._gen_model._pathmc_panel_scan
    coords = {
        "chain": post.chain,
        "draw": post.draw,
        "unit": info.unit_labels,
        "time": info.time_values,
    }
    rng = np.random.default_rng(88)
    return xr.Dataset({
        v: xr.DataArray(
            rng.normal(
                size=(
                    post.sizes["chain"],
                    post.sizes["draw"],
                    info.n_units,
                    info.n_times,
                )
            ),
            dims=("chain", "draw", "unit", "time"),
            coords=coords,
        )
        for v in info.gaussian_states
    })


def test_fixed_innovations_initials_interactions_and_interventions():
    from pathmc.simulate import _execute_gaussian_panel

    model = _model("y ~ 0 + 0.5*lag(y) + 2*lag(x)\nz ~ 0 + 1*y + 3*y:x", units=2)
    post = _posterior(model, draws=3)
    info = model._gen_model._pathmc_panel_scan
    noise = _noise(model, post)
    initial = {
        v: xr.DataArray(vals, dims="unit", coords={"unit": info.unit_labels})
        for v, vals in [("y", [2.0, -3.0]), ("x", [4.0, 5.0])]
    }
    x = xr.DataArray(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        dims=("unit", "time"),
        coords={"unit": info.unit_labels, "time": info.time_values},
    )
    result = _execute_gaussian_panel(
        model._gen_model,
        post,
        info,
        initial=initial,
        innovations=noise,
        interventions={"x": x},
    )
    expected = np.empty((1, 3, 2, 4))
    previous = initial["y"].values
    previous_x = initial["x"].values
    for t in range(4):
        expected[..., t] = 0.5 * previous + 2 * previous_x + noise.y.values[..., t]
        previous = expected[..., t]
        previous_x = x.values[:, t]
    np.testing.assert_allclose(result.y, expected)
    np.testing.assert_allclose(result.z, expected * (1 + 3 * x.values))
    assert result.y.dims == ("chain", "draw", "unit", "time")
    xr.testing.assert_equal(result.chain, post.chain)
    forced = _execute_gaussian_panel(
        model._gen_model,
        post,
        info,
        initial=initial,
        innovations=noise,
        interventions={"y": x, "x": x},
    )
    np.testing.assert_allclose(forced.y, np.broadcast_to(x, forced.y.shape))
    np.testing.assert_allclose(
        forced.z, np.broadcast_to(x * (1 + 3 * x), forced.z.shape)
    )
    again = _execute_gaussian_panel(
        model._gen_model,
        post,
        info,
        initial=initial,
        innovations=noise,
        interventions={"x": x},
        random_seed=400,
    )
    xr.testing.assert_equal(result, again)


@pytest.mark.parametrize(
    "bad", ["initial", "noise_names", "noise_coords", "nonfinite", "intervention"]
)
def test_execution_seam_rejects_ambiguous_inputs(bad):
    from pathmc.simulate import _execute_gaussian_panel

    model = _model()
    post = _posterior(model, draws=2)
    info = model._gen_model._pathmc_panel_scan
    noise = _noise(model, post)
    kwargs = {"innovations": noise}
    if bad == "initial":
        kwargs["initial"] = {}
    elif bad == "noise_names":
        kwargs["innovations"] = noise.drop_vars("z")
    elif bad == "noise_coords":
        kwargs["innovations"] = noise.assign_coords(draw=[1, 2])
    elif bad == "nonfinite":
        noise["y"][:] = np.nan
    else:
        kwargs["interventions"] = {"unknown": noise.y.isel(chain=0, draw=0)}
    with pytest.raises(ValueError):
        _execute_gaussian_panel(model._gen_model, post, info, **kwargs)


def test_predict_and_missing_data_do_not_freeze_future_imputations():
    data = pd.DataFrame({
        "unit": ["u"] * 4,
        "time": range(4),
        "y": [0.0, np.nan, 0.0, 0.0],
        "z": np.zeros(4),
    })
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)\nz ~ 0 + 1*y",
        data=data,
        panel={"unit": "unit", "time": "time"},
    )
    post = _posterior(model, draws=40)
    post["y_unobserved"] = (("chain", "draw", "missing"), np.full((1, 40, 1), 999.0))
    model._idata["posterior"] = xr.DataTree(post)
    first = model.predict(
        one_step_ahead=False,
        random_seed=123,
        extend_inferencedata=False,
        progressbar=False,
    ).posterior_predictive.dataset
    np.testing.assert_array_equal(first.y, first.z)
    assert np.max(np.abs(first.y)) < 10
    before = model.do(kind="predictive", simulate_over="time", random_seed=42).dataset
    model.predict(
        one_step_ahead=True,
        random_seed=456,
        extend_inferencedata=False,
        progressbar=False,
    )
    after = model.do(kind="predictive", simulate_over="time", random_seed=42).dataset
    xr.testing.assert_equal(before, after)
    assert int(model.pymc_model["_use_observed_carry"].get_value()) == 1


def test_simulate_shuffled_rows_preserves_joint_path():
    frame = pd.DataFrame({
        "unit": np.repeat(["b", "a"], 5),
        "time": np.tile(np.arange(5), 2),
        "x": np.arange(10, dtype=float),
    }).sample(frac=1, random_state=30)
    result = pathmc.simulate(
        "y ~ 0 + 1*x\nz ~ 0 + 1*y + 0.5*lag(z)",
        frame,
        params={"sigma_y": 1.0, "sigma_z": 0.0},
        panel={"unit": "unit", "time": "time"},
        random_seed=51,
    )
    np.testing.assert_array_equal(result.unit, frame.unit)
    np.testing.assert_array_equal(result.time, frame.time)
    for _, group in result.sort_values(["unit", "time"]).groupby("unit"):
        np.testing.assert_allclose(group.z.iloc[0], group.y.iloc[0])
        np.testing.assert_allclose(
            group.z.to_numpy()[1:],
            group.y.to_numpy()[1:] + 0.5 * group.z.to_numpy()[:-1],
        )


def test_scaled_two_unit_intervention_current_and_future_states():
    from pathmc import Scaling

    frame = pd.DataFrame({
        "unit": np.repeat(["b", "a"], 4),
        "time": np.tile(np.arange(4), 2),
        "y": [40.0] * 4 + [10.0] * 4,
        "z": np.zeros(8),
    }).sample(frac=1, random_state=8)
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)\nz ~ 0 + 1*y + 2*lag(y)",
        data=frame,
        panel={"unit": "unit", "time": "time"},
        scaling=Scaling(
            target={"method": "divide", "by": {"a": 10.0, "b": 20.0}, "dims": ("unit",)}
        ),
    )
    _posterior(model, draws=3)
    values = np.array([30.0, 50.0, 20.0, 40.0])
    with pytest.warns(UserWarning, match="outside the observed"):
        result = model.do(
            set={"y": values}, kind="predictive", simulate_over="time", random_seed=98
        )
    np.testing.assert_allclose(
        result.dataset.y.transpose("chain", "draw", "time"),
        values[None, None, :] * np.ones((1, 3, 1)),
    )
    # Initial y is still each unit's observed boundary; aggregate only after unscaling.
    expected_z = values + 2 * np.r_[25.0, values[:-1]]
    np.testing.assert_allclose(
        result.dataset.z.transpose("chain", "draw", "time"),
        expected_z[None, None, :] * np.ones((1, 3, 1)),
    )


def test_exception_restores_execution_flags(monkeypatch):
    model = _model()
    _posterior(model, draws=2)
    flag = model._gen_model["_use_observed_carry"]
    flag.set_value(np.array(1, dtype=flag.dtype))

    def fail(*args, **kwargs):
        raise RuntimeError("execution failed")

    monkeypatch.setattr(pm, "sample_posterior_predictive", fail)
    with pytest.raises(RuntimeError, match="execution failed"):
        model.do(kind="predictive", simulate_over="time", random_seed=9)
    assert int(flag.get_value()) == 1


def test_fitted_unit_effects_and_supplied_noise_are_conditioned_once():
    from pathmc.simulate import _execute_gaussian_panel

    frame = pd.DataFrame({
        "unit": np.repeat(["a", "b"], 4),
        "time": np.tile(np.arange(4), 2),
        "y": np.zeros(8),
        "z": np.zeros(8),
    })
    model = pathmc.model(
        "y ~ 0 + 0.5*lag(y)\nz ~ 0 + 1*y",
        data=frame,
        panel={"unit": "unit", "time": "time"},
        pooling="partial",
    )
    post = _posterior(model, draws=3)
    post = post.assign_coords(unit=["a", "b"])
    for var, effects in [("y", [1.0, -2.0]), ("z", [0.3, 0.8])]:
        post[f"alpha_{var}"] = (
            ("chain", "draw", "unit"),
            np.broadcast_to(effects, (1, 3, 2)),
        )
        post[f"mu_alpha_{var}"] = (("chain", "draw"), np.full((1, 3), 100.0))
        post[f"sigma_alpha_{var}"] = (("chain", "draw"), np.full((1, 3), 50.0))
    post["sigma_y"][:] = 0.7
    post["sigma_z"][:] = 0.2
    info = model._gen_model._pathmc_panel_scan
    noise = _noise(model, post)
    result = _execute_gaussian_panel(model._gen_model, post, info, innovations=noise)
    expected = np.empty((1, 3, 2, 4))
    previous = np.zeros(2)
    for t in range(4):
        expected[..., t] = (
            np.array([1.0, -2.0]) + 0.5 * previous + 0.7 * noise.y.values[..., t]
        )
        previous = expected[..., t]
    np.testing.assert_allclose(result.y, expected)
    np.testing.assert_allclose(
        result.z, expected + np.array([0.3, 0.8])[:, None] + 0.2 * noise.z.values
    )


def test_legacy_mean_keeps_lag_noise_and_ignores_nonlag_noise():
    model = _model("y ~ 0 + 1*x\nz ~ 0 + 1*y + 0.5*lag(z)")
    post = _posterior(model, draws=2).assign_coords(chain=[0], draw=[0, 1])
    post["sigma_z"][:] = 1.0
    post["carry_innovations_z"][:] = 0.7
    post["carry_innovations_y"][:] = 100.0
    model._idata["posterior"] = xr.DataTree(post)
    result = model.do(kind="mean", simulate_over="time").dataset
    np.testing.assert_array_equal(
        result.y.transpose("chain", "draw", "time"), np.zeros((1, 2, 4))
    )
    np.testing.assert_allclose(
        result.z.transpose("chain", "draw", "time"),
        np.broadcast_to([0.0, 0.35, 0.525, 0.6125], (1, 2, 4)),
    )


@pytest.mark.slow
def test_fitted_numpy_panel_recovers_and_generates_coherent_paths():
    rng = np.random.default_rng(72)
    rows = []
    for unit in ["a", "b"]:
        previous = 1.0
        for time in range(140):
            x = rng.normal()
            y = 0.8 + 0.6 * x + 0.4 * previous + rng.normal(scale=0.3)
            z = y + rng.normal(scale=0.03)
            rows.append({"unit": unit, "time": time, "x": x, "y": y, "z": z})
            previous = y
    model = pathmc.model(
        "y ~ x + lag(y)\nz ~ 0 + 1*y",
        data=pd.DataFrame(rows),
        panel={"unit": "unit", "time": "time"},
    )
    model.fit(random_seed=51, compute_log_likelihood=False, progressbar=False)
    post = model._idata.posterior.dataset
    for term, truth in [("Intercept", 0.8), ("x", 0.6), ("lag(y)", 0.4)]:
        assert float(post.beta_y.sel(y_predictors=term).mean()) == pytest.approx(
            truth, abs=0.15
        )
    assert float(post.sigma_y.mean()) == pytest.approx(0.3, abs=0.06)
    assert float(post.sigma_z.mean()) == pytest.approx(0.03, abs=0.01)
    generated = model.predict(
        one_step_ahead=False,
        random_seed=85,
        extend_inferencedata=False,
        progressbar=False,
    ).posterior_predictive
    residual = generated.z.values - generated.y.values
    standardized = residual / post.sigma_z.values[..., None, None]
    assert abs(standardized.mean()) < 0.06
    assert standardized.std() == pytest.approx(1.0, abs=0.06)
