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
"""Explicit pre-sample lag states for #544."""

import numpy as np
import pandas as pd
import pytest
import pytensor
import pytensor.tensor as pt
import xarray as xr

import pathmc

PANEL = {"unit": "unit", "time": "time"}
SPEC = "y ~ 0 + 1*x + 0.5*lag(y)"


def _frame(order=("A", "B")):
    rows = []
    values = {"A": (3.0, 4.0), "B": (9.0, 8.0)}
    for unit in order:
        for time, y in zip((1, 2), values[unit], strict=True):
            rows.append({"unit": unit, "time": time, "y": y, "x": 1.0})
    return pd.DataFrame(rows)


def _fit_noise(model, sigma=1.0):
    info = model._gen_model._pathmc_panel_scan
    post = xr.Dataset(coords={"chain": [0], "draw": [0]})
    for var in info.gaussian_states or ("y",):
        if f"sigma_{var}" not in [rv.name for rv in model._gen_model.free_RVs]:
            continue
        post[f"sigma_{var}"] = (("chain", "draw"), np.array([[sigma]]))
        name = f"carry_innovations_{var}"
        if name in [rv.name for rv in model._gen_model.free_RVs]:
            post[name] = (
                ("chain", "draw", f"{var}_dim_0", f"{var}_dim_1"),
                np.zeros((1, 1, info.n_times, info.n_units)),
            )
    model._idata = xr.DataTree.from_dict({"posterior": post})


def _nodes(model):
    return {
        name: np.array(values, copy=True)
        for name, values in model._gen_model._pathmc_panel_scan.initial_values.items()
    }


def test_explicit_initial_overrides_inferred_fill_and_aligns_by_label():
    shuffled = pathmc.model(
        SPEC,
        data=_frame(("B", "A")),
        panel=PANEL,
        initial={"y": {"B": 10.0, "A": 2.0}},
    )
    ordered = pathmc.model(
        SPEC,
        data=_frame(("A", "B")),
        panel=PANEL,
        initial={"y": {"A": 2.0, "B": 10.0}},
    )
    inferred = pathmc.model(SPEC, data=_frame(("B", "A")), panel=PANEL)

    assert shuffled.initial_baseline.explicit is True
    assert inferred.initial_baseline.explicit is False
    for model, expected in (
        (shuffled, [2.0, 10.0]),
        (ordered, [2.0, 10.0]),
        (inferred, [3.0, 9.0]),
    ):
        baseline = model.initial_baseline.values["y"]
        assert list(baseline.coords["unit"].values) == ["A", "B"]
        np.testing.assert_allclose(baseline.values, expected)
        np.testing.assert_allclose(
            model._gen_model._pathmc_panel_scan.initial_values["y"], expected
        )


def test_exogenous_legacy_fill_prefers_lag1_column():
    frame = _frame().assign(x_lag1=[1.5, 5.0, 7.0, 1.0])
    inferred = pathmc.model("y ~ lag(x)", data=frame, panel=PANEL)
    explicit = pathmc.model(
        "y ~ lag(x)",
        data=frame,
        panel=PANEL,
        initial={"x": {"A": 4.0, "B": -1.0}},
    )
    np.testing.assert_allclose(inferred.initial_baseline.values["x"].values, [1.5, 7.0])
    np.testing.assert_allclose(
        explicit.initial_baseline.values["x"].values, [4.0, -1.0]
    )
    assert explicit.initial_baseline.explicit is True


def test_scalar_initial_broadcasts_in_label_order():
    model = pathmc.model(SPEC, data=_frame(), panel=PANEL, initial={"y": 4.0})
    np.testing.assert_allclose(model.initial_baseline.values["y"].values, [4.0, 4.0])


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"data": None, "initial": {"y": 1.0}}, "without data"),
        ({"data": _frame(), "initial": {"y": 1.0}}, "requires a panel"),
        (
            {
                "spec": "y ~ adstock(x)",
                "data": _frame(),
                "panel": PANEL,
                "initial": {"x": 1.0},
            },
            "has none",
        ),
        (
            {"data": _frame(), "panel": PANEL, "initial": {"x": 1.0}},
            r"Missing \['y'\]",
        ),
        (
            {"data": _frame(), "panel": PANEL, "initial": {"y": {"A": 1.0}}},
            r"Missing \['B'\]",
        ),
        (
            {"data": _frame(), "panel": PANEL, "initial": {"y": np.nan}},
            "finite",
        ),
        (
            {"data": _frame(), "panel": PANEL, "initial": {"y": True}},
            "finite number",
        ),
    ],
)
def test_model_rejects_bad_initial_before_compilation(kwargs, fragment):
    spec = kwargs.pop("spec", SPEC)
    with pytest.raises(ValueError, match=fragment):
        pathmc.model(spec, **kwargs)


def test_latent_initial_is_rejected():
    frame = _frame().drop(columns="y")
    with pytest.raises(ValueError, match="latent"):
        pathmc.model(
            "y ~ lag(eta)\neta ~ x",
            data=frame,
            panel=PANEL,
            latent=["eta"],
            initial={"eta": 1.0},
        )


def test_scaled_initial_is_stored_in_model_units():
    scaling = pathmc.Scaling(
        target={"method": "fixed", "value": {"A": 2.0, "B": 10.0}, "dims": ("unit",)}
    )
    model = pathmc.model(
        SPEC,
        data=_frame(),
        panel=PANEL,
        scaling=scaling,
        initial={"y": {"A": 4.0, "B": 30.0}},
    )
    baseline = model.initial_baseline.values["y"]
    assert list(baseline.coords["unit"].values) == ["A", "B"]
    np.testing.assert_allclose(baseline.values, [2.0, 3.0])
    assert model.initial_baseline.explicit is True


def test_do_override_matches_label_oracle_and_leaves_the_baseline():
    model = pathmc.model(
        SPEC,
        data=_frame(),
        panel=PANEL,
        initial={"y": {"A": 0.0, "B": 10.0}},
    )
    _fit_noise(model)
    before_nodes = _nodes(model)
    before = model.initial_baseline.values["y"].values.copy()

    baseline = model.do(simulate_over="time", kind="mean")
    # Free-running mean: t0 = 1 + 0.5*initial, t1 = 1 + 0.5*t0, then the
    # unit mean. A starts at 0 (1, 1.5); B starts at 10 (6, 4).
    np.testing.assert_allclose(baseline.dataset.y.values[:, 0, 0], [3.5, 2.75])

    override = {"y": {"B": 4.0, "A": 2.0}}
    shifted = model.do(simulate_over="time", kind="mean", initial=override)
    # A: 1 + 1 = 2, then 1 + 1 = 2. B: 1 + 2 = 3, then 1 + 1.5 = 2.5.
    np.testing.assert_allclose(shifted.dataset.y.values[:, 0, 0], [2.5, 2.25])
    again = model.do(simulate_over="time", kind="mean")
    xr.testing.assert_allclose(again.dataset.y, baseline.dataset.y)
    np.testing.assert_allclose(model.initial_baseline.values["y"].values, before)
    assert model.initial_baseline.explicit is True
    for name, values in _nodes(model).items():
        np.testing.assert_array_equal(values, before_nodes[name])

    compiled = model._compile_initial_override(
        model._normalize_query_initial(override, "time")
    )
    np.testing.assert_allclose(
        compiled._pathmc_panel_scan.initial_values["y"], [2.0, 4.0]
    )


def test_bad_do_initial_fails_before_sampling_and_does_not_mutate():
    model = pathmc.model(SPEC, data=_frame(), panel=PANEL, initial={"y": 1.0})
    _fit_noise(model)
    before = _nodes(model)
    with pytest.raises(ValueError, match="Missing \\['B'\\]"):
        model.do(simulate_over="time", initial={"y": {"A": 1.0}})
    with pytest.raises(ValueError, match="simulate_over='time'"):
        model.do(initial={"y": 1.0})
    for name, values in _nodes(model).items():
        np.testing.assert_array_equal(values, before[name])


def test_predictive_override_changes_the_path_without_mutating_nodes():
    model = pathmc.model(SPEC, data=_frame(), panel=PANEL, initial={"y": 0.0})
    _fit_noise(model, sigma=0.0)
    before = _nodes(model)
    baseline = model.do(simulate_over="time", kind="predictive", random_seed=3)
    shifted = model.do(
        simulate_over="time",
        kind="predictive",
        random_seed=3,
        initial={"y": {"A": 8.0, "B": 8.0}},
    )
    assert not np.allclose(baseline.dataset.y.values, shifted.dataset.y.values)
    for name, values in _nodes(model).items():
        np.testing.assert_array_equal(values, before[name])
    np.testing.assert_allclose(model.initial_baseline.values["y"].values, [0.0, 0.0])


def test_explicit_initial_shifts_only_the_first_period_logp():
    data = pd.DataFrame({"unit": ["u", "u"], "time": [0, 1], "y": [3.0, 4.0]})
    spec = "y ~ 0 + lag(y)"
    panel = {"unit": "unit", "time": "time"}
    low = pathmc.model(spec, data=data, panel=panel, initial={"y": 1.0})
    high = pathmc.model(spec, data=data, panel=panel, initial={"y": 2.0})
    beta = 0.5
    sigma = 1.0

    def point_logp(model):
        pm_model = model.pymc_model
        point = pm_model.initial_point()
        point["beta_y"] = np.array([beta])
        point["sigma_y_log__"] = np.array(np.log(sigma))
        logp = float(pm_model.compile_logp(vars=pm_model.observed_RVs)(point))
        gradient = pt.grad(
            pm_model.logp(vars=pm_model.observed_RVs),
            [pm_model.rvs_to_values[pm_model["beta_y"]]],
        )
        evaluate = pytensor.function(
            pm_model.value_vars, gradient, on_unused_input="ignore"
        )
        got = evaluate(*[point[var.name] for var in pm_model.value_vars])
        return logp, float(np.asarray(got).reshape(-1)[0])

    low_logp, low_grad = point_logp(low)
    high_logp, high_grad = point_logp(high)

    def period_logp(mu):
        residual = 3.0 - mu
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (residual / sigma) ** 2

    def period_grad(initial):
        return (3.0 - beta * initial) * initial / sigma**2

    assert low_logp - high_logp == pytest.approx(
        period_logp(beta * 1.0) - period_logp(beta * 2.0)
    )
    assert low_grad - high_grad == pytest.approx(period_grad(1.0) - period_grad(2.0))


def test_non_gaussian_initial_compiles_on_a_separate_model():
    frame = pd.DataFrame({
        "unit": ["A", "A", "B", "B"],
        "time": [1, 2, 1, 2],
        "y": [1, 0, 1, 1],
        "x": [0.0, 0.0, 0.0, 0.0],
    })
    model = pathmc.model(
        "y ~ 0 + x + lag(y)",
        data=frame,
        panel=PANEL,
        families={"y": "bernoulli"},
        initial={"y": {"A": 0.0, "B": 1.0}},
    )
    stored = model._gen_model
    np.testing.assert_allclose(
        stored._pathmc_panel_scan.initial_values["y"], [0.0, 1.0]
    )
    compiled = model._compile_initial_override(
        model._normalize_query_initial({"y": {"B": 0.0, "A": 1.0}}, "time")
    )
    assert compiled is not stored
    np.testing.assert_allclose(
        compiled._pathmc_panel_scan.initial_values["y"], [1.0, 0.0]
    )
    np.testing.assert_allclose(
        stored._pathmc_panel_scan.initial_values["y"], [0.0, 1.0]
    )


def test_cross_sectional_do_rejects_initial():
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]})
    model = pathmc.model("y ~ x", data=frame)
    model._idata = xr.DataTree.from_dict({
        "posterior": xr.Dataset(
            {
                "beta_y": (("chain", "draw", "beta_y_dim_0"), np.ones((1, 1, 2))),
                "sigma_y": (("chain", "draw"), np.ones((1, 1))),
            },
            coords={"chain": [0], "draw": [0], "beta_y_dim_0": [0, 1]},
        )
    })
    with pytest.raises(ValueError, match="initial="):
        model.do(initial={"y": 1.0})
    assert model.initial_baseline is None
