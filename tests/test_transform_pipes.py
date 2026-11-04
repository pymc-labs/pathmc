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
"""Pipe syntax lowers to the existing transform expression contract."""

import pytest
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

import pathmc
from pathmc.basis import REGISTRY as BASIS_REGISTRY
from pathmc.basis import Basis, register_basis
from pathmc.exceptions import DuplicateEquationError, ParseError
from pathmc.parse import parse_spec
from pathmc.scaling import Scaling
from pathmc.transforms import REGISTRY as TRANSFORM_REGISTRY
from pathmc.transforms import Transform, register_transform


@pytest.mark.parametrize(
    ("pipe", "nested"),
    [
        ("Y ~ x |> f(p=u)", "Y ~ f(x, p=u)"),
        ("Y ~ (x |> f())", "Y ~ f(x)"),
        ("Y ~ a*(x |> f(p=u) |> g(q=v))", "Y ~ a*g(f(x, p=u), q=v)"),
        ("Y ~ -2*(x |> f(p=u))", "Y ~ -2*f(x, p=u)"),
        ("Y ~ 0*(x |> f(p=u))", "Y ~ 0*f(x, p=u)"),
        ("Y ~ 1e-5*(x |> f(p=u))", "Y ~ 1e-5*f(x, p=u)"),
        (
            "Y ~ adstock |> f(p=u) |> g(q=v) |> h(r=w)",
            "Y ~ h(g(f(adstock, p=u), q=v), r=w)",
        ),
        (
            "Y ~ 0 + x|>f(p=u,) + b*(z |> g(q=v)) + h(w); Z ~ c*t\nY ~~ Z\ntotal := b*c",
            "Y ~ 0 + f(x, p=u) + b*g(z, q=v) + h(w); Z ~ c*t\nY ~~ Z\ntotal := b*c",
        ),
        (
            "Y ~ a*(x\r\n |> f(\r\n p=u,\r\n ))\r\n + z",
            "Y ~ a*f(x, p=u) + z",
        ),
        ("Y ~ x |> f(p=-0.5)", "Y ~ f(x, p=-0.5)"),
    ],
)
def test_pipe_and_nested_forms_have_equal_full_specs(pipe, nested):
    assert parse_spec(pipe) == parse_spec(nested)


@pytest.mark.parametrize(
    "rhs",
    [
        "a*x |> f()",
        "2*x |> f()",
        "|> f()",
        "x |>",
        "x |> |> f()",
        "x |> f",
        "x |> f(x, p=u)",
        "x |> f(p=u, p=v)",
        "x |> f(p=)",
        "x |> f() trailing",
        "a*(x |> f() + z)",
        "x:z |> f()",
        "(x |> f()):z",
        "x |> lag()",
        "x |> hsgp(m=20)",
        "x |> C()",
        "(a*x |> f())",
        "(a*x) |> f()",
        "(x) |> f()",
        "((x |> f()))",
        "(x |> f()) |> g()",
        "f(x) |> g()",
        "g(x |> f())",
        "x |> f(_)",
        "x |> f(p=u,,q=v)",
        "x |> f(p=u) )",
        "x | > f()",
        "x ||> f()",
        "x |> f(p='u')",
        "x |> f(p=u.v)",
        "x |> f(p=g(u))",
    ],
)
def test_malformed_pipes_are_parse_errors(rhs):
    with pytest.raises(ParseError) as exc:
        parse_spec(f"Y ~ {rhs}")
    assert "pipe" in str(exc.value).lower() or "pipeline" in str(exc.value).lower()


@pytest.mark.parametrize("spec", ["Y ~ x\n |> f()", "Y ~ x |>\n f()"])
def test_ungrouped_multiline_pipe_is_rejected(spec):
    with pytest.raises(ParseError, match="group"):
        parse_spec(spec)


@pytest.mark.parametrize("spec", ["x |> f() ~ y", "Y ~~ x |> f()", "total := a |> f()"])
def test_pipe_outside_regression_rhs_is_rejected(spec):
    with pytest.raises(ParseError, match="pipe|Pipeline|pipeline"):
        parse_spec(spec)


def test_pipe_preserves_duplicate_label_error():
    with pytest.raises(DuplicateEquationError):
        parse_spec("Y ~ a*(x |> f()) + a*z")


def test_quoted_pipe_in_categorical_reference_is_legacy_content():
    categorical = (
        parse_spec("Y ~ C(group, reference='a|>b')").regressions[0].terms[0].categorical
    )
    assert categorical is not None
    assert categorical.reference == "a|>b"


class _Identity(Transform):
    name = "pipe_identity_test"
    param_specs = {}
    homogeneous = True

    def apply_pymc(self, x, params, *, panel_info=None, data=None):
        return x


class _Basis(Basis):
    name = "pipe_basis_test"

    def n_basis(self, call):
        return 1

    def build_data(self, x, call, *, state=None):
        return np.asarray(x)[:, None], None


def test_registered_parameter_free_transform_and_custom_basis_boundary():
    old_transform = TRANSFORM_REGISTRY.get(_Identity.name)
    old_basis = BASIS_REGISTRY.get(_Basis.name)
    try:
        register_transform(_Identity())
        register_basis(_Basis())
        assert parse_spec("Y ~ x |> pipe_identity_test()") == parse_spec(
            "Y ~ pipe_identity_test(x)"
        )
        with pytest.raises(ParseError, match="standalone"):
            parse_spec("Y ~ x |> pipe_basis_test()")
    finally:
        if old_transform is None:
            TRANSFORM_REGISTRY.pop(_Identity.name, None)
        else:
            TRANSFORM_REGISTRY[_Identity.name] = old_transform
        if old_basis is None:
            BASIS_REGISTRY.pop(_Basis.name, None)
        else:
            BASIS_REGISTRY[_Basis.name] = old_basis


def test_unknown_transform_resolves_at_existing_model_boundary():
    pipe = "Y ~ x |> unknown_pipe_test(p=u)"
    nested = "Y ~ unknown_pipe_test(x, p=u)"
    assert parse_spec(pipe) == parse_spec(nested)
    data = pd.DataFrame({"x": [0.0, 1.0], "Y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Register custom transforms") as pipe_exc:
        pathmc.model(pipe, data=data)
    with pytest.raises(ValueError) as nested_exc:
        pathmc.model(nested, data=data)
    assert str(pipe_exc.value) == str(nested_exc.value)


PANEL = {"unit": "unit", "time": "time"}
PIPE = "Y ~ 2*(x |> adstock(decay=theta) |> logistic_saturation(lam=lam))"
NESTED = "Y ~ 2*logistic_saturation(adstock(x, decay=theta), lam=lam)"


def _panel_data():
    values = {"A": [1.0, 0.0, 2.0, 0.0], "B": [0.0, 2.0, 0.0, 1.0]}
    return pd.DataFrame([
        {"unit": unit, "time": time, "x": x, "Y": 0.0}
        for unit, series in values.items()
        for time, x in enumerate(series)
    ])


def _oracle(values):
    stock = 0.0
    result = []
    for x in values:
        stock = x + 0.5 * stock
        sat = (1 - np.exp(-0.8 * stock)) / (1 + np.exp(-0.8 * stock))
        result.append(0.3 + 2.0 * sat)
    return np.asarray(result)


def test_pipe_panel_simulation_matches_independent_order_and_coefficient_oracle():
    frame = _panel_data()
    params = {"beta_Y": [0.3], "sigma_Y": 0.0, "theta": 0.5, "lam": 0.8}
    expected = np.concatenate([
        _oracle(frame.loc[frame.unit == unit, "x"].to_numpy()) for unit in ("A", "B")
    ])
    for spec in (PIPE, NESTED):
        result = pathmc.simulate(
            spec, data=frame, params=params, panel=PANEL, random_seed=0
        )
        np.testing.assert_allclose(result["Y"], expected, atol=1e-8)
    x = np.array([1.0, 0.0, 2.0, 0.0])
    sat_then_stock = []
    stock = 0.0
    for value in x:
        saturated_input = (1 - np.exp(-0.8 * value)) / (1 + np.exp(-0.8 * value))
        stock = saturated_input + 0.5 * stock
        sat_then_stock.append(0.3 + 2.0 * stock)
    correct = _oracle(x)
    assert not np.allclose(correct, sat_then_stock)
    stock = 0.0
    coefficient_inside = []
    for value in x:
        stock = value + 0.5 * stock
        coefficient_inside.append(
            0.3 + (1 - np.exp(-0.8 * 2.0 * stock)) / (1 + np.exp(-0.8 * 2.0 * stock))
        )
    assert not np.allclose(correct, coefficient_inside)


def test_pipe_model_metadata_and_time_forward_do_match_nested_oracle():
    frame = _panel_data()
    models = [pathmc.model(spec, data=frame, panel=PANEL) for spec in (PIPE, NESTED)]
    pipe, nested = models
    assert pipe._spec == nested._spec
    assert pipe._graph_info.topological_order == nested._graph_info.topological_order
    assert set(pipe._graph_info._dag.nodes) == set(nested._graph_info._dag.nodes)
    assert set(pipe._graph_info._dag.edges) == set(nested._graph_info._dag.edges)
    assert pipe.pymc_model.coords == nested.pymc_model.coords
    assert {rv.name for rv in pipe.pymc_model.free_RVs} == {
        rv.name for rv in nested.pymc_model.free_RVs
    }
    assert list(pipe.pymc_model.coords["Y_predictors"]) == ["Intercept"]
    pd.testing.assert_frame_equal(pipe.design("Y"), nested.design("Y"))
    assert str(pipe.equations()) == str(nested.equations())
    assert repr(pipe.priors()) == repr(nested.priors())
    assert pipe.graph().source == nested.graph().source

    posterior = xr.Dataset(
        {
            "beta_Y": (("chain", "draw", "Y_predictors"), np.full((1, 2, 1), 0.3)),
            "theta": (("chain", "draw"), np.full((1, 2), 0.5)),
            "lam": (("chain", "draw"), np.full((1, 2), 0.8)),
            "sigma_Y": (("chain", "draw"), np.zeros((1, 2))),
        },
        coords={"chain": [0], "draw": [0, 1], "Y_predictors": ["Intercept"]},
    )
    for model in models:
        compiled = pm.do(
            model._gen_model,
            {
                "beta_Y": np.array([0.3]),
                "theta": np.array(0.5),
                "lam": np.array(0.8),
                "sigma_Y": np.array(0.0),
            },
        )
        expected_by_unit = np.stack(
            [
                _oracle(frame.loc[frame.unit == unit, "x"].to_numpy())
                for unit in ("A", "B")
            ],
            axis=1,
        )
        np.testing.assert_allclose(
            pm.draw(compiled["mu_Y"], draws=1, random_seed=0),
            expected_by_unit,
            atol=1e-8,
        )
        model._idata = xr.DataTree.from_dict({"posterior": posterior.copy(deep=True)})
        natural = model.do(set=None, simulate_over="time", kind="mean")
        forced = model.do(set={"x": 1.0}, simulate_over="time", kind="mean")
        np.testing.assert_allclose(
            natural.by_time("Y").mean(axis=1), expected_by_unit.mean(axis=1), atol=1e-8
        )
        forced_expected = _oracle(np.ones(4))
        np.testing.assert_allclose(
            forced.by_time("Y").mean(axis=1), forced_expected, atol=1e-8
        )


def test_distinct_and_shared_chains_keep_their_panel_state():
    frame = _panel_data()
    pipe = (
        "M ~ 0 + x |> adstock(decay=t1) |> logistic_saturation(lam=l1)\n"
        "Y ~ 0 + x |> adstock(decay=t1) |> logistic_saturation(lam=l1) "
        "+ x |> adstock(decay=t2) |> logistic_saturation(lam=l2)"
    )
    nested = (
        "M ~ 0 + logistic_saturation(adstock(x, decay=t1), lam=l1)\n"
        "Y ~ 0 + logistic_saturation(adstock(x, decay=t1), lam=l1) "
        "+ logistic_saturation(adstock(x, decay=t2), lam=l2)"
    )
    assert parse_spec(pipe) == parse_spec(nested)
    params = {
        "beta_M": [1.0],
        "sigma_M": 0.0,
        "beta_Y": [1.0, 1.0],
        "sigma_Y": 0.0,
        "t1": 0.5,
        "t2": 0.2,
        "l1": 0.8,
        "l2": 0.4,
    }
    for spec in (pipe, nested):
        result = pathmc.simulate(spec, data=frame, params=params, panel=PANEL)
        expected_m = []
        expected_y = []
        for unit in ("A", "B"):
            x = frame.loc[frame.unit == unit, "x"].to_numpy()
            first = _oracle(x) - 0.3
            stock = 0.0
            second = []
            for value in x:
                stock = value + 0.2 * stock
                second.append((1 - np.exp(-0.4 * stock)) / (1 + np.exp(-0.4 * stock)))
            expected_m.extend(first / 2)
            expected_y.extend(first / 2 + np.asarray(second))
        np.testing.assert_allclose(result["M"], expected_m, atol=1e-8)
        np.testing.assert_allclose(result["Y"], expected_y, atol=1e-8)


def test_nonunit_input_scaling_has_same_pipe_and_nested_semantics():
    frame = _panel_data()
    scale = Scaling(channel={"method": "fixed", "value": 10.0})
    params = {"beta_Y": [0.3], "sigma_Y": 0.0, "theta": 0.5, "lam": 0.8}
    outputs = [
        pathmc.simulate(spec, data=frame, params=params, panel=PANEL, scaling=scale)[
            "Y"
        ].to_numpy()
        for spec in (PIPE, NESTED)
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
    expected = np.concatenate([
        _oracle(frame.loc[frame.unit == unit, "x"].to_numpy() / 10)
        for unit in ("A", "B")
    ])
    np.testing.assert_allclose(outputs[0], expected, atol=1e-8)
