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
"""Tests for bs(), ns(), and ps() spline basis terms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from scipy.special import erfinv

import pathmc
from pathmc.basis import get_basis
from pathmc.exceptions import ParseError
from pathmc.parse import BasisCall, parse_spec
from pathmc.splines import (
    DEFAULT_TAU_SIGMA,
    effective_degrees_of_freedom,
    random_walk_map,
)

pytestmark = pytest.mark.filterwarnings("error")


def _frame(x: np.ndarray | None = None, n: int = 24) -> pd.DataFrame:
    if x is None:
        x = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": x,
        "z": rng.normal(size=len(x)),
        "w": rng.normal(size=len(x)),
        "y": rng.normal(size=len(x)),
    })


def _call(name: str, **params: object) -> BasisCall:
    return BasisCall(name, "x", params)


def test_spline_formulas_parse_and_render():
    """The DSL keeps df, knots, and a non-default penalty in the equation."""
    spec = parse_spec(
        "y ~ bs(x, df=5) + ns(z, knots=(0.2, 0.8)) + ps(w, df=4, order=1)"
    )
    rendered = [
        get_basis(term.basis.name).render(term.basis)
        for term in spec.regressions[0].terms
    ]
    assert rendered == [
        "bs(x, df=5)",
        "ns(z, knots=(0.2, 0.8))",
        "ps(w, df=4, order=1)",
    ]


@pytest.mark.parametrize(
    ("formula", "match"),
    [
        ("y ~ bs(x)", "df="),
        ("y ~ bs(x, df=5, knots=(0.2, 0.8))", "not both"),
        ("y ~ bs(x, df=1)", "df must be an integer"),
        ("y ~ ns(x, df=4, degree=2)", "cubic"),
        ("y ~ bs(x, df=4, order=1)", "only valid on ps"),
        ("y ~ ps(x, df=6, order=3)", "order must be 1 or 2"),
        ("y ~ ps(x, df=2)", "df must be an integer"),
        ("y ~ k*bs(x, df=4)", "coefficient prefix"),
        ("y ~ bs(x, df=4) + ps(x, df=5)", "more than one spline"),
        ("y ~ adstock(bs(x, df=4), decay=theta)", "directly"),
        ("y ~ bs(x, knots=(0.8, 0.2))", "strictly increasing"),
    ],
)
def test_invalid_spline_formulas_name_a_fix(formula: str, match: str):
    """A bad spline formula fails at parse time and says what to change."""
    with pytest.raises(ParseError, match=match):
        parse_spec(formula)


def test_centered_columns_match_df_and_drop_the_constant():
    """Training columns are mean-zero and one per requested coefficient."""
    x = np.linspace(10.0, 40.0, 40)
    for name, df in (("bs", 5), ("ns", 4), ("ps", 6)):
        columns, state = get_basis(name).build_data(x, _call(name, df=df))
        assert columns.shape == (len(x), df)
        np.testing.assert_allclose(columns.mean(axis=0), 0.0, atol=1e-10)
        assert state.boundary == (10.0, 40.0)
        assert len(state.interior) > 0


def test_explicit_knots_are_frozen_instead_of_quantiles():
    """knots= is the escape hatch and is stored as the interior knot tuple."""
    x = np.linspace(0.0, 1.0, 20)
    knots = (0.2, 0.5, 0.8)
    _, state = get_basis("bs").build_data(x, _call("bs", knots=knots))
    np.testing.assert_allclose(state.interior, knots)


def test_replay_keeps_training_knots_when_new_data_would_move_them():
    """A shifted input must not recompute quantile knots."""
    train = np.linspace(0.0, 1.0, 30)
    shifted = train + 10.0
    basis = get_basis("bs")
    call = _call("bs", df=5)
    _, frozen = basis.build_data(train, call)
    replayed, _ = basis.build_data(shifted, call, state=frozen)
    _, refit = basis.build_data(shifted, call)

    assert not np.allclose(frozen.interior, refit.interior)
    np.testing.assert_allclose(
        replayed, basis.build_data(shifted, call, state=frozen)[0]
    )
    refit_columns, _ = basis.build_data(shifted, call, state=refit)
    assert not np.allclose(replayed, refit_columns)


def test_ns_extrapolates_linearly_and_bs_does_not():
    """Outside the boundary knots, ns is linear and bs keeps a cubic bend."""
    train = np.linspace(0.0, 1.0, 40)
    outside = np.array([1.2, 1.4, 1.6])
    ns = get_basis("ns")
    ns_call = _call("ns", df=4)
    _, ns_state = ns.build_data(train, ns_call)
    ns_out, _ = ns.build_data(outside, ns_call, state=ns_state)
    second = ns_out[2] - 2.0 * ns_out[1] + ns_out[0]
    np.testing.assert_allclose(second, 0.0, atol=1e-8)

    bs = get_basis("bs")
    bs_call = _call("bs", df=4)
    _, bs_state = bs.build_data(train, bs_call)
    bs_out, _ = bs.build_data(outside, bs_call, state=bs_state)
    bs_second = bs_out[2] - 2.0 * bs_out[1] + bs_out[0]
    assert np.max(np.abs(bs_second)) > 1e-3


def test_spline_does_not_consume_a_plain_predictor_slot():
    """A spline owns its coefficient vector and leaves beta aligned."""
    model = pathmc.model("y ~ x + bs(z, df=4) + 2*w", data=_frame())
    gm = model._gen_model

    assert list(gm.coords["y_predictors"]) == ["Intercept", "x"]
    assert gm["beta_y"].eval().shape == (2,)
    assert gm["beta_bs_y_z"].eval().shape == (4,)
    assert "bs(z, df=4)" in str(model.equations())


def test_centered_spline_rejects_a_formula_without_an_intercept():
    """The centered basis cannot carry the response mean."""
    with pytest.raises(ValueError, match="Drop the '0 \\+' prefix"):
        pathmc.model("y ~ 0 + bs(x, df=4)", data=_frame())


def test_spline_rejects_panel_and_endogenous_inputs():
    """Splines are cross-sectional data bases, not graph or panel terms."""
    panel = pd.DataFrame({
        "region": ["a", "a", "b", "b"],
        "week": [1, 2, 1, 2],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.1, 0.2, 0.3, 0.4],
    })
    with pytest.raises(NotImplementedError, match="panel"):
        pathmc.model(
            "y ~ bs(x, df=3)",
            data=panel,
            panel={"unit": "region", "time": "week"},
        )

    data = _frame()
    data["m"] = data["x"]
    with pytest.raises(NotImplementedError, match="endogenous"):
        pathmc.model("m ~ x\ny ~ ns(m, df=3)", data=data)


def test_default_tau_prior_targets_a_partial_smooth_off_unit_scale():
    """Prior-median effective df sits between a line and the full basis."""
    x = np.linspace(10.0, 40.0, 80)
    scaled = 25.0 * x + 7.0
    median_tau = float(DEFAULT_TAU_SIGMA * np.sqrt(2.0) * erfinv(0.5))
    basis = get_basis("ps")
    for df in (4, 6, 8):
        columns, _ = basis.build_data(x, _call("ps", df=df))
        edf = effective_degrees_of_freedom(columns, median_tau)
        assert 2.0 < edf < df - 0.5
        scaled_columns, _ = basis.build_data(scaled, _call("ps", df=df))
        scaled_edf = effective_degrees_of_freedom(scaled_columns, median_tau)
        assert scaled_edf == pytest.approx(edf, abs=1e-6)


def test_second_order_penalty_leaves_level_and_slope_unpenalized():
    """Order 2 penalizes curvature, not a straight line in coefficient space."""
    mapping = random_walk_map(6, 2)
    second = np.diff(mapping, n=2, axis=0)
    np.testing.assert_allclose(second[:, :2], 0.0)
    assert np.linalg.matrix_rank(second[:, 2:]) == 4


def test_p_spline_introspection_names_the_random_walk_scale():
    """priors() names the innovation scale and the unpenalized coefficient prior."""
    text = str(pathmc.model("y ~ ps(x, df=5)", data=_frame()).priors())
    assert "tau_ps_y_x: HalfNormal(sigma=2" in text
    assert "beta_ps_y_x: Normal" in text
    assert "10." in text


def test_scaling_keeps_spline_knots_in_declared_units():
    """Channel scaling must not move the input the knots were placed on."""
    x = np.linspace(10.0, 40.0, 20)
    data = _frame(x)
    with pytest.warns(UserWarning, match="remain in their declared units"):
        model = pathmc.model(
            "y ~ x + bs(x, df=4)",
            data=data,
            scaling=pathmc.Scaling(channel={"method": "max"}),
        )
    np.testing.assert_array_equal(model._data["x"].to_numpy(), x)
    state = model._gen_model._pathmc_basis_states[("y", "bs", "x", None)]
    assert state.boundary == (float(x.min()), float(x.max()))


def test_predict_and_do_replay_frozen_knots(mock_pymc_sample, monkeypatch):
    """predict() and do() must fail if new data recomputes the knots."""
    train = np.linspace(0.0, 1.0, 12)
    shifted = np.linspace(10.0, 11.0, 12)
    data = _frame(train)
    model = pathmc.model("y ~ bs(x, df=4) + z", data=data)
    model.fit(draws=8, tune=8, chains=1, random_seed=0)

    basis = get_basis("bs")
    call = _call("bs", df=4)
    frozen = model._pymc_model._pathmc_basis_states[("y", "bs", "x", None)]
    frozen_columns, _ = basis.build_data(shifted, call, state=frozen)
    refit_columns, refit = basis.build_data(shifted, call)
    assert not np.allclose(frozen.interior, refit.interior)

    seen: dict[str, np.ndarray] = {}
    original = pm.sample_posterior_predictive

    def _capture(*args, **kwargs):
        seen["columns"] = np.asarray(model._pymc_model["basis_y_bs_x"].get_value())
        return original(*args, **kwargs)

    monkeypatch.setattr(pm, "sample_posterior_predictive", _capture)
    model.predict(data=data.assign(x=shifted), progressbar=False)
    np.testing.assert_allclose(seen["columns"], frozen_columns)
    assert not np.allclose(seen["columns"], refit_columns)

    posterior = model._idata.posterior.copy(deep=True)
    posterior["beta_y"].loc[{"y_predictors": "Intercept"}] = 0.0
    posterior["beta_y"].loc[{"y_predictors": "z"}] = 0.0
    posterior["beta_bs_y_x"] = posterior["beta_bs_y_x"] * 0.0 + 1.0
    posterior["sigma_y"] = posterior["sigma_y"] * 0.0 + 1e-8
    model._idata["posterior"] = posterior
    with pytest.warns(UserWarning, match="outside"):
        intervened = model.do(set={"x": shifted}, kind="mean", average_units=False)

    expected = frozen_columns.sum(axis=1)
    actual = np.asarray(intervened.dataset["y"].mean(("chain", "draw")))
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert not np.allclose(actual, refit_columns.sum(axis=1), atol=1e-3)


@pytest.mark.slow
def test_sampled_spline_recovers_a_known_smooth():
    """A short sample of bs() and ps() follows a planted non-unit-scale curve."""
    rng = np.random.default_rng(1)
    x = np.linspace(10.0, 40.0, 60)
    truth = 1.5 + 0.8 * np.sin((x - 10.0) / 4.0)
    y = truth + rng.normal(scale=0.05, size=x.size)
    data = pd.DataFrame({"x": x, "y": y})
    for formula in ("y ~ bs(x, df=6)", "y ~ ps(x, df=6)"):
        model = pathmc.model(formula, data=data)
        idata = model.fit(
            draws=200,
            tune=200,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=1,
        )
        mu = idata.posterior["mu_y"].mean(("chain", "draw")).values
        assert np.corrcoef(mu, truth)[0, 1] > 0.95
