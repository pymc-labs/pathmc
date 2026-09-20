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
"""Unit-boundary contracts for every numeric PathModel surface."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc import Scaling
from pathmc._model import PathModel, _layout_unscaled_column
from pathmc.parse import parse_spec
from pathmc.scaling import ScaleContext, ScalingFactors, fit_scaling
from pathmc.simulate import _business_exog_value


def _nw(frame: pd.DataFrame):
    import narwhals.stable.v1 as nw

    return nw.from_native(frame, eager_only=True)


@pytest.fixture
def heterogeneous_factors() -> tuple[ScalingFactors, pd.DataFrame]:
    frame = pd.DataFrame({
        "geo": ["large", "small"],
        "X": [20.0, 4.0],
        "Z": [10.0, 2.0],
        "Y": [200.0, 40.0],
    })
    factors = ScalingFactors(
        factors={
            "X": (("geo",), {("large",): 10.0, ("small",): 2.0}),
            "Z": (("geo",), {("large",): 5.0, ("small",): 1.0}),
            "Y": (("geo",), {("large",): 100.0, ("small",): 20.0}),
        },
        roles={
            "X": frozenset({"regressor"}),
            "Z": frozenset({"regressor"}),
            "Y": frozenset({"outcome"}),
        },
    )
    return factors, frame


@pytest.mark.parametrize("kind,column", [("regressor", "X"), ("outcome", "Y")])
def test_pandas_frame_conversion_is_role_aware_and_idempotent(
    heterogeneous_factors, kind, column
):
    factors, frame = heterogeneous_factors

    once = factors.to_internal(frame, kind=kind)
    twice = factors.to_internal(once, kind=kind)

    np.testing.assert_allclose(once[column], [2.0, 2.0])
    pd.testing.assert_frame_equal(twice, once)
    untouched = "Y" if kind == "regressor" else "X"
    np.testing.assert_allclose(once[untouched], frame[untouched])


def test_role_metadata_does_not_fall_back_when_kind_has_no_columns(
    heterogeneous_factors,
):
    factors, frame = heterogeneous_factors
    outcome_only = ScalingFactors(
        factors=factors.factors,
        roles={"Y": frozenset({"outcome"})},
    )

    converted = outcome_only.to_internal(frame, kind="regressor")

    pd.testing.assert_frame_equal(converted, frame)


def test_narwhals_frame_conversion_is_idempotent(heterogeneous_factors):
    factors, frame = heterogeneous_factors
    business = _nw(frame)

    internal = factors.to_internal(business, kind="regressor")
    repeated_internal = factors.to_internal(internal, kind="regressor")
    recovered = factors.to_business(internal, kind="regressor")
    repeated_business = factors.to_business(recovered, kind="regressor")

    np.testing.assert_allclose(internal["X"].to_numpy(), [2.0, 2.0])
    np.testing.assert_allclose(repeated_internal["X"].to_numpy(), [2.0, 2.0])
    np.testing.assert_allclose(recovered["X"].to_numpy(), frame["X"])
    np.testing.assert_allclose(repeated_business["X"].to_numpy(), frame["X"])


def test_frame_idempotence_is_shared_across_semantic_roles():
    frame = pd.DataFrame({"Y": [8.0, 8.0]})
    factors = ScalingFactors(
        factors={"Y": ((), {(): 2.0})},
        roles={"Y": frozenset({"outcome", "regressor"})},
    )

    once = factors.to_internal(frame, kind="regressor")
    cross_role = factors.to_internal(once, kind="outcome")

    pd.testing.assert_frame_equal(cross_role, once)
    np.testing.assert_allclose(cross_role["Y"], [4.0, 4.0])


def test_manual_frame_factors_require_roles_or_explicit_columns():
    frame = pd.DataFrame({"X": [10.0, 20.0]})
    factors = ScalingFactors(factors={"X": ((), {(): 10.0})})

    with pytest.raises(ValueError, match="requires semantic roles"):
        factors.to_internal(frame, kind="regressor")

    converted = factors.to_internal(
        frame,
        kind="regressor",
        dims=ScaleContext(columns=("X",)),
    )
    np.testing.assert_allclose(converted["X"], [1.0, 2.0])


def test_series_array_and_scalar_round_trip_and_idempotence(heterogeneous_factors):
    factors, frame = heterogeneous_factors
    dims = {"term": "X", "data": _nw(frame)}

    for business in (frame["X"], frame["X"].to_numpy()):
        internal = factors.to_internal(business, kind="regressor", dims=dims)
        repeated = factors.to_internal(internal, kind="regressor", dims=dims)
        recovered = factors.to_business(internal, kind="regressor", dims=dims)
        np.testing.assert_allclose(repeated, internal)
        np.testing.assert_allclose(recovered, frame["X"])

    global_factors = ScalingFactors(factors={"X": ((), {(): 10.0})})
    internal_scalar = global_factors.to_internal(
        20.0, kind="regressor", dims={"term": "X"}
    )
    repeated_scalar = global_factors.to_internal(
        internal_scalar, kind="regressor", dims={"term": "X"}
    )
    assert internal_scalar == repeated_scalar == 2.0
    assert (
        global_factors.to_business(
            internal_scalar, kind="regressor", dims={"term": "X"}
        )
        == 20.0
    )


def test_numpy_conversion_state_survives_arithmetic_and_slicing(
    heterogeneous_factors,
):
    factors, frame = heterogeneous_factors
    dims = ScaleContext(term="X", data=_nw(frame))
    converted = factors.to_internal(frame["X"].to_numpy(), kind="regressor", dims=dims)

    for derived in (converted * 1.0, converted[:]):
        repeated = factors.to_internal(derived, kind="regressor", dims=dims)
        np.testing.assert_allclose(repeated, derived)
    assert getattr(converted[:1], "_pathmc_scale_units", None)


def test_conversion_state_is_stable_across_equivalent_factor_instances():
    first = ScalingFactors(factors={"X": ((), {(): 10.0})})
    second = ScalingFactors(factors={"X": ((), {(): 10.0})})
    converted = first.to_internal(
        np.array([10.0, 20.0]),
        kind="regressor",
        dims=ScaleContext(term="X"),
    )

    repeated = second.to_internal(
        converted,
        kind="regressor",
        dims=ScaleContext(term="X"),
    )

    np.testing.assert_allclose(repeated, [1.0, 2.0])


def test_conversion_state_distinguishes_different_resolved_alignment():
    factors = ScalingFactors(factors={"X": (("geo",), {("a",): 2.0, ("b",): 4.0})})
    forward = _nw(pd.DataFrame({"geo": ["a", "b"]}))
    reverse = _nw(pd.DataFrame({"geo": ["b", "a"]}))
    converted = factors.to_internal(
        np.array([8.0, 8.0]),
        kind="regressor",
        dims=ScaleContext(term="X", data=forward),
    )

    realigned = factors.to_internal(
        converted,
        kind="regressor",
        dims=ScaleContext(term="X", data=reverse),
    )

    np.testing.assert_allclose(realigned, [1.0, 1.0])


def test_row_factors_reject_same_size_but_misaligned_array():
    factors = ScalingFactors(factors={"X": (("geo",), {("a",): 1.0, ("b",): 10.0})})
    frame = _nw(pd.DataFrame({"geo": ["a", "a", "b", "b"]}))

    with pytest.raises(ValueError, match="equal element count"):
        factors.to_internal(
            np.ones((2, 2)),
            kind="regressor",
            dims=ScaleContext(term="X", data=frame),
        )


def test_scan_factors_reject_transposed_same_size_array():
    factors = ScalingFactors(
        factors={"X": (("geo",), {("a",): 1.0, ("b",): 10.0, ("c",): 100.0})}
    )
    frame = _nw(pd.DataFrame({"geo": ["a", "a", "b", "b", "c", "c"]}))
    scan = SimpleNamespace(sort_idx=np.arange(6), n_units=3, n_times=2)

    with pytest.raises(ValueError, match=r"expected shape \(2, 3\)"):
        factors.to_internal(
            np.ones((3, 2)),
            kind="regressor",
            dims=ScaleContext(term="X", data=frame, scan_info=scan),
        )


def test_observed_layout_rejects_same_size_shape_coincidence():
    template = xr.DataArray(np.ones((2, 2)), dims=("chain", "row"))

    with pytest.raises(ValueError, match="Shapes must match exactly"):
        _layout_unscaled_column(np.arange(4.0), template, None)


def test_xarray_factors_align_by_named_dimension(heterogeneous_factors):
    factors, _frame = heterogeneous_factors
    internal = xr.DataArray(
        [1.0, 2.0],
        dims=["geo"],
        coords={"geo": ["small", "large"]},
        name="Y",
    )

    business = factors.to_business(internal, kind="outcome")
    repeated = factors.to_business(business, kind="outcome")

    np.testing.assert_allclose(business, [20.0, 200.0])
    xr.testing.assert_identical(repeated, business)


@pytest.mark.parametrize(
    ("term_key", "expected"),
    [
        ("Intercept", 60.0),  # Intercept: F_Y
        (0, 10.0),  # linear X: F_Y / F_X
        (1, 60.0 / 18.0),  # interaction X:Z: F_Y / (F_X F_Z)
        (2, 60.0),  # saturation is dimensionless: F_Y only
    ],
)
def test_coefficient_factor_resolution_is_ast_aware(
    heterogeneous_factors, term_key, expected
):
    factors, frame = heterogeneous_factors
    spec = parse_spec("Y ~ b*X + c*X:Z + d*logistic_saturation(X, lam=lam_x)")
    term = (
        term_key if isinstance(term_key, str) else spec.regressions[0].terms[term_key]
    )

    converted = factors.to_business(
        1.0,
        kind="coefficient",
        dims={"term": term, "outcome": "Y", "data": _nw(frame)},
    )

    assert converted == pytest.approx(expected)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Y ~ b*adstock(X, decay=theta)", 10.0),
        ("Y ~ b*delayed_adstock(X, decay=theta, theta=delay)", 10.0),
        ("Y ~ b*weibull_adstock(X, lam=lam_x, k=k_x)", 10.0),
        ("Y ~ b*logistic_saturation(X, lam=lam_x)", 60.0),
        ("Y ~ b*michaelis_menten(X, alpha=alpha_x, lam=lam_x)", 60.0),
        (
            "Y ~ b*adstock(logistic_saturation(X, lam=lam_x), decay=theta)",
            60.0,
        ),
        ("Y ~ b*lag(X)", 10.0),
        ("Y ~ hsgp(X, m=5, c=1.5)", 60.0),
    ],
)
def test_every_builtin_term_shape_has_an_executable_coefficient_oracle(
    heterogeneous_factors, formula, expected
):
    factors, frame = heterogeneous_factors
    term = parse_spec(formula).regressions[0].terms[0]

    converted = factors.to_business(
        1.0,
        kind="coefficient",
        dims={"term": term, "outcome": "Y", "data": _nw(frame)},
    )

    assert converted == pytest.approx(expected)


def test_grouped_exogenous_fill_converts_rows_before_reducing():
    factors = ScalingFactors(
        factors={
            "X": (("geo",), {("large",): 10.0, ("small",): 20.0}),
        },
        roles={"X": frozenset({"regressor"})},
    )
    internal = _nw(pd.DataFrame({"geo": ["large", "small"], "X": [1.0, 5.0]}))

    full = _business_exog_value("X", internal, None, factors)
    subgroup = _business_exog_value("X", internal, np.array([0]), factors)

    assert full == pytest.approx(55.0)
    assert subgroup == pytest.approx(10.0)


@pytest.mark.parametrize("kind", ["elasticity", "derived"])
def test_dimensionless_kinds_are_identity_and_idempotent(kind):
    factors = ScalingFactors(factors={"X": ((), {(): 10.0})})
    values = np.array([1.0, 2.0])

    once = factors.to_business(values, kind=kind)
    twice = factors.to_business(once, kind=kind)

    np.testing.assert_array_equal(once, values)
    np.testing.assert_array_equal(twice, once)


@dataclass(frozen=True)
class PublicMethodContract:
    """Declared units and executable contract case for one public method."""

    units: str
    executable_case: str | None = None


_EXECUTABLE_NUMERIC_METHODS = frozenset({
    "effects_summary",
    "standardized",
    "effect",
    "predict",
    "refute_placebo",
    "do",
    "counterfactual",
    "ate",
    "cate",
    "att",
    "atu",
    "sensitivity",
    "prob",
    "predictions",
    "comparisons",
    "slopes",
    "datagrid",
})


def _contract(units: str, name: str) -> PublicMethodContract:
    executable = name if name in _EXECUTABLE_NUMERIC_METHODS else None
    return PublicMethodContract(units=units, executable_case=executable)


PUBLIC_METHOD_CONTRACTS = {
    name: _contract(units, name)
    for name, units in {
        "to_graphviz": "structural",
        "design": "internal",
        "graph": "structural",
        "equations": "structural",
        "priors": "structural",
        "set_priors": "mutation",
        "sample_prior_predictive": "internal",
        "summary": "internal",
        "effects_summary": "business",
        "standardized": "dimensionless",
        "effect": "business",
        "fit": "internal",
        "predict": "business",
        "latent_trajectory": "internal",
        "adjustment_sets": "structural",
        "is_identifiable": "dimensionless",
        "adjustment_model": "structural",
        "frontdoor_identifiable": "dimensionless",
        "collider_warnings": "structural",
        "implied_independences": "structural",
        "test_implications": "dimensionless",
        "falsify": "dimensionless",
        "refute_placebo": "business-and-dimensionless",
        "do": "business",
        "counterfactual": "business",
        "ate": "business",
        "cate": "business",
        "att": "business",
        "atu": "business",
        "sensitivity": "business-and-dimensionless",
        "prob": "dimensionless",
        "predictions": "business",
        "comparisons": "business-diff-or-dimensionless-ratio",
        "slopes": "kind-dependent",
        "datagrid": "business",
    }.items()
}


def test_every_public_pathmodel_method_declares_its_unit_contract():
    public_methods = {
        name
        for name, value in PathModel.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert set(PUBLIC_METHOD_CONTRACTS) == public_methods


def test_every_business_facing_numeric_contract_is_executable():
    required_units = {
        "business",
        "business-and-dimensionless",
        "business-diff-or-dimensionless-ratio",
        "kind-dependent",
    }
    required = {
        name
        for name, contract in PUBLIC_METHOD_CONTRACTS.items()
        if contract.units in required_units
    }
    executable = {
        contract.executable_case
        for contract in PUBLIC_METHOD_CONTRACTS.values()
        if contract.executable_case is not None
    }

    assert required <= executable


def test_factor_arithmetic_does_not_escape_scaling_module():
    package = Path(__file__).parents[1] / "pathmc"
    forbidden = (
        "._per_row(",
        ".mean_factor(",
        ".inverse_transform_column(",
        ".unscale_xarray(",
        ".factors",
    )
    leaks: list[str] = []
    for path in package.rglob("*.py"):
        if path.name == "scaling.py":
            continue
        text = path.read_text()
        for token in forbidden:
            if token in text:
                leaks.append(f"{path.name}: {token}")
    assert not leaks, "Factor arithmetic escaped the scaling seam: " + ", ".join(leaks)


def test_fit_scaling_records_semantic_roles():
    frame = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0]})
    factors = fit_scaling(
        Scaling(
            target={"method": "max"},
            channel={"method": "max"},
        ),
        _nw(frame),
        target_columns={"Y"},
        channel_columns={"X"},
    )

    assert factors.roles == {
        "X": frozenset({"regressor"}),
        "Y": frozenset({"outcome"}),
    }


@dataclass(frozen=True)
class EquivalentModels:
    """A scaled fit and its manually parameterized business-unit twin."""

    scaled: PathModel
    unscaled: PathModel


def _pin_linear_posterior(model: PathModel, *, scaled: bool) -> None:
    posterior = model._idata["posterior"].dataset.copy(deep=True)
    values = (
        {"Intercept": 1.0, "T": 2.0, "X": 0.3}
        if scaled
        else {"Intercept": 100.0, "T": 20.0, "X": 3.0}
    )
    for predictor, value in values.items():
        posterior["beta_Y"].loc[{"Y_predictors": predictor}] = value
    posterior["sigma_Y"] = posterior["sigma_Y"] * 0
    model._idata["posterior"].dataset = posterior


@pytest.fixture(scope="module")
def equivalent_models(mock_pymc_sample_module) -> EquivalentModels:
    """Models implementing Y = 100 + 20 T + 3 X in the same raw units."""
    treatment = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    covariate = np.arange(10.0, 70.0, 10.0)
    frame = pd.DataFrame({
        "T": treatment,
        "X": covariate,
        "Y": 100.0 + 20.0 * treatment + 3.0 * covariate,
    })
    scaled = pathmc.model(
        "Y ~ b*T + c*X",
        data=frame,
        scaling=Scaling(
            target={"method": "fixed", "value": 100.0},
            channel={"method": "fixed", "value": 10.0},
        ),
    )
    unscaled = pathmc.model("Y ~ b*T + c*X", data=frame)
    scaled.fit()
    unscaled.fit()
    _pin_linear_posterior(scaled, scaled=True)
    _pin_linear_posterior(unscaled, scaled=False)
    return EquivalentModels(scaled=scaled, unscaled=unscaled)


def _result_draws(result, variable: str = "Y") -> np.ndarray:
    return np.asarray(result.draws(variable), dtype=float)


def _effects_summary(model: PathModel) -> np.ndarray:
    return model.effects_summary().loc[["b", "c"], ["mean", "sd"]].to_numpy()


def _standardized(model: PathModel) -> np.ndarray:
    return model.standardized().loc[["b", "c"], ["mean", "sd"]].to_numpy()


def _effect(model: PathModel) -> np.ndarray:
    return np.asarray(model.effect("T -> Y").draws, dtype=float)


def _predict(model: PathModel) -> np.ndarray:
    predicted = model.predict(
        random_seed=42,
        progressbar=False,
        extend_inferencedata=False,
    )
    return np.asarray(predicted["posterior_predictive"]["Y"], dtype=float)


def _refute_placebo(model: PathModel) -> np.ndarray:
    result = model.refute_placebo("Y", "T", n_permutations=2, random_seed=42)
    return np.asarray(result.observed_ate_draws, dtype=float)


def _do(model: PathModel) -> np.ndarray:
    return _result_draws(model.do(set={"T": 1.0}, kind="mean"))


def _counterfactual(model: PathModel) -> np.ndarray:
    result = model.counterfactual(
        evidence={"T": 1.0, "X": 20.0, "Y": 180.0},
        do={"T": 0.0},
    )
    return _result_draws(result)


def _ate(model: PathModel) -> np.ndarray:
    return _result_draws(model.ate("Y", "T", kind="mean"))


def _cate(model: PathModel) -> np.ndarray:
    return _result_draws(model.cate("Y", "T", condition={"X": 20.0}, kind="mean"))


def _att(model: PathModel) -> np.ndarray:
    return _result_draws(model.att("Y", "T", kind="mean"))


def _atu(model: PathModel) -> np.ndarray:
    return _result_draws(model.atu("Y", "T", kind="mean"))


def _sensitivity(model: PathModel) -> np.ndarray:
    result = model.sensitivity("Y", "T", n_grid=3, kind="mean")
    return np.concatenate([
        np.asarray(result.observed_ate_draws).ravel(),
        np.asarray(result.adjusted_ate_mean).ravel(),
    ])


def _prob(model: PathModel) -> np.ndarray:
    return np.asarray([model.prob("Y > 150", set={"T": 1.0}, kind="mean")])


def _predictions(model: PathModel) -> np.ndarray:
    result = model.predictions("Y", newdata=pd.DataFrame({"T": [1.0], "X": [20.0]}))
    return np.asarray(result.dataset["Y"], dtype=float)


def _comparisons(model: PathModel) -> np.ndarray:
    return _result_draws(model.comparisons("Y", "T", contrast=(0.0, 1.0)))


def _slopes(model: PathModel) -> np.ndarray:
    return _result_draws(model.slopes("Y", "X"))


def _datagrid(model: PathModel) -> np.ndarray:
    return model.datagrid(T=[0.0, 1.0]).select_dtypes(include="number").to_numpy()


SurfaceCase = Callable[[PathModel], np.ndarray]
BUSINESS_SURFACE_CASES: dict[str, SurfaceCase] = {
    "effects_summary": _effects_summary,
    "standardized": _standardized,
    "effect": _effect,
    "predict": _predict,
    "refute_placebo": _refute_placebo,
    "do": _do,
    "counterfactual": _counterfactual,
    "ate": _ate,
    "cate": _cate,
    "att": _att,
    "atu": _atu,
    "sensitivity": _sensitivity,
    "prob": _prob,
    "predictions": _predictions,
    "comparisons": _comparisons,
    "slopes": _slopes,
    "datagrid": _datagrid,
}


@pytest.mark.parametrize(
    "method_name",
    sorted(_EXECUTABLE_NUMERIC_METHODS),
)
def test_public_numeric_surface_matches_equivalent_unscaled_fit(
    method_name,
    equivalent_models,
    monkeypatch,
):
    """Every registered numeric surface executes the same independent oracle."""
    assert set(BUSINESS_SURFACE_CASES) == _EXECUTABLE_NUMERIC_METHODS
    if method_name == "predict":
        import pymc as pm

        business = np.asarray(
            100.0
            + 20.0 * np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
            + 3.0 * np.arange(10.0, 70.0, 10.0)
        )

        def fake_posterior_predictive(idata, **kwargs):
            values = (
                business / 100.0
                if idata is equivalent_models.scaled._idata
                else business
            )
            dataset = xr.Dataset({
                "Y": xr.DataArray(
                    values.reshape(1, 1, -1),
                    dims=("chain", "draw", "Y_dim_0"),
                )
            })
            tree = xr.DataTree()
            tree["posterior_predictive"] = xr.DataTree(dataset)
            return tree

        monkeypatch.setattr(
            pm, "sample_posterior_predictive", fake_posterior_predictive
        )
    elif method_name == "refute_placebo":
        import pathmc.refute as refute

        def fake_permutation(model, outcome, treatment, values, seed, sample_kwargs):
            draws = model.ate(outcome, treatment, values=values, kind="mean").draws()
            return float(np.mean(draws) * 0.05), max(float(np.std(draws)), 1.0)

        def fake_hierarchical(fold_means, fold_sds, sample_kwargs, random_seed):
            zeros = np.zeros(20, dtype=float)
            return zeros, np.ones_like(zeros), zeros

        monkeypatch.setattr(refute, "_permute_and_refit", fake_permutation)
        monkeypatch.setattr(refute, "_fit_hierarchical_null", fake_hierarchical)

    case = BUSINESS_SURFACE_CASES[method_name]
    scaled = case(equivalent_models.scaled)
    expected = case(equivalent_models.unscaled)

    np.testing.assert_allclose(scaled, expected, rtol=1e-6, atol=1e-6)


def test_scaled_placebo_clone_preserves_the_business_unit_boundary(
    equivalent_models,
):
    source = equivalent_models.scaled
    clone = source._refit_permuted("T", 42, {})

    assert clone.fitted_scaling is None
    np.testing.assert_allclose(
        clone._data["X"].to_numpy(), source._data["X"].to_numpy()
    )
    np.testing.assert_allclose(
        clone._data["Y"].to_numpy(), source._data["Y"].to_numpy()
    )
    np.testing.assert_allclose(
        np.sort(clone._data["T"].to_numpy()),
        np.sort(source._data["T"].to_numpy()),
    )


def test_placebo_fold_queries_internal_clone_through_business_boundary(
    equivalent_models,
    monkeypatch,
):
    from copy import copy

    from pathmc.refute import _permute_and_refit

    source = equivalent_models.scaled
    clone = copy(source)
    clone._scaling_factors = None
    monkeypatch.setattr(
        source,
        "_refit_permuted",
        lambda treatment, seed, sample_kwargs: clone,
    )

    mean, sd = _permute_and_refit(source, "Y", "T", (0.0, 1.0), 42, {})

    assert mean == pytest.approx(20.0)
    assert sd == pytest.approx(0.0)
    assert clone.fitted_scaling is None


def _pin_beta(model: PathModel, outcome: str, values: dict[str, float]) -> None:
    posterior = model._idata["posterior"].dataset.copy(deep=True)
    coordinate = f"{outcome}_predictors"
    for predictor, value in values.items():
        posterior[f"beta_{outcome}"].loc[{coordinate: predictor}] = value
    if f"sigma_{outcome}" in posterior:
        posterior[f"sigma_{outcome}"] = posterior[f"sigma_{outcome}"] * 0
    model._idata["posterior"].dataset = posterior


def test_grouped_public_do_matches_equivalent_unscaled_fit(mock_pymc_sample):
    factors_x = {"large": 10.0, "small": 20.0}
    factors_y = {"large": 100.0, "small": 200.0}
    frame = pd.DataFrame({
        "geo": ["large", "large", "small", "small"],
        "X": [10.0, 20.0, 20.0, 40.0],
    })
    frame["Y"] = 20.0 * frame["X"]
    fitted = ScalingFactors(
        factors={
            "X": (("geo",), {(key,): value for key, value in factors_x.items()}),
            "Y": (("geo",), {(key,): value for key, value in factors_y.items()}),
        },
        roles={
            "X": frozenset({"regressor"}),
            "Y": frozenset({"outcome"}),
        },
    )
    scaled = pathmc.model(
        "Y ~ 0 + b*X",
        data=frame,
        scaling=fitted,
    )
    unscaled = pathmc.model("Y ~ 0 + b*X", data=frame)
    scaled.fit()
    unscaled.fit()
    _pin_beta(scaled, "Y", {"X": 2.0})
    _pin_beta(unscaled, "Y", {"X": 20.0})

    actual = scaled.do(set={"X": 20.0}, kind="mean").draws("Y")
    expected = unscaled.do(set={"X": 20.0}, kind="mean").draws("Y")

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual, 400.0)


def test_scan_public_do_matches_equivalent_unscaled_fit(mock_pymc_sample):
    factors_x = {"large": 10.0, "small": 20.0}
    factors_y = {"large": 100.0, "small": 200.0}
    rows = []
    for geo, factor_x in factors_x.items():
        for time, internal_x in enumerate([1.0, 2.0, 3.0]):
            raw_x = internal_x * factor_x
            rows.append({
                "geo": geo,
                "time": time,
                "X": raw_x,
                "Y": 20.0 * raw_x,
            })
    frame = pd.DataFrame(rows)
    panel = {"unit": "geo", "time": "time"}
    spec = "Y ~ 0 + b*X + r*lag(Y)"
    scaled = pathmc.model(
        spec,
        data=frame,
        panel=panel,
        scaling=Scaling(
            target={"method": "divide", "by": factors_y, "dims": ("geo",)},
            channel={"method": "divide", "by": factors_x, "dims": ("geo",)},
        ),
    )
    unscaled = pathmc.model(spec, data=frame, panel=panel)
    scaled.fit()
    unscaled.fit()
    _pin_beta(scaled, "Y", {"X": 2.0, "lag(Y)": 0.5})
    _pin_beta(unscaled, "Y", {"X": 20.0, "lag(Y)": 0.5})

    actual = scaled.do(set={"X": 20.0}, simulate_over="time", kind="mean")
    expected = unscaled.do(set={"X": 20.0}, simulate_over="time", kind="mean")

    xr.testing.assert_allclose(actual.dataset["Y"], expected.dataset["Y"])
    assert scaled.effects_summary().loc["b", "mean"] == pytest.approx(20.0)
    assert scaled.effects_summary().loc["r", "mean"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("Y ~ b*X", 20.0),
        ("Y ~ b*adstock(X, decay=theta)", 20.0),
        ("Y ~ b*logistic_saturation(X, lam=lam_x)", 200.0),
        ("Y ~ b*X:Z", 2.0),
    ],
)
def test_public_effects_summary_term_shape_oracles(
    mock_pymc_sample,
    formula,
    expected,
):
    frame = pd.DataFrame({
        "X": np.arange(10.0, 70.0, 10.0),
        "Z": np.arange(60.0, 0.0, -10.0),
        "Y": np.arange(100.0, 700.0, 100.0),
    })
    model = pathmc.model(
        formula,
        data=frame,
        scaling=Scaling(
            target={"method": "fixed", "value": 100.0},
            channel={"method": "fixed", "value": 10.0},
        ),
    )
    model.fit()
    posterior = model._idata["posterior"].dataset.copy(deep=True)
    posterior["beta_Y"] = posterior["beta_Y"] * 0 + 2.0
    model._idata["posterior"].dataset = posterior

    assert model.effects_summary().loc["b", "mean"] == pytest.approx(expected)


def test_public_defined_parameter_uses_business_coefficients(mock_pymc_sample):
    x = np.arange(10.0, 70.0, 10.0)
    frame = pd.DataFrame({"X": x, "M": 2.0 * x, "Y": 6.0 * x})
    model = pathmc.model(
        "M ~ a*X\nY ~ b*M\nindirect := a*b",
        data=frame,
        scaling=Scaling(
            target={"method": "fixed", "value": 100.0},
            channel={"method": "fixed", "value": 10.0},
        ),
    )
    model.fit()
    _pin_beta(model, "M", {"Intercept": 0.0, "X": 2.0})
    _pin_beta(model, "Y", {"Intercept": 0.0, "M": 3.0})

    summary = model.effects_summary()

    assert summary.loc["a", "mean"] == pytest.approx(20.0)
    assert summary.loc["b", "mean"] == pytest.approx(3.0)
    assert summary.loc["indirect", "mean"] == pytest.approx(60.0)


def test_public_hsgp_do_obeys_the_business_boundary(mock_pymc_sample):
    from copy import copy

    x = np.arange(10.0, 70.0, 10.0)
    frame = pd.DataFrame({"X": x, "Y": 100.0 + 10.0 * np.sin(x / 10.0)})
    model = pathmc.model(
        "Y ~ hsgp(X, m=4, c=1.5)",
        data=frame,
        scaling=Scaling(
            target={"method": "fixed", "value": 100.0},
            channel={"method": "fixed", "value": 10.0},
        ),
    )
    model.fit()
    internal_view = copy(model)
    internal_view._scaling_factors = None

    business = model.do(set={"X": 30.0}, kind="mean").draws("Y")
    internal = internal_view.do(set={"X": 3.0}, kind="mean").draws("Y")

    np.testing.assert_allclose(business, internal * 100.0)
