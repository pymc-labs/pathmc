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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathmc import Scaling
from pathmc._model import (
    PathModel,
    _invert_generated_columns,
    _scale_scalar_intervention,
    _scale_set_for_bounds,
    _unscale_predict_groups,
)
from pathmc.effects import build_effects_summary, compute_path_effect
from pathmc.interpret import _column_in_business_units, _to_frame
from pathmc.parse import parse_spec
from pathmc.scaling import ScalingFactors, fit_scaling
from pathmc.simulate import (
    _business_exog_value,
    _scale_cross_section_intervention,
    _scale_scan_intervention,
    _to_business_units,
    _unscale_do_dataset,
)


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


@pytest.mark.parametrize("kind", ["elasticity", "derived"])
def test_dimensionless_kinds_are_identity_and_idempotent(kind):
    factors = ScalingFactors(factors={"X": ((), {(): 10.0})})
    values = np.array([1.0, 2.0])

    once = factors.to_business(values, kind=kind)
    twice = factors.to_business(once, kind=kind)

    np.testing.assert_array_equal(once, values)
    np.testing.assert_array_equal(twice, once)


PUBLIC_METHOD_UNITS = {
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
}


def test_every_public_pathmodel_method_declares_its_unit_contract():
    public_methods = {
        name
        for name, value in PathModel.__dict__.items()
        if not name.startswith("_") and inspect.isfunction(value)
    }
    assert set(PUBLIC_METHOD_UNITS) == public_methods


@pytest.mark.parametrize(
    ("boundary", "seam"),
    [
        (_scale_set_for_bounds, "to_internal"),
        (_scale_scalar_intervention, "to_internal"),
        (_unscale_predict_groups, "to_business"),
        (_invert_generated_columns, "to_business"),
        (_to_business_units, "to_business"),
        (_unscale_do_dataset, "to_business"),
        (_scale_cross_section_intervention, "to_internal"),
        (_scale_scan_intervention, "to_internal"),
        (_business_exog_value, "to_business"),
        (build_effects_summary, "to_business"),
        (compute_path_effect, "to_business"),
        (_to_frame, "to_internal"),
        (_column_in_business_units, "to_business"),
        (PathModel.datagrid, "to_business"),
    ],
)
def test_scaling_boundary_delegates_to_the_conversion_seam(boundary, seam):
    assert seam in inspect.getsource(boundary)


def test_factor_arithmetic_does_not_escape_scaling_module():
    package = Path(__file__).parents[1] / "pathmc"
    forbidden = (
        "._per_row(",
        ".mean_factor(",
        ".inverse_transform_column(",
        ".unscale_xarray(",
    )
    leaks: list[str] = []
    for path in package.glob("*.py"):
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
