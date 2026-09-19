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
"""Categorical predictor compilation, state replay, and intervention tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.parse import parse_spec


@pytest.fixture
def categorical_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    region = np.tile(["north", "south", "west"], 30)
    x = rng.normal(size=region.size)
    fixed = rng.normal(size=region.size)
    effects = {"north": 0.0, "south": 2.0, "west": -1.0}
    y = 1.0 + x + fixed + np.array([effects[value] for value in region])
    y += rng.normal(scale=0.2, size=region.size)
    return pd.DataFrame({"y": y, "region": region, "x": x, "fixed": fixed})


def test_explicit_categorical_parse():
    term = (
        parse_spec("y ~ C(region, reference='south', prior='hierarchical')")
        .regressions[0]
        .terms[0]
    )
    assert term.categorical is not None
    assert term.categorical.variable == "region"
    assert term.categorical.reference == "south"
    assert term.categorical.prior == "hierarchical"


def test_string_column_is_inferred_and_reference_is_frozen(categorical_data):
    model = pathmc.model("y ~ region", data=categorical_data)
    design = model.design("y")
    assert list(design.columns) == [
        "Intercept",
        "region[T.south]",
        "region[T.west]",
    ]
    text = str(model.equations())
    assert "reference='north'" in text
    assert "levels=['north', 'south', 'west']" in text


def test_pandas_category_order_excludes_unused_levels():
    data = pd.DataFrame({
        "y": [1.0, 2.0, 3.0, 4.0],
        "region": pd.Categorical(
            ["south", "west", "south", "west"],
            categories=["north", "south", "west"],
        ),
    })
    model = pathmc.model("y ~ region", data=data)
    assert list(model.design("y").columns) == ["Intercept", "region[T.west]"]

    with pytest.raises(ValueError, match="region.*unseen.*north"):
        with model._prediction_data(pd.DataFrame({"region": ["north"] * len(data)})):
            pass
    with pytest.raises(ValueError, match="region.*unseen.*north"):
        with model._categorical_intervention({"region": "north"}):
            pass


def test_explicit_reference_controls_design(categorical_data):
    model = pathmc.model("y ~ C(region, reference='south')", data=categorical_data)
    assert list(model.design("y").columns) == [
        "Intercept",
        "region[T.north]",
        "region[T.west]",
    ]


def test_explicit_categorical_requires_data_column():
    data = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(
        ValueError,
        match="Categorical predictor 'region'.*not found.*Add a 'region' column",
    ):
        pathmc.model("y ~ C(region)", data=data)


def test_hierarchical_prior_is_declared(categorical_data):
    model = pathmc.model("y ~ C(region, prior='hierarchical')", data=categorical_data)
    prior_names = set(model.priors()._entries)
    assert {
        "mu_beta_y_region",
        "sigma_beta_y_region",
        "beta_y_region",
    } <= prior_names
    rv_names = {rv.name for rv in model.pymc_model.free_RVs}
    assert prior_names - {"beta_y", "sigma_y"} <= rv_names


def test_beta_alignment_with_plain_and_fixed_terms(categorical_data):
    model = pathmc.model(
        "y ~ x + region + 1*fixed",
        data=categorical_data,
    )
    assert list(model.pymc_model.coords["y_predictors"]) == ["Intercept", "x"]
    assert model.pymc_model["beta_y"].eval().shape == (2,)
    assert model.pymc_model["beta_y_region"].eval().shape == (2,)


@pytest.mark.slow
def test_samples_recovers_level_effects_and_do_accepts_label(categorical_data):
    model = pathmc.model("y ~ region", data=categorical_data)
    model.fit(progressbar=False, random_seed=8, compute_log_likelihood=False)
    south = float(model.do(set={"region": "south"}).mean("y"))
    north = float(model.do(set={"region": "north"}).mean("y"))
    west = float(model.do(set={"region": "west"}).mean("y"))
    assert south - north > 1.0
    assert west - north < -0.4


def test_do_replays_fitted_categorical_basis(categorical_data):
    model = pathmc.model("y ~ region", data=categorical_data)
    basis = model._gen_model["_cat_y_region"]
    original = basis.get_value().copy()

    with model._categorical_intervention({"region": "west"}):
        np.testing.assert_array_equal(
            basis.get_value(),
            np.tile([0.0, 1.0], (len(categorical_data), 1)),
        )
    np.testing.assert_array_equal(basis.get_value(), original)

    with model._categorical_intervention({"region": "north"}):
        np.testing.assert_array_equal(
            basis.get_value(),
            np.zeros((len(categorical_data), 2)),
        )


@pytest.mark.slow
def test_predict_replays_fit_levels_in_new_row_order(categorical_data):
    model = pathmc.model("y ~ region", data=categorical_data)
    model.fit(progressbar=False, random_seed=9, compute_log_likelihood=False)
    new_data = pd.DataFrame({
        "region": np.resize(["west", "north", "south"], len(categorical_data))
    })
    basis = model.pymc_model["_cat_y_region"]
    original = basis.get_value().copy()
    with model._prediction_data(new_data):
        np.testing.assert_array_equal(
            basis.get_value()[:3],
            [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        )
    np.testing.assert_array_equal(basis.get_value(), original)

    predicted = model.predict(
        data=new_data,
        extend_inferencedata=False,
        progressbar=False,
    )
    assert predicted.posterior_predictive["y"].shape[-1] == len(categorical_data)


def test_unseen_levels_name_variable_and_level(categorical_data):
    model = pathmc.model("y ~ region", data=categorical_data)
    with pytest.raises(ValueError, match="region.*unseen.*central"):
        with model._prediction_data(
            pd.DataFrame({"region": ["central"] * len(categorical_data)})
        ):
            pass


def test_simulate_uses_frozen_categorical_design():
    exog = pd.DataFrame({"region": ["north", "south", "west"]})
    simulated = pathmc.simulate(
        "y ~ region",
        data=exog,
        params={
            "beta_y": [1.0],
            "beta_y_region": [2.0, -1.0],
            "sigma_y": 1e-6,
        },
        random_seed=4,
    )
    np.testing.assert_allclose(simulated["y"], [1.0, 3.0, 0.0], atol=1e-4)
