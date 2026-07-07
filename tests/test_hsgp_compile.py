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
"""Compile-level tests for the ``hsgp()`` term (fast; no sampling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def cross_sectional_df():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 5.0, 40)
    return pd.DataFrame({
        "x": x,
        "y": rng.normal(size=x.size),
        "z": rng.normal(size=x.size),
    })


def _gen_model(model):
    model._compile()
    return model._gen_model


def test_hsgp_model_compiles(cross_sectional_df):
    model = pathmc.model("y ~ hsgp(x, m=15, c=1.5)", data=cross_sectional_df)
    gm = _gen_model(model)
    names = {rv.name for rv in gm.free_RVs}
    assert {"ell_y_x", "eta_y_x", "beta_hsgp_y_x"} <= names
    assert gm["beta_hsgp_y_x"].eval().shape == (15,)
    assert gm["f_y_x"].eval().shape == (len(cross_sectional_df),)


def test_hsgp_input_absent_from_predictor_coord(cross_sectional_df):
    model = pathmc.model("y ~ hsgp(x, m=8, c=1.5)", data=cross_sectional_df)
    gm = _gen_model(model)
    # Only the intercept is a plain predictor; 'x' must not appear as a
    # spurious beta column.
    assert list(gm.coords["y_predictors"]) == ["Intercept"]
    assert list(gm.coords["y_x_hsgp"]) == list(range(8))


def test_hsgp_only_regression_has_no_beta(cross_sectional_df):
    model = pathmc.model("y ~ 0 + hsgp(x, m=8, c=1.5)", data=cross_sectional_df)
    gm = _gen_model(model)
    names = {rv.name for rv in gm.free_RVs}
    assert "beta_y" not in names
    assert "beta_hsgp_y_x" in names


@pytest.mark.parametrize("cov", ["expquad", "matern52", "matern32"])
def test_each_cov_compiles(cross_sectional_df, cov):
    model = pathmc.model(
        f"y ~ hsgp(x, m=8, c=1.5, cov='{cov}')", data=cross_sectional_df
    )
    gm = _gen_model(model)
    assert "f_y_x" in gm.named_vars


@pytest.mark.parametrize("centered", ["false", "true"])
def test_both_parametrizations_compile(cross_sectional_df, centered):
    model = pathmc.model(
        f"y ~ hsgp(x, m=8, c=1.5, centered={centered})", data=cross_sectional_df
    )
    gm = _gen_model(model)
    assert "f_y_x" in gm.named_vars


def test_explicit_L_compiles(cross_sectional_df):
    model = pathmc.model("y ~ hsgp(x, m=8, L=6.0)", data=cross_sectional_df)
    gm = _gen_model(model)
    assert gm["f_y_x"].eval().shape == (len(cross_sectional_df),)


def test_priors_listing_includes_hsgp_hyperpriors(cross_sectional_df):
    model = pathmc.model("y ~ hsgp(x, m=8, c=1.5)", data=cross_sectional_df)
    keys = set(model.priors()._entries)
    assert {"ell_y_x", "eta_y_x", "beta_hsgp_y_x"} <= keys


def test_equations_render_hsgp(cross_sectional_df):
    model = pathmc.model("y ~ hsgp(x, m=8, c=1.5)", data=cross_sectional_df)
    text = str(model.equations())
    assert "f_hsgp(x)" in text


def test_panel_model_with_hsgp_raises(cross_sectional_df):
    df = cross_sectional_df.copy()
    df["u"] = np.repeat([0, 1], len(df) // 2)
    df["t"] = list(range(len(df) // 2)) * 2
    with pytest.raises(NotImplementedError, match="panel"):
        pathmc.model(
            "y ~ hsgp(x, m=8, c=1.5)", data=df, panel={"unit": "u", "time": "t"}
        )._compile()


def test_hsgp_in_residual_block_raises(cross_sectional_df):
    with pytest.raises(NotImplementedError, match="residual-covariance"):
        pathmc.model(
            "y ~ hsgp(x, m=8, c=1.5)\ny ~~ z", data=cross_sectional_df
        )._compile()


def test_simulate_with_hsgp_raises(cross_sectional_df):
    with pytest.raises(NotImplementedError, match="hsgp"):
        pathmc.simulate("y ~ hsgp(x, m=8, c=1.5)", data=cross_sectional_df, params={})
