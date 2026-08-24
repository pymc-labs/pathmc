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
"""Tests for pathmc.simulate_params_template() — parameter discovery for simulate()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def mediation_df():
    """Placeholder with one real exogenous column."""
    return pd.DataFrame({"X": np.zeros(10)})


@pytest.fixture
def panel_df():
    """Balanced placeholder panel: 3 units x 4 times."""
    rows = []
    for region in ["North", "South", "East"]:
        for week in range(1, 5):
            rows.append({"region": region, "week": week, "spend": 0.0})
    return pd.DataFrame(rows)


def fill_from_template(template, rng=None):
    """Build a valid ``params`` dict from a template's descriptors."""
    rng = rng or np.random.default_rng(0)
    params = {}
    for name, desc in template.items():
        if desc["kind"] == "scalar":
            params[name] = 1.0
        else:
            params[name] = rng.normal(size=desc["shape"]).tolist()
    return params


class TestTemplateBasic:
    """Cross-sectional specs report every simulate() parameter."""

    def test_mediation_template_keys(self, mediation_df):
        template = pathmc.simulate_params_template(
            "M ~ X\nY ~ M + X", data=mediation_df
        )
        assert sorted(template) == ["beta_M", "beta_Y", "sigma_M", "sigma_Y"]

    def test_descriptor_contents(self, mediation_df):
        template = pathmc.simulate_params_template(
            "M ~ X\nY ~ M + X", data=mediation_df
        )
        # Y has intercept + M + X coefficients.
        assert template["beta_Y"] == {
            "kind": "vector",
            "shape": (3,),
            "dtype": "float64",
        }
        assert template["sigma_M"] == {
            "kind": "scalar",
            "shape": (),
            "dtype": "float64",
        }

    def test_missing_exogenous_column_raises(self):
        """Mirrors the KeyError model()/simulate() raise on missing exog columns."""
        with pytest.raises(KeyError):
            pathmc.simulate_params_template("Y ~ X", data=pd.DataFrame({"Z": [0.0]}))


class TestTemplatePooling:
    """pooling='partial' reports unit-indexed hierarchical parameters."""

    def test_partial_pooling_shapes(self, panel_df):
        template = pathmc.simulate_params_template(
            "sales ~ 0 + spend",
            data=panel_df,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        n_units = panel_df["region"].nunique()
        assert template["alpha_sales"]["shape"] == (n_units,)
        assert template["alpha_sales"]["kind"] == "vector"
        assert template["mu_alpha_sales"]["kind"] == "scalar"
        assert template["sigma_alpha_sales"]["kind"] == "scalar"


class TestTemplatePanelRequired:
    """lag() specs without panel= raise the same clear error as model()."""

    def test_lag_without_panel_raises(self, panel_df):
        with pytest.raises(ValueError, match="lag\\(\\) terms require a panel"):
            pathmc.simulate_params_template("sales ~ lag(spend)", data=panel_df)


class TestTemplateTransforms:
    """Transform parameters appear under their spec-declared names."""

    def test_adstock_theta(self, mediation_df):
        template = pathmc.simulate_params_template(
            "Y ~ adstock(X, decay=theta)", data=mediation_df
        )
        assert template["theta"]["kind"] == "scalar"
        assert "beta_Y" in template

    def test_saturation_lam(self, mediation_df):
        template = pathmc.simulate_params_template(
            "Y ~ logistic_saturation(X, lam=lam_x)", data=mediation_df
        )
        assert "lam_x" in template


class TestTemplateHSGP:
    """hsgp() terms report basis weights and hyperparameters."""

    def test_hsgp_shapes(self):
        df = pd.DataFrame({"x": np.zeros(20)})
        m = 8
        template = pathmc.simulate_params_template(
            f"y ~ hsgp(x, m={m}, c=1.5)", data=df
        )
        assert template["beta_hsgp_y_x"]["shape"] == (m,)
        assert template["ell_y_x"]["kind"] == "scalar"
        assert template["eta_y_x"]["kind"] == "scalar"


class TestTemplateResidualCovariance:
    """~~ blocks report the packed Cholesky vector."""

    def test_chol_shape(self, mediation_df):
        template = pathmc.simulate_params_template(
            "Y1 ~ X\nY2 ~ X\nY1 ~~ Y2", data=mediation_df
        )
        k = 2
        assert template["chol_Y1_Y2"] == {
            "kind": "vector",
            "shape": (k * (k + 1) // 2,),
            "dtype": "float64",
        }
