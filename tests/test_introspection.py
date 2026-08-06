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
"""M5 gate tests: model introspection.

These methods should work BEFORE sampling — they describe model
structure, not posterior results. All tests are fast.
"""

import re

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc
from pathmc.introspect import _latexify_name

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC

# Two bare underscores inside one subscript group (e.g. \beta_{hsgp_y_x}) are
# a TeX double-subscript error: MathJax aborts the whole aligned block and
# dumps raw source. Generated names should carry no bare underscore at all —
# suffix tokens become comma-separated indices — so assert the stricter form.
_DOUBLE_SUBSCRIPT = re.compile(r"_\{[^{}]*_[^{}]*\}")


class TestGraph:
    def test_graph_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        g = model.graph()
        assert g is not None

    def test_graph_for_larger_model(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        g = model.graph()
        assert g is not None


class TestEquations:
    def test_equations_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        assert eqs is not None

    def test_equations_mentions_endogenous(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        text = str(eqs)
        assert "M" in text
        assert "Y" in text

    def test_equations_mentions_predictors(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        text = str(eqs)
        assert "X" in text


class TestDesignIntrospection:
    def test_design_returns_columns(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        design = model.design("M")
        assert hasattr(design, "columns")

    def test_design_for_each_endogenous(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        for var in ["M", "Y"]:
            design = model.design(var)
            assert design is not None
            assert len(design.columns) > 0


class TestPriors:
    def test_priors_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        priors = model.priors()
        assert priors is not None

    def test_priors_mentions_equations(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        priors = model.priors()
        text = str(priors)
        assert "M" in text or "Y" in text


class TestPyMCModelAccess:
    def test_pymc_model_accessible(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        assert isinstance(model.pymc_model, pm.Model)


class TestLatexifyName:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("sigma", r"\sigma"),
            ("theta_tv", r"\theta_{tv}"),
            ("b_tv", "b_{tv}"),
            ("eta_y_x", r"\eta_{y,x}"),
            ("ell_y_x", r"\ell_{y,x}"),
            ("beta_hsgp_y_x", r"\beta_{hsgp,y,x}"),
            ("mu_slope_sales_tv", r"\mu_{slope,sales,tv}"),
            ("slope_demand_price", "slope_{demand,price}"),
        ],
    )
    def test_latexify_name(self, name, expected):
        assert _latexify_name(name) == expected

    @pytest.mark.parametrize(
        "name", ["beta_hsgp_y_x", "mu_slope_sales_tv", "sigma_slope_demand_price"]
    )
    def test_no_double_subscript(self, name):
        assert not _DOUBLE_SUBSCRIPT.search(_latexify_name(name))


class TestLatexRendering:
    @pytest.fixture
    def hsgp_model(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 50)
        df = pd.DataFrame({"x": x, "y": np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 50)})
        return pathmc.model("y ~ hsgp(x, m=20, c=1.5)", data=df)

    def test_hsgp_equations_latex_is_valid(self, hsgp_model):
        latex = hsgp_model.equations()._repr_latex_()
        assert not _DOUBLE_SUBSCRIPT.search(latex), latex

    def test_hsgp_priors_latex_is_valid(self, hsgp_model):
        latex = hsgp_model.priors()._repr_latex_()
        assert not _DOUBLE_SUBSCRIPT.search(latex), latex
