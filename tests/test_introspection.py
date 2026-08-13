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
from pathmc.introspect import (
    _latex_escape,
    _latex_index,
    _latex_symbol,
    _latexify_name,
    _latexify_prior,
)

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC

# Two bare underscores inside one subscript group (e.g. \beta_{hsgp_y_x}) are
# a TeX double-subscript error: MathJax aborts the whole aligned block and
# dumps raw source. Generated names should carry no bare underscore at all —
# suffix tokens become comma-separated indices — so assert the stricter form.
_DOUBLE_SUBSCRIPT = re.compile(r"_\{[^{}]*_[^{}]*\}")

# The invariant that actually covers every emission site: every underscore in
# generated LaTeX is either escaped (``\_``) or opens a subscript group
# (``_{``). A bare ``_`` elsewhere is a subscript operator applied to whatever
# character precedes it — silently wrong inside ``\mathrm{sales_q}``, a hard
# KaTeX parse error inside ``\text{}``, and a double-subscript error as soon
# as a second one lands on the same atom. _DOUBLE_SUBSCRIPT only inspects
# ``_{...}`` groups, so it never sees the ``\mathrm{}``/``\text{}`` cases.
_BARE_UNDERSCORE = re.compile(r"(?<!\\)_(?!\{)")

# Variable names an actual user writes: multi-token, and in the lagged/indexed
# case (``sales_q_1``) a verbatim spelling is a hard KaTeX error.
_UNDERSCORED_SPEC = "birth_weight ~ tv_spend + sales_q_1"


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

    def test_hsgp_latex_has_no_bare_underscore(self, hsgp_model):
        for latex in (
            hsgp_model.equations()._repr_latex_(),
            hsgp_model.priors()._repr_latex_(),
        ):
            assert not _BARE_UNDERSCORE.search(latex), latex


class TestLatexEscaping:
    """Variable names reach LaTeX verbatim; underscores in them must be safe.

    ``_latexify_name`` only sees generated parameter names. Response and
    predictor names are interpolated straight into ``\\mu_{...}``,
    ``\\mathrm{...}`` and ``\\text{...}`` groups, so they need escaping at
    those sites too.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("tv", "tv"),
            ("tv_spend", "tv,spend"),
            ("sales_q_1", "sales,q,1"),
        ],
    )
    def test_latex_index(self, name, expected):
        assert _latex_index(name) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("tv", r"\mathrm{tv}"),
            ("tv_spend", r"\mathrm{tv\_spend}"),
            ("sales_q_1", r"\mathrm{sales\_q\_1}"),
        ],
    )
    def test_latex_symbol(self, name, expected):
        assert _latex_symbol(name) == expected

    def test_latex_escape_leaves_plain_text_alone(self):
        assert _latex_escape("Normal(0, 10)") == "Normal(0, 10)"

    @pytest.mark.parametrize(
        "prior_str",
        [
            "Normal(mu_alpha, sigma_alpha)",
            "LKJCholeskyCov(eta=2, sd_dist=HalfNormal(1))",
            "some_unparseable_prior",
        ],
    )
    def test_latexify_prior_escapes_underscores(self, prior_str):
        assert not _BARE_UNDERSCORE.search(_latexify_prior(prior_str))

    @pytest.mark.parametrize(
        "family", ["gaussian", "bernoulli", "poisson", "negbinomial", "studentt"]
    )
    def test_underscored_names_render_safely(self, family):
        # Data-free: equations() and priors() describe structure, and every
        # family emits a different likelihood line with lhs in a subscript.
        model = pathmc.model(_UNDERSCORED_SPEC, families={"birth_weight": family})
        for latex in (
            model.equations()._repr_latex_(),
            model.priors()._repr_latex_(),
        ):
            assert not _DOUBLE_SUBSCRIPT.search(latex), latex
            assert not _BARE_UNDERSCORE.search(latex), latex

    def test_underscored_transform_renders_safely(self):
        model = pathmc.model("birth_weight ~ logistic_saturation(tv_spend, lam=lam_tv)")
        latex = model.equations()._repr_latex_()
        assert not _BARE_UNDERSCORE.search(latex), latex
        assert not _DOUBLE_SUBSCRIPT.search(latex), latex
