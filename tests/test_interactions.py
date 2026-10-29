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
"""Tests for interaction term support (X:Z syntax).

Covers parsing, graph building, design matrix construction, compilation,
introspection, do() propagation, and effects extraction.
"""

import warnings

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc
from pathmc.compile import get_predictor_columns
from pathmc.graph import build_graph
from pathmc.parse import parse_spec

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTERACTION_SPEC = "Y ~ X + Z + X:Z"
LABELED_INTERACTION_SPEC = "Y ~ a*X + b*Z + c*X:Z"
INTERACTION_ONLY_SPEC = "Y ~ X:Z"
MULTI_EQ_INTERACTION = """\
M ~ a*X
Y ~ b*M + d*X + e*X:M
"""


@pytest.fixture(scope="module")
def interaction_data():
    """Data with true moderation: Y = 1*X + 0.5*Z + 0.8*X*Z + noise."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=n)
    Z = rng.normal(size=n)
    Y = 1.0 * X + 0.5 * Z + 0.8 * X * Z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "Z": Z, "Y": Y})


@pytest.fixture(scope="module")
def mediation_interaction_data():
    """X -> M and X*M -> Y."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    Y = 0.3 * M + 0.4 * X + 0.6 * X * M + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


# ---------------------------------------------------------------------------
# Parse tests
# ---------------------------------------------------------------------------


class TestInteractionParsing:
    def test_basic_interaction(self):
        spec = parse_spec(INTERACTION_SPEC)
        reg = spec.regressions[0]
        variables = [t.variable for t in reg.terms]
        assert "X" in variables
        assert "Z" in variables
        assert "X:Z" in variables

    def test_interaction_of_field(self):
        spec = parse_spec(INTERACTION_SPEC)
        reg = spec.regressions[0]
        xz = next(t for t in reg.terms if t.variable == "X:Z")
        assert xz.interaction_of == ("X", "Z")

    def test_plain_terms_have_no_interaction_of(self):
        spec = parse_spec(INTERACTION_SPEC)
        reg = spec.regressions[0]
        x_term = next(t for t in reg.terms if t.variable == "X")
        assert x_term.interaction_of is None

    def test_labeled_interaction(self):
        spec = parse_spec(LABELED_INTERACTION_SPEC)
        reg = spec.regressions[0]
        xz = next(t for t in reg.terms if t.variable == "X:Z")
        assert xz.label == "c"
        assert xz.interaction_of == ("X", "Z")

    def test_all_labels_parsed(self):
        spec = parse_spec(LABELED_INTERACTION_SPEC)
        reg = spec.regressions[0]
        labels = {t.variable: t.label for t in reg.terms}
        assert labels == {"X": "a", "Z": "b", "X:Z": "c"}

    def test_interaction_only(self):
        spec = parse_spec(INTERACTION_ONLY_SPEC)
        reg = spec.regressions[0]
        assert len(reg.terms) == 1
        assert reg.terms[0].variable == "X:Z"
        assert reg.terms[0].interaction_of == ("X", "Z")

    def test_three_way_interaction(self):
        spec = parse_spec("Y ~ X:Z:W")
        reg = spec.regressions[0]
        assert reg.terms[0].variable == "X:Z:W"
        assert reg.terms[0].interaction_of == ("X", "Z", "W")

    def test_multiple_interactions(self):
        spec = parse_spec("Y ~ X + Z + W + X:Z + X:W")
        reg = spec.regressions[0]
        interaction_vars = [
            t.variable for t in reg.terms if t.interaction_of is not None
        ]
        assert "X:Z" in interaction_vars
        assert "X:W" in interaction_vars

    def test_interaction_with_intercept_suppression(self):
        spec = parse_spec("Y ~ 0 + X + Z + X:Z")
        reg = spec.regressions[0]
        assert reg.has_intercept is False
        assert any(t.variable == "X:Z" for t in reg.terms)

    @pytest.mark.parametrize(
        "bad_spec,match",
        [
            ("Y ~ X:", "Empty variable"),
            ("Y ~ :Z", "Empty variable"),
            ("Y ~ :", "Empty variable"),
        ],
    )
    def test_malformed_interaction_raises(self, bad_spec, match):
        with pytest.raises(Exception, match=match):
            parse_spec(bad_spec)


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------


class TestInteractionGraph:
    def test_interaction_constituents_are_parents(self):
        spec = parse_spec(INTERACTION_SPEC)
        gi = build_graph(spec)
        assert gi.has_edge("X", "Y")
        assert gi.has_edge("Z", "Y")

    def test_interaction_column_not_a_node(self):
        spec = parse_spec(INTERACTION_SPEC)
        gi = build_graph(spec)
        assert "X:Z" not in gi.topological_order

    def test_interaction_only_adds_constituents(self):
        spec = parse_spec(INTERACTION_ONLY_SPEC)
        gi = build_graph(spec)
        assert gi.has_edge("X", "Y")
        assert gi.has_edge("Z", "Y")
        assert "X:Z" not in gi.topological_order

    def test_exogenous_classification(self):
        spec = parse_spec(INTERACTION_SPEC)
        gi = build_graph(spec)
        assert "X" in gi.exogenous
        assert "Z" in gi.exogenous
        assert "Y" in gi.endogenous


# ---------------------------------------------------------------------------
# Design matrix tests
# ---------------------------------------------------------------------------


class TestInteractionDesignMatrix:
    def test_design_includes_interaction_column(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        dm = model.design("Y")
        assert "X:Z" in dm.columns

    def test_interaction_column_is_product(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        dm = model.design("Y")
        expected = interaction_data["X"].values * interaction_data["Z"].values
        np.testing.assert_allclose(dm["X:Z"].values, expected, atol=1e-10)

    def test_predictor_columns_include_interaction(self):
        spec = parse_spec(INTERACTION_SPEC)
        cols = get_predictor_columns(spec.regressions[0])
        assert "X:Z" in cols
        assert "Intercept" in cols

    def test_labeled_interaction_design(self, interaction_data):
        model = pathmc.model(LABELED_INTERACTION_SPEC, data=interaction_data)
        dm = model.design("Y")
        assert "X:Z" in dm.columns


# ---------------------------------------------------------------------------
# Compilation tests
# ---------------------------------------------------------------------------


class TestInteractionCompilation:
    def test_compiles_to_pymc_model(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_beta_has_interaction_coordinate(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        gen = model._gen_model
        beta = gen["beta_Y"]
        coords = gen.coords["Y_predictors"]
        assert "X:Z" in coords

    def test_labeled_interaction_compiles(self, interaction_data):
        model = pathmc.model(LABELED_INTERACTION_SPEC, data=interaction_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_interaction_only_compiles(self, interaction_data):
        model = pathmc.model(INTERACTION_ONLY_SPEC, data=interaction_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_multi_equation_interaction(self, mediation_interaction_data):
        model = pathmc.model(MULTI_EQ_INTERACTION, data=mediation_interaction_data)
        assert isinstance(model.pymc_model, pm.Model)
        coords = model._gen_model.coords["Y_predictors"]
        assert "X:M" in coords


# ---------------------------------------------------------------------------
# Introspection tests
# ---------------------------------------------------------------------------


class TestInteractionIntrospection:
    def test_equations_show_interaction(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        eqs = model.equations()
        eq_str = str(eqs)
        assert "×" in eq_str or "X:Z" in eq_str

    def test_labeled_equations(self, interaction_data):
        model = pathmc.model(LABELED_INTERACTION_SPEC, data=interaction_data)
        eqs = model.equations()
        eq_str = str(eqs)
        assert "c*" in eq_str

    def test_dag_renders(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        dot = model.graph()
        src = dot.source
        assert "X" in src
        assert "Z" in src
        assert "Y" in src

    def test_latex_rendering(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        eqs = model.equations()
        latex = eqs._repr_latex_()
        assert r"\times" in latex

    def test_priors_include_interaction_beta(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        priors = model.priors()
        prior_str = str(priors)
        assert "beta_Y" in prior_str


# ---------------------------------------------------------------------------
# Sampling smoke test (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestInteractionSampling:
    @pytest.fixture(scope="class")
    def sampled_labeled_interaction(self, interaction_data):
        model = pathmc.model(LABELED_INTERACTION_SPEC, data=interaction_data)
        model.fit(draws=500, tune=500, chains=2, random_seed=42)
        return model

    @pytest.fixture(scope="class")
    def sampled_interaction(self, interaction_data):
        model = pathmc.model(INTERACTION_SPEC, data=interaction_data)
        model.fit(draws=200, tune=200, chains=1, random_seed=42)
        return model

    def test_fit_and_sample(self, sampled_labeled_interaction):
        summary = sampled_labeled_interaction.summary()
        assert "beta_Y" in summary.index[0]

    def test_interaction_coefficient_recovery(self, sampled_labeled_interaction):
        """True c (interaction) = 0.8. Check it's in a reasonable range."""
        effects = sampled_labeled_interaction.effects_summary()
        assert "c" in effects.index
        c_mean = effects.loc["c", "mean"]
        assert 0.3 < c_mean < 1.3, f"Interaction coef c={c_mean}, expected ~0.8"

    def test_do_propagates_through_interaction(self, sampled_interaction):
        """do(X=1) vs do(X=0) with Z held at its mean should differ."""
        r0 = sampled_interaction.do(set={"X": 0.0})
        r1 = sampled_interaction.do(set={"X": 1.0})
        diff = r1.mean("Y") - r0.mean("Y")
        assert abs(diff) > 0.1, "Intervention should propagate through interaction"

    def test_do_interaction_depends_on_moderator(self, sampled_interaction):
        """The effect of X on Y should differ at Z=-1 vs Z=+1 (moderation)."""
        cate_low = sampled_interaction.cate("Y", "X", condition={"Z": -1.0})
        cate_high = sampled_interaction.cate("Y", "X", condition={"Z": 1.0})
        diff_low = cate_low.mean("Y")
        diff_high = cate_high.mean("Y")
        assert abs(diff_high - diff_low) > 0.5, (
            f"Moderation effect too small: diff_high={diff_high:.2f}, "
            f"diff_low={diff_low:.2f}"
        )


def _tiny_frame(*names: str) -> pd.DataFrame:
    return pd.DataFrame({
        name: np.linspace(-1, 1, 8) + offset for offset, name in enumerate(names)
    })


def _inject_posterior(model, coefficients: dict[str, dict[str, float]]) -> None:
    """Pin posterior means for ``{outcome: {predictor: value}}``."""
    posterior = {}
    coords = {}
    dims = {}
    for outcome, values in coefficients.items():
        coord = f"{outcome}_predictors"
        order = [str(value) for value in model.pymc_model.coords[coord]]
        vector = [float(values.get(name, 0.0)) for name in order]
        posterior[f"beta_{outcome}"] = np.tile(vector, (1, 4, 1))
        coords[coord] = order
        dims[f"beta_{outcome}"] = [coord]
    model._idata = az.from_dict(
        {"posterior": posterior},
        coords=coords,
        dims=dims,
    )


def _assert_names_interaction(model, path: str, *terms: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        model.effect(path)
    message = str(exc_info.value)
    for term in terms:
        assert term in message
    assert "No direct edge" not in message
    assert "does not exist" not in message
    assert "absent" not in message


class TestInteractionOnlyEffectAgreesWithGraph:
    """A drawn interaction constituent is not a missing edge.

    ``graph()`` draws ``X -> Y`` for ``Y ~ X:Z``. ``effect("X -> Y")`` must
    name that interaction term instead of claiming the edge is absent.
    ``effect("X:Z -> Y")`` still returns the interaction coefficient.
    """

    def test_endpoint_query_names_interaction_term(self):
        model = pathmc.model("Y ~ g*X:Z", data=_tiny_frame("X", "Z", "Y"))
        source = model.graph().source
        assert "X -> Y" in source
        assert "Z -> Y" in source
        _inject_posterior(model, {"Y": {"X:Z": 1.7}})

        assert model.effect("X:Z -> Y").mean == pytest.approx(1.7)
        _assert_names_interaction(model, "X -> Y", "X:Z")
        _assert_names_interaction(model, "Z -> Y", "X:Z")
        _assert_names_interaction(model, "X->Y", "X:Z")

    def test_three_way_names_full_interaction(self):
        model = pathmc.model("D ~ A:B:C", data=_tiny_frame("A", "B", "C", "D"))
        assert "B -> D" in model.graph().source
        _inject_posterior(model, {"D": {"A:B:C": -0.4}})

        assert model.effect("A:B:C -> D").mean == pytest.approx(-0.4)
        _assert_names_interaction(model, "B -> D", "A:B:C")
        _assert_names_interaction(model, "C -> D", "A:B:C")

    def test_names_every_interaction_involving_the_source(self):
        model = pathmc.model("Y ~ X:Z + X:W", data=_tiny_frame("X", "Z", "W", "Y"))
        _inject_posterior(model, {"Y": {"X:Z": 2.25, "X:W": 0.5}})

        assert model.effect("X:W -> Y").mean == pytest.approx(0.5)
        _assert_names_interaction(model, "X -> Y", "X:Z", "X:W")

    def test_missing_edge_still_reports_absence(self):
        model = pathmc.model("Y ~ X:Z", data=_tiny_frame("X", "Z", "Y", "W"))
        _inject_posterior(model, {"Y": {"X:Z": 1.0}})
        with pytest.raises(ValueError, match="No direct edge from 'W' to 'Y'"):
            model.effect("W -> Y")

    def test_main_effect_plus_interaction_still_returns_main_coefficient(self):
        model = pathmc.model("Y ~ X + X:Z", data=_tiny_frame("X", "Z", "Y"))
        _inject_posterior(model, {"Y": {"X": 0.3, "X:Z": 1.7}})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert model.effect("X -> Y").mean == pytest.approx(0.3)
        assert model.effect("X:Z -> Y").mean == pytest.approx(1.7)

    def test_multi_edge_path_names_interaction_on_that_edge(self):
        model = pathmc.model("M ~ X\nY ~ M:Z", data=_tiny_frame("X", "M", "Z", "Y"))
        assert "M -> Y" in model.graph().source
        _inject_posterior(model, {"M": {"X": 0.5}, "Y": {"M:Z": 1.1}})

        _assert_names_interaction(model, "X -> M -> Y", "M:Z")
        assert model.effect("X -> M").mean == pytest.approx(0.5)
        assert model.effect("M:Z -> Y").mean == pytest.approx(1.1)
