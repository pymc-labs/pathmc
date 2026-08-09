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
"""Item 11 (issue #327): DSL parse-tree golden tests for tricky specs.

Covers operator precedence, ``a:b`` vs ``a*b``, nested transforms, the
``0 +``/``-1`` intercept-removal crash vector (#316), hyphenated names
rejected by the top-level minus guard, multi-equation specs, and variable
names colliding with transform names. Every test here only
calls ``parse_spec`` -- no data, no PyMC, no sampling -- so the whole file
runs in well under a second.
"""

import pytest

from pathmc.exceptions import DuplicateEquationError, ParseError
from pathmc.parse import Term, TransformCall, parse_spec

# ---------------------------------------------------------------------------
# Operator precedence: '*' (label) binds a single term; ':' makes an
# interaction; '+' separates terms. These operators are NOT nested R-style
# formula algebra -- each raw '+'-separated chunk is parsed independently.
# ---------------------------------------------------------------------------


class TestOperatorPrecedence:
    def test_label_star_binds_single_term(self):
        spec = parse_spec("y ~ a*b + c")
        terms = spec.regressions[0].terms
        assert terms[0] == Term(variable="b", label="a")
        assert terms[1] == Term(variable="c")

    def test_numeric_star_is_fixed_value_not_label(self):
        spec = parse_spec("y ~ 2*b + c")
        terms = spec.regressions[0].terms
        assert terms[0] == Term(variable="b", fixed_value=2.0)
        assert terms[1] == Term(variable="c")

    def test_label_applies_only_to_its_own_term(self):
        """'a*b + c*d' is two independently-labeled terms, not
        label 'a' distributing over 'b + c*d'."""
        spec = parse_spec("y ~ a*b + c*d")
        terms = spec.regressions[0].terms
        assert terms == [
            Term(variable="b", label="a"),
            Term(variable="d", label="c"),
        ]

    def test_colon_binds_tighter_than_plus(self):
        """'a:b + c' is an interaction term plus a separate plain term,
        not 'a : (b + c)'."""
        spec = parse_spec("y ~ a:b + c")
        terms = spec.regressions[0].terms
        assert terms[0] == Term(variable="a:b", interaction_of=("a", "b"))
        assert terms[1] == Term(variable="c")

    def test_three_way_interaction(self):
        spec = parse_spec("y ~ a:b:c")
        term = spec.regressions[0].terms[0]
        assert term.interaction_of == ("a", "b", "c")
        assert term.variable == "a:b:c"

    def test_label_wraps_interaction(self):
        spec = parse_spec("y ~ lbl*a:b")
        term = spec.regressions[0].terms[0]
        assert term.label == "lbl"
        assert term.interaction_of == ("a", "b")


class TestColonVsStarInteraction:
    """':' is a real interaction (product of variables). '*' with a
    non-numeric prefix is a *coefficient label*, not an R-style
    "main effects + interaction" expansion -- pathmc's DSL never expands
    'a*b' into 'a + b + a:b'."""

    def test_colon_is_interaction(self):
        spec = parse_spec("y ~ a:b")
        term = spec.regressions[0].terms[0]
        assert term.interaction_of == ("a", "b")
        assert term.transform is None

    def test_star_with_named_prefix_is_a_label_not_interaction(self):
        spec = parse_spec("y ~ a*b")
        term = spec.regressions[0].terms[0]
        # This is coefficient label 'a' applied to plain variable 'b' --
        # a single term, not two variables interacting.
        assert term.interaction_of is None
        assert term.label == "a"
        assert term.variable == "b"

    def test_star_never_expands_to_main_effects(self):
        """'a*b' produces exactly one term (label 'a' on 'b'), never the
        three R-formula terms 'a + b + a:b'."""
        spec = parse_spec("y ~ a*b")
        assert len(spec.regressions[0].terms) == 1


# ---------------------------------------------------------------------------
# Nested transforms
# ---------------------------------------------------------------------------


class TestNestedTransforms:
    def test_single_transform(self):
        spec = parse_spec("y ~ adstock(tv, decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.variable == "tv"
        assert term.transform == TransformCall(
            name="adstock", input_expr="tv", params={"decay": "theta"}
        )

    def test_adstock_of_saturation(self):
        spec = parse_spec("y ~ adstock(logistic_saturation(tv, lam=lam), decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.variable == "tv"
        outer = term.transform
        assert outer.name == "adstock"
        assert outer.params == {"decay": "theta"}
        inner = outer.input_expr
        assert isinstance(inner, TransformCall)
        assert inner.name == "logistic_saturation"
        assert inner.input_expr == "tv"
        assert inner.params == {"lam": "lam"}

    def test_saturation_of_adstock(self):
        """Same two transforms, opposite nesting order -> different tree."""
        spec = parse_spec("y ~ logistic_saturation(adstock(tv, decay=theta), lam=lam)")
        term = spec.regressions[0].terms[0]
        outer = term.transform
        assert outer.name == "logistic_saturation"
        assert outer.params == {"lam": "lam"}
        inner = outer.input_expr
        assert isinstance(inner, TransformCall)
        assert inner.name == "adstock"
        assert inner.input_expr == "tv"
        assert inner.params == {"decay": "theta"}

    def test_triple_nested_transform_leaf_variable(self):
        spec = parse_spec("y ~ adstock(logistic_saturation(square(tv)), decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.variable == "tv"
        assert term.transform.name == "adstock"
        assert term.transform.input_expr.name == "logistic_saturation"
        assert term.transform.input_expr.input_expr.name == "square"
        assert term.transform.input_expr.input_expr.input_expr == "tv"

    def test_labeled_nested_transform(self):
        spec = parse_spec("y ~ b_tv*adstock(logistic_saturation(tv), decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.label == "b_tv"
        assert term.transform.name == "adstock"

    def test_lag_rejects_nested_transform(self):
        with pytest.raises(ParseError, match="nested transform"):
            parse_spec("y ~ lag(adstock(tv, decay=theta))")


# ---------------------------------------------------------------------------
# '0 +' / '-1' intercept removal -- '-1' is a known crash vector (#316):
# without an explicit guard it silently becomes a bogus variable name
# instead of raising at parse time.
# ---------------------------------------------------------------------------


class TestInterceptRemoval:
    def test_zero_plus_drops_intercept(self):
        spec = parse_spec("y ~ 0 + x")
        reg = spec.regressions[0]
        assert reg.has_intercept is False
        assert reg.terms == [Term(variable="x")]

    def test_zero_can_appear_after_other_terms(self):
        spec = parse_spec("y ~ x + 0")
        reg = spec.regressions[0]
        assert reg.has_intercept is False
        assert reg.terms == [Term(variable="x")]

    def test_bare_one_is_intercept_only_noop(self):
        spec = parse_spec("y ~ 1 + x")
        reg = spec.regressions[0]
        assert reg.has_intercept is True
        assert reg.terms == [Term(variable="x")]

    @pytest.mark.parametrize(
        "spec_string",
        [
            "y ~ x - 1",
            "y ~ x + z - 1",
            "y ~ -1",
            "y ~ -x",
            # Variable names ending in 'e' must not be mistaken for the
            # mantissa of a float literal and skip the guard.
            "y ~ response-1",
            "y ~ income-1",
            "y ~ x + rate-1",
            "y ~ RATE-1",
            # Hyphenated plain terms and coefficient labels (use underscores).
            "y ~ my-var",
            "y ~ my-label*x",
            # An 'e' terminating an identifier, not a numeric mantissa.
            "y ~ x2e-1",
            # 'e' with no digits after the sign is not an exponent either.
            "y ~ 1e-x",
        ],
    )
    def test_r_style_subtraction_raises_clear_parse_error(self, spec_string):
        """R/patsy '-1' or '-x' syntax must raise a clear ParseError
        immediately, not silently produce a garbage variable name like
        'x - 1' that fails confusingly later at data-lookup time."""
        with pytest.raises(ParseError, match="Unsupported '-'"):
            parse_spec(spec_string)

    def test_negative_fixed_coefficient_still_allowed(self):
        """'-1*x' is NOT intercept-removal syntax -- it's a fixed
        coefficient of -1 for x -- and must keep working."""
        spec = parse_spec("y ~ -1*x")
        term = spec.regressions[0].terms[0]
        assert term.variable == "x"
        assert term.fixed_value == -1.0

    @pytest.mark.parametrize(
        "spec_string, expected",
        [
            ("y ~ 1e-5*x", 1e-5),
            ("y ~ 1E-5*x", 1e-5),
            ("y ~ 2.5e-3*x", 2.5e-3),
            ("y ~ .5e-3*x", 0.5e-3),
            ("y ~ -1e-5*x", -1e-5),
        ],
    )
    def test_scientific_notation_fixed_coefficient_still_allowed(
        self, spec_string, expected
    ):
        """Scientific-notation fixed coefficients must not be mistaken for
        R-style subtraction (the '-' is part of the exponent, not a term
        operator)."""
        spec = parse_spec(spec_string)
        term = spec.regressions[0].terms[0]
        assert term.variable == "x"
        assert term.fixed_value == expected

    @pytest.mark.parametrize(
        "spec_string",
        [
            "y ~ lag(rate-1)",
            "y ~ adstock(my-var)",
            "y ~ logistic_saturation(income-1, lam=lam)",
            "y ~ adstock(logistic_saturation(my-var), decay=theta)",
        ],
    )
    def test_hyphen_inside_transform_arg_raises(self, spec_string):
        """A '-' inside a transform's leaf input is the same garbage-name
        bug as a top-level '-'. The guard must apply to transform
        arguments, not just top-level terms."""
        with pytest.raises(ParseError, match="Unsupported '-'"):
            parse_spec(spec_string)

    def test_negative_transform_param_still_allowed(self):
        """A '-' inside transform parentheses (a negative kwarg value) is
        unrelated to formula subtraction and must not be rejected."""
        spec = parse_spec("y ~ adstock(tv, decay=-0.5)")
        term = spec.regressions[0].terms[0]
        assert term.transform.params == {"decay": "-0.5"}


# ---------------------------------------------------------------------------
# Multi-equation specs
# ---------------------------------------------------------------------------


class TestMultiEquationSpecs:
    def test_newline_separated_equations(self):
        spec = parse_spec("m ~ x\ny ~ m + x")
        assert [r.lhs for r in spec.regressions] == ["m", "y"]

    def test_semicolon_separated_equations(self):
        spec = parse_spec("m ~ x; y ~ m + x")
        assert [r.lhs for r in spec.regressions] == ["m", "y"]

    def test_mixed_newline_and_semicolon(self):
        spec = parse_spec("m ~ x; n ~ x\ny ~ m + n")
        assert [r.lhs for r in spec.regressions] == ["m", "n", "y"]

    def test_continuation_lines_join_to_preceding_equation(self):
        spec = parse_spec(
            "y ~ b_tv*adstock(tv, decay=theta)\n"
            "  + b_dig*logistic_saturation(digital, lam=lam)\n"
            "  + trend"
        )
        assert len(spec.regressions) == 1
        assert len(spec.regressions[0].terms) == 3

    def test_duplicate_lhs_raises(self):
        with pytest.raises(DuplicateEquationError):
            parse_spec("y ~ x\ny ~ z")

    def test_residual_covariance_and_defined_param_alongside_regressions(self):
        spec = parse_spec("m ~ x\ny ~ m\nm ~~ y\ntotal := 1")
        assert [r.lhs for r in spec.regressions] == ["m", "y"]
        assert len(spec.residual_covs) == 1
        assert spec.residual_covs[0].var1 == "m"
        assert spec.residual_covs[0].var2 == "y"
        assert spec.defined_params[0].name == "total"


# ---------------------------------------------------------------------------
# Variable names colliding with transform/keyword names
# ---------------------------------------------------------------------------


class TestNameCollisions:
    def test_variable_named_like_a_transform_as_plain_term(self):
        """A bare variable named 'adstock' (no parens) is just a plain
        term, not a zero-arg transform call attempt."""
        spec = parse_spec("y ~ adstock")
        assert spec.regressions[0].terms == [Term(variable="adstock")]

    def test_variable_named_like_a_transform_as_transform_input(self):
        """A variable literally named 'adstock' used as the *input* to a
        transform of the same name must resolve to the leaf variable, not
        confuse the parser about call boundaries."""
        spec = parse_spec("y ~ adstock(adstock, decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.variable == "adstock"
        assert term.transform.name == "adstock"
        assert term.transform.input_expr == "adstock"

    def test_variable_named_lag_as_plain_term(self):
        spec = parse_spec("y ~ lag")
        assert spec.regressions[0].terms == [Term(variable="lag")]

    def test_variable_named_lag_inside_real_lag_call(self):
        spec = parse_spec("y ~ lag(lag)")
        term = spec.regressions[0].terms[0]
        assert term.lag_of == "lag"
        assert term.variable == "lag(lag)"

    def test_variable_named_hsgp_as_plain_term(self):
        spec = parse_spec("y ~ hsgp")
        assert spec.regressions[0].terms == [Term(variable="hsgp")]

    def test_lhs_named_like_a_transform(self):
        spec = parse_spec("adstock ~ x")
        assert spec.regressions[0].lhs == "adstock"

    def test_interaction_of_two_transform_named_variables(self):
        spec = parse_spec("y ~ lag:adstock")
        term = spec.regressions[0].terms[0]
        assert term.interaction_of == ("lag", "adstock")
