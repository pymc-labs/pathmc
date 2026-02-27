"""M1 gate tests: DSL parsing.

These tests verify that parse_spec() correctly transforms DSL strings
into structured Spec objects. They pin down the parser's public interface.

All tests are fast (no data, no PyMC, no sampling).
"""

import pytest

from pathmc.parse import parse_spec

from conftest import (
    SIMPLE_REGRESSION,
    MEDIATION_SPEC,
    PARALLEL_MEDIATORS_SPEC,
    NO_INTERCEPT_SPEC,
    SEMICOLON_SPEC,
    DUPLICATE_LHS_SPEC,
)


class TestRegressionParsing:
    def test_simple_regression(self):
        spec = parse_spec(SIMPLE_REGRESSION)
        assert len(spec.regressions) == 1
        reg = spec.regressions[0]
        assert reg.lhs == "Y"
        variables = [t.variable for t in reg.terms]
        assert "X1" in variables
        assert "X2" in variables

    def test_multiple_regressions(self):
        spec = parse_spec(MEDIATION_SPEC)
        assert len(spec.regressions) == 2
        lhs_vars = {r.lhs for r in spec.regressions}
        assert lhs_vars == {"M", "Y"}

    def test_default_intercept(self):
        spec = parse_spec(SIMPLE_REGRESSION)
        assert spec.regressions[0].has_intercept is True

    def test_intercept_suppression(self):
        spec = parse_spec(NO_INTERCEPT_SPEC)
        reg = spec.regressions[0]
        assert reg.has_intercept is False
        variables = [t.variable for t in reg.terms]
        assert "X1" in variables
        assert "X2" in variables

    def test_single_predictor(self):
        spec = parse_spec("Y ~ X")
        assert len(spec.regressions) == 1
        assert len(spec.regressions[0].terms) == 1
        assert spec.regressions[0].terms[0].variable == "X"


class TestLabeledCoefficients:
    def test_labeled_terms(self):
        spec = parse_spec(MEDIATION_SPEC)
        m_eq = next(r for r in spec.regressions if r.lhs == "M")
        x_term = next(t for t in m_eq.terms if t.variable == "X")
        assert x_term.label == "a"

    def test_mixed_labeled_unlabeled(self):
        spec = parse_spec("Y ~ a*X1 + X2")
        reg = spec.regressions[0]
        x1_term = next(t for t in reg.terms if t.variable == "X1")
        x2_term = next(t for t in reg.terms if t.variable == "X2")
        assert x1_term.label == "a"
        assert x2_term.label is None

    def test_all_labeled(self):
        spec = parse_spec("Y ~ b1*M1 + b2*M2 + c*T")
        reg = spec.regressions[0]
        labels = {t.variable: t.label for t in reg.terms}
        assert labels == {"M1": "b1", "M2": "b2", "T": "c"}


class TestDefinedParams:
    def test_defined_param_parsed(self):
        spec = parse_spec(MEDIATION_SPEC)
        assert len(spec.defined_params) == 1
        dp = spec.defined_params[0]
        assert dp.name == "indirect"
        assert "a" in dp.expression and "b" in dp.expression

    def test_multiple_defined_params(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        names = {dp.name for dp in spec.defined_params}
        assert names == {"indirect1", "indirect2", "total"}

    def test_defined_param_expression_preserved(self):
        spec = parse_spec("Y ~ a*X\neff := a")
        dp = spec.defined_params[0]
        assert dp.name == "eff"
        assert dp.expression.strip() == "a"


class TestResidualCovariance:
    def test_residual_cov_parsed(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        assert len(spec.residual_covs) >= 1
        pairs = {(rc.var1, rc.var2) for rc in spec.residual_covs}
        assert ("M1", "M2") in pairs or ("M2", "M1") in pairs

    def test_no_residual_cov_when_absent(self):
        spec = parse_spec(MEDIATION_SPEC)
        assert len(spec.residual_covs) == 0


class TestSyntaxVariants:
    def test_semicolons_equivalent_to_newlines(self):
        spec_newline = parse_spec(MEDIATION_SPEC)
        spec_semi = parse_spec(SEMICOLON_SPEC)
        assert len(spec_newline.regressions) == len(spec_semi.regressions)
        assert len(spec_newline.defined_params) == len(spec_semi.defined_params)

    def test_extra_whitespace_tolerated(self):
        spec = parse_spec("  Y  ~  X1  +  X2  ")
        assert len(spec.regressions) == 1
        assert spec.regressions[0].lhs == "Y"

    def test_blank_lines_ignored(self):
        messy = "\n\n  M ~ a*X  \n\n\n  Y ~ b*M + c*X  \n\n"
        spec = parse_spec(messy)
        assert len(spec.regressions) == 2

    def test_trailing_semicolons_tolerated(self):
        spec = parse_spec("Y ~ X;")
        assert len(spec.regressions) == 1


class TestParseErrors:
    def test_duplicate_lhs_raises(self):
        with pytest.raises(Exception, match="(?i)duplicate"):
            parse_spec(DUPLICATE_LHS_SPEC)

    def test_empty_spec_raises(self):
        with pytest.raises(Exception):
            parse_spec("")

    def test_whitespace_only_raises(self):
        with pytest.raises(Exception):
            parse_spec("   \n\n   ")

    def test_malformed_tilde_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ ~ X")

    def test_missing_rhs_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~")
