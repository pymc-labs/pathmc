"""M19 gate tests: Transform DSL parsing.

Tests verify that parse_spec() correctly recognises transform expressions
with named parameters, labels, nesting/composition, and error cases.

All tests are fast (no data, no PyMC, no sampling).
"""

import pytest

from pathmc.parse import parse_spec


class TestSingleTransform:
    """Parse a single transform call in a regression term."""

    def test_basic_transform_parsed(self):
        spec = parse_spec("Y ~ adstock(X, decay=theta)")
        assert len(spec.regressions) == 1
        reg = spec.regressions[0]
        assert len(reg.terms) == 1
        term = reg.terms[0]
        assert term.transform is not None
        assert term.transform.name == "adstock"
        assert term.transform.input_expr == "X"
        assert term.transform.params == {"decay": "theta"}

    def test_transform_variable_name_set(self):
        spec = parse_spec("Y ~ adstock(X, decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.variable == "X" or term.transform.input_expr == "X"

    def test_saturation_transform(self):
        spec = parse_spec("Y ~ logistic_saturation(X, lam=lam_x)")
        term = spec.regressions[0].terms[0]
        assert term.transform.name == "logistic_saturation"
        assert term.transform.params == {"lam": "lam_x"}

    def test_transform_no_label(self):
        spec = parse_spec("Y ~ adstock(X, decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.label is None


class TestLabeledTransform:
    """Parse transforms with coefficient labels."""

    def test_labeled_transform(self):
        spec = parse_spec("Y ~ b*adstock(X, decay=theta)")
        term = spec.regressions[0].terms[0]
        assert term.label == "b"
        assert term.transform is not None
        assert term.transform.name == "adstock"

    def test_label_and_params_independent(self):
        spec = parse_spec("sales ~ b_tv*adstock(tv, decay=theta_tv)")
        term = spec.regressions[0].terms[0]
        assert term.label == "b_tv"
        assert term.transform.name == "adstock"
        assert term.transform.input_expr == "tv"
        assert term.transform.params == {"decay": "theta_tv"}


class TestNestedTransforms:
    """Parse composed/nested transforms."""

    def test_nested_transform(self):
        spec = parse_spec("Y ~ logistic_saturation(adstock(X, decay=theta), lam=lam_x)")
        term = spec.regressions[0].terms[0]
        outer = term.transform
        assert outer.name == "logistic_saturation"
        assert outer.params == {"lam": "lam_x"}
        inner = outer.input_expr
        assert not isinstance(inner, str)
        assert inner.name == "adstock"
        assert inner.input_expr == "X"
        assert inner.params == {"decay": "theta"}

    def test_nested_with_label(self):
        spec = parse_spec(
            "Y ~ b*logistic_saturation(adstock(X, decay=theta), lam=lam_x)"
        )
        term = spec.regressions[0].terms[0]
        assert term.label == "b"
        assert term.transform.name == "logistic_saturation"
        assert term.transform.input_expr.name == "adstock"


class TestMixedSpec:
    """Transforms mixed with plain terms, := definitions, and ~~ ."""

    def test_transforms_with_plain_terms(self):
        spec = parse_spec("Y ~ b_tv*adstock(tv, decay=theta_tv) + trend")
        reg = spec.regressions[0]
        assert len(reg.terms) == 2
        transform_terms = [t for t in reg.terms if t.transform is not None]
        plain_terms = [t for t in reg.terms if t.transform is None]
        assert len(transform_terms) == 1
        assert len(plain_terms) == 1
        assert plain_terms[0].variable == "trend"

    def test_full_spec_with_transforms(self):
        spec = parse_spec("""
        sales ~ b_tv*adstock(tv, decay=theta_tv) + b_dig*logistic_saturation(digital, lam=lam_dig) + trend
        """)
        reg = spec.regressions[0]
        assert len(reg.terms) == 3
        transform_terms = [t for t in reg.terms if t.transform is not None]
        assert len(transform_terms) == 2
        names = {t.transform.name for t in transform_terms}
        assert names == {"adstock", "logistic_saturation"}

    def test_transforms_with_defined_params(self):
        spec = parse_spec("""
        Y ~ a*adstock(X, decay=theta) + b*Z
        total := a + b
        """)
        assert len(spec.regressions) == 1
        assert len(spec.defined_params) == 1
        assert spec.defined_params[0].name == "total"

    def test_transforms_with_residual_cov(self):
        spec = parse_spec("""
        M1 ~ adstock(X, decay=theta1)
        M2 ~ adstock(X, decay=theta2)
        Y ~ M1 + M2
        M1 ~~ M2
        """)
        assert len(spec.regressions) == 3
        assert len(spec.residual_covs) == 1

    def test_multiple_equations_with_transforms(self):
        spec = parse_spec("""
        M ~ adstock(X, decay=theta)
        Y ~ logistic_saturation(M, lam=lam_m) + X
        """)
        assert len(spec.regressions) == 2
        m_reg = next(r for r in spec.regressions if r.lhs == "M")
        y_reg = next(r for r in spec.regressions if r.lhs == "Y")
        assert m_reg.terms[0].transform.name == "adstock"
        assert any(
            t.transform and t.transform.name == "logistic_saturation"
            for t in y_reg.terms
        )


class TestTransformParseErrors:
    """Error handling for malformed transform expressions."""

    def test_unclosed_paren_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ adstock(X, decay=theta")

    def test_empty_transform_name_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ (X, decay=theta)")

    def test_missing_param_value_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ adstock(X, decay=)")

    def test_missing_param_name_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ adstock(X, =theta)")

    def test_no_input_variable_raises(self):
        with pytest.raises(Exception):
            parse_spec("Y ~ adstock(decay=theta)")
