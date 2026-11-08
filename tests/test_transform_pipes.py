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
"""Pipe syntax lowers to the existing transform expression contract."""

import pytest
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

import pathmc
from pathmc.basis import REGISTRY as BASIS_REGISTRY
from pathmc.basis import Basis, register_basis
from pathmc.exceptions import DuplicateEquationError, ParseError
from pathmc.effects import evaluate_defined_params
from pathmc.parse import TransformCall, parse_spec
from pathmc.scaling import Scaling
from pathmc.transforms import REGISTRY as TRANSFORM_REGISTRY
from pathmc.transforms import Transform, register_transform


@pytest.mark.parametrize(
    ("pipe", "nested"),
    [
        ("Y ~ x |> f(p=u)", "Y ~ f(x, p=u)"),
        ("Y ~ (x |> f())", "Y ~ f(x)"),
        ("Y ~ a*(x |> f(p=u) |> g(q=v))", "Y ~ a*g(f(x, p=u), q=v)"),
        ("Y ~ -2*(x |> f(p=u))", "Y ~ -2*f(x, p=u)"),
        ("Y ~ 0*(x |> f(p=u))", "Y ~ 0*f(x, p=u)"),
        ("Y ~ 1e-5*(x |> f(p=u))", "Y ~ 1e-5*f(x, p=u)"),
        (
            "Y ~ adstock |> f(p=u) |> g(q=v) |> h(r=w)",
            "Y ~ h(g(f(adstock, p=u), q=v), r=w)",
        ),
        (
            "Y ~ 0 + x|>f(p=u,) + b*(z |> g(q=v)) + h(w); Z ~ c*t\nY ~~ Z\ntotal := b*c",
            "Y ~ 0 + f(x, p=u) + b*g(z, q=v) + h(w); Z ~ c*t\nY ~~ Z\ntotal := b*c",
        ),
        (
            "Y ~ a*(x\r\n |> f(\r\n p=u,\r\n ))\r\n + z",
            "Y ~ a*f(x, p=u) + z",
        ),
        ("Y ~ x |> f(p=-0.5)", "Y ~ f(x, p=-0.5)"),
    ],
)
def test_pipe_and_nested_forms_have_equal_full_specs(pipe, nested):
    assert parse_spec(pipe) == parse_spec(nested)


@pytest.mark.parametrize(
    "rhs",
    [
        "a*x |> f()",
        "2*x |> f()",
        "|> f()",
        "x |>",
        "x |> |> f()",
        "x |> f",
        "x |> f(x, p=u)",
        "x |> f(p=u, p=v)",
        "x |> f(p=)",
        "x |> f() trailing",
        "a*(x |> f() + z)",
        "x:z |> f()",
        "(x |> f()):z",
        "x |> lag()",
        "x |> hsgp(m=20)",
        "x |> C()",
        "(a*x |> f())",
        "(a*x) |> f()",
        "(x) |> f()",
        "((x |> f()))",
        "(x |> f()) |> g()",
        "f(x) |> g()",
        "g(x |> f())",
        "x |> f(_)",
        "x |> f(p=u,,q=v)",
        "x |> f(p=u) )",
        "x |> f(p='u')",
        "x |> f(p=u.v)",
        "x |> f(p=g(u))",
    ],
)
def test_malformed_pipes_are_parse_errors(rhs):
    with pytest.raises(ParseError) as exc:
        parse_spec(f"Y ~ {rhs}")
    assert "pipe" in str(exc.value).lower() or "pipeline" in str(exc.value).lower()


def test_sum_inside_pipeline_group_says_to_move_it_outside():
    with pytest.raises(ParseError, match=r"Move '\+ z' outside the group") as exc:
        parse_spec("Y ~ a*(x |> f() + z)")
    message = str(exc.value)
    assert "a*(x |> f()) + z" in message
    assert parse_spec("Y ~ x |> f() + z") == parse_spec("Y ~ f(x) + z")


@pytest.mark.parametrize(
    ("spec", "token"),
    [
        ("Y ~ 2*x |> f()", "2*(x |> f())"),
        ("Y ~ 1e-5*x |> f()", "1e-5*(x |> f())"),
        ("Y ~ -1*x |> f()", "-1*(x |> f())"),
        ("Y ~ a*x |> f()", "a*(x |> f())"),
    ],
)
def test_prefixed_pipeline_grouping_keeps_the_coefficient_token(spec, token):
    with pytest.raises(ParseError, match="needs grouping") as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert token in message
    if token != "a*(x |> f())":
        assert "a*(x |> f())" not in message


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ x:z |> f()",
        "Y ~ (x |> f()):z",
        "Y ~ a*(x |> f()):z",
        "Y ~ 2*(x |> f()):z",
    ],
)
def test_pipeline_interaction_names_the_colon_and_skips_the_chain(spec):
    with pytest.raises(ParseError) as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert ":z" in message
    assert "needs grouping" not in message
    assert "x |> f(p=u)" not in message
    if "|>" in spec.split(":", 1)[0]:
        assert "unsupported" in message.lower()
    else:
        assert "plain variable" in message


def test_special_stage_error_does_not_suggest_a_chain():
    with pytest.raises(ParseError, match="standalone") as exc:
        parse_spec("Y ~ x |> lag()")
    assert "x |> f(p=u)" not in str(exc.value)


@pytest.mark.parametrize(
    "spec",
    ["Y ~ a*(x + y |> f())", "Y ~ (x + y |> f())"],
)
def test_sum_before_pipe_inside_group_reuses_move_outside(spec):
    with pytest.raises(ParseError, match="outside the group") as exc:
        parse_spec(spec)
    assert "Missing transform name" not in str(exc.value)


def test_ungrouped_sum_before_pipe_stays_additive():
    assert parse_spec("Y ~ x + y |> f()") == parse_spec("Y ~ x + f(y)")


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ (x:z |> f())",
        "Y ~ a*(x:z |> f())",
        "Y ~ 2*(x:z |> f())",
        "Y ~ (x:z) |> f()",
        "Y ~ a*(x:z) |> f()",
    ],
)
def test_grouped_interaction_seed_skips_the_chain(spec):
    with pytest.raises(ParseError) as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert ":z" in message
    assert "plain variable" in message
    assert "needs grouping" not in message
    assert "x |> f(p=u)" not in message


@pytest.mark.parametrize(
    "spec",
    ["Y ~ (x |> f():z)", "Y ~ a*(x |> f():z)"],
)
def test_grouped_transformed_interaction_skips_the_chain(spec):
    with pytest.raises(ParseError) as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert ":z" in message
    assert "unsupported" in message.lower()
    assert "needs grouping" not in message
    assert "x |> f(p=u)" not in message


def test_colon_inside_stage_stays_a_bad_reference():
    with pytest.raises(ParseError, match="parameter reference") as exc:
        parse_spec("Y ~ a*(x |> f(p=a:b))")
    message = str(exc.value)
    assert "unsupported" not in message.lower()
    assert "needs grouping" not in message
    assert "Transformed interaction" not in message


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ (x + y) |> f()",
        "Y ~ a*(x + y) |> f()",
        "Y ~ 2*(x + y) |> f()",
    ],
)
def test_parenthesized_sum_before_pipe_is_not_torn(spec):
    with pytest.raises(ParseError) as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert "Missing transform name" not in message
    assert "x |> f(p=u)" not in message
    assert "(x + y)" in message
    assert "plain variable" in message


@pytest.mark.parametrize(
    ("spec", "suggestion"),
    [
        ("Y ~ a*(x\n |> f()\nZ ~ w", "a*(x |> f())"),
        ("Y ~ (x\n|> f()\nZ ~ w", "(x |> f())"),
        ("Y ~ a*(x |> f()\nZ ~ w", "a*(x |> f())"),
        ("Y ~ 2*(x |> f(", "2*(x |> f())"),
    ],
)
def test_unclosed_pipeline_group_names_the_closed_form(spec, suggestion):
    with pytest.raises(ParseError) as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert "Unclosed pipeline group" in message
    assert suggestion in message
    assert "Missing transform name" not in message
    assert "x |> f(p=u)" not in message
    assert "needs grouping" not in message


def test_ungrouped_pipeline_still_cannot_span_lines():
    with pytest.raises(ParseError, match="cannot span lines"):
        parse_spec("Y ~ x\n|> f()")


def test_named_call_is_not_an_unclosed_pipeline_group():
    with pytest.raises(ParseError) as exc:
        parse_spec("Y ~ f(x\n |> g()\nZ ~ w")
    message = str(exc.value)
    assert "Write '(x |> f())'" not in message
    assert "Write 'f*(x |> f())'" not in message


@pytest.mark.parametrize(
    ("spec", "forbidden"),
    [
        ("Y ~ 2*x |> lag()", "2*(x |> lag())"),
        ("Y ~ a*x |> hsgp(m=20)", "a*(x |> hsgp(m=20))"),
        ("Y ~ 1e-5*x |> fourier()", "1e-5*(x |> fourier())"),
        ("Y ~ -1*x |> bs(df=5)", "-1*(x |> bs(df=5))"),
        ("Y ~ a*x |> C()", "a*(x |> C())"),
    ],
)
def test_prefixed_special_stage_does_not_suggest_grouping(spec, forbidden):
    with pytest.raises(ParseError, match="standalone") as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert forbidden not in message
    assert "needs grouping" not in message


def test_prefixed_group_with_trailing_text_names_the_closed_form():
    with pytest.raises(ParseError) as exc:
        parse_spec("Y ~ 2*(x |> f()) trailing")
    message = str(exc.value)
    assert "2*(x |> f())" in message
    assert "trailing" in message
    assert "2*((x |> f()) trailing)" not in message
    assert "needs grouping" not in message
    assert "x |> f(p=u)" not in message


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ f(g(x, p=a|>b), q=c+d)",
        "Y ~ logistic_saturation(x, lam=a|>b+c)",
        "Y ~ C(group, reference=a|>b+c) + x",
    ],
)
def test_plus_inside_legacy_call_with_keyword_pipe_still_splits(spec):
    with pytest.raises(ParseError, match="Unclosed parenthesis"):
        parse_spec(spec)


@pytest.mark.parametrize("stage", ["BS(df=5)", "NS(df=4)", "PS(df=4)"])
def test_mixed_case_spline_stage_uses_standalone_guard(stage):
    with pytest.raises(ParseError, match="not a standalone spline term"):
        parse_spec(f"Y ~ x |> {stage}")


@pytest.mark.parametrize("stage", ["Hsgp(m=10)", "HSGP(m=10)"])
def test_mixed_case_hsgp_stage_uses_standalone_guard(stage):
    with pytest.raises(ParseError, match="Did you mean hsgp"):
        parse_spec(f"Y ~ x |> {stage}")


def test_mixed_case_ordinary_transform_stage_still_parses():
    assert parse_spec("Y ~ x |> Fourier()") == parse_spec("Y ~ Fourier(x)")
    assert parse_spec("Y ~ x |> LAG()") == parse_spec("Y ~ LAG(x)")


def test_bare_comparison_and_bar_stay_legacy_variable_text():
    assert parse_spec("Y ~ spend>0").regressions[0].terms[0].variable == "spend>0"
    assert parse_spec("Y ~ x>1").regressions[0].terms[0].variable == "x>1"
    assert parse_spec("Y ~ tv>digital").regressions[0].terms[0].variable == "tv>digital"
    assert parse_spec("Y ~ x | group").regressions[0].terms[0].variable == "x | group"
    terms = parse_spec("Y ~ a>b + c").regressions[0].terms
    assert [term.variable for term in terms] == ["a>b", "c"]
    prefixed = parse_spec("Y ~ 2*x > y").regressions[0].terms[0]
    assert prefixed.variable == "x > y"
    assert prefixed.fixed_value == 2.0


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("Y ~ x | > f()", "Empty transform call"),
        ("Y ~ (x | > f())", "Missing transform name"),
        ("Y ~ a*(x | > f())", "Missing transform name"),
        ("Y ~ x ||> f()", "Empty transform call"),
    ],
)
def test_non_token_pipe_punctuation_keeps_legacy_errors(spec, message):
    with pytest.raises(ParseError, match=message):
        parse_spec(spec)


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ f(x, p=a*(b|>c+d)) + z",
        "Y ~ logistic_saturation(x, lam=a*(b|>c+d))",
        "Y ~ C(group, reference=a*(b|>c+d))",
        "Y ~ f(x, p=a*(b\n|> c))\nZ ~ w",
    ],
)
def test_star_inside_legacy_call_does_not_form_a_pipeline_group(spec):
    with pytest.raises(ParseError, match="Unclosed parenthesis"):
        parse_spec(spec)


@pytest.mark.parametrize("spec", ["Y ~ x\n |> f()", "Y ~ x |>\n f()"])
def test_ungrouped_multiline_pipe_is_rejected(spec):
    with pytest.raises(ParseError, match="group"):
        parse_spec(spec)


@pytest.mark.parametrize("spec", ["x |> f() ~ y", "Y ~~ x |> f()", "total := a |> f()"])
def test_pipe_outside_regression_rhs_is_rejected(spec):
    with pytest.raises(ParseError, match="pipe|Pipeline|pipeline"):
        parse_spec(spec)


def test_pipe_preserves_duplicate_label_error():
    with pytest.raises(DuplicateEquationError):
        parse_spec("Y ~ a*(x |> f()) + a*z")


def test_quoted_pipe_in_categorical_reference_is_legacy_content():
    categorical = (
        parse_spec("Y ~ C(group, reference='a|>b')").regressions[0].terms[0].categorical
    )
    assert categorical is not None
    assert categorical.reference == "a|>b"


def test_pipe_characters_in_legacy_keyword_keep_transform_call():
    spec = parse_spec("Y ~ logistic_saturation(x, lam=a|>b)")
    assert spec.regressions[0].terms[0].transform == TransformCall(
        name="logistic_saturation",
        input_expr="x",
        params={"lam": "a|>b"},
    )


def test_nested_legacy_keyword_pipe_is_not_pipeline_syntax():
    spec = parse_spec("Y ~ f(g(x, p=a|>b), q=c|>d)")
    assert spec.regressions[0].terms[0].transform == TransformCall(
        name="f",
        input_expr=TransformCall(name="g", input_expr="x", params={"p": "a|>b"}),
        params={"q": "c|>d"},
    )


@pytest.mark.parametrize(
    "term",
    ["g(x |> f())", "f(x |> g(), p=u)", "f(g(x |> h()), p=u)"],
)
def test_pipe_in_call_input_is_still_rejected(term):
    with pytest.raises(ParseError, match="pipe|pipeline|Pipeline"):
        parse_spec(f"Y ~ {term}")


def test_unquoted_categorical_reference_pipe_stays_legacy():
    categorical = (
        parse_spec("Y ~ C(group, reference=a|>b)").regressions[0].terms[0].categorical
    )
    assert categorical is not None
    assert categorical.reference == "a|>b"


@pytest.mark.parametrize(
    ("reference", "field", "left"),
    [
        ("'a:=b'", "defined_params", "name"),
        ("'a~~b'", "residual_covs", "var1"),
        ('"a:=b"', "defined_params", "name"),
        ('"a~~b"', "residual_covs", "var1"),
    ],
)
@pytest.mark.parametrize("pipe", ["x |> f()", "(x |> f())"])
def test_quoted_statement_operator_beside_pipe_matches_nested(
    reference, field, left, pipe
):
    """Quoted := / ~~ is literal text, so pipe spelling takes the nested misparse."""
    nested = parse_spec(f"Y ~ C(group, reference={reference}) + f(x)")
    piped = parse_spec(f"Y ~ C(group, reference={reference}) + {pipe}")
    assert piped.regressions == nested.regressions == []
    piped_item = getattr(piped, field)[0]
    nested_item = getattr(nested, field)[0]
    assert getattr(piped_item, left) == getattr(nested_item, left)
    right = "expression" if field == "defined_params" else "var2"
    assert getattr(piped_item, right).replace(pipe, "f(x)") == getattr(
        nested_item, right
    )


def test_escaped_quote_in_legacy_term_survives_adjacent_pipe_term():
    pipe = r"Y ~ C(group, reference='a\')b') + x |> f()"
    nested = r"Y ~ C(group, reference='a\')b') + f(x)"
    assert parse_spec(pipe) == parse_spec(nested)


@pytest.mark.parametrize(
    "literal",
    [
        "'''a'b'''",
        "'''a\"b'''",
        '"""a"b"""',
        "'''a') |> (b'''",
        r"'''a\'''b'''",
        r"r'a\')b'",
        "'a' 'b'",
    ],
)
@pytest.mark.parametrize("multiline", [False, True])
def test_legacy_literals_survive_adjacent_pipe(literal, multiline):
    nested = f"Y ~ C(group, reference={literal}) + f(x) + z"
    pipe = "(x\n |> f())" if multiline else "(x |> f())"
    assert parse_spec(f"Y ~ C(group, reference={literal}) + {pipe} + z") == (
        parse_spec(nested)
    )


@pytest.mark.parametrize(
    ("operator", "expected"),
    [(">", [False, False, True]), (">=", [False, True, True])],
)
@pytest.mark.parametrize("predictor", ["x", "(x |> f())"])
def test_defined_comparisons_remain_usable(operator, expected, predictor):
    spec = parse_spec(f"Y ~ a*{predictor}; positive := a {operator} 0")
    assert spec.defined_params[0].expression == f"a {operator} 0"
    draws = xr.DataArray([-1.0, 0.0, 1.0], dims="draw")
    result = evaluate_defined_params(spec, {"a": draws})
    np.testing.assert_array_equal(result["positive"], expected)


def test_defined_expression_keeps_quoted_additive_and_pipe_content():
    spec = parse_spec("Y ~ a*x; same := a * ('a+|>b' == 'a+|>b')")
    draws = np.array([-1.0, 0.0, 1.0])
    result = evaluate_defined_params(spec, {"a": draws})
    assert isinstance(result["same"], np.ndarray)
    np.testing.assert_array_equal(result["same"], draws)


@pytest.mark.parametrize("value", ["a>b", "a>=b", "a|b"])
@pytest.mark.parametrize("pipe", [False, True])
def test_legacy_argument_punctuation_is_not_a_pipeline(value, pipe):
    legacy = f"C(group, reference={value}) + f(x, p={value})"
    suffix = " + (z\n |> g())" if pipe else ""
    nested_suffix = " + g(z)" if pipe else ""
    spec = parse_spec(f"Y ~ {legacy}{suffix}")
    assert spec == parse_spec(f"Y ~ {legacy}{nested_suffix}")
    categorical, transform = spec.regressions[0].terms[:2]
    assert categorical.categorical.reference == value
    assert transform.transform.params == {"p": value}


@pytest.mark.parametrize("reference", ["a>b", "a|b"])
def test_legacy_bare_categorical_reference_can_build_model(reference):
    frame = pd.DataFrame({"group": [reference, "other"], "Y": [0.0, 1.0]})
    fitted = pathmc.model(f"Y ~ C(group, reference={reference})", data=frame)
    assert fitted._spec.regressions[0].terms[0].categorical.reference == reference


@pytest.mark.parametrize("reference", ["O'Brien", "O'Neal", "O'r'", 'North"West'])
def test_bare_reference_quotes_do_not_hide_multiline_pipeline(reference):
    legacy = f"C(group, reference={reference}) + C(region, reference='a|>b')"
    pipe = f"Y ~ {legacy} + (x\n |> logistic_saturation(lam=lam))"
    nested = f"Y ~ {legacy} + logistic_saturation(x, lam=lam)"
    assert parse_spec(pipe) == parse_spec(nested)
    frame = pd.DataFrame({
        "group": [reference, "other"],
        "region": ["a|>b", "other"],
        "x": [0.0, 1.0],
        "Y": [0.0, 1.0],
    })
    assert (
        pathmc.model(pipe, data=frame)._spec == pathmc.model(nested, data=frame)._spec
    )


@pytest.mark.parametrize(
    "reference",
    ["North(West", "North)West", "r'abc", "u'abc", "b'abc", "a>'b", "'r", 'r"abc'],
)
@pytest.mark.parametrize(
    "chain",
    [
        "x |> logistic_saturation(lam=lam)",
        "(x |> logistic_saturation(lam=lam))",
        "(x\n |> logistic_saturation(lam=lam))",
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_legacy_raw_values_remain_opaque_beside_pipeline(reference, chain, position):
    legacy = f"C(group, reference={reference})"
    nested_term = "logistic_saturation(x, lam=lam)"
    pipe_rhs = f"{legacy} + {chain}" if position == "before" else f"{chain} + {legacy}"
    nested_rhs = (
        f"{legacy} + {nested_term}"
        if position == "before"
        else f"{nested_term} + {legacy}"
    )
    pipe = f"Y ~ {pipe_rhs}\nZ ~ z"
    nested = f"Y ~ {nested_rhs}\nZ ~ z"
    assert parse_spec(pipe) == parse_spec(nested)
    frame = pd.DataFrame({
        "group": [reference, "other"],
        "x": [0.0, 1.0],
        "Y": [0.0, 1.0],
        "z": [0.0, 1.0],
        "Z": [0.0, 1.0],
    })
    assert (
        pathmc.model(pipe, data=frame)._spec == pathmc.model(nested, data=frame)._spec
    )


@pytest.mark.parametrize("reference", ["'90s", "r'abc"])
@pytest.mark.parametrize("separator", [" + ", "; ", "\n"])
@pytest.mark.parametrize(
    "chain",
    ["x |> logistic_saturation(lam=lam)", "(x\n |> logistic_saturation(lam=lam))"],
)
def test_legacy_quote_mask_does_not_cross_term_or_statement(
    reference, separator, chain
):
    first = f"C(decade, reference={reference})"
    second = "C(region, reference='north')"
    prefix = f"Y ~ {first} + " if separator == " + " else f"A ~ {first}{separator}Y ~ "
    pipe = f"{prefix}{second} + {chain}"
    nested = f"{prefix}{second} + logistic_saturation(x, lam=lam)"
    assert parse_spec(pipe) == parse_spec(nested)
    frame = pd.DataFrame({
        "decade": [reference, "other"],
        "region": ["north", "south"],
        "x": [0.0, 1.0],
        "Y": [0.0, 1.0],
        "A": [0.0, 1.0],
    })
    assert (
        pathmc.model(pipe, data=frame)._spec == pathmc.model(nested, data=frame)._spec
    )


@pytest.mark.parametrize(
    "reference", ["North)West>East", "North)West|East", "North*((West"]
)
@pytest.mark.parametrize(
    "chain",
    [
        "logistic_saturation(x, lam=lam)",
        "x |> logistic_saturation(lam=lam)",
        "(x\n |> logistic_saturation(lam=lam))",
    ],
)
@pytest.mark.parametrize("position", ["before", "after"])
def test_legacy_call_contents_do_not_control_pipeline_scanners(
    reference, chain, position
):
    legacy = f"C(group, reference={reference})"
    nested_term = "logistic_saturation(x, lam=lam)"
    pipe_rhs = f"{legacy} + {chain}" if position == "before" else f"{chain} + {legacy}"
    nested_rhs = (
        f"{legacy} + {nested_term}"
        if position == "before"
        else f"{nested_term} + {legacy}"
    )
    pipe = f"Y ~ {pipe_rhs}\nZ ~ z"
    nested = f"Y ~ {nested_rhs}\nZ ~ z"
    assert parse_spec(pipe) == parse_spec(nested)
    frame = pd.DataFrame({
        "group": [reference, "other"],
        "x": [0.0, 1.0],
        "Y": [0.0, 1.0],
        "z": [0.0, 1.0],
        "Z": [0.0, 1.0],
    })
    assert (
        pathmc.model(pipe, data=frame)._spec == pathmc.model(nested, data=frame)._spec
    )


class _Identity(Transform):
    name = "pipe_identity_test"
    param_specs = {}
    homogeneous = True

    def apply_pymc(self, x, params, *, panel_info=None, data=None):
        return x


class _Basis(Basis):
    name = "pipe_basis_test"

    def n_basis(self, call):
        return 1

    def build_data(self, x, call, *, state=None):
        return np.asarray(x)[:, None], None


def test_registered_parameter_free_transform_and_custom_basis_boundary():
    old_transform = TRANSFORM_REGISTRY.get(_Identity.name)
    old_basis = BASIS_REGISTRY.get(_Basis.name)
    try:
        register_transform(_Identity())
        register_basis(_Basis())
        assert parse_spec("Y ~ x |> pipe_identity_test()") == parse_spec(
            "Y ~ pipe_identity_test(x)"
        )
        with pytest.raises(ParseError, match="standalone"):
            parse_spec("Y ~ x |> pipe_basis_test()")
    finally:
        if old_transform is None:
            TRANSFORM_REGISTRY.pop(_Identity.name, None)
        else:
            TRANSFORM_REGISTRY[_Identity.name] = old_transform
        if old_basis is None:
            BASIS_REGISTRY.pop(_Basis.name, None)
        else:
            BASIS_REGISTRY[_Basis.name] = old_basis


def test_unknown_transform_resolves_at_existing_model_boundary():
    pipe = "Y ~ x |> unknown_pipe_test(p=u)"
    nested = "Y ~ unknown_pipe_test(x, p=u)"
    assert parse_spec(pipe) == parse_spec(nested)
    data = pd.DataFrame({"x": [0.0, 1.0], "Y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Register custom transforms") as pipe_exc:
        pathmc.model(pipe, data=data)
    with pytest.raises(ValueError) as nested_exc:
        pathmc.model(nested, data=data)
    assert str(pipe_exc.value) == str(nested_exc.value)


PANEL = {"unit": "unit", "time": "time"}
PIPE = "Y ~ 2*(x |> adstock(decay=theta) |> logistic_saturation(lam=lam))"
NESTED = "Y ~ 2*logistic_saturation(adstock(x, decay=theta), lam=lam)"


def _panel_data():
    values = {"A": [1.0, 0.0, 2.0, 0.0], "B": [0.0, 2.0, 0.0, 1.0]}
    return pd.DataFrame([
        {"unit": unit, "time": time, "x": x, "Y": 0.0}
        for unit, series in values.items()
        for time, x in enumerate(series)
    ])


def _oracle(values):
    stock = 0.0
    result = []
    for x in values:
        stock = x + 0.5 * stock
        sat = (1 - np.exp(-0.8 * stock)) / (1 + np.exp(-0.8 * stock))
        result.append(0.3 + 2.0 * sat)
    return np.asarray(result)


def test_pipe_panel_simulation_matches_independent_order_and_coefficient_oracle():
    frame = _panel_data()
    params = {"beta_Y": [0.3], "sigma_Y": 0.0, "theta": 0.5, "lam": 0.8}
    expected = np.concatenate([
        _oracle(frame.loc[frame.unit == unit, "x"].to_numpy()) for unit in ("A", "B")
    ])
    for spec in (PIPE, NESTED):
        result = pathmc.simulate(
            spec, data=frame, params=params, panel=PANEL, random_seed=0
        )
        np.testing.assert_allclose(result["Y"], expected, atol=1e-8)
    x = np.array([1.0, 0.0, 2.0, 0.0])
    sat_then_stock = []
    stock = 0.0
    for value in x:
        saturated_input = (1 - np.exp(-0.8 * value)) / (1 + np.exp(-0.8 * value))
        stock = saturated_input + 0.5 * stock
        sat_then_stock.append(0.3 + 2.0 * stock)
    correct = _oracle(x)
    assert not np.allclose(correct, sat_then_stock)
    stock = 0.0
    coefficient_inside = []
    for value in x:
        stock = value + 0.5 * stock
        coefficient_inside.append(
            0.3 + (1 - np.exp(-0.8 * 2.0 * stock)) / (1 + np.exp(-0.8 * 2.0 * stock))
        )
    assert not np.allclose(correct, coefficient_inside)


def test_pipe_model_metadata_and_time_forward_do_match_nested_oracle():
    frame = _panel_data()
    models = [pathmc.model(spec, data=frame, panel=PANEL) for spec in (PIPE, NESTED)]
    pipe, nested = models
    assert pipe._spec == nested._spec
    assert pipe._graph_info.topological_order == nested._graph_info.topological_order
    assert set(pipe._graph_info._dag.nodes) == set(nested._graph_info._dag.nodes)
    assert set(pipe._graph_info._dag.edges) == set(nested._graph_info._dag.edges)
    assert pipe.pymc_model.coords == nested.pymc_model.coords
    assert {rv.name for rv in pipe.pymc_model.free_RVs} == {
        rv.name for rv in nested.pymc_model.free_RVs
    }
    assert list(pipe.pymc_model.coords["Y_predictors"]) == ["Intercept"]
    pd.testing.assert_frame_equal(pipe.design("Y"), nested.design("Y"))
    assert str(pipe.equations()) == str(nested.equations())
    assert repr(pipe.priors()) == repr(nested.priors())
    assert pipe.graph().source == nested.graph().source

    posterior = xr.Dataset(
        {
            "beta_Y": (("chain", "draw", "Y_predictors"), np.full((1, 2, 1), 0.3)),
            "theta": (("chain", "draw"), np.full((1, 2), 0.5)),
            "lam": (("chain", "draw"), np.full((1, 2), 0.8)),
            "sigma_Y": (("chain", "draw"), np.zeros((1, 2))),
        },
        coords={"chain": [0], "draw": [0, 1], "Y_predictors": ["Intercept"]},
    )
    for model in models:
        compiled = pm.do(
            model._gen_model,
            {
                "beta_Y": np.array([0.3]),
                "theta": np.array(0.5),
                "lam": np.array(0.8),
                "sigma_Y": np.array(0.0),
            },
        )
        expected_by_unit = np.stack(
            [
                _oracle(frame.loc[frame.unit == unit, "x"].to_numpy())
                for unit in ("A", "B")
            ],
            axis=1,
        )
        np.testing.assert_allclose(
            pm.draw(compiled["mu_Y"], draws=1, random_seed=0),
            expected_by_unit,
            atol=1e-8,
        )
        model._idata = xr.DataTree.from_dict({"posterior": posterior.copy(deep=True)})
        natural = model.do(set=None, simulate_over="time", kind="mean")
        forced = model.do(set={"x": 1.0}, simulate_over="time", kind="mean")
        np.testing.assert_allclose(
            natural.by_time("Y").mean(axis=1), expected_by_unit.mean(axis=1), atol=1e-8
        )
        forced_expected = _oracle(np.ones(4))
        np.testing.assert_allclose(
            forced.by_time("Y").mean(axis=1), forced_expected, atol=1e-8
        )


def test_distinct_and_shared_chains_keep_their_panel_state():
    frame = _panel_data()
    pipe = (
        "M ~ 0 + x |> adstock(decay=t1) |> logistic_saturation(lam=l1)\n"
        "Y ~ 0 + x |> adstock(decay=t1) |> logistic_saturation(lam=l1) "
        "+ x |> adstock(decay=t2) |> logistic_saturation(lam=l2)"
    )
    nested = (
        "M ~ 0 + logistic_saturation(adstock(x, decay=t1), lam=l1)\n"
        "Y ~ 0 + logistic_saturation(adstock(x, decay=t1), lam=l1) "
        "+ logistic_saturation(adstock(x, decay=t2), lam=l2)"
    )
    assert parse_spec(pipe) == parse_spec(nested)
    params = {
        "beta_M": [1.0],
        "sigma_M": 0.0,
        "beta_Y": [1.0, 1.0],
        "sigma_Y": 0.0,
        "t1": 0.5,
        "t2": 0.2,
        "l1": 0.8,
        "l2": 0.4,
    }
    for spec in (pipe, nested):
        result = pathmc.simulate(spec, data=frame, params=params, panel=PANEL)
        expected_m = []
        expected_y = []
        for unit in ("A", "B"):
            x = frame.loc[frame.unit == unit, "x"].to_numpy()
            first = _oracle(x) - 0.3
            stock = 0.0
            second = []
            for value in x:
                stock = value + 0.2 * stock
                second.append((1 - np.exp(-0.4 * stock)) / (1 + np.exp(-0.4 * stock)))
            expected_m.extend(first / 2)
            expected_y.extend(first / 2 + np.asarray(second))
        np.testing.assert_allclose(result["M"], expected_m, atol=1e-8)
        np.testing.assert_allclose(result["Y"], expected_y, atol=1e-8)


def test_nonunit_input_scaling_has_same_pipe_and_nested_semantics():
    frame = _panel_data()
    scale = Scaling(channel={"method": "fixed", "value": 10.0})
    params = {"beta_Y": [0.3], "sigma_Y": 0.0, "theta": 0.5, "lam": 0.8}
    outputs = [
        pathmc.simulate(spec, data=frame, params=params, panel=PANEL, scaling=scale)[
            "Y"
        ].to_numpy()
        for spec in (PIPE, NESTED)
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
    expected = np.concatenate([
        _oracle(frame.loc[frame.unit == unit, "x"].to_numpy() / 10)
        for unit in ("A", "B")
    ])
    np.testing.assert_allclose(outputs[0], expected, atol=1e-8)
