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
"""Parser-level tests for the ``hsgp()`` DSL term (fast, no PyMC)."""

from __future__ import annotations

import pytest

from pathmc.exceptions import ParseError
from pathmc.parse import HSGPCall, parse_spec


def _term(spec_string: str):
    spec = parse_spec(spec_string)
    return spec.regressions[0].terms[0]


def test_parses_basic_hsgp_term():
    term = _term("y ~ hsgp(x, m=20, c=1.5)")
    assert term.variable == "x"
    assert term.hsgp == HSGPCall(
        variable="x", m=20, c=1.5, L=None, cov="expquad", centered=False
    )


def test_literal_coercion_types():
    call = _term("y ~ hsgp(x, m=20, c=1.5)").hsgp
    assert isinstance(call.m, int)
    assert isinstance(call.c, float)
    assert call.L is None
    assert isinstance(call.cov, str)
    assert isinstance(call.centered, bool)


def test_explicit_L_variant():
    call = _term("y ~ hsgp(x, m=8, L=3.0)").hsgp
    assert call.L == 3.0
    assert call.c is None
    assert isinstance(call.L, float)


def test_matern_cov_variant_is_lowercased():
    call = _term("y ~ hsgp(x, m=8, c=1.5, cov='Matern52')").hsgp
    assert call.cov == "matern52"


def test_centered_true_variant():
    call = _term("y ~ hsgp(x, m=8, c=1.5, centered=true)").hsgp
    assert call.centered is True


@pytest.mark.parametrize(
    "spec_string",
    [
        "y ~ hsgp(x, c=1.5)",  # missing m
        "y ~ hsgp(x, m=8, c=1.5, L=3)",  # both c and L
        "y ~ hsgp(x, m=8)",  # neither c nor L
        "y ~ hsgp(x, m=8, c=1.5, by=g)",  # unknown kwarg
        "y ~ hsgp(x1, x2, m=8, c=1.5)",  # multiple positional inputs
        "y ~ adstock(hsgp(x, m=8, c=1.5), decay=d)",  # nested in transform
        "y ~ hsgp(x, m=8, c=1.5):z",  # inside interaction
        "y ~ 2*hsgp(x, m=8, c=1.5)",  # numeric coefficient prefix
        "y ~ b*hsgp(x, m=8, c=1.5)",  # label coefficient prefix
        "y ~ hsgp(x, m=8, c=1.5, cov=rbf)",  # unknown kernel
        "y ~ hsgp(x, m=1.5, c=1.5)",  # non-integer m
        "y ~ hsgp(m=8, c=1.5)",  # keyword-only, no positional input variable
    ],
)
def test_guardrails_raise_parse_error(spec_string):
    with pytest.raises(ParseError):
        parse_spec(spec_string)


def test_duplicate_lhs_var_hsgp_rejected():
    with pytest.raises(ParseError, match="same variable"):
        parse_spec("y ~ hsgp(x, m=20, c=1.5) + hsgp(x, m=30, c=2.0)")


def test_two_hsgp_terms_on_different_vars_allowed():
    spec = parse_spec("y ~ hsgp(x, m=20, c=1.5) + hsgp(z, m=10, c=2.0)")
    hsgp_vars = [t.hsgp.variable for t in spec.regressions[0].terms]
    assert hsgp_vars == ["x", "z"]
