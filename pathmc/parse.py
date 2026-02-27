"""DSL parser for pathmc spec strings.

Parses a lavaan-inspired formula DSL into typed AST nodes.
Supports regression (~), residual covariance (~~), defined parameters (:=),
labeled coefficients (label*variable), and intercept suppression (0 +).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pathmc.exceptions import DuplicateEquationError, ParseError


@dataclass
class Term:
    """A single predictor term in a regression equation."""

    variable: str
    label: str | None = None


@dataclass
class Regression:
    """A parsed regression equation: lhs ~ terms."""

    lhs: str
    terms: list[Term]
    has_intercept: bool = True


@dataclass
class ResidualCov:
    """A residual covariance declaration: var1 ~~ var2."""

    var1: str
    var2: str


@dataclass
class DefinedParam:
    """A user-defined derived parameter: name := expression."""

    name: str
    expression: str


@dataclass
class Spec:
    """Complete parsed specification of a structural model."""

    regressions: list[Regression] = field(default_factory=list)
    residual_covs: list[ResidualCov] = field(default_factory=list)
    defined_params: list[DefinedParam] = field(default_factory=list)


def parse_spec(spec_string: str) -> Spec:
    """Parse a DSL specification string into a Spec object.

    Parameters
    ----------
    spec_string : str
        Multi-line or semicolon-separated specification using lavaan-style
        syntax (``~``, ``~~``, ``:=``).

    Returns
    -------
    Spec
        Structured representation of the parsed model.

    Raises
    ------
    ParseError
        On malformed syntax (missing RHS, invalid operators, etc.).
    DuplicateEquationError
        If two regressions share the same LHS variable.
    """
    raw_statements = re.split(r"[;\n]", spec_string)
    statements = [s.strip() for s in raw_statements if s.strip()]

    if not statements:
        raise ParseError(
            "Empty specification. Provide at least one structural equation."
        )

    regressions: list[Regression] = []
    residual_covs: list[ResidualCov] = []
    defined_params: list[DefinedParam] = []
    seen_lhs: set[str] = set()

    for stmt in statements:
        if ":=" in stmt:
            defined_params.append(_parse_defined_param(stmt))
        elif "~~" in stmt:
            residual_covs.append(_parse_residual_cov(stmt))
        elif "~" in stmt:
            reg = _parse_regression(stmt)
            if reg.lhs in seen_lhs:
                raise DuplicateEquationError(
                    f"Duplicate equation for '{reg.lhs}'. "
                    "Each variable can appear as LHS in at most one regression."
                )
            seen_lhs.add(reg.lhs)
            regressions.append(reg)
        else:
            raise ParseError(
                f"Unrecognized statement: '{stmt}'. "
                "Expected a regression (~), residual covariance (~~), "
                "or defined parameter (:=)."
            )

    return Spec(
        regressions=regressions,
        residual_covs=residual_covs,
        defined_params=defined_params,
    )


# ---------------------------------------------------------------------------
# Statement-level parsers
# ---------------------------------------------------------------------------


def _parse_regression(stmt: str) -> Regression:
    """Parse ``lhs ~ term1 + term2 ...`` into a Regression."""
    parts = stmt.split("~")
    if len(parts) != 2:
        raise ParseError(
            f"Malformed regression: '{stmt}'. Expected exactly one '~' operator."
        )

    lhs = parts[0].strip()
    rhs = parts[1].strip()

    if not lhs:
        raise ParseError(f"Missing left-hand side in: '{stmt}'.")
    if not rhs:
        raise ParseError(
            f"Missing right-hand side in: '{stmt}'. "
            "Provide at least one predictor variable after '~'."
        )

    if not re.match(r"^[A-Za-z_]\w*$", lhs):
        raise ParseError(
            f"Invalid variable name '{lhs}' on left-hand side of '{stmt}'."
        )

    raw_terms = [t.strip() for t in rhs.split("+")]

    has_intercept = True
    terms: list[Term] = []

    for raw in raw_terms:
        if not raw:
            raise ParseError(f"Empty term in regression: '{stmt}'.")
        if raw == "0":
            has_intercept = False
            continue
        terms.append(_parse_term(raw))

    if not terms:
        raise ParseError(
            f"No predictor terms in regression: '{stmt}'. "
            "Add at least one predictor variable."
        )

    return Regression(lhs=lhs, terms=terms, has_intercept=has_intercept)


def _parse_term(raw: str) -> Term:
    """Parse a single term, optionally with a coefficient label (``label*var``)."""
    if "*" in raw:
        label_str, _, var_str = raw.partition("*")
        label = label_str.strip()
        variable = var_str.strip()
        if not label or not variable:
            raise ParseError(f"Malformed labeled term: '{raw}'.")
        return Term(variable=variable, label=label)

    variable = raw.strip()
    if not variable:
        raise ParseError("Empty variable name in term.")
    return Term(variable=variable, label=None)


def _parse_residual_cov(stmt: str) -> ResidualCov:
    """Parse ``var1 ~~ var2`` into a ResidualCov."""
    parts = stmt.split("~~")
    if len(parts) != 2:
        raise ParseError(f"Malformed residual covariance: '{stmt}'.")

    var1 = parts[0].strip()
    var2 = parts[1].strip()

    if not var1 or not var2:
        raise ParseError(
            f"Missing variable in residual covariance: '{stmt}'. "
            "Both sides of '~~' must name a variable."
        )

    return ResidualCov(var1=var1, var2=var2)


def _parse_defined_param(stmt: str) -> DefinedParam:
    """Parse ``name := expression`` into a DefinedParam."""
    name_str, _, expr_str = stmt.partition(":=")
    name = name_str.strip()
    expression = expr_str.strip()

    if not name:
        raise ParseError(f"Missing parameter name in: '{stmt}'.")
    if not expression:
        raise ParseError(
            f"Missing expression in defined parameter: '{stmt}'. "
            "Provide an expression after ':='."
        )

    return DefinedParam(name=name, expression=expression)
