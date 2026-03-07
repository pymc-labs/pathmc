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
class TransformCall:
    """A named transform applied to a variable or nested transform.

    Examples::

        adstock(tv, decay=theta)  ->  TransformCall("adstock", "tv", {"decay": "theta"})
        logistic_saturation(adstock(tv, decay=theta), lam=lam)
          ->  TransformCall("logistic_saturation", TransformCall(...), {"lam": "lam"})
    """

    name: str
    input_expr: str | "TransformCall"
    params: dict[str, str]


@dataclass
class Term:
    """A single predictor term in a regression equation.

    When ``fixed_value`` is set, the coefficient is pinned to that
    numeric constant and no free parameter is created.  The syntax
    ``1*awareness`` produces ``Term(variable="awareness", fixed_value=1.0)``.
    """

    variable: str
    label: str | None = None
    transform: TransformCall | None = None
    lag_of: str | None = None
    interaction_of: tuple[str, ...] | None = None
    fixed_value: float | None = None


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


def _join_continuation_lines(spec_string: str) -> str:
    """Join lines that start with ``+`` to the preceding line.

    This allows multi-line regression statements like::

        sales ~ b_tv*adstock(tv, decay=theta)
              + b_dig*logistic_saturation(digital, lam=lam)
              + trend
    """
    lines = spec_string.split("\n")
    merged: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("+") and merged:
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(line)
    return "\n".join(merged)


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
    spec_string = _join_continuation_lines(spec_string)
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
    """Parse a single term, optionally with a coefficient label and/or transform.

    Numeric labels (e.g. ``1*X``, ``0.5*X``) are treated as fixed
    coefficient values rather than free parameter names.
    """
    raw = raw.strip()
    label: str | None = None
    fixed_value: float | None = None

    if "*" in raw:
        star_pos = _find_top_level_star(raw)
        if star_pos is not None:
            label_str = raw[:star_pos].strip()
            raw = raw[star_pos + 1 :].strip()
            if not label_str:
                raise ParseError(f"Malformed labeled term: '{raw}'.")
            try:
                fixed_value = float(label_str)
                label = None
            except ValueError:
                label = label_str

    if "(" in raw:
        transform = _parse_transform_expr(raw)
        if transform.name == "lag":
            term = _make_lag_term(transform, raw, label)
            term.fixed_value = fixed_value
            return term
        variable = _extract_leaf_variable(transform)
        return Term(
            variable=variable,
            label=label,
            transform=transform,
            fixed_value=fixed_value,
        )

    if ":" in raw:
        term = _parse_interaction_term(raw, label)
        term.fixed_value = fixed_value
        return term

    variable = raw.strip()
    if not variable:
        raise ParseError("Empty variable name in term.")
    return Term(variable=variable, label=label, fixed_value=fixed_value)


def _make_lag_term(tc: TransformCall, raw: str, label: str | None) -> Term:
    """Build a ``Term`` from a ``lag(var)`` expression.

    ``lag()`` is a structural term (not a real transform) that
    references the previous time step of a variable.  Only lag-1 is
    supported — higher-order lags are rejected by design.
    """
    if tc.params:
        raise ParseError(
            f"lag() does not accept parameters: '{raw}'. "
            f"pathmc supports lag-1 only by design. The influence of "
            f"t-2 on t should be mediated through t-1."
        )
    if isinstance(tc.input_expr, TransformCall):
        raise ParseError(
            f"lag() only accepts a plain variable name, not a "
            f"nested transform: '{raw}'."
        )
    base_var = tc.input_expr
    return Term(variable=f"lag({base_var})", label=label, lag_of=base_var)


def _parse_interaction_term(raw: str, label: str | None) -> Term:
    """Parse an interaction term like ``X:Z`` or ``X:Z:W``."""
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) < 2:
        raise ParseError(
            f"Malformed interaction term: '{raw}'. "
            "Use 'X:Z' syntax for two-way interactions."
        )
    for p in parts:
        if not p:
            raise ParseError(f"Empty variable name in interaction term: '{raw}'.")
        if not re.match(r"^[A-Za-z_]\w*$", p):
            raise ParseError(
                f"Invalid variable name '{p}' in interaction term '{raw}'. "
                "Interaction terms only support plain variable names (no transforms)."
            )
    variable = ":".join(parts)
    return Term(variable=variable, label=label, interaction_of=tuple(parts))


def _find_top_level_star(raw: str) -> int | None:
    """Find the position of ``*`` that is NOT inside parentheses."""
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "*" and depth == 0:
            return i
    return None


def _parse_transform_expr(raw: str) -> TransformCall:
    """Recursively parse ``name(input, key=val, ...)``."""
    raw = raw.strip()
    paren_open = raw.index("(")
    name = raw[:paren_open].strip()
    if not name:
        raise ParseError(
            f"Missing transform name in '{raw}'. Expected a function name before '('."
        )

    if raw[-1] != ")":
        raise ParseError(
            f"Unclosed parenthesis in transform expression: '{raw}'. Add a closing ')'."
        )

    inner = raw[paren_open + 1 : -1].strip()
    args = _split_top_level_args(inner)

    if not args:
        raise ParseError(
            f"Empty transform call: '{raw}'. Provide at least an input variable."
        )

    input_raw = args[0].strip()
    if not input_raw:
        raise ParseError(f"Missing input variable in transform: '{raw}'.")

    if "=" in input_raw and "(" not in input_raw:
        raise ParseError(
            f"No input variable in transform: '{raw}'. "
            "The first argument must be a variable or nested transform, "
            "not a keyword parameter."
        )

    if "(" in input_raw:
        input_expr: str | TransformCall = _parse_transform_expr(input_raw)
    else:
        input_expr = input_raw

    params: dict[str, str] = {}
    for arg in args[1:]:
        if "=" not in arg:
            raise ParseError(
                f"Expected keyword parameter (key=value) in transform, "
                f"got '{arg.strip()}' in '{raw}'."
            )
        key, _, val = arg.partition("=")
        key = key.strip()
        val = val.strip()
        if not key:
            raise ParseError(f"Missing parameter name in '{raw}'.")
        if not val:
            raise ParseError(
                f"Missing value for parameter '{key}' in '{raw}'. "
                f"Provide a parameter name after '='."
            )
        params[key] = val

    return TransformCall(name=name, input_expr=input_expr, params=params)


def _split_top_level_args(s: str) -> list[str]:
    """Split a comma-separated argument list, respecting nested parens."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _extract_leaf_variable(tc: TransformCall) -> str:
    """Walk a (possibly nested) TransformCall to find the leaf input variable."""
    if isinstance(tc.input_expr, str):
        return tc.input_expr
    return _extract_leaf_variable(tc.input_expr)


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
