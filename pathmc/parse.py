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
"""DSL parser for pathmc spec strings.

Parses a lavaan-inspired formula DSL into typed AST nodes.
Supports regression (~), residual covariance (~~), defined parameters (:=),
labeled coefficients (label*variable), and intercept suppression (0 +).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pathmc.exceptions import DuplicateEquationError, ParseError

__all__: list[str] = []


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
class HSGPCall:
    """Parsed ``hsgp(...)`` term (Phase 1: a single 1-D input).

    A Hilbert Space Gaussian Process approximation, exposed as a DSL term
    ``hsgp(x, m=..., c=...)`` analogous to Bambi's ``hsgp()``.  Unlike a
    :class:`TransformCall`, the keyword values are compile-time *literals*
    (``m=20``, ``c=1.5``), not random-variable names.

    Parameters
    ----------
    variable : str
        Name of the input column the smooth is a function of.
    m : int
        Number of Laplacian eigenfunction basis vectors. Compile-time literal.
    c : float | None
        Boundary-condition expansion factor. Exactly one of ``c`` or ``L``.
    L : float | None
        Explicit boundary as a user-facing scalar. Exactly one of ``c`` or
        ``L``.  The compiler wraps it into the one-element sequence ``[L]``
        that ``pm.gp.HSGP`` requires.
    cov : str
        Covariance kernel: ``"expquad"``, ``"matern52"``, or ``"matern32"``.
    centered : bool
        If ``True`` use the centered parametrization; otherwise non-centered.
    """

    variable: str
    m: int
    c: float | None = None
    L: float | None = None
    cov: str = "expquad"
    centered: bool = False


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
    hsgp: HSGPCall | None = None


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
        if raw == "1":
            continue
        terms.append(_parse_term(raw))

    if not terms:
        raise ParseError(
            f"No predictor terms in regression: '{stmt}'. "
            "Add at least one predictor variable."
        )

    hsgp_vars = [t.hsgp.variable for t in terms if t.hsgp is not None]
    duplicates = sorted({v for v in hsgp_vars if hsgp_vars.count(v) > 1})
    if duplicates:
        raise ParseError(
            f"Two hsgp() terms on the same variable {duplicates} in equation "
            f"'{lhs}' would collide. Use a single hsgp() per variable per "
            "equation (multiple smooths of one variable are a follow-up)."
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
        func_name = raw[: raw.index("(")].strip()
        if func_name == "hsgp":
            if label is not None or fixed_value is not None:
                raise ParseError(
                    "hsgp(...) cannot take a coefficient prefix; the smooth "
                    "carries its own basis weights. Remove the 'k*' or "
                    "'label*' prefix."
                )
            call = _parse_hsgp_expr(raw)
            return Term(variable=call.variable, hsgp=call)
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
        if input_raw[: input_raw.index("(")].strip() == "hsgp":
            raise ParseError(
                "hsgp(...) cannot be nested inside a transform. "
                "Apply hsgp() directly to a variable."
            )
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


_HSGP_ALLOWED_KWARGS = frozenset({"m", "c", "L", "cov", "centered"})
_HSGP_VALID_COV = frozenset({"expquad", "matern52", "matern32"})


def _parse_hsgp_expr(raw: str) -> HSGPCall:
    """Parse ``hsgp(x, m=..., c=..., cov=..., centered=...)`` into an HSGPCall.

    Unlike transform parsing, keyword values are parsed as literals
    (int / float / bool / bare string), not as random-variable names.

    Raises
    ------
    ParseError
        If ``m`` is missing, both/neither of ``c``/``L`` are given, an
        unknown keyword is used, more than one positional input is given,
        or a value cannot be coerced to its expected literal type.
    """
    raw = raw.strip()
    paren_open = raw.index("(")
    if raw[-1] != ")":
        raise ParseError(
            f"Malformed hsgp() term or combined with another operator: '{raw}'. "
            "Use hsgp() as a standalone term applied directly to one variable, "
            "e.g. 'y ~ hsgp(x, m=20, c=1.5)'."
        )

    inner = raw[paren_open + 1 : -1].strip()
    args = _split_top_level_args(inner)
    if not args or not args[0].strip():
        raise ParseError(
            "hsgp(...) requires an input variable as its first argument. "
            "Example: hsgp(x, m=20, c=1.5)."
        )

    positional: list[str] = []
    kwargs: dict[str, str] = {}
    for arg in args:
        key_part = arg.split("=", 1)[0]
        is_kwarg = "=" in arg and "(" not in key_part
        if is_kwarg:
            key, _, val = arg.partition("=")
            key = key.strip()
            val = val.strip()
            if not key or not val:
                raise ParseError(f"Malformed keyword argument in hsgp(...): '{arg}'.")
            if key in kwargs:
                raise ParseError(f"Duplicate keyword '{key}' in hsgp(...).")
            kwargs[key] = val
        else:
            if kwargs:
                raise ParseError(
                    f"Positional argument '{arg.strip()}' after a keyword "
                    f"argument in hsgp(...): '{raw}'."
                )
            positional.append(arg.strip())

    if len(positional) > 1:
        joined = ", ".join(positional)
        raise ParseError(
            f"Multi-dimensional hsgp({joined}) is not supported yet "
            "(see follow-up). Use a single input variable."
        )
    # A keyword-only call (e.g. ``hsgp(m=20, c=1.5)``) leaves ``positional``
    # empty; the args[0]-non-empty guard above does not catch it, since the
    # first arg is a kwarg. Reject explicitly so the user gets a ParseError
    # instead of a raw IndexError from ``positional[0]``.
    if not positional:
        raise ParseError(
            "hsgp(...) requires an input variable as its first argument. "
            "Example: hsgp(x, m=20, c=1.5)."
        )
    variable = positional[0]
    if not re.match(r"^[A-Za-z_]\w*$", variable):
        raise ParseError(
            f"hsgp(...) input must be a plain variable name, got '{variable}'."
        )

    unknown = sorted(set(kwargs) - _HSGP_ALLOWED_KWARGS)
    if unknown:
        raise ParseError(
            f"hsgp(...) does not support {unknown} in Phase 1. "
            "Supported: m, c, L, cov, centered."
        )

    if "m" not in kwargs:
        raise ParseError(
            "hsgp(...) requires m=<int> (number of basis vectors). "
            "Example: hsgp(x, m=20, c=1.5)."
        )
    try:
        m = int(kwargs["m"])
    except ValueError:
        raise ParseError(
            f"hsgp(...) m must be an integer, got '{kwargs['m']}'."
        ) from None
    if m < 1:
        raise ParseError(f"hsgp(...) m must be >= 1, got {m}.")

    has_c = "c" in kwargs
    has_l = "L" in kwargs
    if has_c == has_l:
        raise ParseError("hsgp(...) needs exactly one of c=<float> or L=<float>.")

    c: float | None = None
    boundary_l: float | None = None
    if has_c:
        try:
            c = float(kwargs["c"])
        except ValueError:
            raise ParseError(
                f"hsgp(...) c must be a number, got '{kwargs['c']}'."
            ) from None
    if has_l:
        try:
            boundary_l = float(kwargs["L"])
        except ValueError:
            raise ParseError(
                f"hsgp(...) L must be a number, got '{kwargs['L']}'."
            ) from None

    cov = kwargs.get("cov", "expquad").strip().strip("'\"").lower()
    if cov not in _HSGP_VALID_COV:
        raise ParseError(
            f"hsgp(...) cov must be one of {sorted(_HSGP_VALID_COV)}, got '{cov}'."
        )

    centered_raw = kwargs.get("centered", "false").strip().strip("'\"").lower()
    if centered_raw not in ("true", "false"):
        raise ParseError(
            f"hsgp(...) centered must be true or false, got '{kwargs['centered']}'."
        )

    return HSGPCall(
        variable=variable,
        m=m,
        c=c,
        L=boundary_l,
        cov=cov,
        centered=centered_raw == "true",
    )


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
