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

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pathmc.exceptions import DuplicateEquationError, ParseError
from pathmc.spline_policy import SPLINE_NAMES, spanned_degree, spline_call_problem

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
    identifier: str | None = None

    @property
    def name(self) -> str:
        """Return the registry name for this backwards-compatible call node."""
        return "hsgp"


@dataclass
class BasisCall:
    """A parsed basis-term call with literal, basis-specific parameters.

    Basis terms produce one or more columns and own their coefficient vector,
    unlike transforms, which preserve the one-column/one-coefficient contract.
    """

    name: str
    variable: str
    params: dict[str, int | float | tuple[float, ...]]
    identifier: str | None = None


@dataclass
class CategoricalCall:
    """A categorical predictor expanded into indicator columns.

    ``levels``, ``reference``, and ``columns`` are fit-time state. They are
    populated from the observed data before compilation and then reused for
    prediction and interventions so the coding cannot silently change.
    Which term supplies the baseline is not stored on the call.
    """

    variable: str
    reference: Any | None = None
    prior: Literal["independent", "hierarchical"] = "independent"
    levels: tuple[Any, ...] = ()
    columns: tuple[str, ...] = ()


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
    basis: BasisCall | HSGPCall | None = None
    categorical: CategoricalCall | None = None

    @property
    def hsgp(self) -> HSGPCall | None:
        """Return the HSGP call for backwards-compatible internal access."""
        return self.basis if isinstance(self.basis, HSGPCall) else None


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


def _unquoted_source(text: str) -> str:
    """Mask literals within the existing additive and statement boundaries.

    Statements split on ``;`` and physical newlines; regression terms also
    split on ``+``, even in quoted content. Defined expressions retain their
    quoted operands. Keeping these existing boundaries prevents raw fallback
    values such as ``'90s`` from pairing with a quote in another term.
    """

    def mask_statement(match: re.Match[str]) -> str:
        statement = match[0]
        if ":=" in statement:
            return _mask_legacy_literals(statement)
        return re.sub(r"[^+]+", lambda term: _mask_legacy_literals(term[0]), statement)

    return re.sub(r"[^;\n]+", mask_statement, text)


def _mask_legacy_literals(text: str) -> str:
    """Mask complete single/triple-quoted spans, preserving source offsets."""
    masked = list(text)
    i = 0
    while i < len(text):
        if text[i] not in {"'", '"'}:
            i += 1
            continue
        start = i
        quote = text[i] * (3 if text.startswith(text[i] * 3, i) else 1)
        i += len(quote)
        while i < len(text):
            if text[i] == "\\":
                i += 2
            elif text.startswith(quote, i):
                i += len(quote)
                masked[start:i] = " " * (i - start)
                break
            else:
                i += 1
    return "".join(masked)


def _pipe_token_at(text: str, index: int) -> bool:
    """True when ``text[index:]`` is a ``|>`` token, not the tail of ``||>``."""
    return text.startswith("|>", index) and (index == 0 or text[index - 1] != "|")


def _find_pipe_token(text: str) -> int | None:
    """Index of a ``|>`` token, ignoring ``||>`` and a bare ``|`` or ``>``."""
    start = 0
    while True:
        index = text.find("|>", start)
        if index < 0:
            return None
        if _pipe_token_at(text, index):
            return index
        start = index + 1


def _has_pipe(text: str) -> bool:
    """Whether a ``|>`` token occurs outside a legacy literal."""
    return _find_pipe_token(_unquoted_source(text)) is not None


def _has_expression_pipe(expr: str) -> bool:
    """Whether an unquoted ``|>`` is pipeline syntax, not keyword text.

    A legacy call stores keyword values as text, so ``f(x, p=a|>b)`` must not
    enter the pipeline grammar. ``g(x |> f())`` must: the pipe is in an
    expression position.
    """
    masked = _unquoted_source(expr)
    if _find_pipe_token(masked) is None:
        return False
    depth = 0
    keyword_depths: list[int] = []
    index = 0
    while index < len(masked):
        ch = masked[index]
        if ch == "(":
            depth += 1
        elif ch == ")":
            if keyword_depths and depth == keyword_depths[-1]:
                keyword_depths.pop()
            depth -= 1
        elif ch == "," and keyword_depths and depth == keyword_depths[-1]:
            keyword_depths.pop()
        elif (
            ch == "="
            and depth > 0
            and (not keyword_depths or depth > keyword_depths[-1])
        ):
            keyword_depths.append(depth)
        elif _pipe_token_at(masked, index):
            if not keyword_depths:
                return True
            index += 2
            continue
        index += 1
    return False


def _pipe_confined_to_keyword_values(expr: str) -> bool:
    """True when every unquoted ``|>`` token is legacy call keyword text."""
    if _find_pipe_token(_unquoted_source(expr)) is None:
        return False
    return not _has_expression_pipe(expr)


def _should_parse_as_pipeline(raw: str) -> bool:
    """Whether ``raw`` uses pipeline grammar rather than a legacy call."""
    if not _has_pipe(raw):
        return False
    return not _pipe_confined_to_keyword_values(raw)


def _matching_paren(masked: str, open_at: int) -> int | None:
    """Close-paren index matching ``masked[open_at]`` on already-masked text."""
    depth = 0
    for index in range(open_at, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


_NEXT_EQUATION = re.compile(r"^[A-Za-z_]\w*\s*(?:~~|:=|~)")


def _pipe_before_next_equation(masked: str, start: int) -> bool:
    """Whether ``|>`` appears after ``start`` before the next equation."""
    index = start
    while index < len(masked):
        if masked[index] == "\n":
            next_line = masked[index + 1 :].split("\n", 1)[0].lstrip()
            if _NEXT_EQUATION.match(next_line):
                return _find_pipe_token(masked[start:index]) is not None
        index += 1
    return _find_pipe_token(masked[start:]) is not None


def _coefficient_before(text: str, open_at: int) -> str | None:
    """Coefficient token immediately before ``text[open_at]``, if any."""
    index = open_at - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    if index < 0 or text[index] != "*":
        return None
    index -= 1
    while index >= 0 and text[index].isspace():
        index -= 1
    end = index + 1
    while index >= 0 and text[index] not in " \t\n+;~":
        index -= 1
    token = text[index + 1 : end]
    return token or None


def _closed_group_example(text: str, open_at: int) -> str:
    """Closed-form suggestion using the real coefficient when one is present."""
    token = _coefficient_before(text, open_at)
    example = "(x |> f())"
    if token:
        return f"{token}*{example}"
    return example


def _group_introducer(masked: str, index: int) -> bool:
    """True when ``masked[index]`` opens a pipeline group after ``~``, ``+``, ``*``, or start.

    A group contains ``|>``, including an unclosed introducer paren when the
    pipe appears before the next equation. A closed span followed by ``|>``,
    such as ``(x + y) |> f()``, is also a group so the sum is not split
    before the pipeline diagnostic. A call such as ``f(x, p=a|>b)`` is not a
    group: its ``(`` is not preceded by ``~``, ``+``, ``*``, or the start.
    """
    if index >= len(masked) or masked[index] != "(":
        return False
    previous = masked[:index].rstrip()[-1:]
    if previous not in {"", "~", "+", "*"}:
        return False
    close = _matching_paren(masked, index)
    if close is None:
        return _pipe_before_next_equation(masked, index)
    if _find_pipe_token(masked[index : close + 1]) is not None:
        return True
    rest = masked[close + 1 :].lstrip()
    return _pipe_token_at(rest, 0)


def _note_call_char(ch: str, depth: int, keyword_depths: list[int]) -> tuple[int, bool]:
    """Advance call depth. The flag is whether this character is keyword text.

    Parentheses inside a keyword value are literal text: an unclosed ``(`` in
    ``reference=North(West`` must not hide a following depth-0 pipeline group.
    An extra ``)`` outside a call stays at depth 0.
    """
    in_keyword = bool(keyword_depths) and depth == keyword_depths[-1]
    if ch == "(" and not in_keyword:
        depth += 1
    elif ch == ")":
        if keyword_depths and depth == keyword_depths[-1]:
            keyword_depths.pop()
        if depth > 0:
            depth -= 1
    elif ch == "," and keyword_depths and depth == keyword_depths[-1]:
        keyword_depths.pop()
    elif ch == "=" and depth > 0 and (not keyword_depths or depth > keyword_depths[-1]):
        keyword_depths.append(depth)
    return depth, in_keyword


def _depth0_group_opens(masked: str) -> set[int]:
    """Indexes of pipeline groups that are not inside a call or keyword value."""
    opens: set[int] = set()
    depth = 0
    keyword_depths: list[int] = []
    for index, ch in enumerate(masked):
        in_keyword = bool(keyword_depths) and depth == keyword_depths[-1]
        if (
            ch == "("
            and not in_keyword
            and depth == 0
            and _group_introducer(masked, index)
        ):
            opens.add(index)
        depth, _ = _note_call_char(ch, depth, keyword_depths)
    return opens


def _pipeline_group_spans(text: str) -> list[tuple[int, int]]:
    """Spans of depth-0 pipeline groups, not every call containing ``|>``."""
    masked = _unquoted_source(text)
    spans: list[tuple[int, int]] = []
    for index in sorted(_depth0_group_opens(masked)):
        close = _matching_paren(masked, index)
        if close is not None:
            spans.append((index, close))
    return spans


def _split_rhs_terms(rhs: str) -> list[str]:
    """Split additive terms without tearing a grouped pipeline.

    ``a*(x |> f()) + z`` is two terms. ``a*(x |> f() + z)``,
    ``(x + y |> f())``, and ``(x + y) |> f()`` stay one term so the
    pipeline diagnostic sees the sum.
    """
    spans = _pipeline_group_spans(rhs)
    parts: list[str] = []
    start = 0
    for index, ch in enumerate(rhs):
        if ch != "+":
            continue
        if any(span_start <= index <= span_end for span_start, span_end in spans):
            continue
        parts.append(rhs[start:index])
        start = index + 1
    parts.append(rhs[start:])
    return [part.strip() for part in parts]


def _top_level_plus(text: str) -> int | None:
    """Index of a ``+`` outside calls and legacy literals, if any."""
    masked = _unquoted_source(text)
    depth = 0
    for index, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "+" and depth == 0:
            return index
    return None


def _prepare_pipe_lines(spec_string: str) -> str:
    """Fold physical lines inside a pipeline group, preserving statement breaks."""
    source = spec_string.replace("\r\n", "\n")
    visible = _unquoted_source(source)
    if _find_pipe_token(visible) is None:
        return source
    out = list(source)
    opens = _depth0_group_opens(visible)
    depth = 0
    keyword_depths: list[int] = []
    i = 0
    while i < len(visible):
        ch = visible[i]
        in_keyword = bool(keyword_depths) and depth == keyword_depths[-1]
        # A ``*`` inside a keyword value is not a coefficient group.
        if i in opens:
            open_at = i
            suggestion = _closed_group_example(visible, open_at)
            group_depth = 1
            i += 1
            while i < len(visible) and group_depth:
                ch = visible[i]
                if ch == "(":
                    group_depth += 1
                elif ch == ")":
                    group_depth -= 1
                elif ch == ";":
                    raise ParseError(
                        "Semicolon inside a pipeline group. Close ')' before "
                        "starting another statement."
                    )
                elif ch == "\n":
                    next_line = visible[i + 1 :].split("\n", 1)[0].strip()
                    if _NEXT_EQUATION.match(next_line):
                        raise ParseError(
                            "Unclosed pipeline group before another equation. "
                            "Close ')' before starting the next statement. "
                            f"Write '{suggestion}'."
                        )
                    out[i] = " "
                i += 1
            if group_depth:
                raise ParseError(
                    "Unclosed pipeline group. Add the missing ')'. "
                    f"Write '{suggestion}'."
                )
            continue
        if ch == "\n" and depth == 0 and not in_keyword:
            next_line = visible[i + 1 :].split("\n", 1)[0].lstrip()
            term_start = max(visible.rfind(mark, 0, i) for mark in ("+", ";", "\n")) + 1
            term = visible[term_start:i]
            pipe_at = _find_pipe_token(term)
            tail = "" if pipe_at is None else term[pipe_at + 2 :]
            if (
                _pipe_token_at(next_line, 0)
                or _pipe_token_at(term.rstrip(), len(term.rstrip()) - 2)
                or (pipe_at is not None and tail.count("(") > tail.count(")"))
            ):
                raise ParseError(
                    "An ungrouped pipeline cannot span lines. "
                    "Write '(x |> f())' to continue a pipeline across lines."
                )
        depth, _ = _note_call_char(ch, depth, keyword_depths)
        i += 1
    return "".join(out)


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
        If two regressions share the same LHS variable, a coefficient
        label is reused across equations, or a defined parameter (``:=``)
        uses a coefficient label's name.
    """
    spec_string = _join_continuation_lines(_prepare_pipe_lines(spec_string))
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
    seen_labels: dict[str, str] = {}

    for stmt in statements:
        visible = _unquoted_source(stmt)
        # Quoted ':=' / '~~' are literal text, not statement operators.
        if _has_pipe(stmt) and (
            ":=" in visible or "~~" in visible or "~" not in visible
        ):
            raise ParseError(
                f"Pipeline syntax is only valid in a regression predictor: '{stmt}'. "
                "Use a plain variable or coefficient expression here."
            )
        if ":=" in stmt:
            defined_params.append(_parse_defined_param(stmt))
        elif "~~" in stmt:
            residual_covs.append(_parse_residual_cov(stmt))
        elif "~" in stmt:
            reg = _parse_regression(stmt)
            _reject_unidentified_splines(reg)
            _assign_basis_identifiers(reg)
            if reg.lhs in seen_lhs:
                raise DuplicateEquationError(
                    f"Duplicate equation for '{reg.lhs}'. "
                    "Each variable can appear as LHS in at most one regression."
                )
            labels_in_stmt: list[str] = []
            for term in reg.terms:
                if term.label is None:
                    continue
                if term.label in labels_in_stmt:
                    raise DuplicateEquationError(
                        f"Label '{term.label}' used more than once in '{stmt}'. "
                        "Each label must be unique. Use different labels "
                        "(e.g., 'a1', 'a2') for distinct coefficients."
                    )
                labels_in_stmt.append(term.label)
                if term.label in seen_labels:
                    raise DuplicateEquationError(
                        f"Label '{term.label}' used in both "
                        f"'{seen_labels[term.label]}' and '{stmt}'. "
                        "Each label must be unique. Use different labels "
                        "(e.g., 'a1', 'a2') for distinct coefficients."
                    )
                seen_labels[term.label] = stmt
            seen_lhs.add(reg.lhs)
            regressions.append(reg)
        else:
            raise ParseError(
                f"Unrecognized statement: '{stmt}'. "
                "Expected a regression (~), residual covariance (~~), "
                "or defined parameter (:=)."
            )

    spec = Spec(
        regressions=regressions,
        residual_covs=residual_covs,
        defined_params=defined_params,
    )
    _reject_defined_param_label_collisions(spec, seen_labels)
    return spec


def _format_transform_call(tc: TransformCall) -> str:
    """Render a transform call, including a nested input."""
    inner = (
        _format_transform_call(tc.input_expr)
        if isinstance(tc.input_expr, TransformCall)
        else tc.input_expr
    )
    if not tc.params:
        return f"{tc.name}({inner})"
    args = ", ".join(f"{key}={value}" for key, value in tc.params.items())
    return f"{tc.name}({inner}, {args})"


def _label_owner(reg: Regression, term: Term) -> str:
    """Name the term that owns a label when no source statement is available."""
    if term.transform is not None:
        predictor = _format_transform_call(term.transform)
    elif term.basis is not None:
        predictor = f"{term.basis.name}({term.basis.variable})"
    else:
        predictor = term.variable
    return f"{reg.lhs} ~ {term.label}*{predictor}"


def _reject_defined_param_label_collisions(
    spec: Spec,
    label_statements: dict[str, str] | None = None,
) -> None:
    """Reject a ``:=`` name that would replace a coefficient label.

    ``effects_summary()`` reports each labeled coefficient and each defined
    parameter. Merging those draws by name lets the defined parameter occupy
    the label, so the collision is rejected instead of overwritten.

    ``label_statements`` maps a label to the source statement. Parsed specs
    pass that map so the error cites the equation the user wrote, including
    transforms. A hand-built ``Spec`` falls back to the term itself.
    """
    statements = label_statements or {}
    label_sites: dict[str, str] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.label is None or term.label in label_sites:
                continue
            label_sites[term.label] = statements.get(term.label) or _label_owner(
                reg, term
            )
    clashes: list[str] = []
    for dp in spec.defined_params:
        if dp.name in label_sites and dp.name not in clashes:
            clashes.append(dp.name)
    if not clashes:
        return
    details = "; ".join(
        f"'{name}' is already the coefficient label in '{label_sites[name]}'"
        for name in clashes
    )
    raise DuplicateEquationError(
        "Defined parameter name collides with a coefficient label: "
        f"{details}. A defined parameter must not occupy a label's name. "
        "Use a distinct name for the defined parameter."
    )


def _reject_unidentified_splines(reg: Regression) -> None:
    """Reject a free column already spanned by a spline on the same input.

    A centered spline includes a straight line. A B-spline or P-spline also
    spans powers through its degree; a natural spline spans only the line.
    A second spline on the same column is the same hole. Fixed offsets stay.
    """
    by_input: dict[str, list[BasisCall]] = {}
    for term in reg.terms:
        basis = term.basis
        if isinstance(basis, BasisCall) and basis.name in SPLINE_NAMES:
            by_input.setdefault(basis.variable, []).append(basis)
    for variable, calls in by_input.items():
        if len(calls) > 1:
            names = ", ".join(call.name for call in calls)
            raise ParseError(
                f"'{reg.lhs}' has more than one spline on '{variable}' ({names}). "
                "Use one spline term per input; a second spline on the same "
                "column is not identified."
            )
        call = calls[0]
        plain_on_input = [
            term
            for term in reg.terms
            if term.basis is None
            and term.categorical is None
            and term.transform is None
            and term.lag_of is None
            and term.fixed_value is None
            and term.variable == variable
        ]
        if plain_on_input:
            raise ParseError(
                f"'{reg.lhs}' has a spline and a plain term on '{variable}'. "
                "The spline already spans that column, including a straight "
                "line, so the two coefficients are not identified. Drop the "
                "plain term."
            )
        spanned = spanned_degree(call.name, call.params)
        self_powers = [
            term
            for term in reg.terms
            if term.interaction_of is not None
            and term.fixed_value is None
            and all(part == variable for part in term.interaction_of)
            and len(term.interaction_of) <= spanned
        ]
        if self_powers:
            shown = ", ".join(f"'{term.variable}'" for term in self_powers)
            raise ParseError(
                f"'{reg.lhs}' has a {call.name} spline and {shown} on "
                f"'{variable}'. A degree-{spanned} {call.name} spline already "
                f"spans powers of '{variable}' through degree {spanned}, "
                "so those coefficients are not identified. Drop the "
                "self-product."
            )


def _assign_basis_identifiers(reg: Regression) -> None:
    """Disambiguate basis calls sharing one equation input."""
    grouped: dict[str, list[Term]] = {}
    for term in reg.terms:
        if term.basis is not None:
            grouped.setdefault(term.basis.variable, []).append(term)
    for terms in grouped.values():
        if len(terms) < 2 or all(isinstance(term.basis, HSGPCall) for term in terms):
            continue
        for index, term in enumerate(terms, start=1):
            basis = term.basis
            if basis is None:
                continue
            basis.identifier = str(index)


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

    if _has_pipe(lhs):
        raise ParseError(
            f"Pipeline syntax is not valid on the left-hand side '{lhs}'. "
            "Move the pipeline into a regression predictor."
        )
    if not re.match(r"^[A-Za-z_]\w*$", lhs):
        raise ParseError(
            f"Invalid variable name '{lhs}' on left-hand side of '{stmt}'."
        )
    raw_terms = _split_rhs_terms(rhs)

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

    regression = Regression(lhs=lhs, terms=terms, has_intercept=has_intercept)

    return regression


#: A float mantissa immediately followed by its exponent marker, anchored so
#: the digits start a fresh token: ``1e``, ``2.5E``, ``.5e`` all match, but the
#: trailing ``2e`` of an identifier such as ``x2e`` does not.
_FLOAT_EXPONENT_RE = re.compile(r"(?:^|[^0-9A-Za-z_.])(?:\d+\.?\d*|\.\d+)[eE]$")


def _is_float_exponent_sign(raw: str, i: int) -> bool:
    """Is ``raw[i]`` (a ``-``) the exponent sign of a float literal?

    Checking only that the preceding character is ``e``/``E`` is not enough:
    an identifier ending in ``e`` (``response-1``, ``income-1``, ``rate-1``)
    would then slip past the R-style-subtraction guard and become a single
    garbage variable name that only fails later, at data binding. The ``e``
    must terminate an actual numeric mantissa and be followed by digits.
    """
    if i == 0 or raw[i - 1] not in {"e", "E"}:
        return False
    if not _FLOAT_EXPONENT_RE.search(raw[:i]):
        return False
    return i + 1 < len(raw) and raw[i + 1].isdigit()


def _reject_top_level_minus(raw: str) -> None:
    """Reject R/patsy-style subtraction such as ``x - 1`` or ``-x``.

    pathmc's DSL is purely additive: terms are joined with ``+``, the
    intercept is controlled with a standalone ``0`` or ``1`` term, and a
    fixed coefficient (which may itself be negative, e.g. ``-1*x``) uses
    ``value*variable``. There is no formula algebra where ``-`` removes a
    term. Without this check, something like ``y ~ x - 1`` parses
    "successfully" into a term literally named ``"x - 1"``, which then
    fails much later with a confusing "column not found" error at data
    binding or compile time instead of a clear parse error (issue #316's
    crash-vector concern for intercept-removal syntax applies here too).

    The same guard also rejects hyphenated plain terms (``my-var``) and
    coefficient labels (``my-label*x``): ``-`` is never a name character
    in the DSL, so use underscores instead (``my_var``, ``my_label*x``).
    """
    star_pos = _find_top_level_star(raw)
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "-" and depth == 0:
            if i == 0 and star_pos is not None and i < star_pos:
                # Leading '-' of a fixed-value coefficient, e.g. "-1*x".
                continue
            if _is_float_exponent_sign(raw, i):
                # Exponent sign in a float literal, e.g. "1e-5*x".
                continue
            raise ParseError(
                f"Unsupported '-' in term '{raw}'. pathmc's formula DSL is "
                "additive only (terms are joined with '+'); it does not "
                "support R/patsy-style subtraction. To drop the intercept, "
                "use '0 + ...' (not '... - 1'); to fix a coefficient, use "
                "'value*variable' (e.g. '-1*x')."
            )


def _parse_term(raw: str) -> Term:
    """Parse a single term, optionally with a coefficient label and/or transform.

    Numeric labels (e.g. ``1*X``, ``0.5*X``) are treated as fixed
    coefficient values rather than free parameter names.
    """
    raw = raw.strip()

    _reject_top_level_minus(raw)
    label: str | None = None
    fixed_value: float | None = None
    coefficient_token: str | None = None

    if "*" in raw:
        star_pos = _find_top_level_star(raw)
        if star_pos is not None:
            label_str = raw[:star_pos].strip()
            raw = raw[star_pos + 1 :].strip()
            if not label_str:
                raise ParseError(f"Malformed labeled term: '{raw}'.")
            coefficient_token = label_str
            try:
                fixed_value = float(label_str)
                label = None
            except ValueError:
                label = label_str

    if label is not None and _has_pipe(label):
        raise ParseError(
            f"Invalid pipeline coefficient label '{label}'. "
            "Use an identifier such as 'a*(x |> f())'."
        )
    if _should_parse_as_pipeline(raw):
        if label is not None and _PIPE_IDENTIFIER.fullmatch(label) is None:
            raise ParseError(
                f"Invalid pipeline coefficient label '{label}'. "
                "Use an identifier such as 'a*(x |> f())'."
            )
        transform = _parse_pipeline(
            raw,
            prefixed=coefficient_token is not None,
            coefficient=coefficient_token,
        )
        return Term(
            variable=_extract_leaf_variable(transform),
            label=label,
            transform=transform,
            fixed_value=fixed_value,
        )

    if _find_top_level_colon(raw) is not None and re.search(r"(?:^|:)\s*C\s*\(", raw):
        raise ParseError(
            f"Categorical interaction '{raw}' is not supported. Use C(...) as "
            "a standalone term and model category-specific effects separately."
        )

    if "(" in raw:
        func_name = raw[: raw.index("(")].strip()
        if func_name == "C":
            if label is not None or fixed_value is not None:
                raise ParseError(
                    "C(...) cannot take a coefficient prefix because it expands "
                    "to one coefficient per non-reference level. Remove the "
                    "'k*' or 'label*' prefix."
                )
            categorical_call = _parse_categorical_expr(raw)
            return Term(
                variable=categorical_call.variable,
                categorical=categorical_call,
            )
        if func_name == "hsgp":
            if label is not None or fixed_value is not None:
                raise ParseError(
                    "hsgp(...) cannot take a coefficient prefix; the smooth "
                    "carries its own basis weights. Remove the 'k*' or "
                    "'label*' prefix."
                )
            call = _parse_hsgp_expr(raw)
            return Term(variable=call.variable, basis=call)
        if func_name == "fourier":
            if label is not None or fixed_value is not None:
                raise ParseError(
                    "fourier(...) cannot take a coefficient prefix; the basis "
                    "carries its own weights. Remove the 'k*' or 'label*' prefix."
                )
            basis_call = _parse_fourier_expr(raw)
            return Term(variable=basis_call.variable, basis=basis_call)
        if func_name in SPLINE_NAMES:
            if label is not None or fixed_value is not None:
                raise ParseError(
                    f"{func_name}(...) cannot take a coefficient prefix; the "
                    "spline carries its own weights. Remove the 'k*' or "
                    "'label*' prefix."
                )
            basis_call = _parse_spline_expr(raw)
            problem = spline_call_problem(
                basis_call.name, basis_call.variable, basis_call.params
            )
            if problem is not None:
                raise ParseError(problem)
            return Term(variable=basis_call.variable, basis=basis_call)
        if any(part.strip().lower() in SPLINE_NAMES for part in func_name.split(":")):
            raise ParseError(
                f"{raw} is not a standalone spline term. Use bs(), ns(), or ps() "
                "directly on one variable, e.g. 'y ~ bs(x, df=5)'."
            )
        if _is_registered_basis(func_name):
            if label is not None or fixed_value is not None:
                raise ParseError(
                    f"{func_name}(...) cannot take a coefficient prefix; basis "
                    "terms carry their own weights."
                )
            basis_call = _parse_registered_basis_expr(raw)
            return Term(variable=basis_call.variable, basis=basis_call)
        # Misspelled/misplaced hsgp would otherwise fall through to the
        # transform registry and fail late with "Unknown transform 'z:hsgp'".
        if any(part.strip().lower() == "hsgp" for part in func_name.split(":")):
            if ":" in func_name:
                raise ParseError(
                    f"hsgp() cannot appear inside a ':' interaction: '{raw}'. "
                    "Use hsgp() as a standalone term applied directly to one "
                    "variable, e.g. 'y ~ hsgp(x, m=20, c=1.5)'."
                )
            raise ParseError(
                f"Unknown function '{func_name}'. Did you mean hsgp(...)? "
                "The hsgp() term is lowercase."
            )
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


_CATEGORICAL_ALLOWED_KWARGS = frozenset({"reference", "prior"})


def _parse_categorical_expr(raw: str) -> CategoricalCall:
    """Parse ``C(variable, reference=..., prior=...)``."""
    if raw[-1] != ")":
        raise ParseError(
            f"Unclosed parenthesis in categorical term: '{raw}'. Add a closing ')'."
        )
    inner = raw[raw.index("(") + 1 : -1].strip()
    args = _split_top_level_args(inner)
    if not args or not args[0].strip() or "=" in args[0]:
        raise ParseError(
            "C(...) requires a column name as its first argument. "
            "Example: C(region, reference='north')."
        )
    variable = args[0].strip()
    if not re.match(r"^[A-Za-z_]\w*$", variable):
        raise ParseError(
            f"C(...) input must be a plain variable name, got '{variable}'."
        )

    kwargs: dict[str, Any] = {}
    for arg in args[1:]:
        if "=" not in arg:
            raise ParseError(
                f"Expected keyword parameter in C(...), got '{arg.strip()}'. "
                "Use reference=... or prior=...."
            )
        key, _, raw_value = arg.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in _CATEGORICAL_ALLOWED_KWARGS:
            raise ParseError(
                f"Unknown C(...) parameter '{key}'. Valid parameters are "
                "'reference' and 'prior'."
            )
        if key in kwargs:
            raise ParseError(f"Duplicate keyword '{key}' in C(...).")
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        kwargs[key] = value

    prior = kwargs.get("prior", "independent")
    if prior not in {"independent", "hierarchical"}:
        raise ParseError(
            f"C(...) prior must be 'independent' or 'hierarchical', got {prior!r}."
        )
    return CategoricalCall(
        variable=variable,
        reference=kwargs.get("reference"),
        prior=prior,
    )


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


def _find_top_level_colon(raw: str) -> int | None:
    """Find a ``:`` outside parentheses, or return ``None``."""
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            return i
    return None


_PIPE_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_PIPE_REFERENCE = re.compile(
    r"(?:[A-Za-z_]\w*|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\Z"
)


def _reject_pipeline_interaction(raw: str) -> None:
    """Reject a top-level ``:`` before grouping or stage parsing.

    ``x:z |> f()`` uses an interaction as a seed. ``(x |> f()):z`` is a
    transformed interaction. Neither correction is a non-interaction chain.
    """
    colon_at = _find_top_level_colon(raw)
    if colon_at is None:
        return
    fragment = raw[colon_at:].strip()
    left = raw[:colon_at]
    if _find_pipe_token(_unquoted_source(left)) is not None:
        raise ParseError(
            f"Transformed interaction '{raw}' is unsupported. "
            f"'{fragment}' cannot follow a pipeline. "
            "Write the factors as separate supported terms."
        )
    raise ParseError(
        f"Invalid pipeline '{raw}': pipeline inputs must be plain variables. "
        f"'{fragment}' is not a valid pipeline input. "
        "Write a plain source variable before the pipe."
    )


def _reject_parenthesized_pipe_seed(raw: str) -> None:
    """Reject a parenthesized sum or interaction used as a pipe seed.

    ``(x + y) |> f()`` and ``(x:z) |> f()`` are one term. Neither correction
    is a chain that includes the sum or the interaction. ``(x) |> f()`` is
    left to the grouped-chain rejection.
    """
    if not raw.startswith("("):
        return
    close = _matching_paren(_unquoted_source(raw), 0)
    if close is None:
        return
    rest = raw[close + 1 :].lstrip()
    if not _pipe_token_at(rest, 0):
        return
    interior = raw[1:close]
    if _top_level_plus(interior) is not None:
        seed = raw[: close + 1]
        raise ParseError(
            f"Invalid pipeline '{raw}': pipeline inputs must be plain variables. "
            f"'{seed}' is a sum, not a pipeline input. "
            "Write a plain source variable before the pipe, or write the sum as "
            "separate terms."
        )
    colon_at = _find_top_level_colon(interior)
    if colon_at is None:
        return
    fragment = f"{interior[colon_at:].strip()} {rest}".strip()
    raise ParseError(
        f"Invalid pipeline '{raw}': pipeline inputs must be plain variables. "
        f"'{fragment}' is not a valid pipeline input. "
        "Write a plain source variable before the pipe."
    )


def _pipeline_stage_names(raw: str) -> list[str]:
    """Stage names after each depth-0 ``|>``."""
    masked = _unquoted_source(raw)
    names: list[str] = []
    depth = 0
    index = 0
    while index < len(masked):
        ch = masked[index]
        if ch == "(":
            depth += 1
            index += 1
            continue
        if ch == ")":
            if depth:
                depth -= 1
            index += 1
            continue
        if depth == 0 and _pipe_token_at(masked, index):
            stage_at = index + 2
            while stage_at < len(masked) and masked[stage_at].isspace():
                stage_at += 1
            stage = _PIPE_IDENTIFIER.match(masked, stage_at)
            if stage is not None:
                names.append(stage.group())
            index = stage_at
            continue
        index += 1
    return names


def _reject_standalone_stage(name: str, original: str) -> None:
    """Raise the existing standalone error for a special pipeline stage."""
    if name.lower() in SPLINE_NAMES and name not in SPLINE_NAMES:
        raise ParseError(
            f"{name}(...) is not a standalone spline term. Use bs(), ns(), or ps() "
            "directly on one variable, e.g. 'y ~ bs(x, df=5)'."
        )
    if name.lower() == "hsgp" and name != "hsgp":
        raise ParseError(
            f"Unknown function '{name}'. Did you mean hsgp(...)? "
            "The hsgp() term is lowercase."
        )
    if name in {"lag", "C", "hsgp", "fourier", *SPLINE_NAMES} or _is_registered_basis(
        name
    ):
        raise ParseError(
            f"Invalid pipeline '{original}': '{name}' is a structural, "
            "categorical, or basis term; "
            f"use '{name}(x, ...)' as a standalone term."
        )


def _reject_structural_pipeline(raw: str, original: str) -> None:
    """Reject a special stage before a grouping suggestion can hide it."""
    for name in _pipeline_stage_names(raw):
        _reject_standalone_stage(name, original)


def _parse_pipeline(
    raw: str, *, prefixed: bool, coefficient: str | None = None
) -> TransformCall:
    """Lower a forward-pipe predictor into nested transform calls."""
    original = raw
    token = coefficient if coefficient else "a"
    _reject_pipeline_interaction(raw)
    _reject_parenthesized_pipe_seed(raw)
    close = _matching_paren(_unquoted_source(raw), 0) if raw.startswith("(") else None
    if raw.startswith("(") and close is None:
        suggestion = f"{token}*(x |> f())" if prefixed else "(x |> f())"
        raise ParseError(
            f"Unclosed pipeline group. Add the missing ')'. Write '{suggestion}'."
        )
    if close is not None:
        rest = raw[close + 1 :].lstrip()
        interior_has_pipe = _find_pipe_token(_unquoted_source(raw[1:close])) is not None
        if interior_has_pipe and rest and not _pipe_token_at(rest, 0):
            group = raw[: close + 1]
            suggestion = f"{token}*{group}" if prefixed else group
            raise ParseError(
                f"Invalid pipeline '{original}': unexpected text "
                f"'{rest.strip()}' after the pipeline group. "
                f"Remove it and write '{suggestion}'."
            )
        if not rest:
            raw = raw[1:close].strip()
            _reject_pipeline_interaction(raw)
            _reject_parenthesized_pipe_seed(raw)
    elif prefixed:
        _reject_structural_pipeline(raw, original)
        raise ParseError(
            f"Coefficient-prefixed pipeline '{original}' needs grouping. "
            f"Write '{token}*({raw})' so the coefficient applies after "
            "the transform."
        )
    if not raw:
        raise ParseError(f"Empty pipeline group '{original}'. Write '(x |> f())'.")
    plus_at = _top_level_plus(raw)
    if plus_at is not None:
        fragment = raw[plus_at:].strip()
        raise ParseError(
            f"Invalid pipeline '{original}': '{fragment}' is inside the pipeline group. "
            f"Move '{fragment}' outside the group, for example 'a*(x |> f()) + z'."
        )

    def fail(problem: str) -> ParseError:
        return ParseError(
            f"Invalid pipeline '{original}': {problem}. "
            "Write a single chain such as 'x |> f(p=u) |> g(q=v)' "
            "(or 'a*(x |> f(p=u))' with a coefficient)."
        )

    pos = 0

    def skip_space() -> None:
        nonlocal pos
        while pos < len(raw) and raw[pos].isspace():
            pos += 1

    skip_space()
    source = _PIPE_IDENTIFIER.match(raw, pos)
    if source is None:
        raise fail("the input must start with a plain source variable")
    expression: str | TransformCall = source.group()
    pos = source.end()
    stage_count = 0
    while True:
        skip_space()
        if pos == len(raw):
            if stage_count == 0:
                raise fail("a transform stage is required after the source variable")
            assert isinstance(expression, TransformCall)
            return expression
        if not raw.startswith("|>", pos):
            raise fail(f"unexpected text '{raw[pos:]}' after a pipeline part")
        pos += 2
        skip_space()
        stage = _PIPE_IDENTIFIER.match(raw, pos)
        if stage is None:
            raise fail("missing transform call after '|>'")
        name = stage.group()
        pos = stage.end()
        skip_space()
        if pos >= len(raw) or raw[pos] != "(":
            raise fail(f"stage '{name}' needs call parentheses, for example '{name}()'")
        pos += 1
        args_start = pos
        while pos < len(raw) and raw[pos] != ")":
            if raw[pos] == "(":
                raise fail(f"nested expression in stage '{name}' is unsupported")
            pos += 1
        if pos >= len(raw):
            raise fail(f"stage '{name}' has an unclosed call")
        args = raw[args_start:pos].strip()
        pos += 1
        _reject_standalone_stage(name, original)
        params: dict[str, str] = {}
        if args:
            parts = args.split(",")
            if parts[-1].strip() == "":
                parts.pop()
            for part in parts:
                if "=" not in part:
                    raise fail(
                        f"stage '{name}' expects keyword references; "
                        "the piped input is inserted automatically"
                    )
                key, _, value = part.partition("=")
                key, value = key.strip(), value.strip()
                if _PIPE_IDENTIFIER.fullmatch(key) is None or not value:
                    raise fail(
                        f"stage '{name}' has an incomplete keyword '{part.strip()}'"
                    )
                if _PIPE_REFERENCE.fullmatch(value) is None:
                    raise fail(
                        f"stage '{name}' parameter '{key}' needs a plain "
                        "parameter reference"
                    )
                if key in params:
                    raise fail(f"stage '{name}' repeats keyword '{key}'")
                params[key] = value
        expression = TransformCall(name=name, input_expr=expression, params=params)
        stage_count += 1


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
        input_name = input_raw[: input_raw.index("(")].strip()
        if input_name in {"hsgp", "fourier", *SPLINE_NAMES} or _is_registered_basis(
            input_name
        ):
            raise ParseError(
                f"Basis expression '{input_raw}' cannot be nested inside "
                f"transform '{raw}'. Apply {input_name}(...) directly to a "
                "variable as a standalone term."
            )
        input_expr: str | TransformCall = _parse_transform_expr(input_raw)
    else:
        _reject_top_level_minus(input_raw)
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


def _validate_hsgp_boundary(name: str, value: float) -> None:
    """Reject non-finite or non-positive HSGP boundary values.

    Both ``c`` (boundary as a proportion of the input half-range) and ``L``
    (absolute boundary) scale the Laplacian eigenvalues as ``(pi * j / (2 * L))``.
    A zero, negative, or non-finite boundary divides by zero or propagates
    ``nan`` into the basis, so the model compiles but every draw of the smooth
    is non-finite. Catch it at the parser boundary instead.
    """
    if not math.isfinite(value):
        raise ParseError(f"hsgp(...) {name} must be a finite number, got {value}.")
    if value <= 0:
        raise ParseError(
            f"hsgp(...) {name} must be > 0, got {value}. {name} scales the HSGP "
            "boundary; a non-positive value produces a non-finite basis."
        )


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
        _validate_hsgp_boundary("c", c)
    if has_l:
        try:
            boundary_l = float(kwargs["L"])
        except ValueError:
            raise ParseError(
                f"hsgp(...) L must be a number, got '{kwargs['L']}'."
            ) from None
        _validate_hsgp_boundary("L", boundary_l)

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


def _parse_fourier_expr(raw: str) -> BasisCall:
    """Parse ``fourier(x, n=..., period=...)`` into a generic basis call."""
    inner = raw[raw.index("(") + 1 : -1].strip()
    args = _split_top_level_args(inner)
    if not args or not args[0].strip() or "=" in args[0]:
        raise ParseError(
            "fourier(...) requires an input variable as its first argument. "
            "Example: fourier(week, n=3, period=52)."
        )
    variable = args[0].strip()
    if not re.match(r"^[A-Za-z_]\w*$", variable):
        raise ParseError(
            f"fourier(...) input must be a plain variable name, got '{variable}'."
        )

    kwargs: dict[str, str] = {}
    for arg in args[1:]:
        if "=" not in arg:
            raise ParseError(
                f"fourier(...) expects keyword parameters, got '{arg.strip()}'."
            )
        key, _, value = arg.partition("=")
        key, value = key.strip(), value.strip()
        if key in kwargs:
            raise ParseError(f"Duplicate keyword '{key}' in fourier(...).")
        kwargs[key] = value
    if set(kwargs) != {"n", "period"}:
        raise ParseError(
            "fourier(...) requires exactly n=<int> and period=<positive number>."
        )
    try:
        n = int(kwargs["n"])
    except ValueError:
        raise ParseError(
            f"fourier(...) n must be an integer, got '{kwargs['n']}'."
        ) from None
    try:
        period = float(kwargs["period"])
    except ValueError:
        raise ParseError(
            f"fourier(...) period must be a number, got '{kwargs['period']}'."
        ) from None
    if n < 1:
        raise ParseError(f"fourier(...) n must be >= 1, got {n}.")
    if not math.isfinite(period) or period <= 0:
        raise ParseError(
            f"fourier(...) period must be a finite number > 0, got {period}."
        )
    return BasisCall(
        name="fourier", variable=variable, params={"n": n, "period": period}
    )


_SPLINE_SCALAR_KEYS = frozenset({"df", "degree", "order"})


def _parse_spline_expr(raw: str) -> BasisCall:
    """Parse ``bs``, ``ns``, or ``ps`` into a generic basis call."""
    if _find_top_level_colon(raw) is not None:
        raise ParseError(
            f"{raw} is not a standalone spline term. Use bs(), ns(), or ps() "
            "directly on one variable, e.g. 'y ~ bs(x, df=5)'."
        )
    if raw[-1] != ")":
        raise ParseError(
            f"Unclosed parenthesis in spline term: '{raw}'. Add a closing ')'."
        )
    name = raw[: raw.index("(")].strip()
    inner = raw[raw.index("(") + 1 : -1].strip()
    args = _split_top_level_args(inner)
    if not args or not args[0].strip() or "=" in args[0]:
        raise ParseError(
            f"{name}(...) requires an input variable as its first argument. "
            f"Example: {name}(x, df=5)."
        )
    variable = args[0].strip()
    if not re.match(r"^[A-Za-z_]\w*$", variable):
        raise ParseError(
            f"{name}(...) input must be a plain variable name, got '{variable}'."
        )
    params: dict[str, int | float | tuple[float, ...]] = {}
    for arg in args[1:]:
        if "=" not in arg:
            raise ParseError(
                f"{name}(...) expects keyword parameters, got '{arg.strip()}'."
            )
        key, _, value = arg.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value or key in params:
            raise ParseError(
                f"Malformed or duplicate parameter '{arg}' in {name}(...)."
            )
        if key == "knots":
            params[key] = _parse_knot_tuple(name, value)
            continue
        if key not in _SPLINE_SCALAR_KEYS:
            raise ParseError(
                f"Unknown {name}(...) parameter '{key}'. Valid parameters are "
                "'df', 'degree', 'knots', and 'order' (ps only)."
            )
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ParseError(
                f"{name}(...) parameter '{key}' must be a number, got '{value}'."
            ) from None
        if isinstance(parsed, bool) or not isinstance(parsed, int | float):
            raise ParseError(
                f"{name}(...) parameter '{key}' must be a number, got '{value}'."
            )
        if isinstance(parsed, float) and parsed.is_integer():
            parsed = int(parsed)
        params[key] = parsed
    return BasisCall(name=name, variable=variable, params=params)


def _parse_knot_tuple(name: str, value: str) -> tuple[float, ...]:
    """Parse a parenthesized knot tuple, including a one-element tuple."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        raise ParseError(
            f"{name}(...) knots must be a parenthesized tuple of numbers, "
            f"got '{value}'. Example: knots=(0.2, 0.5, 0.8)."
        ) from None
    if isinstance(parsed, int | float) and not isinstance(parsed, bool):
        parsed = (parsed,)
    if not isinstance(parsed, tuple) or not parsed:
        raise ParseError(
            f"{name}(...) knots must be a non-empty parenthesized tuple of "
            f"numbers, got '{value}'."
        )
    knots: list[float] = []
    for item in parsed:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ParseError(
                f"{name}(...) knots must contain only numbers, got '{value}'."
            )
        knots.append(float(item))
    return tuple(knots)


def _is_registered_basis(name: str) -> bool:
    """Return whether *name* is currently registered as a custom basis."""
    from pathmc.basis import get_basis

    try:
        get_basis(name)
    except ValueError:
        return False
    return True


def _parse_registered_basis_expr(raw: str) -> BasisCall:
    """Parse a custom basis call with numeric literal keyword parameters."""
    name = raw[: raw.index("(")].strip()
    inner = raw[raw.index("(") + 1 : -1].strip()
    args = _split_top_level_args(inner)
    if not args or not args[0].strip() or "=" in args[0]:
        raise ParseError(
            f"{name}(...) requires an input variable as its first argument."
        )
    variable = args[0].strip()
    if not re.match(r"^[A-Za-z_]\w*$", variable):
        raise ParseError(
            f"{name}(...) input must be a plain variable name, got '{variable}'."
        )
    params: dict[str, int | float | tuple[float, ...]] = {}
    for arg in args[1:]:
        if "=" not in arg:
            raise ParseError(
                f"{name}(...) expects keyword parameters, got '{arg.strip()}'."
            )
        key, _, value = arg.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value or key in params:
            raise ParseError(
                f"Malformed or duplicate parameter '{arg}' in {name}(...)."
            )
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                raise ParseError(
                    f"{name}(...) parameter '{key}' must be a numeric literal, got "
                    f"'{value}'."
                ) from None
    return BasisCall(name=name, variable=variable, params=params)


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
