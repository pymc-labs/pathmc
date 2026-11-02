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
"""Call policy for ``bs``, ``ns``, and ``ps``.

The parser and the spline basis both need these rank rules. This module
imports neither ``pathmc.basis`` nor ``pathmc.parse``: those two import
each other, so the rules cannot live in either without a cycle or a copy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

SPLINE_NAMES = frozenset({"bs", "ns", "ps"})
DEFAULT_DEGREE = 3
DEFAULT_ORDER = 2

ParamValue = int | float | tuple[float, ...]
Params = Mapping[str, ParamValue]

__all__ = [
    "DEFAULT_DEGREE",
    "DEFAULT_ORDER",
    "SPLINE_NAMES",
    "implied_df",
    "minimum_df",
    "spanned_degree",
    "spline_call_problem",
]


def minimum_df(name: str, degree: int, order: int) -> int:
    """Return the smallest ``df`` that keeps the basis and the penalty full rank."""
    if name == "ns":
        return 2
    if name == "ps":
        return max(degree, order + 1)
    return degree


def implied_df(name: str, n_knots: int, degree: int) -> int:
    """Return the coefficient count implied by an explicit knot tuple."""
    if name == "ns":
        return n_knots + 1
    return n_knots + degree


def spanned_degree(name: str, params: Params) -> int:
    """Return the highest power of the input already in the basis.

    A B-spline or P-spline of degree ``d`` spans polynomials through ``d``.
    A natural spline spans only the line. An omitted degree is cubic.
    """
    if name == "ns":
        return 1
    degree = params.get("degree", DEFAULT_DEGREE)
    if isinstance(degree, bool) or not isinstance(degree, int):
        return DEFAULT_DEGREE
    return degree


def spline_call_problem(name: str, variable: str, params: Params) -> str | None:
    """Return a fix-oriented error for an unusable spline call, or ``None``."""
    example = f"{name}({variable}, df=5)"
    if "df" in params and "knots" in params:
        return (
            f"{name}() accepts either df= or knots=, not both. "
            f"Use df= to place quantile knots, or knots= to set them yourself. "
            f"Example: {example}."
        )
    if "df" not in params and "knots" not in params:
        return f"{name}() needs df=<integer> or knots=(...). Example: {example}."
    degree_raw = params.get("degree", DEFAULT_DEGREE)
    if isinstance(degree_raw, bool) or not isinstance(degree_raw, int):
        return (
            f"{name}() degree must be an integer >= 1, got {degree_raw!r}. "
            "Omit degree= for a cubic basis."
        )
    if degree_raw < 1:
        return (
            f"{name}() degree must be an integer >= 1, got {degree_raw}. "
            "Omit degree= for a cubic basis."
        )
    if name == "ns" and degree_raw != DEFAULT_DEGREE:
        return (
            "ns() is a cubic natural spline, so degree must be omitted or "
            f"{DEFAULT_DEGREE}, got {degree_raw}. Use bs() for another "
            "polynomial degree."
        )
    if "order" in params and name != "ps":
        return (
            "order= is the P-spline penalty and is only valid on ps(). "
            f"Remove it from {name}(), or write "
            f"ps({variable}, df=5, order=1)."
        )
    order = DEFAULT_ORDER
    if name == "ps" and "order" in params:
        order_raw = params["order"]
        if isinstance(order_raw, bool) or not isinstance(order_raw, int):
            return (
                "ps() order must be 1 or 2, got "
                f"{order_raw!r}. Use order=1, or omit order= for the "
                "default second-order penalty."
            )
        if order_raw not in (1, 2):
            return (
                f"ps() order must be 1 or 2, got {order_raw}. Use order=1, "
                "or omit order= for the default second-order penalty."
            )
        order = order_raw
    minimum = minimum_df(name, degree_raw, order)
    if "df" in params:
        df = params["df"]
        if isinstance(df, bool) or not isinstance(df, int):
            return (
                f"{name}() df must be an integer >= {minimum}, got {df!r}. "
                f"Example: {example}."
            )
        if df < minimum:
            return (
                f"{name}() df must be an integer >= {minimum}, got {df}. "
                "Raise df, or lower degree=."
            )
        return None
    knots = params["knots"]
    if not isinstance(knots, tuple) or not knots:
        return (
            f"{name}() knots must be a non-empty tuple of finite numbers. "
            "Example: knots=(0.2, 0.5, 0.8)."
        )
    if any(
        isinstance(knot, bool) or not isinstance(knot, int | float) for knot in knots
    ):
        return (
            f"{name}() knots must contain only numbers. Example: knots=(0.2, 0.5, 0.8)."
        )
    if any(not math.isfinite(float(knot)) for knot in knots):
        return (
            f"{name}() knots must be finite, got {knots}. "
            "Pass finite numbers strictly inside the observed range."
        )
    if any(
        float(right) <= float(left)
        for left, right in zip(knots, knots[1:], strict=False)
    ):
        return (
            f"{name}() knots must be strictly increasing, got {knots}. "
            "Sort them and drop duplicates."
        )
    implied = implied_df(name, len(knots), degree_raw)
    if implied < minimum:
        return (
            f"{name}() with these knots has {implied} coefficients, but needs "
            f"at least {minimum}. Add knots, or lower degree= or order=."
        )
    return None
