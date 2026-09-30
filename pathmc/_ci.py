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
"""Shared partial-correlation conditional-independence engine.

Single source of truth for the CI test behind both
:func:`pathmc.identify.test_implications` (edge-by-edge implication
testing) and :func:`pathmc.falsify.falsify_graph` (whole-graph
falsification). Degrees of freedom follow the *effective rank* of the
conditioning design, so constant, duplicated, or collinear conditioners
behave as if they were absent, and every degenerate input maps to a
named skip instead of a NaN or a runtime warning.

The engine itself stays silent about that collapse: ``effective_k`` on
:class:`CIResult` is the number of linearly independent conditioners
actually used. Callers that show a result to a user compare it with the
requested conditioning-set size and warn when the two differ.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    import narwhals.stable.v1 as nw

CISkipReason = Literal[
    "insufficient_observations",
    "zero_variance",
    "nonpositive_df",
    "zero_residual_variance",
    "nan_correlation",
]


@dataclass(frozen=True)
class CIResult:
    """Outcome of one partial-correlation conditional-independence test.

    Attributes
    ----------
    r : float | None
        Partial correlation between ``x`` and ``y`` given ``z``.
        ``None`` when the test was skipped.
    p : float | None
        Two-sided p-value. ``0.0`` for a perfect correlation
        (``r**2 >= 1``). ``None`` when the test was skipped.
    n : int
        Number of complete observations after dropping rows with any
        missing value. Always reported, including for skips.
    df : int | None
        Degrees of freedom of the t-test, ``n - effective_k - 2``
        (``n - 2`` for the marginal test). ``None`` when the test was
        skipped.
    skip_reason : CISkipReason | None
        Why the test could not be run, or ``None`` if it ran.
    effective_k : int | None
        Number of linearly independent conditioning columns,
        ``rank([1, Z]) - 1`` (``0`` for the marginal test). Smaller than
        the number of columns in ``Z`` when a conditioner is constant,
        duplicated, or a linear combination of the others. ``None`` when
        the test was skipped.
    """

    r: float | None
    p: float | None
    n: int
    df: int | None
    skip_reason: CISkipReason | None
    effective_k: int | None


def _skip(reason: CISkipReason, n: int) -> CIResult:
    return CIResult(
        r=None,
        p=None,
        n=n,
        df=None,
        skip_reason=reason,
        effective_k=None,
    )


def _caller_stacklevel() -> int:
    """Stack level that attributes a warning to the first non-pathmc frame."""
    import sys

    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None:
        frame = frame.f_back
        level += 1
        module = frame.f_globals.get("__name__", "")
        if module != "pathmc" and not module.startswith("pathmc."):
            return level
    return level


def warn_conditioning_rank_collapse(statements: list[str]) -> None:
    """Warn once that executed CI tests conditioned on a smaller set.

    Parameters
    ----------
    statements : list[str]
        One short label per collapsed test, including the requested size
        and ``effective_k``. No warning is emitted when empty.
    """
    if not statements:
        return
    n_collapsed = len(statements)
    noun = "test" if n_collapsed == 1 else "tests"
    subject = "This test" if n_collapsed == 1 else "These tests"
    warnings.warn(
        f"Conditioning-set rank collapse in {n_collapsed} "
        f"conditional-independence {noun}: {'; '.join(statements)}. "
        f"{subject} used fewer linearly independent conditioners than "
        f"requested, so the p-value is not evidence about the full "
        f"independence. Drop the constant, duplicated, or collinear "
        f"conditioner — compare effective_k with the conditioning set — "
        f"before reading the result.",
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )


def partial_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> CIResult:
    """Test ``x`` ⊥⊥ ``y`` | ``z`` via partial correlation.

    Regresses ``x`` and ``y`` on ``[1, z]`` and t-tests the correlation
    between the residuals. Rows containing any NaN are dropped first.

    Parameters
    ----------
    x, y : np.ndarray
        1-D arrays of equal length.
    z : np.ndarray | None
        Conditioning matrix of shape ``(len(x), k)``; ``None`` or zero
        columns for the marginal test.
    """
    if z is None:
        z = np.empty((x.shape[0], 0))
    arr = np.column_stack([x, y, z]).astype(float)
    arr = arr[~np.isnan(arr).any(axis=1)]
    n = arr.shape[0]
    if n < 3:
        return _skip("insufficient_observations", n)

    x_vals = arr[:, 0]
    y_vals = arr[:, 1]

    if z.shape[1] == 0:
        if np.std(x_vals) == 0.0 or np.std(y_vals) == 0.0:
            return _skip("zero_variance", n)
        r, p = stats.pearsonr(x_vals, y_vals)
        return CIResult(
            r=float(r),
            p=float(p),
            n=n,
            df=n - 2,
            skip_reason=None,
            effective_k=0,
        )

    z_with_intercept = np.column_stack([np.ones(n), arr[:, 2:]])
    # Effective rank handles constant or collinear conditioning columns:
    # a constant conditioner collapses into the intercept, so the test
    # reduces to the marginal one and the degrees of freedom must reflect
    # the true number of independent predictors, not the column count.
    rank = int(np.linalg.matrix_rank(z_with_intercept))
    effective_k = rank - 1
    df = n - rank - 1
    if df <= 0:
        return _skip("nonpositive_df", n)

    beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x_vals, rcond=None)
    resid_x = x_vals - z_with_intercept @ beta_x
    beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y_vals, rcond=None)
    resid_y = y_vals - z_with_intercept @ beta_y

    if np.std(resid_x) == 0.0 or np.std(resid_y) == 0.0:
        return _skip("zero_residual_variance", n)

    r = float(np.corrcoef(resid_x, resid_y)[0, 1])
    if np.isnan(r):
        return _skip("nan_correlation", n)
    if r * r >= 1.0:
        return CIResult(
            r=r,
            p=0.0,
            n=n,
            df=df,
            skip_reason=None,
            effective_k=effective_k,
        )

    t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df))
    return CIResult(
        r=r,
        p=p,
        n=n,
        df=df,
        skip_reason=None,
        effective_k=effective_k,
    )


class _PartialCorrelationTester:
    """Memoized partial-correlation conditional independence tester.

    Holds a numeric matrix extracted once from the data and caches each
    ``X ⊥⊥ Y | Z`` p-value. Independence is symmetric in ``X`` and ``Y``,
    so the cache key normalizes their order, mirroring dowhy's
    ``_PValuesMemory`` and avoiding recomputation across the many
    permuted graphs that re-test the same triples.
    """

    def __init__(self, data: nw.DataFrame, variables: list[str]) -> None:
        columns = set(data.columns)
        numeric: list[str] = []
        arrays: list[np.ndarray] = []
        for var in variables:
            if var not in columns:
                continue
            try:
                col = data.select(var).to_numpy().astype(float).ravel()
            except (ValueError, TypeError):
                # Non-numeric (string/categorical) columns cannot enter a
                # partial-correlation test; treat them as unavailable so the
                # affected triples are skipped, exactly like a missing column.
                continue
            numeric.append(var)
            arrays.append(col)
        if arrays:
            matrix = np.column_stack(arrays)
        else:
            matrix = np.empty((0, 0), dtype=float)
        self._matrix = matrix
        self._col_idx = {name: i for i, name in enumerate(numeric)}
        self._cache: dict[tuple[frozenset[str], frozenset[str]], CIResult | None] = {}

    def ci_result(self, x: str, y: str, z_vars: tuple[str, ...]) -> CIResult | None:
        """Return the full CI result, or ``None`` if a variable is missing.

        A missing numeric column has no :class:`CIResult`. A degenerate
        test still returns one, with ``skip_reason`` set, so callers can
        tell a skip from a rank collapse (``effective_k`` below the
        requested conditioner count).
        """
        key = (frozenset((x, y)), frozenset(z_vars))
        if key not in self._cache:
            self._cache[key] = self._compute(x, y, z_vars)
        return self._cache[key]

    def p_value(self, x: str, y: str, z_vars: tuple[str, ...]) -> float | None:
        """Return the CI test p-value, or ``None`` if it cannot be run.

        ``None`` signals a skipped test: a required variable has no data
        column, there are too few complete observations, or the test is
        otherwise degenerate (see :class:`CIResult`).
        """
        result = self.ci_result(x, y, z_vars)
        if result is None or result.skip_reason is not None:
            return None
        return result.p

    def _compute(self, x: str, y: str, z_vars: tuple[str, ...]) -> CIResult | None:
        needed = [x, y, *z_vars]
        if any(v not in self._col_idx for v in needed):
            return None
        cols = [self._col_idx[v] for v in needed]
        arr = self._matrix[:, cols]
        return partial_correlation_ci(arr[:, 0], arr[:, 1], arr[:, 2:])
