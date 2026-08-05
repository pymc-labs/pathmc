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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

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
        Degrees of freedom of the t-test, ``n - rank([1, Z]) - 1``
        (``n - 2`` for the marginal test). ``None`` when the test was
        skipped.
    skip_reason : CISkipReason | None
        Why the test could not be run, or ``None`` if it ran.
    """

    r: float | None
    p: float | None
    n: int
    df: int | None
    skip_reason: CISkipReason | None


def _skip(reason: CISkipReason, n: int) -> CIResult:
    return CIResult(r=None, p=None, n=n, df=None, skip_reason=reason)


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
        return CIResult(r=float(r), p=float(p), n=n, df=n - 2, skip_reason=None)

    z_with_intercept = np.column_stack([np.ones(n), arr[:, 2:]])
    # Effective rank handles constant or collinear conditioning columns:
    # a constant conditioner collapses into the intercept, so the test
    # reduces to the marginal one and the degrees of freedom must reflect
    # the true number of independent predictors, not the column count.
    rank = int(np.linalg.matrix_rank(z_with_intercept))
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
        return CIResult(r=r, p=0.0, n=n, df=df, skip_reason=None)

    t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df))
    return CIResult(r=r, p=p, n=n, df=df, skip_reason=None)
