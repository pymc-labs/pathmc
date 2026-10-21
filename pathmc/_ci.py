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
named skip instead of a NaN or a runtime warning. The engine stays
silent: ``requested_k`` and ``effective_k`` record the collapse, and
callers decide whether a collapsed row counts. A named conditioner that
encodes to zero columns is not a smaller rank of the columns that entered;
``dropped_conditioner_names`` reports it, and callers mark that row.

Every variable enters as a *block* of one or more numeric columns. A
continuous variable is one column; a categorical variable is its ``k - 1``
indicator columns, so it can sit on either side of the independence or in
the conditioning set. When both tested blocks are single columns the test
is the classic partial-correlation t-test. Otherwise it is the linear
multivariate generalisation: Wilks' lambda on the residualised blocks
(equivalently their canonical correlations) with Rao's F approximation,
which reduces to the partial F-test when one block is a single column and
to the t-test when both are.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
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
        Partial correlation between ``x`` and ``y`` given ``z`` when both
        are single columns. For multi-column blocks (a categorical on
        either side) it is the largest canonical correlation between the
        residualised blocks, which is non-negative. ``None`` when the test
        was skipped.
    p : float | None
        Two-sided p-value. ``0.0`` for a perfect association
        (``r**2 >= 1`` or Wilks' lambda of zero). ``None`` when the test
        was skipped.
    n : int
        Number of complete observations after dropping rows with any
        missing value. Always reported, including for skips.
    df : int | float | None
        Degrees of freedom of the t-test, ``n - rank([1, Z]) - 1``
        (``n - 2`` for the marginal test). For multi-column blocks, the
        denominator degrees of freedom of Rao's F approximation. ``None``
        when the test was skipped.
    skip_reason : CISkipReason | None
        Why the test could not be run, or ``None`` if it ran.
    requested_k : int
        Conditioning columns that entered the design (``Z.shape[1]``,
        ``0`` for a marginal test). A single-column conditioner contributes
        one; a categorical conditioner contributes its indicator columns.
        Always reported, including for skips.
    effective_k : int | None
        ``rank([1, Z]) - 1`` when the test ran. ``None`` when it was
        skipped. ``rank_collapsed`` is true only when this is smaller than
        ``requested_k``.
    """

    r: float | None
    p: float | None
    n: int
    df: int | float | None
    skip_reason: CISkipReason | None
    requested_k: int
    effective_k: int | None

    @property
    def rank_collapsed(self) -> bool:
        """Whether an executed test used a smaller conditioning design.

        False on a skip. The engine does not warn; callers compare
        ``effective_k`` with ``requested_k``.
        """
        if self.skip_reason is not None or self.effective_k is None:
            return False
        return self.effective_k < self.requested_k


def _skip(reason: CISkipReason, n: int, requested_k: int) -> CIResult:
    return CIResult(
        r=None,
        p=None,
        n=n,
        df=None,
        skip_reason=reason,
        requested_k=requested_k,
        effective_k=None,
    )


def _ran(
    *,
    r: float,
    p: float,
    n: int,
    df: int | float,
    requested_k: int,
    effective_k: int,
) -> CIResult:
    return CIResult(
        r=r,
        p=p,
        n=n,
        df=df,
        skip_reason=None,
        requested_k=requested_k,
        effective_k=effective_k,
    )


def _as_block(values: np.ndarray) -> np.ndarray:
    """Return *values* as a 2-D float array with one column per feature."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(
            f"CI test inputs must be 1-D or 2-D arrays, got {arr.ndim} dimensions."
        )
    return arr


def _effective_rank(matrix: np.ndarray) -> int:
    """Rank of *matrix* with numpy's default ``matrix_rank`` tolerance."""
    if matrix.size == 0:
        return 0
    return int(np.linalg.matrix_rank(matrix))


def partial_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None = None,
) -> CIResult:
    """Test ``x`` ⊥⊥ ``y`` | ``z`` via partial correlation.

    Regresses ``x`` and ``y`` on ``[1, z]`` and tests the association
    between the residuals. Rows containing any NaN are dropped first.

    Parameters
    ----------
    x, y : np.ndarray
        Arrays with the same number of rows. A 1-D array (or a 2-D array
        with one column) is a continuous variable; a 2-D array with
        several columns is a block, for example the indicator columns of a
        categorical variable.
    z : np.ndarray | None
        Conditioning matrix of shape ``(n_rows, k)``; ``None`` or zero
        columns for the marginal test.
    """
    x_block = _as_block(x)
    y_block = _as_block(y)
    n_rows = x_block.shape[0]
    z_block = np.empty((n_rows, 0)) if z is None else _as_block(z)
    p, q = x_block.shape[1], y_block.shape[1]

    arr = np.column_stack([x_block, y_block, z_block])
    arr = arr[~np.isnan(arr).any(axis=1)]
    n = arr.shape[0]
    if n < 3:
        return _skip("insufficient_observations", n, z_block.shape[1])

    x_vals = arr[:, :p]
    y_vals = arr[:, p : p + q]
    z_vals = arr[:, p + q :]
    if p == 1 and q == 1:
        return _scalar_test(x_vals[:, 0], y_vals[:, 0], z_vals, n)
    return _block_test(x_vals, y_vals, z_vals, n)


def _scalar_test(
    x_vals: np.ndarray, y_vals: np.ndarray, z_vals: np.ndarray, n: int
) -> CIResult:
    """Classic partial-correlation t-test for two single-column variables."""
    requested_k = z_vals.shape[1]
    if requested_k == 0:
        if np.std(x_vals) == 0.0 or np.std(y_vals) == 0.0:
            return _skip("zero_variance", n, requested_k)
        r, p = stats.pearsonr(x_vals, y_vals)
        return _ran(
            r=float(r),
            p=float(p),
            n=n,
            df=n - 2,
            requested_k=0,
            effective_k=0,
        )

    z_with_intercept = np.column_stack([np.ones(n), z_vals])
    # Effective rank handles constant or collinear conditioning columns:
    # a constant conditioner collapses into the intercept, so the test
    # reduces to the marginal one and the degrees of freedom must reflect
    # the true number of independent predictors, not the column count.
    rank = _effective_rank(z_with_intercept)
    effective_k = rank - 1
    df = n - rank - 1
    if df <= 0:
        return _skip("nonpositive_df", n, requested_k)

    beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x_vals, rcond=None)
    resid_x = x_vals - z_with_intercept @ beta_x
    beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y_vals, rcond=None)
    resid_y = y_vals - z_with_intercept @ beta_y

    if np.std(resid_x) == 0.0 or np.std(resid_y) == 0.0:
        return _skip("zero_residual_variance", n, requested_k)

    r = float(np.corrcoef(resid_x, resid_y)[0, 1])
    if np.isnan(r):
        return _skip("nan_correlation", n, requested_k)
    if r * r >= 1.0:
        return _ran(
            r=r,
            p=0.0,
            n=n,
            df=df,
            requested_k=requested_k,
            effective_k=effective_k,
        )

    t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df))
    return _ran(
        r=r,
        p=p,
        n=n,
        df=df,
        requested_k=requested_k,
        effective_k=effective_k,
    )


def _orthonormal_residual_basis(
    block: np.ndarray, z_with_intercept: np.ndarray
) -> np.ndarray:
    """Orthonormal basis of *block* residualised on ``[1, Z]``.

    Columns that are constant, collinear, or fully explained by the
    conditioners drop out, so the returned basis has one column per
    effective residual dimension (possibly zero).
    """
    beta, _, _, _ = np.linalg.lstsq(z_with_intercept, block, rcond=None)
    resid = block - z_with_intercept @ beta
    u, s, _ = np.linalg.svd(resid, full_matrices=False)
    if s.size == 0:
        return u[:, :0]
    # The cutoff is relative to the centred block, not to the residual: when
    # [1, Z] spans the block entirely the residual is pure rounding noise,
    # and a residual-relative cutoff would keep every noise direction.
    scale = float(np.linalg.norm(block - block.mean(axis=0), 2))
    if scale == 0.0:
        return u[:, :0]
    tol = scale * max(resid.shape) * np.finfo(resid.dtype).eps
    return u[:, s > tol]


def _block_test(
    x_vals: np.ndarray, y_vals: np.ndarray, z_vals: np.ndarray, n: int
) -> CIResult:
    """Wilks' lambda test between two residualised blocks (Rao's F)."""
    requested_k = z_vals.shape[1]
    if (
        _effective_rank(x_vals - x_vals.mean(axis=0)) == 0
        or _effective_rank(y_vals - y_vals.mean(axis=0)) == 0
    ):
        return _skip("zero_variance", n, requested_k)

    z_with_intercept = np.column_stack([np.ones(n), z_vals])
    rank_z = _effective_rank(z_with_intercept)
    effective_k = rank_z - 1
    ux = _orthonormal_residual_basis(x_vals, z_with_intercept)
    uy = _orthonormal_residual_basis(y_vals, z_with_intercept)
    p_eff, q_eff = ux.shape[1], uy.shape[1]
    if p_eff == 0 or q_eff == 0:
        return _skip("zero_residual_variance", n, requested_k)

    # Rao's F approximation to Wilks' lambda (Rencher, Methods of
    # Multivariate Analysis, §6.1.3), with hypothesis df p_eff, q_eff
    # responses, and error df n - rank_z - p_eff. It is exact when either
    # block has at most two effective columns, which covers the t-test
    # (both scalar) and the partial F-test (one categorical).
    df1 = p_eff * q_eff
    shape = p_eff**2 + q_eff**2 - 5
    t = np.sqrt((df1**2 - 4) / shape) if shape > 0 else 1.0
    w = n - rank_z - (p_eff + q_eff + 1) / 2.0
    df2 = w * t - (df1 - 2) / 2.0
    if df2 <= 0:
        return _skip("nonpositive_df", n, requested_k)

    rho = np.clip(np.linalg.svd(ux.T @ uy, compute_uv=False), 0.0, 1.0)
    r = float(rho.max())
    wilks = float(np.prod(1.0 - rho**2))
    if wilks <= 0.0:
        return _ran(
            r=r,
            p=0.0,
            n=n,
            df=float(df2),
            requested_k=requested_k,
            effective_k=effective_k,
        )

    wilks_t = wilks ** (1.0 / t)
    f_stat = (1.0 - wilks_t) / wilks_t * df2 / df1
    p = float(stats.f.sf(f_stat, df1, df2))
    return _ran(
        r=r,
        p=p,
        n=n,
        df=float(df2),
        requested_k=requested_k,
        effective_k=effective_k,
    )


def _is_numeric_array(values: np.ndarray) -> bool:
    """Whether *values* can be used as a continuous column."""
    if values.dtype.kind in "biuf":
        return True
    try:
        np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return False
    return True


def indicator_block(values: np.ndarray) -> np.ndarray:
    """Encode labels as ``k - 1`` indicator columns for a CI test.

    Any ``k - 1`` indicators span the same space as the full set once an
    intercept is present, so the dropped level (the first in string order)
    does not affect the test. Rows with missing labels become NaN rows so
    the engine's complete-case filter removes them.
    """
    arr = np.asarray(values, dtype=object).reshape(-1)
    missing = pd.isna(arr)
    levels = sorted(set(arr[~missing].tolist()), key=str)
    if len(levels) < 2:
        block = np.empty((arr.shape[0], 0))
    else:
        block = np.column_stack([arr == level for level in levels[1:]]).astype(float)
    block[missing, :] = np.nan
    return block


def variable_blocks(
    data: nw.DataFrame,
    variables: Iterable[str],
    categorical_vars: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return one float matrix per *variable* that has a data column.

    Continuous columns become one column. Columns named in
    *categorical_vars*, or that model fitting would treat as categorical
    (string or pandas categorical labels, including numeric-looking strings
    such as ``"1"``), become indicator blocks via :func:`indicator_block`.
    Other non-numeric columns keep the indicator fallback. Variables without
    a data column are omitted.
    """
    from pathmc.categorical import _is_categorical_series

    categorical = set(categorical_vars or ())
    columns = set(data.columns)
    blocks: dict[str, np.ndarray] = {}
    for var in variables:
        if var not in columns:
            continue
        series = data[var].to_pandas()
        if not isinstance(series, pd.Series):
            series = pd.Series(np.asarray(series), name=var)
        if var in categorical or _is_categorical_series(series):
            blocks[var] = indicator_block(series.to_numpy())
            continue
        values = series.to_numpy()
        if not _is_numeric_array(values):
            blocks[var] = indicator_block(values)
        else:
            blocks[var] = np.asarray(values, dtype=float).reshape(-1, 1)
    return blocks


def stack_blocks(blocks: Mapping[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    """Horizontally stack the blocks of *names* in order."""
    selected = [blocks[name] for name in names]
    if not selected:
        first = next(iter(blocks.values()))
        return np.empty((first.shape[0], 0))
    return np.column_stack(selected)


def dropped_conditioner_names(
    blocks: Mapping[str, np.ndarray],
    names: Iterable[str],
) -> tuple[str, ...]:
    """Named conditioners whose block contributed no columns.

    A single-level categorical encodes to a zero-width indicator block.
    ``stack_blocks`` drops that block, so ``requested_k`` stays the columns
    that entered. Callers mark the written-test row collapsed. The engine
    does not warn.
    """
    dropped: list[str] = []
    for name in names:
        block = blocks.get(name)
        if block is not None and block.ndim == 2 and block.shape[1] == 0:
            dropped.append(name)
    return tuple(dropped)


def named_column_ci(
    data: nw.DataFrame,
    x: str,
    y: str,
    z_vars: Iterable[str],
    categorical_vars: Iterable[str] | None = None,
) -> tuple[CIResult, tuple[str, ...]]:
    """Run one named-column CI test and report dropped conditioners.

    ``requested_k`` is the columns that entered. A named conditioner that
    encoded to zero columns is returned separately so callers can mark the
    row collapsed without changing the engine's rank formula.
    """
    names = tuple(z_vars)
    blocks = variable_blocks(data, [x, y, *names], categorical_vars)
    result = partial_correlation_ci(blocks[x], blocks[y], stack_blocks(blocks, names))
    return result, dropped_conditioner_names(blocks, names)


class _PartialCorrelationTester:
    """Memoized partial-correlation conditional independence tester.

    Holds one numeric block per variable, extracted once from the data,
    and caches each ``X ⊥⊥ Y | Z`` result. Independence is symmetric in
    ``X`` and ``Y``, so the cache key normalizes their order, mirroring
    dowhy's ``_PValuesMemory`` and avoiding recomputation across the many
    permuted graphs that re-test the same triples.
    """

    def __init__(
        self,
        data: nw.DataFrame,
        variables: list[str],
        categorical_vars: Iterable[str] | None = None,
    ) -> None:
        self._blocks = variable_blocks(data, variables, categorical_vars)
        self._cache: dict[tuple[frozenset[str], frozenset[str]], CIResult | None] = {}

    def ci_result(self, x: str, y: str, z_vars: tuple[str, ...]) -> CIResult | None:
        """Return the cached CI result, or ``None`` if a variable is missing.

        A skipped test is a :class:`CIResult` with ``skip_reason`` set, not
        ``None``. ``None`` means a required variable has no data column.
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
        if result is None or result.skip_reason is not None or result.p is None:
            return None
        return result.p

    def dropped_conditioners(self, names: Iterable[str]) -> tuple[str, ...]:
        """Named conditioners that encoded to a zero-column block."""
        return dropped_conditioner_names(self._blocks, names)

    def _compute(self, x: str, y: str, z_vars: tuple[str, ...]) -> CIResult | None:
        needed = [x, y, *z_vars]
        if any(v not in self._blocks for v in needed):
            return None
        return partial_correlation_ci(
            self._blocks[x],
            self._blocks[y],
            stack_blocks(self._blocks, z_vars),
        )
