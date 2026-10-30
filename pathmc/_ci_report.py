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
"""Display and scoring helpers for conditional-independence results.

The CI engine in :mod:`pathmc._ci` stays silent. Callers that show a
result, or that score a DAG against a permutation baseline, use this
module so a rank-collapsed row is neither a pass nor a fail of the
independence that was written down.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import pandas as pd


def rank_collapsed_mask(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that did not test the written conditioning set.

    An empty frame is all False. A non-empty frame must already carry
    ``rank_collapsed``; both producers always write that column.
    """
    if frame.empty:
        return pd.Series(False, index=frame.index)
    return frame["rank_collapsed"].astype(bool)


def written_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that actually tested the independence that was written down."""
    return frame.loc[~rank_collapsed_mask(frame)]


def collapse_statement(
    left: str,
    right: str,
    conditioning: str,
    requested_k: int,
    effective_k: int,
    dropped: Iterable[str] = (),
) -> str:
    """One warning label for a collapsed written-test row."""
    given = f" | {{{conditioning}}}" if conditioning else ""
    names = ", ".join(dropped)
    note = f"; {names} contributed no columns" if names else ""
    return (
        f"{left} ⊥⊥ {right}{given} "
        f"(requested {int(requested_k)}, effective_k {int(effective_k)}{note})"
    )


def written_counts(frame: pd.DataFrame, flag: str) -> tuple[int, int]:
    """Return ``(n_written_tests, n_written_violations)`` for *flag*.

    Rank-collapsed rows are in neither count. An empty frame, or a frame
    whose *flag* column is absent, is ``(0, 0)``.
    """
    scored = written_rows(frame)
    if scored.empty or flag not in scored.columns:
        return 0, 0
    return len(scored), int(scored[flag].astype(bool).sum())


def violation_rows(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Written-test rows where *flag* is true.

    A collapsed row is not in this frame. An empty frame is returned
    unchanged so its columns stay addressable.
    """
    scored = written_rows(frame)
    if scored.empty:
        return scored.copy()
    return scored.loc[scored[flag].astype(bool)].copy()


def n_rank_collapsed(frame: pd.DataFrame) -> int:
    """Count executed rows whose conditioning set was rank-deficient."""
    return int(rank_collapsed_mask(frame).sum())


def row_is_rank_collapsed(row: pd.Series) -> bool:
    """Whether this displayed row did not test the written conditioning set."""
    return bool(row["rank_collapsed"])


def rank_collapse_note(row: pd.Series) -> str:
    """Mark a displayed independence whose conditioning set collapsed.

    A skip, which is not rank-collapsed, gets no marker. A full-rank row
    renders exactly as it did before.
    """
    if not row_is_rank_collapsed(row):
        return ""
    return f" [effective_k={int(row['effective_k'])}]"


def warn_conditioning_rank_collapse(statements: list[str]) -> None:
    """Warn once that executed CI tests did not test the written set.

    The engine stays silent. Callers pass labels built from rows whose
    ``rank_collapsed`` flag is set, including a named conditioner that
    contributed no columns. No warning is emitted when empty.
    """
    if not statements:
        return
    from pathmc.compile import _user_stacklevel

    n_collapsed = len(statements)
    noun = "test" if n_collapsed == 1 else "tests"
    subject = "This test" if n_collapsed == 1 else "These tests"
    warnings.warn(
        f"Conditioning-set rank collapse in {n_collapsed} "
        f"conditional-independence {noun}: {'; '.join(statements)}. "
        f"{subject} did not test the conditioning set that was written "
        f"down, so the p-value is not evidence about the full "
        f"independence. Drop the constant, duplicated, collinear, or "
        f"single-level categorical conditioner before reading the result.",
        UserWarning,
        stacklevel=_user_stacklevel(),
    )
