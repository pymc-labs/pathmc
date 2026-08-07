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
"""Tests for panel shape validation (issue #327, item 3).

The scan-based panel compiler assumes a dense, rectangular (unit x time)
grid: ``n_times = len(data) // n_units`` after sorting by ``(unit, time)``.
Malformed panels (unbalanced, ragged, duplicated, or misaligned rows) used
to silently mis-reshape instead of raising. These tests confirm that
``pathmc.panel.build_panel_info`` (invoked from ``pathmc.model()``) now
fails loudly with a descriptive ``ValueError``, and that well-formed
panels are unaffected.
"""

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v1 as nw

import pathmc
from pathmc.panel import build_panel_info


def _nw(df):
    return nw.from_native(df, eager_only=True)


def _balanced_panel(n_units=3, n_times=5, seed=0):
    rng = np.random.default_rng(seed)
    units = ["A", "B", "C"][:n_units]
    rows = []
    for u in units:
        for t in range(1, n_times + 1):
            rows.append({
                "unit": u,
                "time": t,
                "X": rng.normal(),
                "Y": rng.normal(),
            })
    return pd.DataFrame(rows)


class TestValidPanelShape:
    """Well-formed rectangular panels should pass validation unchanged."""

    def test_balanced_panel_builds_info(self):
        df = _balanced_panel()
        info = build_panel_info(_nw(df), {"unit": "unit", "time": "time"})
        assert info.unit_labels == ["A", "B", "C"]

    def test_balanced_panel_compiles_model(self):
        df = _balanced_panel()
        model = pathmc.model(
            "Y ~ 0 + X",
            data=df,
            panel={"unit": "unit", "time": "time"},
            pooling="partial",
        )
        assert model.pymc_model is not None

    def test_unsorted_balanced_panel_is_still_valid(self):
        # Row order shouldn't matter -- only the (unit, time) grid does.
        df = _balanced_panel().sample(frac=1.0, random_state=1).reset_index(drop=True)
        info = build_panel_info(_nw(df), {"unit": "unit", "time": "time"})
        assert info.unit_labels == ["A", "B", "C"]


class TestMalformedPanelShape:
    """Ragged / unbalanced / duplicated panels must raise a clear error."""

    def test_unbalanced_unit_missing_rows_raises(self):
        df = _balanced_panel(n_units=3, n_times=5)
        # Drop one row for unit "B" -> unbalanced.
        df = df[~((df["unit"] == "B") & (df["time"] == 3))].reset_index(drop=True)
        with pytest.raises(ValueError, match="unbalanced"):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_ragged_panel_via_model_raises(self):
        """A ragged panel must be rejected on the reshape (scan) path.

        The lag term forces the scan compiler, which is the path that
        reshapes rows to a dense (n_times, n_units) grid.
        """
        df = _balanced_panel(n_units=3, n_times=5)
        df = df[~((df["unit"] == "B") & (df["time"] == 3))].reset_index(drop=True)
        with pytest.raises(ValueError):
            pathmc.model(
                "Y ~ X + lag(Y)",
                data=df,
                panel={"unit": "unit", "time": "time"},
                pooling="partial",
            )

    def test_ragged_panel_accepted_by_row_wise_compiler(self):
        """Non-temporal panel models place no rectangularity requirement.

        ``Y ~ 0 + X`` has no lag() term and no adstock() transform, so it
        takes the row-wise compiler (pathmc/compile.py), which indexes rows
        by unit and never reshapes. Unequal observations per unit are fine
        there, and the scan-path validation must not reject them.
        """
        df = _balanced_panel(n_units=3, n_times=5)
        df = df[~((df["unit"] == "B") & (df["time"] == 3))].reset_index(drop=True)
        model = pathmc.model(
            "Y ~ 0 + X",
            data=df,
            panel={"unit": "unit", "time": "time"},
            pooling="partial",
        )
        assert model._panel_info is not None
        assert model._panel_info.unit_labels == ["A", "B", "C"]

    def test_duplicate_unit_time_rows_accepted_by_row_wise_compiler(self):
        """Repeated (unit, time) rows are only a problem for the reshape."""
        df = _balanced_panel(n_units=3, n_times=5)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        model = pathmc.model(
            "Y ~ 0 + X",
            data=df,
            panel={"unit": "unit", "time": "time"},
            pooling="partial",
        )
        assert model._panel_info is not None

    def test_duplicate_unit_time_row_raises(self):
        df = _balanced_panel(n_units=3, n_times=5)
        dup_row = df.iloc[[0]]
        df = pd.concat([df, dup_row], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_extra_unit_row_still_divides_evenly_but_ragged(self):
        # Give unit "A" one extra timepoint and unit "C" one fewer, so the
        # total row count still divides evenly by n_units (masking the
        # naive `len(data) // n_units` check) but the panel is ragged.
        df = _balanced_panel(n_units=3, n_times=5)
        extra = pd.DataFrame([{"unit": "A", "time": 6, "X": 0.1, "Y": 0.1}])
        df = pd.concat([df, extra], ignore_index=True)
        df = df[~((df["unit"] == "C") & (df["time"] == 5))].reset_index(drop=True)
        # Total rows unchanged (still divides evenly by 3 units), but the
        # per-unit row counts / timepoints no longer match.
        with pytest.raises(ValueError):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_misaligned_timepoints_across_units_raises(self):
        # Every unit has exactly 5 rows (balanced row count), but unit "C"
        # is observed at different timepoints than "A" and "B".
        df = _balanced_panel(n_units=3, n_times=5)
        df.loc[(df["unit"] == "C") & (df["time"] == 5), "time"] = 6
        with pytest.raises(ValueError, match="rectangular|timepoints"):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_single_row_panel_with_multiple_units_is_valid_edge_case(self):
        # n_units > 1, n_times = 1 -- degenerate but well-formed.
        df = pd.DataFrame([
            {"unit": "A", "time": 1, "X": 0.0, "Y": 0.0},
            {"unit": "B", "time": 1, "X": 0.5, "Y": 0.5},
        ])
        info = build_panel_info(_nw(df), {"unit": "unit", "time": "time"})
        assert info.unit_labels == ["A", "B"]

    def test_single_unit_single_row_panel_is_valid_edge_case(self):
        # 1 unit, 1 time -- degenerate but well-formed.
        df = pd.DataFrame([{"unit": "A", "time": 1, "X": 0.0, "Y": 0.0}])
        info = build_panel_info(_nw(df), {"unit": "unit", "time": "time"})
        assert info.unit_labels == ["A"]

    def test_empty_data_raises(self):
        df = pd.DataFrame({"unit": [], "time": [], "X": [], "Y": []})
        with pytest.raises(ValueError):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_null_unit_identifier_raises_actionable_error(self):
        df = _balanced_panel(n_units=3, n_times=5)
        df.loc[0, "unit"] = None
        with pytest.raises(ValueError, match="unit.*null"):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_nan_time_identifier_raises_actionable_error(self):
        df = _balanced_panel(n_units=3, n_times=5)
        df.loc[0, "time"] = np.nan
        with pytest.raises(ValueError, match="time.*null"):
            build_panel_info(_nw(df), {"unit": "unit", "time": "time"})

    def test_null_unit_identifier_rejected_on_row_wise_path_too(self):
        # Non-temporal panel models skip the rectangularity checks, but
        # null identifiers are nonsensical there as well and must still
        # raise (rather than the raw TypeError from sorting mixed types).
        df = _balanced_panel(n_units=3, n_times=5)
        df.loc[0, "unit"] = None
        with pytest.raises(ValueError, match="null"):
            pathmc.model(
                "Y ~ 0 + X",
                data=df,
                panel={"unit": "unit", "time": "time"},
                pooling="partial",
            )
