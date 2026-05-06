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
"""Gate tests for M27: Identification helpers."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.graph import build_graph
from pathmc.identify import adjustment_sets, collider_warnings, is_identifiable
from pathmc.parse import parse_spec


def _graph(spec_str):
    return build_graph(parse_spec(spec_str))


class TestForkDAG:
    """Z -> X, Z -> Y, X -> Y. Need to adjust for Z."""

    def test_adjustment_set_includes_confounder(self):
        g = _graph("X ~ Z\nY ~ X + Z")
        sets = adjustment_sets(g, "X", "Y")
        assert {"Z"} in sets

    def test_is_identifiable(self):
        g = _graph("X ~ Z\nY ~ X + Z")
        assert is_identifiable(g, "X", "Y")

    def test_empty_set_not_valid(self):
        g = _graph("X ~ Z\nY ~ X + Z")
        sets = adjustment_sets(g, "X", "Y")
        assert set() not in sets


class TestChainDAG:
    """X -> M -> Y. No confounding; empty set is valid."""

    def test_empty_set_is_valid(self):
        g = _graph("M ~ X\nY ~ M")
        sets = adjustment_sets(g, "X", "Y")
        assert set() in sets

    def test_is_identifiable(self):
        g = _graph("M ~ X\nY ~ M")
        assert is_identifiable(g, "X", "Y")


class TestColliderDAG:
    """X -> C <- Y. No confounding; should not adjust for C."""

    def test_empty_set_is_valid(self):
        g = _graph("C ~ X + Y")
        sets = adjustment_sets(g, "X", "Y")
        assert set() in sets

    def test_collider_not_in_adjustment_set(self):
        g = _graph("C ~ X + Y")
        sets = adjustment_sets(g, "X", "Y")
        for s in sets:
            assert "C" not in s

    def test_collider_warning(self):
        g = _graph("C ~ X + Y")
        warnings = collider_warnings(g, {"C"}, "X", "Y")
        assert len(warnings) > 0
        assert "collider" in warnings[0].lower()


class TestDiamondDAG:
    """Z -> X, Z -> M, X -> M, M -> Y. Adjust for Z."""

    def test_adjustment_sets(self):
        g = _graph("X ~ Z\nM ~ X + Z\nY ~ M")
        sets = adjustment_sets(g, "X", "Y")
        assert len(sets) > 0
        assert is_identifiable(g, "X", "Y")


class TestDirectEffect:
    """X -> Y only. No confounders, no mediators."""

    def test_empty_set_valid(self):
        g = _graph("Y ~ X")
        sets = adjustment_sets(g, "X", "Y")
        assert set() in sets


class TestMediationDAG:
    """X -> M -> Y, X -> Y. Empty set valid for total effect."""

    def test_empty_set_valid_for_total_effect(self):
        g = _graph("M ~ X\nY ~ M + X")
        sets = adjustment_sets(g, "X", "Y")
        assert set() in sets

    def test_m_not_in_adjustment_set(self):
        """M is a descendant of X, so it can't appear in a backdoor set."""
        g = _graph("M ~ X\nY ~ M + X")
        sets = adjustment_sets(g, "X", "Y")
        for s in sets:
            assert "M" not in s


class TestErrorHandling:
    def test_unknown_treatment_raises(self):
        g = _graph("Y ~ X")
        with pytest.raises(ValueError, match="not in DAG"):
            adjustment_sets(g, "UNKNOWN", "Y")

    def test_unknown_outcome_raises(self):
        g = _graph("Y ~ X")
        with pytest.raises(ValueError, match="not in DAG"):
            adjustment_sets(g, "X", "UNKNOWN")


class TestPathModelIntegration:
    """Test identification methods on PathModel."""

    def test_adjustment_sets_method(self):
        df = pd.DataFrame({
            "X": np.random.normal(size=50),
            "Z": np.random.normal(size=50),
            "Y": np.random.normal(size=50),
        })
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        sets = model.adjustment_sets("X", "Y")
        assert {"Z"} in sets

    def test_is_identifiable_method(self):
        df = pd.DataFrame({
            "X": np.random.normal(size=50),
            "Z": np.random.normal(size=50),
            "Y": np.random.normal(size=50),
        })
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        assert model.is_identifiable("X", "Y")

    def test_collider_warnings_method(self):
        df = pd.DataFrame({
            "X": np.random.normal(size=50),
            "Y": np.random.normal(size=50),
            "C": np.random.normal(size=50),
        })
        model = pathmc.model("C ~ X + Y", data=df)
        warnings = model.collider_warnings({"C"}, "X", "Y")
        assert len(warnings) > 0


class TestTemporalEdgesIdentification:
    """Temporal edges must not change identification results (#16)."""

    def test_lag_model_identifiable(self):
        g = _graph("sales ~ spend + lag(sales)")
        assert is_identifiable(g, "spend", "sales")

    def test_lag_adjustment_sets_unchanged(self):
        """Adjustment sets for spend -> sales should be the same
        with or without lag(sales)."""
        g_lag = _graph("sales ~ spend + lag(sales)")
        g_no_lag = _graph("sales ~ spend")
        sets_lag = adjustment_sets(g_lag, "spend", "sales")
        sets_no_lag = adjustment_sets(g_no_lag, "spend", "sales")
        assert sets_lag == sets_no_lag

    def test_lag_no_collider_warnings(self):
        g = _graph("sales ~ spend + lag(sales)")
        warnings = collider_warnings(g, {"lag(sales)"}, "spend", "sales")
        assert len(warnings) == 0
