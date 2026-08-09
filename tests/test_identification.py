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
from pathmc.identify import (
    adjustment_sets,
    collider_warnings,
    implied_independences,
    is_identifiable,
)
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


class TestResidualCovarianceIdentification:
    """``~~`` declares unobserved confounding, so the backdoor
    criterion must not report the effect as identifiable (#344)."""

    IV_SPEC = "T ~ Z\nY ~ T\nT ~~ Y"

    def test_residual_block_blocks_identification(self):
        g = _graph(self.IV_SPEC)
        assert not is_identifiable(g, "T", "Y")

    def test_residual_block_has_no_adjustment_set(self):
        g = _graph(self.IV_SPEC)
        assert adjustment_sets(g, "T", "Y") == []

    def test_without_residual_block_effect_is_identified(self):
        """Same DAG minus the ~~ edge: nothing to adjust for, so the
        empty set is valid and the effect is identifiable."""
        g = _graph("T ~ Z\nY ~ T")
        assert is_identifiable(g, "T", "Y")

    def test_synthetic_confounder_not_offered_for_adjustment(self):
        """The latent node is never a candidate adjustment variable."""
        g = _graph("T ~ Z\nY ~ T + W\nT ~~ Y")
        for s in adjustment_sets(g, "T", "Y"):
            assert not any(v.startswith("_u_resid_") for v in s)

    def test_unrelated_residual_block_does_not_break_identification(self):
        """A ~~ block between two variables that are not the treatment
        and outcome must leave the T -> Y effect identifiable."""
        g = _graph("M1 ~ X\nM2 ~ X\nY ~ T\nM1 ~~ M2")
        assert is_identifiable(g, "T", "Y")

    def test_model_method_reports_not_identifiable(self):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "Z": rng.normal(size=n),
            "T": rng.normal(size=n),
            "Y": rng.normal(size=n),
        })
        model = pathmc.model(self.IV_SPEC, data=df)
        assert not model.is_identifiable("T", "Y")


class TestResidualCovarianceColliderConsistency:
    """``collider_warnings`` must reason about the same residual-
    augmented DAG as the backdoor/frontdoor path, otherwise it can miss
    a collider that only exists because of a declared ``~~`` block."""

    def test_collider_opened_by_residual_block_is_flagged(self):
        """T -> V, O -> Y are separate branches; V and O are joined only
        by an unobserved common cause (V ~~ O). That makes V a collider
        on the path T -> V <- u -> O -> Y, which is invisible if the
        collider search only looks at V's declared (single) parent."""
        g = _graph("T ~ Z\nV ~ T\nO ~ Z2\nY ~ O\nV ~~ O")
        warnings = collider_warnings(g, {"V"}, "T", "Y")
        assert any("collider" in w for w in warnings)

    def test_no_false_collider_without_residual_block(self):
        """Same skeleton without the ~~ block: V has one parent and is
        not a collider anywhere."""
        g = _graph("T ~ Z\nV ~ T\nO ~ Z2\nY ~ O")
        warnings = collider_warnings(g, {"V"}, "T", "Y")
        assert warnings == []


class TestResidualCovarianceIndependenceConsistency:
    """``implied_independences`` must not assert an independence that a
    declared ``~~`` block explicitly contradicts."""

    def test_residual_confounded_pair_not_claimed_independent(self):
        """A and B share no directed edge but are joined by A ~~ B, so
        no conditioning set (observed or not) makes them independent."""
        g = _graph("A ~ Za\nB ~ Zb\nA ~~ B")
        pairs = {(ci.x, ci.y) for ci in implied_independences(g)}
        assert ("A", "B") not in pairs

    def test_unrelated_pair_still_reported_independent(self):
        """Za and Zb share no path at all, residual block or otherwise,
        so the independence still holds and should still be reported."""
        g = _graph("A ~ Za\nB ~ Zb\nA ~~ B")
        pairs = {(ci.x, ci.y) for ci in implied_independences(g)}
        assert ("Za", "Zb") in pairs

    def test_confounder_with_descendants_not_claimed_independent(self):
        """Latent projection must keep X and Y d-connected when the
        confounder has descendants (Z ~ X, T ~ Y)."""
        g = _graph("Z ~ X\nT ~ Y\nX ~~ Y")
        pairs = {(ci.x, ci.y) for ci in implied_independences(g)}
        assert ("X", "Y") not in pairs

    def test_issue_277_no_spurious_test_implications_violation(self):
        """End-to-end regression for #277: a declared ``~~`` block must
        not surface X ⊥ Y as a violated implied independence."""
        rng = np.random.default_rng(31)
        n = 1000
        u = rng.normal(size=n)
        x = 0.9 * u + rng.normal(scale=0.4, size=n)
        y = 0.9 * u + rng.normal(scale=0.4, size=n)
        w = 0.5 * x + 0.5 * y + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": x, "Y": y, "W": w})
        model = pathmc.model("W ~ X + Y\nX ~~ Y", data=df)
        result = model.test_implications()
        assert result.n_violations == 0
        tested_pairs = {
            frozenset({row["x"], row["y"]}) for _, row in result.results.iterrows()
        }
        assert frozenset({"X", "Y"}) not in tested_pairs


class TestResidualNameCollision:
    """The synthetic latent name must never collide with a real,
    user-declared variable, even one named exactly like the generated
    name."""

    def test_user_variable_named_like_synthetic_latent_does_not_collide(self):
        """A real confounder literally named ``_u_resid_0`` opens a
        backdoor path T <- _u_resid_0 -> Y that adjusting for the real
        variable closes. An unrelated ~~ block elsewhere in the graph
        (which would generate a synthetic node with the same index and
        therefore the same default name) must not merge into it, mark
        it latent, or otherwise corrupt it."""
        g = _graph("T ~ Z + _u_resid_0\nY ~ T + _u_resid_0\nA ~ Z\nB ~ Z\nA ~~ B")
        sets = adjustment_sets(g, "T", "Y")
        assert {"_u_resid_0"} in sets
        assert is_identifiable(g, "T", "Y")
