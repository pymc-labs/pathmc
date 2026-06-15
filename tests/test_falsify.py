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
"""Tests for whole-graph DAG falsification (pathmc.falsify)."""

import math

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.falsify import (
    FalsificationResult,
    _parental_triples,
    _validate_tpa,
    falsify_graph,
)
from pathmc.graph import build_graph
from pathmc.parse import parse_spec


def _graph(spec_str):
    return build_graph(parse_spec(spec_str))


@pytest.fixture
def five_node_data():
    """True DAG: A->B, A->C, B->D, C->D, D->E."""
    n = 1500
    rng = np.random.default_rng(3)
    A = rng.normal(size=n)
    B = 0.8 * A + rng.normal(scale=0.5, size=n)
    C = 0.7 * A + rng.normal(scale=0.5, size=n)
    D = 0.6 * B + 0.5 * C + rng.normal(scale=0.5, size=n)
    E = 0.9 * D + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E})


TRUE_SPEC = "B ~ A\nC ~ A\nD ~ B + C\nE ~ D"
WRONG_SPEC = "B ~ A\nC ~ A + E\nD ~ B\nE ~ D"


class TestParentalTriples:
    def test_chain_triples(self):
        g = _graph("M ~ X\nY ~ M")
        triples = _parental_triples(g.contemporaneous_dag, include_unconditional=True)
        # X ⊥ Y | M is the characteristic implication of a chain.
        assert ("X", "Y", ("M",)) in triples or ("Y", "X", ("M",)) in triples

    def test_excludes_descendants(self):
        g = _graph("M ~ X\nY ~ M")
        triples = _parental_triples(g.contemporaneous_dag, include_unconditional=True)
        for node, non_desc, _parents in triples:
            assert node != non_desc

    def test_root_only_when_unconditional(self):
        g = _graph("C ~ X + Y")
        with_uncond = _parental_triples(
            g.contemporaneous_dag, include_unconditional=True
        )
        without_uncond = _parental_triples(
            g.contemporaneous_dag, include_unconditional=False
        )
        # Roots X, Y have no parents; their (unconditional) triples vanish
        # when include_unconditional is False.
        assert len(with_uncond) > len(without_uncond)


class TestValidateTpa:
    def test_zero_violations_against_self(self):
        g = _graph("B ~ A\nC ~ A\nD ~ B + C").contemporaneous_dag
        n_tests, n_viol = _validate_tpa(g, g, include_unconditional=True)
        assert n_tests > 0
        assert n_viol == 0

    def test_markov_equivalent_chain_no_violation(self):
        # X->M->Y and its reverse Y->M->X are Markov equivalent.
        forward = _graph("M ~ X\nY ~ M").contemporaneous_dag
        reverse = _graph("M ~ Y\nX ~ M").contemporaneous_dag
        _, n_viol = _validate_tpa(forward, reverse, include_unconditional=True)
        assert n_viol == 0


class TestVerdict:
    def test_true_dag_not_rejected(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=100, random_seed=1)
        assert r.falsifiable is True
        assert r.falsified is False
        assert r.given_lmc_violations == 0

    def test_wrong_dag_rejected(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=100, random_seed=1)
        assert r.falsifiable is True
        assert r.falsified is True
        assert r.given_lmc_violations > 0
        assert len(r.violations) == r.given_lmc_violations


class TestDecisionRule:
    """The verdict must follow Eulig et al. (2023) exactly."""

    def _result(self, p_lmc, p_tpa, alpha=0.05):
        return FalsificationResult(
            given_lmc_violations=0,
            n_lmc_tests=6,
            given_lmc_violation_fraction=0.0,
            perm_lmc_violation_fractions=np.zeros(20),
            perm_tpa_violation_fractions=np.zeros(20),
            p_value_lmc=p_lmc,
            p_value_tpa=p_tpa,
            n_permutations=20,
            n_in_mec=0,
            significance_level=alpha,
            significance_ci=0.05,
            local_violations=pd.DataFrame(),
        )

    def test_informative_and_beats_baseline_not_rejected(self):
        r = self._result(p_lmc=0.01, p_tpa=0.01)
        assert r.falsifiable is True
        assert r.falsified is False

    def test_informative_but_loses_baseline_rejected(self):
        r = self._result(p_lmc=0.25, p_tpa=0.00)
        assert r.falsifiable is True
        assert r.falsified is True

    def test_not_informative_not_falsifiable(self):
        r = self._result(p_lmc=0.50, p_tpa=0.40)
        assert r.falsifiable is False
        assert r.falsified is False

    def test_boundary_tpa_equals_alpha_is_informative(self):
        r = self._result(p_lmc=0.01, p_tpa=0.05, alpha=0.05)
        assert r.falsifiable is True

    def test_boundary_tpa_equals_alpha_not_rejected(self):
        # dowhy requires significance_level > p_tpa (strict) to reject.
        # At p_tpa == alpha the DAG is informative but NOT rejected, even
        # when it loses the LMC baseline. This is the reachable default
        # case where exactly one of 20 permutations lands in the MEC.
        r = self._result(p_lmc=0.25, p_tpa=0.05, alpha=0.05)
        assert r.falsifiable is True
        assert r.falsified is False

    def test_boundary_lmc_equals_alpha_not_rejected(self):
        # p_lmc == alpha is NOT > alpha, so the DAG is not rejected.
        r = self._result(p_lmc=0.05, p_tpa=0.00, alpha=0.05)
        assert r.falsified is False


class TestCannotEvaluate:
    def test_fully_connected_graph(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(0).normal(size=100),
            "Y": np.random.default_rng(1).normal(size=100),
        })
        # X -> Y is fully connected (no missing edge), no parental
        # non-descendant CIs to test.
        m = pathmc.model("Y ~ X", data=df)
        r = m.falsify(random_seed=0)
        assert r.can_evaluate is False
        assert r.falsifiable is None
        assert r.falsified is None

    def test_repr_handles_cannot_evaluate(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(0).normal(size=100),
            "Y": np.random.default_rng(1).normal(size=100),
        })
        m = pathmc.model("Y ~ X", data=df)
        r = m.falsify(random_seed=0)
        assert "cannot be evaluated" in repr(r).lower()
        assert "Cannot be evaluated" in r._repr_html_()

    def test_plot_raises_when_cannot_evaluate(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(0).normal(size=100),
            "Y": np.random.default_rng(1).normal(size=100),
        })
        m = pathmc.model("Y ~ X", data=df)
        r = m.falsify(random_seed=0)
        with pytest.raises(RuntimeError, match="cannot be evaluated"):
            r.plot()

    def test_violations_keeps_columns_when_empty(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(0).normal(size=100),
            "Y": np.random.default_rng(1).normal(size=100),
        })
        m = pathmc.model("Y ~ X", data=df)
        r = m.falsify(random_seed=0)
        # Accessing documented columns must not raise even when empty.
        assert "node" in r.violations.columns
        assert len(r.violations) == 0
        assert list(r.violations["node"]) == []


class TestSignificanceWiring:
    def test_ci_and_verdict_levels_independent(self, five_node_data):
        # A stricter CI level (fewer violations flagged) is wired
        # separately from the permutation-verdict level.
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        loose = m.falsify(n_permutations=100, significance_ci=0.05, random_seed=1)
        strict = m.falsify(n_permutations=100, significance_ci=1e-6, random_seed=1)
        assert strict.given_lmc_violations <= loose.given_lmc_violations
        assert strict.significance_ci == 1e-6
        assert strict.significance_level == 0.05


class TestLargeGraphNoExplosion:
    """Large graphs must sample, never silently enumerate n! relabelings."""

    def _chain_dag(self, n_nodes):
        spec = "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, n_nodes))
        return build_graph(parse_spec(spec)).contemporaneous_dag

    def test_large_graph_samples_requested_count(self):
        from pathmc.falsify import _permuted_dags

        # 8-node chain (8! = 40320). Requesting a huge count must NOT
        # enumerate 8!; it yields exactly the requested random sample.
        dag = self._chain_dag(8)
        rng = np.random.default_rng(0)
        graphs = list(_permuted_dags(dag, 5, rng))
        assert len(graphs) == 5
        for g in graphs:
            assert g.number_of_nodes() == dag.number_of_nodes()
            assert g.number_of_edges() == dag.number_of_edges()

    def test_small_graph_enumerates_exactly(self):
        from pathmc.falsify import _permuted_dags

        # 3-node chain (3! = 6). A large request enumerates exactly 6.
        dag = self._chain_dag(3)
        rng = np.random.default_rng(0)
        graphs = list(_permuted_dags(dag, 10_000, rng))
        assert len(graphs) == math.factorial(3)


class TestReproducibility:
    def test_same_seed_same_result(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r1 = m.falsify(n_permutations=50, random_seed=7)
        r2 = m.falsify(n_permutations=50, random_seed=7)
        assert r1.p_value_lmc == r2.p_value_lmc
        assert r1.p_value_tpa == r2.p_value_tpa
        np.testing.assert_array_equal(
            r1.perm_lmc_violation_fractions, r2.perm_lmc_violation_fractions
        )


class TestExactEnumeration:
    def test_small_graph_enumerates_all(self, five_node_data):
        # 3-node graph: 3! = 6 distinct relabelings. Requesting more than 6
        # enumerates all of them exactly (deterministic, seed-independent).
        df = five_node_data[["A", "B", "D"]]
        m = pathmc.model("B ~ A\nD ~ B", data=df)
        r_a = m.falsify(n_permutations=1000, random_seed=1)
        r_b = m.falsify(n_permutations=1000, random_seed=999)
        assert r_a.n_permutations == math.factorial(3)
        assert r_a.p_value_tpa == r_b.p_value_tpa
        assert r_a.p_value_lmc == r_b.p_value_lmc


class TestInputValidation:
    def test_bad_significance_level(self, five_node_data):
        with pytest.raises(ValueError, match="significance_level"):
            falsify_graph(
                build_graph(parse_spec(TRUE_SPEC)),
                _to_nw(five_node_data),
                significance_level=1.5,
            )

    def test_bad_significance_ci(self, five_node_data):
        with pytest.raises(ValueError, match="significance_ci"):
            falsify_graph(
                build_graph(parse_spec(TRUE_SPEC)),
                _to_nw(five_node_data),
                significance_ci=0.0,
            )

    def test_bad_n_permutations(self, five_node_data):
        with pytest.raises(ValueError, match="n_permutations"):
            falsify_graph(
                build_graph(parse_spec(TRUE_SPEC)),
                _to_nw(five_node_data),
                n_permutations=0,
            )

    def test_float_n_permutations_rejected(self, five_node_data):
        with pytest.raises(ValueError, match="n_permutations"):
            falsify_graph(
                build_graph(parse_spec(TRUE_SPEC)),
                _to_nw(five_node_data),
                n_permutations=1e7,
            )

    def test_bool_n_permutations_rejected(self, five_node_data):
        with pytest.raises(ValueError, match="n_permutations"):
            falsify_graph(
                build_graph(parse_spec(TRUE_SPEC)),
                _to_nw(five_node_data),
                n_permutations=True,
            )

    def test_huge_n_permutations_on_large_graph_rejected(self):
        # >7 nodes uses sampling; an absurd request is rejected, not hung.
        n = 50
        rng = np.random.default_rng(0)
        cols = {f"N{i}": rng.normal(size=n) for i in range(9)}
        df = pd.DataFrame(cols)
        spec = "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, 9))
        m = pathmc.model(spec, data=df)
        with pytest.raises(ValueError, match="too large"):
            m.falsify(n_permutations=10_000_000)

    def test_huge_n_permutations_on_small_graph_allowed(self, five_node_data):
        # Small graphs (<=7 nodes) cap at n! regardless, so a large request
        # is harmless and must not be rejected.
        df = five_node_data[["A", "B", "C"]]
        m = pathmc.model("B ~ A\nC ~ A", data=df)
        r = m.falsify(n_permutations=500_000, random_seed=0)
        assert r.n_permutations == math.factorial(3)


class TestRequiresData:
    def test_data_free_model_raises(self):
        m = pathmc.model(TRUE_SPEC)
        with pytest.raises(RuntimeError, match="requires data"):
            m.falsify()


class TestLatentSkipped:
    def test_latent_variable_skipped_in_ci(self):
        # M is latent (no data column). Triples needing M are skipped, but
        # M still participates in the d-separation oracle and permutations.
        n = 600
        rng = np.random.default_rng(5)
        X = rng.normal(size=n)
        M = 0.8 * X + rng.normal(scale=0.5, size=n)
        Y = 0.7 * M + rng.normal(scale=0.5, size=n)
        Z = 0.5 * Y + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
        m = pathmc.model("M ~ X\nY ~ M\nZ ~ Y", data=df, latent={"M"})
        r = m.falsify(n_permutations=50, random_seed=1)
        # Tests referencing the latent M must not appear in local results.
        involves_m = r.local_violations.apply(
            lambda row: (
                "M" in {row["node"], row["non_descendant"]}
                or "M" in row["conditioning_set"]
            ),
            axis=1,
        )
        assert not involves_m.any()


def _to_nw(df):
    import narwhals.stable.v1 as nw

    return nw.from_native(df, eager_only=True)
