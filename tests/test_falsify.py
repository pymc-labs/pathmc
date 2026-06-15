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

import itertools
import math

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.falsify import (
    _MAX_EXACT_NODES,
    FalsificationResult,
    _parental_triples,
    _PartialCorrelationTester,
    _permuted_dags,
    _validate_lmc,
    _validate_tpa,
    falsify_graph,
)
from pathmc.graph import build_graph
from pathmc.parse import parse_spec


def _graph(spec_str):
    return build_graph(parse_spec(spec_str))


def _to_nw(df):
    return nw.from_native(df, eager_only=True)


def _dowhy_verdict(p_lmc, p_tpa, alpha):
    """Reference transcription of dowhy's EvaluationResult.__post_init__.

    Returns ``(falsifiable, falsified)``. pathmc must match this exactly
    whenever the result can be evaluated.
    """
    if p_lmc > alpha > p_tpa:
        return True, True
    if alpha < p_tpa:
        return False, False
    return True, False


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


@pytest.fixture
def chain_data():
    """Data faithful to the chain X -> M -> Y (no direct X -> Y)."""
    n = 1200
    rng = np.random.default_rng(11)
    X = rng.normal(size=n)
    M = 0.8 * X + rng.normal(scale=0.5, size=n)
    Y = 0.7 * M + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.fixture
def missing_edge_data():
    """Chain data where X *also* directly affects Y (X -> Y omitted)."""
    n = 1200
    rng = np.random.default_rng(13)
    X = rng.normal(size=n)
    M = 0.8 * X + rng.normal(scale=0.5, size=n)
    Y = 0.7 * M + 0.6 * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.fixture
def collider_data():
    """X and Y independent; both cause C (collider)."""
    n = 1200
    rng = np.random.default_rng(17)
    X = rng.normal(size=n)
    Y = rng.normal(size=n)
    C = 0.7 * X + 0.7 * Y + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "Y": Y, "C": C})


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


class TestPartialCorrelationTester:
    """Unit tests for the memoized partial-correlation CI engine."""

    def _tester(self, df, variables=None):
        if variables is None:
            variables = list(df.columns)
        return _PartialCorrelationTester(_to_nw(df), variables)

    def test_marginal_independence_high_p(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "X": rng.normal(size=500),
            "Y": rng.normal(size=500),
        })
        p = self._tester(df).p_value("X", "Y", ())
        assert p is not None
        assert p > 0.05

    def test_marginal_dependence_low_p(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        df = pd.DataFrame({"X": x, "Y": 0.9 * x + rng.normal(scale=0.3, size=500)})
        p = self._tester(df).p_value("X", "Y", ())
        assert p is not None
        assert p < 1e-6

    def test_conditional_independence_recovered(self):
        # Chain X -> Z -> Y: X and Y are dependent marginally but
        # independent given Z.
        rng = np.random.default_rng(1)
        x = rng.normal(size=800)
        z = 0.9 * x + rng.normal(scale=0.3, size=800)
        y = 0.9 * z + rng.normal(scale=0.3, size=800)
        df = pd.DataFrame({"X": x, "Y": y, "Z": z})
        tester = self._tester(df)
        assert tester.p_value("X", "Y", ()) < 1e-3
        assert tester.p_value("X", "Y", ("Z",)) > 0.05

    def test_missing_column_returns_none(self):
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [1.0, 0.0, 1.0, 0.0]})
        tester = self._tester(df, variables=["X", "Y"])
        assert tester.p_value("X", "MISSING", ()) is None
        assert tester.p_value("X", "Y", ("MISSING",)) is None

    def test_too_few_rows_returns_none(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [2.0, 1.0]})
        assert self._tester(df).p_value("X", "Y", ()) is None

    def test_constant_column_returns_none(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(0).normal(size=50),
            "Y": np.ones(50),
        })
        assert self._tester(df).p_value("X", "Y", ()) is None

    def test_constant_conditioning_equals_marginal(self):
        # A constant conditioning variable is absorbed into the intercept,
        # so the conditional test must exactly equal the marginal one
        # (effective rank handles the degrees of freedom).
        rng = np.random.default_rng(2)
        x = rng.normal(size=200)
        df = pd.DataFrame({
            "X": x,
            "Y": 0.8 * x + rng.normal(scale=0.3, size=200),
            "Z": np.ones(200),
        })
        tester = self._tester(df)
        p_marginal = tester.p_value("X", "Y", ())
        p_conditional = tester.p_value("X", "Y", ("Z",))
        assert p_conditional is not None
        assert p_conditional == pytest.approx(p_marginal)

    def test_collinear_conditioner_equals_single(self):
        # A duplicated conditioning column must not change the result vs.
        # conditioning on one copy (rank-based degrees of freedom).
        rng = np.random.default_rng(6)
        z = rng.normal(size=300)
        x = 0.5 * z + rng.normal(scale=0.5, size=300)
        y = 0.5 * z + rng.normal(scale=0.5, size=300)
        df = pd.DataFrame({"X": x, "Y": y, "Z": z, "Zdup": z.copy()})
        tester = self._tester(df)
        p_one = tester.p_value("X", "Y", ("Z",))
        p_dup = tester.p_value("X", "Y", ("Z", "Zdup"))
        assert p_dup is not None
        assert p_dup == pytest.approx(p_one)

    def test_non_numeric_column_skipped(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        df = pd.DataFrame({
            "X": x,
            "Y": 0.5 * x + rng.normal(scale=0.5, size=40),
            "G": ["a", "b"] * 20,
        })
        tester = self._tester(df, variables=["X", "Y", "G"])
        # Numeric pair still works; any test needing the string column skips.
        assert tester.p_value("X", "Y", ()) is not None
        assert tester.p_value("X", "G", ()) is None
        assert tester.p_value("X", "Y", ("G",)) is None

    def test_perfect_collinearity_returns_zero(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=100)
        df = pd.DataFrame({"X": x, "Y": x.copy()})
        assert self._tester(df).p_value("X", "Y", ()) == 0.0

    def test_nan_rows_dropped(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=200)
        y = 0.8 * x + rng.normal(scale=0.3, size=200)
        x[:5] = np.nan
        df = pd.DataFrame({"X": x, "Y": y})
        p = self._tester(df).p_value("X", "Y", ())
        assert p is not None
        assert p < 1e-3

    def test_symmetry_and_cache(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=300)
        df = pd.DataFrame({"X": x, "Y": 0.5 * x + rng.normal(scale=0.5, size=300)})
        tester = self._tester(df)
        p_xy = tester.p_value("X", "Y", ())
        p_yx = tester.p_value("Y", "X", ())
        assert p_xy == p_yx
        # Second call hits the cache and returns the identical object.
        assert tester.p_value("X", "Y", ()) == p_xy


class TestParentalTriplesDetailed:
    def test_collider_unconditional_triple(self):
        # X -> C <- Y implies X ⊥ Y (unconditional).
        dag = _graph("C ~ X + Y").contemporaneous_dag
        triples = _parental_triples(dag, include_unconditional=True)
        pairs = {(x, y) for x, y, _ in triples}
        assert ("X", "Y") in pairs or ("Y", "X") in pairs
        # The conditioning set for that triple is empty.
        for x, y, parents in triples:
            if {x, y} == {"X", "Y"}:
                assert parents == ()

    def test_collider_no_triples_without_unconditional(self):
        dag = _graph("C ~ X + Y").contemporaneous_dag
        triples = _parental_triples(dag, include_unconditional=False)
        assert triples == []

    def test_diamond_conditioning_sets(self):
        # A->B, A->C, B->D, C->D. B ⊥ C | A is the implication.
        dag = _graph("B ~ A\nC ~ A\nD ~ B + C").contemporaneous_dag
        triples = _parental_triples(dag, include_unconditional=True)
        bc = [t for t in triples if {t[0], t[1]} == {"B", "C"}]
        assert bc
        for _, _, parents in bc:
            assert parents == ("A",)

    def test_parents_excluded_from_non_descendants(self):
        dag = _graph("B ~ A\nC ~ A + B").contemporaneous_dag
        triples = _parental_triples(dag, include_unconditional=True)
        for node, non_desc, parents in triples:
            assert non_desc not in parents
            assert non_desc != node


class TestValidateLmc:
    def test_chain_no_violation(self, chain_data):
        dag = _graph("M ~ X\nY ~ M").contemporaneous_dag
        tester = _PartialCorrelationTester(_to_nw(chain_data), list(chain_data.columns))
        n_tests, n_viol, local = _validate_lmc(dag, tester, 0.05, True)
        assert n_tests == 1
        assert n_viol == 0
        assert len(local) == 1

    def test_missing_edge_violation(self, missing_edge_data):
        dag = _graph("M ~ X\nY ~ M").contemporaneous_dag
        tester = _PartialCorrelationTester(
            _to_nw(missing_edge_data), list(missing_edge_data.columns)
        )
        n_tests, n_viol, _ = _validate_lmc(dag, tester, 0.05, True)
        assert n_tests == 1
        assert n_viol == 1

    def test_skipped_tests_not_counted(self, chain_data):
        # Y has no data column -> every triple needing Y is skipped.
        df = chain_data.drop(columns=["Y"])
        dag = _graph("M ~ X\nY ~ M").contemporaneous_dag
        tester = _PartialCorrelationTester(_to_nw(df), list(dag.nodes))
        n_tests, n_viol, local = _validate_lmc(dag, tester, 0.05, True)
        assert n_tests == 0
        assert n_viol == 0
        assert local == []


class TestValidateTpaDetailed:
    def test_non_equivalent_graph_has_violations(self):
        # A chain and a collider over the same skeleton are NOT Markov
        # equivalent, so d-separations disagree.
        chain = _graph("M ~ X\nY ~ M").contemporaneous_dag
        collider = _graph("M ~ X + Y").contemporaneous_dag
        _, n_viol = _validate_tpa(collider, chain, include_unconditional=True)
        assert n_viol > 0

    def test_self_reference_zero(self, five_node_data):
        dag = _graph(TRUE_SPEC).contemporaneous_dag
        n_tests, n_viol = _validate_tpa(dag, dag, include_unconditional=True)
        assert n_tests > 0
        assert n_viol == 0


class TestPermutedDags:
    def test_preserves_structure(self):
        dag = _graph("B ~ A\nC ~ A\nD ~ B + C").contemporaneous_dag
        rng = np.random.default_rng(0)
        for g in _permuted_dags(dag, 10, rng):
            assert g.number_of_nodes() == dag.number_of_nodes()
            assert g.number_of_edges() == dag.number_of_edges()
            assert nx_is_dag(g)

    def test_identity_included_in_enumeration(self):
        dag = _graph("B ~ A\nC ~ B").contemporaneous_dag
        rng = np.random.default_rng(0)
        graphs = list(_permuted_dags(dag, 10_000, rng))
        assert any(set(g.edges) == set(dag.edges) for g in graphs)

    def test_single_node_graph(self):
        # A one-node DAG cannot be built from a spec (specs need an edge),
        # so construct the networkx graph directly.
        import networkx as nx

        g = nx.DiGraph()
        g.add_node("A")
        rng = np.random.default_rng(0)
        graphs = list(_permuted_dags(g, 5, rng))
        assert len(graphs) == math.factorial(1)

    def test_sampling_path_deterministic_with_seed(self):
        dag = _graph(
            "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, 9))
        ).contemporaneous_dag
        g1 = [
            tuple(sorted(g.edges))
            for g in _permuted_dags(dag, 6, np.random.default_rng(3))
        ]
        g2 = [
            tuple(sorted(g.edges))
            for g in _permuted_dags(dag, 6, np.random.default_rng(3))
        ]
        assert g1 == g2

    def test_exact_node_boundary(self):
        # n == _MAX_EXACT_NODES enumerates; n == _MAX_EXACT_NODES + 1 samples.
        exact = _graph(
            "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, _MAX_EXACT_NODES))
        ).contemporaneous_dag
        assert exact.number_of_nodes() == _MAX_EXACT_NODES
        rng = np.random.default_rng(0)
        assert len(list(_permuted_dags(exact, 10**9, rng))) == math.factorial(
            _MAX_EXACT_NODES
        )
        too_big = _graph(
            "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, _MAX_EXACT_NODES + 1))
        ).contemporaneous_dag
        assert too_big.number_of_nodes() == _MAX_EXACT_NODES + 1
        assert len(list(_permuted_dags(too_big, 7, np.random.default_rng(0)))) == 7


class TestResultProperties:
    def test_pvalues_in_unit_interval(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=100, random_seed=1)
        assert 0.0 <= r.p_value_lmc <= 1.0
        assert 0.0 <= r.p_value_tpa <= 1.0

    def test_n_in_mec_matches_p_tpa(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=100, random_seed=1)
        assert r.p_value_tpa == pytest.approx(r.n_in_mec / r.n_permutations)

    def test_perm_array_lengths(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=64, random_seed=1)
        assert len(r.perm_lmc_violation_fractions) == r.n_permutations
        assert len(r.perm_tpa_violation_fractions) == r.n_permutations

    def test_violation_fraction_consistent(self, missing_edge_data):
        m = pathmc.model("M ~ X\nY ~ M", data=missing_edge_data)
        r = m.falsify(random_seed=0)
        if r.can_evaluate and r.n_lmc_tests:
            assert r.given_lmc_violation_fraction == pytest.approx(
                r.given_lmc_violations / r.n_lmc_tests
            )

    def test_local_violations_schema(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=50, random_seed=1)
        for col in [
            "node",
            "non_descendant",
            "conditioning_set",
            "p_value",
            "violation",
        ]:
            assert col in r.local_violations.columns
        assert r.local_violations["violation"].dtype == bool


class TestResultDisplay:
    def _make(self, p_lmc, p_tpa, n_in_mec, alpha=0.05):
        return FalsificationResult(
            given_lmc_violations=2,
            n_lmc_tests=6,
            given_lmc_violation_fraction=2 / 6,
            perm_lmc_violation_fractions=np.linspace(0, 1, 20),
            perm_tpa_violation_fractions=np.linspace(0, 1, 20),
            p_value_lmc=p_lmc,
            p_value_tpa=p_tpa,
            n_permutations=20,
            n_in_mec=n_in_mec,
            significance_level=alpha,
            significance_ci=0.05,
            local_violations=pd.DataFrame({
                "node": ["C"],
                "non_descendant": ["B"],
                "conditioning_set": ["A"],
                "p_value": [0.001],
                "violation": [True],
            }),
        )

    def test_repr_rejected(self):
        r = self._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        text = repr(r)
        assert "reject" in text.lower()
        assert "we do not reject" not in text.lower()

    def test_repr_not_rejected(self):
        r = self._make(p_lmc=0.0, p_tpa=0.0, n_in_mec=0)
        assert "we do not reject" in repr(r).lower()

    def test_html_states(self):
        rejected = self._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        assert "Rejected" in rejected._repr_html_()
        not_informative = self._make(p_lmc=0.0, p_tpa=0.5, n_in_mec=10)
        assert "Not informative" in not_informative._repr_html_()
        not_rejected = self._make(p_lmc=0.0, p_tpa=0.0, n_in_mec=0)
        assert "Not rejected" in not_rejected._repr_html_()

    def test_plot_returns_figure(self):
        import matplotlib

        matplotlib.use("Agg")
        r = self._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        fig = r.plot()
        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_on_supplied_axis(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r = self._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        fig, ax = plt.subplots()
        out = r.plot(ax=ax, bins=5)
        assert out is fig

    def test_violations_only_true_rows(self):
        r = self._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        assert (r.violations["violation"]).all()


class TestDecisionRuleGrid:
    """Exhaustively match dowhy's verdict across the p-value grid."""

    GRID = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1.0]

    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.1])
    @pytest.mark.parametrize("p_tpa", GRID)
    @pytest.mark.parametrize("p_lmc", GRID)
    def test_matches_reference(self, p_lmc, p_tpa, alpha):
        r = FalsificationResult(
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
        exp_falsifiable, exp_falsified = _dowhy_verdict(p_lmc, p_tpa, alpha)
        assert r.falsifiable is exp_falsifiable
        assert r.falsified is exp_falsified

    def test_falsified_implies_falsifiable(self):
        # Assert the invariant on the actual FalsificationResult object,
        # not just the reference helper.
        for p_lmc, p_tpa in itertools.product(self.GRID, self.GRID):
            r = FalsificationResult(
                given_lmc_violations=0,
                n_lmc_tests=6,
                given_lmc_violation_fraction=0.0,
                perm_lmc_violation_fractions=np.zeros(20),
                perm_tpa_violation_fractions=np.zeros(20),
                p_value_lmc=p_lmc,
                p_value_tpa=p_tpa,
                n_permutations=20,
                n_in_mec=0,
                significance_level=0.05,
                significance_ci=0.05,
                local_violations=pd.DataFrame(),
            )
            if r.falsified:
                assert r.falsifiable


class TestStatisticalBehavior:
    def test_correct_chain_not_rejected(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=100, random_seed=2)
        assert r.given_lmc_violations == 0
        assert r.falsified is False

    def test_missing_edge_flagged_locally(self):
        # Funnel where the proposed chain omits the real Budget -> Sales
        # edge. This is a *local* violation check: the implied independence
        # Budget ⊥ Sales | Clicks is contradicted by the data. (Whether the
        # whole-graph verdict rejects depends on the permutation baseline,
        # tested separately.)
        n = 1500
        rng = np.random.default_rng(21)
        budget = rng.normal(loc=10, scale=2, size=n)
        ads = 0.8 * budget + rng.normal(scale=0.5, size=n)
        clicks = 0.6 * ads + rng.normal(scale=0.5, size=n)
        sales = 0.5 * clicks + 0.5 * budget + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({
            "Budget": budget,
            "Ads": ads,
            "Clicks": clicks,
            "Sales": sales,
        })
        m = pathmc.model("Ads ~ Budget\nClicks ~ Ads\nSales ~ Clicks", data=df)
        r = m.falsify(n_permutations=200, random_seed=1)
        assert r.given_lmc_violations >= 1
        violated_pairs = {
            frozenset((row["node"], row["non_descendant"]))
            for _, row in r.violations.iterrows()
        }
        assert frozenset(("Budget", "Sales")) in violated_pairs

    def test_collider_consistent(self, collider_data):
        # X ⊥ Y unconditionally holds for a collider; the DAG should not be
        # falsified for that reason.
        m = pathmc.model("C ~ X + Y", data=collider_data)
        r = m.falsify(random_seed=0)
        # The only implication is X ⊥ Y (unconditional), which holds.
        assert r.given_lmc_violations == 0


class TestIncludeUnconditional:
    def test_changes_test_count(self, collider_data):
        m = pathmc.model("C ~ X + Y", data=collider_data)
        with_uncond = m.falsify(include_unconditional=True, random_seed=0)
        without_uncond = m.falsify(include_unconditional=False, random_seed=0)
        assert with_uncond.n_lmc_tests > without_uncond.n_lmc_tests
        # Without unconditional tests, the collider DAG has no implications.
        assert without_uncond.can_evaluate is False


class TestApiForwarding:
    def test_method_forwards_significance_level(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=80, significance_level=0.2, random_seed=1)
        assert r.significance_level == 0.2

    def test_works_before_and_after_fit(self, chain_data):
        m = pathmc.model("M ~ X\nY ~ M", data=chain_data)
        before = m.falsify(random_seed=0)
        m.fit(draws=50, tune=50, chains=1, random_seed=0)
        after = m.falsify(random_seed=0)
        assert before.given_lmc_violations == after.given_lmc_violations


class TestBackends:
    def test_polars_backend(self, chain_data):
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(chain_data)
        m = pathmc.model("M ~ X\nY ~ M", data=df)
        r = m.falsify(random_seed=0)
        assert r.given_lmc_violations == 0


class TestEdgeCaseData:
    def test_nan_rows_tolerated(self, five_node_data):
        # Call falsify_graph directly so the model compiler's NaN-imputation
        # path is not exercised; falsify drops incomplete rows per CI test.
        df = five_node_data.copy()
        df.loc[:20, "C"] = np.nan
        r = falsify_graph(
            build_graph(parse_spec(TRUE_SPEC)),
            _to_nw(df),
            n_permutations=50,
            random_seed=1,
        )
        assert r.can_evaluate is True
        assert r.given_lmc_violations == 0

    def test_disconnected_components(self):
        # Two independent chains: A->B and C->D, plus E->F. No cross edges.
        n = 800
        rng = np.random.default_rng(7)
        A = rng.normal(size=n)
        B = 0.8 * A + rng.normal(scale=0.5, size=n)
        C = rng.normal(size=n)
        D = 0.8 * C + rng.normal(scale=0.5, size=n)
        E = rng.normal(size=n)
        F = 0.8 * E + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "E": E, "F": F})
        m = pathmc.model("B ~ A\nD ~ C\nF ~ E", data=df)
        r = m.falsify(n_permutations=100, random_seed=1)
        assert r.can_evaluate is True
        assert r.given_lmc_violations == 0

    def test_n_permutations_one(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=1, random_seed=1)
        assert r.n_permutations == 1
        assert r.p_value_lmc in (0.0, 1.0)

    def test_significance_level_half(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(significance_level=0.5, random_seed=1)
        # default n_permutations = round(1/0.5) = 2
        assert r.n_permutations <= 2
        assert r.significance_level == 0.5


class TestResidualCovariance:
    """`~~` (residual covariance / confounding) is unsupported and rejected.

    Falsification, like dowhy's gcm.falsify_graph, models directed DAGs
    only. Residual covariances encode bidirected (ADMG) confounding whose
    testable implications require m-separation; rather than return wrong
    results, falsify rejects such models with a clear error.
    """

    @pytest.fixture
    def confounded_data(self):
        n = 1000
        rng = np.random.default_rng(31)
        U = rng.normal(size=n)
        X = 0.9 * U + rng.normal(scale=0.4, size=n)
        Y = 0.9 * U + rng.normal(scale=0.4, size=n)
        W = 0.5 * X + 0.5 * Y + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"X": X, "Y": Y, "W": W})

    def test_residual_covariance_rejected(self, confounded_data):
        with pytest.raises(ValueError, match="residual covariance"):
            falsify_graph(
                build_graph(parse_spec("W ~ X + Y\nX ~~ Y")),
                _to_nw(confounded_data),
                random_seed=0,
            )

    def test_residual_covariance_rejected_via_method(self, confounded_data):
        m = pathmc.model("W ~ X + Y\nX ~~ Y", data=confounded_data)
        with pytest.raises(ValueError, match="residual covariance"):
            m.falsify(random_seed=0)

    def test_dangling_residual_endpoint_rejected(self):
        # Y appears only in a ~~ term (not a DAG node): must raise cleanly,
        # not KeyError.
        n = 200
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "X": rng.normal(size=n),
            "Y": rng.normal(size=n),
            "Z": rng.normal(size=n),
        })
        with pytest.raises(ValueError, match="residual covariance"):
            falsify_graph(
                build_graph(parse_spec("Z ~ X\nX ~~ Y")),
                _to_nw(df),
                random_seed=0,
            )

    def test_directed_only_still_works(self, confounded_data):
        # Removing ~~ lets falsification proceed on the directed structure.
        r = falsify_graph(
            build_graph(parse_spec("W ~ X + Y")),
            _to_nw(confounded_data),
            random_seed=0,
        )
        assert r.n_lmc_tests >= 0


class TestStringColumns:
    def test_string_column_does_not_crash(self, chain_data):
        df = chain_data.copy()
        df["G"] = ["a", "b", "c"] * (len(df) // 3) + ["a"] * (len(df) % 3)
        r = falsify_graph(
            build_graph(parse_spec("M ~ X + G\nY ~ M")),
            _to_nw(df),
            random_seed=0,
        )
        # The numeric implications still run; nothing crashes.
        assert r.given_lmc_violations == 0


class TestDocExamplePaths:
    """Lock in the exact behavior shown in the documentation example."""

    def test_five_node_enumerates_120(self, five_node_data):
        # The example calls falsify(n_permutations=200) on a 5-node DAG;
        # since 200 >= 5! = 120, all 120 relabelings are enumerated.
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=200, random_seed=1)
        assert r.n_permutations == math.factorial(5)
        assert r.falsifiable is True
        assert r.falsified is False

    def test_five_node_wrong_dag_rejected(self, five_node_data):
        m = pathmc.model(WRONG_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=200, random_seed=1)
        assert r.n_permutations == math.factorial(5)
        assert r.falsified is True


class TestSignificanceLevelCapping:
    def test_tiny_significance_default_perms_capped(self):
        # On a >7-node graph, a tiny significance_level implies a huge
        # default n_permutations; it must be capped, not error out.
        from pathmc.falsify import _DEFAULT_PERMUTATION_CAP

        n = 60
        rng = np.random.default_rng(0)
        cols = {f"N{i}": rng.normal(size=n) for i in range(9)}
        df = pd.DataFrame(cols)
        spec = "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, 9))
        r = falsify_graph(
            build_graph(parse_spec(spec)),
            _to_nw(df),
            significance_level=1e-9,
            random_seed=0,
        )
        assert r.n_permutations == _DEFAULT_PERMUTATION_CAP

    def test_explicit_huge_still_rejected(self):
        n = 60
        rng = np.random.default_rng(0)
        cols = {f"N{i}": rng.normal(size=n) for i in range(9)}
        df = pd.DataFrame(cols)
        spec = "\n".join(f"N{i} ~ N{i - 1}" for i in range(1, 9))
        with pytest.raises(ValueError, match="too large"):
            falsify_graph(
                build_graph(parse_spec(spec)),
                _to_nw(df),
                n_permutations=10_000_000,
            )


class TestPlotValidation:
    def test_negative_bins_rejected(self):
        import matplotlib

        matplotlib.use("Agg")
        r = TestResultDisplay()._make(p_lmc=0.5, p_tpa=0.0, n_in_mec=0)
        with pytest.raises(ValueError, match="bins"):
            r.plot(bins=0)


class TestMiscEdgeCases:
    def test_integer_columns(self, five_node_data):
        int_df = (five_node_data * 10).round().astype(int)
        m = pathmc.model(TRUE_SPEC, data=int_df)
        r = m.falsify(n_permutations=50, random_seed=1)
        assert r.can_evaluate is True

    def test_isolated_two_node_graph(self):
        # Two unconnected nodes imply a single unconditional independence
        # X ⊥ Y; with no edges the graph is uninformative (every relabeling
        # shares its Markov equivalence class).
        import networkx as nx

        rng = np.random.default_rng(0)
        df = pd.DataFrame({"X": rng.normal(size=200), "Y": rng.normal(size=200)})
        g = nx.DiGraph()
        g.add_nodes_from(["X", "Y"])
        triples = _parental_triples(g, include_unconditional=True)
        assert len(triples) >= 1

    def test_random_seed_none_runs(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(n_permutations=30, random_seed=None)
        assert 0.0 <= r.p_value_tpa <= 1.0

    def test_significance_level_near_one(self, five_node_data):
        m = pathmc.model(TRUE_SPEC, data=five_node_data)
        r = m.falsify(significance_level=0.999, random_seed=1)
        # default n_permutations = round(1/0.999) = 1
        assert r.n_permutations == 1


def nx_is_dag(g):
    import networkx as nx

    return nx.is_directed_acyclic_graph(g)
