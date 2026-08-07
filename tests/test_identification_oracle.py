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
"""Identification oracle tests (issue #327, item 4).

``pathmc.identify.adjustment_sets`` implements the backdoor criterion via a
hand-rolled path-enumeration + d-separation-blocking check
(``_blocks_all_backdoor_paths`` / ``_path_blocked_by``). That implementation
had no regression test against an independent ground truth, and no coverage
of collider / M-bias topologies where naive "adjust for everything" heuristics
give the wrong answer.

This module cross-checks ``adjustment_sets`` against two oracles:

1. Named canonical DAGs (fork, chain, collider, M-bias, bow, unobserved
   confounder) with hand-derived, textbook-known identifiability verdicts.
2. A property-based oracle: for many random DAGs, an adjustment set Z is
   valid for the backdoor criterion iff
   (a) Z contains no descendant of the treatment, and
   (b) ``networkx.is_d_separator`` says treatment and outcome are
   d-separated given Z in the DAG with edges into the treatment removed
   (the "mutilated" graph). This is Pearl's backdoor <-> d-separation
   equivalence theorem, checked with a *different* algorithm
   (networkx's moralization-based d-separation) than pathmc's own
   path-enumeration code, so it serves as an independent oracle.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pytest

from pathmc.graph import GraphInfo
from pathmc.identify import adjustment_sets, is_identifiable


def _graph_info(edges: list[tuple[str, str]], latent: set[str] | None = None) -> GraphInfo:
    """Build a bare-bones GraphInfo directly from edges, bypassing the DSL.

    Useful for oracle DAGs (e.g. M-bias, unobserved confounders) that the
    formula-based DSL cannot express directly (it only allows declaring
    *endogenous* variables as latent, not exogenous common causes).
    """
    latent = latent or set()
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    for u, v in edges:
        dag.add_node(u)
        dag.add_node(v)

    endogenous = {v for _, v in edges}
    exogenous = set(dag.nodes) - endogenous

    return GraphInfo(
        topological_order=list(nx.topological_sort(dag)),
        exogenous=exogenous,
        endogenous=endogenous,
        residual_blocks=[],
        latent=latent,
        _dag=dag,
    )


# ---------------------------------------------------------------------------
# Oracle 1: named canonical DAGs with hand-derived verdicts
# ---------------------------------------------------------------------------


class TestForkOracle:
    """Z -> X, Z -> Y, X -> Y: classic confounding. Must adjust for Z."""

    def test_only_z_is_valid(self):
        g = _graph_info([("Z", "X"), ("Z", "Y"), ("X", "Y")])
        assert adjustment_sets(g, "X", "Y") == [{"Z"}]
        assert is_identifiable(g, "X", "Y")


class TestChainOracle:
    """X -> M -> Y: no confounding, empty set suffices."""

    def test_empty_set_only(self):
        g = _graph_info([("X", "M"), ("M", "Y")])
        assert adjustment_sets(g, "X", "Y") == [set()]
        assert is_identifiable(g, "X", "Y")


class TestColliderOracle:
    """X -> C <- Y: conditioning on C would create bias; must not adjust."""

    def test_empty_set_only_collider_excluded(self):
        g = _graph_info([("X", "C"), ("Y", "C")])
        sets = adjustment_sets(g, "X", "Y")
        assert sets == [set()]


class TestMBiasOracle:
    """Classic M-bias (Greenland 2003).

    U1 -> X, U1 -> M, U2 -> M, U2 -> Y, X -> Y.

    M sits as a collider between the two independent latent causes U1, U2.
    The backdoor path X <- U1 -> M <- U2 -> Y is already blocked at the
    collider M *without* conditioning on it. The empty set is therefore a
    valid (indeed the only minimal) adjustment set; adjusting for M would
    be the classic mistake of opening a spurious path.
    """

    def _dag(self):
        return _graph_info(
            [
                ("U1", "X"),
                ("U1", "M"),
                ("U2", "M"),
                ("U2", "Y"),
                ("X", "Y"),
            ]
        )

    def test_empty_set_is_valid_and_minimal(self):
        g = self._dag()
        sets = adjustment_sets(g, "X", "Y")
        assert sets == [set()]

    def test_m_alone_is_not_a_valid_set(self):
        g = self._dag()
        sets = adjustment_sets(g, "X", "Y")
        assert not any("M" in s for s in sets)

    def test_still_identifiable(self):
        g = self._dag()
        assert is_identifiable(g, "X", "Y")


class TestBowPatternOracle:
    """X -> Y with an additional unobserved confounder U -> X, U -> Y.

    If U is latent (unobserved), there is no way to block the backdoor
    path X <- U -> Y, so the effect is *not* identifiable via the backdoor
    criterion: adjustment_sets must return an empty list.
    """

    def test_not_identifiable_when_confounder_latent(self):
        g = _graph_info(
            [("U", "X"), ("U", "Y"), ("X", "Y")],
            latent={"U"},
        )
        assert adjustment_sets(g, "X", "Y") == []
        assert not is_identifiable(g, "X", "Y")

    def test_identifiable_when_confounder_observed(self):
        g = _graph_info([("U", "X"), ("U", "Y"), ("X", "Y")])
        assert adjustment_sets(g, "X", "Y") == [{"U"}]
        assert is_identifiable(g, "X", "Y")


class TestMultipleConfoundersOracle:
    """Two confounders Z1, Z2 both needed; adjusting for only one fails."""

    def test_joint_adjustment_required(self):
        g = _graph_info(
            [
                ("Z1", "X"),
                ("Z1", "Y"),
                ("Z2", "X"),
                ("Z2", "Y"),
                ("X", "Y"),
            ]
        )
        sets = adjustment_sets(g, "X", "Y")
        assert sets == [{"Z1", "Z2"}]
        # Partial adjustment (only one confounder) must not appear.
        assert {"Z1"} not in sets
        assert {"Z2"} not in sets


class TestDescendantExclusionOracle:
    """A mediator that is also (spuriously) a candidate must be excluded.

    X -> M -> Y, and M has no other parents. M is a descendant of X, so it
    can never appear in a valid backdoor set even though conditioning on it
    would (numerically) block the only X-Y path.
    """

    def test_mediator_excluded_from_all_sets(self):
        g = _graph_info([("X", "M"), ("M", "Y")])
        sets = adjustment_sets(g, "X", "Y")
        assert all("M" not in s for s in sets)
        assert sets == [set()]


# ---------------------------------------------------------------------------
# Oracle 2: property-based cross-check against networkx d-separation
# ---------------------------------------------------------------------------


def _oracle_valid_sets(dag: nx.DiGraph, treatment: str, outcome: str) -> set[frozenset[str]]:
    """Ground-truth *all* valid (not just minimal) backdoor sets.

    Uses Pearl's theorem: Z satisfies the backdoor criterion relative to
    (treatment, outcome) iff
      (1) Z contains no descendant of treatment, and
      (2) treatment and outcome are d-separated by Z in the graph with all
          edges *out of* treatment removed (this deletes the causal paths,
          leaving only backdoor paths for the d-separation check).
    Condition (2) is checked with ``networkx.is_d_separator``, a generic
    moralization-based algorithm that shares no code with pathmc's
    path-enumeration implementation in ``pathmc.identify``.
    """
    descendants = nx.descendants(dag, treatment)
    candidates = set(dag.nodes) - {treatment, outcome} - descendants

    mutilated = dag.copy()
    mutilated.remove_edges_from(list(dag.out_edges(treatment)))

    valid: set[frozenset[str]] = set()
    for size in range(len(candidates) + 1):
        for subset in combinations(sorted(candidates), size):
            z = set(subset)
            if nx.is_d_separator(mutilated, {treatment}, {outcome}, z):
                valid.add(frozenset(z))
    return valid


def _minimal(sets: set[frozenset[str]]) -> set[frozenset[str]]:
    return {s for s in sets if not any(t < s for t in sets)}


# A handful of fixed random-ish DAGs (dense enough to have confounding,
# mediation, and collider structure) used as an exhaustive oracle sweep.
_RANDOM_DAGS: list[list[tuple[str, str]]] = [
    [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("A", "D")],
    [("A", "B"), ("B", "C"), ("A", "C"), ("D", "B"), ("D", "C")],
    [("Z1", "X"), ("Z1", "M"), ("X", "M"), ("M", "Y"), ("Z1", "Y")],
    [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "M")],
    [("A", "X"), ("B", "X"), ("A", "Y"), ("C", "Y"), ("X", "Y"), ("B", "C")],
    [("B", "X"), ("X", "A"), ("A", "Y"), ("X", "Y"), ("B", "Y"), ("B", "A")],
]


@pytest.mark.parametrize("edges", _RANDOM_DAGS, ids=range(len(_RANDOM_DAGS)))
@pytest.mark.parametrize("treatment,outcome", [("X", "Y")])
def test_adjustment_sets_matches_d_separation_oracle(edges, treatment, outcome):
    dag = nx.DiGraph()
    dag.add_edges_from(edges)
    if treatment not in dag.nodes or outcome not in dag.nodes:
        pytest.skip("DAG does not contain both X and Y")

    g = GraphInfo(
        topological_order=list(nx.topological_sort(dag)),
        exogenous={n for n in dag.nodes if dag.in_degree(n) == 0},
        endogenous={n for n in dag.nodes if dag.in_degree(n) > 0},
        residual_blocks=[],
        latent=set(),
        _dag=dag,
    )

    got = {frozenset(s) for s in adjustment_sets(g, treatment, outcome)}
    oracle_all = _oracle_valid_sets(dag, treatment, outcome)
    oracle_minimal = _minimal(oracle_all)

    assert got == oracle_minimal, (
        f"adjustment_sets mismatch for edges={edges}: "
        f"pathmc={got} oracle={oracle_minimal}"
    )

    # is_identifiable must agree with "oracle found at least one valid set".
    assert is_identifiable(g, treatment, outcome) == bool(oracle_all)


@pytest.mark.parametrize("n_nodes", [4, 5, 6])
def test_adjustment_sets_matches_oracle_on_random_dags(n_nodes):
    """Sweep many random small DAGs and cross-check every X->Y* pair."""
    checked = 0
    for trial in range(15):
        p = 0.5
        dag = nx.gnp_random_graph(n_nodes, p, seed=trial + n_nodes * 100, directed=True)
        # Orient by node index to guarantee a DAG.
        dag = nx.DiGraph((u, v) for u, v in dag.edges if u < v)
        nodes = list(dag.nodes)
        if len(nodes) < 3:
            continue
        for treatment, outcome in combinations(nodes, 2):
            if not nx.has_path(dag, treatment, outcome) and not nx.has_path(
                dag, outcome, treatment
            ):
                continue
            if outcome not in nx.descendants(dag, treatment):
                treatment, outcome = outcome, treatment
                if outcome not in nx.descendants(dag, treatment):
                    continue

            g = GraphInfo(
                topological_order=list(nx.topological_sort(dag)),
                exogenous={n for n in dag.nodes if dag.in_degree(n) == 0},
                endogenous={n for n in dag.nodes if dag.in_degree(n) > 0},
                residual_blocks=[],
                latent=set(),
                _dag=dag,
            )
            got = {frozenset(s) for s in adjustment_sets(g, treatment, outcome)}
            oracle_all = _oracle_valid_sets(dag, treatment, outcome)
            oracle_minimal = _minimal(oracle_all)
            assert got == oracle_minimal, (
                f"mismatch dag_edges={list(dag.edges)} treatment={treatment} "
                f"outcome={outcome}: pathmc={got} oracle={oracle_minimal}"
            )
            checked += 1
    assert checked > 0, "no valid treatment/outcome pairs were exercised"
