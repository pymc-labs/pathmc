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
"""Tests for pathmc.discovery (TBFPC causal-discovery front end)."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc import TBFPC
from pathmc import TestResult as _TestResult  # aliased: avoid pytest collection

TEST_RESULT_KEYS = {
    "bic0",
    "bic1",
    "delta_bic",
    "logBF10",
    "BF10",
    "independent",
    "conditioning_set",
}


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _make(**kwargs) -> TBFPC:
    """Construct a TBFPC while suppressing the experimental UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBFPC(**kwargs)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Fixed standardized synthetic dataset matching the docstring example.

    A <- C, D <- C, B <- A, Y <- B + D + C (with target Y).
    """
    rng = np.random.default_rng(7)
    n = 2000
    C = rng.gamma(2, 1, n)
    A = 0.7 * C + rng.gamma(2, 1, n)
    D = 0.5 * C + rng.gamma(2, 1, n)
    B = 0.8 * A + rng.gamma(2, 1, n)
    Y = 0.9 * B + 0.6 * D + 0.7 * C + rng.gamma(2, 1, n)
    df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "Y": Y})
    return (df - df.mean()) / df.std()


DRIVERS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# __init__ warning
# ---------------------------------------------------------------------------


def test_init_emits_experimental_warning():
    with pytest.warns(UserWarning, match="experimental"):
        TBFPC(target="Y")


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["bogus", "anything", "Any", ""])
def test_invalid_target_edge_rule_raises(rule):
    with pytest.raises(ValueError, match="target_edge_rule"):
        _make(target="Y", target_edge_rule=rule)


@pytest.mark.parametrize("bad_target", ["", 123, None, 1.5, ("Y",)])
def test_invalid_target_raises(bad_target):
    with pytest.raises(ValueError, match="target"):
        _make(target=bad_target)


@pytest.mark.parametrize("bad_thresh", [0, 0.0, -1.0, -5])
def test_non_positive_bf_thresh_raises(bad_thresh):
    with pytest.raises(ValueError, match="bf_thresh"):
        _make(target="Y", bf_thresh=bad_thresh)


@pytest.mark.parametrize(
    "required",
    [[("A", "C")], [("C", "A")]],
)
def test_required_conflicting_with_forbidden_raises(required):
    with pytest.raises(ValueError, match="conflict"):
        _make(
            target="Y",
            forbidden_edges=[("A", "C")],
            required_edges=required,
        )


@pytest.mark.parametrize("arg_name", ["forbidden_edges", "required_edges"])
def test_malformed_edges_raise(arg_name):
    # A non-pair entry should be rejected by _coerce_edges.
    with pytest.raises(ValueError):
        _make(target="Y", **{arg_name: [("A",)]})


# ---------------------------------------------------------------------------
# Parameter sweep over the public surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["any", "conservative", "fullS"])
@pytest.mark.parametrize("bf_thresh", [1.0, 3.0])
@pytest.mark.parametrize(
    "forbidden",
    [None, [("A", "C")], [("A", "B")]],
)
def test_param_sweep_public_surface(synthetic_df, rule, bf_thresh, forbidden):
    model = _make(
        target="Y",
        target_edge_rule=rule,
        bf_thresh=bf_thresh,
        forbidden_edges=forbidden,
    )
    returned = model.fit(synthetic_df, drivers=DRIVERS)
    assert returned is model

    # summary()
    summary = model.summary()
    assert isinstance(summary, str)
    assert "=== Directed edges ===" in summary
    assert "=== Undirected edges ===" in summary
    assert "=== Number of CI tests run ===" in summary

    # to_digraph()
    dot = model.to_digraph()
    assert isinstance(dot, str)
    assert dot.startswith("digraph G {")

    # directed / undirected edges are sorted lists of 2-tuples of str
    for getter in (model.get_directed_edges, model.get_undirected_edges):
        edges = getter()
        assert isinstance(edges, list)
        for e in edges:
            assert isinstance(e, tuple)
            assert len(e) == 2
            assert all(isinstance(x, str) for x in e)
        assert edges == sorted(edges)

    # forbidden pairs never appear in either direction
    for fb in forbidden or []:
        u, v = fb
        directed = model.get_directed_edges()
        undirected = model.get_undirected_edges()
        assert (u, v) not in directed
        assert (v, u) not in directed
        assert tuple(sorted((u, v))) not in undirected


# ---------------------------------------------------------------------------
# fit input validation
# ---------------------------------------------------------------------------


def test_fit_missing_driver_column_raises(synthetic_df):
    model = _make(target="Y")
    with pytest.raises(KeyError):
        model.fit(synthetic_df, drivers=["A", "B", "NOPE"])


def test_required_edge_unknown_node_raises_on_fit(synthetic_df):
    model = _make(target="Y", required_edges=[("A", "ZZZ")])
    with pytest.raises(ValueError, match="unknown nodes"):
        model.fit(synthetic_df, drivers=DRIVERS)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def test_key_is_sorted():
    model = _make(target="Y")
    assert model._key("B", "A") == ("A", "B")
    assert model._key("A", "B") == ("A", "B")


def test_set_sep_records_set():
    model = _make(target="Y")
    model._set_sep("A", "B", ["C"])
    assert model.sep_sets[("A", "B")] == {"C"}
    # keyed by sorted pair regardless of call order
    model._set_sep("B", "A", ["D"])
    assert model.sep_sets[("A", "B")] == {"D"}


def test_has_forbidden_both_directions():
    model = _make(target="Y", forbidden_edges=[("A", "C")])
    assert model._has_forbidden("A", "C") is True
    assert model._has_forbidden("C", "A") is True
    assert model._has_forbidden("A", "B") is False


def test_is_required_directional():
    model = _make(target="Y", required_edges=[("B", "Y")])
    assert model._is_required("B", "Y") is True
    assert model._is_required("Y", "B") is False


# ---------------------------------------------------------------------------
# Forbidden edge behaviour
# ---------------------------------------------------------------------------


def test_forbidden_edge_ci_independent_and_absent(synthetic_df):
    model = _make(target="Y", forbidden_edges=[("A", "Y")])
    # _ci_independent short-circuits to True for forbidden pairs
    assert model._ci_independent(synthetic_df, "A", "Y", []) is True

    model.fit(synthetic_df, drivers=DRIVERS)
    directed = model.get_directed_edges()
    undirected = model.get_undirected_edges()
    assert ("A", "Y") not in directed
    assert ("Y", "A") not in directed
    assert ("A", "Y") not in undirected
    assert tuple(sorted(("A", "Y"))) not in undirected


# ---------------------------------------------------------------------------
# Required edge behaviour
# ---------------------------------------------------------------------------


def test_required_edge_behaviour(synthetic_df):
    model = _make(target="Y", required_edges=[("B", "Y")])
    model.fit(synthetic_df, drivers=DRIVERS)

    assert ("B", "Y") in model.get_directed_edges()

    dot = model.to_digraph()
    assert "color=darkgreen" in dot

    summary = model.summary()
    assert "B -> Y [required]" in summary

    results = model.get_test_results("B", "Y")
    assert any(r.get("forced") is True for r in results)


# ---------------------------------------------------------------------------
# get_all_cdags_from_cpdag
# ---------------------------------------------------------------------------


def test_cdags_single_dag_when_no_undirected():
    # A fully directed CPDAG (no dashed edges) -> exactly one DAG.
    model = _make(target="Y", required_edges=[("A", "Y")])
    # Build a model state with only a directed edge and no undirected edges.
    model.nodes_ = ["A", "Y"]
    model._adj_directed = {("A", "Y")}
    model._adj_undirected = set()
    cdags = model.get_all_cdags_from_cpdag()
    assert isinstance(cdags, list)
    assert len(cdags) == 1
    assert cdags[0].startswith("digraph G {")
    assert "dashed" not in cdags[0]


def test_cdags_from_dot_with_undirected_edge():
    # CPDAG DOT with a single dashed/dir=none undirected edge A--B.
    dot = (
        "digraph G {\n"
        "  node [shape=ellipse];\n"
        '  "A";\n'
        '  "B";\n'
        '  "Y" [style=filled, fillcolor="#eef5ff"];\n'
        '  "A" -> "Y";\n'
        '  "A" -> "B" [style=dashed, dir=none];\n'
        "}"
    )
    model = _make(target="Y")
    cdags = model.get_all_cdags_from_cpdag(dot_cpdag=dot)
    assert isinstance(cdags, list)
    # A--B has two acyclic orientations; both are valid.
    assert len(cdags) == 2
    for c in cdags:
        assert c.startswith("digraph")
        assert "dashed" not in c


def test_cdags_multiple_orientations_on_fitted_model(synthetic_df):
    model = _make(target="Y", target_edge_rule="fullS")
    model.fit(synthetic_df, drivers=DRIVERS)
    assert len(model.get_undirected_edges()) > 0
    cdags = model.get_all_cdags_from_cpdag()
    assert len(cdags) > 1
    for c in cdags:
        assert c.startswith("digraph")
        assert "dashed" not in c


def test_cdags_invalid_dot_raises():
    model = _make(target="Y")
    with pytest.raises(ValueError, match="digraph"):
        model.get_all_cdags_from_cpdag(dot_cpdag="not a graph at all")


# ---------------------------------------------------------------------------
# get_test_results
# ---------------------------------------------------------------------------


def test_get_test_results_order_invariant_and_shape(synthetic_df):
    model = _make(target="Y", target_edge_rule="fullS")
    model.fit(synthetic_df, drivers=DRIVERS)

    forward = model.get_test_results("A", "Y")
    backward = model.get_test_results("Y", "A")
    assert forward == backward
    assert len(forward) > 0

    for entry in forward:
        assert TEST_RESULT_KEYS <= set(entry.keys())


# ---------------------------------------------------------------------------
# Determinism / recovery on the fixed synthetic dataset
# ---------------------------------------------------------------------------


def test_recovery_fullS_pins_directed_edges(synthetic_df):
    model = _make(target="Y", target_edge_rule="fullS")
    model.fit(synthetic_df, drivers=DRIVERS)
    # Observed empirically and pinned to lock in behaviour.
    assert model.get_directed_edges() == [("B", "Y"), ("C", "Y"), ("D", "Y")]
    assert model.get_undirected_edges() == [
        ("A", "B"),
        ("A", "C"),
        ("C", "D"),
    ]


def test_fitting_twice_is_deterministic(synthetic_df):
    model = _make(target="Y", target_edge_rule="fullS")
    model.fit(synthetic_df, drivers=DRIVERS)
    first_directed = model.get_directed_edges()
    first_undirected = model.get_undirected_edges()

    model.fit(synthetic_df, drivers=DRIVERS)
    assert model.get_directed_edges() == first_directed
    assert model.get_undirected_edges() == first_undirected


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


def test_public_api_exports():
    assert pathmc.TBFPC is TBFPC
    assert pathmc.TestResult is _TestResult


# ---------------------------------------------------------------------------
# Pinned-edge recovery for the remaining target_edge_rules (regression locks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "expected_directed"),
    [
        ("any", [("B", "Y"), ("C", "Y"), ("D", "Y")]),
        ("conservative", [("A", "Y"), ("B", "Y"), ("C", "Y"), ("D", "Y")]),
    ],
)
def test_recovery_pins_directed_edges_other_rules(
    synthetic_df, rule, expected_directed
):
    model = _make(target="Y", target_edge_rule=rule)
    model.fit(synthetic_df, drivers=DRIVERS)
    # Observed empirically and pinned to lock in behaviour for all three rules.
    assert model.get_directed_edges() == expected_directed
    assert model.get_undirected_edges() == [
        ("A", "B"),
        ("A", "C"),
        ("C", "D"),
    ]


# ---------------------------------------------------------------------------
# Cyclic-orientation exclusion in get_all_cdags_from_cpdag
# ---------------------------------------------------------------------------


def _cyclic_cpdag_dot(*, semicolons: bool) -> str:
    """Return a CPDAG DOT with A->B, B->C and an undirected A--C edge.

    Orienting the undirected A--C edge as C->A would close the cycle
    A -> B -> C -> A, so only the A->C orientation is acyclic.
    """
    term = ";" if semicolons else ""
    return (
        "digraph G {\n"
        f"  node [shape=ellipse]{term}\n"
        f'  "A"{term}\n'
        f'  "B"{term}\n'
        f'  "C"{term}\n'
        f'  "A" -> "B"{term}\n'
        f'  "B" -> "C"{term}\n'
        f'  "A" -> "C" [style=dashed, dir=none]{term}\n'
        "}"
    )


def test_cdags_excludes_cyclic_orientation():
    dot = _cyclic_cpdag_dot(semicolons=True)
    model = _make(target="Y")
    cdags = model.get_all_cdags_from_cpdag(dot_cpdag=dot)

    # Only the A->C orientation is acyclic (C->A would close A->B->C->A).
    assert len(cdags) == 1
    surviving = cdags[0]
    assert '"A" -> "C"' in surviving
    assert '"C" -> "A"' not in surviving

    # Every returned DAG must be acyclic (verified two independent ways).
    from pathmc.cpdag import _parse_dot

    for c in cdags:
        nodes, directed, undirected = _parse_dot(c)
        assert not undirected  # fully oriented, no dashed edges remain
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(directed)
        assert nx.is_directed_acyclic_graph(graph)
        assert TBFPC._is_acyclic(nodes, directed)


# ---------------------------------------------------------------------------
# Semicolon-less external dot_cpdag parses identically
# ---------------------------------------------------------------------------


def test_cdags_semicolonless_dot_matches_semicolon_version():
    model = _make(target="Y")
    with_semis = model.get_all_cdags_from_cpdag(
        dot_cpdag=_cyclic_cpdag_dot(semicolons=True)
    )
    without_semis = model.get_all_cdags_from_cpdag(
        dot_cpdag=_cyclic_cpdag_dot(semicolons=False)
    )
    assert len(without_semis) == len(with_semis)


# ---------------------------------------------------------------------------
# Log-space independence decision / inf-BF10 contract
# ---------------------------------------------------------------------------


def test_ci_independent_log_space_handles_inf_bf10():
    # A near-deterministic dependent pair: BF10 overflows float64 but the
    # log-space decision must still resolve to "dependent".
    rng = np.random.default_rng(123)
    n = 2000
    x = rng.normal(size=n)
    y = 5.0 * x + 1e-6 * rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y})
    df = (df - df.mean()) / df.std()

    model = _make(target="Y")
    assert model._ci_independent(df, "X", "Y", []) is False

    result = model.test_results[("X", "Y", frozenset())]
    assert np.isinf(result["BF10"])
    assert np.isfinite(result["logBF10"])
    assert result["independent"] is False
