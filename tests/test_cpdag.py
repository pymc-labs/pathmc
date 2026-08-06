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
"""Tests for ``pathmc.cpdag.same_markov_equivalence_class``.

Two graphs lie in the same Markov equivalence class iff they share the same
node set, the same skeleton, and the same set of unshielded colliders
(v-structures). The public entry point is exercised here through DOT strings,
``networkx.DiGraph`` instances, ``.source`` objects, and the discovery front
end (:class:`pathmc.TBFPC`).
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc import same_markov_equivalence_class
from pathmc.cpdag import same_markov_equivalence_class as cpdag_smec

# Reusable DOT fixtures ------------------------------------------------------

CHAIN = "digraph { A -> B; B -> C; }"
FORK = "digraph { B -> A; B -> C; }"
COLLIDER = "digraph { A -> B; C -> B; }"


class _DummySource:
    """Minimal stand-in for a ``graphviz.Digraph`` (exposes ``.source``)."""

    def __init__(self, source: str) -> None:
        self.source = source


# ---------------------------------------------------------------------------
# Public re-export
# ---------------------------------------------------------------------------


def test_importable_from_both_locations_is_same_object():
    """The function is exported from both ``pathmc`` and ``pathmc.cpdag``."""
    assert same_markov_equivalence_class is cpdag_smec


# ---------------------------------------------------------------------------
# Markov equivalence: skeleton + v-structures
# ---------------------------------------------------------------------------


def test_chain_and_fork_are_equivalent():
    """A -> B -> C and A <- B -> C: same skeleton, no v-structure -> True."""
    assert same_markov_equivalence_class(CHAIN, FORK) is True


def test_chain_and_collider_not_equivalent():
    """A -> B -> C vs A -> B <- C: collider adds a v-structure -> False."""
    assert same_markov_equivalence_class(CHAIN, COLLIDER) is False


def test_graph_equivalent_to_itself():
    """A graph is always Markov-equivalent to itself."""
    assert same_markov_equivalence_class(CHAIN, CHAIN) is True
    assert same_markov_equivalence_class(COLLIDER, COLLIDER) is True


def test_two_colliders_are_equivalent_regardless_of_edge_order():
    """Same unshielded collider written in two edge orders -> True."""
    one = "digraph { A -> C; B -> C; }"
    two = "digraph { B -> C; A -> C; }"
    assert same_markov_equivalence_class(one, two) is True


def test_same_skeleton_different_vstructure_not_equivalent():
    """Same skeleton {A-C, B-C} but collider vs chain differ in v-structure."""
    collider = "digraph { A -> C; B -> C; }"
    chain = "digraph { A -> C; C -> B; }"
    assert same_markov_equivalence_class(collider, chain) is False


# ---------------------------------------------------------------------------
# Unshielded vs shielded collider
# ---------------------------------------------------------------------------


def test_unshielded_collider_vs_shielded_not_equivalent():
    """A->C<-B with A,B non-adjacent is a v-structure; adding A--B shields it.

    The shielded variant additionally changes the skeleton (it gains the A-B
    adjacency), so the two graphs are not Markov-equivalent either way.
    """
    unshielded = "digraph { A -> C; B -> C; }"
    shielded = "digraph { A -> C; B -> C; A -- B; }"
    assert same_markov_equivalence_class(unshielded, shielded) is False


def test_shielded_collider_has_no_vstructure():
    """With A--B present, A->C<-B is shielded: equivalent to a chain reorient.

    Both graphs share the skeleton {A-B, A-C, B-C} and have no unshielded
    collider, so they are Markov-equivalent.
    """
    shielded_collider = "digraph { A -> C; B -> C; A -- B; }"
    triangle_chain = "digraph { A -> C; C -> B; A -- B; }"
    assert same_markov_equivalence_class(shielded_collider, triangle_chain) is True


# ---------------------------------------------------------------------------
# Undirected blocks and dir=none
# ---------------------------------------------------------------------------


def test_undirected_block_order_independent():
    """``graph { A -- B }`` and ``graph { B -- A }`` are equivalent."""
    assert same_markov_equivalence_class("graph { A -- B }", "graph { B -- A }") is True


def test_strict_graph_parses_as_single_undirected_adjacency():
    """``strict graph { A -- B }`` is one undirected edge, like ``graph``."""
    assert (
        same_markov_equivalence_class("strict graph { A -- B }", "graph { A -- B }")
        is True
    )


def test_dir_none_digraph_edge_is_undirected():
    """``A -> B [dir=none]`` is treated as an undirected adjacency."""
    assert (
        same_markov_equivalence_class(
            "digraph { A -> B [dir=none] }", "graph { A -- B }"
        )
        is True
    )


def test_styled_dir_none_is_undirected():
    """The exact encoding TBFPC.to_digraph emits for unoriented edges."""
    dashed = "digraph { A -> B [style=dashed, dir=none] }"
    assert same_markov_equivalence_class(dashed, "graph { A -- B }") is True
    assert (
        same_markov_equivalence_class(dashed, "digraph { A -> B [dir=none] }") is True
    )


def test_dir_none_breaks_a_vstructure():
    """Marking one collider arm undirected destroys the v-structure.

    ``A -> C <- B`` has a v-structure; ``A -> C [dir=none], B -> C`` does not,
    because C now has only one parent. Same skeleton, different v-structures.
    """
    collider = "digraph { A -> C; B -> C; }"
    half_undirected = "digraph { A -> C [dir=none]; B -> C; }"
    assert same_markov_equivalence_class(collider, half_undirected) is False


# ---------------------------------------------------------------------------
# networkx.DiGraph and .source inputs
# ---------------------------------------------------------------------------


def test_networkx_digraph_matches_dot_equivalent():
    """A ``networkx.DiGraph`` compares equal to its DOT equivalent."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])
    assert same_markov_equivalence_class(g, CHAIN) is True
    assert same_markov_equivalence_class(g, FORK) is True
    assert same_markov_equivalence_class(g, COLLIDER) is False


def test_networkx_digraph_vs_networkx_digraph():
    """Two ``networkx.DiGraph`` instances compare directly."""
    chain = nx.DiGraph()
    chain.add_edges_from([("A", "B"), ("B", "C")])
    fork = nx.DiGraph()
    fork.add_edges_from([("B", "A"), ("B", "C")])
    assert same_markov_equivalence_class(chain, fork) is True


def test_source_attribute_object_is_parsed():
    """An object exposing a ``.source`` DOT string is accepted."""
    obj = _DummySource(CHAIN)
    assert same_markov_equivalence_class(obj, CHAIN) is True
    assert same_markov_equivalence_class(obj, FORK) is True
    assert same_markov_equivalence_class(obj, COLLIDER) is False


def test_source_object_vs_source_object():
    """Two ``.source`` objects compare directly."""
    assert (
        same_markov_equivalence_class(_DummySource(CHAIN), _DummySource(FORK)) is True
    )


# ---------------------------------------------------------------------------
# Node-set differences
# ---------------------------------------------------------------------------


def test_different_node_sets_not_equivalent():
    """Disjoint node sets are never equivalent."""
    assert same_markov_equivalence_class("digraph { A }", "digraph { B }") is False


def test_extra_isolated_node_breaks_equivalence():
    """An extra isolated node changes the node set -> False."""
    assert (
        same_markov_equivalence_class(CHAIN, "digraph { A -> B; B -> C; D }") is False
    )


def test_empty_graphs_are_equivalent():
    """Two node-less, edge-less graphs are trivially equivalent."""
    assert same_markov_equivalence_class("digraph { }", "digraph { }") is True
    assert same_markov_equivalence_class("graph { }", "digraph { }") is True


# ---------------------------------------------------------------------------
# Parsing corner cases (exercised via the public function)
# ---------------------------------------------------------------------------


def test_quoted_identifiers_parse_like_bare():
    """Double-quoted node ids parse the same as bare tokens."""
    quoted = 'digraph { "A" -> "B"; "B" -> "C"; }'
    assert same_markov_equivalence_class(quoted, CHAIN) is True


def test_comments_are_ignored():
    """``//``, ``#`` and ``/* */`` comments are stripped before parsing."""
    commented = (
        "digraph {\n"
        "  A -> B  // first edge\n"
        "  B -> C  # second edge\n"
        "  /* trailing block comment */\n"
        "}"
    )
    assert same_markov_equivalence_class(commented, CHAIN) is True


def test_self_loop_is_ignored_but_node_recorded():
    """A self-loop ``A -> A`` adds the node but no adjacency."""
    looped = "digraph { A -> A; A -> B; B -> C; }"
    # CHAIN has nodes {A, B, C}; the self-loop graph also has {A, B, C} with
    # the same skeleton (the loop contributes no edge), so they are equivalent.
    assert same_markov_equivalence_class(looped, CHAIN) is True


def test_global_style_declarations_ignored():
    """``graph``/``node``/``edge`` attribute statements are not nodes."""
    styled = (
        "digraph {\n"
        "  node [shape=ellipse];\n"
        "  edge [color=black];\n"
        "  A -> B;\n"
        "  B -> C;\n"
        "}"
    )
    assert same_markov_equivalence_class(styled, CHAIN) is True


def test_filled_target_node_attribute_statement_ignored():
    """A node-attribute statement registers the node without an edge.

    Mirrors TBFPC.to_digraph, which emits a styled, edgeless target node.
    """
    with_attr = 'digraph { A -> B; B -> C; "C" [style=filled]; }'
    assert same_markov_equivalence_class(with_attr, CHAIN) is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_non_graph_input_raises_type_error():
    """Passing a non-graph (int) raises ``TypeError``."""
    with pytest.raises(TypeError, match="DOT string"):
        same_markov_equivalence_class(123, CHAIN)


def test_non_graph_second_argument_raises_type_error():
    """The second argument is validated too."""
    with pytest.raises(TypeError):
        same_markov_equivalence_class(CHAIN, [1, 2, 3])


def test_object_with_non_string_source_raises_type_error():
    """A ``.source`` that is not a string is rejected."""
    with pytest.raises(TypeError):
        same_markov_equivalence_class(_DummySource(object()), CHAIN)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", "not dot", "   ", "{ A -> B }"])
def test_unparseable_dot_raises_value_error(bad):
    """Empty / malformed DOT strings raise ``ValueError``."""
    with pytest.raises(ValueError, match="parse DOT"):
        same_markov_equivalence_class(bad, CHAIN)


def test_unparseable_dot_in_second_argument_raises_value_error():
    """The second argument's DOT is parsed and validated as well."""
    with pytest.raises(ValueError, match="parse DOT"):
        same_markov_equivalence_class(CHAIN, "not dot")


# ---------------------------------------------------------------------------
# Integration with the TBFPC discovery front end
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_tbfpc():
    """Fit a TBFPC on a tiny standardized synthetic chain A -> B -> Y."""
    rng = np.random.default_rng(7)
    n = 800
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(scale=0.4, size=n)
    y = 0.9 * b + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"A": a, "B": b, "Y": y})
    df = (df - df.mean()) / df.std()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = pathmc.TBFPC(target="Y", target_edge_rule="fullS")
        model.fit(df, drivers=["A", "B"])
    return model


def test_tbfpc_construction_emits_experimental_warning():
    """Constructing a TBFPC warns that the API is experimental."""
    with pytest.warns(UserWarning, match="experimental"):
        pathmc.TBFPC(target="Y")


def test_tbfpc_cpdag_round_trips_with_itself(fitted_tbfpc):
    """The CPDAG DOT emitted by to_digraph is equivalent to itself."""
    dot = fitted_tbfpc.to_digraph()
    assert same_markov_equivalence_class(dot, dot) is True


def test_tbfpc_each_enumerated_dag_round_trips(fitted_tbfpc):
    """Every DAG enumerated from the CPDAG is equivalent to itself."""
    cdags = fitted_tbfpc.get_all_cdags_from_cpdag()
    assert len(cdags) >= 1
    for dag in cdags:
        assert same_markov_equivalence_class(dag, dag) is True


def test_tbfpc_enumerated_dags_lie_in_cpdag_class(fitted_tbfpc):
    """Each enumerated DAG is Markov-equivalent to the source CPDAG.

    Acyclic orientations of a CPDAG share its skeleton and v-structures, so
    every enumerated DAG must lie in the CPDAG's equivalence class.
    """
    cpdag_dot = fitted_tbfpc.to_digraph()
    cdags = fitted_tbfpc.get_all_cdags_from_cpdag()
    for dag in cdags:
        assert same_markov_equivalence_class(dag, cpdag_dot) is True


# ---------------------------------------------------------------------------
# Regression tests for just-fixed DOT parser behaviors
# ---------------------------------------------------------------------------

from pathmc.cpdag import _parse_dot  # noqa: E402


def test_chained_edges_expand_to_directed_segments():
    """``A -> B -> C`` expands to two directed edges over nodes {A, B, C}."""
    nodes, directed, undirected = _parse_dot("digraph { A -> B -> C; }")
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "B"), ("B", "C")}
    assert undirected == set()
    assert (
        same_markov_equivalence_class(
            "digraph { A -> B -> C; }", "digraph { A -> B; B -> C; }"
        )
        is True
    )


def test_chained_edge_dir_none_makes_all_segments_undirected():
    """``A -> B -> C [dir=none]`` makes BOTH chained segments undirected."""
    nodes, directed, undirected = _parse_dot("digraph { A -> B -> C [dir=none]; }")
    assert nodes == {"A", "B", "C"}
    assert directed == set()
    assert undirected == {frozenset({"A", "B"}), frozenset({"B", "C"})}


def test_undirected_chain_in_graph_block_expands_to_undirected_segments():
    """``graph { A -- B -- C }`` yields two undirected adjacencies."""
    nodes, directed, undirected = _parse_dot("graph { A -- B -- C }")
    assert nodes == {"A", "B", "C"}
    assert directed == set()
    assert undirected == {frozenset({"A", "B"}), frozenset({"B", "C"})}


def test_hash_inside_quoted_attribute_survives_comment_stripping():
    """A ``#`` inside a quoted attribute (e.g. ``fillcolor="#eef5ff"``) is kept.

    Mirrors the TBFPC.to_digraph target-node format; stripping it as a comment
    would drop the node it declares.
    """
    dot = (
        'digraph G { node [shape=ellipse]; "A"; "B"; '
        '"Y" [style=filled, fillcolor="#eef5ff"]; }'
    )
    assert "Y" in _parse_dot(dot)[0]
    assert same_markov_equivalence_class(dot, "digraph { A; B; }") is False


def test_semicolon_inside_quoted_attribute_does_not_split_statement():
    """A ``;`` inside a quoted label must not split the statement."""
    dot = 'digraph { A -> B [label="x; y"]; B -> C; }'
    nodes, directed, undirected = _parse_dot(dot)
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "B"), ("B", "C")}
    assert undirected == set()
    assert same_markov_equivalence_class(dot, "digraph { A -> B; B -> C; }") is True


def test_networkx_self_loop_is_markov_equivalent_to_dot_self_loop():
    """A DiGraph self-loop is ignored consistently with its DOT counterpart."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "A"), ("A", "B")])
    assert same_markov_equivalence_class(g, "digraph { A -> A; A -> B; }") is True
