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


def test_subgraph_wrapper_matches_flat_collider():
    """A graphviz ``subgraph`` wrapper is the same collider, not new nodes.

    Layout clusters do not change the nodes, skeleton, or unshielded colliders.
    The reader must not invent nodes such as ``subgraph cluster { X`` or ``}``
    and report the graphs as inequivalent.
    """
    flat = "digraph { X -> Y; Z -> Y; }"
    wrapped = "digraph { subgraph cluster { X -> Y; Z -> Y; } }"
    nodes, directed, undirected = _parse_dot(wrapped)
    assert nodes == {"X", "Y", "Z"}
    assert directed == {("X", "Y"), ("Z", "Y")}
    assert undirected == set()
    assert same_markov_equivalence_class(wrapped, flat) is True
    assert same_markov_equivalence_class(_DummySource(wrapped), flat) is True


def test_block_comment_does_not_glue_subgraph_keyword():
    """A ``/* */`` between ids is a separator, not a deletion.

    ``subgraph/*c*/cluster`` must stay the keyword plus a cluster id. Deleting
    the comment invents node ``subgraphcluster`` and reports the collider as
    inequivalent.
    """
    flat = "digraph { X -> Y; Z -> Y; }"
    commented = "digraph { subgraph/*c*/cluster { X -> Y; Z -> Y } }"
    nodes, directed, _ = _parse_dot(commented)
    assert "subgraphcluster" not in nodes
    assert nodes == {"X", "Y", "Z"}
    assert directed == {("X", "Y"), ("Z", "Y")}
    assert same_markov_equivalence_class(commented, flat) is True
    split = _parse_dot("digraph { A/*c*/B -> Y }")
    assert split[0] == {"A", "B", "Y"}
    assert split[1] == {("B", "Y")}
    with pytest.raises(ValueError):
        _parse_dot("digraph { sub/*c*/graph cluster { X -> Y } }")


def test_subgraph_layout_variants_do_not_invent_nodes():
    """Nearby subgraph forms keep the same nodes and edges as the flat graph."""
    flat = "digraph { X -> Y; Z -> Y; }"
    variants = [
        "digraph { { X -> Y; Z -> Y; } }",
        "digraph { subgraph { X -> Y; Z -> Y } }",
        "digraph { subgraph outer { subgraph cluster_in { X -> Y; Z -> Y } } }",
        "digraph { subgraph cluster { X -> Y } Z -> Y }",
        (
            "digraph {\n"
            "  rankdir=LR\n"
            "  subgraph cluster_c {\n"
            '    label="collider"\n'
            "    style=dashed\n"
            "    X -> Y\n"
            "    Z -> Y\n"
            "  }\n"
            "}"
        ),
        'digraph { subgraph "c 1" { "X" -> "Y"; "Z" -> "Y"; } }',
        "graph { subgraph cluster { X -- Y; Z -- Y } }",
        "digraph { subgraph cluster { X -> Y Z -> Y } }",
        "digraph { X -> Y subgraph cluster { Z -> Y } }",
        "digraph { rankdir=LR subgraph cluster { X -> Y; Z -> Y } }",
        'digraph { subgraph c { label="collider" style=dashed X -> Y; Z -> Y } }',
    ]
    for dot in variants:
        nodes, directed, undirected = _parse_dot(dot)
        assert "}" not in nodes
        assert all("subgraph" not in node for node in nodes)
        assert all("=" not in node for node in nodes)
        if dot.startswith("graph "):
            assert nodes == {"X", "Y", "Z"}
            assert directed == set()
            assert undirected == {frozenset({"X", "Y"}), frozenset({"Z", "Y"})}
            assert (
                same_markov_equivalence_class(dot, "graph { X -- Y; Z -- Y }") is True
            )
        else:
            assert nodes == {"X", "Y", "Z"}
            assert directed == {("X", "Y"), ("Z", "Y")}
            assert undirected == set()
            assert same_markov_equivalence_class(dot, flat) is True


def test_edge_into_subgraph_expands_to_member_nodes():
    """An edge into a subgraph connects to each node inside, not to the block."""
    dotted = "digraph { A -> subgraph { B; C }; }"
    nodes, directed, undirected = _parse_dot(dotted)
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "B"), ("A", "C")}
    assert undirected == set()
    assert same_markov_equivalence_class(dotted, "digraph { A -> B; A -> C; }") is True

    braced = "digraph { { A B } -> C }"
    nodes, directed, undirected = _parse_dot(braced)
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "C"), ("B", "C")}
    assert undirected == set()

    internal = "digraph { A -> subgraph { B -> C } }"
    nodes, directed, undirected = _parse_dot(internal)
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "B"), ("A", "C"), ("B", "C")}
    assert undirected == set()

    product = "digraph { { A B } -> { C D } }"
    nodes, directed, undirected = _parse_dot(product)
    assert nodes == {"A", "B", "C", "D"}
    assert directed == {("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")}
    assert undirected == set()

    undirected_into = "digraph { A -> { B C } [dir=none] }"
    nodes, directed, undirected = _parse_dot(undirected_into)
    assert nodes == {"A", "B", "C"}
    assert directed == set()
    assert undirected == {frozenset({"A", "B"}), frozenset({"A", "C"})}

    comma_nodes = "digraph { A -> subgraph cluster_c { B, C } }"
    nodes, directed, undirected = _parse_dot(comma_nodes)
    assert nodes == {"A", "B", "C"}
    assert directed == {("A", "B"), ("A", "C")}
    assert undirected == set()


def test_malformed_subgraph_raises_instead_of_inventing_nodes():
    """Unbalanced or bare ``subgraph`` syntax raises, naming that syntax."""
    unbalanced = "digraph { subgraph cluster { X -> Y; Z -> Y; }"
    with pytest.raises(ValueError, match="subgraph"):
        same_markov_equivalence_class(unbalanced, "digraph { X -> Y; Z -> Y; }")
    bare = "digraph { subgraph cluster X -> Y; }"
    with pytest.raises(ValueError, match="subgraph"):
        _parse_dot(bare)


def test_dag_parser_uses_subgraph_nodes_not_invented_ones():
    """The shared body reader must not invent nodes for ``dag`` either."""
    from pathmc.dag import BuildModelFromDAG

    graph = BuildModelFromDAG._parse_dag(
        "digraph { subgraph cluster { X -> Y; Z -> Y } }"
    )
    assert set(graph.nodes) == {"X", "Y", "Z"}
    assert set(graph.edges) == {("X", "Y"), ("Z", "Y")}


def test_comment_and_html_braces_are_not_subgraph_boundaries():
    """A ``}`` in a comment or HTML label is not unsupported subgraph syntax.

    The previous reader accepted these and kept the real edge. Treating the
    brace as a subgraph boundary must not raise or invent a node.
    """
    from pathmc.dag import BuildModelFromDAG

    commented = "digraph { X -> Y; Z -> Y; }\n// note }\n"
    nodes, directed, undirected = _parse_dot(commented)
    assert nodes == {"X", "Y", "Z"}
    assert directed == {("X", "Y"), ("Z", "Y")}
    assert undirected == set()
    assert (
        same_markov_equivalence_class(commented, "digraph { X -> Y; Z -> Y; }") is True
    )

    html = "digraph { X [label=<foo {bar}>]; X -> Y }"
    nodes, directed, undirected = _parse_dot(html)
    assert nodes == {"X", "Y"}
    assert directed == {("X", "Y")}
    assert undirected == set()
    graph = BuildModelFromDAG._parse_dag(html)
    assert set(graph.nodes) == {"X", "Y"}
    assert set(graph.edges) == {("X", "Y")}


def test_html_comment_markers_numerals_and_comma_lists_keep_edges():
    """Layout syntax must not drop the edges of an otherwise identical graph.

    ``//`` and ``#`` inside an HTML label are label text. A leading ``-`` is a
    numeral id. A comma separates a node list, so ``A -> B, C`` is the same
    edges as ``A -> { B C }``.
    """
    flat = "digraph { X -> Y; Z -> Y; }"
    html_url = "digraph {\n  X [label=<http://example.com>];\n  X -> Y;\n  Z -> Y;\n}"
    html_hash = (
        "digraph { subgraph cluster { X [label=<http://ex.com/a#b>]; X -> Y; Z -> Y } }"
    )
    for dot in (html_url, html_hash):
        nodes, directed, undirected = _parse_dot(dot)
        assert nodes == {"X", "Y", "Z"}
        assert directed == {("X", "Y"), ("Z", "Y")}
        assert undirected == set()
        assert same_markov_equivalence_class(dot, flat) is True

    weighted = "digraph { subgraph cluster { X -> Y [weight=-1]; Z -> Y } }"
    nodes, directed, undirected = _parse_dot(weighted)
    assert nodes == {"X", "Y", "Z"}
    assert directed == {("X", "Y"), ("Z", "Y")}
    assert same_markov_equivalence_class(weighted, flat) is True

    numeral = "digraph { -1 -> Y; Z -> Y [weight=-1] }"
    nodes, directed, undirected = _parse_dot(numeral)
    assert nodes == {"-1", "Y", "Z"}
    assert directed == {("-1", "Y"), ("Z", "Y")}

    plus = 'digraph { X -> Y [label="a" + "b"]; Z -> Y }'
    assert same_markov_equivalence_class(plus, flat) is True
    assert _parse_dot('"a" + "b" -> Y'.join(["digraph { ", " }"]))[0] == {"ab", "Y"}

    assert (
        same_markov_equivalence_class(
            "digraph { A -> B, C }", "digraph { A -> { B C } }"
        )
        is True
    )
    nodes, directed, undirected = _parse_dot(
        "digraph { subgraph cluster { B, C -> D } }"
    )
    assert nodes == {"B", "C", "D"}
    assert directed == {("B", "D"), ("C", "D")}
    assert undirected == set()
    nodes, directed, undirected = _parse_dot("digraph { A, B -> C [dir=none] }")
    assert directed == set()
    assert undirected == {frozenset({"A", "C"}), frozenset({"B", "C"})}

    quoted = 'digraph { "node" -> Y; Z -> Y }'
    nodes, directed, undirected = _parse_dot(quoted)
    assert nodes == {"node", "Y", "Z"}
    assert directed == {("node", "Y"), ("Z", "Y")}
    assert same_markov_equivalence_class(quoted, "digraph { Z -> Y }") is False
    assert _parse_dot('digraph { "subgraph" -> Y }')[0] == {"subgraph", "Y"}
    assert _parse_dot('digraph { "no" + "de" -> Y }')[1] == {("node", "Y")}
    assert _parse_dot("digraph { café -> Y }")[0] == {"café", "Y"}
    port_nodes, port_edges, _ = _parse_dot('digraph { "A":n -> B }')
    assert port_nodes == {"A", "B"}
    assert port_edges == {("A", "B")}

    escaped = r'digraph { A -> B [label="\""] }'
    assert _parse_dot(escaped)[1] == {("A", "B")}
    commented = 'digraph { X -> Y [label="a\\" // not comment"]; Z -> Y }'
    assert _parse_dot(commented)[1] == {("X", "Y"), ("Z", "Y")}
    assert _parse_dot("digraph { α -> β; γ -> β }")[0] == {"α", "β", "γ"}
    for ported in (
        'digraph { "A" : n -> B }',
        'digraph { "A": n -> B }',
        'digraph { "A":"n" -> B }',
    ):
        nodes, edges, _ = _parse_dot(ported)
        assert nodes == {"A", "B"}
        assert edges == {("A", "B")}

    backslash = 'digraph { "foo\\bar" -> Y }'
    assert _parse_dot(backslash)[0] == {"foo\\bar", "Y"}
    with pytest.raises(ValueError, match="-"):
        _parse_dot("digraph { my-node -> Y }")
    with pytest.raises(ValueError, match=","):
        _parse_dot("digraph { A -> B [dir=none], C }")

    unicode_id = "digraph { subgraph cluster { café_α -> Y; Z -> Y } }"
    assert _parse_dot(unicode_id)[0] == {"café_α", "Y", "Z"}
    assert (
        same_markov_equivalence_class(unicode_id, 'digraph { "café_α" -> Y; Z -> Y }')
        is True
    )
    for bad, token in (
        ("digraph { Node-2 -> Y }", "-"),
        ("digraph { node-2 -> Y }", "-"),
        ("digraph { foo(bar) -> Y }", r"\("),
        ("digraph { foo\\bar -> Y }", r"\\"),
    ):
        with pytest.raises(ValueError, match=token):
            _parse_dot(bad)

    assert _parse_dot("digraph { © -> Y; Z -> Y }")[0] == {"©", "Y", "Z"}
    fullwidth = "digraph { subgraph cluster { ３abc -> Y; Z -> Y } }"
    assert _parse_dot(fullwidth)[0] == {"３abc", "Y", "Z"}
    with pytest.raises(ValueError, match="-"):
        _parse_dot("digraph { A -> Node-2 }")


def test_unquoted_unicode_space_stays_one_id():
    """NBSP and em space are part of an unquoted id, including in a subgraph.

    ASCII space, tab, CR, and LF still separate tokens. Splitting every
    ``str.isspace()`` character invents nodes and reports a false mismatch
    against the quoted form of the same id.
    """
    for gap in ("\u00a0", "\u2003"):
        node = f"A{gap}B"
        flat = f"digraph {{ {node} -> Y; Z -> Y; }}"
        wrapped = f"digraph {{ subgraph cluster {{ {node} -> Y; Z -> Y; }} }}"
        quoted = f'digraph {{ "{node}" -> Y; Z -> Y; }}'
        assert _parse_dot(flat)[0] == {node, "Y", "Z"}
        assert _parse_dot(wrapped)[0] == {node, "Y", "Z"}
        assert _parse_dot(flat)[1] == {(node, "Y"), ("Z", "Y")}
        assert same_markov_equivalence_class(flat, quoted) is True
        assert same_markov_equivalence_class(wrapped, flat) is True

    for sep in (" ", "\t", "\r", "\n"):
        split = f"digraph {{ A{sep}B -> Y; }}"
        nodes, directed, _ = _parse_dot(split)
        assert "A" in nodes and "B" in nodes
        assert f"A{sep}B" not in nodes
        assert directed == {("B", "Y")}


def test_quoted_string_keeps_every_backslash():
    """Graphviz keeps each backslash in a quoted id; ``\\\\`` is not one slash.

    A quote ends the string only when preceded by an even number of
    backslashes. Collapsing ``\\\\`` makes ``"foo\\\\bar"`` compare equal to
    ``"foo\\bar"``, and a subgraph id ``"C:\\\\"`` is no longer ``C:\\\\``.
    """
    labeled = 'digraph { A [label="foo\\\\"]; A -> B }'
    nodes, directed, _ = _parse_dot(labeled)
    assert nodes == {"A", "B"}
    assert directed == {("A", "B")}
    wrapped = 'digraph { subgraph cluster { "C:\\\\" -> Y; Z -> Y } }'
    nodes, directed, _ = _parse_dot(wrapped)
    assert nodes == {"C:\\\\", "Y", "Z"}
    assert directed == {("C:\\\\", "Y"), ("Z", "Y")}
    assert _parse_dot('digraph { "foo\\bar" -> Y }')[0] == {"foo\\bar", "Y"}
    assert _parse_dot('digraph { "foo\\\\bar" -> Y }')[0] == {"foo\\\\bar", "Y"}
    assert _parse_dot('digraph { "foo\\"bar" -> Y }')[0] == {'foo"bar', "Y"}
    assert (
        same_markov_equivalence_class(
            'digraph { "foo\\bar" -> Y; Z -> Y }',
            'digraph { "foo\\\\bar" -> Y; Z -> Y }',
        )
        is False
    )


def test_subgraph_keeps_empty_html_and_glued_port_ids():
    """Valid subgraph spellings keep Graphviz node identity.

    ``A: n`` and ``A:n`` are both a port on ``A``. ``""`` is a node.
    An HTML statement is a node, not an error. Mixed ``"A"+B`` raises.
    """
    glued = "digraph { subgraph cluster { A: n -> Y; Z -> Y } }"
    nodes, directed, _ = _parse_dot(glued)
    assert nodes == {"A", "Y", "Z"}
    assert directed == {("A", "Y"), ("Z", "Y")}
    assert _parse_dot("digraph { A:n -> B }")[0] == {"A", "B"}
    assert _parse_dot("digraph { subgraph cluster { A:n -> Y; Z -> Y } }")[0] == {
        "A",
        "Y",
        "Z",
    }
    assert _parse_dot('digraph { subgraph cluster { "" } }')[0] == {""}
    assert (
        same_markov_equivalence_class(
            'digraph { subgraph cluster { "" } }', "digraph { }"
        )
        is False
    )
    empty_edge = 'digraph { subgraph cluster { "" -> Y } }'
    nodes, directed, _ = _parse_dot(empty_edge)
    assert "" in nodes and "Y" in nodes
    assert directed == {("", "Y")}
    html = "digraph { subgraph cluster { <!-- x --> X -> Y; Z -> Y } }"
    nodes, directed, _ = _parse_dot(html)
    assert "!-- x --" in nodes
    assert directed == {("X", "Y"), ("Z", "Y")}
    with pytest.raises(ValueError, match=r"\+"):
        _parse_dot('digraph { "A"+B -> Y; Z -> Y }')


def test_subgraph_ports_ids_and_edge_dir_default():
    """Layout ports and cluster ids must not invent a Markov class.

    A second compass piece is not a node. A concatenated or HTML subgraph
    id is not a node and must not raise. ``edge [dir=none]`` inside a
    subgraph is undirected, not a collider.
    """
    ports = "digraph { subgraph cluster { A: n: ne -> B } }"
    nodes, directed, _ = _parse_dot(ports)
    assert nodes == {"A", "B"}
    assert directed == {("A", "B")}
    assert same_markov_equivalence_class(ports, "digraph { A -> B }") is True
    assert same_markov_equivalence_class(ports, "digraph { A; ne -> B }") is False
    assert _parse_dot("digraph { A: n : ne -> B }")[0] == {"A", "B"}
    assert _parse_dot("digraph { A:n -> B }")[0] == {"A", "B"}
    assert _parse_dot('digraph { "A": n: ne -> B }')[0] == {"A", "B"}

    concat = 'digraph { subgraph "A"+"B" { X -> Y; Z -> Y } }'
    nodes, directed, _ = _parse_dot(concat)
    assert nodes == {"X", "Y", "Z"}
    assert "AB" not in nodes
    assert same_markov_equivalence_class(concat, "digraph { X -> Y; Z -> Y }") is True
    assert _parse_dot("digraph { subgraph <BR/> { X -> Y } }")[0] == {"X", "Y"}

    default = "digraph { subgraph cluster { edge [dir=none]; X -> Y; Z -> Y } }"
    nodes, directed, undirected = _parse_dot(default)
    assert nodes == {"X", "Y", "Z"}
    assert directed == set()
    assert undirected == {frozenset(("X", "Y")), frozenset(("Z", "Y"))}
    assert same_markov_equivalence_class(default, "digraph { X -> Y; Z -> Y }") is False


def test_glued_compass_port_is_not_a_node():
    """Graphviz emits ``A:n -> B`` for node ``A``. That is not node ``A:n``."""
    wrapped = "digraph { subgraph cluster { A:n -> Y; Z -> Y } }"
    assert _parse_dot(wrapped)[0] == {"A", "Y", "Z"}
    assert same_markov_equivalence_class(wrapped, "digraph { A -> Y; Z -> Y }") is True
    assert _parse_dot("digraph { A:n -> B }")[0] == {"A", "B"}
    assert _parse_dot('digraph { "A:":n -> B }')[0] == {"A:", "B"}
    nodes, directed, _ = _parse_dot('digraph { "A":"n:" B -> C }')
    assert nodes == {"A", "B", "C"}
    assert directed == {("B", "C")}
    assert _parse_dot("digraph { <A>:n -> Y }")[0] == {"A", "Y"}
    assert _parse_dot("digraph { A:<n> -> B }")[0] == {"A", "B"}
    assert _parse_dot('digraph { "A"+<B> -> Y }')[0] == {"AB", "Y"}
    assert _parse_dot("digraph { <A>+<B> -> Y }")[0] == {"AB", "Y"}
    html_dir = "digraph { subgraph cluster { X -> Y [dir=<none>]; Z -> Y } }"
    _, directed, undirected = _parse_dot(html_dir)
    assert ("X", "Y") not in directed
    assert frozenset(("X", "Y")) in undirected
    assert ("Z", "Y") in directed
    with pytest.raises(ValueError, match="keyword"):
        _parse_dot("digraph { A:node -> B }")
    with pytest.raises(ValueError, match="keyword"):
        _parse_dot("digraph { subgraph node { X -> Y; Z -> Y } }")
    assert _parse_dot('digraph { subgraph "node" { X -> Y } }')[0] == {"X", "Y"}


def test_keyword_endpoint_and_unquoted_plus_do_not_rewrite_the_graph():
    """Invalid spellings raise instead of comparing equal to a different graph."""
    for bad in (
        "digraph { node -> Y }",
        "digraph { subgraph cluster { node -> Y } }",
        "digraph { Node -> Y }",
        "digraph { subgraph cluster { EDGE -> Y } }",
        "digraph { edge -> Y }",
        "digraph { graph -> Y }",
        "digraph { Graph -> Y }",
    ):
        with pytest.raises(ValueError, match="keyword"):
            same_markov_equivalence_class(bad, "digraph { Y }")
    with pytest.raises(ValueError, match=r"\+"):
        same_markov_equivalence_class("digraph { A+B -> Y }", "digraph { AB -> Y }")
    assert _parse_dot('digraph { "A"+"B" -> Y }')[1] == {("AB", "Y")}
    with pytest.raises(ValueError, match="dir=none"):
        _parse_dot("digraph { D -> C; A -> C [dir=none] -> B }")
    split = "digraph { A -> C [dir=none]; C -> B }"
    _, directed, undirected = _parse_dot(split)
    assert ("A", "C") not in directed
    assert frozenset(("A", "C")) in undirected
    assert ("C", "B") in directed
    with pytest.raises(ValueError, match=","):
        _parse_dot("digraph { A -> { B C }, D }")
    with pytest.raises(ValueError, match=","):
        _parse_dot("digraph { { A B }, C -> D }")
    assert _parse_dot("digraph { A -> B, C }")[1] == {("A", "B"), ("A", "C")}


def test_unquoted_graph_keywords_and_node_assignment_are_rejected():
    """Graphviz rejects bare keywords and ``node = box``; do not rewrite them.

    An unquoted ``digraph``, ``subgraph``, or ``strict`` is not a node.
    Keeping it, or dropping ``node = box``, makes the graph compare equal to
    a rewritten valid graph. Quote the keyword, or write ``node [shape=box]``.
    """
    quoted = {
        "digraph": 'digraph { "digraph" -> Y }',
        "strict": 'digraph { "strict" -> Y }',
        "subgraph": 'digraph { "subgraph" -> Y }',
    }
    bare = (
        "digraph { digraph -> Y }",
        "digraph { DIGRAPH -> Y }",
        "digraph { subgraph cluster { digraph -> Y } }",
        "digraph { strict -> Y }",
        "digraph { STRICT; X -> Y }",
        "digraph { subgraph digraph { X -> Y } }",
        "digraph { subgraph strict { X -> Y } }",
        "digraph { A:digraph -> Y }",
        "digraph { A:strict -> Y }",
        "digraph { X [label=strict] }",
        "digraph { X [digraph=box] }",
        "digraph { digraph = box; X -> Y }",
        "digraph { strict = box; X -> Y }",
        "digraph { subgraph = box; X -> Y }",
    )
    for bad in bare:
        with pytest.raises(ValueError, match="keyword"):
            same_markov_equivalence_class(bad, 'digraph { "digraph" -> Y }')
    with pytest.raises(ValueError, match="Quote it"):
        same_markov_equivalence_class("digraph { subgraph -> Y }", quoted["subgraph"])
    for spelling in (
        "digraph { node = box; X -> Y }",
        "digraph { node=box; X -> Y }",
        "digraph { Node = box; X -> Y }",
        "digraph { subgraph cluster { node = box; X -> Y; Z -> Y } }",
        "graph { node = box; A -- B }",
        "digraph G [node = box] { X -> Y; Z -> Y }",
        "digraph G [NODE=box] { X -> Y }",
        "graph G [node = box] { A -- B }",
        "digraph node = box { X -> Y }",
        "digraph G node = box { X -> Y }",
        "digraph { X -> Y [node=box]; Z -> Y }",
        "digraph { X [node = box]; X -> Y }",
    ):
        with pytest.raises(ValueError, match=r"node \[shape=box\]"):
            same_markov_equivalence_class(spelling, "digraph { X -> Y }")
    from pathmc.dag import BuildModelFromDAG, dag_to_spec

    with pytest.raises(ValueError, match=r"node \[shape=box\]"):
        BuildModelFromDAG._parse_dag("digraph G [node = box] { X -> Y; Z -> Y }")
    with pytest.raises(ValueError, match=r"node \[shape=box\]"):
        dag_to_spec("digraph { X -> Y [node = box] }")
    assert _parse_dot('digraph { "digraph" -> Y }')[0] == {"digraph", "Y"}
    assert _parse_dot('digraph { "strict" -> Y }')[0] == {"strict", "Y"}
    assert _parse_dot('digraph { "subgraph" -> Y }')[0] == {"subgraph", "Y"}
    assert _parse_dot("digraph { node [shape=box]; X -> Y }")[1] == {("X", "Y")}
    assert _parse_dot('digraph { "node" = box; X -> Y }')[1] == {("X", "Y")}
    assert _parse_dot("digraph { subgraph cluster { X -> Y; Z -> Y } }")[0] == {
        "X",
        "Y",
        "Z",
    }
    assert (
        same_markov_equivalence_class("strict graph { A -- B }", "graph { A -- B }")
        is True
    )


def test_quoted_odd_backslash_before_lf_is_line_continuation():
    """An odd backslash run before LF joins the lines; an even run does not.

    Graphviz treats ``"foo\\<LF>bar"`` as node ``foobar``. Two backslashes
    before the line feed stay in the id, with the line feed.
    """
    odd = 'digraph { "foo\\\nbar" -> Y; Z -> Y }'
    wrapped = 'digraph { subgraph cluster { "foo\\\nbar" -> Y; Z -> Y } }'
    flat = "digraph { foobar -> Y; Z -> Y }"
    assert _parse_dot(odd)[0] == {"foobar", "Y", "Z"}
    assert _parse_dot(wrapped)[0] == {"foobar", "Y", "Z"}
    assert same_markov_equivalence_class(odd, flat) is True
    assert same_markov_equivalence_class(wrapped, flat) is True
    three = 'digraph { "foo\\\\\\\nbar" }'
    assert _parse_dot(three)[0] == {"foo\\\\bar"}
    even = 'digraph { "foo\\\\\nbar" -> Y }'
    nodes, directed, _ = _parse_dot(even)
    assert nodes == {"foo\\\\\nbar", "Y"}
    assert directed == {("foo\\\\\nbar", "Y")}
    assert same_markov_equivalence_class(even, "digraph { foobar -> Y }") is False


def test_unquoted_keywords_in_values_and_graph_names_are_rejected():
    """A dropped keyword value or graph name must not match a rewritten graph.

    ``label=strict`` and ``digraph strict { ... }`` are not ids. Quote the
    keyword. ``label=node`` and ``graph = box`` are a separate Graphviz
    difference and are not covered here.
    """
    flat = "digraph { X -> Y; Z -> Y }"
    for bad in (
        "digraph { label=strict; X -> Y; Z -> Y }",
        "digraph { rankdir=DIGRAPH; X -> Y }",
        "digraph { foo = SUBGRAPH; X -> Y }",
        'digraph { foo="a"+subgraph; X -> Y }',
        "digraph { subgraph cluster { label=strict; X -> Y } }",
        "graph { label=strict; A -- B }",
        "digraph strict { X -> Y; Z -> Y }",
        "digraph DIGRAPH { X -> Y }",
        "strict digraph strict { X -> Y }",
        "graph SUBGRAPH { A -- B }",
        "DiGraph Strict { X -> Y }",
        "digraph /*c*/ strict { X -> Y }",
        "digraph G [label=strict] { X -> Y }",
        "digraph 0strict { X -> Y; Z -> Y }",
        "digraph 00strict { X -> Y }",
    ):
        with pytest.raises(ValueError, match="unsupported DOT syntax"):
            same_markov_equivalence_class(bad, flat)
    with pytest.raises(ValueError, match=r'label="strict"'):
        _parse_dot("digraph { label=strict; X -> Y }")
    with pytest.raises(ValueError, match=r'digraph "strict"'):
        _parse_dot("digraph strict { X -> Y }")
    with pytest.raises(ValueError, match=r'DiGraph "Strict"'):
        _parse_dot("DiGraph Strict { X -> Y }")
    assert _parse_dot('digraph "strict" { X -> Y }')[0] == {"X", "Y"}
    assert _parse_dot('digraph { label="strict"; X -> Y }')[0] == {"X", "Y"}
    assert _parse_dot("digraph G { X -> Y }")[1] == {("X", "Y")}
    assert _parse_dot("strict digraph { A -> B }")[1] == {("A", "B")}
    assert _parse_dot("digraph { rankdir=LR; X -> Y }")[1] == {("X", "Y")}
    nbsp = "digraph foo\u00a0strict { X -> Y }"
    em = "digraph foo\u2003strict { X -> Y }"
    assert _parse_dot(nbsp)[0] == {"X", "Y"}
    assert _parse_dot(em)[0] == {"X", "Y"}
    assert _parse_dot("digraph \u00a0strict { X -> Y }")[0] == {"X", "Y"}
    assert _parse_dot("digraph strict\u00a0 { X -> Y }")[0] == {"X", "Y"}
    assert _parse_dot("digraph strict\u2003 { X -> Y }")[0] == {"X", "Y"}
    assert _parse_dot("digraph my-graph { X -> Y }")[0] == {"X", "Y"}
    from pathmc.dag import BuildModelFromDAG

    with pytest.raises(ValueError, match=r'digraph "strict"'):
        BuildModelFromDAG._parse_dag("digraph strict { X -> Y; Z -> Y }")
    with pytest.raises(ValueError, match=r'label="strict"'):
        BuildModelFromDAG._parse_dag("digraph G [label=strict] { X -> Y }")


def test_unicode_space_does_not_separate_header_keywords():
    """A non-ASCII space is not a header separator.

    Graphviz rejects ``strict<NBSP>digraph`` and ``digraph<NBSP>foo``.
    Those must raise and name an ASCII-space repair, not match the graph
    with the glued keyword removed. A unicode space inside a name still
    parses. ``digraph.foo`` stays accepted: that mismatch is out of scope.
    """
    from pathmc.dag import BuildModelFromDAG

    flat = "digraph { X -> Y; Z -> Y }"
    repairs = (
        ("strict\u00a0digraph { X -> Y; Z -> Y }", flat, "strict digraph"),
        ("strict\u2003digraph { X -> Y; Z -> Y }", flat, "strict digraph"),
        ("STRICT\u00a0DiGraph { X -> Y }", "digraph { X -> Y }", "STRICT DiGraph"),
        ("strict\u00a0graph { A -- B }", "graph { A -- B }", "strict graph"),
        ("\u00a0digraph { X -> Y }", "digraph { X -> Y }", "digraph"),
        ("digraph\u00a0foo { X -> Y }", "digraph { X -> Y }", "digraph foo"),
        ("digraph\u00a0{ X -> Y }", "digraph { X -> Y }", "digraph {"),
        ("strict \u00a0digraph { X -> Y }", "digraph { X -> Y }", "strict digraph"),
    )
    for src, rewritten, repair in repairs:
        with pytest.raises(ValueError, match=f"Write {repair!r} instead"):
            same_markov_equivalence_class(src, rewritten)
    assert _parse_dot("digraph foo\u00a0bar { X -> Y }")[0] == {"X", "Y"}
    assert _parse_dot("strict digraph { X -> Y }")[1] == {("X", "Y")}
    assert _parse_dot("strict\tdigraph { X -> Y }")[1] == {("X", "Y")}
    assert _parse_dot("strict\ndigraph { X -> Y }")[1] == {("X", "Y")}
    assert _parse_dot("digraph.foo { X -> Y }")[0] == {"X", "Y"}
    with pytest.raises(ValueError, match="Write 'digraph foo' instead"):
        BuildModelFromDAG._parse_dag("digraph\u00a0foo { X -> Y }")
    with pytest.raises(ValueError, match="Write 'digraph' instead"):
        BuildModelFromDAG._parse_dag("\u00a0digraph { X -> Y }")
    parsed = BuildModelFromDAG._parse_dag("digraph foo\u00a0bar { X -> Y }")
    assert set(parsed.nodes) == {"X", "Y"}


def test_keyword_rejections_name_the_position_repair():
    """Attribute values and glued numerals must name the position's repair."""
    from pathmc.dag import BuildModelFromDAG, dag_to_spec

    with pytest.raises(ValueError, match=r'label="strict"'):
        _parse_dot("digraph { X [label=strict] }")
    with pytest.raises(ValueError, match=r'label="strict"'):
        _parse_dot("digraph { X -> Y [label=strict] }")
    with pytest.raises(ValueError, match=r'digraph "0strict"'):
        _parse_dot("digraph 0strict { X -> Y }")
    with pytest.raises(ValueError, match=r'digraph "00strict"'):
        _parse_dot("digraph 00strict { X -> Y }")
    with pytest.raises(ValueError, match=r'subgraph "0strict"'):
        _parse_dot("digraph { subgraph 0strict { X -> Y } }")
    with pytest.raises(ValueError, match=r'digraph "0strict"'):
        BuildModelFromDAG._parse_dag("digraph 0strict { X -> Y }")
    with pytest.raises(ValueError, match=r'subgraph "0strict"'):
        BuildModelFromDAG._parse_dag("digraph { subgraph 0strict { X -> Y } }")
    with pytest.raises(ValueError, match=r'label="strict"'):
        dag_to_spec("digraph { X [label=strict]; X -> Y }")


def test_networkx_self_loop_is_markov_equivalent_to_dot_self_loop():
    """A DiGraph self-loop is ignored consistently with its DOT counterpart."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "A"), ("A", "B")])
    assert same_markov_equivalence_class(g, "digraph { A -> A; A -> B; }") is True
