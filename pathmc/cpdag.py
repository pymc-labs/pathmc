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
"""Markov-equivalence utilities for CPDAGs and DAGs.

Two graphs lie in the same Markov equivalence class iff they share the same
*skeleton* (adjacencies, ignoring orientation) and the same set of
*v-structures* (unshielded colliders) — the Verma–Pearl (1990) criterion.
This module compares graphs supplied as DOT strings, graphviz objects (any
object exposing a ``.source`` attribute), or ``networkx.DiGraph`` instances.
DOT text is read by :mod:`pathmc._dot`; the only third-party requirement is
``networkx``.

A *CPDAG* (completed partially directed acyclic graph) is the canonical
representative of an equivalence class: directed edges are oriented in every
member, undirected edges (encoded in DOT as ``A -> B [dir=none]`` or, in an
undirected ``graph { ... }`` block, ``A -- B``) may point either way. The
discovery front end (:class:`pathmc.discovery.TBFPC`) emits exactly this
encoding, so its CPDAGs and the DAGs it enumerates can be compared directly.
"""

from __future__ import annotations

from itertools import combinations
from typing import Protocol, runtime_checkable

import networkx as nx

from pathmc._dot import parse_dot

__all__ = ["same_markov_equivalence_class"]


@runtime_checkable
class _SupportsSource(Protocol):
    """Anything exposing a DOT ``source`` string (e.g. ``graphviz.Digraph``)."""

    @property
    def source(self) -> str: ...


def same_markov_equivalence_class(
    graph1: str | _SupportsSource | nx.DiGraph,
    graph2: str | _SupportsSource | nx.DiGraph,
) -> bool:
    """Return ``True`` if two graphs share a Markov equivalence class.

    Each graph may be a DOT string, an object with a ``.source`` attribute
    (such as a ``graphviz.Digraph``), or a ``networkx.DiGraph``. Two graphs
    are Markov-equivalent when they have identical node sets, identical
    skeletons, and identical sets of unshielded colliders. A ``subgraph``
    or anonymous ``{ ... }`` block is layout only: the comparison uses the
    nodes and edges inside it, not the cluster id. Node identity is
    the DOT id (or ``networkx`` node); ``label`` attributes are not treated
    as identity. Attribute assignments such as ``rankdir=LR`` are not nodes.
    Unquoted ``digraph``, ``subgraph``, and ``strict`` are keywords, not node
    ids: quote them (``"digraph" -> Y``). A unicode space does not separate
    those keywords; write an ASCII space, as in ``strict digraph``.
    ``node = box`` is rejected; write ``node [shape=box]``. A quoted odd
    backslash run before a line feed is line continuation, so
    ``"foo\\<newline>bar"`` is the node ``foobar``. An even run is not
    continuation.

    Parameters
    ----------
    graph1, graph2 : str | object with ``.source`` | networkx.DiGraph
        The two graphs to compare.

    Returns
    -------
    bool
        ``True`` if the graphs are Markov-equivalent, ``False`` otherwise.

    Raises
    ------
    TypeError
        If either argument is not a DOT string, a ``.source`` object, or a
        ``networkx.DiGraph``.
    ValueError
        If a DOT string cannot be parsed.

    Examples
    --------
    The chain ``A -> B -> C`` and the fork ``A <- B -> C`` are
    Markov-equivalent (same skeleton, no v-structure); the collider
    ``A -> B <- C`` is not::

        from pathmc import same_markov_equivalence_class

        same_markov_equivalence_class(
            "digraph { A -> B; B -> C; }",
            "digraph { B -> A; B -> C; }",
        )  # True
        same_markov_equivalence_class(
            "digraph { A -> B; B -> C; }",
            "digraph { A -> B; C -> B; }",
        )  # False
    """
    nodes1, directed1, undirected1 = _coerce_to_components(graph1)
    nodes2, directed2, undirected2 = _coerce_to_components(graph2)

    if nodes1 != nodes2:
        return False

    skeleton1 = _skeleton(directed1, undirected1)
    skeleton2 = _skeleton(directed2, undirected2)
    if skeleton1 != skeleton2:
        return False

    return _v_structures(directed1, skeleton1) == _v_structures(directed2, skeleton2)


def _coerce_to_components(
    graph: str | _SupportsSource | nx.DiGraph,
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Normalize a graph into ``(nodes, directed_edges, undirected_edges)``."""
    if isinstance(graph, nx.DiGraph):
        # Drop self-loops so a DiGraph compares equal to its DOT form, which
        # also ignores them (a self-loop is not a meaningful adjacency).
        directed = {(u, v) for u, v in graph.edges if u != v}
        return set(graph.nodes), directed, set()
    if isinstance(graph, str):
        return parse_dot(graph)
    source = getattr(graph, "source", None)
    if isinstance(source, str):
        return parse_dot(source)
    raise TypeError(
        "Graph must be a DOT string, an object with a '.source' attribute "
        "(e.g. graphviz.Digraph), or a networkx.DiGraph. "
        f"Got {type(graph).__name__}. Pass one of these, or build a "
        "networkx.DiGraph from your edges first."
    )


def _skeleton(
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
) -> set[frozenset[str]]:
    """Return the undirected skeleton: all adjacencies, orientation discarded."""
    skeleton = set(undirected)
    for u, v in directed:
        skeleton.add(frozenset((u, v)))
    return skeleton


def _v_structures(
    directed: set[tuple[str, str]],
    skeleton: set[frozenset[str]],
) -> set[tuple[tuple[str, str], str]]:
    """Identify unshielded colliders ``a -> c <- b`` with ``a`` and ``b``
    non-adjacent.

    Each v-structure is returned as ``((a, b), c)`` with ``a`` and ``b``
    sorted, so the representation is independent of edge insertion order.
    """
    parents: dict[str, set[str]] = {}
    for a, c in directed:
        parents.setdefault(c, set()).add(a)

    vstructs: set[tuple[tuple[str, str], str]] = set()
    for c, pars in parents.items():
        if len(pars) < 2:
            continue
        for a, b in combinations(pars, 2):
            if frozenset((a, b)) not in skeleton:
                ordered = (a, b) if a < b else (b, a)
                vstructs.add((ordered, c))
    return vstructs
