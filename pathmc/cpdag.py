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
object exposing a ``.source`` attribute), or ``networkx.DiGraph`` instances,
using a dependency-free DOT reader so the only requirement is pathmc's core
``networkx``.

A *CPDAG* (completed partially directed acyclic graph) is the canonical
representative of an equivalence class: directed edges are oriented in every
member, undirected edges (encoded in DOT as ``A -> B [dir=none]`` or, in an
undirected ``graph { ... }`` block, ``A -- B``) may point either way. The
discovery front end (:class:`pathmc.discovery.TBFPC`) emits exactly this
encoding, so its CPDAGs and the DAGs it enumerates can be compared directly.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Protocol, runtime_checkable

import networkx as nx

__all__ = ["same_markov_equivalence_class"]


@runtime_checkable
class _SupportsSource(Protocol):
    """Anything exposing a DOT ``source`` string (e.g. ``graphviz.Digraph``)."""

    @property
    def source(self) -> str: ...


# A DOT identifier: either a double-quoted string or a bare alphanumeric token.
_ID = r'(?:"([^"]+)"|([A-Za-z0-9_]+))'
_RESERVED = frozenset({"graph", "node", "edge"})


def same_markov_equivalence_class(
    graph1: str | _SupportsSource | nx.DiGraph,
    graph2: str | _SupportsSource | nx.DiGraph,
) -> bool:
    """Return ``True`` if two graphs share a Markov equivalence class.

    Each graph may be a DOT string, an object with a ``.source`` attribute
    (such as a ``graphviz.Digraph``), or a ``networkx.DiGraph``. Two graphs
    are Markov-equivalent when they have identical node sets, identical
    skeletons, and identical sets of unshielded colliders. Node identity is
    the DOT id (or ``networkx`` node); ``label`` attributes are not treated
    as identity.

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
        return _parse_dot(graph)
    source = getattr(graph, "source", None)
    if isinstance(source, str):
        return _parse_dot(source)
    raise TypeError(
        "Graph must be a DOT string, an object with a '.source' attribute "
        "(e.g. graphviz.Digraph), or a networkx.DiGraph. "
        f"Got {type(graph).__name__}. Pass one of these, or build a "
        "networkx.DiGraph from your edges first."
    )


def _strip_comments(text: str) -> str:
    """Remove ``//``, ``#`` and ``/* ... */`` comments, ignoring quoted strings.

    A ``#`` or ``//`` inside a double-quoted DOT string (e.g.
    ``fillcolor="#eef5ff"``) is *not* a comment and must survive, otherwise
    the rest of the statement — including the node it declares — is lost.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    out_lines = []
    for raw in text.splitlines():
        kept: list[str] = []
        in_quotes = False
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == '"':
                in_quotes = not in_quotes
                kept.append(ch)
            elif not in_quotes and (
                ch == "#" or (ch == "/" and raw[i + 1 : i + 2] == "/")
            ):
                break
            else:
                kept.append(ch)
            i += 1
        out_lines.append("".join(kept))
    return "\n".join(out_lines)


def _split_statements(body: str) -> list[str]:
    """Split a DOT body on ``;``/newline, ignoring separators inside quotes."""
    statements: list[str] = []
    current: list[str] = []
    in_quotes = False
    for ch in body:
        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
        elif ch in ";\n" and not in_quotes:
            statements.append("".join(current))
            current = []
        else:
            current.append(ch)
    statements.append("".join(current))
    return statements


def _clean_id(token: str) -> str | None:
    """Strip surrounding quotes/whitespace from a DOT identifier token."""
    token = token.strip()
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        return token[1:-1]
    return token or None


def _parse_dot(
    dot_text: str,
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse DOT text into ``(nodes, directed_edges, undirected_edges)``.

    Directed edges are ``(u, v)`` tuples; undirected edges are
    ``frozenset({u, v})``. An ``A -> B`` edge carrying a ``dir=none``
    attribute is treated as undirected, as is any edge in an undirected
    ``graph { ... }`` block. Chained edges (``A -> B -> C``) are expanded, and
    global style declarations (``graph [...]``, ``node [...]``, ``edge [...]``)
    are ignored. Node identity is the DOT id; ``label`` attributes are not
    used as identity.

    Raises
    ------
    ValueError
        If no ``graph``/``digraph`` block can be found.
    """
    header = re.match(r"\s*(?:strict\s+)?(graph|digraph)\b", dot_text, flags=re.I)
    brace_start = dot_text.find("{")
    brace_end = dot_text.rfind("}")
    if header is None or brace_start == -1 or brace_end <= brace_start:
        raise ValueError(
            "Could not parse DOT text. Expected a 'graph { ... }' or "
            "'digraph { ... }' block."
        )

    is_undirected_graph = header.group(1).lower() == "graph"
    body = dot_text[brace_start + 1 : brace_end]
    return _parse_dot_body(body, is_undirected_graph=is_undirected_graph)


def _parse_dot_body(
    body: str, *, is_undirected_graph: bool
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse the inside of a DOT ``{ ... }`` block.

    Shared by :func:`_parse_dot` and (for directed graphs) by
    ``pathmc.dag._parse_dag``, so all DOT consumers agree on the grammar:
    quoted ids, chained edges, attribute brackets, and quote-aware comment
    and statement handling.
    """
    body = _strip_comments(body)

    nodes: set[str] = set()
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set()

    for raw_stmt in _split_statements(body):
        stmt = raw_stmt.strip()
        if not stmt:
            continue

        attrs = ""
        bracket = stmt.find("[")
        if bracket != -1:
            attrs = stmt[bracket:]
            stmt = stmt[:bracket].strip()
        if not stmt:
            continue

        if "->" in stmt or "--" in stmt:
            parts = re.split(r"(->|--)", stmt)
            endpoints = [_clean_id(p) for p in parts[0::2]]
            operators = parts[1::2]
            if len(endpoints) < 2 or not all(endpoints):
                continue
            dir_none = re.search(r"dir\s*=\s*none", attrs, flags=re.I) is not None
            for (u, v), op in zip(zip(endpoints, endpoints[1:]), operators):
                assert u is not None and v is not None
                nodes.update((u, v))
                if u == v:
                    continue
                if is_undirected_graph or op == "--" or dir_none:
                    undirected.add(frozenset((u, v)))
                else:
                    directed.add((u, v))
            continue

        name = _clean_id(stmt)
        if name and name not in _RESERVED:
            nodes.add(name)

    return nodes, directed, undirected


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
