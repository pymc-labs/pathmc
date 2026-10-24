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
    skeletons, and identical sets of unshielded colliders. A ``subgraph``
    or anonymous ``{ ... }`` block is layout only: the comparison uses the
    nodes and edges inside it, not the cluster id. Node identity is
    the DOT id (or ``networkx`` node); ``label`` attributes are not treated
    as identity. Attribute assignments such as ``rankdir=LR`` are not nodes.

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
    """Split a DOT body on ``;``/newline, ignoring separators inside quotes
    or nested ``{ ... }`` blocks.
    """
    statements: list[str] = []
    current: list[str] = []
    in_quotes = False
    depth = 0
    for ch in body:
        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
            continue
        if not in_quotes and ch == "{":
            depth += 1
            current.append(ch)
            continue
        if not in_quotes and ch == "}":
            depth -= 1
            current.append(ch)
            continue
        if ch in ";\n" and not in_quotes and depth == 0:
            statements.append("".join(current))
            current = []
            continue
        current.append(ch)
    statements.append("".join(current))
    return statements


def _separate_closed_blocks(body: str) -> str:
    """Insert ``;`` after a top-level ``}`` that ends a statement.

    DOT makes ``;`` optional, so ``subgraph { A } B -> C`` is two statements.
    ``subgraph { A } -> C`` is one edge, and ``A -> { B } [dir=none]`` keeps
    its attribute list. Neither may be split.
    """
    out: list[str] = []
    in_quotes = False
    depth = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == '"':
            in_quotes = not in_quotes
            out.append(ch)
            i += 1
            continue
        if not in_quotes and ch == "{":
            depth += 1
            out.append(ch)
            i += 1
            continue
        if not in_quotes and ch == "}":
            depth -= 1
            out.append(ch)
            if depth == 0:
                j = i + 1
                while j < len(body) and body[j] in " \t":
                    j += 1
                nxt = body[j : j + 2]
                follows = j < len(body) and body[j] not in ";\n}["
                if follows and nxt not in {"->", "--"}:
                    out.append(";")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _dot_parse_error(detail: str) -> ValueError:
    return ValueError(f"Could not parse DOT text: {detail}")


def _has_subgraph_syntax(text: str) -> bool:
    """Return True if *text* has an unquoted subgraph keyword or brace."""
    in_quotes = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes:
            if ch in "{}":
                return True
            if re.match(r"subgraph\b", text[i:], flags=re.I) and (
                i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            ):
                return True
        i += 1
    return False


def _brace_span(text: str, open_index: int) -> tuple[str, int]:
    """Return ``(body, index_after_close)`` for the ``{`` at *open_index*."""
    if open_index >= len(text) or text[open_index] != "{":
        raise _dot_parse_error("unsupported subgraph syntax (missing '{').")
    depth = 0
    in_quotes = False
    for i in range(open_index, len(text)):
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            continue
        if in_quotes:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : i], i + 1
    raise _dot_parse_error(
        f"unsupported subgraph syntax (unbalanced '{{'). Offending text: {text!r}."
    )


_ASSIGNMENT_RE = re.compile(r'^(?:"[^"]*"|[A-Za-z_][A-Za-z0-9_]*)\s*=')
_SUBGRAPH_ID_RE = re.compile(r'(?:"[^"]*"|[A-Za-z0-9_]+)')


def _match_subgraph(stmt: str) -> str | None:
    """Return the body if *stmt* is exactly one subgraph, else None.

    Raises
    ------
    ValueError
        If *stmt* opens a subgraph whose braces do not balance.
    """
    s = stmt.strip()
    keyword = re.match(r"subgraph\b", s, flags=re.I)
    i = keyword.end() if keyword else 0
    rest = s[i:].lstrip()
    if keyword:
        ident = _SUBGRAPH_ID_RE.match(rest)
        if ident:
            i = len(s) - len(rest) + ident.end()
            rest = s[i:].lstrip()
    if not rest.startswith("{"):
        return None
    if not keyword and s[: len(s) - len(rest)].strip():
        return None
    body, end = _brace_span(s, len(s) - len(rest))
    if s[end:].strip():
        return None
    return body


def _split_top_level_attrs(stmt: str) -> tuple[str, str]:
    """Remove ``[...]`` groups that are outside quotes and braces."""
    kept: list[str] = []
    attrs: list[str] = []
    in_quotes = False
    brace = 0
    bracket = 0
    for ch in stmt:
        if ch == '"':
            in_quotes = not in_quotes
        if not in_quotes:
            if ch == "{":
                brace += 1
            elif ch == "}" and brace:
                brace -= 1
            elif ch == "[" and brace == 0:
                bracket += 1
            elif ch == "]" and bracket:
                attrs.append(ch)
                bracket -= 1
                continue
        if bracket:
            attrs.append(ch)
        else:
            kept.append(ch)
    return "".join(kept).strip(), "".join(attrs)


def _has_edge_op(text: str) -> bool:
    """Return True if *text* has ``->`` or ``--`` outside quotes and braces."""
    in_quotes = False
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0 and text[i : i + 2] in {"->", "--"}:
                return True
        i += 1
    return False


def _split_edge_chain(stmt: str) -> tuple[list[str], list[str]] | None:
    """Split an edge chain on top-level ``->`` / ``--`` operators."""
    if not _has_edge_op(stmt):
        return None
    endpoints: list[str] = []
    operators: list[str] = []
    current: list[str] = []
    in_quotes = False
    depth = 0
    i = 0
    while i < len(stmt):
        ch = stmt[i]
        if ch == '"':
            in_quotes = not in_quotes
            current.append(ch)
            i += 1
            continue
        if not in_quotes:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif depth == 0 and stmt[i : i + 2] in {"->", "--"}:
                endpoints.append("".join(current))
                operators.append(stmt[i : i + 2])
                current = []
                i += 2
                continue
        current.append(ch)
        i += 1
    endpoints.append("".join(current))
    if depth != 0 or not operators:
        return None
    return endpoints, operators


def _plain_node_ids(stmt: str) -> list[str] | None:
    """Return ids if *stmt* is only whitespace-separated DOT ids."""
    ids: list[str] = []
    i = 0
    s = stmt.strip()
    while i < len(s):
        if s[i].isspace():
            i += 1
            continue
        if s[i] == '"':
            end = s.find('"', i + 1)
            if end == -1:
                return None
            ids.append(s[i + 1 : end])
            i = end + 1
            continue
        matched = re.match(r"[A-Za-z0-9_]+", s[i:])
        if not matched:
            return None
        ids.append(matched.group(0))
        i += matched.end()
    return ids or None


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
    ``graph { ... }`` block. Chained edges (``A -> B -> C``) are expanded.
    Global style declarations (``graph [...]``, ``node [...]``, ``edge [...]``,
    and ``ID = value`` assignments such as ``rankdir=LR``) are ignored.
    A ``subgraph`` block, including an anonymous ``{ ... }`` cluster, is layout
    only: its nodes and edges are parsed, and the cluster id is not a node.
    An edge into a subgraph connects to each node in that subgraph. Node
    identity is the DOT id; ``label`` attributes are not used as identity.

    Raises
    ------
    ValueError
        If no ``graph``/``digraph`` block can be found, or a ``subgraph``
        block is not valid DOT.
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
    body: str, *, is_undirected_graph: bool, in_subgraph: bool = False
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse the inside of a DOT ``{ ... }`` block.

    Shared by :func:`_parse_dot` and (for directed graphs) by
    ``pathmc.dag._parse_dag``, so all DOT consumers agree on the grammar:
    quoted ids, chained edges, attribute brackets, quote-aware comment and
    statement handling, and ``subgraph`` blocks. A subgraph does not add a
    node for its id; edges and node statements inside it are real.
    """
    body = _separate_closed_blocks(_strip_comments(body))

    nodes: set[str] = set()
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set()

    for raw_stmt in _split_statements(body):
        _consume_statement(
            raw_stmt,
            nodes,
            directed,
            undirected,
            is_undirected_graph=is_undirected_graph,
            in_subgraph=in_subgraph,
        )
    return nodes, directed, undirected


def _consume_statement(
    raw_stmt: str,
    nodes: set[str],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    *,
    is_undirected_graph: bool,
    in_subgraph: bool,
) -> None:
    stmt = raw_stmt.strip()
    if not stmt:
        return
    subgraph_body = _match_subgraph(stmt)
    if subgraph_body is not None:
        sub_nodes, sub_dir, sub_und = _parse_dot_body(
            subgraph_body,
            is_undirected_graph=is_undirected_graph,
            in_subgraph=True,
        )
        nodes.update(sub_nodes)
        directed.update(sub_dir)
        undirected.update(sub_und)
        return
    if _has_subgraph_syntax(stmt) and not _has_edge_op(stmt):
        raise _dot_parse_error(
            "unsupported subgraph syntax. A subgraph block must be "
            "'subgraph [id] { ... }' (the keyword and id are optional). "
            f"Offending statement: {stmt!r}."
        )

    stmt, attrs = _split_top_level_attrs(stmt)
    if not stmt:
        return
    if _ASSIGNMENT_RE.match(stmt):
        return

    chain = _split_edge_chain(stmt)
    if chain is not None:
        _consume_edge(
            chain,
            attrs,
            nodes,
            directed,
            undirected,
            is_undirected_graph=is_undirected_graph,
        )
        return
    if _has_subgraph_syntax(stmt):
        raise _dot_parse_error(
            "unsupported subgraph syntax. A subgraph block must be "
            "'subgraph [id] { ... }' (the keyword and id are optional). "
            f"Offending statement: {stmt!r}."
        )

    if in_subgraph:
        plain = _plain_node_ids(stmt)
        if plain is not None:
            nodes.update(name for name in plain if name not in _RESERVED)
            return
    name = _clean_id(stmt)
    if name and name not in _RESERVED:
        nodes.add(name)


def _consume_edge(
    chain: tuple[list[str], list[str]],
    attrs: str,
    nodes: set[str],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    *,
    is_undirected_graph: bool,
) -> None:
    endpoints, operators = chain
    incomplete = len(endpoints) != len(operators) + 1 or not all(
        part.strip() for part in endpoints
    )
    if incomplete:
        if any(_has_subgraph_syntax(part) for part in endpoints):
            raise _dot_parse_error(
                "unsupported subgraph syntax in an edge. "
                f"Offending edge: {endpoints!r}."
            )
        return
    resolved = [
        _resolve_endpoint(
            endpoint,
            nodes,
            directed,
            undirected,
            is_undirected_graph=is_undirected_graph,
        )
        for endpoint in endpoints
    ]
    dir_none = re.search(r"dir\s*=\s*none", attrs, flags=re.I) is not None
    for (left, right), op in zip(zip(resolved, resolved[1:]), operators):
        for u in left:
            for v in right:
                if u == v:
                    continue
                if is_undirected_graph or op == "--" or dir_none:
                    undirected.add(frozenset((u, v)))
                else:
                    directed.add((u, v))


def _resolve_endpoint(
    endpoint: str,
    nodes: set[str],
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    *,
    is_undirected_graph: bool,
) -> set[str]:
    endpoint = endpoint.strip()
    body = _match_subgraph(endpoint)
    if body is not None:
        sub_nodes, sub_dir, sub_und = _parse_dot_body(
            body,
            is_undirected_graph=is_undirected_graph,
            in_subgraph=True,
        )
        nodes.update(sub_nodes)
        directed.update(sub_dir)
        undirected.update(sub_und)
        return sub_nodes
    if _has_subgraph_syntax(endpoint):
        raise _dot_parse_error(
            "unsupported subgraph syntax in an edge endpoint. "
            f"Offending endpoint: {endpoint!r}."
        )
    name = _clean_id(endpoint)
    if not name:
        raise _dot_parse_error(f"unsupported edge endpoint {endpoint!r}.")
    nodes.add(name)
    return {name}


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
