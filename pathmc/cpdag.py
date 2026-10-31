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
    ids: quote them (``"digraph" -> Y``). ``node = box`` is rejected; write
    ``node [shape=box]``. A quoted odd backslash run before a line feed is
    line continuation, so ``"foo\\<newline>bar"`` is the node ``foobar``.
    An even run is not continuation.

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


# Reserved statement keywords. A bare ``node`` / ``graph`` / ``edge`` /
# ``digraph`` / ``subgraph`` / ``strict`` is not a node id.
# ``node [shape=box]`` is a style declaration; ``node = box`` is not.
_BARE_KEYWORDS = frozenset({"digraph", "subgraph", "strict"})
_RESERVED = frozenset({"graph", "node", "edge"}) | _BARE_KEYWORDS


def _dot_parse_error(detail: str) -> ValueError:
    return ValueError(f"Could not parse DOT text: {detail}")


def _keyword_id_error(name: str) -> ValueError:
    """Reject an unquoted Graphviz keyword used where an id was accepted."""
    example = f'"{name}" -> Y'
    return _dot_parse_error(
        f"unsupported DOT syntax: {name!r} is a keyword, not a node id. "
        f"Quote it if it is a node, for example {example!r}."
    )


def _node_assignment_error(name: str, value: str | None) -> ValueError:
    """Reject ``node = value``. Graphviz wants an attribute list."""
    if value is not None:
        shown = f"{name} = {value}"
        instead = f"node [shape={value}]"
    else:
        shown = f"{name} = ..."
        instead = "node [shape=box]"
    return _dot_parse_error(
        f"unsupported DOT syntax: {shown!r}. Write {instead!r} instead."
    )


def _keyword_assignment_error(name: str) -> ValueError:
    quoted = f'"{name}" = box'
    return _dot_parse_error(
        f"unsupported DOT syntax: {name!r} is a keyword, not an assignment "
        f"name. Quote it, for example {quoted!r}."
    )


def _keyword_value_error(name: str, lhs: str | None) -> ValueError:
    quoted = f'"{name}"'
    if lhs:
        shown = f"{lhs}={name}"
        instead = f"{lhs}={quoted}"
    else:
        shown = name
        instead = quoted
    return _dot_parse_error(
        f"unsupported DOT syntax: {shown!r}. Write {instead!r} instead."
    )


def _keyword_graph_name_error(kind: str, name: str) -> ValueError:
    shown = f"{kind} {name}"
    instead = f'{kind} "{name}"'
    return _dot_parse_error(
        f"unsupported DOT syntax: {shown!r}. Write {instead!r} instead."
    )


def _end_of_quoted(text: str, start: int) -> int:
    """Index just after the quoted string at *start*, honoring ``\\"``."""
    i = start + 1
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if text[i] == '"':
            return i + 1
        i += 1
    return n


def _strip_comments(text: str) -> str:
    """Remove ``//``, ``#`` and ``/* ... */`` comments, ignoring quoted strings.

    A ``#`` or ``//`` inside a double-quoted DOT string (e.g.
    ``fillcolor="#eef5ff"``) or an HTML label (``label=<http://example.com>``)
    is *not* a comment and must survive, otherwise the rest of the statement
    — including the node it declares — is lost.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            end = _end_of_quoted(text, i)
            out.append(text[i:end])
            i = end
            continue
        if ch == "<":
            end = _skip_html(text, i)
            out.append(text[i:end])
            i = end
            continue
        if ch == "#" or (ch == "/" and text[i + 1 : i + 2] == "/"):
            newline = text.find("\n", i)
            if newline == -1:
                break
            i = newline
            continue
        if ch == "/" and text[i + 1 : i + 2] == "*":
            end = text.find("*/", i + 2)
            if end == -1:
                break
            # Graphviz ends the token at the comment. Deleting it would glue
            # ``subgraph/*c*/cluster`` into one invented id.
            out.append(" ")
            i = end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _skip_html(text: str, start: int) -> int:
    """Return the index just after the HTML label that starts at *start*."""
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise _dot_parse_error("unterminated HTML label.")


def _brace_interior(text: str) -> str | None:
    """Return the interior of the outer ``{ ... }``, ignoring non-structural braces.

    Comments are stripped first. Braces inside quotes or HTML labels
    (``label=<foo {bar}>``) do not open or close the graph. Returns ``None``
    when the text has no matching outer braces. An opening brace that is not
    closed, while some other ``}`` is present, is unbalanced subgraph syntax.
    """
    cleaned = _strip_comments(text)
    start = _opening_brace(cleaned)
    if start is None:
        return None
    return _brace_interior_from(cleaned, start)


def _opening_brace(cleaned: str) -> int | None:
    """Index of the first ``{`` outside quotes and HTML labels."""
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if ch == '"':
            i = _end_of_quoted(cleaned, i)
            continue
        if ch == "<":
            i = _skip_html(cleaned, i)
            continue
        if ch == "{":
            return i
        i += 1
    return None


def _brace_interior_from(cleaned: str, start: int) -> str | None:
    """Match the ``{`` at *start*, skipping quotes and HTML labels."""
    depth = 0
    saw_close = False
    i = start
    while i < len(cleaned):
        ch = cleaned[i]
        if ch == '"':
            i = _end_of_quoted(cleaned, i)
            continue
        if ch == "<":
            i = _skip_html(cleaned, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            saw_close = True
            depth -= 1
            if depth == 0:
                return cleaned[start + 1 : i]
        i += 1
    if not saw_close:
        return None
    raise _dot_parse_error("unsupported subgraph syntax (unbalanced '{').")


def _tokenize(
    text: str, *, skip_unsupported: bool = False
) -> list[str | tuple[str, str]]:
    """Tokenize a DOT body.

    Only ASCII space, tab, CR, and LF separate tokens. Other ``str.isspace()``
    characters, including U+00A0 and U+2003, stay inside an unquoted id.
    Graphviz and the previous reader keep those as one id; splitting them
    invents nodes and reports a false Markov mismatch. Whitespace is not a
    token: DOT makes ``;`` optional, so ``X -> Y Z -> Y`` is two statements.
    ``,`` and ``;`` are explicit separators. Quoted strings and HTML labels
    are single tokens, so a ``}`` inside either is not a subgraph boundary.
    An odd run of backslashes immediately before a line feed is Graphviz line
    continuation: ``"foo\\<LF>bar"`` is the id ``foobar``. An even run is not
    continuation; the backslashes and the line feed stay in the id.
    ``->`` and ``--`` are operators. A bare id may include ``:`` so a compass
    port stays part of the id, matching the previous reader.
    """
    tokens: list[str | tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == '"':
            buf: list[str] = []
            i += 1
            while i < n:
                if text[i] == "\\":
                    start = i
                    while i < n and text[i] == "\\":
                        i += 1
                    count = i - start
                    if i < n and text[i] == "\n" and count % 2 == 1:
                        # Odd run before LF is Graphviz line continuation.
                        if count > 1:
                            buf.append("\\" * (count - 1))
                        i += 1
                        continue
                    if i < n and text[i] == '"' and count % 2 == 1:
                        # The last backslash escapes the quote. Keep the rest.
                        if count > 1:
                            buf.append("\\" * (count - 1))
                        buf.append('"')
                        i += 1
                        continue
                    buf.append("\\" * count)
                    continue
                if text[i] == '"':
                    break
                buf.append(text[i])
                i += 1
            else:
                raise _dot_parse_error("unterminated quoted string.")
            tokens.append(("qid", "".join(buf)))
            i += 1
            continue
        if ch == "<":
            end = _skip_html(text, i)
            tokens.append(("html", text[i:end]))
            i = end
            continue
        if text[i : i + 2] in {"->", "--"}:
            tokens.append(text[i : i + 2])
            i += 2
            continue
        if ch == "+":
            tokens.append("+")
            i += 1
            continue
        if ch in "{}[]=,;":
            tokens.append(ch)
            i += 1
            continue
        numeral = re.match(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", text[i:])
        if numeral and numeral.group(0) not in {"-", "."}:
            tokens.append(("id", numeral.group(0)))
            i += numeral.end()
            continue
        if ch == "-":
            if skip_unsupported:
                i += 1
                continue
            raise _dot_parse_error(
                "unsupported DOT syntax near '-'. A hyphen is not part of a node id."
            )
        if ch in "()\\":
            if skip_unsupported:
                i += 1
                continue
            raise _dot_parse_error(f"unsupported DOT syntax near {ch!r}.")
        if ch == ":":
            tokens.append(":")
            i += 1
            continue
        end = i + 1
        while end < n and text[end] not in ' \t\r\n{}[]=,;+"<()\\:':
            if text[end : end + 2] in {"->", "--"} or text[end] == "-":
                break
            end += 1
        tokens.append(("id", text[i:end]))
        i = end
    return tokens


def _id_text(token: str | tuple[str, str] | None) -> str | None:
    if isinstance(token, tuple) and token[0] in {"id", "qid"}:
        return token[1]
    return None


def _is_id(token: str | tuple[str, str] | None) -> bool:
    return _id_text(token) is not None


def _is_keyword(token: str | tuple[str, str] | None, word: str) -> bool:
    if isinstance(token, tuple) and token[0] == "id":
        return token[1].lower() == word
    return False


class _DotBodyParser:
    """Recursive-descent reader for the inside of a DOT ``{ ... }`` block."""

    def __init__(self, body: str, *, is_undirected_graph: bool) -> None:
        self.tokens = _tokenize(_strip_comments(body))
        self.i = 0
        self.is_undirected_graph = is_undirected_graph
        self.nodes: set[str] = set()
        self.directed: set[tuple[str, str]] = set()
        self.undirected: set[frozenset[str]] = set()
        self.scopes: list[set[str]] = [set()]
        self.edge_dir_none: list[bool] = [False]
        self._dir_was_set = False
        self._endpoint_was_subgraph = False

    def peek(self, offset: int = 0) -> str | tuple[str, str] | None:
        index = self.i + offset
        if index >= len(self.tokens):
            return None
        return self.tokens[index]

    def pop(self) -> str | tuple[str, str]:
        token = self.peek()
        if token is None:
            raise _dot_parse_error(
                "unsupported subgraph syntax (unexpected end of graph)."
            )
        self.i += 1
        return token

    def add_node(self, name: str) -> None:
        # ``""`` is a real DOT id. Dropping it makes ``digraph { "" }`` compare
        # equal to an empty graph, and dropping it from an edge leaves a
        # dangling endpoint.
        self.nodes.add(name)
        self.scopes[-1].add(name)

    def parse(self) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
        self.parse_statements(stop_at_rbrace=False)
        return self.nodes, self.directed, self.undirected

    def parse_statements(self, *, stop_at_rbrace: bool) -> None:
        while True:
            token = self.peek()
            if token is None:
                if stop_at_rbrace:
                    raise _dot_parse_error(
                        "unsupported subgraph syntax (unbalanced '{')."
                    )
                return
            if token == "}":
                if stop_at_rbrace:
                    return
                raise _dot_parse_error("unsupported subgraph syntax (unexpected '}').")
            if token == ";":
                self.pop()
                continue
            self.parse_statement()

    def parse_statement(self) -> None:
        token = self.peek()
        if (
            _is_keyword(token, "graph")
            or _is_keyword(token, "node")
            or _is_keyword(token, "edge")
        ):
            if self.peek(1) == "[":
                is_edge_default = _is_keyword(token, "edge")
                self.pop()
                applied: bool | None = None
                while self.peek() == "[":
                    none = self.take_attr_list()
                    if self._dir_was_set:
                        applied = none
                if is_edge_default and applied is not None:
                    # Later edges in this block inherit dir=none. Ignoring it
                    # reports a directed collider.
                    self.edge_dir_none[-1] = applied
                return
            nxt = _id_text(self.peek(1))
            if nxt is not None and nxt.startswith("-"):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '-'. A hyphen is not part of a node id."
                )
        if _is_id(token) and self.peek(1) == "=":
            name = _id_text(token)
            quoted = isinstance(token, tuple) and token[0] == "qid"
            if not quoted and name is not None and name.lower() == "node":
                value = _id_text(self.peek(2))
                if self.peek(3) == "+":
                    value = None
                raise _node_assignment_error(name, value)
            if not quoted and name is not None and name.lower() in _BARE_KEYWORDS:
                raise _keyword_assignment_error(name)
            self.pop()
            self.pop()
            self.take_value(name)
            return
        self.parse_node_or_edge()

    def take_value(self, lhs: str | None = None) -> None:
        self._take_value_piece(lhs)
        while self.peek() == "+":
            self.pop()
            self._take_value_piece(lhs)

    def _take_value_piece(self, lhs: str | None) -> None:
        token = self.peek()
        if isinstance(token, tuple) and token[0] == "html":
            self.pop()
            return
        if not _is_id(token):
            raise _dot_parse_error(
                "unsupported subgraph syntax in an attribute value. "
                f"Offending token: {token!r}."
            )
        if (
            isinstance(token, tuple)
            and token[0] == "id"
            and token[1].lower() in _BARE_KEYWORDS
        ):
            raise _keyword_value_error(token[1], lhs)
        self.pop()

    def parse_node_or_edge(self) -> None:
        left = self.parse_node_list(on_edge=False)
        if self.peek() not in {"->", "--"}:
            while self.peek() == "[":
                self.take_attr_list()
            return
        hops: list[tuple[str, str, str]] = []
        current = left
        while self.peek() in {"->", "--"}:
            op = self.pop()
            if not isinstance(op, str):
                raise _dot_parse_error(f"unsupported edge endpoint {op!r}.")
            right = self.parse_node_list(on_edge=True)
            for u in current:
                for v in right:
                    if u != v:
                        hops.append((u, v, op))
            current = right
        dir_none = self.edge_dir_none[-1]
        while self.peek() == "[":
            listed = self.take_attr_list()
            if self._dir_was_set:
                dir_none = listed
        for u, v, op in hops:
            if self.is_undirected_graph or op == "--" or dir_none:
                self.undirected.add(frozenset((u, v)))
            else:
                self.directed.add((u, v))

    def parse_node_list(self, *, on_edge: bool) -> set[str]:
        """Parse a comma-separated endpoint list (``B, C`` or ``{B C}``)."""
        names = self.parse_endpoint()
        self.skip_attrs_belonging_to_node(on_edge=on_edge)
        while self.peek() == ",":
            if self._endpoint_was_subgraph:
                raise _dot_parse_error(
                    "unsupported DOT syntax: ',' after a subgraph. A comma "
                    "separates node ids, for example 'A -> B, C', not "
                    "'A -> { B C }, D'."
                )
            self.pop()
            names |= self.parse_endpoint()
            self.skip_attrs_belonging_to_node(on_edge=on_edge)
        return names

    def skip_attrs_belonging_to_node(self, *, on_edge: bool) -> None:
        """Drop ``[attr]`` groups that sit on a node before the next hop.

        ``A [label=x] -> B [dir=none]`` puts the first list on ``A`` and the
        second on the edge. A list followed by ``,`` belongs to that node in
        a node list before the edge operator. After the operator, ``,``
        following an attribute list is unsupported: it must not invent a node
        or drop ``dir=none``.
        """
        saved = self.i
        consumed = False
        saw_dir_none = False
        while self.peek() == "[":
            saw_dir_none = self.take_attr_list() or saw_dir_none
            consumed = True
        if saw_dir_none and self.peek() in {"->", "--"}:
            raise _dot_parse_error(
                "unsupported DOT syntax: dir=none on a node cannot be followed "
                "by an edge operator. Put dir=none on the edge, for example "
                "'A -> C [dir=none]'."
            )
        if self.peek() in {"->", "--"} or (self.peek() == "," and not on_edge):
            return
        if on_edge and consumed and self.peek() == ",":
            raise _dot_parse_error(
                "unsupported DOT syntax: an attribute list cannot be followed "
                "by ','. Offending token: ','."
            )
        self.i = saved

    def parse_endpoint(self) -> set[str]:
        token = self.peek()
        self._endpoint_was_subgraph = False
        if token == "{" or _is_keyword(token, "subgraph"):
            nodes = self.parse_subgraph()
            self._endpoint_was_subgraph = True
            return nodes
        if isinstance(token, tuple) and token[0] == "html":
            self.pop()
            # Graphviz node name is the HTML interior, not the angle brackets.
            name = self._concat_literal(token[1][1:-1])
            if self.peek() == ":":
                self._skip_port_chain()
            self.add_node(name)
            return {name}
        if not _is_id(token):
            if token == "subgraph" or (
                isinstance(token, str) and "subgraph" in token.lower()
            ):
                raise _dot_parse_error(
                    "unsupported subgraph syntax in an edge endpoint. "
                    f"Offending endpoint: {token!r}."
                )
            raise _dot_parse_error(f"unsupported edge endpoint {token!r}.")
        raw_name = _id_text(token)
        if raw_name is None:
            raise _dot_parse_error(f"unsupported edge endpoint {token!r}.")
        name = raw_name
        literal = isinstance(token, tuple) and token[0] == "qid"
        self.pop()
        while self.peek() == "+":
            nxt = self.peek(1)
            if not (literal and self._is_literal_piece(nxt)):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '+'. Only quoted strings can "
                    'be concatenated. Quote each piece, for example \'"A"+"B"\'.'
                )
            self.pop()
            name += self._literal_text(self.pop())
        if self.peek() == ":":
            self._skip_port_chain()
        if not literal and name.lower() in _RESERVED:
            nxt = _id_text(self.peek())
            if nxt is not None and nxt.startswith("-"):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '-'. A hyphen is not part of a node id."
                )
        if not literal and name.lower() in _RESERVED:
            raise _keyword_id_error(name)
        self.add_node(name)
        return {name}

    def _skip_port_chain(self) -> None:
        """Drop at most two ``:id`` pieces. A colon inside an id is not a port.

        ``A:n`` is node ``A`` because the tokenizer splits on ``:``. ``"A:"``
        already includes the colon, so ``"A:":n`` stays node ``A:``. An HTML
        piece is a port, not a node. An unquoted keyword is not a port.
        """
        for _ in range(2):
            if self.peek() != ":":
                return
            self.pop()
            piece = self.peek()
            if isinstance(piece, tuple) and piece[0] == "html":
                self.pop()
                continue
            if not _is_id(piece):
                raise _dot_parse_error(f"unsupported compass port before {piece!r}.")
            if not (isinstance(piece, tuple) and piece[0] == "qid"):
                port_name = _id_text(piece)
                if port_name is not None and port_name.lower() in _RESERVED:
                    raise _keyword_id_error(port_name)
            self.pop()

    def _is_literal_piece(self, token: str | tuple[str, str] | None) -> bool:
        return isinstance(token, tuple) and token[0] in {"qid", "html"}

    def _literal_text(self, token: str | tuple[str, str]) -> str:
        if isinstance(token, tuple) and token[0] == "html":
            return token[1][1:-1]
        text = _id_text(token)
        if text is None:
            raise _dot_parse_error("unsupported '+' in a DOT id.")
        return text

    def _concat_literal(self, name: str) -> str:
        while self.peek() == "+":
            nxt = self.peek(1)
            if not self._is_literal_piece(nxt):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '+'. Only quoted strings can "
                    'be concatenated. Quote each piece, for example \'"A"+"B"\'.'
                )
            self.pop()
            name += self._literal_text(self.pop())
        return name

    def parse_subgraph(self) -> set[str]:
        if _is_keyword(self.peek(), "subgraph"):
            keyword = _id_text(self.peek()) or "subgraph"
            self.pop()
            nxt = self.peek()
            if nxt != "{":
                if not (_is_id(nxt) or (isinstance(nxt, tuple) and nxt[0] == "html")):
                    raise _keyword_id_error(keyword)
                self._skip_subgraph_id()
            if self.peek() != "{":
                raise _dot_parse_error(
                    "unsupported subgraph syntax. A subgraph block must be "
                    "'subgraph [id] { ... }' (the keyword and id are optional). "
                    f"Offending token: {self.peek()!r}."
                )
        if self.peek() != "{":
            raise _dot_parse_error(
                "unsupported subgraph syntax. A subgraph block must be "
                "'subgraph [id] { ... }' (the keyword and id are optional)."
            )
        self.pop()
        self.scopes.append(set())
        self.edge_dir_none.append(self.edge_dir_none[-1])
        self.parse_statements(stop_at_rbrace=True)
        if self.peek() != "}":
            raise _dot_parse_error("unsupported subgraph syntax (unbalanced '{').")
        self.pop()
        self.edge_dir_none.pop()
        inner = self.scopes.pop()
        self.scopes[-1].update(inner)
        return inner

    def _skip_subgraph_id(self) -> None:
        """Consume a cluster id. It is not a node."""
        token = self.peek()
        if isinstance(token, tuple) and token[0] == "html":
            self.pop()
            return
        if not _is_id(token):
            raise _dot_parse_error(
                "unsupported subgraph syntax. A subgraph id must be a DOT id. "
                f"Offending token: {token!r}."
            )
        if not (isinstance(token, tuple) and token[0] == "qid"):
            name = _id_text(token)
            if name is not None and name.lower() in _RESERVED:
                raise _keyword_id_error(name)
        self.pop()
        while self.peek() == "+":
            nxt = self.peek(1)
            if not self._is_literal_piece(nxt):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '+'. Only quoted strings can "
                    "be concatenated in a subgraph id."
                )
            self.pop()
            self.pop()

    def take_attr_list(self) -> bool:
        """Consume one ``[ ... ]`` group. Return True if it sets ``dir=none``."""
        if self.pop() != "[":
            raise _dot_parse_error("unsupported subgraph syntax (expected '[').")
        dir_none = False
        self._dir_was_set = False
        depth = 1
        prev: str | tuple[str, str] | None = None
        while depth:
            token = self.peek()
            if token is None:
                raise _dot_parse_error("unsupported subgraph syntax (unbalanced '[').")
            self.pop()
            if token == "[":
                depth += 1
            elif token == "]":
                depth -= 1
            else:
                if (
                    isinstance(token, tuple)
                    and token[0] == "id"
                    and token[1].lower() in _BARE_KEYWORDS
                ):
                    raise _keyword_id_error(token[1])
                if (
                    token == "="
                    and isinstance(prev, tuple)
                    and prev[0] == "id"
                    and prev[1].lower() == "node"
                ):
                    value = _id_text(self.peek())
                    if self.peek(1) == "+":
                        value = None
                    raise _node_assignment_error(prev[1], value)
                attr = _id_text(prev)
                if token == "=" and attr is not None and attr.lower() == "dir":
                    self._dir_was_set = True
                    value_token = self.peek()
                    if isinstance(value_token, tuple) and value_token[0] == "html":
                        dir_none = value_token[1][1:-1] == "none"
                    else:
                        value = _id_text(value_token)
                        dir_none = value is not None and value.lower() == "none"
            prev = token
        return dir_none


def _reject_keyword_preamble(span: str, kind: str) -> None:
    """Reject unquoted ``digraph`` / ``subgraph`` / ``strict`` before ``{``.

    The graph name and a header attribute list are outside the body. Dropping
    a keyword there compares equal to the graph with that keyword removed.
    Identifiers use the body lexer: a numeral does not glue onto the next
    letters, and U+00A0 / U+2003 stay inside one id. A hyphen is not an id
    character, but it is not this hole, so it is skipped rather than rejected.
    """
    seen_name = False
    depth = 0
    prev_lhs: str | None = None
    saw_eq = False
    tokens = list(_tokenize(span, skip_unsupported=True))
    for index, token in enumerate(tokens):
        if token == "[":
            depth += 1
            saw_eq = False
            prev_lhs = None
            continue
        if token == "]":
            depth = max(0, depth - 1)
            saw_eq = False
            prev_lhs = None
            continue
        if token == "=":
            if prev_lhs is not None and prev_lhs.lower() == "node":
                value = _id_text(tokens[index + 1]) if index + 1 < len(tokens) else None
                if index + 2 < len(tokens) and tokens[index + 2] == "+":
                    value = None
                raise _node_assignment_error(prev_lhs, value)
            saw_eq = True
            continue
        if token == "+":
            continue
        if isinstance(token, tuple) and token[0] == "id":
            word = token[1]
            if word.lower() in _BARE_KEYWORDS:
                if depth == 0 and not seen_name:
                    raise _keyword_graph_name_error(kind, word)
                if saw_eq and prev_lhs:
                    raise _keyword_value_error(word, prev_lhs)
                raise _keyword_id_error(word)
            if depth == 0 and not seen_name:
                seen_name = True
            prev_lhs = word
            saw_eq = False
            continue
        if isinstance(token, tuple) and token[0] in {"qid", "html"}:
            if depth == 0 and not seen_name:
                seen_name = True
            saw_eq = False
            prev_lhs = None
            continue
        saw_eq = False
        prev_lhs = None


def _reject_keywords_before_body(dot_text: str) -> None:
    """Apply the preamble keyword check when a graph header is present."""
    header = re.match(r"\s*(?:strict\s+)?(graph|digraph)\b", dot_text, flags=re.I)
    if header is None:
        return
    cleaned = _strip_comments(dot_text[header.end() :])
    brace = _opening_brace(cleaned)
    if brace is None:
        return
    _reject_keyword_preamble(cleaned[:brace], header.group(1))


def _parse_dot(
    dot_text: str,
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse DOT text into ``(nodes, directed_edges, undirected_edges)``.

    Directed edges are ``(u, v)`` tuples; undirected edges are
    ``frozenset({u, v})``. An ``A -> B`` edge carrying a ``dir=none``
    attribute is treated as undirected, as is any edge in an undirected
    ``graph { ... }`` block. Chained edges (``A -> B -> C``) are expanded.
    Global style declarations (``graph [...]``, ``node [...]``, and ``ID =
    value`` assignments such as ``rankdir=LR``) are ignored. ``node = box``
    is rejected; write ``node [shape=box]``. Unquoted ``digraph``,
    ``subgraph``, and ``strict`` are keywords, not ids, graph names, or
    assignment values. Quote them (``label="strict"``, ``digraph "strict"``).
    A quoted odd backslash run before a line feed continues the string.
    A ``subgraph`` block, including an anonymous ``{ ... }`` cluster, is layout
    only: its nodes and edges are parsed, and the cluster id is not a node.
    An edge into a subgraph connects to each node in that subgraph.
    ``;`` is optional, so ``X -> Y Z -> Y`` is two edges. A comma separates
    a node list (``A -> B, C``), not statements. A ``}`` inside a comment or an
    HTML label is not a subgraph boundary. Node identity is the DOT id;
    ``label`` attributes are not used as identity.

    Raises
    ------
    ValueError
        If no ``graph``/``digraph`` block can be found, or a ``subgraph``
        block is not valid DOT.
    """
    header = re.match(r"\s*(?:strict\s+)?(graph|digraph)\b", dot_text, flags=re.I)
    if header is None:
        raise ValueError(
            "Could not parse DOT text. Expected a 'graph { ... }' or "
            "'digraph { ... }' block."
        )
    _reject_keywords_before_body(dot_text)
    body = _brace_interior(dot_text)
    if body is None:
        raise ValueError(
            "Could not parse DOT text. Expected a 'graph { ... }' or "
            "'digraph { ... }' block."
        )
    is_undirected_graph = header.group(1).lower() == "graph"
    return _parse_dot_body(body, is_undirected_graph=is_undirected_graph)


def _parse_dot_body(
    body: str, *, is_undirected_graph: bool, in_subgraph: bool = False
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse the inside of a DOT ``{ ... }`` block.

    Shared by :func:`_parse_dot` and (for directed graphs) by
    ``pathmc.dag._parse_dag``, so all DOT consumers agree on the grammar:
    quoted ids, chained edges, attribute brackets, quote-aware comments,
    HTML labels, optional semicolons, and ``subgraph`` blocks. A subgraph
    does not add a node for its id; edges and node statements inside it
    are real.
    """
    del in_subgraph
    return _DotBodyParser(body, is_undirected_graph=is_undirected_graph).parse()


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
