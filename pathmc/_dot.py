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
"""Dependency-free DOT reader shared by Markov comparison and DAG construction.

Not a Graphviz implementation. It reads the subset pathmc compares and
builds: node ids, edges, ``dir=none``, and ``subgraph`` blocks as layout.
These Graphviz differences are intentional; do not add branches for them:

- a block comment before the header (``/* lead */ digraph``) still fails
- ``graph = box`` and ``edge = box`` are unchanged
- unquoted ``graph`` / ``node`` / ``edge`` attribute values are unchanged
- ``digraph my-graph`` and ``digraph.foo`` still parse
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ``graph`` / ``node`` / ``edge`` are ids in some positions and keywords in
# others. ``digraph`` / ``subgraph`` / ``strict`` are keywords in every
# unquoted id position this reader accepts. Do not merge the sets.
_BARE_KEYWORDS = frozenset({"digraph", "subgraph", "strict"})
_RESERVED = frozenset({"graph", "node", "edge"}) | _BARE_KEYWORDS

_MISSING_GRAPH = (
    "Could not parse DOT text. Expected a 'graph { ... }' or 'digraph { ... }' block."
)
_DAG_MISSING_BRACES = (
    "Malformed DOT digraph: missing braces. Expected 'digraph { A -> B; ... }'."
)

_Kind = Literal["id", "qid", "html", "op"]


@dataclass(frozen=True, slots=True)
class _Tok:
    """One DOT token.

    ``glued`` is set only by the lexer, for a numeral fused to a bare
    keyword (``0strict``). Callers must not re-detect that spelling.
    HTML ``text`` includes the angle brackets; :attr:`interior` is the
    node or value text.
    """

    kind: _Kind
    text: str
    glued: bool = False

    @property
    def interior(self) -> str:
        if self.kind == "html":
            return self.text[1:-1]
        return self.text

    def __repr__(self) -> str:
        # Match the previous ``str | tuple`` repr so error text stays stable.
        if self.kind == "op":
            return repr(self.text)
        return repr((self.kind, self.text))


def _is_op(token: _Tok | None, *texts: str) -> bool:
    return token is not None and token.kind == "op" and token.text in texts


def _id_text(token: _Tok | None) -> str | None:
    if token is not None and token.kind in {"id", "qid"}:
        return token.text
    return None


def _is_keyword(token: _Tok | None, word: str) -> bool:
    return token is not None and token.kind == "id" and token.text.lower() == word


def _is_literal(token: _Tok | None) -> bool:
    return token is not None and token.kind in {"qid", "html"}


def _literal_text(token: _Tok) -> str:
    return token.interior if token.kind == "html" else token.text


def _rejected(token: _Tok | None, words: frozenset[str]) -> str | None:
    """Spelling to reject, or None if *token* is not an unquoted keyword.

    Quoted ids and HTML labels are never keywords. A numeral glued to
    ``digraph`` / ``subgraph`` / ``strict`` is the whole spelling, so the
    error names ``0strict`` rather than the split keyword.
    """
    if token is None or token.kind != "id":
        return None
    if token.glued or token.text.lower() in words:
        return token.text
    return None


def _shown_assignment_value(value: _Tok | None, *, plus: bool) -> str | None:
    """Value text for ``node = ...``. A following ``+`` hides it."""
    if plus:
        return None
    return _id_text(value)


def _dot_parse_error(detail: str) -> ValueError:
    return ValueError(f"Could not parse DOT text: {detail}")


def _keyword_id_error(name: str) -> ValueError:
    example = f'"{name}" -> Y'
    return _dot_parse_error(
        f"unsupported DOT syntax: {name!r} is a keyword, not a node id. "
        f"Quote it if it is a node, for example {example!r}."
    )


def _node_assignment_error(name: str, value: str | None) -> ValueError:
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
        f"unsupported DOT syntax: {shown!r} is a keyword, not an attribute "
        f"value. Write {instead!r} instead."
    )


def _keyword_graph_name_error(kind: str, name: str) -> ValueError:
    shown = f"{kind} {name}"
    instead = f'{kind} "{name}"'
    return _dot_parse_error(
        f"unsupported DOT syntax: {shown!r}. Write {instead!r} instead."
    )


def _is_id_space(ch: str) -> bool:
    """True for a space the lexer keeps inside an id."""
    return ch.isspace() and ch not in " \t\r\n"


def _header_keyword_error(shown: str, instead: str) -> ValueError:
    return _dot_parse_error(
        f"unsupported DOT syntax: {shown!r} is not an ASCII-separated keyword. "
        f"Write {instead!r} instead."
    )


def _ascii_keyword_form(shown: str) -> str:
    collapsed = re.sub(r"[^\S \t\r\n]+", " ", shown)
    return re.sub(r" +", " ", collapsed).strip()


def _match_graph_header(dot_text: str) -> re.Match[str] | None:
    """Match ``[strict] graph|digraph`` using the body separator rule.

    ASCII space, tab, CR, and LF separate keywords. U+00A0, U+2003, and
    other non-ASCII spaces stay inside an id, so a keyword glued to one is
    not a header. ``digraph.foo`` still matches: a dot is outside this hole.
    """
    leading = re.match(
        r"([^\S \t\r\n]+)[ \t\r\n]*((?:strict|digraph|graph))",
        dot_text,
        flags=re.I,
    )
    if leading:
        raise _header_keyword_error(leading.group(0), leading.group(2))
    match = re.match(
        r"[ \t\r\n]*(?:(?P<strict>strict)(?=[ \t\r\n]))?[ \t\r\n]*"
        r"(?P<kind>digraph|graph)",
        dot_text,
        flags=re.I,
    )
    if match is not None:
        end = match.end()
        if end < len(dot_text) and _is_id_space(dot_text[end]):
            kind = match.group("kind")
            rest = re.match(r"[^\S \t\r\n]+(\S*)", dot_text[end:])
            tail = rest.group(1) if rest else ""
            shown = kind + (rest.group(0) if rest else dot_text[end])
            instead = f"{kind} {tail}".strip() if tail else kind
            raise _header_keyword_error(shown, instead)
        return match
    glued = re.match(
        r"[ \t\r\n]*((?:strict|digraph|graph)[ \t\r\n]*[^\S \t\r\n]+\S*)",
        dot_text,
        flags=re.I,
    )
    if glued:
        shown = glued.group(1)
        raise _header_keyword_error(shown, _ascii_keyword_form(shown))
    return None


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


def _id_stops(text: str, index: int) -> bool:
    ch = text[index]
    if ch in ' \t\r\n{}[]=,;+"<()\\:' or ch == "-":
        return True
    return text[index : index + 2] in {"->", "--"}


def _tokenize(text: str, *, skip_unsupported: bool = False) -> list[_Tok]:
    """Tokenize a DOT span.

    Only ASCII space, tab, CR, and LF separate tokens. Other ``str.isspace()``
    characters, including U+00A0 and U+2003, stay inside an unquoted id.
    Whitespace is not a token: DOT makes ``;`` optional, so ``X -> Y Z -> Y``
    is two statements. ``,`` and ``;`` are explicit separators. Quoted strings
    and HTML labels are single tokens, so a ``}`` inside either is not a
    subgraph boundary. An odd run of backslashes immediately before a line
    feed is Graphviz line continuation: ``"foo\\<LF>bar"`` is the id
    ``foobar``. An even run is not continuation. A numeral immediately
    followed by ``digraph``, ``subgraph``, or ``strict`` is one id with
    :attr:`_Tok.glued` set. ``->`` and ``--`` are operators. A bare id stops
    at ``:`` so a compass port stays a separate token.

    ``skip_unsupported`` drops ``-()\\`` instead of raising. The preamble
    walker uses that so ``digraph my-graph`` stays accepted; the body does not.
    """
    tokens: list[_Tok] = []
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
                        if count > 1:
                            buf.append("\\" * (count - 1))
                        i += 1
                        continue
                    if i < n and text[i] == '"' and count % 2 == 1:
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
            tokens.append(_Tok("qid", "".join(buf)))
            i += 1
            continue
        if ch == "<":
            end = _skip_html(text, i)
            tokens.append(_Tok("html", text[i:end]))
            i = end
            continue
        if text[i : i + 2] in {"->", "--"}:
            tokens.append(_Tok("op", text[i : i + 2]))
            i += 2
            continue
        if ch == "+":
            tokens.append(_Tok("op", "+"))
            i += 1
            continue
        if ch in "{}[]=,;":
            tokens.append(_Tok("op", ch))
            i += 1
            continue
        numeral = re.match(r"-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", text[i:])
        if numeral and numeral.group(0) not in {"-", "."}:
            end = i + numeral.end()
            word_end = end
            while word_end < n and not _id_stops(text, word_end):
                word_end += 1
            word = text[end:word_end]
            if word.lower() in _BARE_KEYWORDS:
                tokens.append(_Tok("id", text[i:word_end], glued=True))
                i = word_end
            else:
                tokens.append(_Tok("id", numeral.group(0)))
                i = end
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
            tokens.append(_Tok("op", ":"))
            i += 1
            continue
        end = i + 1
        while end < n and not _id_stops(text, end):
            end += 1
        tokens.append(_Tok("id", text[i:end]))
        i = end
    return tokens


class _DotBodyParser:
    """Recursive-descent reader for one comment-stripped ``{ ... }`` interior."""

    def __init__(self, body: str, *, is_undirected_graph: bool) -> None:
        self.tokens = _tokenize(body)
        self.i = 0
        self.is_undirected_graph = is_undirected_graph
        self.nodes: set[str] = set()
        self.directed: set[tuple[str, str]] = set()
        self.undirected: set[frozenset[str]] = set()
        self.edge_dir_none: list[bool] = [False]

    def peek(self, offset: int = 0) -> _Tok | None:
        index = self.i + offset
        if index >= len(self.tokens):
            return None
        return self.tokens[index]

    def pop(self) -> _Tok:
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

    def parse(self) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
        self.parse_statements(stop_at_rbrace=False)
        return self.nodes, self.directed, self.undirected

    def parse_statements(self, *, stop_at_rbrace: bool) -> set[str]:
        found: set[str] = set()
        while True:
            token = self.peek()
            if token is None:
                if stop_at_rbrace:
                    raise _dot_parse_error(
                        "unsupported subgraph syntax (unbalanced '{')."
                    )
                return found
            if _is_op(token, "}"):
                if stop_at_rbrace:
                    return found
                raise _dot_parse_error("unsupported subgraph syntax (unexpected '}').")
            if _is_op(token, ";"):
                self.pop()
                continue
            found |= self.parse_statement()

    def parse_statement(self) -> set[str]:
        token = self.peek()
        if (
            _is_keyword(token, "graph")
            or _is_keyword(token, "node")
            or _is_keyword(token, "edge")
        ):
            if _is_op(self.peek(1), "["):
                is_edge_default = _is_keyword(token, "edge")
                self.pop()
                applied: bool | None = None
                while _is_op(self.peek(), "["):
                    none, was_set = self.take_attr_list()
                    if was_set:
                        applied = none
                if is_edge_default and applied is not None:
                    # Later edges in this block inherit dir=none. Ignoring it
                    # reports a directed collider.
                    self.edge_dir_none[-1] = applied
                return set()
        if (
            token is not None
            and token.kind in {"id", "qid"}
            and _is_op(self.peek(1), "=")
        ):
            name = token.text
            if token.kind == "id" and name.lower() == "node":
                raise _node_assignment_error(
                    name,
                    _shown_assignment_value(
                        self.peek(2), plus=_is_op(self.peek(3), "+")
                    ),
                )
            shown = _rejected(token, _BARE_KEYWORDS)
            if shown is not None:
                raise _keyword_assignment_error(shown)
            self.pop()
            self.pop()
            self.take_value(name)
            return set()
        return self.parse_node_or_edge()

    def take_value(self, lhs: str | None = None) -> None:
        self._take_value_piece(lhs)
        while _is_op(self.peek(), "+"):
            self.pop()
            self._take_value_piece(lhs)

    def _take_value_piece(self, lhs: str | None) -> None:
        token = self.peek()
        if token is not None and token.kind == "html":
            self.pop()
            return
        if _id_text(token) is None:
            raise _dot_parse_error(
                "unsupported subgraph syntax in an attribute value. "
                f"Offending token: {token!r}."
            )
        shown = _rejected(token, _BARE_KEYWORDS)
        if shown is not None:
            raise _keyword_value_error(shown, lhs)
        self.pop()

    def parse_node_or_edge(self) -> set[str]:
        left = self.parse_node_list(on_edge=False)
        names = set(left)
        if not _is_op(self.peek(), "->", "--"):
            while _is_op(self.peek(), "["):
                self.take_attr_list()
            return names
        hops: list[tuple[str, str, str]] = []
        current = left
        while _is_op(self.peek(), "->", "--"):
            op = self.pop().text
            right = self.parse_node_list(on_edge=True)
            names |= right
            for u in current:
                for v in right:
                    if u != v:
                        hops.append((u, v, op))
            current = right
        dir_none = self.edge_dir_none[-1]
        while _is_op(self.peek(), "["):
            listed, was_set = self.take_attr_list()
            if was_set:
                dir_none = listed
        for u, v, op in hops:
            if self.is_undirected_graph or op == "--" or dir_none:
                self.undirected.add(frozenset((u, v)))
            else:
                self.directed.add((u, v))
        return names

    def parse_node_list(self, *, on_edge: bool) -> set[str]:
        """Parse a comma-separated endpoint list (``B, C`` or ``{B C}``)."""
        names, was_subgraph = self.parse_endpoint()
        self.skip_attrs_belonging_to_node(on_edge=on_edge)
        while _is_op(self.peek(), ","):
            if was_subgraph:
                raise _dot_parse_error(
                    "unsupported DOT syntax: ',' after a subgraph. A comma "
                    "separates node ids, for example 'A -> B, C', not "
                    "'A -> { B C }, D'."
                )
            self.pop()
            more, was_subgraph = self.parse_endpoint()
            names |= more
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
        while _is_op(self.peek(), "["):
            none, _was_set = self.take_attr_list()
            saw_dir_none = none or saw_dir_none
            consumed = True
        if saw_dir_none and _is_op(self.peek(), "->", "--"):
            raise _dot_parse_error(
                "unsupported DOT syntax: dir=none on a node cannot be followed "
                "by an edge operator. Put dir=none on the edge, for example "
                "'A -> C [dir=none]'."
            )
        if _is_op(self.peek(), "->", "--") or (
            _is_op(self.peek(), ",") and not on_edge
        ):
            return
        if on_edge and consumed and _is_op(self.peek(), ","):
            raise _dot_parse_error(
                "unsupported DOT syntax: an attribute list cannot be followed "
                "by ','. Offending token: ','."
            )
        self.i = saved

    def parse_endpoint(self) -> tuple[set[str], bool]:
        token = self.peek()
        if _is_op(token, "{") or _is_keyword(token, "subgraph"):
            return self.parse_subgraph(), True
        if token is not None and token.kind == "html":
            self.pop()
            name = self._concat_pieces(token.interior)
            if _is_op(self.peek(), ":"):
                self._skip_port_chain()
            self.add_node(name)
            return {name}, False
        if token is None or token.kind not in {"id", "qid"}:
            raise _dot_parse_error(f"unsupported edge endpoint {token!r}.")
        literal = token.kind == "qid"
        shown = _rejected(token, _RESERVED)
        self.pop()
        if literal:
            name = self._concat_pieces(token.text)
        elif _is_op(self.peek(), "+"):
            raise _dot_parse_error(
                "unsupported DOT syntax near '+'. Only quoted strings can "
                'be concatenated. Quote each piece, for example \'"A"+"B"\'.'
            )
        else:
            name = token.text
        if _is_op(self.peek(), ":"):
            self._skip_port_chain()
        if shown is not None:
            nxt = _id_text(self.peek())
            if nxt is not None and nxt.startswith("-"):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '-'. A hyphen is not part of a node id."
                )
            raise _keyword_id_error(shown)
        self.add_node(name)
        return {name}, False

    def _skip_port_chain(self) -> None:
        """Drop at most two ``:id`` pieces. A colon inside an id is not a port.

        ``A:n`` is node ``A`` because the tokenizer splits on ``:``. ``"A:"``
        already includes the colon, so ``"A:":n`` stays node ``A:``. An HTML
        piece is a port, not a node. An unquoted keyword is not a port.
        """
        for _ in range(2):
            if not _is_op(self.peek(), ":"):
                return
            self.pop()
            piece = self.peek()
            if piece is not None and piece.kind == "html":
                self.pop()
                continue
            if _id_text(piece) is None:
                raise _dot_parse_error(f"unsupported compass port before {piece!r}.")
            port = _rejected(piece, _RESERVED)
            if port is not None:
                raise _keyword_id_error(port)
            self.pop()

    def _concat_pieces(self, name: str) -> str:
        while _is_op(self.peek(), "+"):
            nxt = self.peek(1)
            if not _is_literal(nxt):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '+'. Only quoted strings can "
                    'be concatenated. Quote each piece, for example \'"A"+"B"\'.'
                )
            self.pop()
            assert nxt is not None
            name += _literal_text(self.pop())
        return name

    def parse_subgraph(self) -> set[str]:
        if _is_keyword(self.peek(), "subgraph"):
            keyword = self.peek()
            assert keyword is not None
            self.pop()
            nxt = self.peek()
            if not _is_op(nxt, "{"):
                if nxt is None or nxt.kind not in {"id", "qid", "html"}:
                    raise _keyword_id_error(keyword.text)
                self._skip_subgraph_id()
            if not _is_op(self.peek(), "{"):
                raise _dot_parse_error(
                    "unsupported subgraph syntax. A subgraph block must be "
                    "'subgraph [id] { ... }' (the keyword and id are optional). "
                    f"Offending token: {self.peek()!r}."
                )
        self.pop()
        self.edge_dir_none.append(self.edge_dir_none[-1])
        inner = self.parse_statements(stop_at_rbrace=True)
        if not _is_op(self.peek(), "}"):
            raise _dot_parse_error("unsupported subgraph syntax (unbalanced '{').")
        self.pop()
        self.edge_dir_none.pop()
        return inner

    def _skip_subgraph_id(self) -> None:
        """Consume a cluster id. It is not a node."""
        token = self.peek()
        if token is not None and token.kind == "html":
            self.pop()
            return
        if token is None or token.kind not in {"id", "qid"}:
            raise _dot_parse_error(
                "unsupported subgraph syntax. A subgraph id must be a DOT id. "
                f"Offending token: {token!r}."
            )
        if token.glued:
            raise _keyword_graph_name_error("subgraph", token.text)
        shown = _rejected(token, _RESERVED)
        if shown is not None:
            raise _keyword_id_error(shown)
        self.pop()
        while _is_op(self.peek(), "+"):
            nxt = self.peek(1)
            if not _is_literal(nxt):
                raise _dot_parse_error(
                    "unsupported DOT syntax near '+'. Only quoted strings can "
                    "be concatenated in a subgraph id."
                )
            self.pop()
            self.pop()

    def take_attr_list(self) -> tuple[bool, bool]:
        """Consume one ``[ ... ]`` group.

        Return ``(dir_is_none, dir_was_set)``.
        """
        self.pop()
        dir_none = False
        dir_was_set = False
        depth = 1
        prev: _Tok | None = None
        lhs: str | None = None
        while depth:
            token = self.peek()
            if token is None:
                raise _dot_parse_error("unsupported subgraph syntax (unbalanced '[').")
            self.pop()
            if _is_op(token, "["):
                depth += 1
            elif _is_op(token, "]"):
                depth -= 1
            else:
                shown = _rejected(token, _BARE_KEYWORDS)
                if shown is not None:
                    if prev is not None and _is_op(prev, "=") and lhs:
                        raise _keyword_value_error(shown, lhs)
                    raise _keyword_id_error(shown)
                if _is_op(token, "=") and prev is not None and prev.kind == "id":
                    lhs = prev.text
                elif not _is_op(token, "="):
                    lhs = None
                if (
                    _is_op(token, "=")
                    and prev is not None
                    and prev.kind == "id"
                    and prev.text.lower() == "node"
                ):
                    raise _node_assignment_error(
                        prev.text,
                        _shown_assignment_value(
                            self.peek(), plus=_is_op(self.peek(1), "+")
                        ),
                    )
                attr = _id_text(prev)
                if _is_op(token, "=") and attr is not None and attr.lower() == "dir":
                    dir_was_set = True
                    value_token = self.peek()
                    if value_token is not None and value_token.kind == "html":
                        dir_none = value_token.interior == "none"
                    else:
                        value = _id_text(value_token)
                        dir_none = value is not None and value.lower() == "none"
            prev = token
        return dir_none, dir_was_set


def _reject_keyword_preamble(span: str, kind: str) -> None:
    """Reject unquoted ``digraph`` / ``subgraph`` / ``strict`` before ``{``.

    The graph name and a header attribute list are outside the body. Dropping
    a keyword there compares equal to the graph with that keyword removed.
    A hyphen is not an id character, but it is not this hole, so the lexer
    skips it rather than rejecting ``digraph my-graph``.
    """
    seen_name = False
    depth = 0
    prev_lhs: str | None = None
    saw_eq = False
    tokens = _tokenize(span, skip_unsupported=True)
    for index, token in enumerate(tokens):
        if _is_op(token, "["):
            depth += 1
            saw_eq = False
            prev_lhs = None
            continue
        if _is_op(token, "]"):
            depth = max(0, depth - 1)
            saw_eq = False
            prev_lhs = None
            continue
        if _is_op(token, "="):
            if prev_lhs is not None and prev_lhs.lower() == "node":
                value = tokens[index + 1] if index + 1 < len(tokens) else None
                plus = index + 2 < len(tokens) and _is_op(tokens[index + 2], "+")
                raise _node_assignment_error(
                    prev_lhs, _shown_assignment_value(value, plus=plus)
                )
            saw_eq = True
            continue
        if _is_op(token, "+"):
            continue
        if token.kind == "id":
            shown = _rejected(token, _BARE_KEYWORDS)
            if shown is not None:
                if depth == 0 and not seen_name:
                    raise _keyword_graph_name_error(kind, shown)
                if saw_eq and prev_lhs:
                    raise _keyword_value_error(shown, prev_lhs)
                raise _keyword_id_error(shown)
            if depth == 0 and not seen_name:
                seen_name = True
            prev_lhs = token.text
            saw_eq = False
            continue
        if token.kind in {"qid", "html"}:
            if depth == 0 and not seen_name:
                seen_name = True
            saw_eq = False
            prev_lhs = None
            continue
        saw_eq = False
        prev_lhs = None


def _body_after(dot_text: str, header: re.Match[str]) -> str | None:
    """Comment-stripped interior after a matched header.

    Keywords in the preamble are rejected only when an opening brace exists.
    A braceless remainder keeps the caller's missing-braces error. The header
    is not matched again, and the body is not stripped again.
    """
    cleaned = _strip_comments(dot_text[header.end() :])
    start = _opening_brace(cleaned)
    if start is None:
        return None
    kind = header.group("kind")
    if kind is None:
        return None
    _reject_keyword_preamble(cleaned[:start], kind)
    return _brace_interior_from(cleaned, start)


def parse_dot(
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
    A unicode space does not separate a header keyword; write an ASCII space.
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
    header = _match_graph_header(dot_text)
    if header is None:
        raise ValueError(_MISSING_GRAPH)
    body = _body_after(dot_text, header)
    if body is None:
        raise ValueError(_MISSING_GRAPH)
    is_undirected_graph = (header.group("kind") or "").lower() == "graph"
    return _DotBodyParser(body, is_undirected_graph=is_undirected_graph).parse()


def parse_dag_digraph(
    text: str,
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]] | None:
    """Parse a dag ``digraph``, or return None for the edge-list path.

    ``pathmc.dag`` accepts either ``digraph { ... }`` or ``A->B, B->C``.
    Only an ASCII-stripped ``digraph`` prefix is DOT here; ``strict digraph``
    stays on the edge-list path. A leading non-ASCII space is not a header
    separator. Missing braces raise the dag message. Keyword and body errors
    raise the DOT reader's ``ValueError``.
    """
    ascii_s = text.strip(" \t\r\n")
    looks_like_header = ascii_s[:1].isspace() or ascii_s.lower().startswith("digraph")
    header = _match_graph_header(ascii_s) if looks_like_header else None
    if not ascii_s.lower().startswith("digraph"):
        return None
    if header is None:
        raise ValueError(_DAG_MISSING_BRACES)
    body = _body_after(ascii_s, header)
    if body is None:
        raise ValueError(_DAG_MISSING_BRACES)
    return _DotBodyParser(body, is_undirected_graph=False).parse()
