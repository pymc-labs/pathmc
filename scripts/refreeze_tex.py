"""Rewrite generated model-equation TeX inside Quarto freeze artifacts.

The escaping fix in ``pathmc.introspect`` changes what ``equations()`` and
``priors()`` emit, so the committed ``_freeze/**/execute-results/html.json``
caches now disagree with the code that produced them. A full ``quarto
render`` would reconcile them, but it re-executes the notebooks: the three
pages refrozen in the parent commit moved 1343 lines across 36 files, nearly
all of it MCMC nondeterminism (figure PNGs rewritten byte-for-byte, pymc
graph SVGs relaid out). That churn buries the few dozen characters that
actually change.

This applies the same transformation the patched code applies, directly to
the cached TeX, with no notebook execution. The ``hash`` field of a freeze
artifact is derived from the source document, not the result payload (the
parent commit refroze pages while leaving it untouched), so editing
``result`` in place keeps the cache valid.

The transformation is defined by group, mirroring the code exactly:

* a subscript group ``_{...}`` holds indices, so a bare ``_`` becomes ``,``
  (:func:`pathmc.introspect._latex_index`);
* an upright-text group ``\\mathrm{}`` / ``\\operatorname{}`` / ``\\text{}``
  shows a name as written, so a bare ``_`` becomes ``\\_``
  (:func:`pathmc.introspect._latex_escape`).

Groups nest (``f_{\\mathrm{hsgp}}``), so the walk is recursive and the
innermost group wins. ``scripts/verify_refreeze_tex.py`` checks the result
against the real escaper rather than trusting this description.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Groups whose contents are indices: underscores separate them.
_INDEX_OPENERS = ("_{",)

# Groups whose contents are shown verbatim: underscores are literal.
_ESCAPE_OPENERS = (r"\mathrm{", r"\operatorname{", r"\text{")

_ALIGNED = re.compile(r"\\begin\{aligned\}.*?\\end\{aligned\}", re.S)

# A bare ``_`` is one that neither is escaped nor opens a subscript group.
_BARE_UNDERSCORE = re.compile(r"(?<!\\)_(?!\{)")


def _split_group(text: str, open_at: int) -> tuple[str, int]:
    """Return the balanced ``{...}`` body starting at ``open_at`` and its end.

    ``open_at`` indexes the ``{``. The returned index is one past the
    matching ``}``.
    """
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : i], i + 1
    raise ValueError(f"unbalanced brace group at offset {open_at}")


def _rewrite(text: str, mode: str = "escape") -> str:
    r"""Rewrite bare underscores in ``text`` according to enclosing groups.

    ``mode`` is the rule inherited from the enclosing group: ``"index"``
    inside ``_{...}``, ``"escape"`` everywhere else.

    Escaping is the default rather than a no-op because the emitted TeX has
    bare underscores at top level: :func:`~pathmc.introspect._latexify_prior`
    renders ``Normal(mu_alpha, sigma_alpha)`` as ``\text{Normal}(...)`` with
    the arguments *outside* the ``\text{}`` group, and the patched code
    escapes them there. Leaving top level alone would silently disagree with
    the real escaper on exactly those lines.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        for opener in _ESCAPE_OPENERS + _INDEX_OPENERS:
            if text.startswith(opener, i):
                brace = i + len(opener) - 1
                body, end = _split_group(text, brace)
                inner = "index" if opener in _INDEX_OPENERS else "escape"
                out.append(opener + _rewrite(body, inner) + "}")
                i = end
                break
        else:
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                # Consume a control sequence's backslash together with the
                # character it escapes. Without this an existing ``\_`` would
                # be seen as a bare ``_`` and escaped a second time, so the
                # pass would not be idempotent.
                out.append(text[i : i + 2])
                i += 2
                continue
            if ch == "_" and not text.startswith("_{", i):
                out.append("," if mode == "index" else r"\_")
            else:
                out.append(ch)
            i += 1
    return "".join(out)


def rewrite_tex(block: str) -> str:
    """Apply the group rules to one stretch of generated TeX."""
    return _rewrite(block)


def rewrite_markdown(markdown: str) -> tuple[str, int]:
    """Rewrite every ``aligned`` block; return the text and how many changed."""
    changed = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal changed
        new = rewrite_tex(m.group(0))
        if new != m.group(0):
            changed += 1
        return new

    return _ALIGNED.sub(sub, markdown), changed


def rewrite_file(path: Path) -> int:
    """Rewrite one freeze artifact in place; return the blocks changed.

    The file is spliced, not re-serialised: a ``json.load``/``json.dump``
    round-trip does not reproduce Quarto's output byte for byte, which would
    reformat the whole artifact and hide the change. Locating the encoded
    ``markdown`` value in the raw text and substituting it leaves every other
    byte untouched.
    """
    raw = path.read_text(encoding="utf-8")
    markdown = json.loads(raw)["result"]["markdown"]
    new_markdown, changed = rewrite_markdown(markdown)
    if not changed:
        return 0

    for kwargs in ({"ensure_ascii": False}, {"ensure_ascii": True}):
        encoded = json.dumps(markdown, **kwargs)
        if encoded in raw:
            break
    else:
        raise ValueError(f"cannot locate encoded markdown value in {path}")

    path.write_text(
        raw.replace(encoded, json.dumps(new_markdown, **kwargs), 1),
        encoding="utf-8",
    )
    return changed


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("_freeze")
    total_files = 0
    total_blocks = 0
    for path in sorted(root.glob("**/execute-results/html.json")):
        changed = rewrite_file(path)
        if changed:
            total_files += 1
            total_blocks += changed
            print(f"{path}: {changed} block(s)")
    print(f"\n{total_blocks} block(s) across {total_files} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
