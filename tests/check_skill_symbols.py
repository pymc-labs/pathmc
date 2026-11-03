# Copyright 2025 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail when the shipped skill names a symbol the package does not have."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

SKILL = Path("pathmc/skills/pathmc/SKILL.md")
ALLOWLIST: dict[tuple[str, str], str] = {}
TARGETS = {
    "pathmc": "the pathmc package",
    "m": "PathModel",
    "adj": "AdjustmentModel",
}
_FENCE = re.compile(r"^```[^\n]*\n(.*?)```[ \t]*$", re.MULTILINE | re.DOTALL)
_INLINE = re.compile(r"`([^`\n]+)`")
_CALL = re.compile(
    r"\b(pathmc|m|adj)\.([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()"
    r"|\b(pathmc|m|adj)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)


@dataclass(frozen=True)
class Miss:
    """One skill reference that does not resolve."""

    line: int
    receiver: str
    name: str
    snippet: str


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    covered = [(fence.start(), fence.end()) for fence in _FENCE.finditer(text)]
    for fence in _FENCE.finditer(text):
        body_start = fence.start(1)
        body = fence.group(1)
        spans.append((body_start, body_start + len(body), body))
    for inline in _INLINE.finditer(text):
        start, end = inline.span(1)
        if any(left <= start and end <= right for left, right in covered):
            continue
        spans.append((start, end, inline.group(1)))
    return spans


def missing_symbols(
    text: str,
    resolve: Callable[[str, str, bool], bool],
    allowlist: Mapping[tuple[str, str], str] | None = None,
) -> list[Miss]:
    """Return unresolved ``pathmc.``, ``m.``, and ``adj.`` references.

    Parameters
    ----------
    text
        Skill markdown.
    resolve
        ``(receiver, name, call) -> True`` when the reference is valid.
        A call is valid only when the attribute is callable.
    allowlist
        ``(receiver, name)`` to a non-empty reason. A blank reason is an error.
    """
    reasons = dict(allowlist or {})
    for key, reason in reasons.items():
        if not reason.strip():
            raise ValueError(f"allowlist entry {key!r} needs a non-empty reason")
    found: list[Miss] = []
    for start, _end, snippet in _spans(text):
        for match in _CALL.finditer(snippet):
            receiver = match.group(1) or match.group(3)
            name = match.group(2) or match.group(4)
            call = match.group(3) is not None
            if (receiver, name) in reasons or resolve(receiver, name, call):
                continue
            found.append(
                Miss(
                    line=_line_of(text, start + match.start()),
                    receiver=receiver,
                    name=name,
                    snippet=match.group(0),
                )
            )
    return found


def _target(receiver: str) -> object:
    import pathmc
    from pathmc import AdjustmentModel, PathModel

    return {"pathmc": pathmc, "m": PathModel, "adj": AdjustmentModel}[receiver]


def live_resolve(receiver: str, name: str, call: bool = False) -> bool:
    """Resolve a documented attribute against the installed package.

    A call has to be callable. A bare attribute only has to exist, so a
    property used without parentheses still counts. ``hasattr`` alone
    accepts ``pathmc.simulate`` after the function export is removed,
    because that name is then the submodule.
    """
    target = _target(receiver)
    if not hasattr(target, name):
        return False
    if not call:
        return True
    return callable(getattr(target, name))


def main(argv: list[str] | None = None) -> int:
    """Check the shipped skill and print one line per missing symbol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, default=SKILL)
    args = parser.parse_args(argv)
    text = args.skill.read_text(encoding="utf-8")
    try:
        misses = missing_symbols(text, live_resolve, ALLOWLIST)
    except ValueError as exc:
        print(
            f"{args.skill}: {exc}. Give ALLOWLIST a non-empty reason.",
            file=sys.stderr,
        )
        return 1
    if not misses:
        return 0
    print(
        f"{args.skill}: names a symbol the package does not have. "
        "Update the skill, or add the name to ALLOWLIST in "
        "tests/check_skill_symbols.py with a non-empty reason.",
        file=sys.stderr,
    )
    for miss in misses:
        present = hasattr(_target(miss.receiver), miss.name)
        problem = (
            "is not callable" if present and "(" in miss.snippet else "does not resolve"
        )
        print(
            f"  line {miss.line}: {miss.receiver}.{miss.name} "
            f"{problem} on {TARGETS[miss.receiver]}. "
            f"Snippet: {miss.snippet[:120]}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
