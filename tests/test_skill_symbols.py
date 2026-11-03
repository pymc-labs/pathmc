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
"""The shipped skill must not name a symbol the package does not have."""

from __future__ import annotations

from pathlib import Path

import pytest

from check_skill_symbols import missing_symbols

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "pathmc" / "skills" / "pathmc" / "SKILL.md"


def _present(_receiver: str, _name: str) -> bool:
    return True


def _absent(receiver: str, name: str) -> bool:
    return not (receiver == "m" and name == "missing")


def test_fenced_call_is_checked_and_prose_is_ignored() -> None:
    text = "Robins, 1986 and `C(store)` and pm.do() are not calls.\n```\nm.missing()\n```\n"
    misses = missing_symbols(text, _absent)
    assert [(miss.receiver, miss.name) for miss in misses] == [("m", "missing")]
    assert misses[0].line == 3


def test_inline_backtick_outside_a_fence_is_checked() -> None:
    text = "Use `adj.missing` before the fit.\n"
    misses = missing_symbols(text, lambda receiver, name: name != "missing")
    assert [(miss.receiver, miss.name, miss.line) for miss in misses] == [
        ("adj", "missing", 1)
    ]


def test_allowlist_skips_a_documented_non_public_name() -> None:
    text = "`pathmc.not_public` stays out of the guide on purpose.\n"
    misses = missing_symbols(
        text,
        lambda *_args: False,
        allowlist={("pathmc", "not_public"): "intentionally not public"},
    )
    assert misses == []


def test_blank_allowlist_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        missing_symbols(
            "`m.missing()`\n",
            _absent,
            allowlist={("m", "missing"): "  "},
        )


def test_a_missing_method_fails_against_the_package() -> None:
    from check_skill_symbols import live_resolve

    text = "`m.fit()` is real. `m.not_a_real_method()` is not.\n"
    misses = missing_symbols(text, live_resolve)
    assert [(miss.receiver, miss.name) for miss in misses] == [
        ("m", "not_a_real_method")
    ]
