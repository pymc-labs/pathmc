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

from check_skill_symbols import main, missing_symbols

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "pathmc" / "skills" / "pathmc" / "SKILL.md"


def _present(_receiver: str, _name: str, _call: bool = False) -> bool:
    return True


def _absent(receiver: str, name: str, _call: bool = False) -> bool:
    return not (receiver == "m" and name == "missing")


def test_fenced_call_is_checked_and_prose_is_ignored() -> None:
    text = "Robins, 1986 and `C(store)` and pm.do() are not calls.\n```\nm.missing()\n```\n"
    misses = missing_symbols(text, _absent)
    assert [(miss.receiver, miss.name) for miss in misses] == [("m", "missing")]
    assert misses[0].line == 3


def test_inline_backtick_outside_a_fence_is_checked() -> None:
    text = "Use `adj.missing` before the fit.\n"
    misses = missing_symbols(text, lambda receiver, name, _call: name != "missing")
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


def test_shipped_skill_names_only_real_symbols() -> None:
    assert main(["--skill", str(SKILL)]) == 0


def test_hook_reports_a_package_miss_on_the_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("`pathmc.nope()`\n", encoding="utf-8")
    assert main(["--skill", str(skill)]) == 1
    err = capsys.readouterr().err
    assert "pathmc.nope does not resolve on the pathmc package" in err
    assert "PathModel" not in err
    assert "ALLOWLIST" in err


def test_a_call_shadowed_by_its_submodule_is_not_callable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib

    import pathmc

    from check_skill_symbols import live_resolve

    monkeypatch.setattr(pathmc, "simulate", importlib.import_module("pathmc.simulate"))
    misses = missing_symbols("`pathmc.simulate(spec)`\n", live_resolve)
    assert [(miss.receiver, miss.name) for miss in misses] == [("pathmc", "simulate")]
    assert missing_symbols("`pathmc.simulate`\n", live_resolve) == []
    assert missing_symbols("`adj.formula`\n", live_resolve) == []
    skill = tmp_path / "SKILL.md"
    skill.write_text("`pathmc.simulate(spec)`\n", encoding="utf-8")
    assert main(["--skill", str(skill)]) == 1
    assert "is not callable" in capsys.readouterr().err
