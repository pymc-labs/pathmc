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
"""The AGENTS.md module tree must name files that exist."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "AGENTS.md"
_FENCE = re.compile(r"^```[^\n]*\n(.*?)```[ \t]*$", re.MULTILINE | re.DOTALL)


def listed_module_paths(text: str) -> list[str]:
    """Return ``pathmc/<file>.py`` rows from the module-structure fence."""
    for fence in _FENCE.finditer(text):
        lines = fence.group(1).splitlines()
        if not lines or lines[0].strip() != "pathmc/":
            continue
        paths: list[str] = []
        for line in lines[1:]:
            match = re.fullmatch(r"\s*(\w+\.py)\s*(?:#.*)?", line)
            if match:
                paths.append(f"pathmc/{match.group(1)}")
        return paths
    raise AssertionError("AGENTS.md has no pathmc/ module tree")


def test_named_module_files_exist() -> None:
    paths = listed_module_paths(AGENTS.read_text(encoding="utf-8"))
    missing = [path for path in paths if not (ROOT / path).is_file()]
    assert paths
    assert missing == []
