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
"""A test file missing from the committed lists must still be assigned."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_test_groups.py"
ROOT = SCRIPT.parents[1]


@pytest.fixture(scope="module")
def groups_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ci_test_groups", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_groups(root: Path, groups: list[list[str]]) -> None:
    directory = root / ".github" / "ci-groups"
    directory.mkdir(parents=True)
    for number, files in enumerate(groups, start=1):
        (directory / f"group-{number}.txt").write_text(
            "\n".join(files) + ("\n" if files else "")
        )


def test_a_file_missing_from_the_lists_joins_the_smallest_group(groups_mod):
    groups = [
        ["tests/test_fuzz_consistency.py"],
        ["tests/test_a.py", "tests/test_b.py"],
        ["tests/test_c.py", "tests/test_d.py"],
    ]
    present = [
        "tests/test_a.py",
        "tests/test_b.py",
        "tests/test_c.py",
        "tests/test_d.py",
        "tests/test_fuzz_consistency.py",
        "tests/test_new.py",
        "tests/test_newer.py",
    ]
    assigned, extras = groups_mod.assign_groups(present, groups)
    assert extras == ["tests/test_new.py", "tests/test_newer.py"]
    assert assigned[0] == [
        "tests/test_fuzz_consistency.py",
        "tests/test_new.py",
        "tests/test_newer.py",
    ]
    assert "tests/test_new.py" not in assigned[1]
    assert "tests/test_new.py" not in assigned[2]


def test_a_deleted_listed_file_is_omitted(groups_mod):
    assigned, extras = groups_mod.assign_groups(
        ["tests/test_a.py"],
        [["tests/test_gone.py"], ["tests/test_a.py"]],
    )
    assert assigned == [[], ["tests/test_a.py"]]
    assert extras == []


def test_a_tie_goes_to_the_lower_group_index(groups_mod):
    assigned, extras = groups_mod.assign_groups(
        ["tests/test_a.py", "tests/test_b.py", "tests/test_new.py"],
        [["tests/test_a.py"], ["tests/test_b.py"]],
    )
    assert extras == ["tests/test_new.py"]
    assert assigned[0] == ["tests/test_a.py", "tests/test_new.py"]
    assert assigned[1] == ["tests/test_b.py"]


def test_a_path_listed_in_two_groups_is_rejected(groups_mod, tmp_path):
    _write_groups(
        tmp_path,
        [["tests/test_a.py"], ["tests/test_a.py"]],
    )
    with pytest.raises(ValueError, match="group 1 and group 2"):
        groups_mod.load_groups(tmp_path, jobs=2)


def test_cli_prints_one_path_per_line_and_names_an_extra(tmp_path):
    _write_groups(tmp_path, [["tests/test_a.py"], ["tests/test_b.py"]])
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "tests" / "test_b.py").write_text("def test_b():\n    pass\n")
    (tmp_path / "tests" / "test_new.py").write_text("def test_new():\n    pass\n")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--jobs",
            "2",
            "--group",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["tests/test_a.py", "tests/test_new.py"]
    assert "tests/test_new.py" in result.stderr
    assert "group 1" in result.stderr


def test_cli_rejects_an_empty_group(tmp_path):
    _write_groups(tmp_path, [["tests/test_gone.py"]])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--jobs",
            "1",
            "--group",
            "1",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "no test files" in result.stderr
    assert "group-1.txt" in result.stderr


def test_this_checkout_assigns_every_test_file_once(groups_mod):
    discovered = groups_mod.discover_test_files(ROOT)
    assigned, _extras = groups_mod.assign_groups(
        discovered, groups_mod.load_groups(ROOT)
    )
    flat = [path for group in assigned for path in group]
    assert sorted(flat) == discovered
    assert len(flat) == len(set(flat))
    assert "tests/test_ci_test_groups.py" in flat
    assert all(group for group in assigned)
