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
"""The weekly job stays quiet under 2 minutes and does not collect on a bad record."""

from __future__ import annotations
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "duration_drift.py"


def _load():
    spec = importlib.util.spec_from_file_location("duration_drift", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drift = _load()


class _Item:
    """Enough of a pytest item for the library: nodeid lookup, str() tie-break."""

    def __init__(self, nodeid: str) -> None:
        self.nodeid = nodeid

    def __str__(self) -> str:
        return f"<Function {self.nodeid.rsplit('::', 1)[-1]}>"


def test_missing_timing_is_packed_at_the_average() -> None:
    items = [
        _Item("tests/test_a.py::test_known"),
        _Item("tests/test_b.py::test_new"),
    ]
    total = drift.slowest_seconds(items, {"tests/test_a.py::test_known": 10.0}, 1)
    assert total == 20.0


def test_stale_timing_does_not_change_the_slowest_group() -> None:
    items = [_Item("tests/test_a.py::test_one")]
    without_stale = drift.slowest_seconds(items, {"tests/test_a.py::test_one": 30.0}, 2)
    with_stale = drift.slowest_seconds(
        items,
        {
            "tests/test_a.py::test_one": 30.0,
            "tests/test_gone.py::test_old": 10_000.0,
        },
        2,
    )
    assert with_stale == without_stale


def test_two_minute_bar_is_the_slowest_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    items = [
        _Item("tests/test_a.py::test_one"),
        _Item("tests/test_b.py::test_two"),
    ]
    monkeypatch.setattr(drift, "collect_per_commit_items", lambda: items)
    old = {"tests/test_a.py::test_one": 100.0, "tests/test_b.py::test_two": 100.0}
    moved = {"tests/test_a.py::test_one": 220.0, "tests/test_b.py::test_two": 100.0}
    quiet = {"tests/test_a.py::test_one": 219.0, "tests/test_b.py::test_two": 100.0}
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(json.dumps(old))

    new_path.write_text(json.dumps(quiet))
    assert (
        drift.main(["--old", str(old_path), "--new", str(new_path), "--splits", "2"])
        == 3
    )
    assert "No pull request" in capsys.readouterr().out

    new_path.write_text(json.dumps(moved))
    assert (
        drift.main(["--old", str(old_path), "--new", str(new_path), "--splits", "2"])
        == 0
    )
    message = capsys.readouterr().out
    assert "at least 2 minutes" in message
    assert "push straight to main" in message


def test_incomplete_recording_does_not_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def collect() -> list[_Item]:
        raise AssertionError("collected the suite")

    monkeypatch.setattr(drift, "collect_per_commit_items", collect)
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    junit = tmp_path / "junit.xml"
    old_path.write_text("{}")
    new_path.write_text(json.dumps({"tests/test_a.py::test_one": 5.0}))
    junit.write_text(
        '<?xml version="1.0" ?><testsuite>'
        '<testcase classname="tests.test_a" name="test_one"/>'
        '<testcase classname="tests.test_b" name="test_two"/>'
        "</testsuite>"
    )
    code = drift.main([
        "--old",
        str(old_path),
        "--new",
        str(new_path),
        "--junit",
        str(junit),
        "--splits",
        "4",
    ])
    assert code == 1
    assert "incomplete" in capsys.readouterr().err


def test_test_shard_requires_group_and_splits() -> None:
    # CI exports GROUP and SPLITS. Make imports them, so an uncleared
    # environment runs a nested shard instead of the usage guard.
    env = os.environ.copy()
    env["GROUP"] = "1"
    env["SPLITS"] = "4"
    env.pop("GROUP")
    env.pop("SPLITS")
    try:
        result = subprocess.run(
            ["make", "test-shard"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "make test-shard started pytest; clear GROUP and SPLITS before the guard"
        ) from exc
    assert result.returncode != 0
    assert "Usage: make test-shard" in result.stdout + result.stderr
    assert "pytest" not in result.stdout


def test_weekly_split_count_matches_the_ci_matrix() -> None:
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    weekly = (ROOT / ".github/workflows/refresh-test-durations.yml").read_text()
    matrix = re.search(r"group:\s*\[([0-9,\s]+)\]", ci)
    splits = re.search(r"--splits\s+(\d+)", weekly)
    assert matrix is not None and splits is not None
    groups = [part.strip() for part in matrix.group(1).split(",") if part.strip()]
    assert int(splits.group(1)) == len(groups)


def test_timing_record_runs_the_full_selection() -> None:
    result = subprocess.run(
        ["make", "-n", "record-test-durations"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert " -x " not in result.stdout
    assert "--store-durations" in result.stdout
    assert "--junitxml" in result.stdout
    assert "not nightly" in result.stdout
    assert "--splits" not in result.stdout
