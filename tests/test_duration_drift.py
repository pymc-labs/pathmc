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
"""The CI split stays a partition, and the weekly job stays quiet under 2 minutes."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
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


def test_groups_partition_every_test_once() -> None:
    nodeids = [f"tests/test_mod.py::test_{index}" for index in range(7)]
    durations = {nodeid: float(index + 1) for index, nodeid in enumerate(nodeids)}
    groups, totals = drift.least_duration_groups(nodeids, durations, 4)

    assigned = [nodeid for group in groups for nodeid in group]
    assert sorted(assigned) == sorted(nodeids)
    assert len(assigned) == len(set(assigned))
    assert len(groups) == 4
    assert len(totals) == 4
    assert totals[0] == max(totals)


def test_missing_timing_still_assigns_the_test() -> None:
    nodeids = ["tests/test_a.py::test_known", "tests/test_b.py::test_new"]
    groups, _totals = drift.least_duration_groups(
        nodeids, {"tests/test_a.py::test_known": 10.0}, 2
    )
    assigned = [nodeid for group in groups for nodeid in group]
    assert assigned.count("tests/test_b.py::test_new") == 1


def test_stale_timing_does_not_change_the_slowest_group() -> None:
    nodeids = ["tests/test_a.py::test_one"]
    without_stale = drift.slowest_seconds(
        nodeids, {"tests/test_a.py::test_one": 30.0}, 2
    )
    with_stale = drift.slowest_seconds(
        nodeids,
        {
            "tests/test_a.py::test_one": 30.0,
            "tests/test_gone.py::test_old": 10_000.0,
        },
        2,
    )
    assert with_stale == without_stale


def test_two_minute_bar_is_the_slowest_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_incomplete_recording_does_not_open_a_pull_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
    ])
    assert code == 1
    assert "incomplete" in capsys.readouterr().err


def test_assignment_matches_pytest_split(tmp_path: Path) -> None:
    durations = {
        "test_a.py::test_one": 10.0,
        "test_b.py::test_one": 9.0,
        "test_c.py::test_one": 8.0,
        "test_d.py::test_one": 1.0,
        "test_e.py::test_one": 1.0,
        "test_f.py::test_one": 1.0,
    }
    for name in durations:
        path = tmp_path / name.split("::")[0]
        path.write_text("def test_one():\n    pass\n")
    durations_path = tmp_path / ".test_durations"
    durations_path.write_text(json.dumps(durations))
    nodeids = list(durations)
    expected, expected_totals = drift.least_duration_groups(nodeids, durations, 2)

    selected: list[list[str]] = []
    printed: list[float] = []
    for group in (1, 2):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--noconftest",
                f"--rootdir={tmp_path}",
                f"--confcutdir={tmp_path}",
                "-o",
                "addopts=",
                "-q",
                "--collect-only",
                "--splits",
                "2",
                "--group",
                str(group),
                "--splitting-algorithm",
                "least_duration",
                "--durations-path",
                str(durations_path),
                str(tmp_path),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        selected.append([
            line.strip() for line in result.stdout.splitlines() if "::" in line
        ])
        for line in result.stdout.splitlines():
            if "estimated duration:" in line:
                raw = line.split("estimated duration:", 1)[1]
                printed.append(float(raw.strip().removesuffix(")").removesuffix("s")))

    assert [set(group) for group in selected] == [set(group) for group in expected]
    assert printed == pytest.approx(expected_totals)


def test_local_make_test_stops_at_the_first_failure() -> None:
    # The shard job passes PYTEST_FAILFAST= and PYTEST_ARGS=... on the make
    # command line. Make exports both the variables and MAKEFLAGS, so a child
    # make would otherwise copy the shard command.
    env = os.environ.copy()
    for name in ("PYTEST_FAILFAST", "PYTEST_ARGS", "MAKEFLAGS", "MAKEOVERRIDES"):
        env.pop(name, None)
    result = subprocess.run(
        ["make", "-n", "test"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert " -x " in result.stdout
    assert "--splits" not in result.stdout


def test_ci_can_run_a_shard_without_stopping_early() -> None:
    result = subprocess.run(
        [
            "make",
            "-n",
            "test",
            "PYTEST_FAILFAST=",
            "PYTEST_ARGS=--splits 4 --group 1 --splitting-algorithm least_duration --cov-fail-under=0",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert " -x " not in result.stdout
    assert "--splits" in result.stdout
    assert "--cov-fail-under=0" in result.stdout
    assert "--dist loadscope" in result.stdout


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
