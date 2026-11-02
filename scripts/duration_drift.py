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
"""Decide whether refreshed pytest-split timings should open a pull request.

The weekly job records a full per-commit run into a new durations file. This
script collects the same ``-m "not nightly"`` items CI runs and asks
pytest-split how those items pack under the old and new timings. It opens a
pull request only when the slowest group's estimated time would move by at
least two minutes. A missing timing is treated as the average, matching
pytest-split, so a new test is still assigned. Exit 0 means open a pull
request. Exit 3 means the move is below the bar. Exit 1 means the recording
is not safe to publish.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from pytest_split.algorithms import LeastDurationAlgorithm

THRESHOLD_SECONDS = 120.0
EXIT_OPEN = 0
EXIT_ERROR = 1
EXIT_QUIET = 3


def load_durations(path: Path) -> dict[str, float]:
    """Read a pytest-split durations file. A missing file is an empty store."""
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} is not a pytest-split durations object. Re-record timings "
            "with pytest --store-durations; do not open a pull request."
        )
    durations: dict[str, float] = {}
    for nodeid, value in raw.items():
        duration = float(value)
        if duration < 0:
            raise ValueError(
                f"{nodeid} has a negative duration in {path}. Re-record timings; "
                "do not open a pull request."
            )
        durations[str(nodeid)] = duration
    return durations


def junit_testcase_count(path: Path) -> int:
    """Return the number of testcases in a pytest junitxml report."""
    root = ET.parse(path).getroot()
    return len(root.findall(".//testcase"))


def collect_per_commit_items() -> list[Any]:
    """Collect the items ``make test`` and ``make test-shard`` select.

    Real pytest items are required. pytest-split breaks equal durations with
    ``str(item)``, which for a function test is ``<Function test_one>``, not
    the nodeid. A stand-in that stringifies to the nodeid packs new tests
    differently from CI.
    """
    import pytest

    collected: list[Any] = []

    class _Collector:
        def pytest_sessionfinish(self, session: Any) -> None:
            collected.extend(session.items)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = pytest.main(
            [
                "-m",
                "not nightly",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
            ],
            plugins=[_Collector()],
        )
    if code not in (0, 5) or not collected:
        detail = stderr.getvalue().strip() or stdout.getvalue().strip()
        tail = detail[-400:]
        raise ValueError(
            "Could not collect the per-commit tests"
            + (f": {tail}" if tail else "")
            + ". Fix the collection error; do not open a pull request."
        )
    return collected


def slowest_seconds(
    items: list[Any], durations: dict[str, float], splits: int
) -> float:
    """Return the estimated seconds of the slowest pytest-split group."""
    if splits < 1:
        raise ValueError(f"splits must be at least 1, got {splits}")
    if not items:
        raise ValueError(
            "the per-commit selection collected no tests. A full run must record some."
        )
    groups = LeastDurationAlgorithm()(splits, items, durations)
    return max(group.duration for group in groups)


def drift_summary(
    *,
    items: list[Any],
    old: dict[str, float],
    new: dict[str, float],
    splits: int,
    threshold: float,
) -> tuple[bool, str]:
    """Return whether to open a pull request, and the explanation."""
    old_slowest = slowest_seconds(items, old, splits)
    new_slowest = slowest_seconds(items, new, splits)
    delta = abs(new_slowest - old_slowest)
    minutes = threshold / 60
    explanation = (
        f"The slowest of the {splits} CI groups would change by "
        f"{delta / 60:.1f} minutes of estimated total test time "
        f"({old_slowest / 60:.1f} min to {new_slowest / 60:.1f} min). "
        "That is the sum pytest-split uses to pack the groups, not the wall clock."
    )
    if delta < threshold:
        return False, (
            f"No pull request. {explanation} That is under the {minutes:g} minute bar."
        )
    return True, (
        f"{explanation}\n\n"
        "This pull request was opened by the weekly timing job because that "
        f"change is at least {minutes:g} minutes. Merging it keeps the split "
        "fair. A failed run does not open a pull request, and this job cannot "
        "push straight to main."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_SECONDS)
    args = parser.parse_args(argv)
    try:
        old = load_durations(args.old)
        new = load_durations(args.new)
        if args.junit is not None:
            recorded = junit_testcase_count(args.junit)
            if recorded != len(new):
                raise ValueError(
                    f"The timing file has {len(new)} tests but the junit report "
                    f"has {recorded}. The recording is incomplete. Do not open "
                    "a pull request."
                )
        items = collect_per_commit_items()
        should_open, summary = drift_summary(
            items=items,
            old=old,
            new=new,
            splits=args.splits,
            threshold=args.threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(exc, file=sys.stderr)
        return EXIT_ERROR
    print(summary)
    return EXIT_OPEN if should_open else EXIT_QUIET


if __name__ == "__main__":
    raise SystemExit(main())
