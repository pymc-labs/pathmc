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
script compares that file with the committed one. It opens a pull request only
when the slowest group's estimated time would move by at least two minutes.
A missing timing is treated as the average, matching pytest-split, so a new
test is still assigned. Exit 0 means open a pull request. Exit 3 means the
move is below the bar. Exit 1 means the recording is not safe to publish.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SPLITS = 4
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


def least_duration_groups(
    nodeids: list[str], durations: dict[str, float], splits: int
) -> tuple[list[list[str]], list[float]]:
    """Assign tests the way pytest-split's ``least_duration`` algorithm does.

    A test with no stored duration gets the average of the stored durations
    for this collection. With no stored durations, every test weighs 1, which
    is pytest-split's even fallback. Returns groups in ``--group`` order and
    each group's estimated seconds.
    """
    if splits < 1:
        raise ValueError(f"splits must be at least 1, got {splits}")
    if not nodeids:
        raise ValueError("the timing file has no tests. A full run must record some.")
    relevant = [durations[nodeid] for nodeid in nodeids if nodeid in durations]
    average = sum(relevant) / len(relevant) if relevant else 1.0
    indexed = [
        (nodeid, durations.get(nodeid, average), index)
        for index, nodeid in enumerate(nodeids)
    ]
    # Stable sorts: name first, then longest duration, matching pytest-split.
    indexed.sort(key=lambda row: row[0])
    indexed.sort(key=lambda row: row[1], reverse=True)

    selected: list[list[tuple[str, int]]] = [[] for _ in range(splits)]
    totals = [0.0] * splits
    heap = [(0.0, index) for index in range(splits)]
    heapq.heapify(heap)
    for nodeid, duration, original_index in indexed:
        summed, group_index = heapq.heappop(heap)
        totals[group_index] = summed + duration
        selected[group_index].append((nodeid, original_index))
        heapq.heappush(heap, (totals[group_index], group_index))

    groups: list[list[str]] = []
    for group in selected:
        group.sort(key=lambda row: row[1])
        groups.append([nodeid for nodeid, _index in group])
    return groups, totals


def slowest_seconds(
    nodeids: list[str], durations: dict[str, float], splits: int
) -> float:
    """Return the estimated seconds of the slowest group."""
    _groups, totals = least_duration_groups(nodeids, durations, splits)
    return max(totals)


def drift_summary(
    *,
    old: dict[str, float],
    new: dict[str, float],
    splits: int,
    threshold: float,
) -> tuple[bool, str]:
    """Return whether to open a pull request, and the explanation."""
    nodeids = list(new)
    old_slowest = slowest_seconds(nodeids, old, splits)
    new_slowest = slowest_seconds(nodeids, new, splits)
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
    parser.add_argument("--splits", type=int, default=SPLITS)
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
        should_open, summary = drift_summary(
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
