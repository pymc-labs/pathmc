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
"""Assign per-commit test files to the CI jobs in ``.github/ci-groups/``.

The committed lists are a measured packing: group 1 is the fuzz file, whose
wall clock is the indivisible floor. A file that exists on the checkout but
is missing from those lists — a test added on ``main`` after the lists were
cut, for example — is appended to the group with the fewest files. Omitting
it would make the required check green while dropping coverage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

JOBS = 4

# pytest's default ``norecursedirs``, plus this repo's gitignored scratch tree.
# ``.scratch`` is never on the CI checkout; collecting it locally would move
# committed files between groups.
_SKIP_DIRS = {
    ".scratch",
    "CVS",
    "_darcs",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
}


def discover_test_files(root: Path) -> list[str]:
    """Return repo-relative paths pytest would collect as test modules."""
    found: list[str] = []
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _SKIP_DIRS and not name.startswith(".")
        ]
        for name in filenames:
            if not _is_test_module(name):
                continue
            found.append((dirpath / name).relative_to(root).as_posix())
    return sorted(found)


def load_groups(root: Path, jobs: int = JOBS) -> list[list[str]]:
    """Read the committed group lists. A path may appear in only one list."""
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    groups: list[list[str]] = []
    owner: dict[str, int] = {}
    for number in range(1, jobs + 1):
        path = root / ".github" / "ci-groups" / f"group-{number}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Commit that group list before running CI, "
                "or lower --jobs to the number of lists that exist."
            )
        lines = [
            line
            for raw in path.read_text().splitlines()
            if (line := raw.strip()) and not line.startswith("#")
        ]
        for listed in lines:
            previous = owner.get(listed)
            if previous is not None:
                raise ValueError(
                    f"{listed} is listed in CI group {previous} and group "
                    f"{number}. Keep each test file in one group."
                )
            owner[listed] = number
        groups.append(lines)
    return groups


def assign_groups(
    test_files: list[str], groups: list[list[str]]
) -> tuple[list[list[str]], list[str]]:
    """Keep known files in their lists and append unknown files to the smallest.

    A listed file that is no longer on disk is omitted. Returns the assigned
    groups and the paths that were not in any committed list, in the order
    they were placed.
    """
    if not groups:
        raise ValueError("groups must contain at least one list")
    present = set(test_files)
    assigned = [[path for path in group if path in present] for group in groups]
    known = {path for group in groups for path in group}
    extras = sorted(present - known)
    for path in extras:
        index = min(range(len(assigned)), key=lambda i: (len(assigned[i]), i))
        assigned[index].append(path)
    return assigned, extras


def _is_test_module(name: str) -> bool:
    return name.endswith(".py") and (
        name.startswith("test_") or name.endswith("_test.py")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the working directory.",
    )
    parser.add_argument("--jobs", type=int, default=JOBS)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.group <= args.jobs:
        print(
            f"--group must be between 1 and {args.jobs}, got {args.group}.",
            file=sys.stderr,
        )
        return 1
    root = args.root.resolve()
    try:
        groups = load_groups(root, args.jobs)
        assigned, extras = assign_groups(discover_test_files(root), groups)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    files = assigned[args.group - 1]
    if not files:
        print(
            f"CI group {args.group} has no test files. Move a listed file "
            f"into .github/ci-groups/group-{args.group}.txt, or add a test "
            "file so it can land on this group.",
            file=sys.stderr,
        )
        return 1
    for path in extras:
        if path in files:
            print(
                f"assigned {path} to group {args.group}; it is not in "
                ".github/ci-groups/ yet. Commit it there if the placement "
                "should stick.",
                file=sys.stderr,
            )
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
