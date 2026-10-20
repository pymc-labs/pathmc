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
"""Validate a committed PR candidate against its current target branch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _capture(root: Path, *args: str) -> str:
    return subprocess.run(
        args, cwd=root, text=True, check=True, capture_output=True
    ).stdout.strip()


def _clean_head(root: Path) -> str:
    if _capture(root, "git", "status", "--porcelain"):
        raise ValueError(
            "The checkout has uncommitted or untracked changes. Commit or stash "
            "them before make check-pr; ignored .scratch files are allowed."
        )
    return _capture(root, "git", "rev-parse", "HEAD")


def _pull(root: Path, repository: str, number: int) -> dict:
    data = json.loads(_capture(root, "gh", "api", f"repos/{repository}/pulls/{number}"))
    if data["state"] != "open":
        raise ValueError("This PR is not open. Select an open PR to validate.")
    return data


def _fetch(root: Path, url: str, ref: str) -> str:
    _capture(root, "git", "fetch", "--no-tags", url, ref)
    return _capture(root, "git", "rev-parse", "FETCH_HEAD")


def _preview(root: Path, base: str, head: str) -> str:
    merged = subprocess.run(
        ["git", "merge-tree", "--write-tree", base, head],
        cwd=root,
        text=True,
        capture_output=True,
    )
    if merged.returncode:
        remedy = (
            "Resolve the merge conflicts and rerun make check-pr."
            if merged.returncode == 1
            else "Check that Git supports merge-tree --write-tree, fetch the inputs, and rerun."
        )
        raise ValueError(
            f"Cannot compute the merge candidate. {remedy}\n{merged.stdout}{merged.stderr}"
        )
    tree = merged.stdout.splitlines()[0]
    return _capture(
        root,
        "git",
        "-c",
        "user.name=Local CI validation",
        "-c",
        "user.email=ci-preview@localhost.invalid",
        "commit-tree",
        tree,
        "-p",
        base,
        "-p",
        head,
        "-m",
        "Local CI merge preview (not published)",
    )


def _validate(root: Path, number: int | None, branch: str | None, remote: str) -> int:
    head = _clean_head(root)
    remote_head = None
    if number is not None:
        repository = json.loads(
            _capture(root, "gh", "repo", "view", "--json", "nameWithOwner,url")
        )
        name, url = repository["nameWithOwner"], repository["url"]
        pull = _pull(root, name, number)
        remote_head = _fetch(root, url, f"refs/pull/{number}/head")
        if remote_head != pull["head"]["sha"]:
            raise ValueError("The PR head changed while fetching. Rerun make check-pr.")
        descendant = subprocess.run(
            ["git", "merge-base", "--is-ancestor", remote_head, head], cwd=root
        )
        if descendant.returncode:
            raise ValueError(
                "This checkout is not the PR head or a committed descendant. "
                "Check out the PR branch before running make check-pr."
            )
        branch = pull["base"]["ref"]
    else:
        name, url = remote, _capture(root, "git", "remote", "get-url", remote)
    base_ref = f"refs/heads/{branch}"
    base = _fetch(root, url, base_ref)
    mode = "local merge preview"
    merge = None
    if number is not None and head == remote_head:
        try:
            candidate = _fetch(root, url, f"refs/pull/{number}/merge")
            parents = _capture(root, "git", "rev-list", "--parents", "-n1", candidate)
            if parents.split()[1:] == [base, head]:
                merge, mode = candidate, "GitHub merge ref"
        except subprocess.CalledProcessError:
            # GitHub may not have regenerated its merge ref for a new base yet.
            pass
    if merge is None:
        merge = _preview(root, base, head)
    scratch = root / ".scratch" / "ci-preflight"
    scratch.mkdir(parents=True, exist_ok=True)
    report = {
        "pr": number,
        "repository": name,
        "target_branch": branch,
        "mode": mode,
        "head": head,
        "base": base,
        "merge": merge,
        "tree": _capture(root, "git", "rev-parse", f"{merge}^{{tree}}"),
        "status": "running",
    }
    report_path = scratch / "last-run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    result = 1
    try:
        with tempfile.TemporaryDirectory(prefix="pr-", dir=scratch) as temporary:
            worktree = Path(temporary) / "candidate"
            _capture(root, "git", "worktree", "add", "--detach", str(worktree), merge)
            try:
                checked = subprocess.run(["make", "check"], cwd=worktree)
                if checked.returncode:
                    return checked.returncode
                if _clean_head(worktree) != merge:
                    raise ValueError(
                        "The merge candidate changed during validation. Rerun check-pr."
                    )
                if _clean_head(root) != head:
                    raise ValueError(
                        "The local candidate changed. Rerun make check-pr."
                    )
                if _fetch(root, url, base_ref) != base:
                    raise ValueError("The target branch advanced. Rerun make check-pr.")
                if number is not None:
                    latest = _pull(root, name, number)
                    if (
                        latest["head"]["sha"] != remote_head
                        or latest["base"]["ref"] != branch
                    ):
                        raise ValueError(
                            "The pushed PR head or target changed. Rerun make check-pr."
                        )
                result = 0
            finally:
                _capture(root, "git", "worktree", "remove", "--force", str(worktree))
        return result
    finally:
        report["status"] = "passed" if result == 0 else "failed"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Validation record: {report_path}", flush=True)


def main() -> int:
    """Run the shared lint/test gates in an isolated PR merge candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pr", type=int)
    selection.add_argument("--base", help="Target branch for a new, unpublished PR")
    parser.add_argument(
        "--remote", default="origin", help="Remote for --base (default: origin)"
    )
    args = parser.parse_args()
    try:
        commands = ("git", "gh", "make") if args.pr is not None else ("git", "make")
        for command in commands:
            if shutil.which(command) is None:
                raise ValueError(
                    f"{command} is required. Install it and rerun check-pr."
                )
        root = Path(_capture(Path.cwd(), "git", "rev-parse", "--show-toplevel"))
        return _validate(root, args.pr, args.base, args.remote)
    except (ValueError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        print(f"check-pr: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
