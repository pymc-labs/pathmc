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
"""Exercise merge preflight against real local Git histories and fake CI tools."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_pr.py"


def _git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit(root, filename, text):
    (root / filename).write_text(text)
    _git(root, "add", filename)
    _git(root, "commit", "-m", filename)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path):
    root, remote, binaries = (
        tmp_path / name for name in ("checkout", "origin.git", "bin")
    )
    root.mkdir()
    binaries.mkdir()
    _git(tmp_path, "init", "--bare", str(remote))
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@localhost.invalid")
    _git(root, "config", "core.hooksPath", "/dev/null")
    _commit(root, ".gitignore", ".scratch/\n")
    _commit(root, "shared.txt", "initial\n")
    _git(root, "switch", "-c", "feature")
    head = _commit(root, "feature.txt", "feature\n")
    _git(root, "switch", "main")
    base = _commit(root, "target.txt", "target\n")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "origin", "main", "feature:refs/pull/1/head")
    _git(root, "switch", "--detach", base)
    _git(root, "merge", "--no-ff", "feature", "-m", "GitHub merge candidate")
    merge = _git(root, "rev-parse", "HEAD")
    _git(root, "push", "origin", "HEAD:refs/pull/1/merge")
    _git(root, "switch", "feature")
    result, control = tmp_path / "result.json", tmp_path / "control.json"
    control.write_text("{}")
    tools = {
        "gh": f"""
import json, subprocess, sys
from pathlib import Path
if sys.argv[1:3] == ['repo', 'view']:
    print(json.dumps({{'nameWithOwner': 'test/pathmc', 'url': {str(remote)!r}}}))
else:
    head = subprocess.check_output(['git', '--git-dir', {str(remote)!r}, 'rev-parse', 'refs/pull/1/head'], text=True).strip()
    branch = json.loads(Path({str(control)!r}).read_text()).get('base_ref', 'main')
    print(json.dumps({{'state': 'open', 'head': {{'sha': head}}, 'base': {{'ref': branch}}}}))
""",
        "make": f"""
import json, os, subprocess, sys
from pathlib import Path
info = {{'cwd': os.getcwd(), 'args': sys.argv[1:], 'files': sorted(p.name for p in Path.cwd().iterdir()), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}}
Path({str(result)!r}).write_text(json.dumps(info))
control = json.loads(Path({str(control)!r}).read_text())
for key, ref in [('base', 'refs/heads/main'), ('remote_head', 'refs/pull/1/head')]:
    if key in control:
        subprocess.check_call(['git', '--git-dir', {str(remote)!r}, 'update-ref', ref, control[key]])
if control.get('dirty'):
    Path('feature.txt').write_text('validation modified the candidate')
if 'retarget' in control:
    control['base_ref'] = control['retarget']
    Path({str(control)!r}).write_text(json.dumps(control))
sys.exit(control.get('exit', 0))
""",
    }
    for name, code in tools.items():
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n" + code)
        path.chmod(0o755)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(args or ("--pr", "1"))],
            cwd=root,
            env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"]},
            text=True,
            capture_output=True,
        )

    return SimpleNamespace(
        root=root,
        remote=remote,
        head=head,
        base=base,
        merge=merge,
        result=result,
        control=control,
        run=run,
    )


def _assert_cleaned(history, head):
    assert _git(history.root, "rev-parse", "HEAD") == head
    assert _git(history.root, "status", "--porcelain") == ""
    assert _git(history.root, "worktree", "list", "--porcelain").count("worktree ") == 1


def test_checks_exact_github_merge_without_changing_checkout(history):
    completed = history.run()
    assert completed.returncode == 0, completed.stderr
    result = json.loads(history.result.read_text())
    assert result["head"] == history.merge
    assert result["args"] == ["check"]
    assert {"feature.txt", "target.txt"} <= set(result["files"])
    assert result["cwd"] != str(history.root)
    report = json.loads(
        (history.root / ".scratch/ci-preflight/last-run.json").read_text()
    )
    assert report["mode"] == "GitHub merge ref"
    assert report["status"] == "passed"
    assert report["head"] == history.head
    assert report["base"] == history.base
    _assert_cleaned(history, history.head)


def test_unpublished_descendant_is_tested_in_a_merge_preview(history):
    new_head = _commit(history.root, "unpublished.txt", "new candidate\n")
    completed = history.run()
    assert completed.returncode == 0, completed.stderr
    result = json.loads(history.result.read_text())
    assert {"unpublished.txt", "feature.txt", "target.txt"} <= set(result["files"])
    report = json.loads(
        (history.root / ".scratch/ci-preflight/last-run.json").read_text()
    )
    assert report["mode"] == "local merge preview"
    assert report["head"] == new_head
    assert _git(history.root, "rev-parse", f"{report['merge']}^2") == new_head
    assert _git(history.root, "rev-parse", f"{report['merge']}^1") == history.base
    _assert_cleaned(history, new_head)


def test_new_branch_can_be_checked_before_a_pr_exists(history):
    new_head = _commit(history.root, "unpublished.txt", "new candidate\n")
    completed = history.run("--base", "main")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(history.result.read_text())
    assert {"unpublished.txt", "feature.txt", "target.txt"} <= set(result["files"])
    report = json.loads(
        (history.root / ".scratch/ci-preflight/last-run.json").read_text()
    )
    assert report["pr"] is None
    assert report["mode"] == "local merge preview"
    _assert_cleaned(history, new_head)


def test_wrong_pr_checkout_is_rejected(history):
    _git(history.root, "switch", "main")
    completed = history.run()
    assert completed.returncode != 0
    assert "Check out the PR branch" in completed.stderr
    assert not history.result.exists()
    _assert_cleaned(history, history.base)


def test_dirty_checkout_is_rejected_before_validation(history):
    (history.root / "feature.txt").write_text("uncommitted work")
    completed = history.run()
    assert completed.returncode != 0
    assert "Commit or stash" in completed.stderr
    assert not history.result.exists()
    assert (history.root / "feature.txt").read_text() == "uncommitted work"


def test_validation_failure_is_propagated_and_worktree_removed(history):
    history.control.write_text(json.dumps({"exit": 7}))
    assert history.run().returncode == 7
    report = json.loads(
        (history.root / ".scratch/ci-preflight/last-run.json").read_text()
    )
    assert report["status"] == "failed"
    _assert_cleaned(history, history.head)


@pytest.mark.parametrize("changed", ["base", "remote_head"])
def test_remote_inputs_changing_during_checks_invalidate_the_result(history, changed):
    _git(history.root, "switch", "main" if changed == "base" else "feature")
    next_commit = _commit(history.root, "next.txt", "advance input\n")
    _git(history.root, "push", "origin", f"{next_commit}:refs/heads/validation-next")
    _git(history.root, "switch", "--detach", history.head)
    history.control.write_text(json.dumps({changed: next_commit}))
    completed = history.run()
    assert completed.returncode != 0
    assert "Rerun make check-pr" in completed.stderr
    _assert_cleaned(history, history.head)


def test_gate_cannot_pass_after_modifying_the_merge_candidate(history):
    history.control.write_text(json.dumps({"dirty": True}))
    completed = history.run()
    assert completed.returncode != 0
    assert "Commit or stash" in completed.stderr
    _assert_cleaned(history, history.head)


def test_retargeting_pr_during_validation_invalidates_the_result(history):
    history.control.write_text(json.dumps({"retarget": "different-target"}))
    completed = history.run()
    assert completed.returncode != 0
    assert "target changed" in completed.stderr
    _assert_cleaned(history, history.head)


def test_conflicts_are_reported_without_running_checks_or_changing_checkout(history):
    new_head = _commit(history.root, "shared.txt", "feature edit\n")
    _git(history.root, "switch", "main")
    _commit(history.root, "shared.txt", "target edit\n")
    _git(history.root, "push", "origin", "main")
    _git(history.root, "switch", "feature")
    completed = history.run()
    assert completed.returncode != 0
    assert "Resolve the merge conflicts" in completed.stderr
    assert not history.result.exists()
    _assert_cleaned(history, new_head)
