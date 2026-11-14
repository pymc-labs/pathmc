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
"""Release commands against a temporary git remote. They must not touch GitHub."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release.py"
TODAY = datetime.now(timezone.utc).date().isoformat()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _write_versions(root: Path, version: str, released: str) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "pathmc"\nversion = "{version}"\n'
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\nversion: {version}\ndate-released: {released}\n"
    )
    (root / "uv.lock").write_text(
        textwrap.dedent(
            f"""\
            [[package]]
            name = "pathmc"
            version = "{version}"
            source = {{ editable = "." }}

            [[package]]
            name = "numpy"
            version = "2.0.0"
            """
        )
    )


def _repo(tmp_path: Path, version: str = "0.6.0", released: str = "2020-01-01"):
    root = tmp_path / "checkout"
    remote = tmp_path / "origin.git"
    binaries = tmp_path / "bin"
    root.mkdir()
    binaries.mkdir()
    real_git = shutil.which("git")
    assert real_git
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@localhost.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "tag.gpgsign", "false")
    _git(root, "config", "core.hooksPath", "/dev/null")
    (root / "pathmc").mkdir()
    (root / "pathmc" / "__init__.py").write_text(
        '__version__ = _pkg_version("pathmc")\n'
    )
    _write_versions(root, version, released)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "origin", "main")
    git_log = tmp_path / "git-pushes.jsonl"
    gh_log = tmp_path / "gh.json"
    pr_body = tmp_path / "pr-body.md"
    uv_seen = tmp_path / "uv-seen.txt"
    tools = {
        "git": f"""
import json, os, subprocess, sys
from pathlib import Path
if len(sys.argv) > 1 and sys.argv[1] == "push":
    with Path({str(git_log)!r}).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({{
            "argv": sys.argv[1:],
            "github_token": os.environ.get("GITHUB_TOKEN"),
            "gh_token": os.environ.get("GH_TOKEN"),
            "github_enterprise_token": os.environ.get("GITHUB_ENTERPRISE_TOKEN"),
            "gh_enterprise_token": os.environ.get("GH_ENTERPRISE_TOKEN"),
        }}) + "\\n")
raise SystemExit(subprocess.run([{real_git!r}, *sys.argv[1:]]).returncode)
""",
        "uv": f"""
import os, sys
from pathlib import Path
assert sys.argv[1:] == ["lock"], sys.argv
if os.environ.get("PATHMC_UV_FAIL") == "1":
    print("uv lock failed in the stub", file=sys.stderr)
    raise SystemExit(1)
lock = Path("uv.lock")
seen = lock.read_text()
Path({str(uv_seen)!r}).write_text(seen)
version = None
for line in Path("pyproject.toml").read_text().splitlines():
    if line.startswith('version = "'):
        version = line.split('"', 2)[1]
        break
assert version
lines = seen.splitlines(keepends=True)
for index, line in enumerate(lines):
    if line.startswith('name = "pathmc"'):
        lines[index + 1] = 'version = "' + version + '"\\n'
        break
else:
    if seen and not seen.endswith("\\n"):
        lines.append("\\n")
    lines.extend([
        "[[package]]\\n",
        'name = "pathmc"\\n',
        'version = "' + version + '"\\n',
    ])
if os.environ.get("PATHMC_UV_EXTRA") == "1":
    lines.append("# unrelated pin changed by uv lock\\n")
lock.write_text("".join(lines))
""",
        "gh": f"""
import json, os, sys
from pathlib import Path
Path({str(gh_log)!r}).write_text(json.dumps({{
    "argv": sys.argv[1:],
    "github_token": os.environ.get("GITHUB_TOKEN"),
    "gh_token": os.environ.get("GH_TOKEN"),
    "github_enterprise_token": os.environ.get("GITHUB_ENTERPRISE_TOKEN"),
    "gh_enterprise_token": os.environ.get("GH_ENTERPRISE_TOKEN"),
}}))
if sys.argv[1:3] != ["pr", "create"]:
    print("unexpected gh call", sys.argv, file=sys.stderr)
    raise SystemExit(2)
body = Path(sys.argv[sys.argv.index("--body-file") + 1]).read_text()
Path({str(pr_body)!r}).write_text(body)
print("https://example.invalid/pull/1")
""",
    }
    for name, code in tools.items():
        path = binaries / name
        path.write_text(f"#!{sys.executable}\n" + code)
        path.chmod(0o755)
    return root, remote, binaries, git_log, gh_log, pr_body, uv_seen


def _run(root: Path, binaries: Path, *args: str, **env_extra: str):
    env = os.environ.copy()
    # The Actions test job sets this on every process. The refusal is for an
    # operator running the release command inside Actions, which one test sets.
    env.pop("GITHUB_ACTIONS", None)
    env["PATH"] = str(binaries) + os.pathsep + env["PATH"]
    for name in (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_ENTERPRISE_TOKEN",
        "GH_ENTERPRISE_TOKEN",
    ):
        env[name] = "must-not-be-used"
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
    )


def _pushes(git_log: Path) -> list[dict]:
    if not git_log.exists():
        return []
    return [json.loads(line) for line in git_log.read_text().splitlines() if line]


def _assert_main_untouched(root: Path, remote: Path, main_sha: str) -> None:
    assert _git(root, "rev-parse", "main") == main_sha
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "tag", "--list") == ""
    remote_main = _git(root, "ls-remote", str(remote), "refs/heads/main").split()[0]
    assert remote_main == main_sha


@pytest.mark.parametrize(
    ("part", "expected"),
    [("patch", "0.6.1"), ("minor", "0.7.0"), ("major", "1.0.0")],
)
def test_bump_opens_pr_without_pushing_main_or_tagging(tmp_path, part, expected):
    root, remote, binaries, git_log, gh_log, pr_body, uv_seen = _repo(tmp_path)
    main_sha = _git(root, "rev-parse", "HEAD")
    completed = _run(root, binaries, "bump", part)
    assert completed.returncode == 0, completed.stderr
    assert "https://example.invalid/pull/1" in completed.stdout
    assert _git(root, "rev-parse", "main") == main_sha
    assert (
        _git(root, "rev-parse", "--abbrev-ref", "HEAD") == f"release/pathmc-{expected}"
    )
    assert _git(root, "tag", "--list") == ""
    assert 'version = "' + expected + '"' in (root / "pyproject.toml").read_text()
    citation = (root / "CITATION.cff").read_text()
    assert f"version: {expected}\n" in citation
    assert f"date-released: {TODAY}\n" in citation
    assert f'version = "{expected}"' in (root / "uv.lock").read_text()
    assert 'version = "2.0.0"' in (root / "uv.lock").read_text()
    assert 'version = "0.6.0"' in uv_seen.read_text()
    assert (
        '__version__ = _pkg_version("pathmc")'
        in (root / "pathmc" / "__init__.py").read_text()
    )
    body = pr_body.read_text()
    assert "0.6.0 -> " + expected in body
    assert "make release-publish" in body
    assert "Protect main cannot rewrite the tip" in body
    assert "before it is the tip" not in body
    assert "No other lockfile changes." in body
    pushes = _pushes(git_log)
    assert len(pushes) == 1
    assert pushes[0]["argv"] == [
        "push",
        "--no-follow-tags",
        "origin",
        f"HEAD:refs/heads/release/pathmc-{expected}",
    ]
    assert pushes[0]["github_token"] is None
    for key in ("gh_token", "github_enterprise_token", "gh_enterprise_token"):
        assert pushes[0][key] is None
    gh = json.loads(gh_log.read_text())
    assert gh["github_token"] is None
    for key in ("gh_token", "github_enterprise_token", "gh_enterprise_token"):
        assert gh[key] is None
    assert "--base" in gh["argv"] and "main" in gh["argv"]
    remote_main = _git(root, "ls-remote", str(remote), "refs/heads/main").split()[0]
    assert remote_main == main_sha
    assert _git(root, "ls-remote", "--tags", str(remote)) == ""


def test_bump_does_not_push_reachable_annotated_tags(tmp_path):
    root, remote, binaries, *_ = _repo(tmp_path)
    _git(root, "tag", "-a", "0.5.0", "-m", "old")
    _git(root, "config", "push.followTags", "true")
    completed = _run(root, binaries, "bump", "patch")
    assert completed.returncode == 0, completed.stderr
    assert _git(root, "ls-remote", "--tags", str(remote)) == ""
    assert _git(root, "rev-parse", "refs/tags/0.5.0")


def test_patch_bump_increments_the_numeric_patch(tmp_path):
    root, _, binaries, *_ = _repo(tmp_path, version="0.6.9")
    completed = _run(root, binaries, "bump", "patch")
    assert completed.returncode == 0, completed.stderr
    assert 'version = "0.6.10"' in (root / "pyproject.toml").read_text()


def test_other_lock_diff_is_included_in_the_pr_body(tmp_path):
    root, _, binaries, _, _, pr_body, _ = _repo(tmp_path)
    completed = _run(root, binaries, "bump", "patch", PATHMC_UV_EXTRA="1")
    assert completed.returncode == 0, completed.stderr
    body = pr_body.read_text()
    assert "unrelated pin changed by uv lock" in body
    assert "No other lockfile changes." not in body


def test_failed_uv_lock_leaves_main_untouched(tmp_path):
    root, remote, binaries, git_log, gh_log, _, _ = _repo(tmp_path)
    main_sha = _git(root, "rev-parse", "HEAD")
    completed = _run(root, binaries, "bump", "patch", PATHMC_UV_FAIL="1")
    assert completed.returncode == 1
    assert "uv lock failed" in completed.stderr
    assert "Do not hand-edit uv.lock" in completed.stderr
    _assert_main_untouched(root, remote, main_sha)
    assert not gh_log.exists()
    assert _pushes(git_log) == []
    assert _git(root, "branch", "--list", "release/*") == ""


@pytest.mark.parametrize(
    ("setup", "needles"),
    [
        ("dirty", ("uncommitted changes", "commit, stash")),
        ("branch", ("not main", "git switch main")),
        ("behind", ("origin/main", "will not update main")),
        ("prerelease", ("not MAJOR.MINOR.PATCH", "will not guess")),
    ],
)
def test_bump_refusals_name_the_fix_and_do_not_push(tmp_path, setup, needles):
    version = "0.6.0rc1" if setup == "prerelease" else "0.6.0"
    root, remote, binaries, git_log, gh_log, _, _ = _repo(tmp_path, version=version)
    main_sha = _git(root, "rev-parse", "HEAD")
    if setup == "dirty":
        (root / "notes.txt").write_text("local\n")
    elif setup == "branch":
        _git(root, "switch", "-c", "feature")
    elif setup == "behind":
        _git(root, "commit", "--allow-empty", "-m", "remote moved")
        _git(root, "push", "origin", "main")
        _git(root, "reset", "--hard", "HEAD^")
    completed = _run(root, binaries, "bump", "patch")
    assert completed.returncode == 1
    for needle in needles:
        assert needle in completed.stderr
    assert not gh_log.exists()
    assert _pushes(git_log) == []
    if setup != "branch":
        assert _git(root, "rev-parse", "HEAD") == main_sha
    assert _git(root, "ls-remote", "--tags", str(remote)) == ""


def test_publish_tags_only_the_merged_bump(tmp_path):
    root, remote, binaries, git_log, gh_log, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.1", TODAY)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Bump version to 0.6.1")
    _git(root, "push", "origin", "main")
    head = _git(root, "rev-parse", "HEAD")
    completed = _run(root, binaries, "publish")
    assert completed.returncode == 0, completed.stderr
    assert f"Pushed tag 0.6.1 at {head[:12]}" in completed.stdout
    assert _git(root, "rev-parse", "HEAD") == head
    assert _git(root, "status", "--porcelain") == ""
    assert _git(root, "rev-parse", "refs/tags/0.6.1") == head
    remote_tag = _git(root, "ls-remote", str(remote), "refs/tags/0.6.1").split()[0]
    assert remote_tag == head
    pushes = _pushes(git_log)
    assert pushes == [
        {
            "argv": ["push", "--no-follow-tags", "origin", "refs/tags/0.6.1"],
            "github_token": None,
            "gh_token": None,
            "github_enterprise_token": None,
            "gh_enterprise_token": None,
        }
    ]
    assert not gh_log.exists()
    remote_main = _git(root, "ls-remote", str(remote), "refs/heads/main").split()[0]
    assert remote_main == head


def test_publish_refuses_a_later_commit_existing_tag_and_stale_date(tmp_path):
    root, remote, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.1", TODAY)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Bump version to 0.6.1")
    _git(root, "commit", "--allow-empty", "-m", "later")
    _git(root, "push", "origin", "main")
    later = _run(root, binaries, "publish")
    assert later.returncode == 1
    assert "not the merged bump" in later.stderr
    assert "Protect main" in later.stderr
    assert "new bump" in later.stderr
    assert "Reset" not in later.stderr
    assert "fast-forward" not in later.stderr
    assert "will not move main" in later.stderr
    assert _pushes(git_log) == []

    _git(root, "reset", "--hard", "HEAD^")
    _git(root, "push", "--force", "origin", "main")
    _git(root, "tag", "0.6.1")
    exists = _run(root, binaries, "publish")
    assert exists.returncode == 1
    assert "Tag 0.6.1 already exists" in exists.stderr
    assert "do not force-push" in exists.stderr
    assert _pushes(git_log) == []

    _git(root, "tag", "-d", "0.6.1")
    citation = (root / "CITATION.cff").read_text().replace(TODAY, "2020-01-01")
    (root / "CITATION.cff").write_text(citation)
    _git(root, "add", "CITATION.cff")
    _git(root, "commit", "--amend", "--no-edit")
    _git(root, "push", "--force", "origin", "main")
    stale = _run(root, binaries, "publish")
    assert stale.returncode == 1
    assert "date-released is 2020-01-01" in stale.stderr
    assert "Protect main" in stale.stderr
    assert "new bump" in stale.stderr
    assert "before it is the tip" not in stale.stderr
    assert "will not commit that change" in stale.stderr
    assert _git(root, "tag", "--list") == ""
    assert _pushes(git_log) == []


def test_publish_refuses_when_the_tag_would_not_equal_the_package_version(tmp_path):
    root, _, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.1", TODAY)
    citation = (
        (root / "CITATION.cff").read_text().replace("version: 0.6.1", "version: 0.6.2")
    )
    (root / "CITATION.cff").write_text(citation)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "inconsistent bump")
    _git(root, "push", "origin", "main")
    completed = _run(root, binaries, "publish")
    assert completed.returncode == 1
    assert "would not equal the package version" in completed.stderr
    assert "CITATION.cff has 0.6.2" in completed.stderr
    assert "Protect main" in completed.stderr
    assert "new bump" in completed.stderr
    assert "Do not align the merged tip" in completed.stderr
    assert "align pyproject.toml" not in completed.stderr
    assert "will not commit" in completed.stderr
    assert _pushes(git_log) == []
    assert _git(root, "tag", "--list") == ""


def test_publish_refuses_an_invalid_version_without_rewriting_main(tmp_path):
    root, _, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.0rc1", TODAY)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "invalid version")
    _git(root, "push", "origin", "main")
    before = _git(root, "rev-parse", "HEAD")
    completed = _run(root, binaries, "publish")
    assert completed.returncode == 1
    assert "not MAJOR.MINOR.PATCH" in completed.stderr
    assert "Protect main" in completed.stderr
    assert "will not guess" in completed.stderr
    assert "will not commit" in completed.stderr
    assert "before tagging" not in completed.stderr
    assert _git(root, "rev-parse", "HEAD") == before
    assert _pushes(git_log) == []
    assert _git(root, "tag", "--list") == ""


def test_publish_refuses_github_actions(tmp_path):
    root, _, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.1", TODAY)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Bump version to 0.6.1")
    _git(root, "push", "origin", "main")
    completed = _run(root, binaries, "publish", GITHUB_ACTIONS="true")
    assert completed.returncode == 1
    assert "GitHub Actions" in completed.stderr
    assert "GITHUB_TOKEN" in completed.stderr
    assert _pushes(git_log) == []
    assert _git(root, "tag", "--list") == ""


def test_init_keeps_metadata_version_and_workflows_listen_for_the_tag(tmp_path):
    del tmp_path
    init = (ROOT / "pathmc" / "__init__.py").read_text()
    assert '__version__ = _pkg_version("pathmc")' in init
    assert re.search(r"^__version__\s*=\s*['\"]", init, re.M) is None
    makefile = (ROOT / "Makefile").read_text()
    for target, arg in (
        ("release-patch", "bump patch"),
        ("release-minor", "bump minor"),
        ("release-major", "bump major"),
        ("release-publish", "publish"),
    ):
        assert f"{target}:" in makefile
        assert f"scripts/release.py {arg}" in makefile
    dry = subprocess.run(
        [
            "make",
            "-n",
            "release-patch",
            "release-minor",
            "release-major",
            "release-publish",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for arg in ("bump patch", "bump minor", "bump major", "publish"):
        assert (
            "uv run --no-project --no-config python scripts/release.py "
            f"{arg}" in dry.stdout
        )
    assert "--locked" not in dry.stdout
    assert "--frozen" not in dry.stdout
    assert "uv lock" not in dry.stdout
    assert "$(UV_RUN) python scripts/release.py" not in makefile
    publish = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    created = (ROOT / ".github" / "workflows" / "github-release.yml").read_text()
    assert "workflow_dispatch" not in publish
    assert "workflow_dispatch" not in created
    assert "types: [published]" not in publish
    assert "pull_request:" in publish
    assert "branches: [main]" in publish
    assert '"[0-9]+.[0-9]+.[0-9]+"' in publish
    assert "github.ref_type == 'tag'" in publish
    assert "environment: release" in created or "environment: release" in publish
    assert "gh release create" in created
    assert "--generate-notes" in created
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "make release-patch" in contributing
    assert "make release-publish" in contributing
    assert "Protect main cannot rewrite the tip" in contributing
    assert "Do not reset or align `main`" in contributing


def test_ahead_of_origin_names_a_local_reset(tmp_path):
    root, remote, binaries, git_log, gh_log, _, _ = _repo(tmp_path)
    _git(root, "commit", "--allow-empty", "-m", "local only")
    local = _git(root, "rev-parse", "HEAD")
    completed = _run(root, binaries, "bump", "patch")
    assert completed.returncode == 1
    assert "reset --hard origin/main" in completed.stderr
    assert "another branch" in completed.stderr
    assert "Do not push main" in completed.stderr
    assert "ff-only" not in completed.stderr
    assert _git(root, "rev-parse", "HEAD") == local
    assert not gh_log.exists()
    assert _pushes(git_log) == []
    assert _git(root, "ls-remote", "--tags", str(remote)) == ""


def test_bump_inserts_missing_citation_and_lock_fields(tmp_path):
    root, _, binaries, _, _, pr_body, _ = _repo(tmp_path)
    citation = (root / "CITATION.cff").read_text()
    citation = citation.replace("version: 0.6.0\n", "").replace(
        "date-released: 2020-01-01\n", ""
    )
    (root / "CITATION.cff").write_text(citation)
    lock = (
        (root / "uv.lock")
        .read_text()
        .replace(
            '[[package]]\nname = "pathmc"\nversion = "0.6.0"\n'
            'source = { editable = "." }\n\n',
            "",
        )
    )
    (root / "uv.lock").write_text(lock)
    _git(root, "add", ".")
    _git(root, "commit", "--amend", "--no-edit")
    _git(root, "push", "--force", "origin", "main")
    completed = _run(root, binaries, "bump", "patch")
    assert completed.returncode == 0, completed.stderr
    written = (root / "CITATION.cff").read_text()
    assert "version: 0.6.1\n" in written
    assert f"date-released: {TODAY}\n" in written
    assert 'name = "pathmc"' in (root / "uv.lock").read_text()
    assert 'version = "0.6.1"' in (root / "uv.lock").read_text()
    assert "0.6.0 -> 0.6.1" in pr_body.read_text()


def test_publish_missing_fields_name_a_new_bump(tmp_path):
    root, _, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    _write_versions(root, "0.6.1", TODAY)
    citation = (root / "CITATION.cff").read_text().replace("version: 0.6.1\n", "")
    (root / "CITATION.cff").write_text(citation)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Bump version to 0.6.1")
    _git(root, "push", "origin", "main")
    missing_version = _run(root, binaries, "publish")
    assert missing_version.returncode == 1
    assert "no version field" in missing_version.stderr
    assert "new bump" in missing_version.stderr
    assert "add `" not in missing_version.stderr
    assert "and retry" not in missing_version.stderr
    assert _pushes(git_log) == []

    citation = (root / "CITATION.cff").read_text()
    citation = citation.replace(f"date-released: {TODAY}\n", "")
    if "version: 0.6.1\n" not in citation:
        citation = f"version: 0.6.1\n{citation}"
    (root / "CITATION.cff").write_text(citation)
    _git(root, "add", "CITATION.cff")
    _git(root, "commit", "--amend", "--no-edit")
    _git(root, "push", "--force", "origin", "main")
    missing_date = _run(root, binaries, "publish")
    assert missing_date.returncode == 1
    assert "no date-released field" in missing_date.stderr
    assert "new bump" in missing_date.stderr
    assert "add `" not in missing_date.stderr
    assert _pushes(git_log) == []

    _write_versions(root, "0.6.1", TODAY)
    lock = (
        (root / "uv.lock")
        .read_text()
        .replace(
            '[[package]]\nname = "pathmc"\nversion = "0.6.1"\n'
            'source = { editable = "." }\n\n',
            "",
        )
    )
    (root / "uv.lock").write_text(lock)
    _git(root, "add", ".")
    _git(root, "commit", "--amend", "--no-edit")
    _git(root, "push", "--force", "origin", "main")
    missing_lock = _run(root, binaries, "publish")
    assert missing_lock.returncode == 1
    assert "no readable pathmc version" in missing_lock.stderr
    assert "new bump" in missing_lock.stderr
    assert "uv lock" not in missing_lock.stderr
    assert _pushes(git_log) == []


def test_publish_unparseable_parent_names_a_new_bump(tmp_path):
    root, _, binaries, git_log, _, _, _ = _repo(tmp_path, released=TODAY)
    (root / "pyproject.toml").write_text("[\n")
    _git(root, "add", "pyproject.toml")
    _git(root, "commit", "-m", "break parent")
    _write_versions(root, "0.6.1", TODAY)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Bump version to 0.6.1")
    _git(root, "push", "origin", "main")
    completed = _run(root, binaries, "publish")
    assert completed.returncode == 1
    assert "parent blob cannot be repaired" in completed.stderr
    assert "new bump" in completed.stderr
    assert "restore and retry" not in completed.stderr
    assert "restore a valid" not in completed.stderr
    assert _pushes(git_log) == []
    assert _git(root, "tag", "--list") == ""


def _strip_pathmc_member(lock: str) -> str:
    blocks = lock.split("\n[[package]]\n")
    kept = [blocks[0]]
    removed = 0
    for block in blocks[1:]:
        if block.startswith('name = "pathmc"\n'):
            removed += 1
            continue
        kept.append(block)
    assert removed == 1
    return "\n[[package]]\n".join(kept)


def _committed_release_tree(tmp_path: Path, pyproject: str, lock: str) -> Path:
    """A clean main matching a local origin, without the stub uv on PATH."""
    root = tmp_path / "checkout"
    remote = tmp_path / "origin.git"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True)
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@localhost.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "config", "tag.gpgsign", "false")
    _git(root, "config", "core.hooksPath", "/dev/null")
    (root / "pyproject.toml").write_text(pyproject)
    (root / "CITATION.cff").write_text(
        "cff-version: 1.2.0\nversion: 0.6.0\ndate-released: 2020-01-01\n"
    )
    (root / "uv.lock").write_text(lock)
    (root / ".gitignore").write_text(".venv/\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "initial")
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "origin", "main")
    return root


def _without_actions_flag() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GITHUB_ACTIONS", None)
    return env


def _release_argv(*flags: str) -> list[str]:
    uv = shutil.which("uv")
    assert uv
    return [uv, "run", *flags, "python", str(SCRIPT), "publish"]


def test_make_release_command_reaches_the_script_without_locking(tmp_path):
    """The four targets share this argv. It must start the script, not uv lock."""
    real_lock = (ROOT / "uv.lock").read_text()
    project = '[project]\nname = "pathmc"\nversion = "0.6.0"\n'
    command = _release_argv("--no-project", "--no-config")

    missing = _committed_release_tree(
        tmp_path / "missing", project, _strip_pathmc_member(real_lock)
    )
    head = _git(missing, "rev-parse", "HEAD")
    completed = subprocess.run(
        command,
        cwd=missing,
        env=_without_actions_flag(),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 1
    assert "no readable pathmc version" in completed.stderr
    assert "new bump" in completed.stderr
    assert "uv lock" not in completed.stderr
    assert "Missing workspace member" not in completed.stderr
    assert _git(missing, "status", "--porcelain") == ""
    assert _git(missing, "rev-parse", "HEAD") == head
    assert _git(missing, "rev-parse", "--abbrev-ref", "HEAD") == "main"

    frozen = subprocess.run(
        _release_argv("--frozen", "--all-extras"),
        cwd=missing,
        env=_without_actions_flag(),
        text=True,
        capture_output=True,
    )
    assert frozen.returncode != 0
    assert "no readable pathmc version" not in frozen.stderr
    assert "uv lock" in frozen.stderr
    assert _git(missing, "rev-parse", "HEAD") == head
    assert _git(missing, "status", "--porcelain") == ""

    broken = _committed_release_tree(tmp_path / "broken", "[\n", real_lock)
    broken_head = _git(broken, "rev-parse", "HEAD")
    parsed = subprocess.run(
        command,
        cwd=broken,
        env=_without_actions_flag(),
        text=True,
        capture_output=True,
    )
    assert parsed.returncode == 1
    assert "could not be parsed" in parsed.stderr
    assert "restores a valid" in parsed.stderr
    assert "uv lock" not in parsed.stderr
    assert _git(broken, "status", "--porcelain") == ""
    assert _git(broken, "rev-parse", "HEAD") == broken_head

    consistent = _committed_release_tree(tmp_path / "consistent", project, real_lock)
    consistent_head = _git(consistent, "rev-parse", "HEAD")
    started = subprocess.run(
        command,
        cwd=consistent,
        env=_without_actions_flag(),
        text=True,
        capture_output=True,
    )
    assert started.returncode == 1
    assert "not a merged version bump" in started.stderr
    assert "uv lock" not in started.stderr
    assert "Missing workspace member" not in started.stderr
    assert _git(consistent, "status", "--porcelain") == ""
    assert _git(consistent, "rev-parse", "HEAD") == consistent_head
    assert (consistent / "uv.lock").read_text() == real_lock
    assert (consistent / "pyproject.toml").read_text() == project


def test_lock_reader_and_rewriter_share_one_version_line():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pathmc_release_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    indented = '[[package]]\nname = "pathmc"\n  version = "0.6.0"\n'
    assert module._pathmc_lock_version(indented) == "0.6.0"
    rewritten = module._with_pathmc_lock_version(indented, "0.6.1")
    assert module._pathmc_lock_version(rewritten) == "0.6.1"

    commented = '[[package]]\nname = "pathmc"\nversion = "0.6.0" # pin\n'
    with pytest.raises(module.ReleaseError, match="no version"):
        module._pathmc_lock_version(commented)
    with pytest.raises(module.ReleaseError, match="no version"):
        module._with_pathmc_lock_version(commented, "0.6.1")
