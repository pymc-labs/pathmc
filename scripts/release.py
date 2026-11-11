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
"""Open a version-bump pull request, or tag the merged bump.

bumpver is not used. Its commit, tag, and push run on the current branch, and
``date-released`` is not a version field. This script must not push ``main``
or create a tag until ``publish``.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

_SEMVER = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")
_VERSION_FILES = ("pyproject.toml", "CITATION.cff", "uv.lock")
_REMOTE = "origin"
_MAIN = "main"


class ReleaseError(Exception):
    """A refused release action. The message names the fix."""


def next_version(current: str, part: str) -> str:
    """Return the next MAJOR.MINOR.PATCH. Refuse a pre-release instead of guessing."""
    match = _SEMVER.fullmatch(current)
    if not match:
        raise ReleaseError(
            f"Version {current!r} is not MAJOR.MINOR.PATCH. "
            'Fix: set [project].version to "X.Y.Z" and retry. '
            "This command will not guess a bump from a pre-release."
        )
    major, minor, patch = (int(piece) for piece in match.groups())
    if part == "patch":
        patch += 1
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ReleaseError(
            f"Unknown bump part {part!r}. Fix: use patch, minor, or major."
        )
    return f"{major}.{minor}.{patch}"


def _run(
    root: Path,
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=root,
        text=True,
        capture_output=True,
        env=env,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            f"{' '.join(args)} failed.\n{detail}\nFix the error above and retry."
        )
    return result


def _stdout(root: Path, args: list[str]) -> str:
    return _run(root, args).stdout.strip()


def _caller_env() -> dict[str, str]:
    """Drop the Actions token so git and gh use the caller's own credentials."""
    env = os.environ.copy()
    env.pop("GITHUB_TOKEN", None)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _refuse_actions() -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise ReleaseError(
            "Refusing to run from GitHub Actions. "
            "Fix: run make release-patch or make release-publish locally so "
            "git uses your admin credentials, not GITHUB_TOKEN."
        )


def _require_root(root: Path) -> None:
    probe = _run(root, ["git", "rev-parse", "--show-toplevel"], check=False)
    if probe.returncode != 0:
        raise ReleaseError(
            "This directory is not a git repository. "
            "Fix: run the command from the pathmc checkout."
        )
    toplevel = Path(probe.stdout.strip()).resolve()
    if toplevel != root.resolve():
        raise ReleaseError(
            f"Run this from the repository root ({toplevel}), not {root}. "
            "Fix: cd to the root and retry."
        )


def _require_clean_main(root: Path) -> str:
    """Return origin/main. Refuse unless this checkout is that commit and clean."""
    _require_root(root)
    _refuse_actions()
    branch = _stdout(root, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch != _MAIN:
        raise ReleaseError(
            f"Checkout is on {branch!r}, not main. "
            "Fix: git switch main, then retry. This command will not switch branches for you."
        )
    dirty = _stdout(root, ["git", "status", "--porcelain"])
    if dirty:
        raise ReleaseError(
            "Checkout has uncommitted changes. "
            "Fix: commit, stash, or remove them, then retry. Ignored files are allowed."
        )
    fetch = _run(root, ["git", "fetch", _REMOTE, _MAIN], check=False)
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout).strip()
        raise ReleaseError(
            f"Could not fetch {_REMOTE} {_MAIN}.\n{detail}\n"
            "Fix: check the origin remote and network, then retry."
        )
    head = _stdout(root, ["git", "rev-parse", "HEAD"])
    remote = _run(root, ["git", "rev-parse", f"{_REMOTE}/{_MAIN}"], check=False)
    if remote.returncode != 0:
        raise ReleaseError(
            f"{_REMOTE}/{_MAIN} is missing after fetch. "
            "Fix: git fetch origin main, then retry."
        )
    remote_sha = remote.stdout.strip()
    if head != remote_sha:
        raise ReleaseError(
            f"main is {head[:12]}, but {_REMOTE}/main is {remote_sha[:12]}. "
            "Fix: git fetch origin && git merge --ff-only origin/main, then retry. "
            "This command will not update main."
        )
    return head


def _project_version(text: str) -> str:
    try:
        version = tomllib.loads(text).get("project", {}).get("version")
    except tomllib.TOMLDecodeError as exc:
        raise ReleaseError(
            f"pyproject.toml could not be parsed ({exc}). "
            "Fix: restore a valid [project].version and retry."
        ) from exc
    if not isinstance(version, str):
        raise ReleaseError(
            "pyproject.toml [project].version is missing. "
            'Fix: set version = "X.Y.Z" under [project] and retry.'
        )
    return version


def _set_project_version(text: str, version: str) -> str:
    """Replace only [project].version, keeping the double-quoted form #696 uses."""
    lines = text.splitlines(keepends=True)
    in_project = False
    replaced = False
    header = re.compile(r"^\[([^\[\]\s]+)\]\s*$")
    version_line = re.compile(r"""^version\s*=\s*(['"])([^'"]+)\1""")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        match_header = header.match(stripped)
        if match_header:
            in_project = match_header.group(1) == "project"
        if in_project and not replaced and version_line.match(stripped):
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            out.append(f'{indent}version = "{version}"{newline}')
            replaced = True
            continue
        out.append(line)
    if not replaced:
        raise ReleaseError(
            "pyproject.toml [project].version was not found. "
            'Fix: set version = "X.Y.Z" under [project] and retry.'
        )
    rewritten = "".join(out)
    if _project_version(rewritten) != version:
        raise ReleaseError(
            "Rewriting [project].version did not round-trip. "
            'Fix: keep version = "X.Y.Z" directly under [project] and retry.'
        )
    return rewritten


def _citation_field(text: str, key: str) -> str:
    match = re.search(
        rf"^{re.escape(key)}:[ \t]*['\"]?([^'\"\s#]+)['\"]?[ \t]*$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ReleaseError(
            f"CITATION.cff has no {key} field. "
            f"Fix: add `{key}: ...` in the existing unquoted form and retry."
        )
    return match.group(1)


def _set_citation_field(text: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(key)}:[ \t]*['\"]?[^'\"\s#]+['\"]?[ \t]*$",
        flags=re.MULTILINE,
    )
    rewritten, count = pattern.subn(f"{key}: {value}", text, count=1)
    if count != 1:
        raise ReleaseError(
            f"CITATION.cff has no {key} field. "
            f"Fix: add `{key}: {value}` in the existing unquoted form and retry."
        )
    if text.endswith("\n") and not rewritten.endswith("\n"):
        rewritten += "\n"
    return rewritten


def _pathmc_lock_version(text: str) -> str:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "[[package]]":
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].startswith("["):
            end += 1
        block = lines[index + 1 : end]
        if any(line.strip() == 'name = "pathmc"' for line in block):
            for line in block:
                match = re.fullmatch(r"""version = (['"])([^'"]+)\1""", line.strip())
                if match:
                    return match.group(2)
            raise ReleaseError(
                'uv.lock entry name = "pathmc" has no version. '
                "Fix: run uv lock and retry. Do not hand-edit the lock."
            )
        index = end
    raise ReleaseError(
        'uv.lock has no package entry name = "pathmc". '
        "Fix: run uv lock from the repository root and retry."
    )


def _with_pathmc_lock_version(text: str, version: str) -> str:
    """Return text with only the pathmc version line replaced. Does not write the lock."""
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        if lines[index].strip() != "[[package]]":
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].startswith("["):
            end += 1
        block = [line.strip() for line in lines[index + 1 : end]]
        if any(line == 'name = "pathmc"' for line in block):
            for cursor in range(index + 1, end):
                if lines[cursor].startswith("version = "):
                    newline = "\n" if lines[cursor].endswith("\n") else ""
                    lines[cursor] = f'version = "{version}"{newline}'
                    return "".join(lines)
        index = end
    raise ReleaseError(
        'uv.lock has no package entry name = "pathmc". '
        "Fix: run uv lock from the repository root and retry."
    )


def _other_lock_diff(before: str, after: str, old: str) -> str:
    if _with_pathmc_lock_version(after, old) == before:
        return ""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="uv.lock (before uv lock)",
            tofile="uv.lock (after uv lock)",
        )
    )


def _changed_paths(root: Path) -> set[str]:
    status = _run(root, ["git", "status", "--porcelain"]).stdout
    paths: set[str] = set()
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path.strip('"'))
    return paths


def _ref_exists(root: Path, ref: str) -> bool:
    result = _run(root, ["git", "show-ref", "--verify", "--quiet", ref], check=False)
    return result.returncode == 0


def _remote_has(root: Path, kind: str, name: str) -> bool:
    result = _run(root, ["git", "ls-remote", f"--{kind}", _REMOTE, name], check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            f"Could not list {_REMOTE} {kind}.\n{detail}\n"
            "Fix: check the origin remote and retry."
        )
    return bool(result.stdout.strip())


def _push(root: Path, refspec: str) -> None:
    destination = refspec.split(":", 1)[-1]
    if destination in {"main", "HEAD", "refs/heads/main"} or "--force" in refspec:
        raise ReleaseError(
            "Refusing to push main. "
            "Fix: push only the release branch or the bare version tag."
        )
    command = ["git", "push", _REMOTE, refspec]
    if any(arg in {"--force", "--force-with-lease", "-f"} for arg in command):
        raise ReleaseError(
            "Refusing to force-push. Fix: delete a mistaken tag by hand, or bump again."
        )
    result = _run(root, command, env=_caller_env(), check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ReleaseError(
            f"git push {_REMOTE} {refspec} failed.\n{detail}\n"
            "Fix: authenticate with your admin credentials and retry. "
            "Do not set GITHUB_TOKEN for this push, and do not force-push."
        )


def _abandon(root: Path, branch: str, main_sha: str) -> None:
    current = _stdout(root, ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if current == branch:
        _run(root, ["git", "reset", "--hard", main_sha])
        _run(root, ["git", "switch", _MAIN])
    _run(root, ["git", "branch", "-D", branch], check=False)


def _pr_body(old: str, new: str, released: str, other_diff: str) -> str:
    if other_diff:
        shown = other_diff
        if len(shown) > 40_000:
            shown = (
                shown[:40_000]
                + "\n... diff truncated; the bump commit has the full lockfile change.\n"
            )
        lock_note = (
            f"`uv lock` updated the `pathmc` entry from {old} to {new}. "
            "It also changed other pins. Review this diff before merging:\n\n"
            f"```diff\n{shown}```"
        )
    else:
        lock_note = (
            f"`uv lock` updated the `pathmc` entry from {old} to {new}. "
            "No other lockfile changes."
        )
    return (
        "## Version bump\n\n"
        f"{old} -> {new}\n\n"
        f"`CITATION.cff` `date-released` is {released}, the UTC date this "
        "pull request was opened.\n\n"
        "## After merge\n\n"
        "`main` requires a linear history, so squash or rebase this pull request. "
        "Then, from a clean `main` that matches `origin/main`, on the same UTC "
        "day as `date-released`:\n\n"
        "```bash\n"
        "make release-publish\n"
        "```\n\n"
        "That tags only the merged bump commit and pushes the bare tag "
        f"`{new}` with your credentials. It does not push `main`, does not commit, "
        "and does not use `GITHUB_TOKEN`. It refuses if `main` is not this bump, "
        f"if `{new}` already exists, if the tag would not equal the package version, "
        "or if `date-released` is not today's UTC date. It will not commit a date "
        "change. If you tag on a later day, set `date-released` to that UTC date in "
        "this bump before it is the tip of `main`, then run `make release-publish`.\n\n"
        "The tag push creates the GitHub Release with generated notes and starts "
        "the PyPI jobs. Test PyPI runs, then the real upload waits for approval "
        "on the `release` environment.\n\n"
        "## Lockfile\n\n"
        f"{lock_note}\n"
    )


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def bump(root: Path, part: str) -> None:
    main_sha = _require_clean_main(root)
    pyproject = root / "pyproject.toml"
    citation = root / "CITATION.cff"
    lock = root / "uv.lock"
    old = _project_version(pyproject.read_text())
    new = next_version(old, part)
    released = _today()
    branch = f"release/pathmc-{new}"
    if _ref_exists(root, f"refs/heads/{branch}") or _remote_has(
        root, "heads", f"refs/heads/{branch}"
    ):
        raise ReleaseError(
            f"Branch {branch} already exists. "
            "Fix: open the pull request from that branch, or delete it if the "
            "previous attempt failed before the pull request opened. "
            "This command will not push main."
        )
    remote_main = _stdout(root, ["git", "ls-remote", _REMOTE, "refs/heads/main"])
    created = False
    pushed = False
    try:
        _run(root, ["git", "switch", "-c", branch])
        created = True
        pyproject.write_text(_set_project_version(pyproject.read_text(), new))
        citation_text = _set_citation_field(citation.read_text(), "version", new)
        citation.write_text(
            _set_citation_field(citation_text, "date-released", released)
        )
        before_lock = lock.read_text()
        lock_result = _run(root, ["uv", "lock"], check=False)
        if lock_result.returncode != 0:
            detail = (lock_result.stderr or lock_result.stdout).strip()
            raise ReleaseError(
                f"uv lock failed.\n{detail}\n"
                "Fix: run uv lock from the repository root and retry. "
                "Do not hand-edit uv.lock."
            )
        after_lock = lock.read_text()
        locked = _pathmc_lock_version(after_lock)
        if locked != new:
            raise ReleaseError(
                f"uv lock left the pathmc entry at {locked}, not {new}. "
                "Fix: run uv lock so it rewrites that entry. Do not hand-edit uv.lock."
            )
        unexpected = _changed_paths(root) - set(_VERSION_FILES)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ReleaseError(
                f"The bump changed unexpected files: {names}. "
                "Fix: revert those files and retry. This command will not commit them."
            )
        other = _other_lock_diff(before_lock, after_lock, old)
        _run(root, ["git", "add", "--", *_VERSION_FILES])
        commit = _run(
            root,
            ["git", "commit", "-m", f"Bump version {old} -> {new}"],
            check=False,
        )
        if commit.returncode != 0:
            detail = (commit.stderr or commit.stdout).strip()
            raise ReleaseError(
                f"The bump commit failed.\n{detail}\n"
                "Fix: resolve the hook or identity error and retry. "
                "This command will not use --no-verify, push main, or create a tag."
            )
        if _stdout(root, ["git", "rev-parse", "--abbrev-ref", "HEAD"]) != branch:
            raise ReleaseError(
                "The bump commit did not stay on the release branch. "
                "Fix: inspect HEAD and retry. This command must not commit on main."
            )
        if _stdout(root, ["git", "rev-parse", "refs/heads/main"]) != main_sha:
            raise ReleaseError(
                "main moved during the bump. "
                "Fix: inspect the repository. This command must not commit on main."
            )
        _push(root, f"HEAD:refs/heads/{branch}")
        pushed = True
        if (
            _stdout(root, ["git", "ls-remote", _REMOTE, "refs/heads/main"])
            != remote_main
        ):
            raise ReleaseError(
                "origin/main changed during the branch push. "
                "Fix: do not use this branch if main was pushed. "
                "This command must not push main."
            )
        body = _pr_body(old, new, released, other)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", encoding="utf-8", delete=False
        ) as handle:
            handle.write(body)
            body_path = handle.name
        try:
            gh = _run(
                root,
                [
                    "gh",
                    "pr",
                    "create",
                    "--base",
                    _MAIN,
                    "--head",
                    branch,
                    "--title",
                    f"Bump version {old} -> {new}",
                    "--body-file",
                    body_path,
                ],
                env=_caller_env(),
                check=False,
            )
        finally:
            Path(body_path).unlink(missing_ok=True)
        if gh.returncode != 0:
            detail = (gh.stderr or gh.stdout).strip()
            raise ReleaseError(
                f"Pushed {branch} but could not open the pull request.\n{detail}\n"
                "Fix: gh auth login, then:\n"
                f"gh pr create --base main --head {branch} "
                f'--title "Bump version {old} -> {new}" --body-file BODY\n'
                "This command will not push main or create a tag.\n"
                f"Pull request body:\n{body}"
            )
    except ReleaseError as exc:
        if created and not pushed:
            try:
                _abandon(root, branch, main_sha)
            except ReleaseError as abandon_error:
                raise ReleaseError(
                    f"{exc}\nAlso failed to restore the checkout: {abandon_error}\n"
                    f"Fix: git switch main && git reset --hard {main_sha} && "
                    f"git branch -D {branch}"
                ) from abandon_error
        raise
    print(gh.stdout.strip())
    print(
        f"Opened a pull request to bump {old} -> {new} on {branch}. "
        "main was not pushed and no tag was created."
    )


def _versions_agree(root: Path) -> str:
    version = _project_version((root / "pyproject.toml").read_text())
    if not _SEMVER.fullmatch(version):
        raise ReleaseError(
            f"Package version {version!r} is not MAJOR.MINOR.PATCH. "
            'Fix: set [project].version to "X.Y.Z" before tagging. '
            "This command will not invent a version."
        )
    citation = _citation_field((root / "CITATION.cff").read_text(), "version")
    locked = _pathmc_lock_version((root / "uv.lock").read_text())
    mismatches = [
        (label, found)
        for label, found in (("CITATION.cff", citation), ("uv.lock", locked))
        if found != version
    ]
    if mismatches:
        details = ", ".join(f"{label} has {found}" for label, found in mismatches)
        raise ReleaseError(
            f"Refusing to tag {version}: it would not equal the package version "
            f"({details}). Fix: align pyproject.toml, CITATION.cff, and the "
            "uv.lock pathmc entry, then retry. This command will not commit."
        )
    return version


def publish(root: Path) -> None:
    _require_clean_main(root)
    version = _versions_agree(root)
    parent = _run(root, ["git", "rev-parse", "--verify", "HEAD^"], check=False)
    if parent.returncode != 0:
        raise ReleaseError(
            "main has no parent commit, so it is not a merged version bump. "
            "Fix: tag only a commit that changes [project].version. "
            "This command will not commit."
        )
    parent_text = _run(root, ["git", "show", "HEAD^:pyproject.toml"]).stdout
    parent_version = _project_version(parent_text)
    if parent_version == version:
        head = _stdout(root, ["git", "rev-parse", "--short", "HEAD"])
        raise ReleaseError(
            f"main ({head}) is not the merged bump: its parent already has "
            f"version {version}. Fix: check out the merged bump commit and "
            "fast-forward main to it before tagging. This command will not move main."
        )
    if _ref_exists(root, f"refs/tags/{version}") or _remote_has(
        root, "tags", f"refs/tags/{version}"
    ):
        raise ReleaseError(
            f"Tag {version} already exists. Fix: do not force-push the tag. "
            "If this version was already released, bump again with make release-patch. "
            "If the tag is a local mistake, delete it and retry."
        )
    released = _citation_field((root / "CITATION.cff").read_text(), "date-released")
    today = _today()
    if released != today:
        raise ReleaseError(
            f"CITATION.cff date-released is {released}, not today's UTC date {today}. "
            f"Fix: set date-released to {today} in the bump commit before it is the "
            "tip of main. This command will not commit that change."
        )
    head = _stdout(root, ["git", "rev-parse", "HEAD"])
    _run(root, ["git", "tag", "--", version, head])
    try:
        _push(root, f"refs/tags/{version}")
    except ReleaseError:
        _run(root, ["git", "tag", "-d", "--", version], check=False)
        raise
    if _stdout(root, ["git", "rev-parse", "HEAD"]) != head:
        raise ReleaseError(
            "HEAD moved while tagging. Fix: inspect the repository. "
            "This command must not commit."
        )
    print(f"Pushed tag {version} at {head[:12]}. main was not pushed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bump_parser = sub.add_parser("bump", help="open a version-bump pull request")
    bump_parser.add_argument("part", choices=("patch", "minor", "major"))
    sub.add_parser("publish", help="tag the merged bump and push that tag")
    args = parser.parse_args(argv)
    try:
        if args.command == "bump":
            bump(Path.cwd(), args.part)
        else:
            publish(Path.cwd())
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
