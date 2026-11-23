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
"""Tests for the first-interaction welcome decision.

These tests do not call GitHub. They pass author history in memory.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "welcome_decision.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "welcome.yml"
PR_TEMPLATE = "Thanks for opening your first pull request, @{actor}\n"
ISSUE_TEMPLATE = "Thanks for opening your first issue, @{actor}\n"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("welcome_decision", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def _decide(script_module, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event": "pull_request_target",
        "actor": "newcomer",
        "number": 709,
        "authored": [{"number": 709, "pull_request": True}],
        "pr_message": PR_TEMPLATE,
        "issue_message": ISSUE_TEMPLATE,
    }
    payload.update(overrides)
    return script_module.welcome_decision(
        event=payload["event"],
        actor=payload["actor"],
        number=payload["number"],
        authored=payload["authored"],
        pr_message=payload["pr_message"],
        issue_message=payload["issue_message"],
    )


def test_pr_author_with_earlier_pr_and_no_issue_is_not_welcomed(script_module) -> None:
    """leninworld had merged PR 554 before opening PR 709, and had no issue."""
    decision = _decide(
        script_module,
        actor="leninworld",
        number=709,
        authored=[
            {"number": 554, "pull_request": True},
            {"number": 709, "pull_request": True},
            {"number": 710, "pull_request": True},
        ],
    )
    assert decision["post"] is False
    assert decision["body"] is None
    assert decision["kind"] is None


def test_first_pr_author_receives_exactly_one_pr_welcome(script_module) -> None:
    decision = _decide(
        script_module,
        actor="newcomer",
        number=800,
        authored=[{"number": 800, "pull_request": True}],
    )
    assert decision["post"] is True
    assert decision["kind"] == "pr"
    assert decision["body"] == "Thanks for opening your first pull request, @newcomer\n"
    assert "first issue" not in str(decision["body"])
    unindexed = _decide(script_module, actor="newcomer", number=800, authored=[])
    assert unindexed["post"] is True
    assert unindexed["kind"] == "pr"
    assert unindexed["body"] == decision["body"]


def test_earlier_issue_does_not_block_or_replace_a_first_pr_welcome(
    script_module,
) -> None:
    decision = _decide(
        script_module,
        actor="newcomer",
        number=800,
        authored=[
            {"number": 12, "pull_request": False},
            {"number": 800, "pull_request": {"html_url": "https://example.test/800"}},
        ],
    )
    assert decision == {
        "post": True,
        "kind": "pr",
        "body": "Thanks for opening your first pull request, @newcomer\n",
        "reason": "first pr",
    }


def test_issue_author_with_earlier_issue_and_no_pr_is_not_welcomed(
    script_module,
) -> None:
    decision = _decide(
        script_module,
        event="issues",
        actor="reporter",
        number=40,
        authored=[{"number": 3}, {"number": 40}],
    )
    assert decision["post"] is False
    assert decision["body"] is None


def test_first_issue_author_receives_exactly_one_issue_welcome(script_module) -> None:
    decision = _decide(
        script_module,
        event="issues",
        actor="reporter",
        number=40,
        authored=[{"number": 40}],
    )
    assert decision["post"] is True
    assert decision["kind"] == "issue"
    assert decision["body"] == "Thanks for opening your first issue, @reporter\n"
    assert "pull request" not in str(decision["body"])


def test_earlier_pr_does_not_block_or_replace_a_first_issue_welcome(
    script_module,
) -> None:
    decision = _decide(
        script_module,
        event="issues",
        actor="reporter",
        number=40,
        authored=[
            {"number": 554, "pull_request": True},
            {"number": 40, "pull_request": None},
        ],
    )
    assert decision["kind"] == "issue"
    assert decision["body"] == "Thanks for opening your first issue, @reporter\n"


def test_current_and_later_items_are_not_history(script_module) -> None:
    decision = _decide(
        script_module,
        number=709,
        authored=[
            {"number": 709, "pull_request": True},
            {"number": 710, "pull_request": True},
        ],
    )
    assert decision["post"] is True
    assert decision["kind"] == "pr"


@pytest.mark.parametrize(
    "actor", ["github-actions[bot]", "dependabot[bot]", "renovate"]
)
def test_bot_actors_are_not_welcomed(script_module, actor: str) -> None:
    decision = _decide(
        script_module,
        actor=actor,
        event="pull_request_target",
        authored=[],
    )
    assert decision["post"] is False
    issue = _decide(script_module, actor=actor, event="issues", number=1, authored=[])
    assert issue["post"] is False


def test_query_filters_to_this_author_and_all_states(script_module) -> None:
    args = script_module.query_args(
        owner="pymc-labs", repo="pathmc", author="leninworld"
    )
    assert args == {
        "owner": "pymc-labs",
        "repo": "pathmc",
        "creator": "leninworld",
        "state": "all",
        "per_page": 100,
    }
    assert "author" not in args


def test_query_refuses_empty_author(script_module) -> None:
    with pytest.raises(ValueError, match="Refusing to list every issue"):
        script_module.query_args(owner="pymc-labs", repo="pathmc", author="")


def test_omitted_history_does_not_welcome_everyone(
    script_module, tmp_path: Path
) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "event": "pull_request_target",
            "actor": "newcomer",
            "number": 1,
            "pr_message": PR_TEMPLATE,
            "issue_message": ISSUE_TEMPLATE,
        }),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "decide", "--payload-file", str(payload)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "authored is required" in result.stderr


def test_cli_reported_history_posts_nothing(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "event": "pull_request_target",
            "actor": "leninworld",
            "number": 709,
            "authored": [
                {"number": 554, "pull_request": True},
                {"number": 709, "pull_request": True},
            ],
            "pr_message": PR_TEMPLATE,
            "issue_message": ISSUE_TEMPLATE,
        }),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "decide", "--payload-file", str(payload)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["post"] is False


def _workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _literal_block(workflow: str, key: str) -> str:
    prefix = f"{key}: |"
    lines = workflow.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == prefix)
    indent = None
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() == "":
            body.append("")
            continue
        current = len(line) - len(line.lstrip(" "))
        if indent is None:
            indent = current
        if current < indent:
            break
        body.append(line[indent:])
    return "\n".join(body) + "\n"


def test_workflow_runs_the_tested_decision_without_first_interaction() -> None:
    workflow = _workflow_text()
    assert "actions/first-interaction" not in workflow
    assert "actions/checkout" not in workflow
    assert "contents:" not in workflow
    script = workflow.split("script: |", 1)[1]
    assert "pulls.list" not in script
    assert "head.sha" not in workflow
    assert "github.event.pull_request.head" not in workflow
    assert "github.paginate(github.rest.issues.listForRepo, query)" in workflow
    assert workflow.count("createComment") == 1
    assert "body: decision.body" in workflow
    assert "event: context.eventName" in workflow
    embedded = _literal_block(workflow, "WELCOME_DECISION_PY")
    assert embedded == SCRIPT_PATH.read_text(encoding="utf-8")


def test_workflow_keeps_the_pr_welcome_text() -> None:
    workflow = _workflow_text()
    pr_message = _literal_block(workflow, "pr_message")
    issue_message = _literal_block(workflow, "issue_message")
    assert "LinkedIn" not in pr_message
    assert "review your changes soon" not in pr_message
    assert "CONTRIBUTING.md#first-pull-requests" in pr_message
    assert "assignment" in pr_message
    assert "spam?" in pr_message
    assert "{actor}" in pr_message
    assert "first issue" in issue_message
    assert "spam?" not in issue_message


def test_first_pr_posts_the_workflow_pr_message(script_module) -> None:
    workflow = _workflow_text()
    decision = _decide(
        script_module,
        actor="ada",
        pr_message=_literal_block(workflow, "pr_message"),
        issue_message=_literal_block(workflow, "issue_message"),
    )
    assert decision["kind"] == "pr"
    body = str(decision["body"])
    assert body.count("Welcome to pathmc") == 1
    assert "@ada" in body
    assert "first pull request" in body
    assert "first issue" not in body
    assert "spam?" in body


def test_bot_markers_match_the_workflow_skip(script_module) -> None:
    job_if = _workflow_text().split("if: >-", 1)[1].split("permissions:", 1)[0]
    for marker in script_module.BOT_ACTOR_MARKERS:
        assert f"contains(github.actor, '{marker}')" in job_if


def test_embedded_copy_rejects_the_reported_history(tmp_path: Path) -> None:
    embedded = _literal_block(_workflow_text(), "WELCOME_DECISION_PY")
    helper = tmp_path / "welcome_decision.py"
    helper.write_text(embedded, encoding="utf-8")
    payload = tmp_path / "payload.json"
    payload.write_text(
        json.dumps({
            "event": "pull_request_target",
            "actor": "leninworld",
            "number": 709,
            "authored": [{"number": 554, "pull_request": True}],
            "pr_message": PR_TEMPLATE,
            "issue_message": ISSUE_TEMPLATE,
        }),
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(helper), "decide", "--payload-file", str(payload)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["post"] is False
