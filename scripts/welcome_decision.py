#!/usr/bin/env python3
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

"""Decide whether to post a first-issue or first-pull-request welcome.

The pull-request welcome depends only on earlier pull requests by this
author. The issue welcome depends only on earlier issues. A prior issue
must not make a returning pull-request author look new, and a prior pull
request must not make a returning issue author look new.

This module does not call GitHub. The workflow passes event data as a JSON
file and posts ``body`` at most once. It inlines this source: checking the
file out would need contents permission, and the job must not gain that or
run pull-request code. ``tests/test_welcome_decision.py`` fails if the
copies diverge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Same substring checks as .github/workflows/welcome.yml. GitHub's contains()
# is case-sensitive, so these markers are too.
BOT_ACTOR_MARKERS = ("[bot]", "dependabot", "renovate")
PR_EVENTS = frozenset({"pull_request", "pull_request_target"})
ISSUE_EVENTS = frozenset({"issues"})


def is_skipped_bot(actor: str) -> bool:
    """Return whether ``actor`` matches the welcome-workflow bot skip."""
    return any(marker in actor for marker in BOT_ACTOR_MARKERS)


def query_args(*, owner: str, repo: str, author: str) -> dict[str, object]:
    """Return ``issues.listForRepo`` parameters for this author's history.

    ``pulls.list`` cannot filter by creator. An unfiltered pull list treats
    another author's earlier pull request as this author's, so the history
    query has to be the issues list with ``creator`` set. ``state=all``
    counts closed and merged items. Paginate the result; ``per_page`` only
    sizes each page.
    """
    if not owner or not repo:
        raise ValueError(
            "owner and repo must be non-empty. "
            "Pass context.repo so the query cannot run against an unspecified repository."
        )
    if not author:
        raise ValueError(
            "author login is empty. Refusing to list every issue in the repository. "
            "Pass the event author's login as creator."
        )
    return {
        "owner": owner,
        "repo": repo,
        "creator": author,
        "state": "all",
        "per_page": 100,
    }


def welcome_decision(
    *,
    event: str,
    actor: str,
    number: int,
    authored: Sequence[Mapping[str, object]],
    pr_message: str,
    issue_message: str,
) -> dict[str, object]:
    """Return whether to post one welcome, and the body if so.

    ``authored`` is the creator-filtered ``issues.listForRepo`` result.
    Items with ``number`` less than ``number`` are earlier. The current
    item and any later item are not history. A truthy ``pull_request``
    value, including the API object, marks a pull request.
    """
    if not pr_message or not issue_message:
        raise ValueError(
            "pr_message and issue_message must be non-empty. "
            "Restore the welcome text in .github/workflows/welcome.yml."
        )
    if "{actor}" not in pr_message or "{actor}" not in issue_message:
        raise ValueError(
            "welcome text is missing the {actor} placeholder. "
            "Keep the @ mention so the comment names the author."
        )
    if is_skipped_bot(actor):
        return _skip("bot actor")
    if not actor:
        raise ValueError(
            "actor login is empty. Refusing to post a welcome that cannot name the author."
        )
    if event in PR_EVENTS:
        if _has_earlier(authored, number, want_pr=True):
            return _skip("author already opened a pull request")
        return _post("pr", pr_message.replace("{actor}", actor))
    if event in ISSUE_EVENTS:
        if _has_earlier(authored, number, want_pr=False):
            return _skip("author already opened an issue")
        return _post("issue", issue_message.replace("{actor}", actor))
    raise ValueError(
        f"unsupported event {event!r}. "
        "Pass context.eventName (pull_request_target or issues)."
    )


def _skip(reason: str) -> dict[str, object]:
    return {"post": False, "kind": None, "body": None, "reason": reason}


def _post(kind: str, body: str) -> dict[str, object]:
    return {"post": True, "kind": kind, "body": body, "reason": f"first {kind}"}


def _has_earlier(
    authored: Sequence[Mapping[str, object]], number: int, *, want_pr: bool
) -> bool:
    for item in authored:
        if not isinstance(item, Mapping):
            raise ValueError(
                "each authored item must be an object with a number. "
                "Pass issues.listForRepo items, not bare numbers."
            )
        if int(item["number"]) < number and _is_pull_request(item) is want_pr:
            return True
    return False


def _is_pull_request(item: Mapping[str, object]) -> bool:
    marker = item.get("pull_request", False)
    if isinstance(marker, Mapping):
        return True
    return bool(marker)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    query = commands.add_parser("query-args")
    query.add_argument("--payload-file", required=True)
    query.set_defaults(func=_command_query_args)

    decide = commands.add_parser("decide")
    decide.add_argument("--payload-file", required=True)
    decide.set_defaults(func=_command_decide)
    return parser


def _command_query_args(args: argparse.Namespace) -> int:
    payload = _read_payload(args.payload_file)
    result = query_args(
        owner=_required_text(payload, "owner"),
        repo=_required_text(payload, "repo"),
        author=_required_text(payload, "author"),
    )
    _write_result(result)
    return 0


def _command_decide(args: argparse.Namespace) -> int:
    payload = _read_payload(args.payload_file)
    if "authored" not in payload:
        raise ValueError(
            "authored is required. Pass the creator-filtered issues.listForRepo "
            "results; an omitted list would welcome every author."
        )
    authored = payload["authored"]
    if not isinstance(authored, list):
        raise ValueError(
            "authored must be a list of issue objects. "
            "Pass the creator-filtered issues.listForRepo result, not one issue."
        )
    result = welcome_decision(
        event=_required_text(payload, "event"),
        actor=_required_text(payload, "actor"),
        number=int(payload["number"]),
        authored=authored,
        pr_message=_required_text(payload, "pr_message"),
        issue_message=_required_text(payload, "issue_message"),
    )
    _write_result(result)
    return 0


def _read_payload(path: str) -> Mapping[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            "payload must be a JSON object. Write the event fields to the payload file."
        )
    return payload


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{key} must be a non-empty string. "
            "Write it to the payload file; do not pass it as a shell argument."
        )
    return value


def _write_result(result: Mapping[str, object]) -> None:
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run a subcommand. User-controlled text is read from files, not argv."""
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"welcome_decision: {exc}. "
            "Fix the payload or the welcome text; do not list unfiltered pulls.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
