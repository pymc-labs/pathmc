# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

This file records pathmc's repository-specific issue conventions. General-purpose planning, implementation, TDD, and review skills are intentionally not vendored in this repository; developers can install and use preferred workflows in their user-level skills.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments`
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When an agent needs the issue tracker

Create or update a GitHub issue using the commands above, preserving any existing human-authored content.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
