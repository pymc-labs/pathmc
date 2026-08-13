# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

This file was written for pathmc by adapting the template in `.agents/skills/setup-matt-pocock-skills/issue-tracker-github.md`. We did not run the interactive setup skill — the result is the same as accepting its GitHub defaults. See the **Bringing work to a new repo** section in `.agents/skills/work/SKILL.md` when porting this stack elsewhere.

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

## When a skill says "publish to the issue tracker"

Create or update a GitHub issue. For `to-spec` and `work`, append a `## Spec` section to the issue body.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
