---
name: work
description: >-
  State-aware orchestrator for enhancements and features. Detects where work
  left off (spec, implement, review) and advances it one phase. Delegates to
  grill-with-docs, to-spec, implement, tdd, and code-review internally — the
  user invokes only this skill. Use when the user says "work #N", "work on
  #N", "advance #N", or "work" to see in-flight items.
---

# Work — State-Aware Feature Orchestrator

One command advances a feature issue through spec → implement → review. You never invoke the sub-skills directly; this skill reads state from GitHub labels and PR markers, then runs the right phase.

For **bugs**, delegate to the `fix-bug` skill instead.

Read **Repo conventions** below and `docs/agents/*.md` before changing code.

## Entry points

| You say | What the agent does |
|---------|---------------------|
| `work #284` / `work on #284` | Load issue → detect state → run next phase |
| `work PR #123` | Load PR → detect state → resume implement/review |
| `work` | List in-flight issues/PRs with state and suggested next action |

## Repo conventions

Customize when copying to another repo. pathmc subsection below is authoritative for this clone.

- **Agent guide**: `AGENTS.md` at repo root.
- **Issue tracker**: `docs/agents/issue-tracker.md`
- **Triage labels**: `docs/agents/triage-labels.md`
- **Domain docs**: `docs/agents/domain.md`
- **Branch naming**: `feat/<issue-number>-<short-slug>`
- **Escalation label**: `needs developer attention`
- **Ready label**: `ready-for-agent` (spec approved; safe to implement)

### pathmc (this repo)

- **Environment**: `uv run` for all commands. Never system Python.
- **Tests**: `uv run pytest tests/test_<module>.py -x -v` targeted; `make test-fast` before PR ready.
- **Lint**: `make lint`
- **Do not modify existing test assertions** unless adding new tests or removing obsolete ones.

## Phase 1: State detection

### Resolve issue and PR

```bash
gh issue view $ISSUE_NUMBER --json title,body,state,labels,comments
gh pr list --search "#$ISSUE_NUMBER" --state all --json number,headRefName,state,title
gh pr view $PR_NUMBER --json number,title,body,state,headRefName,comments,statusCheckRollup,labels
```

If started from a PR: `gh pr checkout $PR_NUMBER`

### Marker prefixes (portable across repos)

| Marker | Purpose |
|--------|---------|
| `work-spec` | Spec published; records decisions |
| `work-summary` | Implementer summary after each push |
| `work-round` | Reviewer round comment |
| `work-approved` | Review loop complete |
| `work-escalation` | Handoff to human |

```bash
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("work-round"))] | length'
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("work-approved"))] | length'
gh issue view $ISSUE_NUMBER --json labels -q \
  '[.labels[].name | select(. == "ready-for-agent")] | length'
```

### State table (match top-to-bottom)

| Detected state | Action |
|----------------|--------|
| Issue closed or PR merged | Exit — done |
| Human pushed or commented since last automation | Escalate |
| Issue is `[Epic]` / `[Meta]` / `[proposal]` with open children | Pick one child (smallest scope); `work #child` — do not implement the parent |
| Issue open, no `ready-for-agent`, no `## Spec` in body | **Spec phase** |
| Issue has `ready-for-agent`, no PR | **Implement phase** (create branch + PR) |
| PR exists, CI failing | Fix CI → push → continue |
| PR exists, unaddressed `work-round` comment | Address review → push → continue |
| PR exists, CI green, no unaddressed review | **Review phase** (spawn reviewers) |
| PR has `work-approved` and CI passing | Exit — done |
| 3+ `work-round` markers | Escalate |

**Tie-breaking**: When several child issues or checklist items look equal, pick the first by number or top-to-bottom. Document the choice; do not ask the user.

### Dashboard (`work` with no issue)

```bash
gh issue list --label "ready-for-agent" --state open --json number,title
gh pr list --search "work-round OR work-summary" --state open --json number,title,headRefName
```

Summarise each item: issue/PR number, title, detected state, one-line next action.

## Phase 2: Spec

Run when the issue lacks a published spec and `ready-for-agent`.

1. Read the issue, linked epics, and relevant `docs/adr/`.
2. If acceptance criteria are already complete (module paths, explicit non-goals, test plan): skip grilling; go to step 4.
3. Otherwise run **`grill-with-docs`** (uses `grilling` + `domain-modeling`). Resolve design branches; do not implement code.
4. Run **`to-spec`**: publish spec to the issue body under `## Spec` (or create a child issue for umbrella parents).
5. Apply label `ready-for-agent`; remove `needs-human` if present.
6. Post issue comment with marker `<!-- work-spec -->` summarising decisions and out-of-scope items.

**Human gate**: If the issue has `API options` or unresolved design forks the grilling did not settle, apply `needs-human` and stop — do not label `ready-for-agent`.

## Phase 3: Implement

Run when issue has `ready-for-agent` and no open PR (or resuming a PR).

1. Read `## Spec` on the issue (source of truth).
2. Branch: `feat/<issue>-<slug>`; check out or create.
3. Run **`implement`** skill (uses **`tdd`** at seams named in the spec).
4. Run targeted tests + `make lint`.
5. Commit, push, open PR if needed:

```bash
gh pr create --title "Feat: <short description>" --body "$(cat <<'EOF'
## Summary
<1-2 sentences>

## Issue links
- Implements #<issue> (or Part of #<parent>)

## Spec
<Link or paste key acceptance criteria from issue>

## Test plan
- [ ] Targeted tests pass
- [ ] make test-fast passes
- [ ] make lint passes
EOF
)"
```

6. Post fixer summary on PR:

```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- work-summary -->
## Work summary

**Scope**: Implements #<issue> — <one line>
**Approach**: <why this shape>
**Not in scope**: <deliberate omissions>
EOF
)"
```

## Phase 4: Review loop

Spawn **two parallel** Task subagents following the **`code-review`** skill:

- **Standards** axis: repo conventions + smell baseline.
- **Spec** axis: diff vs issue `## Spec` and acceptance criteria.

Reviewer posts ONE comment starting with `<!-- work-round:$ROUND -->`. If no 🔴 or 🟡 findings, post `<!-- work-approved -->` instead.

```
round = count work-round markers (max 3)

while round < 3:
    wait_for_ci()
    spawn standards_reviewer and spec_reviewer in parallel
    if work-approved posted: break
    address findings; push; post work-summary
    round += 1

if round >= 3 and not approved: escalate()
```

Address all 🔴 and 🟡; skip 🟢 unless trivial. Never self-approve.

## Phase 5: Escalation

Same pattern as `fix-bug`: label PR + issue with `needs developer attention`, post `<!-- work-escalation -->` summary.

## Safety caps

| Cap | Limit |
|-----|-------|
| Review rounds per session | 3 |
| Pushes per session | 6 |
| CI wait | 30 minutes |

## Sub-skills (internal only)

| Phase | Skill |
|-------|-------|
| Spec / decisions | `grill-with-docs`, `to-spec` |
| Build | `implement`, `tdd` |
| Review | `code-review` |
| Bugs | `fix-bug` (separate entry point) |

## Copying to another repo

1. Copy `.agents/skills/work/SKILL.md`; update **Repo conventions**.
2. Install Matt Pocock sub-skills: `npx skills@latest add mattpocock/skills --skill grill-with-docs --skill to-spec --skill implement --skill tdd --skill code-review --skill grilling --skill domain-modeling --agent cursor --copy -y`
3. Run `setup-matt-pocock-skills` once, or copy `docs/agents/` from a configured repo.
4. Keep marker prefixes identical for portable resume logic.
