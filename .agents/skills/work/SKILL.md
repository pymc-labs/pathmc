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

Skills are plain Markdown (`SKILL.md`). Any agent harness — Cursor, VS Code extensions, or a CLI — reads the same files; only the skill discovery path differs per tool. State lives on GitHub (labels, issue bodies, PR comment markers), not in local agent state, so work resumes across machines and tools.

## Phase 0: Bootstrap (first use on a machine or repo)

**pathmc**: `docs/agents/` is committed — skip this phase after clone.

**Another repo** copying `work`: before Phase 1, check that `docs/agents/issue-tracker.md` exists.

| Check | Action |
|-------|--------|
| `docs/agents/issue-tracker.md` present | Continue to Phase 1 |
| Missing, `git remote` points at GitHub | Copy templates from `.agents/skills/setup-matt-pocock-skills/issue-tracker-github.md`, `domain.md`, and `triage-labels.md` into `docs/agents/` without an interview; add the Agent skills block to `AGENTS.md` |
| Missing, not GitHub or ambiguous | Stop and tell the developer to run `setup-matt-pocock-skills` once |

The setup skill does **not** run automatically on its own. `work` performs this lightweight bootstrap when config is missing. Full interactive setup is only needed for non-GitHub trackers or custom label vocabularies.

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

| Marker | Who posts | Purpose |
|--------|-----------|---------|
| `work-spec` | Orchestrator | Spec published on the issue |
| `work-summary` | Implementer | Summary after each implementation push |
| `work-round:$N:standards` | Standards reviewer | Round *N* findings on repo conventions |
| `work-round:$N:spec` | Spec reviewer | Round *N* findings vs issue spec |
| `work-approved:standards` | Standards reviewer | Round *N* standards axis clean (no 🔴/🟡) |
| `work-approved:spec` | Spec reviewer | Round *N* spec axis clean (no 🔴/🟡) |
| `work-review-complete` | Orchestrator | Both axes approved; **CI not yet green** |
| `work-approved` | Orchestrator | Both axes approved **and CI green** — merge-ready |
| `work-escalation` | Orchestrator | Handoff to human |

**Comment style**: Marker comments are for humans reading the PR. Write in plain language — full sentences, concrete file or behaviour references. Do **not** use orchestrator shorthand (`axis clean`, `public export`, `spec axis`) without explaining what was checked and what changed.

```bash
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("work-round:[0-9]+:"))] | length'
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("<!-- work-approved -->"))] | length'
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("work-review-complete"))] | length'
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
| PR exists, unaddressed `work-round` on either axis | Address review → push → `work-summary` → continue |
| PR exists, CI green, both axes need another review pass | **Review phase** (spawn reviewers) |
| PR has `work-review-complete` and CI still failing/pending | Wait for CI or fix CI → continue |
| PR has `work-approved` and CI passing | Exit — done |
| 3+ `work-round` markers (any axis) | Escalate |

**Tie-breaking**: When several child issues or checklist items look equal, pick the first by number or top-to-bottom. Document the choice; do not ask the user.

### Dashboard (`work` with no issue)

```bash
gh issue list --label "ready-for-agent" --state open --json number,title
gh pr list --search "work-round OR work-summary OR work-review-complete" --state open --json number,title,headRefName
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

Each reviewer posts **one** PR comment on its axis. Use the templates below. Reviewers never modify code.

### Review comment quality

Review comments are the **primary handoff to the implementer agent**. If you spent effort finding a problem, convey that effort: enough context that the implementer can fix it without re-deriving your reasoning.

**Every 🔴 and 🟡 finding must include:**

| Field | What to write |
|-------|----------------|
| **Where** | File path and line(s), or symbol name (e.g. `DoResult.plot` in `simulate.py`) |
| **What** | What the code does today vs what it should do |
| **Why** | Cite the repo rule (`AGENTS.md`, convention) or spec line; say user/regression impact |
| **Fix** | Concrete steps — rename X, move Y, add test Z, change guard to … |

Quote a short hunk from the diff when it clarifies the issue. One-line bullet findings are too thin for agent handoff.

**Approval comments** should still be substantive: list what was reviewed (files/areas), note any 🟢 nits optionally, and confirm no 🔴/🟡. Do not post empty approvals.

When spawning reviewers via **`code-review`**, tell them: *comments are for an implementer agent, not a human skimming — prefer detail over brevity.*

### Reviewer output (one comment per axis)

**If 🔴 or 🟡 findings** — post round comment:

```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- work-round:1:standards -->
## Standards review (round 1)

Reviewed <scope — e.g. `pathmc/simulate.py`, `pathmc/panel.py`, `tests/test_do_plot.py`> against `AGENTS.md` and CONTRIBUTING conventions.

### Must fix

#### 🔴 <short title>
- **Where**: `<path>:<lines>` (`<symbol>`)
- **What**: <current behaviour in plain language>
- **Why**: <which standard or smell; why it matters>
- **Fix**: <numbered steps or exact change to make>

```diff
<optional: short quoted hunk from the PR diff>
```

### Should fix

#### 🟡 <short title>
- **Where**: …
- **What**: …
- **Why**: …
- **Fix**: …

### Nits (optional)
- 🟢 <finding + optional one-line suggestion>
EOF
)"
```

Use `work-round:$N:spec` for the spec reviewer. Quote the spec requirement for each finding. Add a **Requirements checked** subsection listing spec items verified (even when passing) so the implementer sees coverage.

**If no 🔴 or 🟡** — post per-axis approval (not the umbrella `work-approved`):

```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- work-approved:standards -->
## Standards review (round 1)

Reviewed <files/modules> against `AGENTS.md` and code-smell baseline.

**Checked**: <bullets — e.g. public API surface, error messages, lazy imports, test style, docstrings on new public methods>.

No must-fix or should-fix items. <Optional: one or two 🟢 nits with brief suggestions.>
EOF
)"
```

Use `work-approved:spec` for the spec reviewer. Include **Requirements checked** (bullets mapping spec items to what you verified in the diff) and the issue number.

### Implementer: addressing review

When `work-round` comments exist, the implementer must:

1. Read **every** finding block (Where / What / Why / Fix) before editing.
2. In the follow-up `work-summary`, reference each 🔴/🟡 by title and state what changed (or why deferred with reason).
3. Not close a finding with a one-word fix — match the specificity the reviewer provided.

### Orchestrator loop

```
round = count of work-round markers on the PR (max 3)

while round < 3:
    wait_for_ci()   # gh pr checks --watch, cap 30 min

    spawn standards_reviewer(round + 1) and spec_reviewer(round + 1) in parallel

    if either axis posted work-round (🔴/🟡):
        address all 🔴 and 🟡; skip 🟢 unless trivial
        run tests + lint; push
        post work-summary (what changed in response to review)
        round += 1
        continue

    # Both axes posted work-approved:standards and work-approved:spec
    wait_for_ci()

    if CI green:
        post umbrella work-approved (template below); break
    else:
        post work-review-complete (template below)
        wait_for_ci()   # cap 30 min total CI wait for this session
        if CI green: post umbrella work-approved; break
        else: stop — do not post work-approved until CI is green

if round >= 3 and not umbrella work-approved: escalate()
```

Never post umbrella `<!-- work-approved -->` while CI is failing or still pending. Use `work-review-complete` instead.

### Orchestrator: review complete, CI pending

Post when both axes approved the code but required CI checks are not all green yet:

```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- work-review-complete -->
## Review complete — waiting on CI

Standards and spec reviews are clean for round <N>.

**Standards**: <one sentence — e.g. "No convention issues; one fix landed in <sha> (removed internal helper from `__all__`).">
**Spec**: <one sentence — e.g. "Implements #111 — trajectory plot, observed overlay, cross-sectional redirect, tests.">

CI is still running or failing: <list pending or failed check names>. Merge when CI is green; umbrella `work-approved` will follow.
EOF
)"
```

### Orchestrator: merge-ready approval

Post **only** when both axes approved **and** required CI checks are green:

```bash
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- work-approved -->
## Ready to merge

**Issue**: Implements #<issue> — <one line on what shipped>.

**Standards**: <one sentence on review outcome; cite fix commit if any>.
**Spec**: <one sentence confirming spec requirements are met>.

**CI**: All required checks passing. Review loop complete.
EOF
)"
```

Address all 🔴 and 🟡 before any approval marker. Never self-approve an axis you implemented without an independent reviewer comment on that axis.

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

## Bringing `work` to a new repo

An agent asked to port this stack should follow this checklist. Do **not** run `setup-matt-pocock-skills` unless the target repo is non-GitHub or needs custom triage labels — on a normal GitHub repo, Phase 0 or copying `docs/agents/` is enough.

### What to copy

| Copy | Why |
|------|-----|
| `.agents/skills/work/SKILL.md` | The orchestrator |
| `.agents/skills/fix-bug/SKILL.md` | Bug workflow (separate entry point, same loop pattern) |
| Matt sub-skills (see below) or run `npx skills@latest add mattpocock/skills --skill grill-with-docs --skill to-spec --skill implement --skill tdd --skill code-review --skill grilling --skill domain-modeling --copy -y` | Internal phases `work` delegates to |
| `docs/agents/issue-tracker.md`, `triage-labels.md`, `domain.md` | Repo config sub-skills read |
| `skills-lock.json` | Optional; tracks vendored skill versions for `npx skills update` |

Update the **Repo conventions** section in `work` and `fix-bug` for the target repo (test commands, branch naming, lint). Add an **Agent skills** block to the target's `AGENTS.md`.

Keep marker prefixes (`work-*`, `fix-bug-*`) identical so GitHub state is portable.

### Config: three ways to get `docs/agents/`

| Situation | What to do |
|-----------|------------|
| **pathmc itself** | Nothing — `docs/agents/` is already committed. No setup ever. |
| **New GitHub repo** | Copy `docs/agents/` from pathmc and edit repo conventions; **or** let `work` Phase 0 auto-copy from `.agents/skills/setup-matt-pocock-skills/issue-tracker-github.md` on first run. |
| **GitLab, Linear, local markdown, or custom labels** | Run `setup-matt-pocock-skills` **once**, interactively, with a human. |

### Do we need `setup-matt-pocock-skills`?

**Usually no.** It is a human-facing wizard for edge cases. pathmc never ran it — we wrote `docs/agents/` directly from the templates that ship inside the setup skill folder. That produces the same result as running setup on a GitHub repo and accepting all defaults.

Keep the setup skill vendored so an agent *can* invoke it when the target repo is not plain GitHub. For the common case (GitHub + default labels), copying `docs/agents/` or Phase 0 bootstrap is faster and needs no human in the loop.

### After copying

1. Edit **Repo conventions** in `work` and `fix-bug`.
2. Create GitHub labels from `docs/agents/triage-labels.md` if they do not exist (`ready-for-agent`, `needs-human`, `needs developer attention`).
3. Pilot: `work #<small-issue>` on a branch.
