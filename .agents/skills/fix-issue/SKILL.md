---
name: fix-issue
description: >-
  Autonomous bug-fix workflow for pathmc. Guides the orchestrator/fixer through
  state detection, implementation, test/lint validation, and a bounded
  fix/review loop with a reviewer subagent. Use when fixing GitHub issues
  autonomously, or when the user says "fix issue", "bugfix", or references
  fixing a specific issue number like "fix #149" or "bugfix #149".
---

# Fix Issue — Autonomous Bug-Fix Workflow

## Repo conventions

- **Environment**: `uv run` for all commands. Never system Python.
- **Tests**: `uv run pytest tests/test_<module>.py -x -v` for targeted; `make test-fast` for full suite (skips MCMC).
- **Lint**: `make lint` (runs ruff, ruff-format, mypy, license checks via `prek`).
- **Branch naming**: `fix/<issue-number>-<short-slug>` (e.g. `fix/149-non-gaussian-guards`).
- **Commit messages**: Imperative mood, `Fixes #N` in body. Focus on *why* not *what*.
- **PR body**: Include `Fixes #N`, a summary of the approach, and a test plan.
- **Do not modify existing test assertions** unless adding new tests or removing obsolete ones.
- **Architecture**: See `AGENTS.md` at repo root for module structure and principles.

## Phase 1: State detection

Before doing anything, determine the current state of the issue/PR:

```bash
# Does a PR already exist?
gh pr list --search "Fixes #$ISSUE_NUMBER" --json number,headRefName,state

# If PR exists, get details:
gh pr view $PR_NUMBER --json comments,statusCheckRollup,labels,headRefName

# Count round markers:
gh pr view $PR_NUMBER --json comments -q \
  '[.comments[].body | select(test("pathmc-autofix-round"))] | length'

# CI status:
gh pr checks $PR_NUMBER
```

| Detected state | Action |
|----------------|--------|
| No PR exists | Full flow: understand → plan → implement → push → create PR → review loop |
| PR exists, CI passing, no unaddressed review | Enter review loop |
| PR exists, unaddressed review comment (has marker) | Address findings → push → wait CI → continue |
| PR exists, CI failing | Read failures → fix → push → wait CI → continue |
| 3+ round markers already present | Escalate immediately |
| PR merged or issue closed | Exit — already resolved |
| Human has pushed/commented since last automation activity | Escalate — don't overwrite human work |

**Fallback when markers are missing**: Use `max(marker_count, substantive_review_comment_count)` as round estimate. If ambiguous, treat as round 0 and start fresh.

## Phase 2: Fix implementation

1. **Read the issue thoroughly** — body, all comments (especially triage bot comments), any linked docs.
2. **Identify root cause** — trace the actual code path, don't guess from the description alone.
3. **Consider 2-3 approaches** — pick the smallest correct diff. Prefer fixing the shared function once over patching each caller.
4. **Implement** — follow AGENTS.md style. No narrating comments. Type hints on public functions.
5. **Validate**:
   ```bash
   uv run pytest tests/test_<relevant_module>.py -x -v
   make lint
   ```
6. **Commit and push**:
   ```bash
   git add -A
   git commit -m "$(cat <<'EOF'
   Short imperative summary

   Fixes #N. Explanation of why this approach was chosen.
   EOF
   )"
   git push -u origin HEAD
   ```
7. **Create PR** (if new):
   ```bash
   gh pr create --title "Fix: <short description>" --body "$(cat <<'EOF'
   ## Summary
   Fixes #N. <1-2 sentences on the approach.>

   ## Test plan
   - [ ] Targeted tests pass
   - [ ] `make test-fast` passes
   - [ ] `make lint` passes
   EOF
   )"
   ```

## Phase 3: Review loop

### Spawning the reviewer subagent

Spawn a Task subagent with these characteristics:
- **Model**: `composer-2.5`
- **Fresh context**: The reviewer has NO knowledge of your fix reasoning.
- **Prompt**: Include the PR number, the diff (or instruct it to read via `gh pr diff`), and the review criteria below.
- **Role boundary**: The reviewer NEVER modifies code. Its only output is a PR comment.

### Reviewer prompt template

> You are an independent code reviewer for the `pathmc` project. You have never seen this code before.
>
> **Your job**: Review PR #$PR_NUMBER. Post ONE review comment on the PR with your findings. You NEVER modify code, push, or run commands other than reading the diff.
>
> **Read the diff**: `gh pr diff $PR_NUMBER`
>
> **Review criteria** (check each):
> - Correctness: Does the fix address the root cause? Any logic errors?
> - Regressions: Could this break existing behavior? Check callers of modified functions.
> - Edge cases: Are boundary conditions handled?
> - Test coverage: Are the new/changed paths tested?
> - Style: Follows ruff formatting, type hints on public APIs, no narrating comments.
> - Performance: No unnecessary O(n²) or repeated computation.
> - Error messages: Name the problem AND suggest a fix.
>
> **Severity levels**:
> - 🔴 **Must fix**: Correctness bug, regression, or missing guard. Blocks merge.
> - 🟡 **Should fix**: Missing test, unclear naming, style violation.
> - 🟢 **Nitpick**: Optional improvement, not blocking.
>
> **Output format**: Post a PR comment via `gh pr comment $PR_NUMBER --body "..."`.
> Start the comment with the marker: `<!-- pathmc-autofix-round:$ROUND -->`
> If you find NO actionable issues (no 🔴 or 🟡), post an approval comment WITHOUT the marker:
> "Reviewed — no actionable issues found. Approving."
>
> Be specific and actionable. Another agent will read your comments and fix what you raise.

### Loop control

```
round = count_existing_markers()  # or internal counter during live session

while round < 3:
    wait_for_ci()  # gh pr checks --watch $PR_NUMBER
    spawn_reviewer(round + 1)
    verdict = read_reviewer_output()

    if verdict == "approved" (no marker posted):
        break  # Done — natural termination

    address_review_findings()
    push_fixes()
    round += 1

if round >= 3 and not approved:
    escalate()
```

### Addressing review findings

- Read the reviewer's comment carefully.
- Address all 🔴 (must fix) and 🟡 (should fix) items.
- 🟢 (nitpick) items: fix if trivial, skip if not.
- Run tests and lint before pushing.
- Do NOT post your own review or evaluate your changes — the reviewer will check on the next pass.

## Phase 4: Escalation

When the loop is exhausted (3 rounds) or a safety cap is hit:

```bash
# Label the PR
gh pr edit $PR_NUMBER --add-label "needs developer attention"

# Label the linked issue
gh issue edit $ISSUE_NUMBER --add-label "needs developer attention"

# Post summary comment
gh pr comment $PR_NUMBER --body "$(cat <<'EOF'
<!-- pathmc-autofix-escalation -->
## Automated fix escalation

This PR has been through 3 review/fix rounds without full resolution.
Remaining findings from the last review are above.

**Handing off to a developer.** The automated agents were unable to fully
resolve the reviewer's concerns. Please review the outstanding items
and either fix or close as acceptable.
EOF
)"
```

## Safety caps (hard limits)

These apply regardless of marker state:

| Cap | Limit | On breach |
|-----|-------|-----------|
| Reviewer spawns per session | 3 | Escalate |
| Total pushes per session | 6 | Escalate |
| Single CI wait | 30 minutes | Post timeout note, exit |
| Consecutive reviewer failures | 2 | Escalate |

## Robustness notes

- **Internal counter is primary** during a live session. Markers are for observability and resume.
- **On resume with missing markers**: estimate round from comment count. If ambiguous, start at round 0.
- **Reviewer subagent failure**: If it returns without posting a comment, check the PR. If no comment appeared, count as a failed round.
- **Human intervention**: If a human has pushed commits or posted comments since the last automation activity, escalate rather than overwriting.
- **Already resolved**: If the issue is closed or PR is merged, exit immediately.
