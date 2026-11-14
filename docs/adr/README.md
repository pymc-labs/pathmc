# Architecture decision records

An ADR records a decision a human has closed, so the next agent does not re-decide it. Add one only when a human closes one of the four questions below. Do not pre-fill records from the architecture principles in `AGENTS.md`. This file is the template. It is not a decision, and it does not mean a question is settled.

## When to stop

Stop when no accepted ADR covers the question and the change would:

- break the public API
- change a numerical tolerance
- add a dependency
- reject an alternative

Apply `needs-human` only when that stop records the decision, the recommendation and its consequence, the other live option and its cost, the fact that would flip it, what is already settled, and the next owner. A label without those fields is not a completed stop.

Not every non-obvious line is one of these. Rejecting an alternative means a later agent would otherwise re-litigate that choice. A routine implementation choice continues. Output that contradicts an accepted ADR is surfaced, not silently overridden. See `docs/agents/domain.md`.

## Record

One file per decision: `docs/adr/NNNN-short-slug.md`. Put the status in the title, `accepted` or `superseded`. Supersede by adding a new record that names the one it replaces. Leave the old decision text in place. The only edit to the old record is the title status and a pointer to the new file.

```markdown
# NNNN. Short title (accepted)

Decision: what was closed.

Why: the reason, including the alternative that was rejected.

What would reverse it: the observation that would reopen the question.
```
