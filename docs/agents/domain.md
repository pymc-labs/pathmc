# Domain Documentation

How contributors and agents should consume pathmc domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, when present.
- **`docs/adr/`** — read accepted ADRs that touch the area you are about to work in. `docs/adr/README.md` is the template and the stop rule, not a decision.
- **`docs/agents/traps.md`** — failures that have already bitten an agent, and the file or command that now prevents each one.

## Use the glossary's vocabulary

When naming domain concepts in issues, tests, or APIs, use terms from `CONTEXT.md` when present.

## Flag ADR conflicts

If output contradicts an accepted ADR in `docs/adr/`, surface the conflict explicitly rather than silently overriding.

## When no ADR covers the question

An ADR is added only when a human closes one of the four questions below. Do not pre-fill records from the architecture principles in `AGENTS.md`.

Stop when no accepted ADR covers the question and the change would break the public API, change a numerical tolerance, add a dependency, or reject an alternative a later agent would otherwise re-litigate. Do not apply `needs-human` unless that stop records the decision, the recommendation and its consequence, the other live option and its cost, the fact that would flip it, what is already settled, and the next owner. Do not invent the decision. A routine implementation choice is not one of those triggers.
