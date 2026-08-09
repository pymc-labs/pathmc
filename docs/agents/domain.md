# Domain Docs

How agent skills should consume pathmc domain documentation.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root (created lazily by `grill-with-docs` / `domain-modeling` when needed).
- **`docs/adr/`** — read ADRs that touch the area you're about to work in.

If these files don't exist, proceed silently. The grilling phase creates them when terms or decisions are resolved.

## Use the glossary's vocabulary

When naming domain concepts in issues, tests, or APIs, use terms from `CONTEXT.md` when present.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding.
