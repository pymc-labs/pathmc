# Triage labels (agent workflow)

Labels used by the `work` and `fix-bug` skills. Create on GitHub if missing.

| Label | Meaning |
|-------|---------|
| `ready-for-agent` | Spec approved; safe for autonomous implement + review loop |
| `needs-human` | Blocked on a design decision only a human can make |
| `needs developer attention` | Automation exhausted review rounds; handoff |

Apply `ready-for-agent` only after a `## Spec` section exists on the issue (or acceptance criteria are already agent-complete) and no open design forks remain.
