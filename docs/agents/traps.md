# Traps

Append-only. One failure per bullet: what bit, and the file or command that now prevents it. Do not delete a trap because the session that hit it is over.

- `make docs` renders the committed `_freeze/` cache. A local preview is not the edit just made. Refresh with `make freeze-page` and commit `_freeze/`. See the docs section of `AGENTS.md`.
- Simulate-and-recover is blind to a silent data-generating bug, because the same code generates and checks the data. Prefer a hand-written NumPy oracle or a round-trip through a different entry point. See the `simulate()` paragraph in `AGENTS.md`.
- `make test` on the feature head is not the merge GitHub tests. Final validation is `make check-pr` or `make check-merge`. See the validation paragraph in `AGENTS.md`.
- git-ai ignores this repo until `allow_repositories` includes it. See the git-ai section of `CONTRIBUTING.md`.
