# PR: Remove incorrect ancestry pre-filter in collider_warnings()

Closes #148

## Issue Summary

`collider_warnings()` in `identify.py` had an ancestry pre-filter that silently skipped valid colliders not descended from either treatment or outcome, causing false negatives.

## Root Cause

The ancestry check (`treatment_ancestor or outcome_ancestor`) before calling `_is_collider_on_path` incorrectly assumed that a collider must be a descendant of the treatment or outcome. In DAGs like `T <- A -> C <- B -> Y`, collider C is not a descendant of T or Y, so the pre-filter evaluated `False or False` and skipped the `_is_collider_on_path` call entirely.

Additionally, each side of the ancestry check was redundant (`var in nx.descendants(dag, X)` and `X in nx.ancestors(dag, var)` test the same relationship).

## Solution

Removed the ancestry pre-filter entirely. The `len(parents) >= 2` check is already a correct and sufficient guard — a node with fewer than 2 parents can never be a collider. After that check, `_is_collider_on_path` is called directly.

## Changes Made

- `pathmc/identify.py`: Removed the 6-line ancestry pre-filter block (lines 253–259), keeping only `len(parents) >= 2` + `_is_collider_on_path`.

## Testing

- [x] Existing tests pass (388 passed)
- [x] Verified with the counterexample DAG from the issue (`T <- A -> C <- B -> Y`)
- [x] Lint and format clean

## Notes

This is a ~3-line net deletion. No new dependencies or API changes.
