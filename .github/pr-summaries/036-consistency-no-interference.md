# PR: Add consistency and no-interference to causal assumptions

Closes #36

## Issue Summary

The causal assumptions page omits two important identification assumptions — consistency (well-defined interventions) and no interference (SUTVA) — that hold implicitly in pathmc's structural equations but should be stated explicitly.

## Root Cause

The original assumptions list covered DAG correctness, unmeasured confounders, model misspecification, and positivity, but did not mention the two assumptions that connect potential outcomes to observed data.

## Solution

Added two new numbered items to the "Causal assumptions and limitations" section, inserted after "No unmeasured confounders" and before "No model misspecification" to maintain a logical grouping (identification assumptions first, then statistical/practical ones).

## Changes Made

- `docs/concepts/causal_inference.qmd`: Added items 3 (Consistency) and 4 (No interference / SUTVA) to the assumptions list; renumbered existing items 3–4 to 5–6.

## Testing

- [x] Existing tests pass (222 passed, 96 deselected)
- [x] Linting clean (ruff check + format)

## Notes

- Consistency explains why `do(X = x)` must correspond to a single well-defined intervention.
- No interference notes relevance to panel models with potential cross-unit spillover.
