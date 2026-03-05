# PR: Docs — Interaction terms DSL reference and moderation example

Closes #62

## Issue Summary

PR #61 added `X:Z` interaction term support to the DSL but left the documentation referencing the old precomputed-column workaround. The moderation example notebook and DSL reference needed updating to reflect the native syntax.

## Root Cause

The `model_specification.qmd` had no mention of interaction terms, and `moderation.qmd` explicitly stated that pathmc "does not yet parse interaction syntax" — both outdated after the #50/#61 merge.

## Solution

1. Added an "Interaction terms" section to the DSL reference covering `X:Z`/`X:Z:W` syntax, labeled interactions, unsupported syntax (`X*Z`, transforms in interactions), and equation rendering.
2. Rewrote `moderation.qmd` to use native `X:Z` syntax and `model.cate()` instead of manual column construction and `do()` workarounds. Added a comparison section explaining why the symbolic approach matters for correct `do()` propagation.

## Changes Made

- `docs/concepts/model_specification.qmd`: Added "Interaction terms" section between "Regression equations" and "Defined parameters"
- `docs/examples/moderation.qmd`: Rewrote to use native `X:Z` DSL syntax and `model.cate()` for CATE computation; added symbolic vs precomputed comparison section

## Testing

- [x] All 270 fast tests pass
- [x] Ruff lint and format pass
- [x] `model_specification.qmd` renders successfully
- [x] `moderation.qmd` renders and executes cleanly (all 11 cells)

## Notes

- The introspection rendering (`X × Z` in plain text, `\times` in LaTeX) is documented in the new DSL reference section; no separate introspection guide exists to update.
- The `_quarto.yml` and `examples/index.qmd` required no changes since `moderation.qmd` already existed and is auto-listed.
