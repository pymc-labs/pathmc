# PR: Add interaction term support (X:Z syntax) to DSL

Closes #50

## Issue Summary

The parser did not support `X:Z` interaction syntax. Users had to pre-compute interaction columns in their data, which meant `do()` could not automatically recompute the interaction when a constituent variable was intervened on.

## Root Cause

The DSL parser, compiler, and graph layer had no concept of interaction terms — only plain variables, labeled coefficients, and transforms.

## Solution

Added end-to-end interaction term support across the full stack:

- **Parser**: Detects `:` in terms and creates `Term` objects with an `interaction_of` tuple holding constituent variable names (e.g., `("X", "Z")`). The `variable` field becomes the column name `"X:Z"`.
- **Graph**: Interaction terms add edges from each constituent variable to the LHS (not from `"X:Z"` as a node), keeping the DAG clean.
- **Compiler**: Interaction terms are computed as **symbolic products** of their constituents' `pm.Data` or upstream free RVs, ensuring `pm.do()` interventions propagate correctly through interactions. Both the cross-sectional `_build_mu_symbolic` and the scan-panel `step_fn` handle interactions.
- **Design matrix**: Updated missing-variable detection to check constituent variables (not `"X:Z"` literal). Patsy natively handles `X:Z` syntax for the design matrix.
- **Introspect**: Equations display interactions as `X × Z` (plain) and `X \times Z` (LaTeX). DAG edges are drawn from constituents without duplicates.
- **Effects**: Standardization (`stdyx`) skips interaction terms since `sd(X*Z)` is not well-defined for this purpose.

## Changes Made

- `pathmc/parse.py`: Added `interaction_of` field to `Term`; added `_parse_interaction_term()` helper
- `pathmc/graph.py`: Handle `term.interaction_of` in `build_graph()` to add edges from constituents
- `pathmc/compile.py`: Added `_resolve_interaction_symbolic()` for symbolic product computation; updated `_build_mu_symbolic()`, `build_design_matrix()`, `_term_base_vars()`, and scan `step_fn`
- `pathmc/model.py`: Fixed missing-variable check to handle interaction terms
- `pathmc/effects.py`: Skip interaction terms in `build_standardized_effects()`
- `pathmc/introspect.py`: Updated `_format_term()`, `_format_term_latex()`, and `build_dag_viz()` for interactions
- `tests/test_interactions.py`: 34 new tests covering parsing, graph, design matrix, compilation, introspection, and sampling with do() propagation

## Testing

- [x] All 252 existing fast tests pass (no regressions)
- [x] 30 new fast tests pass (parsing, graph, design matrix, compilation, introspection)
- [x] 4 new slow tests added (sampling, coefficient recovery, do() propagation, moderation via CATE)
- [x] Linting passes (`ruff check && ruff format`)

## Notes

- Three-way interactions (`X:Z:W`) are parsed and compiled but not extensively tested
- The `X*Z` shorthand (main effects + interaction) is not supported because `*` is already used for labels (`label*variable`). Users should write `Y ~ X + Z + X:Z` explicitly
- Interactions with transforms (e.g., `adstock(X):Z`) are not supported; the parser rejects them with a clear error
