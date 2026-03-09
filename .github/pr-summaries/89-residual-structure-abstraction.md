# PR: Residual structure abstraction (Phase 1)

Closes #89

## Issue Summary

The `~~` residual covariance was hardcoded as LKJ Cholesky with no abstraction layer, no `mu_{var}` deterministics for block variables, and broken `do()` propagation through block variables to downstream equations.

## Root Cause

`_compile_residual_block` emitted LKJ + MvNormal inline without creating `mu_{var}` deterministics or registering block variables in `endogenous_rvs`. This meant downstream equations resolved block variable predictors from raw data tensors (not the model graph), so `pm.do()` interventions on block variables couldn't propagate. Additionally, blocks were compiled after all non-block variables, so even if they were registered, downstream equations were already compiled.

## Solution

Phase 1 of the residual structure abstraction — a refactor that extracts the LKJ logic into a pluggable `ResidualStructure` protocol and fixes the `mu_{var}` / `endogenous_rvs` / `do()` gaps.

## Changes Made

- `pathmc/residuals.py` (new): `ResidualStructure` protocol and `LKJResidual` implementation. The protocol owns only the covariance parameterization and likelihood emission; coefficient betas and mu construction stay in the main compiler.
- `pathmc/compile.py`: Refactored `_compile_residual_block` to (1) create `pm.Deterministic(f"mu_{var}", mu)` for each block variable, (2) register block variables in `endogenous_rvs` so downstream equations wire through the model graph, and (3) delegate covariance + likelihood to `LKJResidual`. Moved block compilation into the topological order loop so blocks are compiled before downstream variables that depend on them. Block members are processed in topological order within the block to handle intra-block dependencies correctly.
- `pathmc/simulate.py`: Updated `run_do_pymc` to detect block variables (endogenous, no free RV, not latent) and handle them: map intervention keys through `mu_{var}`, skip free RV replacement in kind="mean", and compute `mu_{var}` deterministics for block variables in kind="predictive".
- `pathmc/introspect.py`: Updated `build_priors` to show `chol_{block_name}` entries for residual blocks with the LKJ prior description.

## Testing

- [x] All 351 existing fast tests pass
- [x] All 133 existing slow (sampling) tests pass
- [x] Verified `mu_{var}` deterministics exist for block variables
- [x] Verified LKJ prior visible in `model.priors()` output

## Notes

- This is Phase 1 only (pure refactor + behavioral fixes). Phase 2 (alternative structures like Diagonal/LowRank, `residual_structure=` parameter, full prior customization) and Phase 3 (panel `~~` support) are separate follow-ups.
- Block variables use `mu_{var}` deterministics for `do()` propagation rather than individual free RVs. This is a pragmatic choice since the observed MvNormal block can't easily be split into separate free RVs.
- The `sigma_{var}` prior entries for block variables are retained in introspection for backward compatibility, even though they aren't used by the compiler (the covariance is modeled jointly by LKJ). Phase 2 should clean this up.
