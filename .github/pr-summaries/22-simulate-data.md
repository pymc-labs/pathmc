# PR: Add `simulate()` function for model-based data generation

Closes #22

## Issue Summary

Users wanted to generate synthetic datasets directly from a pathmc model specification with known parameter values, rather than writing manual NumPy simulation code. This enables simulate-and-recover workflows, power analysis, and pedagogical dataset creation.

## Root Cause

The existing `fit()` function requires observed data for all endogenous variables — there was no way to build a generative model and draw from it with fixed parameters.

## Solution

Added a `simulate()` function that:

1. Parses the spec and compiles a generative PyMC model (all endogenous vars as free RVs)
2. Creates placeholder data for endogenous variables (only exogenous data required from user)
3. Fixes all parameter RVs (betas, sigmas, etc.) at user-provided values via `pm.do()`
4. Draws the endogenous variables via `pm.draw()`, which evaluates the graph consistently — respecting multi-equation chains
5. Returns a DataFrame with original exogenous columns + simulated endogenous columns

## Changes Made

- `pathmc/model.py`: Added `simulate()` function after `fit()` with full docstring, parameter validation, and helpful error messages
- `pathmc/__init__.py`: Exported `simulate` in the public API
- `tests/test_simulate_data.py`: 13 tests covering basic usage, multi-equation models, non-Gaussian families (Bernoulli, Poisson), validation errors, and edge cases
- `docs/examples/data_simulation.qmd`: New example page with 3 examples of increasing complexity (simple regression, mediation chain, binary outcomes)
- `.github/pr-summaries/22-simulate-data.md`: This file

## Testing

- [x] All 13 new tests pass
- [x] All 283 existing fast tests pass (no regressions)
- [x] Ruff linting and formatting clean

## Notes

- Residual covariance (`~~`) models are not yet supported by `simulate()` — raises `NotImplementedError` with guidance to use NumPy simulation instead
- Panel models are not yet supported — a follow-up could extend `simulate()` for panel data
