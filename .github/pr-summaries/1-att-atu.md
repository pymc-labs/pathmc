# PR: Add canonical ATT/ATU with subgroup-aware intervention integration

Closes #1

## Issue Summary

Add first-class `att()` and `atu()` convenience methods with canonical potential-outcomes semantics, backed by subgroup-aware intervention integration in the `do()` engine.

## Root Cause

The existing `do()` operator computes potential outcomes for all N data rows and averages over the full covariate distribution (correct for ATE). However, ATT and ATU require averaging over the covariate distribution of the treated and untreated subgroups respectively. There was no mechanism to restrict the empirical integration to a subset of rows.

## Solution

Added a `subgroup_indices` parameter to `run_do_pymc()` that subsets the `(chain, draw, N)` output arrays to `(chain, draw, n_sub)` before flattening. This works for both `kind="mean"` (via `compute_deterministics`) and `kind="predictive"` (via `sample_posterior_predictive`). The core computation is unchanged — only the result extraction is filtered.

`PathModel.att()` and `PathModel.atu()` are thin wrappers that select the appropriate subgroup rows (where treatment equals `treated_value` or `untreated_value`) and call `run_do_pymc` with those indices.

## Changes Made

- `pathmc/simulate.py`: Added `subgroup_indices` parameter to `run_do_pymc()`. When provided, endogenous variable draws are subset to the specified rows before flattening, and exogenous fill values use subgroup means.
- `pathmc/model.py`: Added `att()` and `atu()` methods to `PathModel` with full docstrings, parameter validation, and examples.
- `tests/test_att_atu.py`: Comprehensive test suite (28 tests) covering API surface, binary treatment baseline, non-default coding, interaction-driven ATT≠ATU divergence, both `kind` variants, and existing `ate()`/`cate()` backward compatibility.

## Testing

- [x] All 289 existing fast tests pass (no regressions)
- [x] All 28 new ATT/ATU tests pass (6 fast + 22 slow)
- [x] Linting clean (`ruff check` + `ruff format`)

## Notes

- Panel model support for ATT/ATU is deferred (`NotImplementedError` raised with clear message).
- The issue's "nice-to-have" integration policy flag (`mean_exogenous` vs `empirical_rows`) is not included in this PR but could be added as a follow-up.
