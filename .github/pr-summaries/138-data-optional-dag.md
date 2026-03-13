# PR: Allow data-free DAG exploration and identification

Closes #138

## Issue Summary

`pathmc.model(spec, data=df)` required a DataFrame upfront, blocking the common workflow of exploring a DAG's structure, checking identification, and visualizing equations before having data.

## Root Cause

`PathModel.__init__()` coupled construction with compilation — it immediately built design matrices and compiled the PyMC model, making `data` a hard requirement even though the underlying introspection and identification functions already worked without data.

## Solution

Make `data` optional in `model()` and `PathModel.__init__()`. When `data=None`:
- Parse spec, build graph, and compute default priors as usual
- Skip design matrix construction and PyMC compilation
- Introspection (`graph()`, `equations()`, `priors()`) and identification helpers work immediately
- Data-requiring methods (`fit()`, `do()`, `design()`, etc.) raise `RuntimeError` with an actionable message
- `set_priors()` updates priors without recompilation

## Changes Made

- `pathmc/model.py`: Made `data` parameter optional (`pd.DataFrame | None = None`) in both `model()` factory and `PathModel.__init__()`. Added `_require_data()` guard method. Applied guard to 17 data-requiring methods. Adapted `set_priors()` to skip recompilation when no data. Guarded `pymc_model` property.
- `tests/test_data_optional.py`: New test file with 34 tests covering data-free creation, introspection, identification, RuntimeError guards for all data-requiring methods, `set_priors()` without data, and regression test for data-bound models.
- `.github/pr-summaries/138-data-optional-dag.md`: This PR summary.

## Testing

- [x] All 34 new tests pass
- [x] All 388 existing fast tests pass (no regressions)
- [x] `ruff check --fix && ruff format` passes clean

## Notes

- A `with_data()` convenience method is explicitly out of scope per the issue; trivial to add later.
- No changes to any existing test files.
- No changes to lower-level modules (parser, graph, identify, introspect).
