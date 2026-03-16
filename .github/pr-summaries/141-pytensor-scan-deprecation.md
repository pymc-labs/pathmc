# PR: Silence PyTensor scan deprecation warning in panel models

Closes #141

## Issue Summary

Panel models emitted a `DeprecationWarning` from PyTensor's `scan` function every time a panel model was compiled, warning that the return signature will change and advising to pass `return_updates=False`.

## Root Cause

PyTensor deprecated the old `scan()` return convention (returning a `(results, updates)` tuple) starting in v2.36.0. The codebase was unpacking both values even though `updates` was never used.

## Solution

Pass `return_updates=False` to all `pytensor.scan()` calls and adjust the return value unpacking to match the new API (results returned directly, no updates dict). Bumped the minimum PyMC version from `>=5.22.0` to `>=5.27.0` to guarantee `pytensor>=2.36.0` which includes the `return_updates` parameter.

## Changes Made

- `pathmc/compile.py`: Pass `return_updates=False` to `pytensor.scan()`, change unpacking from `results, _updates = ...` to `results = ...`
- `benchmarks/scan_vs_conv.py`: Same fix for all three `pytensor.scan()` calls (adstock-only, adstock+AR, multi-equation)
- `pyproject.toml`: Bump minimum PyMC from `>=5.22.0` to `>=5.27.0` (ensures `pytensor>=2.36.0`)

## Testing

- [x] Existing tests pass (388 passed)
- [x] Lint clean (ruff check + format)
- [x] Manual verification: panel model compiles with no scan deprecation warning
