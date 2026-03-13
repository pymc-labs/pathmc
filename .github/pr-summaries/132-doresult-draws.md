# PR: Add `DoResult.draws()` and fix private attribute accesses in notebooks

Closes #132

## Issue Summary

Documentation notebooks accessed `DoResult._values` and `PathModel._idata` directly, bypassing the public API. A public `.draws(var)` method was needed as a prerequisite for the ArviZ plotting migration (#123).

## Root Cause

`DoResult` only exposed `.mean(var)`, `.hdi(var)`, `.by_time(var)`, and subtraction. Notebooks that needed raw posterior draws (for KDE plotting, decomposition, etc.) had to access the private `._values` dict. Similarly, three notebooks accessed `model._idata` instead of capturing the return value of `model.fit()`.

## Solution

1. Added `DoResult.draws(var)` — a public method returning `self._values[var]`, mirroring the `EffectResult.draws` pattern.
2. Replaced all 15 occurrences of `._values[var]` with `.draws(var)` across 4 notebooks.
3. Changed 3 notebooks to capture `idata = model.fit(...)` and use `idata` instead of `model._idata`.

## Changes Made

- `pathmc/simulate.py`: Added `DoResult.draws(var)` method
- `docs/examples/vaccine_surrogates.qmd`: `._values` → `.draws()`
- `docs/examples/saas_funnel.qmd`: `._values` → `.draws()`
- `docs/examples/mmm.qmd`: `._values` → `.draws()` (6 occurrences)
- `docs/examples/ate_estimation.qmd`: `._values` → `.draws()`
- `docs/examples/moderation.qmd`: `model._idata` → `idata`
- `docs/examples/did.qmd`: `model._idata` → `idata`
- `docs/examples/data_simulation.qmd`: `model_med._idata` → `idata_med`

## Testing

- [x] Existing tests pass (354 passed)
- [x] No new tests needed (method is trivially correct)
- [x] Ruff lint and format clean

## Notes

This is the prerequisite for the ArviZ plotting migration tracked in #123. Follow-up sub-issues:
- #133: Replace `gaussian_kde` boilerplate with `az.plot_kde`
- #134: Replace manual histograms with `az.plot_posterior`
- #135: Replace manual HDI `fill_between` with `az.plot_hdi`
