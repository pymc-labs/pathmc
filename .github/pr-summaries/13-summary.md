# PR: DSL `lag(var)` syntax for autoregressive terms (lag-1 only)

Closes #13

## Issue Summary

Replace the implicit `sales_lag1` column-name convention with an explicit `lag(sales)` syntax in the DSL, eliminating the `add_lags()` preprocessing step. Only lag-1 is supported by design.

## Root Cause

The prior approach relied on regex-matching `_lag\d+$` on column names to detect temporal dependencies — implicit and fragile. Users had to call `add_lags()` to create lag columns, then `dropna()` to remove the warm-up row, before passing data to `fit()`. The formula was not self-describing.

## Solution

Add `lag(var)` as a structural term in the DSL parser, wired through the compiler's scan infrastructure. The formula is now fully self-describing: `sales ~ spend + lag(sales)` requires no external preprocessing.

**Parser:** `lag(var)` is parsed as a structural term (not a real transform) with a new `lag_of` field on `Term`. Rejects parameters (`lag(x, k=2)`) and nested transforms (`lag(adstock(...))`) with clear errors.

**Compiler:** A new `_build_lag_map()` extracts lag terms from the AST. `_has_temporal_deps()` and the scan step function use this alongside the existing regex path for backward compatibility. Lag terms are wired to `prev_endo` or `prev_exog` carry state in the scan.

**Validation:** `lag()` terms require `panel=` to be passed to `fit()`.

**Deprecation:** `add_lags()` emits a `DeprecationWarning` but continues to work. The `_lag\d+$` regex path remains for backward compatibility.

## Changes Made

- `pathmc/parse.py`: Add `lag_of` field to `Term`; `_make_lag_term()` validates and builds lag terms
- `pathmc/compile.py`: Add `_build_lag_map()`; update `_has_temporal_deps()` and `_compile_scan_panel()` column classification and step function for AST-driven lag detection
- `pathmc/model.py`: Validate that `lag()` terms require `panel=`
- `pathmc/panel.py`: Add `DeprecationWarning` to `add_lags()`
- `AGENTS.md`: Mark `add_lags()` as deprecated in public API list
- `benchmarks/scan_vs_conv.py`: Update specs from `sales_lag1` to `lag(sales)`; remove `add_lags()` calls
- `tests/test_lag_syntax.py`: New test file — parsing, validation, and compilation tests for `lag()` syntax
- `tests/test_panel_smoke.py`: Replace `add_lags()` + `spend_lag1` with `lag(spend)` syntax
- `tests/test_panel_do.py`: Same
- `tests/test_panel_engines.py`: Replace `add_lags()` + `sales_lag1` with `lag(sales)` syntax; add `lag_requires_panel` test
- `tests/test_add_lags.py`: Add `pytestmark` to filter deprecation warnings (function still works)

## Testing

- [x] All 221 fast tests pass
- [x] All 27 slow panel tests pass (excluding pre-existing Bernoulli do() bug on main)
- [x] 10 new tests for lag() parsing, validation, and compilation
- [x] Backward compatibility: `add_lags()` + `_lag1` column names still work (with deprecation warning)
- [x] Linter clean (`ruff check` + `ruff format`)

## Notes

- `lag_fill` parameter for `fit()` (configuring initial carry values) is deferred to a follow-up — initial carry defaults to zero for endogenous lags, first-observation for exogenous lags (matching current scan behaviour)
- The pre-existing `TestPanelBernoulli::test_panel_bernoulli_works` failure is unrelated (fails on `main` too — a `pm.do()` type mismatch with Bernoulli models)
