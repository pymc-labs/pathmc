# PR: Unify panel do() engines via scan-based generative models

Closes #10

## Issue Summary

pathmc had three separate panel `do()` engines (numpy, batched, scan) with ~1200 lines of
simulation code. The issue called for unifying them by compiling panel models with temporal
dependencies directly as `pytensor.scan`-based generative models, letting `pm.do()` handle
interventions natively.

## Root Cause

The three engines existed because the flat (convolution-based) compiler produced models
where temporal state (adstock carry, lag feedback) was baked into the data, not the model
graph. `pm.do()` couldn't propagate interventions through time, so each engine reimplemented
temporal propagation differently.

## Solution

Move temporal structure into the generative model itself via `pytensor.scan`:

1. **Scan compiler** (`compile.py`): Panel models with adstock transforms or lag terms are
   now compiled with `pytensor.scan`. The scan step function computes all equations in
   topological order per time step, carrying adstock state and lag values as scan outputs.
   Free RVs have shape `(n_times, n_units)`.

2. **Unified do()** (`simulate.py`): A single `run_do_panel_unified()` function replaces
   all three engines. It calls `pm.do()` on the scan generative model and extracts
   per-time-step results. ~1200 lines of engine code removed.

3. **Transform.step()** (`transforms.py`): Added `step()` and `has_state` to the Transform
   interface for use in the scan body. `Adstock.step()` implements the recursive
   `y_t = x_t + decay * y_{t-1}`.

4. **Exogenous lag carry**: The scan model correctly handles lag columns that reference
   exogenous variables (e.g., `spend_lag1`), carrying previous exogenous values through
   scan state so `do(set={"spend": ...})` propagates correctly.

## Benchmark Results

Phase 1 benchmarks (`benchmarks/scan_vs_conv.py`) showed scan-based estimation is 3-7x
slower than convolution-based. The decision was to proceed anyway, accepting the cost for
architectural simplicity, with the expectation that upstream PyTensor improvements will
close the gap.

## Changes Made

- `pathmc/compile.py`: Added `PanelScanInfo`, `_has_temporal_deps()`, `_compile_scan_panel()`,
  and helpers for scan-based compilation (+472 lines)
- `pathmc/simulate.py`: Removed 3 panel engines and helpers (~1200 lines), added
  `run_do_panel_unified()` (~150 lines). Net -1281 lines.
- `pathmc/transforms.py`: Added `has_state`, `step()` to Transform base and subclasses
- `pathmc/model.py`: Updated `PathModel.__init__` for scan observation reshaping, updated
  `do()` for unified dispatch, deprecated `panel_engine` parameter
- `tests/test_panel_engines.py`: Updated 3 tests for deprecated engine infrastructure
- `docs/`: Updated `panel_interventions.qmd` and `panel_engines.qmd` for new architecture
- `benchmarks/scan_vs_conv.py`: Phase 1 benchmark script (new)

## Testing

- [x] All 212 fast tests pass
- [x] All 21 panel engine tests pass (including slow)
- [x] All 6 panel do tests pass
- [x] Integration tests for adstock-only, adstock+AR, multi-equation, and exogenous lag models
- [x] Pre-existing Bernoulli/Poisson do() failures unchanged (not introduced by this PR)

## Notes

- The `panel_engine` parameter is accepted but deprecated — emits `DeprecationWarning`
- Models without temporal dependencies (no adstock, no lags) still use flat compilation
- The benchmark script in `benchmarks/` documents the performance gap for future reference
