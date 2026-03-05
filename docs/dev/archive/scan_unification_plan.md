# Scan + do() Unification — Implementation Plan

> **Background**: See `panel_engine_unification.md` (same directory) for the full analysis of why three panel engines exist and the case for unifying them via scan-based generative models.

## Goal

Replace the three panel `do()` engines (numpy, batched, scan) with a single architecture: build the generative model with `pytensor.scan` for temporal structure, so `pm.do()` handles panel interventions natively — the same way it already handles cross-sectional interventions.

## Phasing

This is a two-phase effort. Phase 1 is a benchmark that gates the decision. Phase 2 is the implementation.

---

## Phase 1: Benchmark (gate)

### Objective

Determine whether `pytensor.scan`-based estimation is acceptably performant compared to convolution-based estimation for typical pathmc model sizes.

### Deliverable

A standalone benchmark script (`benchmarks/scan_vs_conv.py`) that fits the same DGP two ways and reports wall-clock time, ESS/second, divergences, and coefficient recovery.

### Models to benchmark

1. **Adstock only** (no AR terms):
   - Spec: `sales ~ adstock(spend, decay=theta)`
   - DGP: T=100, 5 units, true decay=0.7, true beta=0.5
   - Variant A: convolution-based estimation (current `compile.py`)
   - Variant B: scan-based estimation (hand-built `pm.Model` with `pytensor.scan`)

2. **Adstock + AR**:
   - Spec: `sales ~ adstock(spend, decay=theta) + lag(sales)`
   - Same DGP but with AR(1) coefficient 0.3
   - Variant A: convolution + data column for lag (current)
   - Variant B: scan with carry for both adstock state and previous mu

3. **Multi-equation**:
   - Spec: `awareness ~ adstock(spend, decay=theta); sales ~ awareness + lag(sales)`
   - Two-equation DAG with temporal dependencies in both
   - Same two variants

### Metrics to report

| Metric | Why |
|--------|-----|
| Wall-clock sampling time (2 chains, 500 draws, 500 tune) | Primary cost concern |
| ESS/second for key parameters (beta, decay) | Efficiency-adjusted cost |
| Number of divergences | Gradient quality |
| Posterior mean recovery vs. true values | Correctness check |
| `do()` wall-clock time | Confirm do() is at least as fast |

### Gate criteria

- If scan is **≤3x slower** than convolution on ESS/second across all three models → proceed to Phase 2
- If scan is **3–5x slower** → proceed with a note that a convolution fast-path may be needed later for large models
- If scan is **>5x slower** → abort scan+do unification; promote the current scan engine as sole panel engine instead (simpler version of the change)

---

## Phase 2: Implementation

### Overview

Modify `compile.py` to emit `pytensor.scan`-based generative models for panel data. This makes the generative model self-contained for temporal propagation, so `pm.do()` works without a separate simulation engine.

### M-P1: Scan-based compiler for panel models

**Files**: `compile.py`

**Change**: When `panel_info` is not None and the model has temporal dependencies (adstock transforms or lag terms), compile the generative model using `pytensor.scan` instead of flat vectorised operations.

The compiler should:
1. Identify all temporal dependencies (adstock transforms, lag columns)
2. Build a scan step function that:
   - Takes exogenous inputs at time t as sequences
   - Carries endogenous mu values and adstock state as carry variables
   - Computes the linear predictor for each endogenous variable in topological order
   - Returns updated carry (mu values + adstock state)
3. Emit `pytensor.scan(fn=step, sequences=exog, outputs_info=carry, non_sequences=params)`
4. Emit `pm.Deterministic('mu_{var}', scan_results[i])` for each endogenous variable
5. Emit `pm.Normal('{var}', mu=mu_all, sigma=sigma)` as free RVs with shape `(T, n_units)`

**Key design decisions**:
- Exogenous data is reshaped to `(T, n_units)` and passed as scan sequences
- Parameters (betas, sigmas, transform params) are non-sequences
- Random intercepts/slopes: fold into the step function via unit-indexed tensors
- For models WITHOUT temporal dependencies, keep the current flat compilation (no scan overhead)

**Acceptance criteria**:
- `pm.observe(gen_model, {'sales': sales_obs})` produces a valid estimation model
- `pm.sample()` on the estimation model recovers known DGP coefficients
- `pm.do(gen_model, {'spend': new_spend})` produces correct counterfactual trajectories
- Existing cross-sectional compilation is unchanged

### M-P2: Unified panel do() in model.py

**Files**: `model.py`, `simulate.py`

**Change**: Panel `do()` uses the same code path as cross-sectional `do()`: `pm.do()` on the generative model + `compute_deterministics` (mean) or PPC (predictive).

Specifically:
- `PathModel.do()` with `simulate_over="time"` calls `run_do_pymc()` (the existing cross-sectional function), passing the scan-based generative model
- `run_do_pymc()` may need minor adjustments to handle the `(T, n_units)` output shape and produce per-time-step results in `DoResult`
- The `panel_engine` parameter is **deprecated** with a warning, then removed

**Acceptance criteria**:
- `model.do(set={'spend': 30.0}, simulate_over='time')` produces correct results without specifying `panel_engine`
- `DoResult` includes `by_time()` data with correct shape
- Passing `panel_engine=` emits a `DeprecationWarning`

### M-P3: Remove old panel engines

**Files**: `simulate.py`, `model.py`

**Change**: Delete the following functions and all their helpers:
- `run_panel_do()` (numpy engine, ~210 lines)
- `run_panel_do_batched()` (~180 lines)
- `run_panel_do_scan()` (~140 lines)
- `_build_step_model()` (~170 lines)
- `_build_scan_model()` (~180 lines)
- `_apply_panel_transform()`, `_apply_single_step_transform_pt()`, `_get_panel_col_value()`, `_prepare_panel_data()`, `_reshape_to_panel()`, and related helpers

Also remove:
- `panel_engine` parameter from `PathModel.do()` signature
- The engine dispatch logic in `model.py`
- Imports of the removed functions

**Estimated deletion**: ~700–900 lines from `simulate.py`

**Acceptance criteria**:
- All remaining tests pass
- `simulate.py` exports only `run_do_pymc` and `DoResult` (plus any shared helpers)
- No references to numpy/batched/scan engines remain in the codebase

### M-P4: Transform interface update

**Files**: `transforms.py`

**Change**: The `Transform` base class gains a `step()` method for single-timestep computation inside the scan body. This replaces the ad-hoc `_apply_single_step_transform_pt()` function currently in `simulate.py`.

```python
class Transform:
    def apply_pymc(self, x, params, *, panel_info=None, data=None):
        """Full-series transform for compilation (convolution, vectorised)."""
        raise NotImplementedError

    def step(self, x_t, state, params):
        """Single time-step for scan body. Returns (output_t, new_state).

        Only needed for temporal transforms (adstock). Pointwise transforms
        (saturation) can use a default that returns (apply(x_t), state).
        """
        raise NotImplementedError
```

For `Adstock`: `step()` returns `(x_t + decay * prev_state, x_t + decay * prev_state)`
For `LogisticSaturation`: `step()` returns `(1 - exp(-lam * x_t), state)` (no state change)

**Note**: `apply_pymc()` stays as-is — it's still used for cross-sectional models (no scan needed). For panel models, the compiler calls `step()` inside the scan body. The `apply_numpy()` method can be removed once the numpy engine is deleted.

**Acceptance criteria**:
- `Adstock.step()` and `LogisticSaturation.step()` exist and are tested
- The scan compiler uses `transform.step()` in the step function
- `apply_numpy()` is removed from all transforms

### M-P5: Update tests

**Files**: `tests/test_panel_engines.py` and others

**Change**: The panel engine comparison tests become single-engine tests. The core assertions (DGP recovery, ATE correctness, by_time shape) remain; the engine loop is removed.

Specific changes:
- `test_panel_engines.py`: Rewrite to test the unified `do()` without `panel_engine` parameter. Keep the DGP fixtures and correctness assertions. Remove engine comparison tests (they're meaningless with one engine). Remove batched mean-field tolerance tests. Remove `test_unknown_engine_raises`.
- Other test files referencing `panel_engine`: Remove the parameter. Tests should call `do(simulate_over='time')` without engine selection.
- Add new tests for scan-based estimation: coefficient recovery on the benchmark DGPs.

**Acceptance criteria**:
- `pytest -x -v` passes with no `panel_engine` references
- DGP recovery tests still verify correct causal effects
- No test imports `run_panel_do`, `run_panel_do_batched`, or `run_panel_do_scan`

### M-P6: Documentation update

**Files**: `docs/concepts/panel_interventions.qmd`, `docs/how-it-works.qmd`

**Change**: Remove references to engine selection. Update the panel `do()` documentation to reflect the unified architecture. The `panel_engine` parameter should not appear in any user-facing docs.

**Acceptance criteria**:
- `quarto render docs/` exits 0
- No mention of `panel_engine`, "numpy engine", "batched engine", or "scan engine" in rendered docs

---

## Milestone Ordering

```
Phase 1: Benchmark
  └── benchmarks/scan_vs_conv.py
  └── Gate decision (≤3x → proceed)

Phase 2: Implementation
  M-P1: Scan-based compiler ──┐
  M-P2: Unified panel do()  ──┤ (can overlap)
  M-P3: Remove old engines  ──┘ (after M-P1 + M-P2 pass tests)
  M-P4: Transform interface    (can parallel with M-P1–P3)
  M-P5: Update tests           (after M-P3)
  M-P6: Documentation          (after M-P5)
```

## Risk: Models without temporal dependencies

Models that are panel but have no adstock or lags (e.g., `sales ~ spend` with `panel=...`) should NOT use scan. The compiler should detect whether temporal structure exists and fall back to the current flat compilation. Scan adds overhead for models that don't need it.

Detection logic: a model has temporal structure if any regression term has an adstock transform OR any term has `lag_of` set (i.e., `lag(var)` in the spec).

## Risk: Teacher forcing vs. free-running

The scan-based model is free-running (mu_t depends on mu_{t-1}, not sales_obs_{t-1}). This is a different model from teacher-forced regression. The benchmark (Phase 1) will reveal if this causes estimation issues. If it does, a hybrid approach is possible: teacher-force during estimation by injecting observed values into the scan carry, but use free-running for do(). This would be an M-P1 variant, not a separate milestone.
