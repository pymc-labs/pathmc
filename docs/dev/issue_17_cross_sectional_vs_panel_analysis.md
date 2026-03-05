# Issue #17: Do We Still Need Cross-Sectional vs Panel Modes?

> **Issue**: "Now everything is just a symbolic graph through pytensor, then using pm.do, do we even need different modes?"

## TL;DR

**Mostly no — but the remaining differences are real and well-justified.**

The scan-based unification already eliminated the biggest sources of divergence (three competing panel `do()` engines). What remains is a single architectural branch point in the compiler (`compile.py`) that produces either a flat model or a scan model depending on whether temporal dependencies exist. The `do()` code has two functions (`run_do_pymc` and `run_do_panel_unified`) but they share the same mechanism (`pm.do()` + `compute_deterministics` or PPC). The user-facing API is already unified — there is no "mode" the user selects; pathmc auto-detects.

The remaining separation is a legitimate architectural concern driven by the physics of the problem (temporal vs atemporal data), not by implementation debt.

---

## What Has Already Been Unified

The repo has already gone through a major unification effort (documented in `archive/panel_engine_unification.md` and `archive/scan_unification_plan.md`). Here's what was eliminated:

| Before (3+ engines) | After (current state) |
|---|---|
| `run_panel_do()` — numpy engine (~210 lines) | Deleted |
| `run_panel_do_batched()` — batched engine (~180 lines) | Deleted |
| `run_panel_do_scan()` — separate scan model (~140 lines) | Replaced by unified scan in compiler |
| `_build_step_model()` (~170 lines) | Deleted |
| `_build_scan_model()` (~180 lines) | Deleted |
| `panel_engine` parameter dispatch logic | Deprecated (emits warning, ignored) |

**Total removed**: ~880 lines of simulation code that duplicated compiler logic.

**What replaced it**: The scan is now in the generative model itself (`_compile_scan_panel()` in `compile.py`). The same `pm.do()` mechanism works for both cross-sectional and panel — no separate simulation engine needed.

## What Still Differs

### 1. Compiler branch: flat vs scan (`compile.py`)

The compiler has two code paths gated by `_has_temporal_deps()`:

```python
if panel_info is not None and _has_temporal_deps(spec, graph_info):
    return _compile_scan_panel(...)   # ~350 lines
else:
    # flat generative model            # ~80 lines of the main path
```

**Is this separation necessary?** Yes. These produce fundamentally different PyMC models:

| | Flat (cross-sectional / non-temporal panel) | Scan (temporal panel) |
|---|---|---|
| **Variable shapes** | `(N,)` where N = total rows | `(n_times, n_units)` |
| **Temporal state** | None | Carry variables in `pytensor.scan` |
| **Exogenous data** | `pm.Data(var, data[var].values)` | `pm.Data(var, mat.astype(float))` reshaped to `(T, U)` |
| **Observation reshaping** | Direct `data[var].values` | Sorted → reshaped to `(T, U)` via `PanelScanInfo` |
| **Transform application** | `transform.apply_pymc()` (vectorised) | `transform.step()` inside scan body |
| **Lag terms** | Not supported (raises error) | Carry state feeds `prev_endo[base_var]` |

A flat model with `pytensor.scan` wrapping it when there's no temporal dependency would add overhead for no benefit. The detection logic (`_has_temporal_deps`) is clean — it checks for `lag()` DSL terms, `_lag\d+$` columns, and `adstock()` transforms.

### 2. Simulation functions: `run_do_pymc` vs `run_do_panel_unified` (`simulate.py`)

These are the two functions called from `PathModel.do()`:

| | `run_do_pymc` (145 lines) | `run_do_panel_unified` (160 lines) |
|---|---|---|
| **Called when** | No `simulate_over="time"`, or panel without scan | `simulate_over="time"` with scan-compiled model |
| **Replacements shape** | `np.full(N, val)` | `np.full((n_times, n_units), val)` or `val[:, None]` broadcast |
| **Core mechanism** | `pm.do(gen_model, replacements)` | `pm.do(gen_model, replacements)` |
| **Propagation** | `compute_deterministics` or PPC | `compute_deterministics` or PPC |
| **Output** | `DoResult(values={...})` | `DoResult(values={...}, values_by_time={...}, time_index=...)` |
| **Reshape logic** | Flatten `.values.flatten()` | `.reshape(-1, n_times, n_units).mean(axis=2).T` |

The core mechanism is identical: `pm.do()` + either `compute_deterministics` (mean) or `sample_posterior_predictive` (predictive). The divergence is in:

1. **Shape handling**: panel outputs are `(n_times, n_units)` and need averaging over units and per-time decomposition
2. **Per-time results**: `DoResult.by_time()` is only meaningful for panel models — it requires `values_by_time` which comes from the `(T, U)` reshape

### 3. User-facing API: `PathModel.do()` (`model.py`)

```python
def do(self, set=None, shift=None, kind="mean",
       simulate_over=None, panel_engine="numpy"):
```

The dispatch logic:

```python
if simulate_over == "time":
    # validate panel exists, check array lengths
    if scan_info is not None:
        return run_do_panel_unified(...)   # panel path
    return run_do_pymc(...)                # fallback (non-temporal panel)
return run_do_pymc(...)                    # cross-sectional path
```

The `panel_engine` parameter is already deprecated and ignored. The user doesn't choose a "mode" — they either pass `simulate_over="time"` or not. Internally, pathmc auto-detects whether a scan model exists.

---

## Could These Be Merged Further?

### Could `run_do_pymc` and `run_do_panel_unified` become one function?

**In principle, yes.** They share the same algorithm:

1. Build replacements dict
2. For `kind="mean"`: replace endogenous RVs with their `mu_*` deterministics
3. Call `pm.do(gen_model, replacements)`
4. Call `compute_deterministics` or `sample_posterior_predictive`
5. Extract and reshape results

The differences are purely in shape handling (flat vs `(T, U)`) and the extra `values_by_time` output. A single function with shape-aware logic could handle both.

**Estimated savings**: ~80-100 lines (the functions share ~60% of their logic).

**Risk**: The unified function would have more `if scan_info is not None:` branches, potentially reducing readability. The current separation is clean and each function is self-contained.

**Verdict**: A reasonable refactor, but not urgent. The code duplication is modest and the separation aids readability.

### Could the compiler always use scan?

**No, and it shouldn't.** For models without temporal dependencies:

- `pytensor.scan` adds compilation overhead and gradient complexity
- The flat model is simpler, faster to compile, and faster to sample
- Cross-sectional models don't have a time dimension to scan over
- Panel models without lags/adstock (e.g., `sales ~ spend` with `panel=...`) benefit from flat compilation

The `archive/panel_engine_unification.md` doc explicitly notes this: "Models that are panel but have no adstock or lags should NOT use scan."

### Could `simulate_over="time"` be auto-detected?

**Possibly.** If a model has a scan-compiled generative model, `do()` could automatically use `run_do_panel_unified` without requiring `simulate_over="time"`. The argument would become a no-op or an opt-out.

However, there's a semantic reason to keep it: a panel model *without* temporal deps might still want cross-sectional-style `do()` (one intervention applied uniformly, no time decomposition). And users may want to see the familiar cross-sectional `DoResult` even from a panel model. The explicit `simulate_over="time"` flag serves as an intent declaration.

---

## What "Modes" Actually Remain

The word "modes" implies a user-facing choice, but pathmc doesn't really have that anymore. The user makes **data-level declarations**:

| User declaration | What pathmc does internally |
|---|---|
| No `panel=` arg | Flat generative model, `run_do_pymc` |
| `panel=` but no `lag()`/`adstock()` | Flat generative model with random effects, `run_do_pymc` |
| `panel=` with `lag()` or `adstock()` | Scan generative model, `run_do_panel_unified` |
| `do(simulate_over="time")` | Time-forward propagation with per-time results |

This is not a "mode" — it's the compiler responding to the model structure. The distinction is analogous to how a SQL engine chooses between a sequential scan and an index scan: the user declares what they want (the query), and the engine picks the execution strategy.

## Remaining Cleanup Opportunities

### 1. Remove `panel_engine` parameter entirely

The `panel_engine` parameter is deprecated but still present in the signature. It could be removed in a future version. The tests in `test_panel_engines.py` still reference it (the `TestNoTemporalState` class passes `panel_engine=engine` in a loop over `("numpy", "batched", "scan")`). These tests are testing the deprecation warning, not engine behavior.

### 2. Merge `run_do_pymc` and `run_do_panel_unified`

As discussed above, a shape-aware single function would eliminate ~80-100 lines of duplication. The function signature would gain `panel_info` and `scan_info` optional parameters.

### 3. Rename `simulate_over` to something clearer

The `simulate_over="time"` argument name is somewhat opaque. An alternative like `temporal=True` or `time_forward=True` might be clearer. But this is cosmetic and would be a breaking change.

### 4. Clean up `test_panel_engines.py`

The `TestNoTemporalState` class still loops over three engines, all of which trigger deprecation warnings. These tests should be simplified to test the unified path without engine selection. (Note: per AGENTS.md, test files should not be modified by agents — this is flagged for human review.)

## Recommendations

1. **Don't introduce any new "mode" abstraction.** The current auto-detection is the right design. Users declare data structure; pathmc picks the execution strategy.

2. **Consider merging `run_do_pymc` + `run_do_panel_unified`** into a single `run_do()` function in a future PR. Low urgency but reduces surface area.

3. **Remove `panel_engine` from the API** in the next minor version. It's already dead code.

4. **Keep the compiler branch** (`flat` vs `_compile_scan_panel`). These produce genuinely different PyMC models and merging them would add complexity for no benefit.

5. **Close issue #17** with the conclusion: the answer is "no, we don't need separate modes anymore — and in fact, we already unified them. The remaining differences are shape handling driven by the data structure, not competing implementations."

## Summary

| Question | Answer |
|---|---|
| Do we need cross-sectional vs panel "modes"? | **No user-facing modes exist anymore.** The compiler auto-detects. |
| Do we need separate compiler paths? | **Yes.** Flat vs scan are genuinely different PyMC models. |
| Do we need separate `do()` functions? | **Debatable.** Could merge, but current separation is clean. |
| Is there dead code from the old engine design? | **Yes.** `panel_engine` parameter and some tests still reference it. |
| Does `pm.do()` unify the mechanism? | **Yes.** Both paths use `pm.do()` + `compute_deterministics`/PPC. |
| What should we do? | **Minor cleanup** (remove `panel_engine`, optionally merge `do()` functions). No architectural changes needed. |
