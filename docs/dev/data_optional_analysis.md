# Analysis: Data-Free DAG Exploration in `pathmc`

> **Status:** Proposal — ready for review.

## Motivation

Today, `pathmc.model(spec, data=df)` requires a DataFrame upfront. This blocks a valuable workflow: exploring a DAG's structure, checking identification, and visualizing equations *before* having data. Analysts often sketch candidate DAGs on a whiteboard before collecting or cleaning data. The package should support that.

## Current Architecture: Where Is Data Actually Required?

I audited every layer of the codebase to determine which operations genuinely need data and which only need the parsed spec and graph structure.

### No data needed (work today at the lower level, but not exposed via `PathModel`)

| Layer | Function | Inputs |
|-------|----------|--------|
| Parser | `parse_spec()` | spec string |
| Graph | `build_graph()` | `Spec` |
| Introspection | `build_dag_viz()` | `Spec`, `GraphInfo` |
| Introspection | `build_equations()` | `Spec` |
| Introspection | `build_priors()` | `Spec`, families, pooling config |
| Identification | `adjustment_sets()` | `GraphInfo` |
| Identification | `is_identifiable()` | `GraphInfo` |
| Identification | `frontdoor_identifiable()` | `GraphInfo` |
| Identification | `collider_warnings()` | `GraphInfo` |
| Identification | `implied_independences()` | `GraphInfo` |

### Data required

| Layer | Function | Why |
|-------|----------|-----|
| Identification | `test_implications()` | Partial correlation tests against observed data |
| Compiler | `build_design_matrix()` | Needs column values to construct X matrices |
| Compiler | `compile_to_pymc()` | Needs design matrices + data shapes |
| Model | `PathModel.__init__()` | Builds design matrices and compiles immediately |
| Post-fit | `fit()`, `do()`, `ate()`, `cate()`, `att()`, `atu()`, `prob()`, `sensitivity()`, `predict()` | Need compiled model + posterior draws |
| Post-fit | `summary()`, `effects_summary()`, `standardized()` | Need posterior draws |

### The blocker

The bottleneck is `PathModel.__init__()`. It takes `data: pd.DataFrame` (not Optional) and immediately:

1. Builds design matrices from data (lines 99–115)
2. Calls `_compile()` which creates the PyMC model (line 126)

The `model()` factory function enforces the same — `data` is a required positional argument.

**Verdict: Data is NOT a hard requirement for DAG exploration and identification.** The requirement is entirely an artifact of `PathModel` coupling compilation with construction. The underlying functions (`parse_spec`, `build_graph`, all of `identify.py`, most of `introspect.py`) already work without data.

## Proposed Design

### Allow `data=None` in `model()`

```python
# DAG-only — no data, no compilation
m = pathmc.model("""
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
""")

# These all work immediately:
m.graph()                              # DAG visualization
m.equations()                          # structural equations + priors
m.priors()                             # prior table
m.adjustment_sets("X", "Y")           # backdoor sets
m.is_identifiable("X", "Y")           # identification check
m.collider_warnings({"C"}, "X", "Y")  # collider detection
m.implied_independences()              # testable implications
m.frontdoor_identifiable("X", "M", "Y")

# These raise a clear error:
m.fit()           # RuntimeError: "No data provided..."
m.do(set={"X": 1})
m.design("Y")
m.test_implications()
```

### Obtaining a data-bound model

Two options evaluated:

**Option A — Create a new model (recommended)**

```python
# Later, when data arrives:
m_data = pathmc.model(spec, data=df)
idata = m_data.fit(draws=1000)
```

Users just call `pathmc.model()` again with the same spec string and data. Simple, no new API surface, no state confusion. This is what the user would do today — the only difference is that the first call *without* data is now possible.

**Option B — Mutable `.with_data()` method**

```python
m_data = m.with_data(df)  # returns a NEW PathModel
```

A convenience method returning a *new* PathModel instance (not mutating in place). Preserves any custom priors or families from the original. This is syntactically nicer for notebook workflows where you've already configured priors on the data-free model.

**Option C — In-place `.set_data()` (rejected)**

Mutating the model in place (like `set_priors()` does) is risky here because adding data changes the fundamental capabilities of the object. `set_priors()` is a smaller configuration change that recompiles an already-compilable model. Going from "no data" to "has data" is a larger state transition that would confuse users about what's valid.

### Recommendation

**Implement Option A as the baseline. Consider Option B as a convenience if demand arises.**

Option A requires no new public API — just making `data` optional. Option B can be added later as sugar without breaking anything. The key point is that we should NOT try to support in-place mutation from data-free to data-bound.

## Implementation Plan

### Changes to `model()` factory

```python
def model(
    spec_string: str,
    data: pd.DataFrame | None = None,  # was: data: pd.DataFrame
    families: dict[str, str] | None = None,
    ...
) -> PathModel:
```

When `data is None`:
- Parse spec and build graph (same as today)
- Skip the endogenous-variable-in-data check
- Skip panel info construction
- Still validate lag terms require panel (can check this structurally)
- Pass `data=None` to `PathModel.__init__()`

### Changes to `PathModel.__init__()`

When `data is None`:
- Store `self._data = None`
- Skip design matrix construction (`self._design_matrices = {}`)
- Skip `_compile()` (`self._pymc_model = None`, `self._gen_model = None`)
- Still compute priors (they don't need data)

### Guard methods that need data

Add a helper:

```python
def _require_data(self, method_name: str) -> None:
    if self._data is None:
        raise RuntimeError(
            f"{method_name}() requires data. Create a data-bound model: "
            f"m = pathmc.model(spec, data=df)"
        )
```

Call it at the top of: `fit()`, `do()`, `ate()`, `cate()`, `att()`, `atu()`, `prob()`, `sensitivity()`, `predict()`, `design()`, `test_implications()`, `sample_prior_predictive()`, `summary()`, `effects_summary()`, `standardized()`, `effect()`.

### Methods that work without data (no changes needed)

- `graph()` — already uses only `self._spec` and `self._graph_info`
- `equations()` — already uses only `self._spec`, `self._latent`, `self._families`
- `priors()` — already uses only `self._spec` and config
- `set_priors()` — updates `self._priors`, recompiles only if data exists
- `adjustment_sets()` — uses only `self._graph_info`
- `is_identifiable()` — uses only `self._graph_info`
- `frontdoor_identifiable()` — uses only `self._graph_info`
- `collider_warnings()` — uses only `self._graph_info`
- `implied_independences()` — uses only `self._graph_info`

### `set_priors()` adaptation

When no data is present, `set_priors()` should still update `self._priors` but skip recompilation. When data is later provided (via creating a new model), the priors carry through.

### What about `with_data()`?

If we add it later, it would look like:

```python
def with_data(
    self,
    data: pd.DataFrame,
    panel: dict[str, str] | None = None,
    pooling: str | dict | None = None,
) -> "PathModel":
    """Return a new data-bound PathModel preserving spec, families, and priors."""
    return PathModel(
        spec=self._spec,
        graph_info=self._graph_info,
        data=data,
        families=self._families,
        panel_info=build_panel_info(data, panel) if panel else None,
        pooling=pooling or self._pooling,
        latent=self._latent,
        priors=self._priors,
    )
```

This is trivial to add later and doesn't affect the core design.

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Users confused about which methods work | Low | Clear error messages naming the missing piece |
| `set_priors()` behavior diverges for data-free vs data-bound | Low | Skip recompile when no data; document clearly |
| Type hints on `self._data` become `Optional[pd.DataFrame]` | Low | Internal only; guard methods handle it |
| Test coverage for data-free path | Medium | Add tests for each data-free method |

## Test Strategy

No modification to existing test files. New tests would verify:

1. `pathmc.model(spec)` without data succeeds
2. All introspection methods work on data-free model
3. All identification methods work on data-free model
4. Data-requiring methods raise `RuntimeError` with helpful message
5. Creating a new model with same spec + data works normally
6. `set_priors()` on data-free model updates priors without error

## Summary

The data requirement for DAG exploration is **not fundamental** — it's an artifact of `PathModel` coupling construction with compilation. Making `data` optional in `model()` requires:

- ~20 lines changed in `model()` and `PathModel.__init__()`
- ~15 guard calls added to data-requiring methods
- A small helper method `_require_data()`
- No changes to any lower-level modules (parser, graph, identify, introspect)
- No changes to test files

The "add data later" question resolves cleanly: users create a new model with `pathmc.model(spec, data=df)`. A `with_data()` convenience method can be added later if the notebook workflow calls for it, but it's not needed for the core feature.
