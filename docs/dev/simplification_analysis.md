# pathmc Simplification Analysis

> March 2026 — Codebase review for consolidation and simplification opportunities.

## Current state

pathmc is ~6,000 lines across 13 Python source files (plus ~5,000 lines of tests across 33 test files). The codebase is well-structured with clean layer separation: parse → graph → compile → model, with simulation, effects, introspection, and identification as lateral modules.

| Module | Lines | Role |
|--------|-------|------|
| `compile.py` | 1,404 | PyMC model compilation (cross-sectional + scan/panel) |
| `model.py` | 1,293 | PathModel facade, `model()`, `simulate()` |
| `identify.py` | 678 | Backdoor/front-door, adjustment sets, d-sep tests |
| `simulate.py` | 509 | `do()` via `pm.do()`, DoResult |
| `introspect.py` | 500 | DAG viz, equations, priors (LaTeX rendering) |
| `parse.py` | 442 | DSL → typed AST |
| `effects.py` | 312 | Labeled coefficients, defined params, stdyx |
| `sensitivity.py` | 258 | Unmeasured confounding sensitivity |
| `transforms.py` | 230 | Transform registry (adstock, saturation) |
| `panel.py` | 151 | PanelInfo, add_lags (deprecated) |
| `graph.py` | 140 | Spec → DAG (GraphInfo via NetworkX) |
| `__init__.py` | 19 | Re-exports |
| `exceptions.py` | 15 | Custom error types |

The architecture is sound — the generative model + `pm.observe()` / `pm.do()` pattern is genuinely elegant and enables both estimation and intervention from one compiled model. The question is whether the implementation carries unnecessary weight.

---

## Verdict: mostly well-justified, with targeted cleanup opportunities

After reading every source file, the honest assessment is that **pathmc is not dramatically over-engineered for what it does**. The pipeline stages exist because they solve genuinely different problems (parsing, graph construction, PyMC compilation, causal simulation), and the PyMC ecosystem doesn't offer higher-level abstractions that cover pathmc's specific pattern of generative-model-first compilation with `pm.do()`.

That said, there are concrete opportunities ranging from quick wins (removing duplication) to more ambitious structural changes.

---

## Opportunity 1: Merge `att()` and `atu()` into a shared helper

**Impact: ~80 lines removed | Effort: small | Risk: none**

`att()` (lines 743–842) and `atu()` (lines 844–938) in `model.py` are nearly identical — same validation, same structure, same call to `run_do_pymc` with `subgroup_indices`. The only differences are the parameter name (`treated_value` vs `untreated_value`) and the error messages. A private `_subgroup_ate()` helper would implement both:

```python
def _subgroup_ate(self, treatment, values, subgroup_value, kind, label):
    mask = np.isclose(self._data[treatment].values.astype(float), subgroup_value)
    subgroup_idx = np.where(mask)[0]
    if len(subgroup_idx) == 0:
        raise ValueError(f"No observations with {treatment} ≈ {subgroup_value}.")
    lo, hi = values
    r_lo = run_do_pymc(..., set={treatment: lo}, subgroup_indices=subgroup_idx)
    r_hi = run_do_pymc(..., set={treatment: hi}, subgroup_indices=subgroup_idx)
    return r_hi - r_lo
```

Then `att()` and `atu()` become thin wrappers.

---

## Opportunity 2: Eliminate duplicate residual-block construction

**Impact: ~15 lines removed, cleaner data flow | Effort: small | Risk: none**

`graph.py` has `_build_residual_blocks()` (lines 131–140) which builds connected components from `~~` declarations. `compile.py` has `_identify_residual_blocks()` (lines 744–756) that does the exact same thing — builds an undirected graph from `spec.residual_covs` and finds connected components.

The compiler should use `graph_info.residual_blocks` (which is already computed and available) instead of recomputing from the spec. The duplication exists because of a historical accident where the compiler was written independently of the graph layer.

---

## Opportunity 3: Centralize the "ensure sampled" guard

**Impact: cleaner code, ~30 lines of boilerplate removed | Effort: small | Risk: none**

At least 12 methods in `PathModel` start with:

```python
if self._idata is None:
    raise RuntimeError("No posterior samples available. Call .fit() before ...")
```

A private `_require_posterior(method_name)` helper or a decorator would eliminate the repetition.

---

## Opportunity 4: Extract shared pooling-inspection helpers

**Impact: ~20 lines removed from introspect.py | Effort: small | Risk: none**

`compile.py` already has `_has_random_intercepts()` and `_get_slope_vars()` which decode the `pooling` argument. `introspect.py` reimplements the same logic inline when building the priors table. `introspect.py` should import and call the helpers from `compile.py`.

---

## Opportunity 5: Unify `run_do_pymc` and `run_do_panel_unified` value extraction

**Impact: ~60 lines consolidated | Effort: medium | Risk: low**

Both do-operator functions in `simulate.py` have similar value-extraction loops that iterate over `graph_info.topological_order` and branch on exogenous/endogenous/intervened/latent. The loops differ in where values come from (posterior predictive vs scan output) but the branching structure is identical. A shared helper that takes a "value source" abstraction could reduce this.

---

## Opportunity 6: Delegate identification to an external package

**Impact: ~400–500 lines removed from identify.py | Effort: large | Risk: medium**

`identify.py` (678 lines) implements backdoor criterion, front-door criterion, adjustment set enumeration, collider warnings, implied conditional independences, and partial-correlation testing. Several external packages provide overlapping functionality:

| Feature | pathmc (custom) | DoWhy | pgmpy | NetworkX |
|---------|----------------|-------|-------|----------|
| Backdoor adjustment sets | ✓ | ✓ | ✓ | — |
| Front-door identification | ✓ | ✓ | — | — |
| D-separation | ✓ (path-level) | ✓ | ✓ | ✓ (`is_d_separator`) |
| Collider warnings | ✓ | — | — | — |
| Implied CI testing | ✓ | — | — | — |

**Trade-offs:**

- **DoWhy** would be the most natural fit, but it expects its own `CausalGraph` format, so an adapter layer is needed. DoWhy is a substantial dependency (~40 transitive deps).
- **pgmpy** similarly has its own graph format and a heavy dependency footprint.
- pathmc's custom identification code is actually quite clean and well-tested (152 lines of tests, 17 test cases). The main argument for keeping it is zero-dependency simplicity.
- The partial-correlation CI testing (`test_implications`) is relatively niche and not well-covered by external packages in a form that integrates easily with pathmc's Bayesian output.

**Recommendation:** Keep for now. The code is correct, well-tested, and avoids a heavy transitive dependency. Revisit if DoWhy or pgmpy become existing dependencies for other reasons.

---

## Opportunity 7: Simplify `_compile_scan_panel`

**Impact: improved maintainability of the largest function (~387 lines) | Effort: large | Risk: medium**

`_compile_scan_panel` is a single 387-line function that handles:
1. Sequence/non-sequence classification for scan inputs
2. Carry variable setup (lagged endogenous + adstock state)
3. The scan step function body (variable resolution, mu construction, family-specific transforms)
4. Emission and reshaping of scan outputs

This is the most complex function in the codebase. It could be broken into ~4 focused helpers (setup, step function builder, emission, reshape). The logic itself is necessary — pytensor scan is inherently complex — but the current monolithic structure makes it hard to modify safely.

Note: This is a maintainability improvement, not a simplification of what the code does.

---

## Opportunity 8: Consider whether `patsy` is still needed

**Impact: one fewer dependency, modest code reduction | Effort: medium | Risk: low-medium**

`patsy` is used solely for `dmatrix()` calls in `build_design_matrix()` to produce design matrices from column names. pathmc doesn't use patsy's formula parsing (the DSL parser handles that), interaction terms (handled by `MuSpec`), or transformations. The actual patsy usage amounts to: "given a list of column names and whether to include an intercept, produce a DataFrame with those columns plus optionally an Intercept column."

This could be replaced with ~15 lines of pandas:

```python
def build_design_matrix(columns, data, has_intercept):
    dm = data[columns].copy()
    if has_intercept:
        dm.insert(0, "Intercept", 1.0)
    return dm
```

The `formulaic` package (patsy's successor) is another option if formula features are needed later.

**Trade-off:** patsy is battle-tested for edge cases. But pathmc's usage is so simple that the dependency mostly adds import-time overhead and a fragile pinning story (patsy is not actively maintained).

---

## Opportunity 9: Family registry instead of if/elif chains

**Impact: cleaner extensibility, ~30 lines saved | Effort: medium | Risk: low**

Family-specific behavior is currently scattered across if/elif chains in three locations:

1. `compile.py` → `_emit_free_rv()`: selects `pm.Normal`, `pm.Bernoulli`, etc.
2. `compile.py` → scan step function: applies inverse link (`pt.exp`, sigmoid)
3. `simulate.py` → `_apply_inverse_link()`: NumPy inverse link for `kind="mean"`

A small `Family` dataclass or registry could consolidate:

```python
@dataclass
class Family:
    name: str
    distribution: type       # pm.Normal, pm.Bernoulli, ...
    link: str                # "identity", "logit", "log"
    inverse_link_pt: Callable  # pytensor version
    inverse_link_np: Callable  # numpy version
    extra_params: list[str]  # ["sigma"], ["alpha_disp"], etc.
```

This would make adding new families (e.g., ordered probit, zero-inflated) a single registration instead of edits to three places.

---

## Opportunity 10: Reduce LaTeX rendering code in `introspect.py`

**Impact: ~80 lines reduced | Effort: medium | Risk: low**

`introspect.py` has ~200 lines of LaTeX rendering helpers (`_latexify_name`, `_latexify_prior`, `_format_term_latex`, `_format_transform_latex`, `_build_equation_latex`, `_latexify_expression`). These are well-written but represent a meaningful maintenance burden for what is essentially display logic.

Two options:
1. **Use sympy** for the arithmetic parts (`a*b` → LaTeX). This would replace `_latexify_expression()` but not the structural equation formatting.
2. **Move to a Jinja2 template** for the LaTeX output. The current Python string concatenation is hard to read; a template would be clearer.

Neither option removes a lot of code, but either would improve maintainability. The honest assessment: the current code works and rarely changes, so this is low priority.

---

## Opportunity 11: Consider Bambi for the compilation layer

**Impact: potentially large (could replace compile.py) | Effort: very large | Risk: high**

Bambi (Bayesian Modeling Made Easy) is a higher-level interface to PyMC that handles formula parsing, design matrices, families, random effects, and prior specification. In principle, pathmc's compilation layer overlaps with Bambi's scope.

**Why this doesn't work today:**

1. **Generative model pattern**: pathmc compiles a generative model where endogenous variables are free RVs, then uses `pm.observe()` to condition them. Bambi always produces observed models — there is no `pm.do()` compatible generative model.
2. **Multi-equation structure**: Bambi handles single-equation GLMMs. pathmc compiles a system of equations where downstream variables depend on upstream free RVs. This causal wiring is the whole point.
3. **Scan/panel compilation**: Bambi doesn't support pytensor scan for temporal dependencies.
4. **Transforms**: Bambi doesn't have a concept of domain transforms (adstock, saturation) with estimable parameters.

**Could Bambi be extended?** In theory, yes — Bambi could support multi-equation generative models. In practice, this would require deep changes to Bambi's internals, and the pathmc team doesn't control Bambi's roadmap.

**Recommendation:** Not viable as a simplification path. pathmc's compilation needs are genuinely outside Bambi's scope.

---

## Opportunity 12: `simulate()` function placement

**Impact: code organization improvement | Effort: small | Risk: none**

The `simulate()` function (for simulate-and-recover / prior predictive simulation) lives in `model.py` (lines 1165–1293) alongside the `model()` entry point. It uses `build_graph`, `build_design_matrix`, `compile_to_pymc` — it's essentially a second entry point into the pipeline. Moving it to its own small module (or into `simulate.py` alongside the do-operator code) would reduce `model.py`'s size and improve cohesion.

---

## What NOT to simplify

Several areas look like they might be over-engineered but are actually well-justified:

### The parse → graph → compile pipeline

Three stages for what could be "spec string → PyMC model" might seem like over-engineering. But the separation is load-bearing:
- The parser catches syntax errors cheaply (no PyMC, no data).
- The graph layer provides topological ordering and identification (no data needed).
- The compiler needs both the AST and the graph.
- Tests can exercise each layer independently.

### The `MuSpec` / `PredictorSlot` / resolver abstraction

This might seem like unnecessary indirection for "multiply coefficients by predictors." But it's what allows the same mu-construction code to work for both cross-sectional (resolve from `pm.Data` and free RVs) and scan/panel (resolve from scan carry variables). Without it, the mu construction would be duplicated.

### The transform registry

A registration-based system for two transforms (adstock, saturation) might seem over-engineered. But the DSL and compiler are designed for extensibility — users can register custom transforms. The registry pattern keeps this clean.

### Separate `_gen_model` and `_pymc_model`

Having two PyMC model objects looks redundant until you realize that `pm.do()` operates on the generative model (free RVs) while `pm.sample()` operates on the estimation model (`pm.observe()`-conditioned). This duality is the core architectural insight.

---

## Priority ranking

| # | Opportunity | Lines saved | Effort | Risk | Priority |
|---|------------|-------------|--------|------|----------|
| 1 | Merge `att()`/`atu()` | ~80 | Small | None | High |
| 2 | Eliminate duplicate residual blocks | ~15 | Small | None | High |
| 3 | Centralize "ensure sampled" guard | ~30 | Small | None | High |
| 4 | Extract shared pooling helpers | ~20 | Small | None | High |
| 5 | Unify do-result extraction | ~60 | Medium | Low | Medium |
| 9 | Family registry | ~30 | Medium | Low | Medium |
| 12 | Move `simulate()` function | 0 | Small | None | Medium |
| 7 | Break up `_compile_scan_panel` | 0 | Large | Medium | Medium |
| 8 | Replace patsy | ~30 | Medium | Low-Med | Low |
| 10 | Reduce LaTeX code | ~80 | Medium | Low | Low |
| 6 | Delegate identification | ~400 | Large | Medium | Low |
| 11 | Use Bambi for compilation | ~800 | Very large | High | Not viable |

The high-priority items (1–4) are mechanical cleanups that could be done in a single PR with ~145 lines net reduction and zero risk. The medium-priority items require more design thought but would improve long-term maintainability.

---

## Summary

pathmc's ~6,000 lines serve a genuinely complex purpose: compiling a multi-equation causal DSL into PyMC generative models that support both Bayesian estimation and interventional simulation. The layered architecture is well-justified, and no single external package can replace the core pipeline.

The realistic simplification path is not "rewrite on top of Bambi/DoWhy" but rather:
1. **Remove internal duplication** (opportunities 1–5): ~200 lines, easy, zero risk.
2. **Introduce small abstractions** where if/elif chains repeat (family registry, pooling helpers).
3. **Improve the largest function** (`_compile_scan_panel`) by breaking it into helpers.
4. **Evaluate patsy** — it's doing very little for its dependency weight.

The codebase is in good shape. The ratio of "necessary complexity" to "accidental complexity" is favorable.
