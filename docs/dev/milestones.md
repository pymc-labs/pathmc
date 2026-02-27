# Milestones — pathmc

These milestones cover the v0.1 (MVP) and v0.2 (Panel mode) scope defined in `prd_v1.md`.

## How Milestones Work

Each milestone has **gate tests** in `tests/`. A milestone is done when **all** its gate tests pass. Run only the specific test file for the current milestone:

```bash
pytest tests/test_<milestone>.py -x -v
```

Do **not** modify test files. If a test seems wrong, flag it for review.

After tests pass, also verify:
- `ruff check` reports no errors
- `ruff format --check` reports no changes needed

## Milestone Overview

| #   | Name                       | Gate tests                                  | Depends on | Status |
| --- | -------------------------- | ------------------------------------------- | ---------- | ------ |
| M1  | DSL Parser                 | `test_parse.py`                             | —          | ✓      |
| M2  | Graph Builder              | `test_graph.py`                             | M1         | ✓      |
| M3  | PathModel + Design Matrices| `test_compile.py::TestDesignMatrix`         | M1, M2     | ✓      |
| M4  | Gaussian Compiler          | `test_compile.py` (all)                     | M3         | ✓      |
| M5  | Introspection              | `test_introspection.py`                     | M4         | ✓      |
| M6  | do() Cross-sectional       | `test_do.py`                                | M4         | ✓      |
| M7  | Residual Covariance (~~)   | `test_residual_cov.py`                      | M4         | ✓      |
| M8  | Effects + Defined Params   | `test_effects.py`                           | M4         | ✓      |
| M9  | Integration Smoke Tests    | `test_smoke.py`                             | M1–M8      | ✓      |
| M10 | Documentation              | `cd docs && quarto render` exits 0          | M9         | ✓      |
| M11 | Bernoulli-logit Family     | `test_bernoulli.py`                         | M4         | ✓      |
| M12 | Predictive do()            | `test_do_predictive.py`                     | M6, M11    | ✓      |
| M13 | `add_lags()` utility       | `test_add_lags.py`                          | —          | ✓      |
| M14 | Panel fit + random intercepts | `test_panel.py`                          | M13        | ✓      |
| M15 | Random slopes              | `test_random_slopes.py`                     | M14        | ✓      |
| M16 | Time-forward do()          | `test_panel_do.py`                          | M13, M14   | ✓      |
| M17 | Panel smoke tests          | `test_panel_smoke.py`                       | M13–M16    | ✓      |
| M18 | Panel documentation        | `cd docs && quarto render` exits 0          | M17        | ✓      |

## Required Module Structure

The test files import from these specific paths — they are **not negotiable**:

- `pathmc.parse` — must export `parse_spec(spec_string: str) -> Spec`
- `pathmc.graph` — must export `build_graph(spec: Spec) -> GraphInfo`
- `pathmc` (top-level) — must export `fit(spec: str, data: pd.DataFrame, **kwargs) -> PathModel`
- `pathmc.exceptions` — must export custom exception classes

Internal helpers and additional submodules can be organized as you see fit.

## Data Structures

The gate tests assert against these interfaces. Implement them however you like (dataclasses, attrs, plain classes) as long as the attribute access patterns below work.

### Spec (returned by `parse_spec`)

```
Spec
  .regressions: list[Regression]
  .residual_covs: list[ResidualCov]
  .defined_params: list[DefinedParam]

Regression
  .lhs: str                    # left-hand side variable name
  .terms: list[Term]           # right-hand side terms
  .has_intercept: bool         # True unless "0 +" appears

Term
  .variable: str               # variable name
  .label: str | None           # coefficient label (e.g., "a" in "a*X")

ResidualCov
  .var1: str
  .var2: str

DefinedParam
  .name: str                   # e.g., "indirect"
  .expression: str             # e.g., "a*b"
```

### GraphInfo (returned by `build_graph`)

```
GraphInfo
  .topological_order: list[str]         # valid topological sort of the DAG
  .exogenous: set[str]                  # nodes with no parents in the DAG
  .endogenous: set[str]                 # nodes with at least one parent
  .residual_blocks: list[set[str]]      # connected components of ~~ edges
  .has_edge(source: str, target: str) -> bool
```

### PathModel (returned by `fit`)

```
PathModel
  .pymc_model: pm.Model
  .graph() -> object                              # DAG representation
  .equations() -> object                          # human-readable equation list
  .design(var: str) -> object with .columns       # design matrix info
  .priors() -> object                             # resolved priors
  .sample(**kwargs) -> az.InferenceData
  .do(set=None, shift=None, kind="mean") -> DoResult
  .summary() -> pd.DataFrame (or similar)
  .effects_summary() -> pd.DataFrame (or similar)
  .effect(path: str) -> object
```

### DoResult (returned by `do`)

```
DoResult
  .mean(var: str) -> float
  .hdi(var: str, prob: float = 0.94) -> array-like with 2 elements (lower, upper)
  .__sub__(other: DoResult) -> DoResult    # contrast arithmetic
```

## Milestone Details

### M1: DSL Parser

**Goal**: Parse the spec string into a `Spec` object.

**What to handle**:
- Regression statements: `y ~ x1 + x2`, `y ~ a*x1 + b*x2`, `y ~ 0 + x1`
- Residual covariance: `y1 ~~ y2`
- Defined parameters: `name := expression`
- Statement separators: newlines and semicolons
- Whitespace robustness (extra spaces, blank lines)
- Error cases: duplicate LHS, empty spec, malformed syntax

**Implementation notes**:
- A hand-written parser or regex-based parser is fine; no need for a parser generator.
- Return typed dataclass nodes.
- Keep the parser pure — no side effects, no data dependency.

### M2: Graph Builder

**Goal**: Convert `Spec` → `GraphInfo`.

**What to handle**:
- Directed edges from regression terms (each RHS variable → LHS)
- Topological ordering of the DAG
- Exogenous vs endogenous classification
- Connected components of `~~` edges → residual blocks
- Cycle detection with clear error message

**Implementation notes**:
- Add `networkx` to `pyproject.toml` dependencies during this milestone.
- `build_graph` should raise a descriptive error (from `pathmc.exceptions`) on cycles.

### M3: PathModel + Design Matrices

**Goal**: Create the `PathModel` class and build design matrices from parsed formulas.

**What to handle**:
- `pathmc.fit(spec_string, data=df)` returns a `PathModel`
- `.design(var)` returns a DataFrame-like object with correct column names
- Intercept included by default, suppressed by `0 +`
- Uses `patsy` for formula → design matrix conversion

### M4: Gaussian Compiler

**Goal**: Compile parsed spec + design matrices into a `pm.Model`.

**What to handle**:
- One Gaussian likelihood per endogenous variable
- Coefficient priors (default: `Normal(0, 10)` or similar weakly informative)
- Scale priors (default: `HalfNormal` or `HalfCauchy`)
- Stable parameter naming with ArviZ coords
- `.pymc_model` attribute on PathModel
- `.sample()` wraps `pm.sample()` and stores the InferenceData

### M5: Introspection

**Goal**: Implement `graph()`, `equations()`, `design()`, `priors()` on PathModel.

These methods should work **before** sampling (they describe model structure, not results).

- `graph()`: return a representation of the DAG (graphviz Digraph recommended for notebook rendering)
- `equations()`: return a human-readable list of structural equations
- `design(var)`: return design matrix info (already built in M3)
- `priors()`: return resolved prior specifications per parameter

### M6: do() Cross-sectional

**Goal**: Implement `do(set=..., kind="mean")` on PathModel. Requires sampling first.

**What to handle**:
- Mean propagation through DAG in topological order
- Intervened variable's structural equation is skipped (parents have no influence)
- `DoResult` with `.mean(var)`, `.hdi(var)`, and contrast arithmetic (`__sub__`)
- Raise an appropriate error if called before `.sample()`

### M7: Residual Covariance (~~)

**Goal**: Modify the compiler to produce MvNormal blocks for ~~-connected variables.

**What to handle**:
- LKJ prior for the correlation matrix of each residual block
- Priors for residual standard deviations
- Guard: `~~` only between Gaussian outcomes; raise error if a Bernoulli/other family variable is involved
- This requires at least a stub `families` parameter on `fit()` to distinguish Gaussian from non-Gaussian

### M8: Effects + Defined Params

**Goal**: Implement `effects_summary()` and defined parameter evaluation.

**What to handle**:
- Labeled coefficient extraction from posterior draws
- Arithmetic evaluation of `:=` expressions over posterior draws (e.g., `indirect := a*b`)
- `effects_summary()` returns a DataFrame with posterior summaries for all labels + defined params
- `effect(path)` for path-based effect queries (e.g., `"X -> M -> Y"`)

### M9: Integration Smoke Tests

**Goal**: All end-to-end smoke tests pass. These verify the full pipeline with actual MCMC sampling.

Key verifications:
- Fit → sample → summary workflow completes
- Defined params (`:=`) appear in effects summary with finite values
- `do()` ATE has correct sign for a known DGP (positive X→Y effect in simulated data)
- Correlated residuals model fits and produces summaries

### M10: Documentation ✓

**Goal**: Quarto site builds cleanly.

Required pages:
- `index.qmd`: landing page (already exists)
- `intro.qmd`: conceptual overview (DSL, DAGs, do-operator, causal assumptions/limitations)
- Example notebooks:
  - Mediation (cross-sectional, labels, `:=`, `do()`)
  - Correlated residuals (`~~`)
  - Simple `do()` queries
  - (Optional) One applied example demonstrating the full workflow

### M11: Bernoulli-logit Family

**Goal**: Support binary outcomes via `families={"Y": "bernoulli"}`.

**What to handle**:
- Bernoulli likelihood with logit link (`pm.Bernoulli(logit_p=...)`)
- No sigma parameter for Bernoulli variables
- `~~` guard already rejects non-Gaussian variables
- `do(kind="mean")` applies inverse-logit for Bernoulli outcomes (returns probabilities in (0, 1))
- Introspection: `priors()` omits sigma for Bernoulli; `equations()` notes logit link

**Implementation notes**:
- The `families` dict is resolved at compile time; defaults to `"gaussian"`.
- The resolved families must be stored on `PathModel` and passed to `run_do()`.

### M12: Predictive do() ✓

**Goal**: Implement `do(kind="predictive")` which adds residual noise at each propagation step.

**What to handle**:
- Gaussian: after computing `mu`, draw `mu + Normal(0, sigma)` using posterior sigma draws
- Bernoulli: draw binary `Bernoulli(expit(linear_predictor))` values
- `DoResult` interface unchanged; HDIs will be wider than mean propagation
- Contrast arithmetic still works

---

## v0.2 Milestones — Panel Mode

### M13: `add_lags()` utility

**Goal**: Standalone data-preprocessing helper that creates lag columns within each panel unit.

**Public API**: `pathmc.add_lags(df, variables, lags, panel={"unit": ..., "time": ...})`

**What to handle**:
- Sort within unit by time
- Create `{var}_lag{k}` columns via `groupby(unit).shift(k)`
- First `k` rows per unit get `NaN` for lag-k columns
- Validate: `unit` and `time` columns exist; `variables` exist in df

### M14: Panel-aware `fit()` + random intercepts

**Goal**: Accept `panel=` and `pooling=` on `fit()`, compile hierarchical intercepts per unit.

**What to handle**:
- `panel={"unit": "region", "time": "week"}` parameter on `fit()`
- `pooling="partial"`: random intercepts per unit for each endogenous variable
- `pooling=None` (default): cross-sectional behavior unchanged
- Compiler emits group-level intercepts (`mu_alpha`, `sigma_alpha`, `alpha` per variable)
- `priors()` includes group-level parameters
- `summary()` includes group-level parameters
- `do()` uses mean of group intercepts for propagation

### M15: Random slopes

**Goal**: Optional per-unit slopes for specified predictors.

**What to handle**:
- `pooling={"intercept": True, "slopes": ["var1"]}` syntax
- Partially pooled slopes: `beta_j ~ Normal(mu_beta, sigma_beta)` per unit
- Fixed coefficients still work for non-slope variables
- Start with independent random slopes (correlated random effects are a stretch goal)

### M16: Time-forward `do(simulate_over="time")`

**Goal**: Panel `do()` that propagates interventions forward through time.

**What to handle**:
- `simulate_over="time"`: activates temporal propagation
- `init_from="observed"`: use observed data for initial conditions
- For each unit, iterate through time steps, computing lagged values from previously simulated values
- Intervened variables use the fixed value; endogenous variables propagate using coefficients
- `kind="predictive"` adds residual noise at each step
- Error if `simulate_over="time"` used without `panel=`

### M17: Panel smoke tests

**Goal**: End-to-end integration tests for the full panel pipeline.

**What to handle**:
- `add_lags()` → `fit(panel=..., pooling="partial")` → `sample()` → `summary()` completes
- `do(simulate_over="time")` produces sensible ATE (correct sign for known DGP)
- Random intercepts produce per-unit variation
- Panel model with Bernoulli outcomes works
- `model.graph()` works for panel models

### M18: Panel documentation

**Goal**: Two new example pages with rendered outputs and visualizations.

**Required pages**:
- `docs/examples/did.qmd`: Difference-in-Differences with panel data
- `docs/examples/mmm_basic.qmd`: Basic Media Mix Model with lags and panel structure

Both pages include DOT diagrams, `model.graph()`, matplotlib visualizations, and rendered cell outputs. Add a "Panel mode" section to `intro.qmd`.

---

## v0.3 Milestones — Transforms, Families, PPC

| #   | Name                          | Gate tests                   | Depends on   | Status |
| --- | ----------------------------- | ---------------------------- | ------------ | ------ |
| M19 | Transform parser              | `test_transforms_parse.py`   | M1           | ✓      |
| M20 | Transform registry + compiler | `test_transforms_compile.py` | M19, M4      | ✓      |
| M21 | Transforms under do()         | `test_transforms_do.py`      | M20, M6, M16 | ✓      |
| M22 | Additional families           | `test_families.py`           | M4           | ✓      |
| M23 | Posterior predictive          | `test_ppc.py`                | M4           | ✓      |
| M24 | v0.3 smoke tests              | `test_v03_smoke.py`          | M19–M23      | ✓      |
| M25 | v0.3 documentation            | `quarto render` exits 0      | M24          | ✓      |

### M19: Transform Parser

**Goal**: Extend the DSL parser to recognize transform expressions with named parameters and nesting.

**New AST node**: `TransformCall` with `name`, `input_expr` (string or nested `TransformCall`), and `params` dict.

**What to parse**:
- `y ~ adstock(x, decay=theta)` — single transform
- `y ~ a*adstock(x, decay=theta)` — labeled + transform
- `y ~ logistic_saturation(adstock(x, decay=theta), lam=lam)` — nested composition
- Error cases: missing closing paren, empty param name

### M20: Transform Registry + Compiler

**Goal**: Create `pathmc/transforms.py` with a registration mechanism, implement `adstock` and `logistic_saturation`, and compile transform parameters into the PyMC model.

**Built-in transforms**:
- `adstock(x, decay=theta)`: geometric decay with `Beta(2, 2)` prior on decay, panel-aware accumulation
- `logistic_saturation(x, lam=lam)`: `1 - exp(-lam * x)` with `HalfNormal(1)` prior on lam

**Compiler changes**: detect transform terms, emit constrained priors, compute transformed predictors as PyMC tensors, use in regression.

### M21: Transforms under do()

**Goal**: Recompute transforms during interventional simulation in both cross-sectional and panel modes.

For cross-sectional: saturation applies pointwise; adstock degenerates to identity for single-value interventions.
For panel `do(simulate_over="time")`: adstock accumulates correctly in time-forward simulation.

### M22: Additional Families

**Goal**: Support Poisson, NegBinomial, and StudentT likelihoods.

- **Poisson**: log link, no sigma, `~~` guard rejects
- **NegBinomial**: log link, dispersion parameter, `~~` guard rejects
- **StudentT**: identity link, degrees of freedom `nu`, sigma retained

`do()` applies the correct inverse link function for each family.

### M23: Posterior Predictive Checks

**Goal**: Add `.predict()` method on `PathModel` wrapping `pm.sample_posterior_predictive()`.

Returns InferenceData with `posterior_predictive` group. Works for all families and `~~` block models.

### M24: v0.3 Smoke Tests

**Goal**: End-to-end integration tests combining transforms, new families, and PPC.

- MMM with `adstock` + `logistic_saturation` + panel mode: fit, sample, do(), predict
- Poisson and StudentT pipelines
- Transform parameter recovery from known DGP

### M25: v0.3 Documentation

**Goal**: Update MMM example with adstock + saturation transforms and PPC. Add transforms section to intro.qmd. Append v0.3 milestones.

---

## v0.4 Milestones — Causal Workbench

| #   | Name                       | Gate tests                  | Depends on | Status |
| --- | -------------------------- | --------------------------- | ---------- | ------ |
| M26 | Fix do() random slopes     | `test_do_slopes.py`         | M15, M16   | ✓      |
| M27 | Identification helpers     | `test_identification.py`    | M2         | ✓      |
| M28 | Causal query sugar         | `test_causal_queries.py`    | M6, M26    | ✓      |
| M29 | Standardized effects       | `test_standardized.py`      | M8         | ✓      |
| M30 | v0.4 smoke tests           | `test_v04_smoke.py`         | M26–M29    | ✓      |
| M31 | v0.4 documentation         | `quarto render` exits 0     | M30        | ✓      |

### M26: Fix do() random slopes

**Goal**: Make `do()` propagate random slopes, not just random intercepts.

**What to handle**:
- In `run_do` (cross-sectional): detect `slope_{var}_{predictor}` in posterior, average over units, add `slope_mean * values[predictor]` to linear predictor
- In `run_panel_do` (panel): select unit-specific slope `slope_arr.sel(unit=unit)`, add `slope_unit * parent_val` to linear predictor
- Result: `do()` queries on models with random slopes now reflect geo/unit-varying effects

### M27: Identification helpers

**Goal**: Provide backdoor adjustment set computation, collider warnings, and identifiability checks.

**New module**: `pathmc/identify.py`

**Public functions**:
- `adjustment_sets(graph_info, treatment, outcome)` → `list[set[str]]`
- `collider_warnings(graph_info, adjustment_vars, treatment, outcome)` → `list[str]`
- `is_identifiable(graph_info, treatment, outcome)` → `bool`

**PathModel methods**: `.adjustment_sets(treatment, outcome)`, `.is_identifiable(treatment, outcome)`

### M28: Causal query sugar

**Goal**: Provide convenience methods for common causal queries.

**PathModel methods**:
- `.ate(outcome, treatment, values=(0.0, 1.0))` → `DoResult` (contrast)
- `.cate(outcome, treatment, values=(0.0, 1.0), condition={...})` → `DoResult`
- `.prob(expr, set=None, kind="predictive")` → `float`

### M29: Standardized effects

**Goal**: Compute `stdyx`-style standardized coefficients from posterior draws and data moments.

**New function** in `pathmc/effects.py`: `build_standardized_effects(spec, idata, data)` → `pd.DataFrame`

**PathModel method**: `.standardized()` → DataFrame with mean, sd, HDI of standardized coefficients.

### M30: v0.4 smoke tests

**Goal**: End-to-end integration tests for the full v0.4 feature set.

- Full pipeline: fit → adjustment_sets → ate → standardized
- Panel model with random slopes → do() → verify slopes affect result
- Mediation: indirect standardized effect

### M31: v0.4 documentation

**Goal**: Update intro.qmd with sections on identification helpers, causal queries, and standardized effects. Verify docs build.
