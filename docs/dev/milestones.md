# Milestones — pathmc v0.1

These milestones cover the v0.1 (MVP) scope defined in `prd_v1.md`. Post-v0.1 scope (panel mode, transforms, causal workbench) is tracked separately.

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

| #   | Name                       | Gate tests                                  | Depends on |
| --- | -------------------------- | ------------------------------------------- | ---------- |
| M1  | DSL Parser                 | `test_parse.py`                             | —          |
| M2  | Graph Builder              | `test_graph.py`                             | M1         |
| M3  | PathModel + Design Matrices| `test_compile.py::TestDesignMatrix`         | M1, M2     |
| M4  | Gaussian Compiler          | `test_compile.py` (all)                     | M3         |
| M5  | Introspection              | `test_introspection.py`                     | M4         |
| M6  | do() Cross-sectional       | `test_do.py`                                | M4         |
| M7  | Residual Covariance (~~)   | `test_residual_cov.py`                      | M4         |
| M8  | Effects + Defined Params   | `test_effects.py`                           | M4         |
| M9  | Integration Smoke Tests    | `test_smoke.py`                             | M1–M8      |
| M10 | Documentation              | `cd docs && quarto render` exits 0          | M9         |

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

### M10: Documentation

**Goal**: Quarto site builds cleanly.

Required pages:
- `index.qmd`: landing page (already exists)
- `intro.qmd`: conceptual overview (DSL, DAGs, do-operator, causal assumptions/limitations)
- Example notebooks:
  - Mediation (cross-sectional, labels, `:=`, `do()`)
  - Correlated residuals (`~~`)
  - Simple `do()` queries
  - (Optional) One applied example demonstrating the full workflow
