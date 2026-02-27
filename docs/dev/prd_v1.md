# Product Requirements Document (PRD): `pathmc`

> **Status: v1 complete.** All milestones (M1–M31) pass. All Definition of Done criteria met. See [milestones.md](milestones.md) for details and [roadmap_post_v1.md](roadmap_post_v1.md) for planned future work.

## 1. Purpose

`pathmc` is a Python package for **Bayesian path analysis** (observed-variable SEM) designed for **rapid, iterative exploration of causal DAGs**. It compiles a concise, lavaan-inspired **formula-string DSL** into **PyMC** models, provides rich **model introspection**, and supports a first-class **do() operator** for **g-computation / interventional simulation**.

The package operates in two first-class modes:

- **Cross-sectional mode**: the DAG is applied to every observation independently.
- **Panel mode**: the DAG applies per time step, where each row is a unit-time observation. Dynamics are achieved through explicit lagged terms.

The DSL supports **user-defined transformations with estimable parameters**, enabling domain-specific applications such as Media Mix Modeling (MMM) with adstock and saturation curves.

## 2. Non-goals (v1)

- Full SEM with latent variables (CFA/SEM measurement models)
- Cyclic structural systems (feedback loops) as a default (may be future)
- Automatic causal identification beyond common helpers (e.g., full do-calculus)
- Full time-series residual structures (state-space, ARMA errors) as default
- Policy optimization (future feature, built on `do()` simulator)

## 3. Target Users

- Applied Bayesian analysts exploring **candidate causal DAGs**
- Researchers wanting a **lavaan-like** path-spec experience but with **Bayesian inference** and **interventional simulation**
- Domain practitioners (e.g., MMM, epidemiology, social science) who need to specify, fit, and interrogate structural causal models quickly

## 4. Core Concepts

- **Path analysis**: directed acyclic graph (DAG) over observed variables; endogenous nodes are modeled by regression-like structural equations.
- **Two modes**:
  - *Cross-sectional*: each row is an independent observation; the full DAG applies per row.
  - *Panel*: long-format data with `unit` and `time` indexes; dynamics via explicit lagged columns; the DAG applies per time step within unit.
- **Residual covariance** (`~~`): correlates residuals among *eligible continuous outcomes*.
- **Causal queries**: effects and predictions under intervention via `do()`; explicit caveats about causal assumptions.
- **Transformations with estimable parameters**: the DSL supports named transformations (e.g., `adstock`, `logistic_saturation`) whose parameters can be estimated during inference, enabling domain-specific nonlinearities.

## 5. Product Goals and Success Criteria

### Goals

1. Ergonomic DSL for systems of equations with `~`, `~~`, labels, and defined parameters (`:=`).
2. Correct compilation to PyMC with clear parameter naming and stable coordinates.
3. Strong introspection: graph, equations, design matrices, resolved priors.
4. First-class `do()` for cross-sectional and panel forward simulation.
5. Transformations with estimable parameters that recompute under `do()`.

### Success Criteria

- Users can:
  - specify a path model in ≤20 lines
  - fit it via `pathmc.fit()`
  - inspect graph and equations
  - compute causal effects via `do()` and effect helpers
  - use transformations with estimable parameters and simulate counterfactual scenarios
- Repo includes:
  - passing contract tests
  - Quarto docs with intro and executable examples

## 6. Scope

### v0.1 (MVP)

- DSL parsing: `~`, `~~`, coefficient labels, `:=` defined parameters
- Graph building + validation: acyclic directed graph, topological order
- Design matrices: formula parsing, intercept rules (`1` vs `0 +`)
- Families:
  - Gaussian (required)
  - Bernoulli-logit (optional but recommended)
- `~~` residual covariance for Gaussian outcomes only (LKJ + multivariate normal blocks)
- Cross-sectional `do()`:
  - mean-propagation (`kind="mean"`)
  - predictive sampling (`kind="predictive"`) for Gaussian
- Introspection: `graph()`, `equations()`, `design()`, `priors()`
- Effects: labeled coefficient extraction + `:=` evaluation; basic path total effects for linear-Gaussian
- Docs: Quarto intro + at least 3 examples (mediation, correlated residuals, simple `do()`)

### v0.2 — Panel mode

- Panel/longitudinal support:
  - long-format with `panel={"unit":..., "time":...}`
  - `add_lags()` helper
  - time-forward `do(simulate_over="time")` when time index present
  - random intercepts by unit; optional random slopes
- Docs: at least 2 panel examples:
  - difference-in-differences (simple treatment/control over time, `do()` to estimate ATT)
  - basic MMM without transforms (linear spend → sales with lags, panel over regions/weeks)

### v0.3 — Transformations & additional families

- Transformations with estimable parameters:
  - general mechanism for registering named transforms with constrained parameters
  - built-in transforms: `adstock(x, decay=theta)`, `logistic_saturation(x, lam=lam)`
  - transforms recompute under `do()` interventions
- Additional families (prioritized): Poisson, NegBin; StudentT
- Diagnostics:
  - posterior predictive checks for outcomes
  - residual-corr PPC for Gaussian blocks
- Docs: update MMM example to include adstock/logistic_saturation transforms with estimable parameters

### v0.4 — Causal workbench

- Fix `do()` random slopes propagation:
  - `run_do` and `run_panel_do` must propagate `slope_` variables alongside `alpha_` intercepts
  - Cross-sectional: average slopes over units; panel: use unit-specific slopes
- Identification helpers:
  - minimal adjustment sets (backdoor criterion) for a given DAG and target effect
  - collider warnings / forbidden adjustments
  - report whether a target effect is identifiable under standard assumptions
- Causal query language — thin sugar on `do()`:
  - `.ate(outcome, treatment, values=(lo, hi))` → `DoResult` contrast
  - `.cate(outcome, treatment, values, condition={...})` → conditional ATE
  - `.prob(expr, set={...})` → `P(expr | do(set))` via predictive draws
- Standardized effects as a post-fit view (`stdyx`-style), computed from posterior draws + data moments

## 7. DSL Specification

### Statements

- **Regression/path**: `y ~ x1 + x2 + ...`
  - remove intercept: `y ~ 0 + x1 + x2`
  - labeled coefficients: `y ~ a*x1 + b*x2`
- **Residual covariance**: `y1 ~~ y2`
  - variance (optional): `y ~~ y`
- **Defined parameters**: `name := expression` (references labels)

### Transform expressions

Transforms are named functions with estimable parameters that can appear on the RHS of structural equations:

- `adstock(var, decay=theta)` — geometric adstock with estimable decay
- `logistic_saturation(var, lam=lam)` — logistic saturation with estimable steepness
- Composability: `logistic_saturation(adstock(tv_spend, decay=theta_tv), lam=lam_tv)`

The transform mechanism is general: users can register custom transforms with constrained parameters.

### Parsing Requirements

- robust to whitespace
- statements separated by newline or `;`
- clear error messages for:
  - duplicate LHS equations
  - unknown references in `:=`
  - cycles in directed graph
  - `~~` involving non-Gaussian outcomes unless explicitly enabled (future)

### Examples

#### Fork (common cause)

`Z` causes both `X` and `Y`. Conditioning on `Z` blocks the non-causal path.

```python
spec = """
X ~ Z
Y ~ X + Z
"""
fit = pathmc.fit(spec, data=df)
```

#### Chain (mediation)

`X → M → Y` with labeled coefficients and a defined indirect effect.

```python
spec = """
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""
fit = pathmc.fit(spec, data=df)
```

#### Collider

`X` and `Y` both cause `C`. No confounding of `X → Y` unless you condition on `C`.

```python
spec = """
C ~ X + Y
"""
fit = pathmc.fit(spec, data=df)
```

#### Multiple mediators with correlated residuals

A richer model: treatment `T` affects outcome `Y` through two parallel mediators `M1` and `M2`, whose residuals are correlated.

```python
spec = """
M1 ~ a1*T
M2 ~ a2*T
Y  ~ b1*M1 + b2*M2 + c*T

M1 ~~ M2

indirect1 := a1*b1
indirect2 := a2*b2
total     := c + a1*b1 + a2*b2
"""
fit = pathmc.fit(spec, data=df)
```

#### Panel mode with lagged effects

Weekly sales driven by advertising spend with a one-period lag, fit per unit over time.

```python
spec = """
sales ~ sales_lag1 + spend_lag1 + trend
"""
fit = pathmc.fit(
    spec,
    data=df,
    panel={"unit": "region", "time": "week"},
)
```

## 8. Public API

### Model construction & fitting

```python
fit = pathmc.fit(spec, data=df, families=..., priors=..., panel=..., pooling=...)
idata = fit.sample(draws=..., tune=..., chains=..., target_accept=...)
```

### Introspection

```python
fit.graph()          # DAG and residual/bidirected edges
fit.equations()      # resolved equations (expanded terms)
fit.design("y")      # design matrix columns + metadata
fit.priors()         # resolved priors per parameter
fit.pymc_model       # underlying pm.Model
```

### Summaries

```python
fit.summary()
fit.effects_summary()        # includes := params
```

### `do()` operator

```python
baseline = fit.do(kind="mean")
scenario = fit.do(set={"x": 1}, kind="mean")
contrast = scenario - baseline
contrast.mean("y")
contrast.hdi("y", 0.95)
```

#### Panel `do()`

```python
scenario = fit.do(
  set={"spend": 120},
  simulate_over="time",
  init_from="observed",
  kind="mean",
)
```

### Effects API

```python
fit.effect("x -> y")
fit.effect("x -> m -> y")
fit.effects_summary()    # includes := defined params like indirect := a*b
```

### Causal queries (v0.4)

```python
fit.ate("y", "x", values=(0.0, 1.0))
fit.cate("y", "x", values=(0.0, 1.0), condition={"z": 2.0})
fit.prob("y > 0", set={"x": 1.0})
```

### Identification (v0.4)

```python
fit.adjustment_sets("x", "y")
fit.is_identifiable("x", "y")
fit.collider_warnings({"c"}, "x", "y")
```

### Standardized effects (v0.4)

```python
fit.standardized()  # stdyx-standardized coefficients
```

## 9. Modeling Semantics

### Directed edges

- Each `lhs ~ rhs` creates directed edges `parent -> lhs` for each RHS variable.
- Directed graph must be acyclic (unless an explicit future option is used).

### Residual covariance (`~~`)

- `y1 ~~ y2` specifies **residual covariance** among eligible continuous outcomes.
- v0.1/v1 restriction: only among Gaussian/StudentT outcomes.
- Implementation: connected components of `~~` form multivariate blocks:
  - LKJ prior for correlation + priors for residual scales
  - `MvNormal` likelihood for the block

### Cross-sectional mode

- Default mode. Each observation is independent.
- `do()` propagates interventions through the DAG in topological order for each observation.

### Panel mode

- Activated by `panel={"unit": ..., "time": ...}`.
- Data is long-format with `unit` and `time` columns.
- Dynamics are represented via **explicit lag columns** (e.g., `sales_lag1`).
- `add_lags()` produces lag columns within unit sorted by time.
- `do(simulate_over="time")` propagates forward in time, using simulated values for lagged dependencies.

### Transformations with estimable parameters

- Named transforms (e.g., `adstock`, `logistic_saturation`) appear on the RHS and introduce constrained latent parameters.
- `adstock(x, decay=theta)`: `theta` constrained to (0,1) via Beta prior or logistic transform; computed within unit over time when panel mode is active.
- `logistic_saturation(x, lam=lam)`: `lam > 0` with positive-support prior; applies `1 - exp(-lam * x)` pointwise.
- Transforms must recompute under `do()` interventions.

### `do()` semantics

- `do(set={...})` overrides the intervened variable(s) and ignores their structural equations.
- `do(shift={...})` is reserved for soft interventions (not yet implemented).
- In panel/time-forward mode:
  - simulation proceeds in time order and uses simulated values for lagged dependencies
  - `init_from` controls initial state (observed vs user-specified)
- Output object supports `.mean(var)`, `.hdi(var)`, `.sample(var, n)` and arithmetic contrasts.

## 10. Diagnostics & Reporting

- Posterior summaries via ArviZ
- Posterior predictive checks (PPC) for each outcome
- Residual correlation PPC for Gaussian blocks
- (Future) exploratory suggestion tools (edge/covariance suggestions) must be opt-in and clearly labeled.

## 11. Engineering Requirements

### Repository deliverables

- Working Python package with `pyproject.toml`
- Tests:
  - contract tests (fast) verifying parsing, graph, compilation artifacts, transform wiring, `do()` plan
  - minimal smoke sampling tests (small draws) to ensure end-to-end runs
- Documentation site in **Quarto**:
  - `index.qmd`: landing
  - `intro.qmd`: conceptual overview (DSL, DAGs, do-operator, limitations)
  - examples:
    - mediation (cross-sectional)
    - correlated residuals (`~~`)
    - panel + lags + time-forward `do()`
    - MMM with estimable transforms + scenario simulation (application showcase)

### Naming & coordinates (stability)

- Parameter naming conventions must be documented and stable.
- Use xarray/ArviZ coords for:
  - equations
  - coefficient names
  - multivariate blocks

### Implementation order (recommended)

1. DSL parser → AST
2. Graph builder + validators
3. Design matrix builder
4. Gaussian compiler (univariate)
5. `do()` cross-sectional
6. Panel indexing + `add_lags()`
7. Time-forward `do()`
8. `~~` multivariate blocks
9. Effects (`:=`) evaluation
10. Transformations with estimable parameters + recomputation under `do()`
11. Docs & examples stabilization

### Example use case: Media Mix Modeling

MMM is a motivating application for pathmc. A typical MMM workflow would:

- specify a DAG with marketing channels → sales
- use `adstock()` and `logistic_saturation()` transforms with estimable parameters
- fit in panel mode (units = regions or markets, time = weeks)
- use `do()` to simulate counterfactual spend scenarios
- (future) use `optimize_policy()` to find optimal budget allocations

This workflow exercises most of pathmc's core features and serves as a flagship example in the documentation.

## 12. Roadmap (post v1)

See [roadmap_post_v1.md](roadmap_post_v1.md) for planned future features and their architectural implications.

## 13. Contract Scenarios (Acceptance)

`pathmc` must satisfy a small suite of contract scenarios:

1. Cross-sectional mediation with labels + `:=` + `do()` ATE sanity
2. Gaussian `~~` block compiles to LKJ + MvNormal; residual corr exposed
3. Mixed families compile independently; `~~` disallowed unless enabled
4. Panel `add_lags()` aligns within unit/time; dynamic spec remains acyclic
5. Time-forward `do()` propagates through lag terms
6. Transforms with estimable parameters are constrained, appear in transform graph, and recompute under `do()`

## 14. Risks and Mitigations

- **Identifiability with estimable transforms** (e.g., adstock vs lag vs saturation): mitigate with strong priors, start with simple examples, provide diagnostics and warnings.
- **Panel ordering bugs**: enforce sorting checks; provide panel report.
- **Over-claiming causality**: docs must emphasize assumptions; provide warnings on `do()` use.

## 15. Definition of Done (v1)

- `pip install -e .` works
- `pytest` passes
- Quarto site builds with intro + ≥4 executable examples
- Both cross-sectional and panel modes functional
- `do()` works in both modes
- At least one example with estimable transform parameters
- Identification helpers report adjustment sets and collider warnings
- `ATE()` / `CATE()` query sugar functional
- Standardized effects available as post-fit view
