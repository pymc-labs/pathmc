# Generality Audit Report

**Issue**: #48 — Audit the DSL's range and scope for model generality

## Executive summary

pathmc's DSL can express a **broad and practically useful** subset of structural causal models. The lavaan-style syntax (`~`, `~~`, `:=`) covers the core building blocks of path analysis, and the extensions — panel mode, hierarchical pooling, transforms with estimable parameters, latent deterministic mediators, and five distribution families — push the package well beyond basic regression toward a genuine causal modeling workbench.

The current DSL covers the most common model patterns that applied analysts encounter: mediation, moderation (via data interactions), parallel mediators with correlated residuals, hierarchical/panel models with random intercepts and slopes, autoregressive dynamics, and media mix models. The generative model + `pm.observe()` + `pm.do()` architecture is the correct foundation for extensibility.

There are, however, meaningful classes of models that PyMC can express but the current DSL cannot. This audit identifies them, assesses their importance for pathmc's positioning, and recommends which gaps to close — ordered by the value they would add to the package's appeal as a "causal orchestration layer around PyMC."

## What the DSL can express today

### Structural equation features

| Feature | DSL syntax | Status |
|---------|-----------|--------|
| Directed edges (regression) | `Y ~ X1 + X2` | Complete |
| Labeled coefficients | `Y ~ a*X + b*M` | Complete |
| Intercept suppression | `Y ~ 0 + X` | Complete |
| Defined parameters | `indirect := a*b` | Complete |
| Residual covariance | `M1 ~~ M2` | Complete (Gaussian only) |
| Named transforms | `adstock(x, decay=θ)` | Complete |
| Nested transforms | `sat(adstock(x, decay=θ), lam=λ)` | Complete |
| Custom transforms | `register_transform()` | Complete |
| Lag terms | `lag(sales)` | Complete (lag-1 only) |
| Latent deterministic mediators | `latent=["M"]` | Complete |
| Multi-line continuation | `+ term` on new line | Complete |

### Distribution families

| Family | Link | Use case | Docs example? |
|--------|------|----------|--------------|
| Gaussian | Identity | Continuous outcomes | Yes (many) |
| Bernoulli | Logit | Binary outcomes | Yes (5 examples) |
| Poisson | Log | Count data | No |
| NegBinomial | Log | Overdispersed counts | No |
| StudentT | Identity | Heavy-tailed continuous | No |

### Panel and hierarchical features

| Feature | API | Status |
|---------|-----|--------|
| Panel mode | `panel={"unit": ..., "time": ...}` | Complete |
| Random intercepts | `pooling="partial"` | Complete |
| Random slopes | `pooling={"intercept": True, "slopes": ["X"]}` | Complete |
| Time-forward simulation | `do(simulate_over="time")` | Complete |
| Time-varying interventions | `do(set={"X": np.array(...)})` | Complete |

### Causal query features

| Feature | API | Status |
|---------|-----|--------|
| do-operator | `model.do(set={...})` | Complete |
| ATE | `model.ate(outcome, treatment, values)` | Complete |
| CATE | `model.cate(outcome, treatment, values, condition)` | Complete |
| Probability queries | `model.prob("Y > 0", set={...})` | Complete |
| Path effects | `model.effect("X -> M -> Y")` | Complete |
| Adjustment sets | `model.adjustment_sets(treatment, outcome)` | Complete |
| Identifiability | `model.is_identifiable(treatment, outcome)` | Complete |
| Collider warnings | `model.collider_warnings(vars, treatment, outcome)` | Complete |
| Standardized effects | `model.standardized()` | Complete |
| Positivity warnings | Automatic in `do()` | Complete |
| Posterior predictive | `model.predict()` | Complete |

### Model patterns demonstrated in docs

| Pattern | Example count | Notes |
|---------|:---:|-------|
| Cross-sectional mediation | 8 | Core use case, well covered |
| Correlated residuals (`~~`) | 1 | Only two-variable block |
| Panel / DiD | 1 | Basic panel DiD |
| Autoregressive (AR1) | 1 | `lag(sales)` |
| MMM (adstock + saturation) | 4 | Flagship application |
| Mixed families (Gaussian + Bernoulli) | 5 | Strong coverage |
| Latent deterministic mediators | 1 | MMM awareness only |
| Front-door criterion | 1 | Conceptual (not automated) |
| Multi-equation funnels | 2 | SaaS + vaccine |

## Model types the DSL can express but lacks examples for

These models work today with the existing DSL but are not demonstrated in the documentation. Each represents an opportunity for a new example notebook.

### 1. Count outcome models (Poisson, NegBinomial)

The DSL supports `families={"clicks": "poisson"}` and `families={"events": "negbinomial"}`, and the compiler correctly emits log-link likelihoods with `do()` support. However, no example notebook demonstrates these families.

**Impact**: High. Count outcomes are ubiquitous in marketing (clicks, conversions, sign-ups), epidemiology (disease counts), and web analytics (page views, events). Demonstrating these families would immediately broaden pathmc's perceived applicability.

**Recommendation**: New example notebook — e.g., "Marketing funnel with count outcomes" showing `impressions → clicks (Poisson) → conversions (Bernoulli)` or "Epidemiological count model" with Poisson/NegBin outcomes.

### 2. Heavy-tailed outcomes (StudentT)

The `studentt` family is implemented and tested but has no dedicated example. Heavy-tailed outcomes are common in finance (returns), pricing (willingness to pay), and any domain with outliers.

**Recommendation**: Either add a standalone example or add a StudentT variant to an existing example (e.g., dynamic pricing with heavy-tailed demand).

### 3. Larger residual covariance blocks (3+ variables)

The `~~` operator supports arbitrary connected components — if you write `M1 ~~ M2; M2 ~~ M3`, all three share a single MvNormal block with an LKJ prior. Only the two-variable case (`M1 ~~ M2`) is demonstrated.

**Recommendation**: Extend the correlated residuals example or add a new one with 3+ mediators sharing residual covariance.

### 4. Cross-sectional latent mediators

The `latent=["M"]` feature works for any model, but the only example uses it in a panel MMM context. A cross-sectional latent mediator (e.g., unobserved "customer satisfaction" mediating between service quality and retention) would broaden understanding.

**Recommendation**: New cross-sectional example with a latent deterministic mediator.

### 5. Interaction terms (moderation) via data columns

The DSL does not parse interaction syntax (no `X:Z` or `X*Z`), but users can create interaction columns in their data (`df["XZ"] = df["X"] * df["Z"]`) and include them as predictors. This works but is not documented.

**Recommendation**: Document the interaction pattern with an example showing moderation analysis via pre-computed interaction columns.

## Model types the DSL cannot currently express

These are models that PyMC itself can handle but the pathmc DSL cannot encode. Each is assessed for importance to pathmc's positioning.

### Priority 1: High value, architecturally feasible

#### 1a. Interaction terms in the DSL

**What's missing**: The parser does not support `Y ~ X + Z + X:Z` or `Y ~ X*Z` interaction syntax. Users must pre-compute interaction columns. This is the single most commonly requested regression feature that the DSL lacks.

**Why it matters**: Moderation analysis (effect modification) is central to CATE estimation and heterogeneous treatment effects. It's also a bread-and-butter SEM feature. Bambi and lavaan both support interactions in-formula.

**Complexity**: Medium. The parser needs a new `Interaction` term type; the compiler needs to compute the product term symbolically from `pm.Data` or upstream RVs. The `do()` operator must recompute interactions when either constituent variable is intervened on.

#### 1b. Higher-order lags (lag-2, lag-3)

**What's missing**: The DSL enforces lag-1 only by design (higher lags rejected with "the influence of t-2 on t should be mediated through t-1"). This is a principled design choice but limits expressiveness for time series practitioners who expect direct higher-order lag terms.

**Why it matters**: Many applied time series models need AR(2) or AR(3) structures. Forcing everything through lag-1 imposes a chain structure (t-2 → t-1 → t) that may not match the data-generating process. For example, seasonality at lag-7 (daily data) or lag-12 (monthly data) is awkward to express as seven or twelve cascading lag-1 steps.

**Complexity**: Medium. The scan loop already handles lag-1 carry state; extending to lag-k requires k carry variables per lagged term and a wider `outputs_info`.

#### 1c. Custom priors

**What's missing**: The `fit()` function accepts `**kwargs` for future options, and the transform system has `emit_prior()`, but there is no user-facing API for specifying custom priors on regression coefficients (e.g., `priors={"beta_Y_X": pm.Normal.dist(0, 1)}`).

**Why it matters**: Prior specification is fundamental to Bayesian modeling. The current defaults (`Normal(0, 10)` for betas, `HalfNormal(1)` for sigmas) are reasonable but generic. Domain experts need to encode prior knowledge — strong priors for expected effect sizes, informative priors from previous studies, shrinkage priors for variable selection.

**Complexity**: Low-Medium. The compiler already emits priors in predictable locations; accepting a user-supplied prior dict and overriding the defaults is architecturally straightforward.

### Priority 2: Medium value, moderate complexity

#### 2a. Nonlinear structural equations

**What's missing**: Structural equations are linear in coefficients: `Y = β₀ + β₁X + β₂M`. Transforms (adstock, saturation) add nonlinearity in predictors, but only for registered transform functions. Arbitrary nonlinear structural equations (e.g., `Y = β * X^α` with estimated `α`, or `Y = f(X)` with a GP/spline) are not expressible.

**Why it matters**: Many real-world relationships are fundamentally nonlinear — dose-response curves, diminishing returns beyond what logistic saturation captures, threshold effects, polynomial relationships. The transform registry is the right extension point, but users need to know how to register custom transforms.

**Recommendation**: Prioritize documentation of the custom transform API (`register_transform()`) with worked examples. A spline/GP transform would be a compelling addition but is architecturally complex.

#### 2b. Measurement models / latent factors (CFA/SEM)

**What's missing**: pathmc supports latent *deterministic* mediators (no measurement error, fully determined by parents) but not latent *factor* models. In classical SEM, latent factors are estimated from multiple observed indicators: `η =~ y1 + y2 + y3`. This is CFA (confirmatory factor analysis) and is semopy's core strength.

**Why it matters**: Latent factor models are widely used in psychology, marketing (brand perception scales), and social science. The `=~` operator would be the natural extension. However, this is a significant architectural addition — it requires a measurement model layer, priors on loadings, and integration with the structural layer.

**Assessment**: This is correctly identified as a non-goal for v1 in the PRD. It would substantially increase complexity and overlap with semopy's niche. **Recommend deferring** and clearly documenting pathmc's scope as observed-variable path analysis with deterministic latent mediators.

#### 2c. Spline / smooth terms

**What's missing**: No way to specify nonlinear effects via splines or smooth functions in the DSL (e.g., `Y ~ s(X)` as in GAMs/Bambi).

**Why it matters**: Many confounders need flexible adjustment (e.g., age, income). Bambi supports `bs(x, ...)` spline terms in its formula language. For pathmc, the primary concern is correct confounder adjustment — if the functional form is wrong, the causal estimate is biased.

**Complexity**: High. Would require integration with a spline basis function library and careful handling under `do()`.

#### 2d. Multi-family residual covariance

**What's missing**: The `~~` operator is restricted to Gaussian outcomes. Correlated residuals between a Bernoulli and a Gaussian variable, or between two Bernoulli variables, are not supported.

**Why it matters**: In multi-equation models with mixed outcome types (e.g., a SaaS funnel with continuous engagement + binary conversion), correlated residuals across families would better capture shared unmeasured causes. The current restriction forces independence across families.

**Complexity**: High. Would require a copula-based or latent probit approach rather than MvNormal.

### Priority 3: Lower value or correctly deferred

#### 3a. Cyclic structural systems (feedback loops)

**What's missing**: The DAG constraint prevents feedback loops (e.g., price → demand → price). Some economic systems have simultaneous equations that are inherently cyclic.

**Assessment**: Correctly deferred. Cyclic systems require fundamentally different estimation (e.g., 2SLS, equilibrium-based) and would undermine pathmc's clean do-operator semantics. The acyclicity constraint is a feature, not a limitation.

#### 3b. Instrumental variables

**What's missing**: No IV/2SLS estimation or IV identification automation. The front-door criterion is demonstrated but not automated.

**Assessment**: Partially deferred. A `frontdoor_identifiable()` check in `identify.py` would add value. Full IV estimation is a different estimation strategy from pathmc's g-computation approach.

#### 3c. State-space models / ARMA errors

**What's missing**: No ARMA error structure or state-space representation. The panel scan supports AR(1) via `lag()` but not moving average components or state-space transitions.

**Assessment**: Out of scope for pathmc's SEM-based design. CausalPy is the better tool for time series causal inference with complex temporal structures.

#### 3d. Nonparametric estimation (BART, GP)

**What's missing**: No built-in support for BART or GP regression as structural equations. PyMC supports both via `pm.BART` and `pm.gp`.

**Assessment**: This targets a different estimation philosophy (flexibility vs. interpretability). EconML's CATE methods are better positioned for nonparametric heterogeneous effects. pathmc's value is in interpretable structural models.

## Comparison with "just regression" packages

The issue text draws a comparison to Bambi, noting that "Bambi is supposedly 'just regression' but it can handle a very broad range of models." This comparison is apt and informative.

### What Bambi can do that pathmc cannot

| Bambi feature | pathmc equivalent | Gap? |
|---------------|-------------------|------|
| Single-equation regression | `Y ~ X1 + X2` | No gap |
| Interactions (`X:Z`, `X*Z`) | Pre-computed columns | **Yes — DSL gap** |
| Splines (`bs(X, ...)`) | Not available | **Yes — DSL gap** |
| Distributional models (varying σ) | Not available | **Yes — DSL gap** |
| Crossed/nested random effects | `pooling=` dict | Partial (simpler) |
| GAMs with smooth terms | Not available | Yes |
| Zero-inflated / hurdle | Not available | Yes |

### What pathmc can do that Bambi cannot

| pathmc feature | Bambi equivalent |
|----------------|-----------------|
| Multi-equation systems | Not available |
| Residual covariance (`~~`) | Not available |
| `do()` operator (graph surgery) | Not available |
| Time-forward panel simulation | Not available |
| Transforms with estimable parameters | Not available |
| Causal identification checks | Not available |
| Defined parameters (`:=`) | Not available |
| Latent deterministic mediators | Not available |
| Path-specific effects | Not available |
| Probability queries under intervention | Not available |

**Assessment**: pathmc is not competing with Bambi on single-equation flexibility. Its comparative advantage is the *system*: multiple equations, causal graph, and interventional simulation. The gaps that matter most are those that limit the range of *systems* the DSL can express — interaction terms and custom priors are the highest-priority additions.

## Documentation coverage gaps

### Families without examples

Three of five supported families have no dedicated example notebook:

1. **Poisson** — needed for count outcomes
2. **NegBinomial** — needed for overdispersed counts
3. **StudentT** — needed for heavy-tailed data

### DSL features without examples

1. **Intercept suppression (`0 +`)** — documented but not demonstrated
2. **Interaction terms** via pre-computed columns — not documented at all
3. **Custom transforms** via `register_transform()` — API exists, no example
4. **`pooling="full"` or `pooling="none"`** — not tested or documented

### Undertested feature combinations

1. Multi-family mediator chains (Bernoulli → Bernoulli)
2. `prob()` with non-Gaussian families
3. `standardized()` with non-Gaussian families
4. NegBinomial `predict()`

## Recommendations: priority-ordered feature roadmap

### Tier 1: Close before stealth exit (high impact, low-medium complexity)

These features would most impress PyMC core developers evaluating pathmc as a "causal orchestration layer":

| # | Feature | Type | Complexity | Impact |
|---|---------|------|-----------|--------|
| 1 | Interaction terms in DSL (`X:Z`) | Feature + Docs | Medium | High — enables moderation, CATE |
| 2 | Custom priors API | Feature + Docs | Low-Medium | High — fundamental Bayesian feature |
| 3 | Count outcomes example (Poisson/NegBin) | Docs | Low | High — demonstrates breadth |
| 4 | StudentT example | Docs | Low | Medium — demonstrates robustness |
| 5 | Custom transform example | Docs | Low | High — shows extensibility |
| 6 | Interaction/moderation example | Docs | Low | High — common analysis pattern |

### Tier 2: Strengthen the system (medium complexity)

| # | Feature | Type | Complexity | Impact |
|---|---------|------|-----------|--------|
| 7 | Higher-order lags (`lag(x, k=2)`) | Feature + Docs | Medium | Medium — time series users |
| 8 | Front-door identification helper | Feature | Medium | Medium — automated `frontdoor_identifiable()` |
| 9 | Cross-sectional latent mediator example | Docs | Low | Medium — broadens latent usage |
| 10 | Larger residual covariance block example | Docs | Low | Low — niche but demonstrates depth |

### Tier 3: Future vision (high complexity, defer)

| # | Feature | Complexity | Notes |
|---|---------|-----------|-------|
| 11 | Spline / smooth terms | High | Consider via transform registry |
| 12 | Measurement models (`=~`) | Very High | Different architecture; semopy's niche |
| 13 | Multi-family `~~` (copula) | High | Architecturally complex |
| 14 | Distributional models | High | Varying σ across observations |

## Summary for PyMC core developer audience

**Pitch**: pathmc is not trying to replace PyMC or Bambi for single-equation modeling. Its value proposition is:

1. **System-level specification**: Multiple structural equations compiled as a single generative model
2. **Native causal operations**: `pm.do()` graph surgery propagates through the full DAG
3. **Domain-specific transforms**: Extensible transform registry with estimable parameters (adstock, saturation, custom)
4. **Identification-first workflow**: Check your assumptions before you estimate
5. **Panel + temporal**: Time-forward simulation with hierarchical pooling

**What could be upstreamed to PyMC**: The generative model + `pm.observe()` + `pm.do()` pattern for causal inference. The transform registry concept. The scan-based panel compilation strategy.

**Current breadth**: The DSL handles the most common 80% of applied structural causal models. The missing 20% (interactions, custom priors, higher-order lags, CFA) represents the difference between "useful tool" and "comprehensive platform."

**Recommendation**: Close the Tier 1 gaps before going public. The Tier 2 items can be positioned as a near-term roadmap. Tier 3 items demonstrate architectural foresight but are not blockers.

## Follow-up issues

Based on this audit, the following issues have been created:

1. **#50 — Feature: Interaction terms in DSL** — Add `X:Z` syntax for interactions; recompute under `do()`
2. **#52 — Feature: Custom priors API** — Accept user-specified priors via `priors=` kwarg in `fit()`
3. **#53 — Docs: Count outcome example (Poisson/NegBinomial)** — New example notebook
4. **#54 — Docs: StudentT heavy-tailed example** — New or extended example
5. **#55 — Docs: Custom transform tutorial** — Show `register_transform()` end-to-end
6. **#56 — Docs: Moderation / interaction example** — Document pre-computed interaction pattern
7. **#57 — Feature: Higher-order lags** — Extend `lag(x, k=2)` syntax
8. **#58 — Docs: Cross-sectional latent mediator example** — Non-MMM latent mediator
