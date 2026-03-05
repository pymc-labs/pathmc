# Causal Inference Audit Report

**Issue**: #34 — Audit this repo from a causal inference perspective

## Executive summary

pathmc is deeply and deliberately grounded in **Pearl's Structural Causal Model (SCM) framework**. The codebase faithfully implements DAGs, structural equations, the do-operator (graph surgery), backdoor identification, and d-separation — the core primitives of the SCM approach. The documentation clearly distinguishes interventions (Pearl's Ladder rung 2) from unit-level counterfactuals (rung 3), and a dedicated notebook walks through the three-step abduction procedure.

The package is **not too narrow**. Its Pearlian foundation is the most general and rigorous framework for causal reasoning with structural models, and the SEM/lavaan heritage gives it natural ergonomics for applied analysts. The ATE/CATE terminology used by pathmc is recognized across both the SCM and potential outcomes traditions, so results are intelligible to practitioners from either background.

There are, however, several areas where the causal framing can be strengthened — primarily around **explicit estimand language**, **assumption documentation**, **cross-framework bridge notes**, and **identification gaps**. These are surfaced as actionable follow-up issues below.

## Framework alignment analysis

### Pearl's SCM framework — primary alignment (strong)

pathmc is a Pearlian tool. The alignment is deep and consistent across all layers:

| SCM concept | pathmc implementation | Quality |
|---|---|---|
| Structural equations | `Spec` AST with `Regression` nodes; `compile_to_pymc()` emits symbolic parent wiring | Excellent |
| DAG | `GraphInfo` via NetworkX; topological order, exogenous/endogenous classification | Excellent |
| do-operator (graph surgery) | `pm.do()` on generative model; severs incoming edges, fixes value | Excellent |
| Backdoor criterion | `identify.py`: `adjustment_sets()`, `is_identifiable()`, d-separation | Good |
| Collider bias detection | `collider_warnings()` flags conditioning on colliders | Good |
| Exogenous / endogenous | `GraphInfo.exogenous`, `GraphInfo.endogenous` with clear semantics | Excellent |
| L2 vs L3 distinction | `do()` = interventional; docs explain counterfactual requires abduction | Excellent |
| Three-step counterfactual | `docs/examples/counterfactual.qmd`: abduction, action, prediction | Excellent |
| Residual covariance (`~~`) | LKJ + MvNormal blocks for correlated residuals | Good |
| Front-door criterion | Concept demonstrated in `docs/examples/front_door.qmd` via mediation | Good (conceptual, not automated) |

**Assessment**: pathmc's Pearlian alignment is its core strength. The do-operator implementation via `pm.do()` is architecturally clean and semantically faithful. The documentation is unusually clear about what `do()` does and does not claim.

### Potential outcomes / Rubin Causal Model — partial alignment

pathmc uses PO-compatible terminology (ATE, CATE, treatment, outcome) but does not adopt the PO framework itself:

| PO concept | pathmc status | Notes |
|---|---|---|
| ATE estimand | `ate()` method computes `do(hi) - do(lo)` | Correct via g-computation |
| CATE estimand | `cate()` method with `condition=` | Correct via conditional do-operator |
| ATT/ATU | Not implemented | Mentioned once for DiD in PRD |
| Y(0), Y(1) notation | Not used | Uses Pearl's `P(Y \| do(X=x))` instead |
| SUTVA / no interference | Not discussed | Implicitly assumed but undocumented |
| Consistency assumption | Not discussed | Implicitly assumed |
| Propensity scores / IPW | Not applicable | Different estimation strategy |

**Assessment**: pathmc's g-computation approach to ATE/CATE is theoretically equivalent to the PO-framework's standardization estimator under the same identification assumptions. This equivalence is not documented, which is a missed opportunity for cross-framework communication. The lack of explicit SUTVA/consistency discussion is a gap, as these assumptions are relevant regardless of framework.

### SEM / lavaan tradition — strong alignment

| SEM concept | pathmc implementation | Quality |
|---|---|---|
| Formula DSL (`~`, `~~`, `:=`) | `parse.py` with lavaan-inspired syntax | Excellent |
| Labeled coefficients | `Term.label` parsed from `Y ~ a*X` syntax | Excellent |
| Defined parameters | `:=` expressions evaluated over posterior draws | Excellent |
| Residual covariance | `~~` operator with LKJ + MvNormal blocks | Good |
| stdyx standardization | `build_standardized_effects()` | Good |
| Path effects | `compute_path_effect("X -> M -> Y")` | Good |
| Indirect effects | `indirect := a*b` evaluated draw-wise | Excellent |

**Assessment**: The SEM heritage is well-executed. The Bayesian twist (full posterior distributions for defined parameters, no delta method needed) is a genuine advantage over classical SEM tools.

## Terminology audit

### Terms used correctly

| Term | Usage | Framework origin | Cross-framework? |
|---|---|---|---|
| do-operator | `do()` method, graph surgery semantics | Pearl | Pearl-specific |
| DAG | Throughout; graph layer, docs, examples | Both | Yes |
| Backdoor criterion | `identify.py`, concept docs | Pearl | Pearl-specific |
| Collider | `collider_warnings()`, examples | Both | Yes |
| Confounder / confounding | Docs, examples | Both | Yes |
| Mediator / mediation | Examples, PRD | Both | Yes |
| ATE | `ate()` method | Both | Yes |
| CATE | `cate()` method | Both | Yes |
| Treatment / outcome | API parameter names | Both | Yes |
| Intervention | Docs, docstrings | Pearl | Yes |
| Counterfactual | Correctly distinguished from intervention in docs | Pearl | Partially (term used differently in PO) |
| Structural equation | Spec, compile, docs | SEM / Pearl | Yes |
| Graph surgery | Docs, how-it-works | Pearl | Pearl-specific |

### Terms absent but potentially useful

| Term | Why it matters | Recommendation |
|---|---|---|
| **Estimand** | Central to modern causal practice across all frameworks. Currently appears only 3 times in the entire repo, always in passing. | Add a conceptual section defining estimand → estimator → estimate; frame `ate()`, `cate()`, `prob()` as estimands |
| **Consistency / SUTVA** | Necessary assumptions for causal interpretation of `do()` results, even in the SCM framework | Add to causal assumptions documentation |
| **Positivity** | Mentioned once in causal_inference.qmd but not checked at runtime | Document more prominently; consider runtime warnings for out-of-support interventions |
| **G-computation** | pathmc's `do()` is literally g-computation; naming it connects to a huge literature | Mention in docs that `do()` implements g-computation / the truncated factorization formula |
| **Adjustment formula** | The mathematical identity that backdoor adjustment computes; connects `adjustment_sets()` to `do()` | Could strengthen the identification docs |

### Terms deliberately avoided (correctly)

| Term | Why it's absent | Assessment |
|---|---|---|
| Propensity score | pathmc uses structural equations, not weighting | Correct — different estimation strategy |
| IPW / matching | Not the approach | Correct |
| Causal discovery | pathmc takes DAGs as input | Correct — clearly out of scope |
| Full do-calculus | Only backdoor criterion implemented | Honest; acknowledged in comparison page |

## Strengths

1. **Clean Pearlian architecture.** The generative model + `pm.observe()` + `pm.do()` pattern is a faithful implementation of Pearl's do-operator. The separation between the generative model (for interventions) and the estimation model (for fitting) is architecturally sound and causally principled.

2. **Honest about scope.** The docs, comparison page, and causal assumptions section all clearly state what pathmc can and cannot do. The "pathmc computes; you identify" callout is exemplary.

3. **Correct L2/L3 distinction.** The counterfactual notebook is among the clearest explanations of the intervention vs. counterfactual distinction in any Python package's documentation.

4. **Cross-framework output.** The `ate()` and `cate()` methods produce quantities that are recognized across frameworks, so results are interpretable by practitioners from any tradition.

5. **Rich causal examples.** The documentation includes seeing-vs-doing, causal identification, front-door, collider bias, and counterfactual examples — a strong pedagogical suite.

6. **Identification before estimation.** The `adjustment_sets()`, `is_identifiable()`, and `collider_warnings()` methods encourage the right workflow: check identification before fitting.

## Gaps and recommendations

### Gap 1: No explicit estimand → estimator → estimate framework

**Problem**: The estimand-estimator-estimate pipeline is the standard conceptual framework across all schools of causal inference. pathmc has estimands (`ate()`, `cate()`, `prob()`) and estimates (the `DoResult` values), but the mapping is not made explicit. A user from the PO tradition or from the DoWhy workflow would expect this language.

**Recommendation**: Add a short section to the causal inference concept page defining the three terms and mapping pathmc's API to them:
- **Estimand**: "What is the ATE of X on Y?" → defined by the DAG and the target query
- **Estimator**: g-computation via the do-operator on the structural model
- **Estimate**: the posterior distribution returned by `ate()` / `do()`

### Gap 2: Causal assumptions not fully enumerated

**Problem**: The causal assumptions page in `docs/concepts/causal_inference.qmd` lists four assumptions (correct DAG, no unmeasured confounders, no model misspecification, positivity) but omits two important ones:
- **Consistency** (well-defined interventions): if a unit actually receives treatment value x, their outcome equals the potential outcome Y(x). This rules out "multiple versions of treatment."
- **No interference / SUTVA**: one unit's treatment does not affect another unit's outcome. This is particularly relevant for panel models where units may interact.

**Recommendation**: Add consistency and no-interference to the documented assumptions. These hold implicitly in pathmc's structural equations but should be stated for completeness and to help users reason about when they are violated.

### Gap 3: G-computation not named

**Problem**: pathmc's `do()` operator is an implementation of g-computation (Robins, 1986) — the procedure of standardizing over the structural model to compute interventional distributions. This term connects pathmc to a large literature in epidemiology and causal inference, but the word "g-computation" appears only once in the entire repo (in the PRD, as "g-computation / interventional simulation").

**Recommendation**: Mention in the causal inference concept page and how-it-works page that pathmc's `do()` implements g-computation: fit the structural model, then forward-simulate under the intervention. This bridges the Pearl and Robins literatures and helps epidemiologists and PO practitioners locate pathmc in their conceptual map.

### Gap 4: No cross-framework bridge notes

**Problem**: The comparison page positions pathmc relative to other _packages_ but does not position it relative to other _frameworks_. A reader who thinks in potential outcomes (Y(0), Y(1)) may not immediately see how pathmc's `do()` relates to their estimands.

**Recommendation**: Add a brief conceptual note — either in the causal inference concept page or as a sidebar — explaining the equivalence:
- `E[Y | do(X=1)] - E[Y | do(X=0)]` (Pearl) = `E[Y(1) - Y(0)]` (PO) under the same identification assumptions
- pathmc's g-computation is the structural analog of the standardization / G-formula in the PO tradition
- The backdoor criterion in Pearl corresponds to conditional ignorability / unconfoundedness in PO

This is a documentation-only change, not a code change.

### Gap 5: Identification limited to backdoor criterion

**Problem**: `identify.py` only implements the backdoor criterion. The front-door criterion is demonstrated conceptually in a notebook but not automated. No instrumental variable or general do-calculus identification is available.

**Recommendation**: This is already acknowledged in the comparison page and the roadmap. The front-door notebook is a good interim measure. For a next step:
- Add `frontdoor_identifiable(treatment, mediator, outcome)` to `identify.py` — a targeted check for the three front-door conditions
- Longer term: consider integrating with DoWhy's identification engine for more complex DAGs

### Gap 6: No sensitivity analysis for unmeasured confounding

**Problem**: Every causal claim from pathmc rests on the assumption that the DAG is correct and all confounders are measured. Sensitivity analysis quantifies how strong an unmeasured confounder would need to be to overturn the conclusion. This is critical for credibility.

**Recommendation**: Already in the post-v1 roadmap. Priority should be high given pathmc's positioning as a tool for causal reasoning. Even a simple parametric sensitivity approach (e.g., "add an unobserved confounder with effect size γ on treatment and δ on outcome; how does the ATE change?") would be valuable.

### Gap 7: No runtime positivity checks

**Problem**: Positivity (overlap) is mentioned in the assumptions but not checked. When a user calls `do(set={"X": 100})` and the observed data has max(X) = 5, the result is pure extrapolation. No warning is issued.

**Recommendation**: Add an optional positivity warning when intervention values fall outside the observed data range. This could be a simple check in `PathModel.do()`:
```
if val < data[var].min() or val > data[var].max():
    warnings.warn(f"Intervention value {val} for '{var}' is outside the observed data range [{data[var].min()}, {data[var].max()}]. Results are extrapolations.")
```

## Assessment: is pathmc too narrow or too opinionated?

**No.** pathmc occupies a well-defined niche — Bayesian structural causal models with interventional simulation — and does it well. The Pearlian foundation is the right choice for a package that specifies DAGs and structural equations: Pearl's framework was literally designed for this setting.

The package is **appropriately opinionated** in the following ways:
- **Requiring a DAG**: Correct. Causal inference requires causal assumptions. Making them explicit (via a DAG) is a feature, not a limitation.
- **Linear structural equations**: A reasonable starting point. Transforms (adstock, saturation) extend the range. Fully nonlinear structural equations are a natural next step but not a v1 requirement.
- **Bayesian estimation**: Gives full posterior uncertainty over causal effects. This is strictly more informative than point estimates.
- **g-computation as the estimation strategy**: Natural for structural models. Other strategies (IPW, AIPW, DML) target different settings (treatment effect estimation without structural models) and are out of scope.

The areas where pathmc could be **less narrow** are primarily documentation and terminology, not architecture:
- Name g-computation explicitly to connect to epidemiology literature
- Add bridge notes for PO practitioners
- Strengthen assumption documentation
- Expand identification helpers (front-door, eventually IV)

## Follow-up issues

Based on this audit, the following specific issues are recommended:

1. **Docs: Add estimand → estimator → estimate framework to causal inference concept page** — Frame `ate()`, `cate()`, and `prob()` as estimands; name g-computation as the estimator; clarify that `DoResult` contains the estimate.

2. **Docs: Add consistency and no-interference to causal assumptions** — These assumptions hold implicitly but should be stated and explained, especially for panel models where interference between units is plausible.

3. **Docs: Add cross-framework bridge notes** — A sidebar or section explaining the equivalence between `do()` / Pearl notation and potential outcomes / PO notation, plus the connection between backdoor criterion and conditional ignorability.

4. **Docs: Name g-computation explicitly** — In the causal inference concept page and how-it-works page, note that pathmc's `do()` implements Robins' g-computation / the truncated factorization formula.

5. **Feature: Add positivity warnings to `do()`** — Warn when intervention values fall outside the observed data range.

6. **Feature: Front-door identification helper** — Add `frontdoor_identifiable()` to `identify.py` to automate checking the three front-door conditions.

7. **Feature: Unmeasured confounding sensitivity analysis** — Already in roadmap; this audit elevates its priority. Even a simple parametric approach would significantly strengthen causal credibility.

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N.P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Pearl, J. & Mackenzie, D. (2018). *The Book of Why*. Basic Books.
- Robins, J.M. (1986). A new approach to causal inference in mortality studies with sustained exposure periods. *Mathematical Modelling*, 7, 1393–1512.
- Rubin, D.B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. *Journal of Educational Psychology*, 66(5), 688–701.
- Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Imbens, G.W. & Rubin, D.B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
