# Workshop Fit Assessment: pathmc × Causal Inference Workshop

> Assessment of where `pathmc` adds unique value, where it needs feature development, and where other packages (Bambi, CausalPy) are adequate — to inform feature issue creation.

## Workshop Overview

The [causal-inference-workshop](~/git/causal-inference-workshop/) is an 8-session (16-hour) live course titled *Applied Causal Inference for Business Impact*. It teaches causal inference through business use cases using Bambi, CausalPy, and PyMC. pathmc is not currently referenced in any session materials.

| Session | Title | Primary Tool | Core Methods |
|---------|-------|-------------|--------------|
| 1 | Causal Decision-Making | Bambi | DAGs, backdoor criterion, regression adjustment |
| 2 | Marketing Attribution | Bambi | Logistic regression, G-computation, geo-experiments, MMM intro |
| 3 | Price Elasticity | CausalPy | Instrumental variables (2SLS), revenue optimization |
| 4 | Customer Churn | Bambi | Survival analysis (Weibull), hazard ratios, immortal time bias |
| 5 | Product Impact | Bambi | Bayesian A/B tests, hierarchical models, network effects, switchback |
| 6 | Policy Evaluation | CausalPy | Difference-in-differences, synthetic control, interrupted time series |
| 7 | Personalization | Bambi | CATE, interaction terms, splines, S-learner / T-learner |
| 8 | Communication | None (conceptual) | Sensitivity analysis, E-values, triangulation, pitfalls |

---

## Assessment Categories

### A. Natural Fit — pathmc adds unique value today
### B. Good Fit — requires feature development
### C. Not Suitable — other packages are adequate

---

## A. Natural Fit: pathmc adds unique value today

These topics are where pathmc's DAG-first design, `do()` operator, and structural model approach provide pedagogical and ergonomic advantages over Bambi or CausalPy.

### A1. DAG-First Causal Reasoning (Session 1, used throughout)

**Workshop need:** Session 1 teaches participants to draw DAGs, identify confounders, mediators, and colliders, and apply the backdoor criterion. DAGs are referenced in every subsequent session.

**pathmc advantage:** No other single package lets you specify the DAG *and* fit it *and* query identification from the same object.

```python
model = pathmc.model("Y ~ X + Z", data=df)
model.graph()                          # visualise the DAG
model.adjustment_sets("X", "Y")        # backdoor criterion
model.is_identifiable("X", "Y")        # identifiability check
model.collider_warnings({"C"}, "X", "Y")  # collider bias warning
```

**Existing pathmc docs:** `causal_identification.qmd`, `collider_bias.qmd`, `simpsons_paradox.qmd`, `dag_testing.qmd`

**Compared to Bambi:** Bambi fits a regression but provides no graph introspection, no `adjustment_sets()`, no `collider_warnings()`. The DAG is implicit in the formula — pathmc makes it explicit. This is the core pedagogical win: *the DAG is the model*.

**Workshop integration:** Replace or supplement the Bambi regression adjustment exercise in Session 1 with pathmc, which makes the DAG-to-model connection visible. Students would draw a DAG and then *type it directly into pathmc*, rather than mentally translating from DAG to a Bambi formula.

**Feature gaps:** None — this works today.

---

### A2. Mediation Analysis with Labeled Effects (Sessions 1, 2, 7)

**Workshop need:** Mediation appears implicitly throughout the workshop (marketing → brand awareness → sales, treatment → engagement → retention) but is never formally taught as a method. The workshop currently uses Bambi for single-equation regression adjustment.

**pathmc advantage:** Mediation is pathmc's flagship workflow. Labeled coefficients, `:=` defined parameters, and `effect()` give ergonomic direct/indirect effect decomposition.

```python
spec = """
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""
model = pathmc.model(spec, data=df)
model.effects_summary()           # posterior summaries for a, b, c, indirect
model.effect("X -> M -> Y")       # path-traced indirect effect
model.ate("Y", "X", values=(0, 1))  # total effect via g-computation
```

**Existing pathmc docs:** `mediation.qmd`, `correlated_residuals.qmd`, `mmm_mediation.qmd`

**Compared to Bambi:** Bambi requires fitting two separate models and manually multiplying posterior draws for the indirect effect. pathmc encodes the entire structural model in one spec and computes `indirect := a*b` automatically.

**Workshop integration:** Could add a mediation mini-module in Session 1 or Session 2 (marketing funnel mediation). The `mmm_mediation.qmd` example (brand → search → sales) is directly relevant to Session 2's marketing attribution topic.

**Feature gaps:** None — this works today.

---

### A3. G-Computation via do() (Sessions 1, 2, 7)

**Workshop need:** Sessions 2 and 7 teach G-computation for ATE and CATE. The current approach uses Bambi's `predict()` with manual counterfactual datasets — creating two copies of the data (everyone treated / everyone untreated), predicting from both, and taking the difference. This is tedious and error-prone (6-10 lines per G-computation call).

**pathmc advantage:** `do()`, `ate()`, `cate()` are single-call operations.

```python
# Bambi G-computation (current workshop approach)
data_t = data.copy(); data_t['treatment'] = 1
data_c = data.copy(); data_c['treatment'] = 0
pred_t = model.predict(idata, data=data_t, kind='mean')
pred_c = model.predict(idata, data=data_c, kind='mean')
ate = (pred_t.posterior['Y_mean'] - pred_c.posterior['Y_mean']).mean(dim='obs')

# pathmc equivalent
model.ate("Y", "treatment", values=(0, 1))
```

**Existing pathmc docs:** `do_queries.qmd`, `ate_estimation.qmd`, `seeing_vs_doing.qmd`, `moderation.qmd`

**Compared to Bambi:** pathmc's `do()` also handles multi-step propagation through the DAG (topological order), which Bambi's `predict()` cannot do — Bambi applies the intervention to one equation, not to a system. For multi-equation models, pathmc propagates interventions correctly through all downstream nodes.

**Workshop integration:** Session 2's marginal effects section and Session 7's CATE computation could be dramatically simplified with pathmc. The `ate_estimation.qmd` example directly addresses the workshop's ATE-vs-coefficient discussion for linear and logistic models.

**Feature gaps:** None — this works today. However, `ate()` and `cate()` could benefit from better default visualizations (see B5).

---

### A4. Collider Bias and Simpson's Paradox (Session 1)

**Workshop need:** Session 1 teaches four DAG patterns including collider bias and confounding (Simpson's paradox is an instance of confounding where the sign reverses). These are taught conceptually with DAG drawings.

**pathmc advantage:** pathmc can *demonstrate* collider bias and Simpson's paradox computationally, not just draw them.

```python
# Collider: conditioning on C opens a spurious path
model.collider_warnings({"C"}, "X", "Y")
# → "Warning: C is a collider on the path X → C ← Y. Conditioning on C
#    opens a non-causal path between X and Y."

# Simpson's paradox: show that naive vs. adjusted estimates differ
model.adjustment_sets("X", "Y")
# → {"Z"} — must adjust for Z to get the correct effect direction
```

**Existing pathmc docs:** `collider_bias.qmd` (birth-weight paradox), `simpsons_paradox.qmd`

**Workshop integration:** These examples could replace or supplement the conceptual DAG pattern discussion in Session 1 Part 3. Students would see the paradox *numerically* — fitting the naive model yields a positive effect, adjusting for the confounder reverses it.

**Feature gaps:** None — this works today.

---

### A5. DAG Testing / Implied Independences (Session 8, could be Session 1)

**Workshop need:** Session 8 discusses robustness and sensitivity. The workshop doesn't currently teach DAG testing — checking whether the DAG's conditional independence implications are consistent with the data.

**pathmc advantage:** `implied_independences()` and `test_implications()` allow empirical DAG validation.

```python
model.implied_independences()  # list conditional independences implied by DAG
model.test_implications()       # test each against data
```

**Existing pathmc docs:** `dag_testing.qmd`

**Workshop integration:** This could be a powerful addition to Session 1 (after drawing the DAG, *test it*) or Session 8 (as a robustness check). No other tool in the workshop stack provides this.

**Feature gaps:** None — this works today.

---

### A6. Sensitivity Analysis for Unmeasured Confounding (Session 8)

**Workshop need:** Session 8 teaches sensitivity analysis conceptually (Rosenbaum bounds, E-values). The current plan is to compute E-values manually from posterior samples.

**pathmc advantage:** pathmc has a `sensitivity()` method that automates this.

**Existing pathmc docs:** `sensitivity.qmd`

**Workshop integration:** Could replace the manual E-value computation in Session 8.

**Feature gaps:** Unclear how comprehensive the current implementation is — need to verify it covers the tipping-point analysis and contour plots described in the workshop plan.

---

### A7. MMM with Structural Causal Model (Session 2)

**Workshop need:** Session 2 introduces MMM conceptually (adstock, saturation, channel contributions). The current plan shows PyMC-Marketing as a demo only.

**pathmc advantage:** pathmc can fit a *structural* MMM where the DAG shows *why* channels work, not just that they work. The mediation structure (spend → brand awareness → sales) is a natural pathmc model with adstock and saturation transforms.

```python
spec = """
awareness ~ lag(awareness) + adstock(tv_spend, decay=theta_tv)
sales ~ a*awareness + b*adstock(digital_spend, decay=theta_d)
indirect_tv := a * 1  # TV effect mediated through awareness
"""
model = pathmc.model(spec, data=df, panel={"unit": "region", "time": "week"})
model.do(set={"tv_spend": 150}, simulate_over="time")
```

**Existing pathmc docs:** `mmm_basic.qmd`, `mmm_mediation.qmd`, `mmm_geo.qmd`, `mmm_awareness.qmd`

**Compared to PyMC-Marketing:** PyMC-Marketing is a black-box MMM (`DelayedSaturatedMMM`). pathmc lets you specify the causal structure explicitly, which is pedagogically superior for a causal inference course. It also supports counterfactual simulation via `do()`, which PyMC-Marketing does through a different (budget optimization) interface.

**Workshop integration:** pathmc could replace or supplement the PyMC-Marketing demo in Session 2 with a structural MMM that students build from a DAG. This ties the MMM directly to the causal reasoning taught in Session 1.

**Feature gaps:** None — this works today. However, the workshop's MMM section is deliberately brief (demo only), so the integration would be modest unless an extended MMM module is added.

---

### A8. Panel DiD (Session 6, partial)

**Workshop need:** Session 6 teaches DiD as the workhorse quasi-experimental method, currently implemented in CausalPy (and manual Bambi as alternative).

**pathmc advantage:** pathmc has a `did.qmd` example using panel mode with `treat_post` interaction.

**Existing pathmc docs:** `did.qmd`

**Compared to CausalPy:** CausalPy provides automatic counterfactual plots, placebo tests, and a purpose-built DiD API. pathmc's DiD is a panel regression model — more manual but structurally transparent (students see the full DAG).

**Workshop integration:** pathmc could be shown as the "manual Bambi alternative" in Session 6, replacing the `bmb.Model('revenue ~ treated + post + treated_post', data)` code with a structurally transparent pathmc spec. The pathmc version makes the parallel trends assumption visible in the DAG. However, CausalPy's built-in diagnostics (parallel trends plots, event study plots) remain superior for practical use.

**Feature gaps:** See B1 (parallel trends testing) and B2 (counterfactual visualization).

---

## B. Good Fit — Requires Feature Development

These topics are natural extensions of pathmc's capabilities but need new features before they can serve the workshop.

### B1. Parallel Trends Testing for DiD

**Workshop need:** Session 6 emphasizes testing the parallel trends assumption visually and via placebo DiD. This is the most important assumption check for any DiD analysis.

**Current gap:** pathmc's `did.qmd` example fits a DiD model but provides no built-in parallel trends diagnostic.

**Proposed feature:**
- `model.parallel_trends_test(treatment_var, time_var, pre_period)` — fit a placebo DiD in the pre-period and report whether the "effect" is zero
- `model.plot_trends(treatment_var, time_var)` — visualize pre/post trends by group

**Value:** Would make pathmc competitive with CausalPy for DiD, while retaining the DAG-first advantage.

**Priority:** Medium — CausalPy already handles this well; pathmc's edge is the structural model, not the diagnostics.

---

### B2. Counterfactual Visualization for Panel do()

**Workshop need:** Sessions 2 and 6 produce counterfactual time-series plots: "what would have happened without the intervention?" CausalPy generates these automatically.

**Current gap:** pathmc's `do(simulate_over="time")` returns a `DoResult` with numerical summaries, but has no built-in counterfactual time-series plot.

**Proposed feature:**
- `do_result.plot(var, vs="observed")` — overlay observed vs. counterfactual trajectories with HDI bands
- `do_result.plot_contrast(var)` — plot the difference over time with HDI

**Value:** Critical for making pathmc's panel `do()` results accessible and comparable to CausalPy's output.

**Priority:** High — visualization is essential for the "last mile" in communicating results (Session 8's point).

---

### B3. Event Study / Dynamic Treatment Effects

**Workshop need:** Session 6 covers event studies — estimating time-period-specific treatment effects for DiD models. This shows that the effect appears only post-treatment and checks for pre-trends.

**Current gap:** pathmc has no event study functionality.

**Proposed feature:**
- `model.event_study(treatment_var, time_var, reference_period)` — estimate and plot period-specific treatment effects relative to a reference period
- Pre-treatment coefficients should be near zero (visual parallel trends check)

**Value:** Combines pathmc's panel mode with a standard DiD diagnostic. Provides both the parallel trends check and the dynamic effect trajectory in one call.

**Priority:** Medium — useful but CausalPy covers this.

---

### B4. Instrumental Variables via DAG

**Workshop need:** Session 3 is entirely about instrumental variables for price elasticity. This is the workshop's hardest session and uses CausalPy.

**Current gap:** pathmc has no IV support. However, the DAG already encodes the IV structure: an instrument Z that affects treatment T but not outcome Y directly.

**Proposed feature:** Two options:

**Option A — DAG-informed IV detection:**
```python
spec = """
price ~ cost_shock + demand_shock
quantity ~ price + demand_shock
"""
model = pathmc.model(spec, data=df)
model.instruments("price", "quantity")
# → {"cost_shock"} — cost_shock is a valid instrument for price→quantity
```

The identification module already computes adjustment sets; computing valid instruments is a graph-theoretic extension (find variables that satisfy the IV conditions in the DAG).

**Option B — Full IV compilation:**
The compiler would detect IV structure in the DAG and compile to a two-stage model internally. This is more ambitious but would be unique: no other package lets you specify a DAG and automatically compiles to IV when standard adjustment is insufficient.

**Value:** Would make pathmc the first package to unify DAG specification, identification diagnosis, *and* IV estimation. For the workshop, it would replace CausalPy in Session 3 while maintaining the DAG-first philosophy from Session 1.

**Priority:** High potential impact, but high development effort. Option A (detection only) is lower effort and still valuable.

---

### B5. ATE/CATE Visualization Helpers

**Workshop need:** Sessions 1, 2, and 7 visualize treatment effects extensively: `az.plot_posterior` with `ref_val=0`, CATE distributions, treatment effect forest plots by subgroup.

**Current gap:** pathmc's `DoResult` provides `.mean()` and `.hdi()` but no built-in plotting.

**Proposed features:**
- `do_result.plot(var)` — posterior distribution of the causal effect with HDI and reference line
- `model.plot_ate(outcome, treatment, values)` — posterior plot for the ATE
- `model.plot_cate(outcome, treatment, values, by=var)` — CATE by subgroup as forest plot
- `model.plot_effects()` — forest plot of all labeled coefficients and defined parameters

**Value:** Reduces friction between estimation and communication. Currently users must extract draws and use ArviZ/matplotlib manually.

**Priority:** Medium-high — ergonomic improvement that would make pathmc more workshop-friendly.

---

### B6. Overlap / Positivity Diagnostics

**Workshop need:** Session 1 checks the positivity assumption by plotting covariate distributions by treatment group. Session 8 emphasizes assumption checking.

**Current gap:** pathmc provides no overlap diagnostics.

**Proposed feature:**
- `model.overlap_check(treatment_var)` — for each treatment group, plot the distribution of confounders (or propensity scores)
- Could also flag extreme propensity scores or lack of common support

**Value:** Complements the identification helpers (which check the graph) with data-level diagnostics (which check whether the data supports identification).

**Priority:** Low-medium — helpful but not unique to pathmc's mission.

---

### B7. Controlled Direct Effect (CDE) Helper

**Workshop need:** Session 7's CATE estimation implicitly involves controlled direct effects when mediators are present. The `prd_v1.md` roadmap already identifies CDE as a future convenience feature.

**Current gap:** Users must manually set up two `do()` calls — one fixing the mediator, one not.

**Proposed feature:**
```python
model.cde("Y", "X", mediator="M", mediator_value=None)  # fix M at mean
model.proportion_mediated("Y", "X", mediator="M")
```

**Value:** Automates the mediation decomposition for non-linear models (where `indirect := a*b` is only approximate).

**Priority:** Medium — identified in roadmap, directly useful for the workshop's mixed-family mediation scenarios.

---

### B8. Single-Equation Convenience Mode

**Workshop need:** Sessions 1 and 5 use simple single-equation models (`Y ~ T + X1 + X2`). pathmc currently requires this to be specified as a structural equation, which works but feels heavy for one-equation models.

**Current gap:** For a single regression, pathmc and Bambi produce identical results, but Bambi is simpler (auto priors, no graph overhead). Students may wonder "why not just use Bambi?"

**Proposed feature / positioning:** Rather than a new feature, this is about documentation and messaging:
- Show that pathmc handles single-equation models gracefully
- Emphasize the value-add: even for one equation, you get `adjustment_sets()`, `collider_warnings()`, and `do()`
- Provide a "getting started" example that begins with one equation and progressively adds structure

**Priority:** Low (documentation, not code).

---

## C. Not Suitable: Other Packages Are Adequate

These workshop topics do not align with pathmc's design or are well-served by existing tools.

### C1. Instrumental Variables — 2SLS Implementation (Session 3)

CausalPy provides built-in Bayesian IV with formula interface, first-stage diagnostics, and F-statistics. Unless pathmc develops DAG-informed IV compilation (see B4), CausalPy is the right tool.

### C2. Survival Analysis (Session 4)

Bambi's survival family (`weibull`, `exponential`) with `cens()` syntax handles Weibull regression with confounders. Survival analysis requires fundamentally different likelihoods (censored observations) that don't fit pathmc's current compiler architecture. Adding survival families would be a large, low-ROI effort given that Bambi already works.

### C3. Simple Bayesian A/B Testing (Session 5)

A two-group comparison (`Y ~ group`) is a single-equation problem where Bambi is the simplest tool. pathmc adds no value here — no DAG structure, no mediation, no system of equations.

### C4. Network Effects / Interference (Session 5)

Network effects and SUTVA violations require modeling interference patterns (spillover terms, cluster randomization). This is a specialized problem outside pathmc's DAG-based path analysis framework.

### C5. Switchback Designs (Session 5)

Switchback experiments are time-based randomization with carryover effects. While pathmc has panel mode and `lag()`, switchback analysis is better served by Bambi's random effects (`(1 | market_id) + (1 | week)`). The DAG structure adds little.

### C6. Synthetic Control (Session 6)

CausalPy is purpose-built for synthetic control (weight optimization, placebo tests, automatic plots). Synthetic control is not a structural model — it's a prediction problem (what would the treated unit have looked like?). Not a fit for pathmc's DAG framework.

### C7. Interrupted Time Series (Session 6)

CausalPy handles ITS with counterfactual extrapolation and automatic visualization. A single time series without a structural system of equations doesn't benefit from pathmc.

### C8. BART / GP for Flexible CATE (Session 7)

PyMC-BART and custom GP models are specialized non-parametric methods. pathmc's framework is parametric (linear equations with optional transforms). Non-parametric CATE estimation is outside pathmc's scope.

### C9. Sequential Testing / Power Analysis (Session 5)

These are tool-agnostic Bayesian concepts that use posterior updating. No package advantage.

---

## Summary Matrix

| Session | Topic | Fit | pathmc Role | Alternative |
|---------|-------|-----|-------------|-------------|
| 1 | DAGs, backdoor criterion | **A** | Primary tool for DAG reasoning | Bambi (regression only) |
| 1 | Regression adjustment | **A** | System-of-equations + identification | Bambi (single equation) |
| 1 | Collider bias, Simpson's | **A** | Computational demonstration | Conceptual only |
| 2 | G-computation (binary) | **A** | `ate()` / `do()` replaces manual predict() | Bambi (manual G-comp) |
| 2 | CATE by segment | **A** | `cate()` with conditions | Bambi (interaction + predict) |
| 2 | Geo-experiment DiD | **A/B** | Panel mode DiD | CausalPy (better diagnostics) |
| 2 | MMM intro | **A** | Structural MMM with `do()` | PyMC-Marketing (black box) |
| 3 | Instrumental variables | **B/C** | Needs IV feature (B4) | CausalPy (works today) |
| 3 | Revenue optimization | **A** | `do()` for price counterfactuals | Manual loop |
| 4 | Survival analysis | **C** | Not supported | Bambi (survival family) |
| 4 | Churn DAG reasoning | **A** | Identification + collider warnings | Conceptual only |
| 5 | A/B testing | **C** | Overkill for single equation | Bambi |
| 5 | Hierarchical experiments | **C** | Panel mode possible but heavy | Bambi |
| 5 | Network effects | **C** | Not supported | Bambi (spillover terms) |
| 6 | Difference-in-differences | **A/B** | Panel mode; needs diagnostics (B1-B3) | CausalPy (purpose-built) |
| 6 | Synthetic control | **C** | Not a structural model problem | CausalPy |
| 6 | ITS | **C** | Not a structural model problem | CausalPy |
| 7 | CATE / moderation | **A** | `cate()` + interactions | Bambi (manual G-comp) |
| 7 | Optimal targeting | **A** | `do()` per customer for CATE | Manual code |
| 7 | BART / GP | **C** | Non-parametric, outside scope | PyMC-BART |
| 8 | Sensitivity analysis | **A** | `sensitivity()` method | Manual E-value |
| 8 | DAG testing | **A** | `implied_independences()` | No alternative |
| 8 | Communication | Neutral | Tool-agnostic session | — |

---

## Recommended Feature Issues (Prioritized)

Based on the assessment above, the following features would maximize pathmc's utility for the workshop. Issues are ordered by impact-to-effort ratio.

### Tier 1: High Impact, Moderate Effort

| # | Feature | Serves Sessions | Rationale |
|---|---------|-----------------|-----------|
| 1 | **Counterfactual time-series plot** for panel `do()` | 2, 6 | Essential for communicating panel intervention results. Currently requires manual matplotlib. CausalPy's automatic plots are the benchmark. |
| 2 | **ATE/CATE posterior plot** helpers | 1, 2, 7 | `model.plot_ate()`, `model.plot_cate(by=...)`, `do_result.plot()`. Removes ArviZ boilerplate and makes pathmc results presentation-ready. |
| 3 | **Workshop-oriented tutorial**: "From DAG to Causal Effect in 5 Minutes" | 1 | Show pathmc's unique value for single-equation models: draw DAG → check identification → fit → `ate()`. Addresses the "why not just Bambi?" question. |

### Tier 2: High Impact, High Effort

| # | Feature | Serves Sessions | Rationale |
|---|---------|-----------------|-----------|
| 4 | **IV instrument detection** from DAG (`model.instruments(T, Y)`) | 3 | Graph-theoretic extension of `adjustment_sets()`. Even without IV compilation, telling users which variables are valid instruments from the DAG is unique and valuable. |
| 5 | **IV compilation** (compile DAG with IV structure to 2SLS) | 3 | Ambitious but transformative. Would make pathmc the only package that unifies DAG → identification → IV estimation. |
| 6 | **Controlled Direct Effect** helper (`model.cde()`) | 7 | Already on roadmap. Automates the two-`do()` pattern for mediation decomposition in non-linear models. |

### Tier 3: Medium Impact, Low-Medium Effort

| # | Feature | Serves Sessions | Rationale |
|---|---------|-----------------|-----------|
| 7 | **Parallel trends diagnostic** for panel DiD | 6 | `model.parallel_trends_test()` — fit placebo DiD in pre-period. Makes pathmc's DiD competitive with CausalPy. |
| 8 | **Event study** plot for dynamic DiD effects | 6 | Period-specific treatment effects with pre-treatment falsification. Standard DiD diagnostic. |
| 9 | **Effects forest plot** | 1, 2, 7 | `model.plot_effects()` — forest plot of all labeled coefficients and `:=` parameters. Common reporting format. |
| 10 | **Overlap diagnostics** for positivity check | 1, 8 | Covariate distribution overlap by treatment group. Complements graph-level identification with data-level checks. |

### Tier 4: Nice to Have

| # | Feature | Serves Sessions | Rationale |
|---|---------|-----------------|-----------|
| 11 | **Proportion mediated** helper | 7 | `model.proportion_mediated()` — (total - CDE) / total. Requires CDE helper. |
| 12 | **Comparison table** of naive vs. adjusted estimates | 1 | Show the bias from omitting confounders by fitting both models. Pedagogically powerful for Session 1. |

---

## Strategic Recommendation

pathmc's strongest workshop fit is in the **conceptual foundation** sessions (1, 2, 7, 8) where DAG reasoning, g-computation, and identification are central. These sessions currently use Bambi with manual DAG drawings and manual G-computation — pathmc replaces both with a single, integrated workflow.

The **weakest fit** is in specialized method sessions (3: IV, 4: survival, 5: A/B testing, 6: synthetic control / ITS) where purpose-built packages (CausalPy, Bambi survival) handle the methods directly.

**The highest-ROI investment** is visualization (issues 1–2): pathmc already *computes* the right quantities via `do()`, `ate()`, `cate()`, and `sensitivity()`, but lacks the plotting layer to make results immediately presentable. Adding `plot_ate()`, `do_result.plot()`, and counterfactual time-series plots would make pathmc workshop-ready for Sessions 1, 2, 7, and 8 with no other changes.

**The highest-ambition investment** is IV support (issues 4–5): if pathmc could detect valid instruments from the DAG and compile to 2SLS, it would be the only package that goes from DAG → identification → estimation for *all three* standard identification strategies (backdoor adjustment, front-door, IV). This would make it the natural primary tool for the entire workshop.
