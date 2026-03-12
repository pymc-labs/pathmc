# Notebook Audit Report

**Date:** 2026-03-09
**Scope:** All 40 `.qmd` notebooks in `docs/`
**Question:** Do any notebooks demonstrate or fake code functionality that doesn't exist?

## Verdict

**No notebook fakes functionality.** Every `pathmc` API call across all 40 notebooks corresponds to a real, working method in the codebase. No fabricated outputs, no phantom parameters, no calls to nonexistent methods.

## Quality issues found

While no functionality is faked, the audit surfaced three categories of code-quality issues.

### 1. Reversed argument order in showcase code

`docs/index.qmd` line 106 has `model.sensitivity("X", "Y")` — the signature is `sensitivity(outcome, treatment)`, so this should be `model.sensitivity("Y", "X")`. This is in a non-executable code block (no `{python}` tag), so it doesn't produce faked output — it's a documentation typo.

### 2. Private attribute access: `DoResult._values`

Seven notebooks access `DoResult._values[var]` to extract raw posterior draws for KDE plotting. The public `DoResult` API only exposes `.mean(var)`, `.hdi(var)`, `.by_time(var)`, and subtraction. There is no public `.draws(var)` accessor.

| Notebook | Usage |
|----------|-------|
| `examples/vaccine_surrogates.qmd` | element-wise total − direct decomposition |
| `examples/ate_estimation.qmd` | KDE plotting of ATE draws |
| `examples/mmm_mediation.qmd` | lift draw extraction |
| `examples/mmm_basic.qmd` | 6 occurrences for plotting |
| `examples/saas_funnel.qmd` | contrast and lift draw extraction |

**Recommendation:** Add a public `.draws(var)` method to `DoResult` (mirroring `EffectResult.draws`), then replace all `._values` accesses.

### 3. Private attribute access: `PathModel._idata`

Three notebooks access `model._idata` instead of capturing the return value of `model.fit()`:

| Notebook | Usage |
|----------|-------|
| `examples/moderation.qmd` | extract posterior draws for interaction plotting |
| `examples/data_simulation.qmd` | posterior draws for recovery plots |
| `examples/did.qmd` | `az.plot_posterior()` and `.posterior.stack()` |

**Fix:** Assign `idata = model.fit(...)` and use `idata` instead of `model._idata`.

### 4. Internal module imports (pedagogically justified)

Three notebooks import from internal modules (`pathmc.graph`, `pathmc.identify`, `pathmc.parse`) to demonstrate concepts on hypothetical DAGs without fitting a model:

- `examples/front_door.qmd` — checks front-door criterion on the *causal* graph (without adjustment covariates)
- `examples/dag_testing.qmd` — shows implied independences before fitting
- `examples/causal_identification.qmd` — front-door criterion on a DAG with unobserved `U`

These are pedagogically motivated: the public `PathModel` API requires data and a fitted model, but the concept being taught is graph-only. The internal functions do exist and work correctly.

## Notebooks audited (all 40)

### Clean (30 notebooks)

`mediation`, `seeing_vs_doing`, `binary_outcomes`, `correlated_residuals`, `custom_priors`, `do_queries`, `sensitivity`, `simpsons_paradox`, `autoregressive`, `counterfactual`, `collider_bias`, `cross_sectional_vs_panel`, `latent_mediator`, `mmm_awareness`, `mmm_awareness_surveys`, `mmm_geo`, `dynamic_pricing`, `panel_interventions`, `how-it-works`, `comparison`, `examples/index`, `concepts/panel_data`, `concepts/bayesian_workflow`, `concepts/standardized_effects`, `concepts/model_specification`, `concepts/causal_inference`, `concepts/transforms_families`, `concepts/panel_interventions`

### Minor issues (10 notebooks)

See sections 1–4 above for details on: `index`, `vaccine_surrogates`, `ate_estimation`, `mmm_mediation`, `mmm_basic`, `saas_funnel`, `moderation`, `data_simulation`, `did`, `front_door`, `dag_testing`, `causal_identification`.
