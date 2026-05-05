---
name: pathmc
description: >
  Bayesian path analysis (observed-variable SEM) in PyMC. Compiles a
  lavaan-inspired formula DSL into a generative PyMC model, then layers
  introspection, identification diagnostics, the `do()` operator, and
  causal estimands (ATE/CATE/ATT/ATU/prob) on top. Use when the user
  asks to specify, fit, or query a Bayesian structural causal model;
  estimate average treatment effects via g-computation; check
  identification with adjustment sets or the front-door criterion;
  or simulate panel/longitudinal counterfactuals.
license: MIT
compatibility: Requires Python >=3.11, PyMC >=5.22.0.
metadata:
  homepage: https://github.com/pymc-labs/pathmc
  tags:
    - causal-inference
    - bayesian
    - sem
    - path-analysis
    - pymc
---

# pathmc

`pathmc` lets you specify a system of structural equations as a string,
compile it to a generative PyMC model, fit with MCMC, and reason about
causal effects using the do-operator.

## Quick start

```python
import pathmc

spec = """
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""

m = pathmc.model(spec, data=df)   # returns a PathModel (NOT a fitted result)
m.fit(draws=1000, chains=2)       # MCMC happens here

m.effects_summary()                # labeled coefficients + defined params
m.ate("Y", "X", values=(0, 1))     # average treatment effect via do()
m.adjustment_sets("X", "Y")        # valid backdoor adjustment sets
```

The DSL is lavaan-inspired:

- `Y ~ X` — regression
- `Y ~~ X` — residual covariance
- `indirect := a*b` — defined parameter
- `a*X` — labeled coefficient
- Transforms: `adstock(x, decay=...)`, `logistic_saturation(x, lam=...)`

## Decision table

| Need                                              | Use                                                    |
| ------------------------------------------------- | ------------------------------------------------------ |
| Build a model from a spec + data                  | `m = pathmc.model(spec, data=df)`                      |
| Explore the DAG without data                      | `m = pathmc.model(spec)` (data-free mode)              |
| Inspect causal DAG                                | `m.graph()`                                            |
| Inspect structural equations + priors             | `m.equations()` or `m.model_equations()`               |
| Inspect priors only                               | `m.priors()`                                           |
| Refine priors                                     | `m.set_priors({"beta_Y": Prior(...)})`                 |
| Prior predictive check                            | `m.sample_prior_predictive()`                          |
| Run MCMC                                          | `m.fit(draws=1000, chains=2)`                          |
| Summarize posteriors                              | `m.summary()` or `m.effects_summary()`                 |
| Standardized (stdyx) coefficients                 | `m.standardized()`                                     |
| Path-specific effect (e.g. `X -> M -> Y`)         | `m.effect("X -> M -> Y")`                              |
| Posterior predictions                             | `m.predict(...)`                                       |
| **Average treatment effect**                      | `m.ate(outcome, treatment, values=(0, 1))`             |
| **Conditional ATE** (effect modification)         | `m.cate(outcome, treatment, condition={"Z": z0})`      |
| ATE on the treated / untreated                    | `m.att(...)` / `m.atu(...)`                            |
| Probability under intervention                    | `m.prob("Y > 0", set={"X": 1})`                        |
| Manual intervention                               | `m.do(set={"X": 1})`                                   |
| Counterfactual / time-forward (panel)             | `m.do(set={...}, kind="time-forward")`                 |
| Adjustment sets for identification                | `m.adjustment_sets(treatment, outcome)`                |
| Yes/no identification check                       | `m.is_identifiable(treatment, outcome)`                |
| Front-door identification                         | `m.frontdoor_identifiable(treatment, outcome)`         |
| Warn about colliders in an adjustment set         | `m.collider_warnings(adjust, treatment, outcome)`      |
| Enumerate implied conditional independences       | `m.implied_independences()`                            |
| Test DAG implications against data                | `m.test_implications()`                                |
| Sensitivity analysis (unmeasured confounding)     | `m.sensitivity(outcome, treatment)`                    |
| Add lag columns to a panel DataFrame              | `pathmc.add_lags(df, columns, by, time)`               |
| Simulate from a fully-specified model             | `pathmc.simulate(spec, data, params=...)`              |

## Gotchas

1. **`pathmc.model(...)` returns a `PathModel`, not a fitted result.**
   You must call `.fit()` separately. `model()` only parses, builds the
   DAG, and compiles the PyMC graph — it does not sample.
2. **`m.do(...)` is a structural intervention, not conditioning.**
   It applies `pm.do()` graph surgery and forward-simulates from the
   intervened model, propagating posterior uncertainty through the
   causal chain (g-computation; Robins, 1986). It is **not** the same as
   conditioning on observed values. For typical user-facing queries,
   prefer the wrappers `m.ate()`, `m.cate()`, `m.att()`, `m.atu()`,
   `m.prob()`.
3. **The DSL is lavaan-*inspired*, not a 1:1 reimplementation.**
   `~`, `~~`, `:=`, and labeled coefficients all work. Latent-variable
   measurement models (`=~`) are out of scope in v0.1 — see the user
   guide for the full operator list.
4. **`Prior` is re-exported from `pymc_extras` for convenience.**
   `from pathmc import Prior` is a shortcut for
   `from pymc_extras.prior import Prior`. The canonical reference and
   list of supported distributions live in `pymc_extras`.
5. **Panel mode requires `add_lags()` *before* `model()`.**
   `pathmc.add_lags(df, columns=["sales"], by="region", time="week")`
   creates the lagged columns in the DataFrame; the `lag(...)` term in
   a spec then references those columns. Forgetting `add_lags()` raises
   a missing-column error from the compiler.
6. **Data-free models have a partial method surface.**
   When `data=None`, `graph()`, `equations()`, `priors()`,
   `adjustment_sets()`, `is_identifiable()`, `collider_warnings()`,
   `implied_independences()` all work. `fit()`, `do()`, `ate()`,
   `cate()`, `design()`, `sample_prior_predictive()`,
   `test_implications()`, `sensitivity()` raise `RuntimeError` until
   the model is rebuilt with data.
7. **`PathModel` is not in `pathmc.__all__`** — it's the class returned
   by `model()`. You don't import it directly; you receive it. Type
   annotations can use `pathmc.PathModel` (it is reachable as an
   attribute) but the public entrypoint is the `model()` function.

## Capabilities and boundaries

**Agents using pathmc can:**

- Write spec strings in the DSL (regressions, residual covariances,
  defined parameters, labeled coefficients, transforms).
- Configure custom priors via `Prior` objects from `pymc_extras`.
- Run `fit()` with PyMC's NUTS sampler (or `nutpie` / `numpyro` via
  the `samplers` extra).
- Query `ate`/`cate`/`att`/`atu`/`prob`/`effect` with full posterior
  uncertainty.
- Check identification (`adjustment_sets`, `is_identifiable`,
  `frontdoor_identifiable`, `collider_warnings`).
- Test the DAG's conditional-independence implications against data
  (`test_implications`).
- Build hierarchical panel models with random intercepts/slopes and
  use `lag()` terms.
- Run sensitivity analysis to quantify robustness to unmeasured
  confounding.

**Out of scope (do not attempt):**

- **Latent variables / SEM measurement models** (the `=~` operator).
  Out of scope in v0.1; on the post-v1 roadmap.
- **Categorical mediators or treatments with >2 levels in
  `ate()`/`cate()`** without manual `do()` calls. Use `m.do(set={...})`
  with explicit values for non-binary interventions.
- **Editing the compiled `pm.Model` object directly.** pathmc owns the
  graph; mutating it bypasses the introspection layer and breaks
  `do()` propagation. To customize, change the spec or pass `priors=`
  / `families=` to `model()`.

## Patterns

### Inspect before sampling (data-free DAG exploration)

```python
m = pathmc.model("""
    M ~ a*X
    Y ~ b*M + c*X
    indirect := a*b
""")
m.graph()                      # DAG plot
m.equations()                  # structural equations + priors
m.adjustment_sets("X", "Y")    # what to adjust for
m.is_identifiable("X", "Y")    # can we estimate the effect at all?
```

### Standard fit-and-query workflow

```python
m = pathmc.model(spec, data=df)
m.fit(draws=1000, chains=2)
m.effects_summary()                          # labeled coefs
m.ate("Y", "X", values=(0, 1))               # ATE
m.cate("Y", "X", condition={"Z": 1})         # CATE | Z=1
m.test_implications()                        # DAG vs data check
```

### Panel model

```python
import pathmc
df = pathmc.add_lags(df, columns=["sales"], by="region", time="week")
m = pathmc.model(
    "sales ~ b*price + a*lag(sales) + trend",
    data=df,
    panel={"unit": "region", "time": "week"},
    pooling="partial",
)
m.fit()
m.do(set={"price": 1.5}, kind="time-forward")
```

### Custom priors

```python
from pathmc import Prior   # re-export of pymc_extras.prior.Prior

m = pathmc.model(
    spec,
    data=df,
    priors={
        "beta_Y": Prior("Normal", mu=0, sigma=2),
        "sigma_Y": Prior("HalfNormal", sigma=1),
    },
)
m.priors()                  # confirm overrides applied
m.sample_prior_predictive() # check the priors imply plausible data
```

## Resources

- Docs site: <https://pymc-labs.github.io/pathmc/> *(once Phase 4b lands)*
- `llms.txt` — indexed API reference for LLMs
- `llms-full.txt` — comprehensive API documentation for LLMs
- GitHub: <https://github.com/pymc-labs/pathmc>
