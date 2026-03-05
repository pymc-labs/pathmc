# Counterfactual Inference and Abduction Notes

## Why this note exists

Counterfactual inference is exciting and easy to misstate. This note captures the
shared understanding from recent discussion so we can reuse it in docs, APIs, and
implementation decisions.

## Core distinction: intervention vs counterfactual

- **Intervention (`do`)** asks population-level questions like: "What happens on
  average if we set `H = 2`?"
- **Counterfactual** asks unit-level questions like: "For this specific person
  with observed evidence, what would `Y` have been if `H = 2`?"

The key difference is the treatment of exogenous factors `U`.

- Interventions typically integrate over population uncertainty in `U`.
- Counterfactuals must infer this unit's `U` from factual evidence, then carry
  the same `U` into the intervened world.

## Why `pm.observe` + `pm.do` alone is not enough

`pm.observe` and `pm.do` on a single model are powerful, but they do not
automatically produce Pearl-style unit-level counterfactuals.

Reason: counterfactuals are a **two-world** computation.

1. **Factual world**: condition on observed evidence to infer latent exogenous
   values (`abduction`).
2. **Counterfactual world**: modify structural equation(s) with `do(...)`, then
   predict while reusing the same abducted `U`.

If we skip abduction, we generally compute an interventional quantity
(`E[Y | do(X=x)]` or `E[Y | do(X=x), covariates]`), not `Y_x` for the same unit.

## The 3-step SCM procedure

1. **Abduction**: infer `p(U | evidence, theta)` (or joint with `theta`).
2. **Action**: apply intervention (`do`) to structural equation(s).
3. **Prediction**: simulate outcome with intervened equations and fixed abducted
   `U`.

This is the standard SCM counterfactual pipeline.

## Residuals vs structural `U`

It is tempting to think "just use residuals." That is valid only in specific
model classes.

- Structural `U` is not just generic model error; it is part of the causal data
  generating process.
- Counterfactual consistency requires per-unit, per-posterior-draw reuse of the
  same latent exogenous realization across worlds.

Aggregate residual distributions are usually insufficient for this requirement.

## When abduction is easy: linear Gaussian additive models

For additive equations

`V = intercept_V + sum(beta * parents) + U_V`

we can invert algebraically for each posterior draw:

`U_V = evidence[V] - intercept_V - sum(beta * evidence[parent])`

This is fast and exact for the modeled structure (within posterior uncertainty).
No second sampling pass over `U` is required.

## When abduction is hard: non-Gaussian or non-additive models

For Bernoulli/Poisson/nonlinear/non-invertible equations, there is often no
single residual-like inversion.

Then abduction is itself an inference problem over latent exogenous factors.
Operationally, that means explicitly representing latent noise terms (or an
equivalent latent-variable parameterization) and sampling/inference under
factual evidence.

## Why this belongs in a causal layer on top of PyMC

PyMC/PyTensor can express and infer all needed pieces, but generic models do not
carry enough causal semantics by default.

A robust counterfactual API needs explicit metadata and orchestration:

- structural equations
- DAG/parent relationships
- endogenous vs exogenous roles
- intervention targets and validation
- abduction strategy (analytic vs sampled)
- world bookkeeping (shared `U`, different equations)

This is exactly what a package like `pathmc` can provide cleanly on top of PyMC
primitives.

## Practical API direction (pathmc)

Desired user shape:

```python
result = model.counterfactual(
    evidence={"X": 0.5, "H": 1.0, "Y": 1.5},
    do={"H": 2.0},
)
```

Implementation sketch:

1. Use posterior draws of structural parameters.
2. Abduct `U` draw-wise from factual `evidence`.
3. Apply `do` to structural equations.
4. Predict forward with fixed abducted `U`.
5. Return posterior summaries in a `DoResult`-compatible interface.

## Common mistakes to avoid

- Calling interventional predictions "counterfactuals" without abduction.
- Conditioning on outcome in the counterfactual world rather than factual world.
- Re-sampling new `U` after intervention (breaks same-unit counterfactual logic).
- Confusing mean-response (`mu`) comparisons with full outcome distributions.

## Quick checklist for validity

- Do we condition on factual evidence first?
- Do we recover/infer unit-level `U`?
- Do we hold `U` fixed across factual and intervened worlds?
- Is `do(...)` applied only in the prediction world?
- Are reported estimands clearly labeled (interventional vs counterfactual)?

## References

- Pearl, Glymour, Jewell (2016), *Causal Inference in Statistics: A Primer*,
  Section 4.2
- `docs/dev/issue_29_draft.md`
- `docs/examples/counterfactual.qmd`
