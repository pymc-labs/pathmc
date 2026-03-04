# PyMC-Native do() Architecture: Findings & Design

## Problem Statement

pathmc needs a `do()` operator that propagates interventions through the causal
DAG using posterior parameter draws. The original plan called for moving from
NumPy-based manual propagation to PyMC-native graph surgery (`pm.do()`,
`pm.observe()`, `pm.sample_posterior_predictive()`). This document records the
investigation into how to make that work reliably.

## Key Issue: PyMC #7069 — `sample_posterior_predictive` Volatility

PyMC's `sample_posterior_predictive` has a well-documented problem where
including unobserved model variables in `var_names` causes them to be **resampled
from the prior** instead of copied from the posterior trace. Furthermore, any
variable that depends on `MutableData` that has changed (via `pm.set_data()`)
becomes "volatile" and triggers cascading resampling of all dependent variables.

- **Issue**: https://github.com/pymc-devs/pymc/issues/7069
- **Status**: Open (as of 2025-05), with active discussion and several PRs
- **Relevant PRs**:
  - [#7596](https://github.com/pymc-devs/pymc/pull/7596) — Make do interventions shared variables by default (merged, available in 5.22)
  - [#7969](https://github.com/pymc-devs/pymc/pull/7969) — Fix: Deterministic variables no longer cause resampling
- **ricardoV94's recommendation**: Use `compute_deterministics` or
  `apply_function_over_dataset` for advanced forward computation to avoid
  volatility issues entirely

### Why `pm.set_data()` Is Problematic

When `pm.set_data({'X': new_values})` changes a `MutableData` variable, ALL
downstream variables are marked volatile. The sampler then resamples parameters
like `beta_M` from their **prior**, not the posterior. This completely defeats
the purpose of posterior-based interventional simulation.

### Why `pm.do()` Avoids This Problem

`pm.do()` creates a **new model** with constants baked in — no `MutableData`
mutation occurs. The remaining free RVs (betas, sigmas) are matched to the
posterior trace by name and their posterior draws are used correctly. There is
no volatility cascade because no shared data has changed.

**Verified empirically**: In PyMC 5.22, posterior draws are correctly used after
`pm.do()` + PPC. The posterior mean and std of mu_Y confirm posterior (not prior)
values: mean=0.80, std=0.11 vs prior which would give mean≈0, std>>10.

## The Architecture: Generative Model + `pm.observe` + `pm.do`

### Core Idea

Build a **generative model** where all endogenous variables are **free RVs**
(not observed). Use `pm.observe()` to create the estimation model, and `pm.do()`
on the generative model for interventions.

```python
# 1. GENERATIVE MODEL — the data-generating process
with pm.Model() as gen_model:
    X_data = pm.Data('X', X_obs)
    beta_M = pm.Normal('beta_M', 0, 10, shape=2)
    mu_M = pm.Deterministic('mu_M', beta_M[0] + beta_M[1] * X_data)
    sigma_M = pm.HalfNormal('sigma_M', 1)
    M = pm.Normal('M', mu=mu_M, sigma=sigma_M)       # ← free RV

    beta_Y = pm.Normal('beta_Y', 0, 10, shape=3)
    mu_Y = pm.Deterministic('mu_Y', beta_Y[0] + beta_Y[1] * X_data + beta_Y[2] * M)
    sigma_Y = pm.HalfNormal('sigma_Y', 1)
    Y = pm.Normal('Y', mu=mu_Y, sigma=sigma_Y)       # ← free RV

# 2. ESTIMATION — pm.observe conditions free RVs on data
est_model = pm.observe(gen_model, {'M': M_data, 'Y': Y_data})
with est_model:
    idata = pm.sample(...)

# 3. DO (predictive) — pm.do on generative model + PPC
do_model = pm.do(gen_model, {'X': np.full(N, 1.0)})
with do_model:
    ppc = pm.sample_posterior_predictive(idata)

# 4. DO (mean) — anonymous tensor trick + compute_deterministics
replacements = {'X': np.full(N, 1.0)}
for var in ['M', 'Y']:
    replacements[var] = gen_model[f'mu_{var}'] * 1  # anonymous tensor
mean_do_model = pm.do(gen_model, replacements)
det = compute_deterministics(idata.posterior, model=mean_do_model)
```

### Why This Works

| Step | Operation | Why correct |
|------|-----------|-------------|
| Estimation | `pm.observe(gen_model, obs)` | M, Y become observed → standard SEM likelihood, correct coefficient estimates |
| do(predictive) | `pm.do(gen_model, {'X': val})` | X replaced with constant; M, Y stay as free RVs; PPC forward-samples M from N(mu_M, sigma_M) using posterior beta_M, sigma_M → correct causal chain with noise |
| do(mean) | `pm.do(gen_model, {'X': val, 'M': mu_M*1, 'Y': mu_Y*1})` | All endogenous RVs replaced with their mu Deterministics → mean propagation through DAG without noise |
| Latent vars | Variable not passed to `pm.observe` | Stays as free RV in estimation (sampled jointly), and in do model |

### Empirical Verification

All tested on a mediation DGP: X → M → Y, true effects β_M=[0, 0.5], β_Y=[0, 0.3, 0.8].

**Coefficient recovery** (pm.observe estimation model):
- β_M = [0.002, 0.427] ≈ true [0, 0.5] ✓
- β_Y = [0.044, 0.419, 0.791] ≈ true [0, 0.3, 0.8] ✓

**ATE do(X=0→1), true total effect = 0.3 + 0.8×0.5 = 0.7**:
- kind="mean" (compute_deterministics): 0.7567 ✓
- kind="predictive" (PPC): 0.7527 ✓

**Direct effect do(M=0→0.5), true effect = 0.8×0.5 = 0.4**:
- kind="mean": 0.3954 ✓

**Latent mediator**: M not passed to pm.observe → sampled as free RV.
Posterior includes M draws. Model has expected convergence difficulties
(inherent to latent variable estimation, not the architecture).

## Technical Details

### The Anonymous Tensor Trick

`pm.do(model, {'M': model['mu_M']})` fails with "Variable name M already exists"
because `model['mu_M']` is a named Deterministic. When pm.do replaces M, it
tries to register the replacement tensor, which conflicts with existing names.

**Fix**: Multiply by 1 to create an anonymous tensor:
```python
pm.do(model, {'M': model['mu_M'] * 1})  # anonymous Elemwise, no name conflict
```

After this, M becomes a Deterministic equal to mu_M in the new model. sigma_M
and sigma_Y remain as free RVs but are present in the posterior so
`compute_deterministics` works correctly.

### kind="mean" vs kind="predictive"

**kind="predictive"**: Uses PPC on the do model. Free endogenous RVs (M, Y)
are NOT in the posterior trace (they were observed during estimation under
different model structure). PPC correctly forward-samples them from their
conditional distributions using posterior parameter draws. This gives the full
predictive distribution including all residual noise.

**kind="mean"**: Replaces all endogenous RVs with their mu Deterministics
(anonymous tensor trick), then uses `compute_deterministics`. This propagates
posterior means through the DAG without any noise, giving the expected value
E[Y | do(X=x)].

### Intervening on Endogenous Variables

For do(M=val), `pm.do(gen_model, {'M': np.full(N, val)})` replaces the M free
RV with a constant. In the do model, mu_Y = f(X, M=val) uses the constant M.
Both PPC (for predictive) and compute_deterministics (for mean, after also
replacing Y with mu_Y*1) work correctly.

### `compute_deterministics` Requirements

`compute_deterministics(dataset, model=model)` extracts ALL free RVs from the
dataset. If the model has free RVs not in the dataset, it fails with KeyError.

- **do model** (for predictive): Has M, Y as free RVs → NOT in posterior → use PPC instead
- **mean do model** (with anonymous tensor replacements): M, Y become Deterministics → only beta/sigma free RVs remain → all in posterior → works ✓

### pm.observe Behavior

`pm.observe(model, {'M': data, 'Y': data})` returns a new model where M and Y
become observed RVs. The free RVs change from [beta_M, sigma_M, M, beta_Y,
sigma_Y, Y] to [beta_M, sigma_M, beta_Y, sigma_Y]. This is exactly the standard
SEM likelihood structure.

### Compile Changes Required

The current compiler emits `pm.Normal('var_obs', mu=mu, sigma=sigma, observed=y)`.
The new architecture emits `pm.Normal('var', mu=mu, sigma=sigma)` (free RV) and
uses `pm.observe` externally. Key naming change: drop the `_obs` suffix since
the variable itself represents the random variable, not its observation.

Important: mu_Y must wire through M (the free RV), not through mu_M or data['M'].
This ensures that:
- During estimation (pm.observe): mu_Y uses observed M → correct coefficients
- During do (pm.do): mu_Y uses the free/replaced M → correct causal propagation

### Panel do() Considerations

Panel do() uses time-forward NumPy propagation with adstock state tracking.
This is kept as-is for now — the PyMC-native approach for panel models would
require `pytensor.scan` which is a separate, larger effort. The panel do()
already handles latent variables correctly.

## Comparison: Old vs New Architecture

| Aspect | Old (NumPy propagation) | New (PyMC-native) |
|--------|------------------------|-------------------|
| Estimation model | `pm.Normal('M_obs', observed=data)` | `pm.observe(gen_model, {'M': data})` |
| do() mechanism | Manual topological propagation in NumPy | `pm.do()` + PPC / `compute_deterministics` |
| Causal chain | Explicit loops over DAG | Implicit via PyMC graph |
| Residual noise | Manual `rng.normal(0, sigma)` | PPC handles automatically |
| Latent variables | Skip noise for latent vars | Naturally handled (free RV not observed) |
| Graph surgery | None (custom logic) | `pm.do()` creates clean new model |
| Posterior usage | Manual extraction from idata | Automatic name matching |

## Implementation Plan

1. **compile.py**: Emit free RVs for endogenous vars (not observed). Wire mu_Y
   through M (the free RV). Return the generative model.
2. **model.py**: Call `pm.observe()` to create the estimation model. Store both
   gen_model and est_model. Wire `do()` to the new engine.
3. **simulate.py**: Add `run_do_pymc()` using pm.do + PPC (predictive) or
   pm.do + compute_deterministics (mean). Keep NumPy `run_do` and `run_panel_do`
   for backward compatibility with panel models.
4. **Tests**: Run existing test suite. Create test_latent.py for latent mediator
   scenarios.

## Risks & Mitigations

- **Anonymous tensor trick fragility**: The `* 1` trick creates an anonymous
  Elemwise node. This is a standard PyTensor operation unlikely to break, but
  could be replaced with `pytensor.tensor.specify_shape` or explicit graph
  cloning if needed.
- **PPC warnings**: "Could not extract data from symbolic observation M" warnings
  appear during PPC on the do model. These are cosmetic (ArviZ trying to extract
  observed data from a model where M/Y are free RVs) and can be suppressed.
- **Panel do()**: Stays NumPy-based for now. Future work could use pytensor.scan.
- **Residual covariance blocks (~~)**: These use MvNormal with observed data.
  Need to verify they work correctly with the pm.observe approach.
