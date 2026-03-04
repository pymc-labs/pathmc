# Panel Intervention Engine Unification: Analysis

## Current State

pathmc has **three** panel `do()` engines in `simulate.py`:

| Engine | Mechanism | Temporal state | Adstock impl |
|--------|-----------|----------------|--------------|
| **numpy** (default) | Python loop over `(unit, time, draw)` | Per-draw correct | Manual `adstock_state` dict in NumPy |
| **batched** | Single-timestep `pm.Model` + `pm.do()` per time step (Python loop) | Mean-field (averaged across draws between steps) | `pm.Data` for adstock state, `compute_deterministics` per step |
| **scan** | Full time loop encoded as `pytensor.scan` in one `pm.Model` | Per-draw correct | Adstock state carried in `outputs_info` |

Meanwhile, the **cross-sectional** `do()` uses a fourth approach: PyMC-native graph surgery (`pm.do()` on the generative model + PPC or `compute_deterministics`). This is clean and requires no custom propagation logic.

The **estimation model** (`compile.py`) uses pymc-marketing's convolution-based adstock — a vectorised `pt.signal.convolve1d` applied to the entire time series at once. No scan. No recurrence. Lags are expressed via `lag(var)` in the spec and compiled into the model.

This creates a fundamental mismatch: the estimation model "sees" the full time series as a flat vector (convolution operates on all T points simultaneously), but `do()` needs to generate *new* trajectories where each time step depends on simulated values at previous steps.

## Why Three Engines Exist

The three engines are a consequence of the estimation model not encoding temporal dependencies in its computation graph.

### The core problem

In the estimation model:
- `adstock(spend)` is a convolution over the observed spend series → produces a vector of length N
- `lag(sales)` is a structural term in the spec → compiled into the temporal model
- `mu_sales = beta_0 + beta_1 * adstock(spend) + beta_2 * lag(sales)`

When you do `pm.do(gen_model, {'spend': new_spend})`, the adstock convolution recomputes correctly over the new spend values. With the old `add_lags()` approach, the lag was a pre-computed data column — the model had no knowledge that it was the previous value of `sales`. The `lag()` DSL syntax encodes this structurally.

This breaks causal propagation through time. Each engine is a different workaround:

1. **numpy**: Manually implement the full time-forward loop in NumPy, bypassing the PyMC graph entirely. Correct but duplicates all the model logic.

2. **batched**: Build a one-step PyMC model, step forward in a Python loop, pass simulated values back as `pm.Data`. Approximates because `pm.Data` is shared across draws (mean-field).

3. **scan**: Encode the full time loop as `pytensor.scan` in a *new* model (separate from the estimation model), then use `pm.do()` once. Per-draw correct, but this is essentially building a second model that mirrors the estimation model's structure.

## Two Architectures for Temporal Dependencies

### Architecture 1: Scan in the generative model ("scan + do")

Build the generative model with `pytensor.scan` encoding the full temporal structure. `pm.do()` on this model handles everything — no separate simulation engine needed.

### Architecture 2: The scan engine (current approach)

Use convolution for estimation. At `do()` time, build a *separate* scan-based model that mirrors the estimation model's structure, then apply `pm.do()` to that.

### What's actually different?

The difference is **where the scan lives**:

| | Scan + do (Architecture 1) | Scan engine (Architecture 2) |
|--|---|---|
| **Estimation model** | Uses `pytensor.scan` for adstock and temporal deps | Uses convolution + data columns |
| **do() model** | IS the estimation model (with `pm.do()` applied) | Separate model built by `_build_scan_model()` |
| **# of model representations** | 1 | 2 |
| **do() model construction** | None needed (already exists) | ~300 lines of model-building code at do() time |
| **Risk of estimation↔simulation divergence** | Zero (same model) | Real (must keep `_build_scan_model` in sync with `compile.py`) |

### do() performance: scan + do should be comparable or faster

For the `do()` operation itself, both architectures run the same computation: `pytensor.scan` over T time steps, evaluated against posterior draws via `compute_deterministics`.

Scan + do may be **slightly faster** because:
- No model construction overhead at `do()` time (the scan model already exists)
- The scan graph is already compiled and optimised by PyTensor
- Parameter names are guaranteed to match (same model)

The scan engine must build a fresh model, compile the scan graph, and map parameter names every time `do()` is called.

## Adstock-Only vs. Autoregressive: A Critical Distinction

### Adstock on exogenous variables: scan + do works cleanly

For models like `sales ~ adstock(spend)` where adstock operates only on exogenous inputs:

- The scan carry tracks `adstock_state`, not any endogenous value
- `pm.do(gen_model, {'spend': new_spend})` replaces the spend input
- The scan recomputes the adstock chain over the new spend values
- `mu_sales` updates automatically
- `pm.observe({'sales': sales_obs})` conditions the free RVs on data for estimation

This is fully correct and elegant. Actually, for this case, convolution-based `pm.do()` *already works* — the convolution recomputes over the new spend. No scan needed at all.

### Autoregressive terms: the teacher-forcing question

For models with lags of endogenous variables (`sales ~ spend + lag(sales)`), there's a deeper issue.

**Teacher forcing** (standard regression approach): Each time step conditions on the *observed* previous outcome. `mu_t = β₀ + β₁·spend_t + β₂·sales_{t-1}^obs`. The old `add_lags()` approach produced pre-computed data columns for this.

**Free-running** (state-space approach): Each time step conditions on the *model-predicted* previous outcome. `mu_t = β₀ + β₁·spend_t + β₂·mu_{t-1}`. This is what a scan-based generative model would produce — the carry variable feeds the previous prediction forward.

These are **different models** with different likelihoods:

| | Teacher forcing | Free running (scan) |
|--|--|--|
| **Likelihood at time t** | `p(sales_t \| sales_{t-1}^obs, params)` | `p(sales_t \| mu_{t-1}(params), params)` |
| **Conditional on** | Observed previous value | Model-predicted previous value |
| **Prediction errors** | Independent across time steps | Correlated (error at t affects mu at t+1) |
| **Gradients** | Simple (no backprop through time) | Complex (backprop through T steps of scan) |
| **Estimation** | Fast, well-conditioned | Slower, can be more complex |
| **Causal simulation** | Needs special engine | `pm.do()` just works |

For `do()` / causal simulation, we *always* want free-running: when we intervene on spend, the entire sales trajectory changes, and each sales prediction depends on the previous simulated sales, not observed ones.

The question is whether to use free-running for estimation too.

### The case for free-running estimation (scan + do)

The free-running / state-space formulation is actually **more principled** in several ways:

1. **One model, one truth**: The estimation model IS the generative model. There's no translation step, no risk of divergence, no `_build_scan_model()` to maintain.

2. **Coherent joint distribution**: The model defines `p(sales_1:T | spend_1:T, params)` as a coherent joint distribution. Teacher forcing decomposes this into `∏ p(sales_t | sales_{t-1}^obs, params)`, which is correct but loses the generative interpretation.

3. **Naturally handles do()**: `pm.do()` on the generative model produces correct counterfactual trajectories with no special machinery.

4. **Future-proof**: Any new temporal transform only needs a `step()` function for the scan body. No separate convolution kernel needed. No `_build_scan_model()` update needed.

5. **Handles latent variables correctly**: If an endogenous variable is latent (not observed), the scan naturally propagates the latent trajectory. No special cases.

### The case against: estimation cost

The honest concern is **NUTS performance**. Backpropagating through `pytensor.scan` for T time steps is more expensive than teacher-forced regression where each step's gradient is independent.

However, the cost may be overstated for pathmc's use case:

1. **Short time series**: Marketing/social science panel data typically has T = 50–200 time steps. This is very different from RNNs with T = 10,000. Scan overhead is roughly proportional to T.

2. **Simple step functions**: The scan body is a linear combination + one multiply for adstock. This is trivial compared to e.g. an LSTM cell. The per-step gradient is cheap.

3. **pymc-marketing's context is different**: They chose convolution because their MMMs have 3–5 channels each with adstock, T = 100–200, and they need fast iteration. But their models don't have autoregressive terms or causal DAG structure — the temporal dependency is purely in the exogenous adstock transformation, where convolution is a perfect fit.

4. **Only applies to panel models**: Cross-sectional models don't use scan at all. The scan cost is only paid when the model has temporal structure.

**We don't actually know the cost.** The "5–20x slower" claim in the original analysis was inherited from general RNN wisdom, not benchmarked on pathmc-scale models. For T = 100 with a trivial step function, the difference might be 2x or less.

### What scan + do looks like concretely

```python
with pm.Model() as gen_model:
    spend_data = pm.Data('spend', spend_obs)       # (T, n_units)
    alpha_decay = pm.Beta('alpha_decay', 2, 2)
    beta = pm.Normal('beta', 0, 10, shape=3)
    sigma = pm.HalfNormal('sigma', 1)

    def step(spend_t, prev_sales_mu, prev_adstock, alpha, beta):
        adstock_t = spend_t + alpha * prev_adstock
        mu_t = beta[0] + beta[1] * adstock_t + beta[2] * prev_sales_mu
        return mu_t, adstock_t

    [mu_all, adstock_all], _ = pytensor.scan(
        fn=step,
        sequences=[spend_data],
        outputs_info=[pt.zeros(n_units), pt.zeros(n_units)],
        non_sequences=[alpha_decay, beta],
    )
    pm.Deterministic('mu_sales', mu_all)
    sales = pm.Normal('sales', mu=mu_all, sigma=sigma)   # free RV (T, n_units)

# Estimation: condition on observed data
est_model = pm.observe(gen_model, {'sales': sales_obs})
with est_model:
    idata = pm.sample()

# do(): just use pm.do — the scan handles everything
do_model = pm.do(gen_model, {'spend': new_spend_scenario})
det = pm.compute_deterministics(idata.posterior, model=do_model)
# or: pm.sample_posterior_predictive(idata, model=do_model)
```

No `_build_scan_model()`. No `_build_step_model()`. No `run_panel_do()`. No `run_panel_do_batched()`. No `run_panel_do_scan()`. No engine selection. The panel `do()` code path is identical to the cross-sectional one: `pm.do()` on the generative model.

### The `prev_sales_mu` subtlety

In the scan above, the carry variable is `prev_sales_mu` — the model's predicted mean, not the observed sales. During estimation (`pm.observe`), the likelihood evaluates `p(sales_obs_t | mu_t, sigma)` where `mu_t` depends on `mu_{t-1}`. This is different from teacher forcing where `mu_t` depends on `sales_obs_{t-1}`.

For `do()`, this is exactly right: the scan propagates the model's prediction forward, which is what counterfactual simulation requires.

For estimation, the question is whether this produces the same coefficient estimates as teacher forcing. In general:
- For well-identified models with low noise, the difference is negligible
- For high-noise autoregressive models, teacher forcing can be more stable
- The free-running formulation is the correct generative model — teacher forcing is an approximation that happens to be convenient

## Recommendation: Benchmark scan + do, then decide

The original recommendation (Path B: keep convolution, promote scan engine) was conservative. The scan + do approach (Path A) is more elegant and maintainable. The deciding factor is estimation performance, which should be **benchmarked, not assumed**.

### Proposed benchmark

1. Take 2-3 representative panel models:
   - Adstock only: `sales ~ adstock(spend, decay)` (T=100, 5 units)
   - Adstock + AR: `sales ~ adstock(spend, decay) + lag(sales)` (T=100, 5 units)
   - Multi-equation: `awareness ~ adstock(spend, decay); sales ~ awareness + lag(sales)`
2. Compile each with convolution (current) and with scan
3. Compare: sampling time, ESS/second, divergences, coefficient recovery

If scan-based estimation is within 2–3x of convolution for typical model sizes, the maintenance and correctness benefits of scan + do make it the clear winner.

### Fallback if scan is too slow for estimation

If benchmarks show scan is prohibitively slow for some model sizes, a hybrid is possible:
- Compile with convolution for estimation (fast NUTS)
- Auto-generate the scan model for do() (current scan engine approach)
- But only maintain this as an optimisation, not the primary architecture

This is similar to how JAX can JIT-compile different graph representations for forward vs. backward passes. The generative model conceptually uses scan; the estimation model is an equivalent convolution-based form.

## Summary

| Question | Answer |
|----------|--------|
| Do we need 3 panel engines? | **No.** At most one. |
| Scan + do vs. scan engine for do()? | **Scan + do should be comparable or faster** (no model construction overhead). |
| Scan + do: correct? | **Yes.** Free-running is the correct generative model. Teacher forcing is a convenient approximation. |
| Scan + do: elegant? | **Very.** One model, `pm.do()` just works, ~300 lines of `simulate.py` deleted. |
| Should we use it? | **Benchmark first.** The deciding factor is estimation performance, not do() performance. |
| What if scan is too slow for estimation? | Fall back to convolution for estimation + scan engine for do() (current Path B). |
| What do future transforms need? | A `step()` function for the scan body. That's it. |

The strongest argument for scan + do: the code that doesn't exist can't have bugs. Eliminating the scan engine, the batched engine, the numpy engine, and `_build_scan_model()` removes hundreds of lines of model-mirroring code that must stay in sync with the compiler.
