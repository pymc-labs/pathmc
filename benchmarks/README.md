# Posterior-query execution benchmarks

Repeated mean `do()` and ATE queries on fully observed cross-sectional linear Gaussian models now reuse prepared PyTensor functions. Ordinary `predict()` reuses PyMC's own frozen-model compilation cache when available. The public query methods and result classes are unchanged. This report records the staged execution work in [#526](https://github.com/pymc-labs/pathmc/issues/526); timing results are review evidence, not CI thresholds.

## Scope and execution contract

| Model/query capability | Execution route | Measurement or correctness coverage |
| --- | --- | --- |
| Fully observed Gaussian numeric linear terms, mean `do`/ATE | Prepared symbolic posterior/scenario batches | Small/large timings; analytic NumPy mediation oracles with distinct chains/draws |
| Scalar and per-row interventions, endogenous targets, same-shape `pm.set_data` | Runtime numeric inputs to the same plan | Value changes, posterior replacement, chain/draw labels, and unit retention checked |
| Ordinary `predict`, including output selection, same-row-count `data=`, and changed seeds | Instance-owned frozen observed model, using PyMC's reseedable cache | Compilation counts, data restoration, posterior replacement, seeded repeatability, and extension checked |
| Advanced prediction kwargs, unsupported frozen-model state, unavailable `freeze_model` | Existing PyMC prediction route | Capability fallback checks; no dependency bump |
| Predictive `do`, interactions, scaling, categorical and data-basis terms | Existing executor | Predictive-do timings; interactions/scaling/category/basis timings not collected, with existing regression suites retained |
| Bernoulli/count intermediaries | Existing executor | Small/large timings and independent response-link oracles |
| Gaussian exogenous lag, endogenous AR(1), adstock followed by saturation | Existing panel executor | Small/large query timings and optimized likelihood/gradient profiles |
| Pooling, missing observations, latent state, released interventions | Existing executor | Existing regression suites; latent fitting probes; individual query stress timings not collected |
| History-dependent policy trajectories | Existing policy executor | Twenty-replication timing; existing policy oracle/lifecycle suites. One replication is unsupported by the existing policy API, which requires at least two |

Prepared means clone the already compiled `mu_*` expressions and use symbolic posterior, scenario, and numeric-data inputs. They do not implement another set of structural equations. PyMC's `vectorize_over_posterior` supplies concrete posterior arrays; retaining such a graph would retain stale draws. The prepared route instead uses PyTensor's underlying graph vectorization with symbolic inputs. The retained per-draw control in the harness uses the same expression with one draw per function invocation, isolating the effect of batching.

Each `PathModel` retains at most 16 execution plans in an LRU. Plans contain executable graphs and layout metadata, not results or posterior copies. Recompilation, changed priors, and refitting clear the cache; deserialization creates an empty cache with a fresh lock. The structural signature includes graph generation, posterior schema, mutable-input layout/dtype, and effective compilation settings. Numeric intervention/data/posterior values and random seeds are supplied anew. ATE evaluates its two arms against the same ordered draws; a public intervention vector continues to mean observation-row values.

Draw/scenario chunks limit the observation-expanded working batch to one million elements per graph intermediate, or one observation vector when that vector already exceeds the budget. Averaging happens inside the compiled expression; retaining unit-level results still requires the requested output storage. Cached functions and prediction-data updates are serialized by an instance lock. This does not add a concurrent serving contract for arbitrary external graph or posterior mutations.

Only the observed prediction graph is frozen. Freezing a generative graph can freeze data needed by free outcome RVs, so it is unsuitable for changing interventions. PyMC backends that detach RNG state, including JAX, use the ordinary uncached route here. Query compiler choice remains separate from sampler choice. The PyTensor version tested resolves `FAST_RUN` to `NumbaLinker`; explicit C/VM and Numba measurements are therefore reported separately.

## Reproduction and measurement

Run commands from the repository root through the locked environment:

```bash
uv run --locked --all-extras python scripts/benchmark_queries.py --executor legacy --output .scratch/linear-legacy.json
uv run --locked --all-extras python scripts/benchmark_queries.py --output .scratch/linear-prepared.json
uv run --locked --all-extras python scripts/benchmark_queries.py --large --scenarios 25 --output .scratch/large-scenarios.json
uv run --locked --all-extras python scripts/benchmark_queries.py --workload panel_policy --simulations 20 --output .scratch/policy.json
uv run --locked --all-extras python scripts/benchmark_queries.py --mode CVM --output .scratch/cvm.json
uv run --locked --all-extras python scripts/benchmark_queries.py --cold-cache --output .scratch/cold.json
uv run --locked --all-extras python scripts/benchmark_backends.py --experiment kernel --workload linear --mode JAX --output .scratch/kernel.json
uv run --locked --all-extras python scripts/benchmark_backends.py --workload panel_ar --large --output .scratch/scan-profile.json
uv run --locked --all-extras python scripts/benchmark_backends.py --experiment fit --workload latent --sampler nutpie-jax --output .scratch/fit.json
```

The query harness runs each workload in a subprocess with fixed synthetic posterior draws and an `observed_data` group matching a fitted model's metadata. Small cross-sectional workloads have 100 rows and 100 posterior samples; large ones have 10,000 rows and 2,000 samples. Panels use 1 unit × 20 periods or 20 units × 200 periods. Fitting probes generate a separate noisy dataset and run two chains with 200 tuning and 200 retained draws; their timings do not infer sampling speed from query execution.

JSON files under `results/` retain the source commit, versions, effective linker, dtype, seed, hardware/platform, thread settings, cache state, first request, five warmed repetitions, median/range, output size, and peak process RSS. Each workload runs serially in its own process. First-request measurements use an existing native disk cache unless explicitly labeled isolated-empty-disk. `--cold-cache` starts with an isolated PyTensor compilation directory; it does not promise an empty operating-system or external compiler cache. ATE's first request can reuse the preceding mean-do plan, so it is not another cold-preparation measurement.

The harness counts actual `pm.do`, PyMC compiler, and `pytensor.function` entries independently of the disk cache. It also times plan preparation, numeric function execution/JIT, xarray assembly/aggregation, and predictive conversion. These are inclusive instrumentation buckets and can overlap; do not sum them. End-to-end times include instrumentation overhead. Peak RSS covers the whole workload process, including imports, model construction, and all query types; it is not a per-query allocation estimate. The `legacy` comparison selects the retained executor on the same source revision.

## Measured warm queries

Measurements below are medians [minimum, maximum] in milliseconds from five warmed requests. Each JSON file records the exact execution commit; result-only and report-only descendants do not change those measured inputs.

| Size | Query | Retained executor (ms) | Prepared route (ms) | Median speedup |
| --- | --- | ---: | ---: | ---: |
| small | `mean_do` | 26.17 [24.96, 93.09] | 1.52 [1.49, 1.89] | 17.20× |
| small | `ate` | 48.73 [47.82, 48.96] | 2.39 [2.34, 2.39] | 20.43× |
| small | `predictive_do` | 35.50 [35.40, 35.96] | 35.49 [35.41, 36.56] | 1.00× |
| small | `predict` | 36.16 [35.27, 36.61] | 9.87 [9.73, 10.02] | 3.66× |
| large | `mean_do` | 102.10 [94.28, 161.62] | 26.76 [26.67, 27.04] | 3.81× |
| large | `ate` | 186.92 [185.96, 191.27] | 55.32 [55.29, 55.43] | 3.38× |
| large | `predictive_do` | 335.01 [331.17, 337.31] | 338.74 [332.11, 363.72] | 0.99× |
| large | `predict` | 329.23 [329.02, 340.01] | 309.22 [304.82, 324.01] | 1.06× |

Predictive `do` is delegated. After preparation, the retained mean and ordinary-prediction plans make zero additional PyTensor/PyMC function compilations or query-graph preparations on these warmed calls; changed values and seeds are covered separately by the numerical/lifecycle tests.

For the small full workload process, peak RSS was 429 MiB with the retained executor and 413 MiB with prepared queries.
For the large full workload process, peak RSS was 1699 MiB with the retained executor and 1349 MiB with prepared queries.


The largest prepared-versus-retained median ratio among the small delegated workloads was 1.081; none exceeded the proposed 1.10 regression threshold in this run. These are machine-specific observations, not timing assertions in pytest.

| Rows / samples | Scenarios | Batched median [range] (ms) | Reused per-draw median [range] (ms) |
| --- | ---: | ---: | ---: |
| 100 / 100 | 1 | 2.07 [2.05, 2.11] | 2.42 [2.38, 2.44] |
| 100 / 100 | 2 | 2.85 [2.85, 2.91] | 3.84 [3.83, 3.87] |
| 100 / 100 | 25 | 17.52 [17.41, 18.49] | 32.91 [32.81, 33.09] |
| 10000 / 2000 | 2 | 55.86 [55.78, 63.70] | 82.27 [82.14, 82.62] |
| 10000 / 2000 | 25 | 675.22 [672.48, 685.00] | 1005.70 [1002.41, 1012.62] |

The batched route is faster than the retained per-draw control on these deterministic workloads and is the selected implementation. Twenty-five scenarios reuse the same prepared graph, with bounded draw/scenario batches. Chunked and unchunked values agree in independent oracle tests.

The isolated-PyTensor-cache first mean query took 4.016 seconds; inspect `linear-cold.json` alongside its construction/preparation/first-execution buckets rather than treating that number as warm latency.

The delegated 20-replication feedback-policy query took 140.11 [139.59, 157.33] ms. `policy-simulations-1.json` explicitly records the existing unsupported one-replication combination.

## Panel scan decision

| Panel workload | Size | Graph/function preparation (s) | Warm logp + gradient median [range] (ms) |
| --- | --- | ---: | ---: |
| panel_lag | small | 0.736 | 0.211 [0.210, 0.247] |
| panel_lag | large | 0.782 | 1.609 [1.591, 1.695] |
| panel_ar | small | 0.908 | 0.247 [0.245, 0.274] |
| panel_ar | large | 0.802 | 1.929 [1.909, 2.436] |
| panel_transform | small | 1.149 | 0.251 [0.250, 0.281] |
| panel_transform | large | 1.049 | 2.042 [1.973, 2.399] |

The optimized graphs already fuse forward scans where possible. The profile separates forward and gradient scans and retains per-Op times; exogenous lag handling and shared transform/state work remain in the existing compiler. On these workloads, warm optimized likelihood/gradient execution is much cheaper than preparing a new function. Retain the current scan implementation and defer specialized recurrence rewrites. Compilation reuse should be evaluated for panels before considering a custom rewrite. No custom Op, global rewrite, finite-convolution replacement of panel adstock, or alternative AR recurrence is introduced.

That decision preserves realized Gaussian carry innovations, observed-history prediction, noise-free legacy mean trajectories, endogenous interventions and releases, initial states, unit boundaries, and the full geometric adstock recursion. The existing scan/intervention, Gaussian-trajectory, mean-carry, initial-state, shared-adstock, scaling, and policy oracle suites remain applicable. No rewrite is proposed, so no new logp/gradient equivalence claim is needed.

## Backend and fitting handoff

The prepared-kernel experiment compiles the same symbolic linear Gaussian graph with C/VM, Numba, and JAX. It measures one bounded deterministic batch and excludes labeled-result assembly. This isolates backend compilation/evaluation; public JAX queries still use the delegated route and should not be ranked against cached end-to-end CPU queries.

| Size | Mode | Draws × scenarios × rows in batch | Preparation (s) | First execution/oracle (s) | Warm kernel median [range] (ms) |
| --- | --- | --- | ---: | ---: | ---: |
| small | CVM | 100 × 2 × 100 | 0.423 | 0.001 | 0.402 [0.392, 0.413] |
| small | NUMBA | 100 × 2 × 100 | 0.243 | 0.130 | 0.032 [0.031, 0.055] |
| small | JAX | 100 × 2 × 100 | 0.449 | 0.043 | 0.056 [0.049, 0.123] |
| large | CVM | 50 × 2 × 10000 | 0.421 | 0.011 | 8.912 [8.482, 8.996] |
| large | NUMBA | 50 × 2 × 10000 | 0.238 | 0.132 | 1.349 [1.303, 1.394] |
| large | JAX | 50 × 2 × 10000 | 0.439 | 0.062 | 0.627 [0.620, 0.666] |

| Fitting workload | Sampler | Total time (s) | Minimum bulk ESS | Maximum R-hat | Divergences | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear | pymc | 0.937 | 265 | 1.02 | 0 | 347 |
| linear | nutpie-numba | 2.758 | 200 | 1.03 | 0 | 413 |
| linear | nutpie-jax | 1.329 | 168 | 1.02 | 0 | 487 |
| linear | numpyro | 2.239 | 213 | 1.01 | 0 | 648 |
| panel_ar | pymc | 3.462 | 484 | 1.02 | 0 | 357 |
| panel_ar | nutpie-numba | 5.234 | 257 | 1.01 | 0 | 464 |
| panel_ar | nutpie-jax | scan shape failure (#85) | — | — | — | 306 |
| panel_ar | numpyro | scan shape failure (#85) | — | — | — | 335 |
| latent | pymc | 8.718 | 24 | 1.11 | 56 | 378 |
| latent | nutpie-numba | 7.020 | 18 | 1.10 | 27 | 490 |
| latent | nutpie-jax | scan shape failure (#85) | — | — | — | 309 |
| latent | numpyro | scan shape failure (#85) | — | — | — | 336 |

These short fitting runs establish feasibility and expose failures; they do not select a default. Latent fits have poor effective sample sizes and substantial divergences, so their raw elapsed times do not establish useful inference speed. Compiler events and total fit time are recorded separately. Per-chain tuning/sampling clocks are reported only when the backend exposes them; unavailable clocks are null. JAX first-call compilation can remain inside the sampler's reported total, so no fabricated compilation/tuning split is supplied.

The current evidence for [#511](https://github.com/pymc-labs/pathmc/issues/511) is a reusable fixed-shape execution layer with measured warm-query benefit; variable rows, new panel shapes, public warmup, and service serialization remain separate work. For [#514](https://github.com/pymc-labs/pathmc/issues/514), the report supplies backend feasibility, diagnostics, timing, and memory without changing sampler selection. Installed PyMC 6.3 already has an automatic sampler selector; its behavior must be considered when revisiting pathmc defaults.

The JAX fitting shape error reproduces the historical report in [#85](https://github.com/pymc-labs/pathmc/issues/85), while the tested forward-query routes work. Sparse latent fitting with nutpie/Numba succeeds on this stack, but the convergence diagnostics are poor. Retain the scalar-beta workaround tracked by [#338](https://github.com/pymc-labs/pathmc/issues/338): upstream [PyMC #8337](https://github.com/pymc-devs/pymc/issues/8337) remains open, and [#8338](https://github.com/pymc-devs/pymc/pull/8338) closed without merging.

## Compatibility limits and follow-up evidence

No dependency requirement changes. Absence of `freeze_model` uses the existing prediction executor. A supplemental PyMC 6.1 / ArviZ 1.1 probe exercises that fallback; the older stack needs `matplotlib.style.core` imported before ArviZ because of an existing import-order problem. PyMC 6.0 itself cannot resolve against the repository's already-declared `pytensor>=3.1.1` requirement, so this change does not claim to repair that existing metadata inconsistency.

PyMC 6.3.2's `compute_deterministics` drops non-default sample labels. Combining its result with arrays retaining the original labels can introduce NaNs through xarray alignment. The prepared mean route retains current sample labels directly; delegated routes still inherit this upstream limitation. Timing baselines use ordinary fitted-posterior sample labels, while the prepared-route correctness tests deliberately use distinct non-default labels and draws. A minimal upstream reproducer is:

```python
import numpy as np
import pymc as pm
import xarray as xr

with pm.Model() as model:
    beta = pm.Normal("beta")
    pm.Deterministic("doubled", 2 * beta)
post = xr.Dataset(
    {"beta": (("chain", "draw"), np.array([[1., 2.], [3., 4.]]))},
    coords={"chain": [7, 11], "draw": [23, 25]},
)
result = pm.compute_deterministics(post, model=model, progressbar=False)
assert np.array_equal(result.chain, post.chain)
```

The assertion fails on the measured stack. This warrants a focused upstream report; no upstream issue was submitted in this pass.
