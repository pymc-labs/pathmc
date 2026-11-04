# Sampling backend comparison

`scripts/benchmark_backends.py` extends the fitting probe introduced in #526 / PR #683 at the same script path. It compares native PyMC NUTS, nutpie/Numba, NumPyro, and nutpie/JAX on the same anchored stochastic latent AR(1) model. Query-cache and deterministic-kernel timings answer different questions and are not sampling benchmarks.

Install all extras, then run outside pytest (the test fixture overrides sampling sizes):

```bash
uv sync --locked --all-extras
uv run python scripts/benchmark_backends.py --smoke --cache warm --output .scratch/sampling-smoke
uv run python scripts/benchmark_backends.py --matrix --output .scratch/sampling-matrix
```

The matrix covers `(units, periods) = (1, 30), (3, 30), (1, 300), (10, 300)`, each with dense measurements and a fixed schedule observing approximately 20% of periods. Data come from a handwritten NumPy recurrence, including an innovation at the first transition. `init_a` precedes that transition. The observation equation is `y ~ 0 + 1*a`, with separate process/measurement noise and regularizing priors identical across routes. Missing measurements remain imputed unknown values; no panel rows are dropped.

Defaults are two sequential chains, 1,000 tuning steps and 1,000 retained draws per chain, three sampler seeds/repetitions, fixed data seed 85, `target_accept=0.95`, CPU/float64, one virtual JAX CPU device and single-thread BLAS/OpenMP/Numba settings. Noise priors are `LogNormal(log(0.2), 0.25)` for process SD and `LogNormal(log(0.1), 0.25)` for measurement SD; these regularize variance identification without changing priors across routes. `--seed` varies the sampler seed across repetitions; `--data-seed` keeps the generated observations identical so frozen JAX data do not invalidate primed compiler keys. Use `--gradient-backend pytensor` to measure nutpie/JAX's default differentiation route instead of this harness's explicit JAX autodifferentiation. `--sampler`, `--units`, `--periods`, `--observed-fraction`, `--cache`, `--target-accept`, and the sampling-size flags permit individual experiments. `--smoke` forces one unit/eight periods, 20 tuning/retained draws and one chain: those results are execution checks, never speed or convergence evidence.

The default `--initialization stable` uses the same constrained prior support point plus 0.01 for every free coordinate, without random jitter. Native PyMC uses `init="adapt_diag"`, NumPyro uses `nuts={"jitter": False, "chain_method": "sequential"}`, and nutpie uses `compile_kwargs={"jitter_rvs": set()}`; all receive the same explicit `initvals`. Preparation is timed separately and included in end-to-end cost. The reports record each initial array's shape/digest and starting AR coefficient (0.61), so large innovation/imputation arrays need not be duplicated in every report. These controls are tested on the locked PyMC 6.3.2 / nutpie 0.16.10 stack. `--initialization default` retains each public fitting route's initializer for comparison.

Each worker runs in a fresh process. **Cold** means empty PyTensor, JAX and Numba compiler-cache directories, not flushed OS filesystem caches. **Warm** means those disk directories are primed by a separately retained repetition `-1`; subsequent repetitions still reconstruct the model and compile process-local functions. This is distinct from reusing a compiled model in a long-lived service. Priming/failure reports stay in the results. Workers run sequentially; do not run CPU-heavy tests alongside a comparison.

Per-run JSON records the source commit/tree and dirty flag, versions, hardware, dtype/devices/thread settings, seed, parameter count, cache locations, sampler/compiler/gradient settings, peak process RSS, system load before/after, diagnostics, and inclusive compiler entry timings. Construction, initialization preparation, public `fit()` (compilation/JIT, tuning, retained sampling and conversion), and pointwise likelihood have separate clocks. JAX postprocessing is synchronized. Tuning/retained per-chain clocks are recorded when exposed; unexposed boundaries remain `null`. Inclusive compiler buckets overlap: subtracting them does not produce pure sampling time. End-to-end time includes construction, initialization preparation, fit and pointwise likelihood and excludes imports, data generation and diagnostic calculation. Shared-machine contention can affect wall time; inspect load and spread before interpreting modest differences.

Diagnostics include all dynamics coefficients, initial states, noise scales, and latent states at the first, middle and last periods. Reports retain per-parameter R-hat and bulk/tail ESS, divergences, minimum ESS per end-to-end second, and peak RSS. A run is eligible for a diagnostic comparison only with at least two chains/500 retained draws, no divergences, maximum R-hat ≤ 1.01 and minimum bulk/tail ESS ≥ 100. Timing medians still include successful poorly mixing runs, explicitly counted; they must not be interpreted as speedups. `summary.json` contains medians/ranges and links to the complete per-run reports. Backend failures remain visible instead of disappearing from the denominator.

## Why initialization is controlled

A completed 48-run baseline on source `709c715` used public initialization defaults: one unit, 30 periods, dense/sparse measurements, four routes, two cache regimes and three sampler seeds, with 1,000 tuning and 1,000 retained draws in each of two chains. All fits executed, but only 28/48 met the predeclared diagnostic criteria. The [summary](results/default-initialization/summary.json) and per-run reports retain every run, including failures to mix. Counts below combine cold and warm regimes (six measured runs per row); priming runs are excluded.

| Observed | Sampler | Diagnostic passes | Divergences (sum) | Maximum R-hat | Minimum bulk ESS |
| --- | --- | ---: | ---: | ---: | ---: |
| 100% | pymc | 4/6 | 2 | 1.009 | 367.1 |
| 100% | nutpie-numba | 4/6 | 2 | 1.004 | 367.8 |
| 100% | numpyro | 2/6 | 2 | 2.234 | 2.5 |
| 100% | nutpie-jax | 4/6 | 0 | 1.011 | 418.1 |
| 20% | pymc | 4/6 | 2 | 1.007 | 281.5 |
| 20% | nutpie-numba | 2/6 | 0 | 2.183 | 2.6 |
| 20% | numpyro | 2/6 | 4 | 1.829 | 2.9 |
| 20% | nutpie-jax | 6/6 | 0 | 1.005 | 269.6 |

A [direct initialization probe](results/default-initialization/initialization-probe.json) through the installed PyMC seed-routing/initialization helpers produced AR starts above 1.5 despite the regularizing `Normal(0.6, 0.1)` prior. Uniform jitter is independent of that prior width. These finite but extremely poor initial targets can dominate short-panel fitting; this is evidence for controlling starts, not a claim that every poor run has the same cause. The main comparison uses a shared interior starting point so initialization instability does not obscure compiler/sampler differences. The initial [HalfNormal-noise pilots](results/pilots/half-normal/summary.json) and [LogNormal/0.9-target pilot](results/pilots/log-normal-target-090/summary.json) also remain available; their poor diagnostics and machine contention make them unsuitable for timing ranks.

The exact unperturbed support point is rejected by tested nutpie even when compiled logp/gradients are finite: a zero gradient in one unused Normal coordinate suffices in the standalone reproducer. Moving that coordinate by 0.01 permits sampling without changing the model. The common 0.01 perturbation handles this initializer limitation identically across routes. This warrants an upstream nutpie/nuts-rs discussion; the pathmc maintainers own filing the ready-to-file reproducer. Searches for initial/zero-gradient nutpie reports found no matching issue; [nutpie #186](https://github.com/pymc-devs/nutpie/issues/186) concerns missing optional dependencies instead.

```bash
uv run python scripts/repro_nutpie_zero_gradient.py                # finite target/gradient, initializer rejects it
uv run python scripts/repro_nutpie_zero_gradient.py --offset 0.01  # executes; not convergence evidence
```

The controlled full workload comparison is still being collected; the baseline above is not a substitute for its larger workloads.

## Compatibility and upstream ownership

The singleton scan regression is tested with Python 3.12.12, PyMC 6.3.2, PyTensor 3.3.6, ArviZ 1.2.0, NumPy 2.4.6, nutpie 0.16.10, NumPyro 0.21.0, and JAX/jaxlib 0.10.1 on macOS arm64 CPU/float64. Tests complete fitting, posterior deterministics and default pointwise likelihood for sparse stochastic AR(1) panels at one and three units. Controls cover dense latent dynamics, observed/exogenous lags, adstock, pooling and non-scan regression. This is bounded compatibility evidence, not support for every model/backend/device combination.

The upstream failure is independent of pathmc and missing values:

```bash
uv run python scripts/repro_single_unit_scan.py                   # reproduces the upstream broadcast failure
uv run python scripts/repro_single_unit_scan.py --declare-shapes  # native/JAX logp and gradients agree
```

On the versions above, PyMC shared-data replacement or nutpie's `freeze_dims_and_data()` specializes an outer sequence from `(?, ?)` to `(8, 1)`, making its unit axis broadcastable after the scan's inner vector has been traced. The failure occurs in `Scan.make_node()`'s outer/inner sequence broadcast check before JAX executes. pathmc declares fixed sequence/carry and per-unit non-sequence shapes before tracing; shared values remain mutable. No matching upstream report was found in the issue-title searches performed for #85. **Ownership:** the pathmc maintainers can file the retained standalone reproducer against PyMC/PyTensor; this change does not claim an upstream fix or create an upstream issue. Remove the local declarations only after both shared-replacement routes, native gradients, mutable data, and queries pass without them at the supported dependency floor. The separate scalar-beta workaround tracked by #338 / PyMC #8337 remains.

The old blanket nutpie/Numba warning about unnamed imputation variables is obsolete for the tested sparse cases. The production shape repair preserves imputed RVs, priors and the unit axis. The benchmark uses its explicitly stated regularizing priors and initialization controls. Sampler defaults/approximation policy belong to #514; broader query performance belongs to #526.
