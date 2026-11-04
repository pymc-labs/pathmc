# Sampling backend comparison

`scripts/benchmark_backends.py` extends the fitting probe introduced in #526 / PR #683 at the same script path. It compares native PyMC NUTS, nutpie/Numba, NumPyro, and nutpie/JAX on the same anchored stochastic latent AR(1) model. Query-cache and deterministic-kernel timings answer different questions and are not sampling benchmarks.

Install all extras, then run outside pytest (the test fixture overrides sampling sizes):

```bash
uv sync --locked --all-extras
uv run python scripts/benchmark_backends.py --smoke --cache warm --output .scratch/sampling-smoke
uv run python scripts/benchmark_backends.py --matrix --output .scratch/sampling-matrix
uv run python scripts/benchmark_backends.py --summarize-only --output .scratch/sampling-matrix
```

The matrix covers `(units, periods) = (1, 30), (3, 30), (1, 300), (10, 300)`, each with dense measurements and a fixed schedule observing approximately 20% of periods. Data come from a handwritten NumPy recurrence, including an innovation at the first transition. `init_a` precedes that transition. The observation equation is `y ~ 0 + 1*a`, with separate process/measurement noise and regularizing priors identical across routes. Missing measurements remain imputed unknown values; no panel rows are dropped.

Defaults are two sequential chains, 1,000 tuning steps and 1,000 retained draws per chain, three sampler seeds/repetitions, fixed data seed 85, `target_accept=0.95`, CPU/float64, one virtual JAX CPU device and single-thread BLAS/OpenMP/Numba settings. Noise priors are `LogNormal(log(0.2), 0.25)` for process SD and `LogNormal(log(0.1), 0.25)` for measurement SD; these regularize variance identification without changing priors across routes. `--seed` varies the sampler seed across repetitions; `--data-seed` keeps the generated observations identical so frozen JAX data do not invalidate primed compiler keys. Use `--gradient-backend pytensor` to measure nutpie/JAX's default differentiation route instead of this harness's explicit JAX autodifferentiation. `--sampler`, `--units`, `--periods`, `--observed-fraction`, `--cache`, `--target-accept`, and the sampling-size flags permit individual experiments. `--smoke` forces one unit/eight periods, 20 tuning/retained draws and one chain: those results are execution checks, never speed or convergence evidence.

The default `--initialization stable` uses the same constrained prior support point plus 0.01 for every free coordinate, without random jitter. Native PyMC uses `init="adapt_diag"`, NumPyro uses `nuts={"jitter": False, "chain_method": "sequential"}`, and nutpie uses `compile_kwargs={"jitter_rvs": set()}`; all receive the same explicit `initvals`. Both chains start at this common point. The diagnostic checks describe this controlled experiment; assess sensitivity to dispersed starting points separately for substantive models. Preparation is timed separately and included in end-to-end cost. The reports record each initial array's shape/digest and starting AR coefficient (0.61), so large innovation/imputation arrays need not be duplicated in every report. These controls are tested on the locked PyMC 6.3.2 / nutpie 0.16.10 stack. `--initialization default` retains each public fitting route's initializer for comparison.

Each worker runs in a fresh process. **Cold** means empty PyTensor, JAX and Numba compiler-cache directories, not flushed OS filesystem caches. **Warm** means those disk directories are primed by a separately retained repetition `-1`; subsequent repetitions still reconstruct the model and compile process-local functions. This is distinct from reusing a compiled model in a long-lived service. Priming/failure reports stay in the results. Workers run sequentially; do not run CPU-heavy tests alongside a comparison.

Per-run JSON records the source commit/tree and dirty flag, versions, hardware, dtype/devices/thread settings, seed, parameter count, cache locations, sampler/compiler/gradient settings, peak process RSS, system load before/after, diagnostics, and inclusive compiler entry timings. Construction, initialization preparation, public `fit()` (compilation/JIT, tuning, retained sampling and conversion), and pointwise likelihood have separate clocks. JAX postprocessing is synchronized. Tuning/retained per-chain clocks are recorded when exposed; unexposed boundaries remain `null`. Inclusive compiler buckets overlap: subtracting them does not produce pure sampling time. End-to-end time includes construction, initialization preparation, fit and pointwise likelihood and excludes imports, data generation and diagnostic calculation. Shared-machine contention can affect wall time; inspect load and spread before interpreting modest differences.

Diagnostics include all dynamics coefficients, initial states, noise scales, and latent states at the first, middle and last periods. Reports retain per-parameter R-hat and bulk/tail ESS, divergences, minimum ESS per end-to-end second, and peak RSS. A run is eligible for a diagnostic comparison only with at least two chains/500 retained draws, no divergences, maximum R-hat ≤ 1.01 and minimum bulk/tail ESS ≥ 100. Timing medians still include successful poorly mixing runs, explicitly counted; they must not be interpreted as speedups. `summary.json` contains medians/ranges and links to the complete per-run reports. Backend failures remain visible instead of disappearing from the denominator.

## Controlled workload comparison

The [full summary](results/controlled/summary.json) retains 192 measured runs and 32 priming fits from clean source `b1c0423`. There were 0 execution failures and 54/192 diagnostic passes. The data were collected on 4 November 2026 on an Apple M1 Max, 10 logical CPUs, 64 GiB RAM, macOS 27.2 arm64, using the dependency versions listed below (Numba 0.65.1). CSV rates and tables were derived after collection from the recorded ESS and clocks; the raw worker JSON remains unchanged.

For the **one-unit, 30-period, 20%-observed fixture**, all six measured runs per route pass the criteria. NumPyro has the shortest median fitting times: 13.5s cold and 8.1s warm. Its warm median minimum monitored bulk ESS/s is 50.9, compared with 29.2 for native PyMC. Its warm peak-RSS median is 0.92 GiB versus 0.45 GiB. This makes NumPyro a useful candidate for this small sparse CPU workload when its extra memory fits the application. In the one-unit dense 30-period case, nutpie/Numba is the only route with all six runs passing; the other routes need further diagnostic work before an efficiency comparison.

**At 300 periods, 0/96 completed fits pass the criteria.** The lowest monitored bulk ESS belongs to `sigma_y` in 96/96 of those runs. Shorter elapsed times there do not establish an inference-speed improvement. Inspect measurement-noise mixing before comparing routes on these larger panels; these measurements leave their useful-inference ranking unresolved.

Repeated fresh-process fits amortize setup in the healthy small sparse NumPyro case. Its full-length primer costs 12.6s on the fitting clock. On the recorded worker-experiment clock, including imports and diagnostics, the primer costs 15.2s and cold/warm medians are 15.9/10.2s: a deliberately discarded primer needs roughly 3 subsequent fits to repay its cost under these measurements. A useful first posterior fit can itself prime the disk cache. This calculation depends on process lifetime, model/data cache keys and the measured workload.

Workers ran sequentially in a fixed route order on a shared host. Recorded one-minute system load ranged from 2.9 to 91.9 on 10 logical CPUs; that includes experiment and background activity. Treat modest timing differences as descriptive. Cold/warm refers to compiler directories, not OS caches. Peak RSS includes worker imports. The table shows peak-RSS medians; the summary contains their min/max and every raw result. The [controlled-start pilot](results/pilots/stable-protocol/summary.json) is retained separately because validation probes could run alongside it; it is excluded from these timing medians.

### Elapsed fitting time and memory

Times include construction, initialization preparation, fit/conversion and pointwise likelihood. Each cold/warm cell contains three measured seeds; passes use the unrounded diagnostic criteria above. Every successful run contributes to the displayed timings, including ineligible runs. Read the pass counts before comparing efficiency.

| Units × periods | Observed | Route | Cold seconds, median (range) | Warm seconds, median (range) | Passes cold / warm | Peak RSS GiB, cold / warm medians |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 × 30 | 100% | pymc | 11.4 (10.2–12.0) | 8.9 (8.5–9.6) | 2/3 / 2/3 | 0.44 / 0.44 |
| 1 × 30 | 100% | nutpie-numba | 8.0 (7.9–8.3) | 7.9 (7.3–8.0) | 3/3 / 3/3 | 0.53 / 0.53 |
| 1 × 30 | 100% | numpyro | 6.9 (5.9–7.2) | 3.9 (3.9–4.5) | 1/3 / 1/3 | 0.88 / 0.83 |
| 1 × 30 | 100% | nutpie-jax | 8.0 (7.8–8.2) | 8.8 (7.2–10.3) | 2/3 / 2/3 | 0.51 / 0.46 |
| 1 × 30 | 20% | pymc | 18.0 (17.7–19.1) | 13.9 (13.9–14.1) | 3/3 / 3/3 | 0.45 / 0.45 |
| 1 × 30 | 20% | nutpie-numba | 15.4 (15.0–15.9) | 12.3 (11.7–12.8) | 3/3 / 3/3 | 0.54 / 0.54 |
| 1 × 30 | 20% | numpyro | 13.5 (13.2–13.7) | 8.1 (7.7–8.3) | 3/3 / 3/3 | 0.98 / 0.92 |
| 1 × 30 | 20% | nutpie-jax | 17.3 (16.9–18.1) | 15.8 (14.2–16.6) | 3/3 / 3/3 | 0.53 / 0.47 |
| 3 × 30 | 100% | pymc | 13.5 (13.2–15.8) | 11.8 (11.8–12.0) | 0/3 / 0/3 | 0.46 / 0.45 |
| 3 × 30 | 100% | nutpie-numba | 10.3 (9.9–13.7) | 9.2 (9.0–10.1) | 1/3 / 2/3 | 0.54 / 0.54 |
| 3 × 30 | 100% | numpyro | 7.5 (7.4–9.1) | 4.7 (4.6–4.9) | 1/3 / 1/3 | 0.88 / 0.82 |
| 3 × 30 | 100% | nutpie-jax | 8.7 (8.2–10.1) | 13.6 (12.2–15.2) | 0/3 / 0/3 | 0.53 / 0.48 |
| 3 × 30 | 20% | pymc | 28.4 (27.8–35.3) | 23.3 (23.1–24.8) | 0/3 / 0/3 | 0.47 / 0.47 |
| 3 × 30 | 20% | nutpie-numba | 21.0 (19.6–25.1) | 17.5 (16.7–18.1) | 3/3 / 2/3 | 0.56 / 0.56 |
| 3 × 30 | 20% | numpyro | 14.9 (14.7–15.7) | 8.6 (8.5–8.9) | 0/3 / 0/3 | 0.97 / 0.91 |
| 3 × 30 | 20% | nutpie-jax | 39.0 (35.3–39.6) | 31.7 (30.0–36.0) | 2/3 / 2/3 | 0.55 / 0.49 |
| 1 × 300 | 100% | pymc | 53.0 (52.8–61.1) | 54.7 (51.6–54.9) | 0/3 / 0/3 | 0.49 / 0.49 |
| 1 × 300 | 100% | nutpie-numba | 30.0 (28.0–41.7) | 32.3 (29.8–36.5) | 0/3 / 0/3 | 0.58 / 0.58 |
| 1 × 300 | 100% | numpyro | 10.3 (9.6–11.2) | 9.2 (7.7–9.9) | 0/3 / 0/3 | 0.93 / 0.87 |
| 1 × 300 | 100% | nutpie-jax | 15.4 (15.2–17.7) | 12.9 (12.4–14.0) | 0/3 / 0/3 | 0.57 / 0.51 |
| 1 × 300 | 20% | pymc | 125.9 (123.8–132.8) | 118.7 (118.4–137.8) | 0/3 / 0/3 | 0.53 / 0.53 |
| 1 × 300 | 20% | nutpie-numba | 108.0 (105.9–110.4) | 100.2 (81.8–102.8) | 0/3 / 0/3 | 0.62 / 0.62 |
| 1 × 300 | 20% | numpyro | 22.7 (21.6–24.9) | 15.6 (15.6–15.9) | 0/3 / 0/3 | 1.04 / 0.96 |
| 1 × 300 | 20% | nutpie-jax | 42.2 (42.1–45.3) | 41.5 (40.9–46.9) | 0/3 / 0/3 | 0.62 / 0.55 |
| 10 × 300 | 100% | pymc | 84.7 (84.6–99.2) | 84.4 (75.0–85.7) | 0/3 / 0/3 | 0.98 / 0.98 |
| 10 × 300 | 100% | nutpie-numba | 80.2 (78.7–85.2) | 76.0 (67.2–80.5) | 0/3 / 0/3 | 1.07 / 1.08 |
| 10 × 300 | 100% | numpyro | 70.3 (62.8–70.5) | 65.1 (60.7–65.2) | 0/3 / 0/3 | 1.27 / 1.19 |
| 10 × 300 | 100% | nutpie-jax | 47.5 (45.7–47.9) | 47.2 (46.2–52.1) | 0/3 / 0/3 | 1.07 / 1.01 |
| 10 × 300 | 20% | pymc | 93.6 (89.5–139.8) | 91.3 (89.6–93.1) | 0/3 / 0/3 | 1.34 / 1.32 |
| 10 × 300 | 20% | nutpie-numba | 271.7 (203.7–283.6) | 266.5 (237.6–271.0) | 0/3 / 0/3 | 1.41 / 1.42 |
| 10 × 300 | 20% | numpyro | 160.5 (159.7–204.4) | 153.4 (147.6–186.9) | 0/3 / 0/3 | 1.45 / 1.36 |
| 10 × 300 | 20% | nutpie-jax | 174.7 (148.7–233.2) | 168.0 (142.8–237.2) | 0/3 / 0/3 | 1.41 / 1.34 |

### Diagnostic extremes and efficiency

Divergences are summed and R-hat/ESS extremes span all six measured runs per row. ESS/s columns are medians of each run's minimum across the monitored coefficients, initial states, noise scales and selected latent states; they use the complete fitting clock. Displayed values are rounded; eligibility uses unrounded values. Rows with poor diagnostics provide descriptive rates, not validated inference-speed rankings.

| Units × periods | Observed | Route | Divergences, sum | Maximum R-hat | Minimum bulk / tail ESS | Minimum bulk ESS/s, cold / warm medians | Minimum tail ESS/s, cold / warm medians |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 30 | 100% | pymc | 4 | 1.0057 | 336.5 / 526.5 | 37.05 / 49.71 | 65.44 / 79.06 |
| 1 × 30 | 100% | nutpie-numba | 0 | 1.0094 | 279.6 / 522.4 | 52.67 / 52.20 | 65.88 / 72.16 |
| 1 × 30 | 100% | numpyro | 4 | 1.0134 | 384.8 / 518.4 | 57.87 / 97.53 | 77.42 / 140.71 |
| 1 × 30 | 100% | nutpie-jax | 0 | 1.0111 | 331.0 / 587.4 | 47.42 / 37.81 | 75.17 / 68.88 |
| 1 × 30 | 20% | pymc | 0 | 1.0067 | 395.3 / 647.2 | 22.36 / 29.21 | 36.60 / 48.86 |
| 1 × 30 | 20% | nutpie-numba | 0 | 1.0085 | 342.2 / 517.8 | 22.75 / 29.34 | 34.46 / 44.43 |
| 1 × 30 | 20% | numpyro | 0 | 1.0060 | 364.2 / 587.1 | 30.57 / 50.85 | 44.42 / 76.57 |
| 1 × 30 | 20% | nutpie-jax | 0 | 1.0047 | 290.8 / 457.0 | 17.99 / 21.85 | 36.56 / 39.81 |
| 3 × 30 | 100% | pymc | 0 | 1.0719 | 35.9 / 118.4 | 4.83 / 6.45 | 9.75 / 13.03 |
| 3 × 30 | 100% | nutpie-numba | 0 | 1.0162 | 125.6 / 181.6 | 12.72 / 15.45 | 19.28 / 22.13 |
| 3 × 30 | 100% | numpyro | 0 | 1.0576 | 70.3 / 172.9 | 16.81 / 31.43 | 27.00 / 50.57 |
| 3 × 30 | 100% | nutpie-jax | 0 | 1.0283 | 89.3 / 142.9 | 10.27 / 6.63 | 18.83 / 12.55 |
| 3 × 30 | 20% | pymc | 4 | 1.0609 | 74.6 / 165.9 | 3.59 / 4.40 | 6.09 / 8.76 |
| 3 × 30 | 20% | nutpie-numba | 0 | 1.0155 | 114.1 / 189.9 | 5.82 / 7.91 | 9.04 / 11.37 |
| 3 × 30 | 20% | numpyro | 0 | 1.0407 | 72.3 / 88.6 | 5.02 / 9.15 | 8.37 / 15.26 |
| 3 × 30 | 20% | nutpie-jax | 0 | 1.0258 | 100.1 / 221.1 | 3.21 / 3.53 | 6.62 / 7.28 |
| 1 × 300 | 100% | pymc | 0 | 1.1312 | 12.6 / 34.0 | 1.10 / 1.13 | 2.08 / 2.01 |
| 1 × 300 | 100% | nutpie-numba | 0 | 1.0536 | 47.6 / 68.7 | 2.09 / 1.96 | 3.28 / 3.08 |
| 1 × 300 | 100% | numpyro | 0 | 1.0351 | 59.0 / 68.2 | 7.80 / 7.54 | 10.75 / 10.39 |
| 1 × 300 | 100% | nutpie-jax | 0 | 1.0536 | 47.6 / 68.7 | 3.81 / 4.71 | 5.97 / 7.39 |
| 1 × 300 | 20% | pymc | 0 | 1.0897 | 21.5 / 36.0 | 0.42 / 0.44 | 0.59 / 0.62 |
| 1 × 300 | 20% | nutpie-numba | 0 | 1.1210 | 13.0 / 39.0 | 0.30 / 0.33 | 0.63 / 0.65 |
| 1 × 300 | 20% | numpyro | 10 | 1.1403 | 13.9 / 35.8 | 1.26 / 1.82 | 1.62 / 2.58 |
| 1 × 300 | 20% | nutpie-jax | 0 | 1.0372 | 29.5 / 50.7 | 0.81 / 0.78 | 1.38 / 1.40 |
| 10 × 300 | 100% | pymc | 0 | 1.1131 | 17.4 / 35.9 | 0.33 / 0.32 | 0.49 / 0.49 |
| 10 × 300 | 100% | nutpie-numba | 0 | 1.3820 | 5.4 / 7.5 | 0.19 / 0.24 | 0.44 / 0.55 |
| 10 × 300 | 100% | numpyro | 0 | 1.1547 | 11.7 / 14.6 | 0.29 / 0.31 | 0.37 / 0.40 |
| 10 × 300 | 100% | nutpie-jax | 0 | 1.3820 | 5.4 / 7.5 | 0.34 / 0.31 | 0.78 / 0.72 |
| 10 × 300 | 20% | pymc | 68 | 1.6221 | 3.3 / 7.2 | 0.04 / 0.05 | 0.10 / 0.09 |
| 10 × 300 | 20% | nutpie-numba | 0 | 1.5264 | 3.6 / 5.4 | 0.02 / 0.02 | 0.05 / 0.06 |
| 10 × 300 | 20% | numpyro | 0 | 1.4645 | 4.1 / 6.0 | 0.03 / 0.03 | 0.06 / 0.07 |
| 10 × 300 | 20% | nutpie-jax | 0 | 1.8604 | 2.9 / 13.3 | 0.03 / 0.04 | 0.09 / 0.10 |

The [per-parameter CSV](results/controlled/parameter-diagnostics.csv) gives bulk/tail ESS/s and R-hat for each dynamics parameter and each selected latent state in every run. Filter `repetition >= 0` for measured runs, then inspect the eligibility flag and run-level diagnostics. Priming and failure rows remain present; empty metric cells mean unavailable values. It is derived directly from the unchanged per-run ESS divided by recorded end-to-end fitting seconds. Regenerate summaries and this CSV from saved reports with `--summarize-only`.

Actual compiler traces show [native PyMC's Numba linker](results/controlled/u10-t300-obs1-pymc-cold-0.json) and [nutpie/JAX's JAX linker and JAX differentiation](results/controlled/u10-t300-obs1-nutpie-jax-cold-0.json). Compiler-entry clocks include graph preparation and overlap other buckets; lazy machine compilation may occur during fitting. They are partial measurements, and subtraction cannot recover exclusive JIT, tuning or retained-sampling intervals.

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

## Compatibility and upstream ownership

The singleton scan regression is tested with Python 3.12.12, PyMC 6.3.2, PyTensor 3.3.6, ArviZ 1.2.0, NumPy 2.4.6, nutpie 0.16.10, NumPyro 0.21.0, and JAX/jaxlib 0.10.1 on macOS arm64 CPU/float64. Tests complete fitting, posterior deterministics and default pointwise likelihood for sparse stochastic AR(1) panels at one and three units. Controls cover dense latent dynamics, observed/exogenous lags, adstock, pooling and non-scan regression. This is bounded compatibility evidence, not support for every model/backend/device combination.

The upstream failure is independent of pathmc and missing values:

```bash
uv run python scripts/repro_single_unit_scan.py                   # reproduces the upstream broadcast failure
uv run python scripts/repro_single_unit_scan.py --declare-shapes  # native/JAX logp and gradients agree
```

On the versions above, PyMC shared-data replacement or nutpie's `freeze_dims_and_data()` specializes an outer sequence from `(?, ?)` to `(8, 1)`, making its unit axis broadcastable after the scan's inner vector has been traced. The failure occurs in `Scan.make_node()`'s outer/inner sequence broadcast check before JAX executes. pathmc declares fixed sequence/carry and per-unit non-sequence shapes before tracing; shared values remain mutable. No matching upstream report was found in the issue-title searches performed for #85. **Ownership:** the pathmc maintainers can file the retained standalone reproducer against PyMC/PyTensor; this change does not claim an upstream fix or create an upstream issue. Remove the local declarations only after both shared-replacement routes, native gradients, mutable data, and queries pass without them at the supported dependency floor. The separate scalar-beta workaround tracked by #338 / PyMC #8337 remains.

The old blanket nutpie/Numba warning about unnamed imputation variables is obsolete for the tested sparse cases. The production shape repair preserves imputed RVs, priors and the unit axis. The benchmark uses its explicitly stated regularizing priors and initialization controls. Sampler defaults/approximation policy belong to #514; broader query performance belongs to #526.
