# Reproducing posterior-query benchmarks

The benchmark generators measure posterior-query latency, compilation reuse, memory, panel scans, and alternative backends. Recorded measurements and their interpretation live in the pull request that produced them; the initial report is in [PR #683](https://github.com/pymc-labs/pathmc/pull/683). Keep generated JSON in ignored scratch space.

## Query latency and compilation reuse

Run from the repository root through the locked environment:

```bash
uv run --locked --all-extras python scripts/benchmark_queries.py --executor legacy --output .scratch/benchmarks/linear-legacy.json
uv run --locked --all-extras python scripts/benchmark_queries.py --output .scratch/benchmarks/linear-prepared.json
uv run --locked --all-extras python scripts/benchmark_queries.py --large --scenarios 25 --output .scratch/benchmarks/large-scenarios.json
uv run --locked --all-extras python scripts/benchmark_queries.py --workload panel_policy --simulations 20 --output .scratch/benchmarks/policy.json
uv run --locked --all-extras python scripts/benchmark_queries.py --cold-cache --output .scratch/benchmarks/cold.json
```

The `legacy` comparison selects the retained executor on the same revision. Each workload runs in an isolated subprocess with fixed synthetic posterior draws; linear fixtures include fitted response means and constant-data metadata. Query measurements exclude MCMC. The first request and at least five warmed requests are recorded separately, with medians and ranges. Keep large timing runs outside pytest and execute comparisons serially.

| Option | Coverage |
| --- | --- |
| `--workload` | `linear`, `bernoulli`, `poisson`, `panel_lag`, `panel_ar`, `panel_transform`, `panel_policy` |
| `--large` | Cross-sectional rows/samples: 100/100 → 10,000/2,000; panel units/periods: 1/20 → 20/200 |
| `--scenarios` | 1, 2, or 25 internal deterministic intervention scenarios |
| `--simulations` | 1 or 20 policy replications; the existing API rejects one replication and the report records it as unsupported |
| `--mode` | `FAST_RUN`, `CVM`, `VM`, `NUMBA`, or `JAX` |
| `--repetitions` | At least five warmed requests; default five |

Run the small and large workload pairs, retained/prepared comparisons, scenario/replication variants, and explicit compiler modes to regenerate the query matrix. The generators record unsupported combinations and failures alongside successful cases.

## Panel profiles and backend experiments

```bash
uv run --locked --all-extras python scripts/benchmark_backends.py --workload panel_ar --large --output .scratch/benchmarks/scan-profile.json
uv run --locked --all-extras python scripts/benchmark_backends.py --experiment kernel --workload linear --mode JAX --output .scratch/benchmarks/kernel.json
uv run --locked --all-extras python scripts/benchmark_backends.py --experiment fit --workload latent --sampler nutpie-jax --output .scratch/benchmarks/fit.json
```

The default `profile` experiment evaluates the optimized likelihood and gradient for `panel_lag`, `panel_ar`, or `panel_transform`, with small/large sizes. Use `CVM` for per-Op profiling. The `kernel` experiment compares the same prepared linear deterministic batch under `CVM`, `NUMBA`, and `JAX`; labeled-result assembly is excluded. A direct kernel probe can exercise a backend that public query caching delegates.

The `fit` experiment uses separate noisy datasets for `linear`, `panel_ar`, and `latent`, with explicit `pymc`, `nutpie-numba`, `nutpie-jax`, and `numpyro` sampler choices. Defaults are two chains with 200 tuning and 200 retained draws. Compare diagnostics alongside time: ESS per total second, R-hat, divergences, and peak RSS. Missing backend clocks are recorded as null. Query compiler measurements and sampler measurements answer different questions.

## Reading generated reports

JSON reports record the source commit, dirty state, dependency versions, effective linker, dtype, seed, platform, thread settings, cache state, output size, and peak process RSS. Compiler and graph-construction counters measure actual entry calls independently of the native disk cache. Function execution, graph preparation, compilation, conversion, and aggregation are inclusive timing buckets that can overlap; do not sum them.

First requests normally use an existing native disk cache. `--cold-cache` isolates the PyTensor compilation directory without promising an empty operating-system or external compiler cache. ATE can reuse a preceding mean-query plan within a workload process. Peak RSS covers imports, model construction, and every query in that process, rather than one query's allocation.

Check each report's status and independent oracle results before comparing timing. Publish the relevant measurements, environment, capability limits, and decisions on the PR; regenerate JSON locally when more detail is needed. The full result directory `benchmarks/results/` is also ignored to prevent accidental snapshot commits.
