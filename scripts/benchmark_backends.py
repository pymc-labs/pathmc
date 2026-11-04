#   Copyright 2025 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Compare exact sampler routes for dense/sparse stochastic latent AR(1) panels.

Extends the fitting probe developed for #526 / PR #683 at this same script path.
Query latency and scan-kernel experiments are separate from these fitting times.
Run outside pytest (whose fixture overrides the requested sampling sizes).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from importlib.metadata import PackageNotFoundError, version
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any
from unittest.mock import patch

import numpy as np


def environment() -> dict[str, Any]:
    """Describe the source, hardware, dependencies and actual compiler settings."""
    import jax
    import pytensor

    versions = {}
    for name in (
        "pathmc",
        "pymc",
        "pytensor",
        "arviz",
        "numpy",
        "nutpie",
        "numba",
        "numpyro",
        "jax",
        "jaxlib",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None

    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version,
        "versions": versions,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if sys.platform == "darwin"
        else platform.processor(),
        "logical_cpus": os.cpu_count(),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        "floatX": pytensor.config.floatX,
        "pytensor_mode": str(pytensor.config.mode),
        "compiledir": str(pytensor.config.compiledir),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "jax_enable_x64": jax.config.jax_enable_x64,
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    }


@contextmanager
def instrumentation() -> Iterator[dict[str, dict[str, float | int]]]:
    """Time real compiler/postprocessing entry points; inclusive buckets overlap."""
    import jax
    import pymc as pm
    import pytensor
    import pymc.sampling.jax as sampling_jax

    events: dict[str, dict[str, float | int]] = {}

    def measured(name, original):
        def call(*args, **kwargs):
            start = perf_counter()
            try:
                result = original(*args, **kwargs)
                # JAX dispatch is asynchronous, including deterministic expansion.
                if name == "jax_posterior_postprocessing":
                    jax.block_until_ready(result)
                return result
            finally:
                event = events.setdefault(name, {"calls": 0, "seconds": 0.0})
                event["calls"] += 1
                event["seconds"] += perf_counter() - start

        return call

    with ExitStack() as stack:
        entries = [
            (pytensor, "function", "pytensor_function_compilation"),
            (sampling_jax, "get_jaxified_graph", "jax_graph_conversion"),
            (sampling_jax, "_postprocess_samples", "jax_posterior_postprocessing"),
            (pm, "compute_deterministics", "pymc_deterministic_postprocessing"),
        ]
        try:
            import nutpie
        except ImportError:
            pass
        else:
            entries.append((nutpie, "compile_pymc_model", "nutpie_compilation"))
        for owner, name, label in entries:
            if hasattr(owner, name):
                stack.enter_context(
                    patch.object(owner, name, measured(label, getattr(owner, name)))
                )
        yield events


def fixture(units: int, periods: int, observed_fraction: float, seed: int):
    """Generate data with an independent NumPy recurrence and anchored scale."""
    import pandas as pd

    import pathmc
    from pathmc.priors import Prior

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(units):
        state = 0.2 + 0.1 * unit
        for time in range(periods):
            x = rng.normal()
            state = 0.1 + 0.6 * state + 0.25 * x + rng.normal(scale=0.2)
            y = state + rng.normal(scale=0.1)
            # Fixed schedule, identical across routes and repetitions; NaNs are
            # unknown measurements, not zeros or missing panel rows.
            observed = observed_fraction == 1.0 or time % 5 == 1
            rows.append({
                "unit": unit,
                "time": time,
                "x": x,
                "y": y if observed else np.nan,
            })
    frame = pd.DataFrame(rows)
    start = perf_counter()
    fitted = pathmc.model(
        "a ~ lag(a) + x\ny ~ 0 + 1*a",
        data=frame,
        panel={"unit": "unit", "time": "time"},
        latent=["a"],
        families={"a": "latent_normal"},
        priors={
            "beta_a": Prior("Normal", mu=[0.1, 0.6, 0.25], sigma=[0.2, 0.1, 0.2]),
            "sigma_a": Prior("HalfNormal", sigma=0.35),
            "sigma_y": Prior("HalfNormal", sigma=0.2),
            "init_a": Prior("Normal", mu=0.2 + 0.1 * np.arange(units), sigma=0.3),
        },
    )
    return fitted, perf_counter() - start, int(frame.y.notna().sum())


def fit_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Fit identical models through the public API and keep failed diagnostics."""
    import arviz as az
    import jax
    import pymc as pm
    import xarray as xr

    jax.config.update("jax_enable_x64", True)
    fitted, construction, observed = fixture(
        args.units, args.periods, args.observed_fraction, args.data_seed
    )
    sampler = "nutpie" if args.sampler.startswith("nutpie") else args.sampler
    options: dict[str, Any] = {}
    if sampler == "nutpie":
        options.update(backend=args.sampler.split("-")[1])
        if args.sampler == "nutpie-jax":
            options["compile_kwargs"] = {"gradient_backend": args.gradient_backend}
    elif sampler == "numpyro":
        options["nuts"] = {"chain_method": "sequential"}
    settings = dict(
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=1,
        nuts_sampler=sampler,
        random_seed=args.seed,
        progressbar=False,
        target_accept=0.9,
        compute_convergence_checks=False,
        discard_tuned_samples=False,
        idata_kwargs={"save_warmup": True},
        **options,
    )
    start = perf_counter()
    with instrumentation() as events:
        idata = fitted.fit(compute_log_likelihood=False, **settings)
        # Force host materialization before stopping the end-to-end timer.
        for variable in idata.posterior.to_dataset().data_vars.values():
            np.asarray(variable)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    with fitted.pymc_model:
        pm.compute_log_likelihood(idata, progressbar=False)
    loglik_seconds = perf_counter() - start
    post = idata.posterior.to_dataset()
    selected = post.a.isel({
        post.a.dims[-2]: sorted({0, args.periods // 2, args.periods - 1})
    })
    diagnostic_data = xr.Dataset({
        name: post[name] for name in ("beta_a", "sigma_a", "sigma_y", "init_a")
    })
    diagnostic_data["selected_latent_a"] = selected
    summary = az.summary(diagnostic_data, round_to="none")
    stats = idata.sample_stats.to_dataset()

    def clock(group):
        if group not in idata:
            return None
        dataset = idata[group].to_dataset()
        return (
            float(dataset.perf_counter_diff.sum())
            if "perf_counter_diff" in dataset
            else None
        )

    divergences = int(stats.diverging.sum())
    bulk = float(np.min(summary.ess_bulk.to_numpy()))
    tail = float(np.min(summary.ess_tail.to_numpy()))
    rhat = float(np.max(summary.r_hat.to_numpy()))
    total = construction + fit_seconds + loglik_seconds
    return {
        "status": "ok",
        "model_construction_seconds": construction,
        "fit_including_compilation_tuning_sampling_and_conversion_seconds": fit_seconds,
        "pointwise_log_likelihood_seconds": loglik_seconds,
        "end_to_end_seconds": total,
        "compiler_and_postprocessing_events_inclusive": events,
        "tuning_sum_of_chain_clocks_seconds": clock("warmup_sample_stats"),
        "retained_sum_of_chain_clocks_seconds": clock("sample_stats"),
        "clock_note": "Inclusive compiler buckets overlap. JIT, tuning, sampling and deterministic expansion are not separately exposed on every backend; absent boundaries are null, never inferred by subtraction. Chain clocks are sums, not elapsed wall time. End-to-end excludes data generation/imports/diagnostic calculation and includes construction, fit and pointwise likelihood.",
        "sampler_settings": settings,
        "gradient_backend": args.gradient_backend
        if args.sampler == "nutpie-jax"
        else ("jax" if sampler == "numpyro" else "pytensor"),
        "free_parameter_count": sum(
            np.size(value) for value in fitted.pymc_model.initial_point().values()
        ),
        "observed_measurements": observed,
        "imputed_measurements": args.units * args.periods - observed,
        "divergences": divergences,
        "maximum_rhat": rhat,
        "minimum_bulk_ess": bulk,
        "minimum_tail_ess": tail,
        "minimum_bulk_ess_per_end_to_end_second": bulk / total,
        "minimum_tail_ess_per_end_to_end_second": tail / total,
        "diagnostics": summary.to_dict(orient="index"),
        "diagnostics_acceptable_for_comparison": args.chains >= 2
        and args.draws >= 500
        and divergences == 0
        and np.isfinite(rhat)
        and rhat <= 1.01
        and bulk >= 100
        and tail >= 100,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def worker(args: argparse.Namespace) -> None:
    """Write a run even on backend failure; subprocess exit never hides failure."""
    started = perf_counter()
    try:
        report = fit_probe(args)
    except Exception as error:
        import traceback

        report = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(
        environment=environment(),
        units=args.units,
        periods=args.periods,
        observed_fraction=args.observed_fraction,
        sampler=args.sampler,
        seed=args.seed,
        data_seed=args.data_seed,
        cache=args.cache,
        repetition=args.repetition,
        smoke=args.smoke,
        process_experiment_seconds=perf_counter() - started,
        peak_process_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
    )
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )


def main() -> None:
    """Run isolated sequential workers with explicit cold versus warm disk caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for per-run JSON and summary.json",
    )
    parser.add_argument(
        "--sampler",
        choices=["all", "pymc", "nutpie-numba", "numpyro", "nutpie-jax"],
        default="all",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="1/3 x 30 and 1/10 x 300, dense and 20% observed",
    )
    parser.add_argument("--units", type=int, default=1)
    parser.add_argument("--periods", type=int, default=30)
    parser.add_argument(
        "--observed-fraction", type=float, choices=[0.2, 1.0], default=0.2
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=85,
        help="Fixed across repetitions so warm JAX graphs retain identical embedded data",
    )
    parser.add_argument("--cache", choices=["cold", "warm", "both"], default="both")
    parser.add_argument(
        "--gradient-backend", choices=["jax", "pytensor"], default="jax"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Execution check only: 8 periods, 20 draws/tune, one chain",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repetition", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            args.units,
            args.periods,
            args.draws,
            args.tune,
            args.chains,
            args.repetitions,
        )
        < 1
    ):
        parser.error(
            "Sizes must be positive; supply at least one unit, period, draw, tuning step, chain and repetition."
        )
    if args.worker:
        worker(args)
        return
    if args.smoke:
        (
            args.units,
            args.periods,
            args.draws,
            args.tune,
            args.chains,
            args.repetitions,
        ) = 1, 8, 20, 20, 1, 1
    args.output.mkdir(parents=True, exist_ok=True)
    workloads = (
        [(1, 30), (3, 30), (1, 300), (10, 300)]
        if args.matrix and not args.smoke
        else [(args.units, args.periods)]
    )
    fractions = (
        [1.0, 0.2] if args.matrix and not args.smoke else [args.observed_fraction]
    )
    samplers = (
        ["pymc", "nutpie-numba", "numpyro", "nutpie-jax"]
        if args.sampler == "all"
        else [args.sampler]
    )
    regimes = ["cold", "warm"] if args.cache == "both" else [args.cache]
    all_runs = []
    scratch = Path(__file__).resolve().parents[1] / ".scratch" / "sampler-caches"
    scratch.mkdir(parents=True, exist_ok=True)
    for (units, periods), fraction, sampler, regime in itertools.product(
        workloads, fractions, samplers, regimes
    ):
        with tempfile.TemporaryDirectory(prefix="run-", dir=scratch) as cache_root:
            repetitions = range(-1 if regime == "warm" else 0, args.repetitions)
            for repetition in repetitions:
                # Warm runs reuse one disk cache across fresh processes, primed
                # with the same model/settings. Cold runs each start empty.
                cache = Path(cache_root) / (
                    str(repetition) if regime == "cold" else "warm"
                )
                cache.mkdir(parents=True, exist_ok=True)
                name = f"u{units}-t{periods}-obs{fraction:g}-{sampler}-{regime}-{repetition}"
                destination = args.output / f"{name}.json"
                if destination.exists():
                    parser.error(
                        f"Refusing to overwrite {destination}; use a new output directory."
                    )
                env = os.environ.copy()
                env.update({
                    name: "1"
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "NUMBA_NUM_THREADS",
                    )
                })
                env.update(
                    PYTENSOR_FLAGS=f"floatX=float64,compiledir={cache / 'pytensor'}",
                    JAX_ENABLE_X64="true",
                    JAX_PLATFORMS="cpu",
                    JAX_COMPILATION_CACHE_DIR=str(cache / "jax"),
                    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
                    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
                    NUMBA_CACHE_DIR=str(cache / "numba"),
                )
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--output",
                    str(destination),
                    "--sampler",
                    sampler,
                    "--units",
                    str(units),
                    "--periods",
                    str(periods),
                    "--observed-fraction",
                    str(fraction),
                    "--draws",
                    str(args.draws),
                    "--tune",
                    str(args.tune),
                    "--chains",
                    str(args.chains),
                    "--seed",
                    str(args.seed + max(0, repetition)),
                    "--data-seed",
                    str(args.data_seed),
                    "--cache",
                    regime,
                    "--repetition",
                    str(repetition),
                    "--gradient-backend",
                    args.gradient_backend,
                ]
                if args.smoke:
                    command.append("--smoke")
                print(name, flush=True)
                with destination.with_suffix(".log").open("w") as log:
                    result = subprocess.run(
                        command, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                if not destination.exists():
                    destination.write_text(
                        json.dumps({
                            "status": "failed",
                            "error": f"Worker exited {result.returncode}; see sibling .log",
                            "sampler": sampler,
                            "units": units,
                            "periods": periods,
                            "observed_fraction": fraction,
                            "cache": regime,
                            "repetition": repetition,
                        })
                        + "\n"
                    )
                all_runs.append({
                    "file": destination.name,
                    **json.loads(destination.read_text()),
                })
                (args.output / "summary.json").write_text(
                    json.dumps(
                        _json_safe(summarize(all_runs)), indent=2, allow_nan=False
                    )
                    + "\n"
                )


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep failures/priming visible and report median/range without speedup claims."""
    groups: dict[tuple, list[dict]] = {}
    for run in runs:
        if run["repetition"] < 0:
            continue
        key = tuple(
            run[name]
            for name in ("units", "periods", "observed_fraction", "sampler", "cache")
        )
        groups.setdefault(key, []).append(run)
    rows = []
    for key, group in groups.items():
        successful = [run for run in group if run["status"] == "ok"]
        row = dict(
            zip(
                ("units", "periods", "observed_fraction", "sampler", "cache"),
                key,
                strict=True,
            )
        )
        row.update(
            runs=len(group),
            failures=len(group) - len(successful),
            diagnostic_passes=sum(
                run.get("diagnostics_acceptable_for_comparison", False)
                for run in successful
            ),
        )
        for name in (
            "end_to_end_seconds",
            "minimum_bulk_ess_per_end_to_end_second",
            "minimum_tail_ess_per_end_to_end_second",
            "peak_process_rss_bytes",
        ):
            values = [
                run[name]
                for run in successful
                if run.get(name) is not None and np.isfinite(run[name])
            ]
            row.setdefault("metric_available_runs", {})[name] = len(values)
            row[name] = (
                {
                    "median": float(np.median(values)),
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
        rows.append(row)
    return {
        "note": "Warm means primed disk caches in fresh processes, not retained compiled functions. Failures and poor mixing remain in runs; timing medians are descriptive, not speedups. Smoke results cannot support performance claims.",
        "groups": rows,
        "runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"environment", "diagnostics"}
            }
            for run in runs
        ],
    }


if __name__ == "__main__":
    main()
