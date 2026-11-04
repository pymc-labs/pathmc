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
"""Profile optimized panel logp/gradients or probe fitting backends for #526.

Sampler timings are separate experiments from posterior query timings. Missing
backend clocks are reported as null, not estimated from compilation overhead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from benchmark_queries import environment, fixture, instrumentation


def scan_profile(args) -> dict:
    """Profile the actual optimized likelihood and gradient, including scan Ops."""
    fitted, construction, shape = fixture(args.workload, args.large)
    model = fitted.pymc_model
    point = model.initial_point()
    start = perf_counter()
    function = model.compile_fn(
        [model.logp(), model.dlogp()],
        inputs=model.value_vars,
        point_fn=True,
        profile=True,
    )
    compilation = perf_counter() - start
    first = perf_counter()
    logp, gradient = function(point)
    first = perf_counter() - first
    assert np.isfinite(logp) and np.isfinite(gradient).all()
    elapsed = []
    for _ in range(100):
        start = perf_counter()
        function(point)
        elapsed.append(perf_counter() - start)
    profile = function.f.profile
    times = {}
    for (_, node), seconds in profile.apply_time.items():
        name = str(node.op)
        times[name] = times.get(name, 0) + seconds
    nodes = function.f.maker.fgraph.toposort()
    return {
        "shape": shape,
        "construction_seconds": construction,
        "graph_and_function_compilation_seconds": compilation,
        "first_execution_seconds": first,
        "warm_median_seconds": float(np.median(elapsed)),
        "warm_min_seconds": min(elapsed),
        "warm_max_seconds": max(elapsed),
        "optimized_node_count": len(nodes),
        "optimized_scan_ops": [
            str(node.op) for node in nodes if type(node.op).__name__ == "Scan"
        ],
        "op_seconds_including_first_execution": dict(
            sorted(times.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def fit_probe(args) -> dict:
    """Fit identical synthetic data without changing pathmc's sampler defaults."""
    import arviz as az
    import pandas as pd

    import pathmc

    rng = np.random.default_rng(526)
    rows = 100
    x = rng.normal(size=rows)
    m = 1 + 0.5 * x + rng.normal(scale=0.5, size=rows)
    y = 0.3 + 0.8 * m + 0.2 * x + rng.normal(scale=0.5, size=rows)
    frame = pd.DataFrame({
        "X": x,
        "M": m,
        "Y": y,
        "unit": np.zeros(rows, dtype=int),
        "time": np.arange(rows),
    })
    kwargs = {}
    spec = "M ~ X\nY ~ M + X"
    if args.workload == "panel_ar":
        for time in range(1, rows):
            y[time] = 0.3 + 0.5 * y[time - 1] + 0.2 * x[time] + rng.normal(scale=0.5)
        frame["Y"] = y
        spec = "Y ~ X + lag(Y)"
        kwargs["panel"] = {"unit": "unit", "time": "time"}
    elif args.workload == "latent":
        for time in range(1, rows):
            m[time] = 0.5 * m[time - 1] + 0.2 * x[time] + rng.normal(scale=0.5)
        frame["Y"] = m + rng.normal(scale=0.2, size=rows)
        frame.loc[np.arange(rows) % 4 != 0, "Y"] = np.nan
        frame = frame.drop(columns="M")
        spec = "M ~ X + lag(M)\nY ~ 0 + 1*M"
        kwargs.update(
            panel={"unit": "unit", "time": "time"},
            latent=["M"],
            families={"M": "latent_normal"},
        )
    fitted = pathmc.model(spec, data=frame, **kwargs)
    sampler = "nutpie" if args.sampler.startswith("nutpie") else args.sampler
    options = (
        {"nuts_sampler_kwargs": {"backend": args.sampler.split("-")[1]}}
        if sampler == "nutpie"
        else {}
    )
    start = perf_counter()
    with instrumentation() as events:
        idata = fitted.fit(
            draws=args.draws,
            tune=args.tune,
            chains=2,
            cores=1,
            nuts_sampler=sampler,
            random_seed=526,
            progressbar=False,
            compute_log_likelihood=False,
            discard_tuned_samples=False,
            idata_kwargs={"save_warmup": True},
            **options,
        )
    elapsed = perf_counter() - start
    stats = idata.sample_stats.to_dataset()
    summary = az.summary(
        idata,
        var_names=[
            name
            for name in idata.posterior.data_vars
            if name.startswith("beta_") or name.startswith("sigma_")
        ],
    )

    def clock(group):
        if group not in idata:
            return None
        ds = idata[group].to_dataset()
        return float(ds.perf_counter_diff.sum()) if "perf_counter_diff" in ds else None

    return {
        "sampler": args.sampler,
        "draws_per_chain": args.draws,
        "tune_per_chain": args.tune,
        "tuning_sampling_conversion_seconds": elapsed,
        "compiler_events_inclusive": dict(events),
        "tuning_chain_cpu_wall_seconds": clock("warmup_sample_stats"),
        "sampling_chain_cpu_wall_seconds": clock("sample_stats"),
        "clock_note": "Per-chain clocks are summed when exposed; null means the backend did not expose this boundary. Total time includes compilation and result conversion.",
        "minimum_bulk_ess": float(summary.ess_bulk.min()),
        "minimum_bulk_ess_per_total_second": float(summary.ess_bulk.min()) / elapsed,
        "maximum_rhat": float(summary.r_hat.max()),
        "divergences": int(stats.diverging.sum()),
    }


def main() -> None:
    """Run one backend experiment in an isolated process and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=["profile", "fit"], default="profile")
    parser.add_argument(
        "--workload",
        choices=["linear", "panel_lag", "panel_ar", "panel_transform", "latent"],
        default="panel_ar",
    )
    parser.add_argument(
        "--sampler",
        choices=["pymc", "nutpie-numba", "nutpie-jax", "numpyro"],
        default="pymc",
    )
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--mode", choices=["CVM", "NUMBA", "JAX"], default="CVM")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=200)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        args.output.parent.mkdir(parents=True, exist_ok=True)
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
        mode = args.mode if args.experiment == "profile" else "FAST_RUN"
        env["PYTENSOR_FLAGS"] = (
            f"floatX=float64,mode={mode},profiling__time_thunks=True"
        )
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker"], env=env, check=True
        )
        return
    try:
        result = (scan_profile if args.experiment == "profile" else fit_probe)(args)
        report = {"status": "ok", **result}
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report.update(
        environment=environment(),
        experiment=args.experiment,
        workload=args.workload,
        sampler=args.sampler,
        large=args.large,
        peak_process_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
