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
"""Measure posterior queries in isolated processes, without running MCMC.

Run with ``uv run python scripts/benchmark_queries.py --output .scratch/query.json``.
Timings are evidence for review, never unit-test assertions. Each workload runs in
its own process; peak RSS includes imports, graph construction and query execution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext, ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from typing import Any, TYPE_CHECKING
from unittest.mock import patch

import numpy as np

if TYPE_CHECKING:
    from pathmc._model import PathModel


def query_adapter() -> ModuleType | None:
    """Allow this baseline harness to run before the prepared executor exists."""
    if importlib.util.find_spec("pathmc._query") is None:
        return None
    return importlib.import_module("pathmc._query")


def environment() -> dict[str, Any]:
    """Describe the executable environment, including explicit thread settings."""
    import pytensor
    from pytensor.compile.mode import get_mode

    return {
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pathmc", "pymc", "pytensor", "arviz", "numpy")
        },
        "optional_backend_versions": {
            name: importlib.metadata.version(name)
            for name in ("numba", "jax", "numpyro", "nutpie")
            if importlib.util.find_spec(name) is not None
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
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
        "mode": str(pytensor.config.mode),
        "effective_linker": type(get_mode().linker).__name__,
        "compiledir": str(pytensor.config.compiledir),
        "seed": 526,
    }


@contextmanager
def instrumentation() -> Iterator[dict[str, dict[str, Any]]]:
    """Count actual compiler entries, graph surgery and numeric function calls.

    Inclusive durations may overlap (PyMC compilation calls PyTensor); consumers
    must not sum these buckets. Numeric time includes first execution/JIT and is
    reported separately for the first request and warmed requests.
    """
    import pymc as pm
    import pymc.model.core as core
    import pytensor
    import xarray as xr
    from pytensor.compile.executor import Function

    events = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def wrap(name, original):
        def measured(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                events[name]["calls"] += 1
                events[name]["seconds"] += perf_counter() - start

        return measured

    with ExitStack() as stack:
        entries = [
            (pm, "do", "pm.do"),
            (pm, "to_inference_data", "predictive_result_conversion"),
            (core, "compile", "pymc.compile"),
            (pytensor, "function", "pytensor.function"),
            (Function, "__call__", "function_execution"),
            (xr.Dataset, "__init__", "xarray_dataset_construction"),
            (xr.DataArray, "mean", "xarray_aggregation"),
        ]
        query = query_adapter()
        if query is not None:
            entries.append((query, "prepare_mean", "mean_plan_preparation"))
            if query.freeze_model is not None:
                entries.append((query, "freeze_model", "prediction_graph_preparation"))
        for owner, name, event in entries:
            stack.enter_context(
                patch.object(owner, name, wrap(event, getattr(owner, name)))
            )
        yield events


def fixture(workload: str, large: bool) -> tuple[PathModel, float, dict[str, int]]:
    """Construct deterministic inputs and fixed synthetic parameter draws."""
    import pandas as pd
    import xarray as xr

    import pathmc

    rows, draws = (10_000, 2_000) if large else (100, 100)
    rng = np.random.default_rng(526)
    data = pd.DataFrame({
        "X": rng.normal(size=rows),
        "M": np.zeros(rows),
        "Y": np.zeros(rows),
    })
    kwargs = {}
    spec = "M ~ X\nY ~ X + M"
    if workload in ("bernoulli", "poisson"):
        kwargs["families"] = {"M": workload}
    elif workload in ("panel_lag", "panel_ar", "panel_transform", "panel_policy"):
        units, periods = (20, 200) if large else (1, 20)
        data = pd.DataFrame({
            "unit": np.repeat(np.arange(units), periods),
            "time": np.tile(np.arange(periods), units),
            "X": rng.uniform(0, 1, units * periods),
            "Y": np.zeros(units * periods),
        })
        kwargs["panel"] = {"unit": "unit", "time": "time"}
        spec = {
            "panel_lag": "Y ~ X + lag(X)",
            "panel_ar": "Y ~ X + lag(Y)",
            "panel_policy": "Y ~ X + lag(Y)",
            "panel_transform": "Y ~ logistic_saturation(adstock(X, decay=decay), lam=lam)",
        }[workload]
        if workload == "panel_policy":
            kwargs["initial"] = {"Y": 0.0}
    start = perf_counter()
    model = pathmc.model(spec, data=data, **kwargs)
    construction_seconds = perf_counter() - start
    # Normal fitted-posterior coordinates keep the baseline comparable even on
    # PyMC releases whose compute_deterministics drops custom sample labels.
    posterior = xr.Dataset(coords={"chain": [0, 1], "draw": np.arange(draws // 2)})
    shape = (2, draws // 2)
    for reg in model._spec.regressions:
        names = list(model.pymc_model.coords[f"{reg.lhs}_predictors"])
        beta = np.zeros((*shape, len(names)))
        values = {
            "Intercept": 1 if reg.lhs == "M" else 3,
            "X": 2 if reg.lhs == "M" else 4,
            "M": 5,
            "lag(X)": 0.3,
            "lag(Y)": 0.5,
        }
        for index, name in enumerate(names):
            beta[..., index] = values.get(name, 0.8)
        posterior[f"beta_{reg.lhs}"] = (
            ("chain", "draw", f"{reg.lhs}_predictors"),
            beta,
        )
        posterior = posterior.assign_coords({f"{reg.lhs}_predictors": names})
        if kwargs.get("families", {}).get(reg.lhs, "gaussian") == "gaussian":
            posterior[f"sigma_{reg.lhs}"] = (("chain", "draw"), np.full(shape, 0.5))
    for name, value in (("decay", 0.7), ("lam", 0.5)):
        if name in model.pymc_model.named_vars:
            posterior[name] = (("chain", "draw"), np.full(shape, value))
    observed = xr.Dataset({
        reg.lhs: (("obs",), data[reg.lhs].to_numpy()) for reg in model._spec.regressions
    })
    model._idata = xr.DataTree.from_dict({
        "posterior": posterior,
        "observed_data": observed,
    })
    if workload == "linear":
        for name, values in (
            ("M", 1 + 2 * data.X.to_numpy()),
            ("Y", 3 + 4 * data.X.to_numpy() + 5 * data.M.to_numpy()),
        ):
            posterior[f"mu_{name}"] = (
                ("chain", "draw", "obs"),
                np.broadcast_to(values, (*shape, len(data))),
            )
        model._idata = xr.DataTree.from_dict({
            "posterior": posterior,
            "observed_data": observed,
            "constant_data": xr.Dataset({"X": ("obs", data.X.to_numpy())}),
        })
    return model, construction_seconds, {"rows": len(data), "samples": draws}


def measure_request(call: Callable[[], Any]) -> tuple[dict[str, Any], Any]:
    """Measure inclusive end-to-end latency and force output materialization."""
    import xarray as xr

    start = perf_counter()
    with instrumentation() as events:
        result = call()
    converted = perf_counter()
    if isinstance(result, list):
        dataset = xr.concat([value.dataset for value in result], dim="scenario")
    elif isinstance(result, xr.DataTree):
        dataset = result.posterior_predictive.to_dataset()
    else:
        dataset = result.dataset
    output_bytes = sum(np.asarray(value).nbytes for value in dataset.data_vars.values())
    end = perf_counter()
    return {
        "end_to_end_seconds": end - start,
        "result_materialization_seconds": end - converted,
        "output_bytes": output_bytes,
        "events": dict(events),
    }, dataset


def worker(args: argparse.Namespace) -> dict[str, Any]:
    """Run one workload in a fresh process with independently counted requests."""
    model, construction, shape = fixture(args.workload, args.large)
    query = query_adapter()

    time_kwargs = (
        {"simulate_over": "time"} if args.workload.startswith("panel_") else {}
    )
    requests = {
        "mean_do": lambda: model.do(set={"X": 1.0}, **time_kwargs),
        "ate": lambda: model.ate("Y", "X", **time_kwargs),
        "predictive_do": lambda: model.do(
            set={"X": 1.0}, kind="predictive", random_seed=526, **time_kwargs
        ),
        "predict": lambda: model.predict(
            extend_inferencedata=False, progressbar=False, random_seed=526
        ),
    }
    if args.workload == "panel_policy":
        import pathmc

        policy = pathmc.TreatmentPolicy(
            "feedback", "X", pathmc.where(pathmc.previous("Y") > 1, 0.0, 1.0)
        )
        requests = {
            "policy_mean": lambda: model.do(
                policy=policy,
                horizon=model._gen_model._pathmc_panel_scan.n_times,
                n_simulations=args.simulations,
                simulate_over="time",
                random_seed=526,
            )
        }
    if args.workload == "linear":
        from itertools import cycle

        frame = model._data.to_pandas()
        changes = cycle((0.0, 1.0, 2.0, 0.0))
        requests["predict_changed_data"] = lambda: model.predict(
            data=frame.assign(X=frame.X + next(changes)),
            extend_inferencedata=False,
            progressbar=False,
            random_seed=526,
        )
        scenarios = [{"X": float(x)} for x in np.linspace(0, 1, args.scenarios)]
        requests["scenarios"] = lambda: (
            query.mean_scenarios(model, scenarios)
            if query is not None and query.eligible(model)
            else [model.do(set=arm) for arm in scenarios]
        )
        if query is not None and query.eligible(model):
            requests["prepared_per_draw"] = lambda: query.mean_scenarios(
                model, scenarios, max_elements=len(model._data)
            )
    report = {
        "environment": environment(),
        "shape": shape,
        "model_construction_seconds": construction,
        "queries": {},
    }
    for name, call in requests.items():
        if args.workload == "panel_policy" and args.simulations == 1:
            report["queries"][name] = {
                "status": "unsupported",
                "reason": "The policy API requires at least two simulations; one simulation is not a supported policy query.",
            }
            continue
        try:
            first, dataset = measure_request(call)
            if args.workload == "linear" and name in ("mean_do", "ate"):
                np.testing.assert_allclose(
                    dataset.Y, 22 if name == "mean_do" else 14, rtol=1e-8, atol=1e-10
                )
            if args.workload in ("bernoulli", "poisson") and name in ("mean_do", "ate"):
                link = (
                    (lambda x: 1 / (1 + np.exp(-x)))
                    if args.workload == "bernoulli"
                    else np.exp
                )
                expected = (
                    7 + 5 * link(3)
                    if name == "mean_do"
                    else 4 + 5 * (link(3) - link(1))
                )
                np.testing.assert_allclose(dataset.Y, expected, rtol=1e-8, atol=1e-10)
            warm = [measure_request(call)[0] for _ in range(args.repetitions)]
            latencies = [run["end_to_end_seconds"] for run in warm]
            report["queries"][name] = {
                "status": "ok",
                "first_request": first,
                "warmed_requests": warm,
                "warm_median_seconds": float(np.median(latencies)),
                "warm_min_seconds": min(latencies),
                "warm_max_seconds": max(latencies),
            }
        except Exception as error:
            report["queries"][name] = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = int(
        rss if sys.platform == "darwin" else rss * 1024
    )
    return report


def main() -> None:
    """Write a JSON benchmark report; keep large runs outside pytest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--workload",
        choices=[
            "linear",
            "bernoulli",
            "poisson",
            "panel_lag",
            "panel_ar",
            "panel_transform",
            "panel_policy",
        ],
        default="linear",
    )
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--scenarios", type=int, choices=[1, 2, 25], default=2)
    parser.add_argument("--simulations", type=int, choices=[1, 20], default=20)
    parser.add_argument(
        "--mode", choices=["FAST_RUN", "CVM", "VM", "NUMBA", "JAX"], default="FAST_RUN"
    )
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--executor", choices=["auto", "legacy"], default="auto")
    args = parser.parse_args()
    if args.repetitions < 5:
        parser.error("Comparisons need at least five warmed repetitions.")
    if args.worker:
        try:
            with (
                patch("pathmc._query.eligible", return_value=False)
                if args.executor == "legacy" and query_adapter() is not None
                else nullcontext()
            ):
                report = worker(args)
                statuses = [q["status"] for q in report["queries"].values()]
                report["status"] = (
                    "ok"
                    if all(s == "ok" for s in statuses)
                    else "unsupported"
                    if all(s == "unsupported" for s in statuses)
                    else "partial"
                    if "ok" in statuses
                    else "failed"
                )
        except Exception as error:
            report = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "environment": environment(),
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        env[name] = "1"
    with tempfile.TemporaryDirectory(prefix="pathmc-query-") as cache:
        flags = [f"mode={args.mode}", "floatX=float64"]
        if args.cold_cache:
            flags.append(f"base_compiledir={cache}")
        env["PYTENSOR_FLAGS"] = ",".join(flags)
        command = [sys.executable, __file__, *sys.argv[1:], "--worker"]
        subprocess.run(command, env=env, check=True)
    report = json.loads(args.output.read_text())
    report["cache_state"] = (
        "isolated-empty-disk"
        if args.cold_cache
        else "existing-disk-cache-first-request"
    )
    report["workload"] = args.workload
    report["large"] = args.large
    report["executor"] = args.executor
    report["scenarios"] = args.scenarios
    report["simulations"] = args.simulations
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
