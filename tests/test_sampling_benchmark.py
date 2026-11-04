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
"""Benchmark summaries must not conceal failed/poorly mixing runs or priming."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_backends.py"
    spec = importlib.util.spec_from_file_location("sampling_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_preserves_failures_and_diagnostic_denominator():
    module = _benchmark_module()

    def run(status="ok", quality=True, repetition=0, seconds=3):
        result = dict(
            file=f"{repetition}.json",
            units=1,
            periods=30,
            observed_fraction=0.2,
            sampler="pymc",
            cache="warm",
            status=status,
            repetition=repetition,
            diagnostics_acceptable_for_comparison=quality,
        )
        if status == "ok":
            result.update(
                end_to_end_seconds=seconds,
                minimum_bulk_ess_per_end_to_end_second=30,
                minimum_tail_ess_per_end_to_end_second=20,
                peak_process_rss_bytes=100,
            )
        return result

    summary = module.summarize([
        run(repetition=-1, seconds=100),
        run(),
        run(quality=False, repetition=1, seconds=5),
        run(status="failed", repetition=2),
    ])
    (group,) = summary["groups"]
    assert group["runs"] == 3
    assert group["failures"] == 1
    assert group["diagnostic_passes"] == 1
    assert group["end_to_end_seconds"] == {"median": 4.0, "min": 3, "max": 5}
    assert len(summary["runs"]) == 4

    # Mirror the worker's JSON round-trip: nonfinite diagnostics become null.
    unavailable = run(quality=False, repetition=3)
    unavailable["minimum_bulk_ess_per_end_to_end_second"] = np.nan
    unavailable["minimum_tail_ess_per_end_to_end_second"] = np.nan
    decoded = json.loads(json.dumps(module._json_safe(unavailable)))
    summary = module.summarize([decoded])
    group = summary["groups"][0]
    assert group["runs"] == 1 and group["failures"] == 0
    assert group["diagnostic_passes"] == 0
    assert group["minimum_bulk_ess_per_end_to_end_second"] is None
    assert group["minimum_tail_ess_per_end_to_end_second"] is None
    assert group["metric_available_runs"]["minimum_bulk_ess_per_end_to_end_second"] == 0
    assert summary["runs"][0]["status"] == "ok"


def test_failed_worker_preserves_requested_sampler_configuration(tmp_path, monkeypatch):
    module = _benchmark_module()
    args = SimpleNamespace(
        output=tmp_path / "failed.json",
        units=3,
        periods=30,
        observed_fraction=0.2,
        sampler="nutpie-jax",
        draws=500,
        tune=600,
        chains=4,
        target_accept=0.97,
        gradient_backend="pytensor",
        seed=87,
        data_seed=85,
        cache="warm",
        repetition=2,
        smoke=False,
    )

    def fail(_):
        raise ValueError("injected backend failure")

    monkeypatch.setattr(module, "fit_probe", fail)
    monkeypatch.setattr(module, "environment", lambda: {"commit": "test-candidate"})
    module.worker(args)
    report = json.loads(args.output.read_text())
    assert report["status"] == "failed"
    assert "injected backend failure" in report["error"]
    for key in (
        "draws",
        "tune",
        "chains",
        "target_accept",
        "gradient_backend",
        "sampler",
        "seed",
        "data_seed",
    ):
        assert report["requested_settings"][key] == getattr(args, key)


def test_crashed_worker_records_reproduction_command_and_settings(
    tmp_path, monkeypatch
):
    module = _benchmark_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            module.__file__,
            "--output",
            str(tmp_path),
            "--sampler",
            "numpyro",
            "--units",
            "3",
            "--periods",
            "8",
            "--cache",
            "cold",
            "--repetitions",
            "1",
            "--draws",
            "500",
            "--tune",
            "600",
            "--chains",
            "4",
            "--target-accept",
            ".97",
        ],
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=-11)
    )
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: "test-candidate\n"
    )
    module.main()
    report = json.loads(next(tmp_path.glob("*cold-0.json")).read_text())
    assert report["status"] == "failed"
    assert "-11" in report["error"]
    assert report["source_commit"] == "test-candidate"
    assert "--draws" in report["worker_command"]
    assert report["requested_settings"]["draws"] == 500
    assert report["requested_settings"]["tune"] == 600
    assert report["requested_settings"]["chains"] == 4
    assert report["requested_settings"]["target_accept"] == 0.97
    assert report["requested_settings"]["units"] == 3
    assert (
        report["worker_environment"]["XLA_FLAGS"]
        == "--xla_force_host_platform_device_count=1"
    )
