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
from pathlib import Path


def test_comparison_preserves_failures_and_diagnostic_denominator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_backends.py"
    spec = importlib.util.spec_from_file_location("sampling_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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
