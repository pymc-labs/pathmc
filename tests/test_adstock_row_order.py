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
"""Non-panel adstock must not silently convolve across stacked rows.

Issue #655. Geometric panel scan stays ``y_t = x_t + decay * y_{t-1}``.
Without ``panel=``, the three convolution transforms warn that row order
is one series. With ``panel=``, an impulse in one unit stays out of the next.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytensor.tensor as pt
import pytest

import pathmc
from pathmc.transforms import Adstock, _geometric_adstock

DECAY = 0.8
SPECS = {
    "adstock": (
        "Y ~ 0 + adstock(X, decay=theta)",
        {"beta_Y": [1.0], "sigma_Y": 0.0, "theta": DECAY},
    ),
    "delayed_adstock": (
        "Y ~ 0 + delayed_adstock(X, decay=theta, theta=peak)",
        {"beta_Y": [1.0], "sigma_Y": 0.0, "theta": DECAY, "peak": 0.0},
    ),
    "weibull_adstock": (
        "Y ~ 0 + weibull_adstock(X, lam=lam, k=k)",
        {"beta_Y": [1.0], "sigma_Y": 0.0, "lam": 1.5, "k": 1.2},
    ),
}
# Finite-kernel leak into the next unit's first row, observed on main
# before the warning. Pinning it rejects a fix that zeroes the series
# or swaps the kernel while still emitting a warning.
LEAKED_GEO_B = {
    "adstock": 0.8,
    "delayed_adstock": 0.8,
    "weibull_adstock": 0.5173965,
}


def _stacked_units() -> pd.DataFrame:
    rows = []
    for geo, impulse_week in [("A", 5), ("B", None)]:
        for week in range(6):
            rows.append({
                "geo": geo,
                "week": week,
                "X": 1.0 if week == impulse_week else 0.0,
            })
    return pd.DataFrame(rows)


def _series_warnings(
    caught: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    return [item for item in caught if "one series" in str(item.message)]


def _series_messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(item.message) for item in _series_warnings(caught)]


def _attributed_to_this_file(caught: list[warnings.WarningMessage]) -> bool:
    """The shared walker must still name this file, not a pathmc frame."""
    caller = Path(__file__).resolve()
    series = _series_warnings(caught)
    return bool(series) and all(
        Path(item.filename).resolve() == caller for item in series
    )


def test_kernel_helper_does_not_warn() -> None:
    """The numeric kernel is shared with the per-unit path; warning there
    would fire on panel convolution too."""
    x = pt.as_tensor_variable(np.array([1.0, 0.0, 0.0]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _geometric_adstock(x, alpha=DECAY, l_max=12).eval()
    assert _series_messages(caught) == []


def test_geometric_step_stays_unbounded_recursion() -> None:
    out, new_state = Adstock().step(
        pt.as_tensor_variable(1.0),
        pt.as_tensor_variable(0.5),
        {"decay": DECAY},
    )
    assert float(out.eval()) == pytest.approx(1.0 + DECAY * 0.5)
    assert float(new_state.eval()) == pytest.approx(1.0 + DECAY * 0.5)


@pytest.mark.parametrize("name", list(SPECS))
def test_nonpanel_probe_warns_and_keeps_the_kernel(name: str) -> None:
    spec, params = SPECS[name]
    frame = _stacked_units()
    with pytest.warns(UserWarning, match="one series") as caught:
        flat = pathmc.simulate(spec, data=frame[["X"]], params=params, random_seed=0)
    messages = _series_messages(list(caught))
    assert messages
    assert _attributed_to_this_file(list(caught))
    assert all("row order" in message for message in messages)
    assert all("panel=" in message for message in messages)
    if name == "adstock":
        assert any(
            "12-lag convolution" in message
            and "not the full recursive stock used when panel= is set" in message
            for message in messages
        )
    else:
        assert all("recurs" not in message.lower() for message in messages)
        assert all(
            re.search(r"\bstock\b", message.lower()) is None for message in messages
        )
    leaked = float(np.asarray(flat["Y"])[6])
    assert leaked == pytest.approx(LEAKED_GEO_B[name], rel=1e-4, abs=1e-4)


@pytest.mark.parametrize("name", list(SPECS))
def test_panel_probe_isolates_units_without_row_order_warning(name: str) -> None:
    spec, params = SPECS[name]
    frame = _stacked_units()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = pathmc.simulate(
            spec,
            data=frame,
            params=params,
            panel={"unit": "geo", "time": "week"},
            random_seed=0,
        )
    assert _series_messages(list(caught)) == []
    assert float(np.asarray(panel["Y"])[6]) == 0.0
    assert np.asarray(panel["Y"], dtype=float)[6:].tolist() == [0.0] * 6


@pytest.mark.parametrize("name", list(SPECS))
def test_model_without_panel_warns(name: str) -> None:
    spec, _params = SPECS[name]
    frame = _stacked_units()[["X"]].assign(Y=0.0)
    with pytest.warns(UserWarning, match="one series") as caught:
        pathmc.model(spec, data=frame)
    messages = _series_messages(list(caught))
    assert messages
    assert all("row order" in message for message in messages)
    assert _attributed_to_this_file(list(caught))


@pytest.mark.parametrize("name", ["adstock", "delayed_adstock", "weibull_adstock"])
def test_model_with_panel_does_not_warn_about_row_order(name: str) -> None:
    spec = {
        "adstock": "Y ~ 0 + adstock(X, decay=theta)",
        "delayed_adstock": "Y ~ 0 + delayed_adstock(X, decay=theta, theta=peak)",
        "weibull_adstock": "Y ~ 0 + weibull_adstock(X, lam=lam, k=k)",
    }[name]
    frame = _stacked_units().assign(Y=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pathmc.model(
            spec,
            data=frame,
            panel={"unit": "geo", "time": "week"},
        )
    assert _series_messages(list(caught)) == []


def test_nonpanel_geometric_stays_a_12_lag_window() -> None:
    """A single series longer than 12 must still drop the oldest input.

    Widening this kernel is a different bug. Truncating the panel scan to
    match it is the #638 mistake and is locked by the companion test.
    """
    x = np.zeros(13)
    x[0] = 1.0
    frame = pd.DataFrame({"X": x})
    with pytest.warns(UserWarning, match="12-lag convolution"):
        out = pathmc.simulate(
            "Y ~ 0 + adstock(X, decay=theta)",
            data=frame,
            params={"beta_Y": [1.0], "sigma_Y": 0.0, "theta": DECAY},
            random_seed=0,
        )
    assert float(np.asarray(out["Y"])[12]) == 0.0


def test_panel_geometric_scan_keeps_history_past_l_max() -> None:
    x = np.zeros(13)
    x[0] = 1.0
    frame = pd.DataFrame({"unit": "A", "week": np.arange(13), "X": x, "Y": 0.0})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = pathmc.simulate(
            "Y ~ 0 + adstock(X, decay=theta)",
            data=frame,
            params={"beta_Y": [1.0], "sigma_Y": 0.0, "theta": DECAY},
            panel={"unit": "unit", "time": "week"},
            random_seed=0,
        )
    assert _series_messages(list(caught)) == []
    assert float(np.asarray(out["Y"])[12]) == pytest.approx(DECAY**12)


def test_custom_l_max_names_its_own_window() -> None:
    with pytest.warns(UserWarning, match="8-lag convolution") as caught:
        Adstock(l_max=8).apply_pymc(
            pt.as_tensor_variable(np.zeros(3)),
            {"decay": DECAY},
        )
    messages = _series_messages(list(caught))
    assert messages
    assert _attributed_to_this_file(list(caught))
    assert all("12-lag" not in message for message in messages)
    assert any("recursive stock" in message for message in messages)
