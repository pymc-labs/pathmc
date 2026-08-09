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
"""Panel data utilities for longitudinal path models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import narwhals.stable.v1 as nw
import numpy as np

__all__ = ["PanelInfo", "observed_means_by_time"]


@dataclass
class PanelInfo:
    """Metadata describing panel structure in the data.

    Parameters
    ----------
    unit : str
        Column name identifying the panel unit (e.g. ``"region"``).
    time : str
        Column name identifying the time index (e.g. ``"week"``).
    unit_labels : list[str]
        Unique unit labels in sorted order.
    """

    unit: str
    time: str
    unit_labels: list[str]


def _validate_panel_args(
    df: nw.DataFrame,
    panel: dict[str, str],
    variables: list[str],
) -> None:
    """Validate panel arguments, raising KeyError on problems."""
    if "unit" not in panel:
        raise KeyError(
            "panel dict must contain 'unit' key. "
            "Example: panel={'unit': 'region', 'time': 'week'}"
        )
    if "time" not in panel:
        raise KeyError(
            "panel dict must contain 'time' key. "
            "Example: panel={'unit': 'region', 'time': 'week'}"
        )

    unit_col = panel["unit"]
    time_col = panel["time"]

    _require_column(df, unit_col, "Unit column")
    _require_column(df, time_col, "Time column")
    for var in variables:
        _require_column(df, var, "Variable")


def _require_column(df: nw.DataFrame, col: str, label: str) -> None:
    """Raise ``KeyError`` if *col* is absent from *df*."""
    if col not in df.columns:
        raise KeyError(
            f"{label} '{col}' not found in data. "
            f"Available columns: {', '.join(df.columns)}"
        )


def _reject_null_identifiers(df: nw.DataFrame, col: str, label: str) -> None:
    """Raise ``ValueError`` if *col* contains any null/NaN values.

    Null unit or time identifiers break every downstream assumption: they
    cannot be sorted alongside non-null labels (``TypeError`` from Python's
    comparison operators), cannot be meaningfully grouped into a panel row
    count, and cannot be reshaped into a rectangular grid. Reject them here
    with an actionable message rather than letting a ``TypeError`` (or a
    silently wrong panel) surface later.
    """
    null_mask = df[col].is_null()
    n_null = int(null_mask.sum())
    if n_null == 0:
        return
    raise ValueError(
        f"{label} '{col}' contains {n_null} null/NaN value(s). Every row "
        "must have a valid unit and time identifier; drop or fill rows "
        f"with missing '{col}' values before building a panel model."
    )


def _validate_panel_shape(
    df: nw.DataFrame,
    unit_col: str,
    time_col: str,
    unit_labels: list[str],
    *,
    require_rectangular: bool = True,
) -> None:
    """Validate that panel rows form a complete, rectangular (unit x time) grid.

    The scan-based panel compiler (``pathmc/compile.py``) assumes the data
    can be losslessly reshaped to a dense ``(n_times, n_units)`` array via
    ``n_times = len(data) // n_units`` after sorting by ``(unit, time)``.
    That assumption silently breaks -- producing mis-aligned, garbage rows
    with no error -- if the panel is unbalanced, has duplicate ``(unit,
    time)`` rows, or has missing unit/time combinations. This function
    fails loudly with a descriptive error instead.

    The rectangularity requirement belongs to the scan path only. A panel
    model with no temporal dependency (no ``lag()`` term, no ``adstock()``
    transform) takes the row-wise compiler, which indexes rows by unit and
    never reshapes, so it supports unbalanced panels, unequal timepoints and
    repeated ``(unit, time)`` rows. Pass ``require_rectangular=False`` for
    that path: only the empty-panel check then applies.
    """
    n_rows = df.shape[0]
    n_units = len(unit_labels)

    if n_units == 0:
        raise ValueError(
            f"Panel unit column '{unit_col}' has no values; cannot build a "
            "panel model from empty data."
        )

    if not require_rectangular:
        return

    units = df[unit_col].to_list()
    times = df[time_col].to_list()

    # 1. Duplicate (unit, time) rows: reshape cannot represent them.
    pair_counts = Counter(zip(units, times))
    duplicates = sorted(
        (pair for pair, count in pair_counts.items() if count > 1),
        key=lambda p: (str(p[0]), str(p[1])),
    )
    if duplicates:
        shown = ", ".join(f"({u!r}, {t!r})" for u, t in duplicates[:5])
        more = f" (+{len(duplicates) - 5} more)" if len(duplicates) > 5 else ""
        raise ValueError(
            "Panel data has duplicate (unit, time) rows, which cannot be "
            f"reshaped into a rectangular panel: {shown}{more}. Each "
            f"combination of '{unit_col}' and '{time_col}' must appear "
            "exactly once."
        )

    # 2. Balanced panel: every unit must contribute the same row count.
    if n_rows % n_units != 0:
        raise ValueError(
            f"Panel data is unbalanced: {n_rows} row(s) do not divide "
            f"evenly across {n_units} unit(s) in '{unit_col}'. Every unit "
            "must have the same number of time observations. Found row "
            f"counts per unit: {
                dict(sorted(Counter(units).items(), key=lambda kv: str(kv[0])))
            }."
        )
    n_times = n_rows // n_units

    unit_row_counts = Counter(units)
    unbalanced_units = {
        unit: count for unit, count in unit_row_counts.items() if count != n_times
    }
    if unbalanced_units:
        raise ValueError(
            f"Panel data is unbalanced: expected {n_times} observations per "
            f"unit (based on {n_rows} total rows / {n_units} units), but "
            f"the following units have a different count: {unbalanced_units}. "
            "Ragged or missing panel rows are not supported; every unit "
            "must be observed at every timepoint."
        )

    # 3. Every unit must be observed at exactly the same set of timepoints,
    # in the same order once sorted -- otherwise column position within
    # the reshaped (n_times, n_units) array would refer to different
    # timepoints for different units.
    expected_times = sorted(set(times))
    if len(expected_times) != n_times:
        raise ValueError(
            f"Panel data is not rectangular: found {len(expected_times)} "
            f"distinct value(s) of '{time_col}' but {n_times} observations "
            "per unit. Every unit must be observed at exactly the same set "
            "of timepoints."
        )

    times_by_unit: dict[object, list] = {}
    for u, t in zip(units, times):
        times_by_unit.setdefault(u, []).append(t)

    mismatched = []
    for unit, unit_times in times_by_unit.items():
        if sorted(unit_times) != expected_times:
            mismatched.append(unit)
    if mismatched:
        raise ValueError(
            "Panel data is not rectangular: the following unit(s) are not "
            f"observed at the same timepoints as the rest of the panel: "
            f"{sorted(mismatched, key=str)}. Every unit must share an "
            f"identical set of '{time_col}' values."
        )


def build_panel_info(
    df: nw.DataFrame,
    panel: dict[str, str],
    *,
    require_rectangular: bool = True,
) -> PanelInfo:
    """Build panel metadata from data and panel specification.

    Parameters
    ----------
    df : nw.DataFrame
        Panel data.
    panel : dict[str, str]
        Must contain ``"unit"`` and ``"time"`` keys.
    require_rectangular : bool
        Whether the data must form a dense ``(unit x time)`` grid. True for
        models that compile to the temporal scan (any ``lag()`` term or
        ``adstock()`` transform), which reshapes the rows. False for
        non-temporal panel models, which take the row-wise compiler and
        place no shape constraint beyond a non-empty unit column.

    Returns
    -------
    PanelInfo
        Panel metadata for use by compiler and simulator.

    Raises
    ------
    ValueError
        If the unit or time column contains null/NaN values, if the unit
        column is empty, or -- when *require_rectangular* is True -- if the
        panel is unbalanced, ragged, has duplicate ``(unit, time)`` rows, or
        units do not share an identical set of timepoints.
    """
    unit_col = panel["unit"]
    time_col = panel["time"]
    _reject_null_identifiers(df, unit_col, "Panel unit column")
    _reject_null_identifiers(df, time_col, "Panel time column")
    unit_labels = sorted(df[unit_col].unique().to_list())
    _validate_panel_shape(
        df,
        unit_col,
        time_col,
        unit_labels,
        require_rectangular=require_rectangular,
    )
    return PanelInfo(unit=unit_col, time=time_col, unit_labels=unit_labels)


def observed_means_by_time(
    data: nw.DataFrame,
    panel_info: PanelInfo,
    time_index: np.ndarray,
    variables: list[str],
) -> dict[str, np.ndarray]:
    """Unit-mean observed series per time step, aligned to *time_index*.

    Parameters
    ----------
    data : nw.DataFrame
        Panel data used to fit the model.
    panel_info : PanelInfo
        Panel column metadata.
    time_index : np.ndarray
        Time labels in the order used by panel ``do(simulate_over="time")``.
    variables : list[str]
        Variables to aggregate (ignored when absent from *data*).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from variable name to a length-``n_times`` mean array.
    """
    time_col = panel_info.time
    result: dict[str, np.ndarray] = {}
    for var in variables:
        if var not in data.columns:
            continue
        means: list[float] = []
        for t in time_index:
            mask = data[time_col] == t
            col = data.filter(mask)[var].to_numpy()
            arr = np.asarray(col, dtype=float)
            means.append(float(np.nanmean(arr)) if arr.size else float("nan"))
        result[var] = np.asarray(means, dtype=float)
    return result
