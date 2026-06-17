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
"""Panel data utilities for longitudinal path models.

Provides helpers for preparing panel data (lag creation, validation)
and metadata structures used by the panel-aware compiler and simulator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import narwhals.stable.v1 as nw
from narwhals.stable.v1.typing import IntoFrameT

__all__ = ["PanelInfo"]


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


def add_lags(
    df: IntoFrameT,
    variables: list[str],
    lags: int | list[int],
    panel: dict[str, str],
) -> IntoFrameT:
    """Create lag columns within each panel unit.

    Parameters
    ----------
    df : IntoFrame
        Panel data in long format (pandas or polars DataFrame).
    variables : list[str]
        Column names to lag.
    lags : int | list[int]
        Lag order(s). A single integer ``k`` is treated as ``[1, ..., k]``.
        A list specifies exact lag orders.
    panel : dict[str, str]
        Must contain ``"unit"`` and ``"time"`` keys mapping to column names.

    Returns
    -------
    IntoFrame
        Copy of *df* (same backend as the input) with additional
        ``{var}_lag{k}`` columns, sorted by unit then time. Rows at the
        start of each unit's series will contain null for the lag columns.

    Raises
    ------
    KeyError
        If required columns are missing from *df* or *panel*.
    """
    warnings.warn(
        "add_lags() is deprecated. Use lag(var) syntax in the model "
        "spec instead, e.g. 'sales ~ spend + lag(sales)'. "
        "add_lags() will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    data = nw.from_native(df, eager_only=True)
    _validate_panel_args(data, panel, variables)

    unit_col = panel["unit"]
    time_col = panel["time"]

    if isinstance(lags, int):
        lag_orders = list(range(1, lags + 1))
    else:
        lag_orders = list(lags)

    result = data.sort([unit_col, time_col])
    result = result.with_columns([
        nw.col(var).shift(k).over(unit_col).alias(f"{var}_lag{k}")
        for var in variables
        for k in lag_orders
    ])

    return result.to_native()


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


def build_panel_info(df: nw.DataFrame, panel: dict[str, str]) -> PanelInfo:
    """Build panel metadata from data and panel specification.

    Parameters
    ----------
    df : nw.DataFrame
        Panel data.
    panel : dict[str, str]
        Must contain ``"unit"`` and ``"time"`` keys.

    Returns
    -------
    PanelInfo
        Panel metadata for use by compiler and simulator.
    """
    unit_col = panel["unit"]
    time_col = panel["time"]
    unit_labels = sorted(df[unit_col].unique().to_list())
    return PanelInfo(unit=unit_col, time=time_col, unit_labels=unit_labels)
