"""Panel data utilities for longitudinal path models.

Provides helpers for preparing panel data (lag creation, validation)
and metadata structures used by the panel-aware compiler and simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    df: pd.DataFrame,
    variables: list[str],
    lags: int | list[int],
    panel: dict[str, str],
) -> pd.DataFrame:
    """Create lag columns within each panel unit.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data in long format.
    variables : list[str]
        Column names to lag.
    lags : int | list[int]
        Lag order(s). A single integer ``k`` is treated as ``[1, ..., k]``.
        A list specifies exact lag orders.
    panel : dict[str, str]
        Must contain ``"unit"`` and ``"time"`` keys mapping to column names.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with additional ``{var}_lag{k}`` columns, sorted by
        unit then time. Rows at the start of each unit's series will
        contain ``NaN`` for the lag columns.

    Raises
    ------
    KeyError
        If required columns are missing from *df* or *panel*.
    """
    _validate_panel_args(df, panel, variables)

    unit_col = panel["unit"]
    time_col = panel["time"]

    if isinstance(lags, int):
        lag_orders = list(range(1, lags + 1))
    else:
        lag_orders = list(lags)

    result = df.sort_values([unit_col, time_col]).copy()
    result.reset_index(drop=True, inplace=True)

    grouped = result.groupby(unit_col, sort=False)
    for var in variables:
        for k in lag_orders:
            result[f"{var}_lag{k}"] = grouped[var].shift(k)

    return result


def _validate_panel_args(
    df: pd.DataFrame,
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

    if unit_col not in df.columns:
        raise KeyError(
            f"Unit column '{unit_col}' not found in data. "
            f"Available columns: {', '.join(df.columns)}"
        )
    if time_col not in df.columns:
        raise KeyError(
            f"Time column '{time_col}' not found in data. "
            f"Available columns: {', '.join(df.columns)}"
        )

    for var in variables:
        if var not in df.columns:
            raise KeyError(
                f"Variable '{var}' not found in data. "
                f"Available columns: {', '.join(df.columns)}"
            )


def build_panel_info(df: pd.DataFrame, panel: dict[str, str]) -> PanelInfo:
    """Build panel metadata from data and panel specification.

    Parameters
    ----------
    df : pd.DataFrame
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
    unit_labels = sorted(df[unit_col].unique().tolist())
    return PanelInfo(unit=unit_col, time=time_col, unit_labels=unit_labels)
