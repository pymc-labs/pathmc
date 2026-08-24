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

from dataclasses import dataclass
from typing import Any

import narwhals.stable.v1 as nw

__all__ = ["PanelInfo"]


@dataclass
class PanelInfo:
    """Metadata describing panel structure in the data.

    Parameters
    ----------
    unit : str
        Column name identifying the panel unit. For multi-dimensional
        panels this is the derived composite key column (values joined
        with ``"|"``, e.g. ``"North|Acme"``), present in the working
        frame.
    time : str
        Column name identifying the time index (e.g. ``"week"``).
    unit_labels : list[str]
        Unique unit labels in sorted order.
    unit_columns : tuple[str, ...]
        Original data columns composing the unit identifier. A
        single-element tuple is the classic one-column panel.
    """

    unit: str
    time: str
    unit_labels: list[str]
    unit_columns: tuple[str, ...] = ()

    @property
    def is_multi_dim(self) -> bool:
        """True when the unit key spans more than one data column."""
        return len(self.unit_columns) > 1


_SEPARATOR = "|"


def _require_column(df: nw.DataFrame, col: str, label: str) -> None:
    """Raise ``KeyError`` if *col* is absent from *df*."""
    if col not in df.columns:
        raise KeyError(
            f"{label} '{col}' not found in data. "
            f"Available columns: {', '.join(df.columns)}"
        )


def build_panel_info(
    df: nw.DataFrame, panel: dict[str, Any]
) -> tuple[PanelInfo, nw.DataFrame]:
    """Build panel metadata from data and panel specification.

    Parameters
    ----------
    df : nw.DataFrame
        Panel data.
    panel : dict[str, Any]
        Must contain ``"unit"`` and ``"time"`` keys. ``"unit"`` may be
        a single column name or a list of columns forming a composite
        unit key (multi-dimensional panel); the data must be
        rectangular — every unit shares the same set of time points.

    Returns
    -------
    tuple[PanelInfo, nw.DataFrame]
        Panel metadata and the frame to use downstream. For
        multi-dimensional panels the frame gains a derived composite
        key column (``"|".join(unit_columns)``) that all downstream
        unit indexing reads.
    """
    unit_spec: str | list[str] | tuple[str, ...] = panel["unit"]
    time_col = panel["time"]
    if isinstance(unit_spec, (list, tuple)):
        if not unit_spec:
            raise ValueError(
                "panel['unit'] must name at least one column, got an "
                "empty list. Example: panel={'unit': ['geo', 'brand'], "
                "'time': 'week'}."
            )
        unit_columns = tuple(unit_spec)
    else:
        unit_columns = (unit_spec,)

    for col in (*unit_columns, time_col):
        _require_column(df, col, "Panel column")

    if len(unit_columns) == 1:
        unit_col = unit_columns[0]
    else:
        unit_col = "|".join(unit_columns)
        if unit_col in df.columns:
            raise ValueError(
                f"The derived composite unit key {unit_col!r} collides with "
                f"an existing column. Rename either the column or the unit "
                "columns passed via panel['unit']."
            )
        df = df.with_columns(_composite_unit_column(df, unit_columns).alias(unit_col))

    _require_rectangular(df, unit_col, time_col)
    unit_labels = sorted(df[unit_col].unique().to_list())
    return (
        PanelInfo(
            unit=unit_col,
            time=time_col,
            unit_labels=unit_labels,
            unit_columns=unit_columns,
        ),
        df,
    )


def _composite_unit_column(
    df: nw.DataFrame, unit_columns: tuple[str, ...]
) -> nw.Series:
    """Join the unit columns row-wise into one composite-key series."""
    keys = df[unit_columns[0]].cast(nw.String)
    for col in unit_columns[1:]:
        keys = keys + "|" + df[col].cast(nw.String)
    return keys


def _require_rectangular(df: nw.DataFrame, unit_col: str, time_col: str) -> None:
    """Raise ``ValueError`` if units do not share an identical time grid."""
    grids: dict[str, set[Any]] = {}
    for unit, time in zip(df[unit_col].to_list(), df[time_col].to_list()):
        grids.setdefault(unit, set()).add(time)
    reference = next(iter(grids.values()))
    offenders = sorted(u for u, t in grids.items() if t != reference)
    if offenders:
        shown = ", ".join(repr(u) for u in offenders[:5])
        more = "" if len(offenders) <= 5 else f" (and {len(offenders) - 5} more)"
        raise ValueError(
            f"Panel data is not rectangular: unit(s) {shown}{more} do not "
            f"share the same set of '{time_col}' values as the other units. "
            "Multi-dimensional panels require the Cartesian product of "
            "units × times; drop or fill the missing rows and retry."
        )
