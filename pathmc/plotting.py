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
"""Shared posterior-density rendering for models and result objects.

Centralizes ``plot_dist`` on :class:`~pathmc._model.PathModel`,
:class:`~pathmc.effects.EffectResult`, :class:`~pathmc.simulate.EstimandResult`,
and :class:`~pathmc.simulate.DoResult`. A leaf module so the model layer and
the result objects can both import it without cycles.

``plot_density`` returns a :class:`matplotlib.figure.Figure` and does not
display it. Supported 1-D draw types:

- continuous, non-constant: KDE curve and translucent fill
- constant: a point mass at the unique value (ArviZ's KDE is undefined here)
- integer-valued, non-constant: empirical probability bars on the observed
  integers, with no density between them
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import arviz as az
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["plot_density"]

_SAMPLE_DIMS = ("chain", "draw")
_AGGREGATES = ("time_mean", "unit_pool")
_POINT_MASS_LABEL = "_pathmc_point_mass"


def _as_1d(draws: np.ndarray) -> np.ndarray:
    """Return finite 1-D draws, or raise a message that names the fix."""
    values = np.asarray(draws, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError(
            "draws is empty. Pass a 1-D array with at least one finite sample."
        )
    if not np.isfinite(values).all():
        raise ValueError(
            "draws contains non-finite values. Drop or impute them before plotting."
        )
    return values


def _integer_valued(draws: np.ndarray) -> bool:
    """True when every draw is an integer (Bernoulli, Poisson, and kin)."""
    return bool(np.all(draws == np.round(draws)))


_DENSITY_KIND = "_pathmc_density_kind"


def _density_kind(draws: np.ndarray) -> str:
    """Return the scale a 1-D sample will be drawn on."""
    if np.ptp(draws) == 0 or _integer_valued(draws):
        return "probability"
    return "density"


def _reject_mixed_scale(ax: matplotlib.axes.Axes, kind: str) -> None:
    """Refuse a density curve and a probability series on one axes.

    A point mass pins the y limit near 1, and a later KDE can peak far above
    that. Relabeling the axes would then describe one of the series wrongly.
    """
    if not ax.has_data():
        if hasattr(ax, _DENSITY_KIND):
            delattr(ax, _DENSITY_KIND)
        return
    existing = getattr(ax, _DENSITY_KIND, None)
    if existing is None or existing == kind:
        return
    raise ValueError(
        "Cannot overlay a density curve with a point mass or probability bars "
        "on the same axes: they do not share a scale, and the point-mass "
        "limits would clip the density. Plot them on separate axes."
    )


def _label_scan_axes(
    da: xr.DataArray,
    *,
    n_times: int,
    n_units: int,
    time_values: Sequence[Any],
    unit_labels: Sequence[Any],
) -> xr.DataArray:
    """Rename anonymous trailing ``(n_times, n_units)`` axes to time and unit.

    Scan-compiled panel RVs are stored as ``{name}_dim_0`` / ``{name}_dim_1``.
    ``latent_trajectory`` already treats those trailing axes as time and unit;
    plot_dist must do the same or ``aggregate='time_mean'`` denies a time axis
    that is present. Named coordinates such as ``y_predictors`` are left alone.
    """
    extra = [dim for dim in da.dims if dim not in ("chain", "draw", "time", "unit")]
    if (
        len(extra) < 2
        or "time" in da.dims
        or "unit" in da.dims
        or int(da.sizes[extra[-2]]) != int(n_times)
        or int(da.sizes[extra[-1]]) != int(n_units)
        or "_dim_" not in str(extra[-2])
        or "_dim_" not in str(extra[-1])
    ):
        return da
    return da.rename({extra[-2]: "time", extra[-1]: "unit"}).assign_coords(
        time=list(time_values),
        unit=list(unit_labels),
    )


def _legend(ax: matplotlib.axes.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")


def _coord_value(value: Any) -> Any:
    """Return a Python scalar so the error repr keeps ints unquoted."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sync_probability_ticks(ax: matplotlib.axes.Axes) -> None:
    """Keep bar centers and point-mass locations on the x ticks."""
    xs: list[float] = []
    for line in ax.lines:
        if line.get_marker() == "o" and line.get_linestyle() in ("None", "none"):
            xs.extend(float(x) for x in np.asarray(line.get_xdata(), dtype=float))
    for container in ax.containers:
        for patch in container:
            xs.append(float(patch.get_x() + patch.get_width() / 2))
    if not xs or len(xs) > 20:
        return
    ordered = sorted(xs)
    unique = [ordered[0]]
    for x in ordered[1:]:
        if abs(x - unique[-1]) > 1e-8:
            unique.append(x)
    ax.set_xticks(unique)


def _point_mass_labels(ax: matplotlib.axes.Axes) -> list[Any]:
    return [text for text in ax.texts if getattr(text, _POINT_MASS_LABEL, False)]


def _label_inside_axes(box: Any, ax_box: Any) -> bool:
    return box.x0 >= ax_box.x0 - 1 and box.x1 <= ax_box.x1 + 1


def _measure_renderer(figure: Any) -> Any:
    """Measure text without drawing *figure*.

    ``canvas.draw()`` runs a tight or constrained layout engine and moves
    axes the caller already placed. A private Agg renderer matches the
    canvas size and does not.
    """
    from matplotlib.backends.backend_agg import RendererAgg

    width, height = figure.canvas.get_width_height()
    return RendererAgg(int(width), int(height), float(figure.dpi))


def _set_point_mass_side(ann: Any, *, offset: float, ha: str) -> None:
    ann.xyann = (offset, -14)
    ann.set_ha(ha)


def _reposition_point_mass_labels(ax: matplotlib.axes.Axes) -> None:
    """Move point-mass text inside the axes after later artists change xlim.

    A label placed against the limits at draw time stays put when a later
    series, reference line, or tight_layout moves that point toward an edge.
    The side comes from those final limits. Pixel measurement uses a private
    renderer, so a supplied axes does not run the figure layout engine and a
    canvas without ``get_renderer`` still returns a figure.
    """
    labels = _point_mass_labels(ax)
    figure = ax.figure
    if not labels or figure is None:
        return
    left, right = ax.get_xlim()
    for ann in labels:
        value = float(ann.xy[0])
        if (value - left) >= (right - value):
            _set_point_mass_side(ann, offset=-6, ha="right")
        else:
            _set_point_mass_side(ann, offset=6, ha="left")
    renderer = _measure_renderer(figure)
    for ann in labels:
        box = ann.get_window_extent(renderer)
        ax_box = ax.get_window_extent(renderer)
        if _label_inside_axes(box, ax_box):
            continue
        if ann.get_ha() == "right":
            _set_point_mass_side(ann, offset=6, ha="left")
        else:
            _set_point_mass_side(ann, offset=-6, ha="right")
        box = ann.get_window_extent(renderer)
        ax_box = ax.get_window_extent(renderer)
        if _label_inside_axes(box, ax_box):
            continue
        dx_px = 0.0
        if box.x0 < ax_box.x0:
            dx_px = (ax_box.x0 - box.x0) + 1
        elif box.x1 > ax_box.x1:
            dx_px = (ax_box.x1 - box.x1) - 1
        ann.xyann = (float(ann.xyann[0]) + dx_px * 72.0 / figure.dpi, -14)


def _place_point_mass_label(
    ax: matplotlib.axes.Axes,
    value: float,
    text: str,
    *,
    color: str,
) -> None:
    """Create the point-mass annotation. Final placement is a later pass."""
    ann = ax.annotate(
        text,
        xy=(value, 1.0),
        xytext=(6, -14),
        textcoords="offset points",
        ha="left",
        va="top",
        color=color,
        fontsize=9,
        clip_on=False,
    )
    setattr(ann, _POINT_MASS_LABEL, True)


def _plot_point_mass(
    ax: matplotlib.axes.Axes,
    draws: np.ndarray,
    *,
    label: str | None,
    color: str,
) -> None:
    """Mark a constant sample as a point mass instead of a blank KDE."""
    value = float(draws[0])
    fresh = not ax.has_data()
    vline_kwargs: dict[str, Any] = {"color": color, "lw": 2}
    if label is not None:
        vline_kwargs["label"] = label
    ax.axvline(value, **vline_kwargs)
    ax.plot(
        [value],
        [1.0],
        marker="o",
        color=color,
        ms=8,
        linestyle="None",
        label="_nolegend_",
    )
    _place_point_mass_label(
        ax,
        value,
        f"point mass at {value:g} (n={draws.size})",
        color=color,
    )
    if fresh:
        ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability")
    _sync_probability_ticks(ax)
    _legend(ax)


def _plot_discrete(
    ax: matplotlib.axes.Axes,
    draws: np.ndarray,
    *,
    label: str | None,
    color: str,
) -> None:
    """Plot an empirical PMF. Bars sit on the observed integers only."""
    values, counts = np.unique(draws, return_counts=True)
    probs = counts / counts.sum()
    if values.size == 1:
        width = 0.8
    else:
        gaps = np.diff(values.astype(float))
        width = float(min(0.8, np.min(gaps) * 0.8))
    ax.bar(
        values,
        probs,
        width=width,
        align="center",
        color=color,
        alpha=0.6,
        label=label if label is not None else "_nolegend_",
    )
    _sync_probability_ticks(ax)
    ax.set_ylabel("Probability")
    _legend(ax)


def _plot_kde(
    ax: matplotlib.axes.Axes,
    draws: np.ndarray,
    *,
    label: str | None,
    color: str,
) -> None:
    # Constant arrays are handled before this call. ArviZ 1.2 ``az.kde``
    # returns x=0 and non-finite y for those, which would draw a blank plot.
    x_kde, y_kde, _ = az.kde(draws)
    if not np.isfinite(x_kde).all() or not np.isfinite(y_kde).all():
        raise ValueError(
            "Could not compute a finite density for these draws "
            f"(n={draws.size}, min={float(draws.min())}, max={float(draws.max())}). "
            "Constant draws are drawn as a point mass and integer draws as a "
            "probability mass function; report this if the draws are neither."
        )
    ax.plot(x_kde, y_kde, color=color, lw=2, label=label)
    ax.fill_between(x_kde, y_kde, alpha=0.3, color=color)
    ax.set_ylabel("Density")
    _legend(ax)


def plot_density(
    draws: np.ndarray,
    *,
    label: str | None = None,
    ref: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    color: str = "C0",
) -> matplotlib.figure.Figure:
    """Plot a 1-D sample as a density, a point mass, or an integer PMF.

    Continuous non-constant draws use a KDE curve with a translucent fill.
    Constant draws (including a fixed intervention) are a point mass at that
    value: ArviZ's KDE is undefined for a single value and would otherwise
    return a non-finite curve. Integer-valued draws, such as Bernoulli or
    Poisson predictive samples, are empirical probability bars on the observed
    integers, so the plot does not put mass between values the outcome cannot
    take.

    Parameters
    ----------
    draws : np.ndarray
        Flat ``(n_samples,)`` array. Non-finite values are rejected.
    label : str | None
        Legend label. Omitted when ``None``.
    ref : float | None
        Reference value marked with a dashed vertical line (e.g. ``0`` for a
        null effect). No line when ``None``.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates a new figure if ``None``.
    color : str
        Line and fill color (default ``"C0"``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot. The caller displays or saves it.

    Raises
    ------
    ValueError
        If *draws* is empty, non-finite, or a continuous sample whose KDE
        is non-finite.
    """
    import matplotlib.pyplot as plt

    values = _as_1d(draws)
    if ax is None:
        fig, target = plt.subplots(figsize=(8, 4))
        created = True
    else:
        target = ax
        figure = target.get_figure()
        if figure is None or target not in figure.axes:
            raise ValueError(
                "ax is not attached to a figure. Pass an axes from "
                "plt.subplots(), or omit ax to create a new figure."
            )
        fig = cast("matplotlib.figure.Figure", figure)
        created = False

    kind = _density_kind(values)
    _reject_mixed_scale(target, kind)
    setattr(target, _DENSITY_KIND, kind)
    if kind == "probability" and np.ptp(values) == 0:
        _plot_point_mass(target, values, label=label, color=color)
    elif kind == "probability":
        _plot_discrete(target, values, label=label, color=color)
    else:
        _plot_kde(target, values, label=label, color=color)
    if ref is not None:
        target.axvline(ref, color=color, linestyle="--", linewidth=1.5)
    # A caller-supplied axes may sit beside other panels. tight_layout would
    # move those siblings.
    if created:
        fig.tight_layout()
    _reposition_point_mass_labels(target)
    return fig


def _reject_unused_coords(
    coords: Mapping[str, Any] | None,
    applied: set[str],
    names: Sequence[str],
) -> None:
    """Raise when a requested selector matched no selected variable."""
    if not coords:
        return
    unused = [key for key in coords if key not in applied]
    if unused:
        raise ValueError(
            f"coords keys {unused} did not match any dimension of {list(names)}. "
            "Drop selectors that none of the selected variables have, or check "
            "the coordinate names."
        )


def _select_draws(
    da: xr.DataArray,
    *,
    name: str,
    coords: Mapping[str, Any] | None = None,
    aggregate: str | None = None,
    time_inspection: str = "Inspect the per-time draws separately. ",
) -> tuple[np.ndarray, str, set[str]]:
    """Reduce *da* to a 1-D sample and a legend label.

    Applies only *coords* keys that are dimensions of *da*. Size-1 non-sample
    dimensions need no selector. Every remaining non-sample dimension of size
    greater than 1 must be selected, except ``time`` and ``unit``, which
    require an explicit aggregate instead of being silently collapsed.

    Returns the flat draws, the legend label, and the coord keys that were
    applied to this variable.
    """
    if aggregate not in (None, *_AGGREGATES):
        raise ValueError(
            f"Unknown aggregate={aggregate!r}. "
            "Supported values: 'time_mean', 'unit_pool', or omit aggregate."
        )

    selected = da
    applied: set[str] = set()
    if coords:
        applicable = {
            key: value for key, value in coords.items() if key in selected.dims
        }
        if applicable:
            try:
                selected = selected.sel(applicable)
            except KeyError as err:
                available = {
                    dim: [_coord_value(v) for v in selected.coords[dim].values]
                    for dim in applicable
                    if dim in selected.coords
                }
                raise KeyError(
                    f"coords {applicable} did not match '{name}'. "
                    f"Available values: {available}. "
                    "Pass a coordinate value that exists on that dimension."
                ) from err
            applied = set(applicable)

    label = name
    selected_bits = [
        f"{key}={value}"
        for key, value in (coords or {}).items()
        if key in applied and key not in _SAMPLE_DIMS
    ]
    if selected_bits:
        label = f"{label} [{', '.join(selected_bits)}]"

    n_times = int(selected.sizes["time"]) if "time" in selected.dims else 0
    if n_times > 1 and aggregate != "time_mean":
        times = [str(v) for v in selected.coords["time"].values]
        shown = ", ".join(times[:6]) + ("" if len(times) <= 6 else ", ...")
        ignored = (
            "aggregate='unit_pool' does not collapse time. "
            if aggregate == "unit_pool"
            else ""
        )
        raise ValueError(
            f"'{name}' has {n_times} time points ({shown}). "
            "plot_dist() will not average over time and present that as the "
            f"posterior of '{name}'. {ignored}{time_inspection.format(name=name)}"
            "To plot the density of the time-averaged draws, pass "
            "aggregate='time_mean' (the legend will mark it as a time mean)."
        )
    if aggregate == "time_mean" and n_times <= 1:
        raise ValueError(
            f"aggregate='time_mean' requires '{name}' to have more than one "
            "time point. Omit aggregate when the variable is already univariate "
            "in time."
        )
    if aggregate == "time_mean":
        values = np.asarray(selected.values)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"'{name}' contains non-finite values, so aggregate='time_mean' "
                "would skip missing times and still be labeled a time mean. "
                "Drop or impute those draws before plotting."
            )
        selected = selected.mean(dim="time", skipna=False)
        label = f"{label} (time mean)"
    elif "time" in selected.dims:
        selected = selected.squeeze("time", drop=True)

    n_units = int(selected.sizes["unit"]) if "unit" in selected.dims else 0
    if n_units > 1 and aggregate != "unit_pool":
        ignored = (
            "aggregate='time_mean' does not select or pool units. "
            if aggregate == "time_mean"
            else ""
        )
        raise ValueError(
            f"'{name}' has {n_units} units. plot_dist() will not pool them into "
            f"one density labeled '{name}'. {ignored}"
            "Select one unit with coords={'unit': <id>}, or pass "
            "aggregate='unit_pool' to plot the pooled distribution "
            "(the legend will say it is pooled over unit)."
        )
    if aggregate == "unit_pool" and n_units <= 1:
        raise ValueError(
            f"aggregate='unit_pool' requires '{name}' to have more than one "
            "unit. Omit aggregate when the variable has no unit dimension, "
            "or select the single unit implicitly by omitting coords."
        )
    if aggregate == "unit_pool":
        label = f"{label} (pooled over unit)"
    elif "unit" in selected.dims:
        selected = selected.squeeze("unit", drop=True)

    extra = [
        dim
        for dim in selected.dims
        if dim not in (*_SAMPLE_DIMS, "unit", "time") and int(selected.sizes[dim]) > 1
    ]
    if extra:
        hint = ", ".join(f"{dim!r}: '<value>'" for dim in extra)
        raise ValueError(
            f"'{name}' has multiple coordinates on dimension(s) {extra} "
            "(size > 1) after selection. Pass coords to select one value on "
            f"each remaining dimension, e.g. coords={{{hint}}}."
        )
    squeeze_dims = [
        dim
        for dim in selected.dims
        if dim not in (*_SAMPLE_DIMS, "unit") and int(selected.sizes[dim]) <= 1
    ]
    if squeeze_dims:
        selected = selected.squeeze(dim=squeeze_dims, drop=True)

    stack_dims = [dim for dim in (*_SAMPLE_DIMS, "unit") if dim in selected.dims]
    if stack_dims:
        stacked = selected.stack(sample=stack_dims).transpose("sample")
        flat = np.asarray(stacked.values).ravel()
    else:
        flat = np.asarray(selected.values).ravel()
    return flat, label, applied
