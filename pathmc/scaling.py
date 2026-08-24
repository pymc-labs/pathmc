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
"""Scaling of heterogeneous units onto a common internal scale.

Estimating a shared coefficient across units measured in different
magnitudes (large vs. small geos, high- vs. low-volume brands) makes
default priors badly calibrated for some units. Passing a
:class:`Scaling` object via ``scaling=`` to :func:`pathmc.model` divides
the affected data columns by fitted scale *factors* before compilation,
so estimation happens on a common internal scale — the same pattern as
``pymc_marketing.mmm.scaling.Scaling`` / ``FixedScaling``, but
domain-general: any panel outcome or predictor can be scaled, not just
MMM targets and media channels.

The fitted factors are stored on the returned :class:`~pathmc.PathModel`
(accessible as ``model.fitted_scaling``) so that
:func:`pathmc.simulate` can apply the inverse transform and return
generated columns in their original business units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw
import numpy as np

if TYPE_CHECKING:
    from xarray import DataArray

__all__ = ["Scaling", "ScalingFactors"]

#: Supported scaling methods.
_METHODS = ("max", "mean", "fixed", "divide")

#: Per-role configuration keys each method accepts.
_ALLOWED_KEYS = {
    "max": {"method", "dims"},
    "mean": {"method", "dims"},
    "fixed": {"method", "dims", "value"},
    "divide": {"method", "dims", "by"},
}


@dataclass
class Scaling:
    """Scaling configuration passed as ``scaling=`` to ``model()`` / ``simulate()``.

    Each role (``target`` for endogenous/outcome columns, ``channel``
    for exogenous/predictor columns) takes a spec dictionary:

    - ``"method"`` (required): one of ``"max"``, ``"mean"``, ``"fixed"``,
      or ``"divide"``.
    - ``"dims"``: panel unit columns defining the groups the scaling is
      computed per (e.g. ``("geo",)``). Default ``()`` — a single global
      scale. Every name must appear in ``panel["unit"]``.
    - ``"value"`` (``"fixed"`` only): a positive constant, or a grid
      (:class:`xarray.DataArray` or dict keyed like ``"by"``) of
      constants per unit.
    - ``"by"`` (``"divide"`` only): an external grid —
      :class:`xarray.DataArray` whose coords are named after ``dims``,
      or a dict mapping unit labels (single dim) or label tuples /
      ``"|"-joined composite labels`` (multiple dims) to divisors.

    ``"max"`` / ``"mean"`` divide each column by the group-wise maximum
    or mean of that same column; ``"fixed"`` divides by a supplied
    constant or grid; ``"divide"`` divides by an external grid such as
    population.

    This mirrors ``pymc_marketing.mmm.scaling.Scaling(method="max",
    dims=...)`` and ``FixedScaling(values=<DataArray>)``: ``target`` /
    ``channel`` correspond to pymc-marketing's identically-named MMM
    scaling slots, while remaining usable for arbitrary panels.

    Example::

        scaling = pathmc.Scaling(
            target={"method": "max", "dims": ("geo",)},
            channel={"method": "divide", "by": population_da},
        )
        model = pathmc.model(spec, data=df, panel=panel, scaling=scaling)
    """

    target: dict[str, Any] | None = None
    channel: dict[str, Any] | None = None


@dataclass
class ScalingFactors:
    """Fitted per-row scale factors, as returned by ``model.fitted_scaling``.

    Each entry maps a scaled data column to a divisor array with one
    element per data row (a constant grid broadcasts to a constant
    array). Forward scaling divides by these values; the inverse
    transform multiplies by them.

    Pass a fitted instance back as ``scaling=`` to
    :func:`pathmc.simulate` (or :func:`pathmc.model`) to reuse the exact
    estimation-time scales instead of refitting from data.
    """

    #: Column name -> per-row divisor array.
    factors: dict[str, np.ndarray] = field(default_factory=dict)

    def transform(self, df: nw.DataFrame) -> nw.DataFrame:
        """Divide the fitted columns of *df* by their scale factors.

        Columns with fitted factors but no matching *df* column (e.g.
        simulated outcomes) are skipped.
        """
        new_cols: list[nw.Series] = []
        for col, factor in self.factors.items():
            if col not in df.columns:
                continue
            vals = df[col].to_numpy().astype(float)
            new_cols.append(
                nw.new_series(col, vals / factor, backend=df.implementation)
            )
        return df.with_columns(new_cols) if new_cols else df

    def inverse_transform_column(self, values: np.ndarray, column: str) -> np.ndarray:
        """Multiply *values* by the fitted factor of *column* (the inverse transform)."""
        return np.asarray(values, dtype=float) * self.factors[column]


def _validate_spec(role: str, cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Validate one role's config dict and return it unchanged."""
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError(
            f"Scaling.{role} must be a dict like "
            f'{{"method": "max", "dims": ("geo",)}}, got {type(cfg).__name__}.'
        )
    if "method" not in cfg:
        raise ValueError(
            f'Scaling.{role} requires a "method" key (one of {", ".join(_METHODS)}).'
        )
    method = cfg["method"]
    if method not in _METHODS:
        raise ValueError(
            f"Unknown Scaling.{role} method {method!r}. "
            f"Valid methods: {', '.join(_METHODS)}."
        )
    extra = set(cfg) - _ALLOWED_KEYS[method]
    if extra:
        raise ValueError(
            f"Scaling.{role} keys {sorted(extra)} are not valid for "
            f"method {method!r}. Allowed keys: {sorted(_ALLOWED_KEYS[method])}."
        )
    if method == "fixed" and "value" not in cfg:
        raise ValueError(
            'Scaling method "fixed" requires a "value": a positive constant or a grid.'
        )
    if method == "divide" and "by" not in cfg:
        raise ValueError(
            'Scaling method "divide" requires "by": an external grid '
            "(xarray DataArray or dict) keyed by the unit dims."
        )
    dims = cfg.get("dims")
    if dims is not None and not isinstance(dims, (list, tuple)):
        raise TypeError(
            f'Scaling.{role} "dims" must be a tuple of panel unit column '
            f"names, e.g. ('geo',), got {type(dims).__name__}."
        )
    return cfg


def _dataarray_grid(da: Any, dims: tuple[str, ...]) -> dict[tuple[str, ...], float]:
    """Flatten an xarray DataArray into unit-key -> divisor entries."""
    import xarray as xr

    if not isinstance(da, xr.DataArray):
        raise TypeError(
            "Scaling grids accept an xarray.DataArray or a dict, got "
            f"{type(da).__name__}."
        )
    if set(da.dims) != set(dims):
        raise ValueError(
            f"Scaling grid coords {tuple(da.dims)} do not match the "
            f"configured dims {dims}. Name the DataArray's dimensions "
            "after the panel unit columns."
        )
    axis_labels = [[str(v) for v in da.coords[dim].values] for dim in da.dims]
    out: dict[tuple[str, ...], float] = {}
    for idx in np.ndindex(*(len(labels) for labels in axis_labels)):
        key: list[str] = [""] * len(dims)
        for pos, dim in enumerate(da.dims):
            key[list(dims).index(str(dim))] = axis_labels[pos][idx[pos]]
        out[tuple(key)] = float(np.asarray(da.values)[idx])
    return out


def _as_grid(
    grid: "DataArray | dict[str, Any]", dims: tuple[str, ...]
) -> dict[tuple[str, ...], float]:
    """Normalize a grid into a mapping from unit-key tuples to divisors."""
    if hasattr(grid, "dims") and hasattr(grid, "coords"):
        return _dataarray_grid(grid, dims)
    if not isinstance(grid, dict):
        raise TypeError(
            "Scaling grids accept an xarray.DataArray or a dict, got "
            f"{type(grid).__name__}."
        )
    out: dict[tuple[str, ...], float] = {}
    normed: tuple[str, ...]
    for key, val in grid.items():
        if isinstance(key, str) and "|" in key and len(dims) > 1:
            normed = tuple(key.split("|"))
        elif isinstance(key, (list, tuple)):
            normed = tuple(str(k) for k in key)
        else:
            normed = (str(key),)
        out[normed] = float(val)
    return out


def _check_dims(dims: tuple[str, ...], panel_info: Any) -> None:
    """Raise unless every dim names a panel unit column."""
    if not dims:
        return
    if panel_info is None:
        raise ValueError(
            "Scaling dims require a panel model: pass panel={'unit': ..., "
            "'time': ...} alongside scaling=."
        )
    known = panel_info.unit_columns
    unknown = [d for d in dims if d not in known]
    if unknown:
        raise ValueError(
            f"Scaling dims {tuple(unknown)} are not panel unit columns. "
            f"panel['unit'] columns: {list(known)}."
        )


def _group_stat(
    values: np.ndarray, keys: list[tuple[str, ...]], method: str
) -> np.ndarray:
    """Per-row group max/mean of *values* grouped by *keys*."""
    stats: dict[tuple[str, ...], float] = {}
    buckets: dict[tuple[str, ...], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)
    for key, rows in buckets.items():
        subset = values[rows]
        stat = float(np.max(subset) if method == "max" else np.mean(subset))
        if not np.isfinite(stat) or stat <= 0:
            raise ValueError(
                f"Cannot scale by method {method!r}: the group(s) {key!r} "
                f"have a non-positive or non-finite column statistic "
                f"({stat}). Scales must be positive and finite; use "
                "'fixed' or 'divide' with explicit scales instead."
            )
        stats[key] = stat
    return np.array([stats[key] for key in keys])


def _grid_factors(
    grid: DataArray | dict[str, Any],
    dims: tuple[str, ...],
    row_keys: list[tuple[str, ...]],
) -> np.ndarray:
    """Per-row divisors looked up in an external grid."""
    table = _as_grid(grid, dims)
    missing = sorted({key for key in row_keys if key not in table})
    if missing:
        shown = ", ".join(repr(m) for m in missing[:5])
        more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise KeyError(
            f"Scaling grid is missing entries for unit(s) {shown}{more}. "
            f"Grid covers {len(table)} unit(s); every observed combination "
            "of the configured dims must have an entry."
        )
    factors_ = np.array([table[key] for key in row_keys])
    if np.any(factors_ == 0):
        zeroed = sorted({key for key, val in zip(row_keys, factors_) if val == 0})
        raise ValueError(
            f"Scaling grid contains zero entries for unit(s) {zeroed[:5]}: "
            "dividing by zero would produce inf/nan. Every divisor must be "
            "non-zero."
        )
    return factors_


def _row_keys(df: nw.DataFrame, dims: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Composite unit key per row over the given dim columns."""
    if not dims:
        return [()] * len(df)
    columns = [df[d].to_numpy().astype(str).tolist() for d in dims]
    return list(zip(*columns))


def _numeric_columns(df: nw.DataFrame, candidates: set[str]) -> list[str]:
    """Candidates present in *df* with a numeric dtype, in stable order."""
    out: list[str] = []
    for col in sorted(candidates):
        if col in df.columns and df[col].dtype.is_numeric():
            out.append(col)
    return out


def _fit_role(
    role: str,
    cfg: dict[str, Any],
    df: nw.DataFrame,
    panel_info: Any,
    candidates: set[str],
    has_data: bool,
) -> dict[str, np.ndarray]:
    """Fit per-row divisors for one role's columns."""
    method = cfg["method"]
    dims = tuple(cfg.get("dims") or ())
    _check_dims(dims, panel_info)

    if method in ("max", "mean"):
        if not has_data:
            raise ValueError(
                f"Scaling.{role} method {method!r} derives its scale from "
                f"the {role} column values themselves, which do not exist "
                f"when simulating. Pass method='fixed' or 'divide' with the "
                "estimation-time scales (available on "
                "fitted_model.fitted_scaling), or pass the fitted "
                "ScalingFactors object directly as scaling=."
            )
        columns = _numeric_columns(df, candidates)
        if not columns and candidates:
            raise ValueError(
                f"Scaling.{role} was configured, but none of its candidate "
                f"columns ({sorted(candidates)}) were found in the data as "
                "numeric columns."
            )
    else:
        # Grid-based methods also fit factors for generated (simulated)
        # columns that have no observed data column yet.
        columns = sorted(candidates)
        for col in columns:
            if col in df.columns and not df[col].dtype.is_numeric():
                raise ValueError(
                    f"Scaling.{role} targets non-numeric column {col!r}; "
                    "only numeric columns can be scaled."
                )

    if method in ("max", "mean"):
        factors: dict[str, np.ndarray] = {}
        for col in columns:
            keys = _row_keys(df, dims)
            factors[col] = _group_stat(df[col].to_numpy().astype(float), keys, method)
        return factors

    if method == "fixed":
        value = cfg["value"]
        keys = _row_keys(df, dims)
        if np.isscalar(value) and not hasattr(value, "dims"):
            factor = float(value)  # type: ignore[arg-type]
            if factor == 0:
                raise ValueError("Scaling 'fixed' value must be non-zero.")
            return {col: np.full(len(keys), factor) for col in columns}
        return {col: _grid_factors(value, dims, keys) for col in columns}

    # method == "divide"
    if not dims:
        raise ValueError(
            'Scaling method "divide" requires "dims" naming the panel unit '
            "columns the external grid is keyed by."
        )
    keys = _row_keys(df, dims)
    grid_factors = _grid_factors(cfg["by"], dims, keys)
    return {col: grid_factors.copy() for col in columns}


def fit_scaling(
    scaling: Scaling | ScalingFactors,
    df: nw.DataFrame,
    *,
    panel_info: Any = None,
    target_columns: set[str] | None = None,
    channel_columns: set[str] | None = None,
    roles_with_data: frozenset[str] | set[str] = frozenset({"target", "channel"}),
) -> ScalingFactors:
    """Fit per-row scale factors for the columns affected by *scaling*.

    Parameters
    ----------
    scaling : Scaling | ScalingFactors
        Configuration. A pre-fitted :class:`ScalingFactors` is returned
        unchanged (its columns must exist in *df* only when transformed).
    df : nw.DataFrame
        Data the factors are computed from. Roles outside
        *roles_with_data* must not use data-derived methods (``max`` /
        ``mean``).
    panel_info : PanelInfo | None
        Panel metadata; required whenever a spec declares non-empty
        ``dims``.
    target_columns, channel_columns : set[str] | None
        Candidate columns per role (endogenous / exogenous variables);
        intersected with *df* and filtered to numeric dtypes.
    roles_with_data : set[str]
        Which roles' columns exist in *df*. ``simulate()`` passes only
        ``{"channel"}``: generated outcomes are absent, so target specs
        there must be grid-based (``fixed`` / ``divide``).

    Returns
    -------
    ScalingFactors
    """
    if scaling is None:
        return ScalingFactors()
    if isinstance(scaling, ScalingFactors):
        return scaling
    if not isinstance(scaling, Scaling):
        raise TypeError(
            "scaling= accepts pathmc.Scaling or a fitted "
            f"pathmc.ScalingFactors, got {type(scaling).__name__}."
        )

    factors: dict[str, np.ndarray] = {}
    for role, candidates in (
        ("target", target_columns or set()),
        ("channel", channel_columns or set()),
    ):
        cfg = _validate_spec(role, getattr(scaling, role))
        if cfg:
            factors.update(
                _fit_role(
                    role,
                    cfg,
                    df,
                    panel_info,
                    candidates,
                    has_data=role in roles_with_data,
                )
            )
    return ScalingFactors(factors=factors)


def validate_scaling_config(scaling: Any) -> None:
    """Validate a ``scaling=`` argument structurally, without data.

    Used by :func:`pathmc.simulate_params_template`, which reports
    parameter shapes that scaling does not change but still accepts the
    argument for forward compatibility with :func:`pathmc.simulate`.
    """
    if scaling is None:
        return
    if isinstance(scaling, (Scaling, ScalingFactors)):
        if isinstance(scaling, Scaling):
            _validate_spec("target", scaling.target)
            _validate_spec("channel", scaling.channel)
        return
    raise TypeError(
        "scaling= accepts pathmc.Scaling or a fitted pathmc.ScalingFactors, "
        f"got {type(scaling).__name__}."
    )
