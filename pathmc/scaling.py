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
generated columns in their original business units, and so that
:meth:`~pathmc.PathModel.do` can divide ``set`` values from business
units into the internal scale, and user-facing outputs (``do()``,
``predict()``, ``effects_summary()``, and related helpers) are returned
in business units.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import blake2b
from typing import TYPE_CHECKING, Any, Literal, Mapping, cast

import narwhals.stable.v1 as nw
import numpy as np
from narwhals.stable.v1.typing import IntoFrame

from pathmc.panel import _SEPARATOR

if TYPE_CHECKING:
    from pathmc.parse import Term
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

ScaleKind = Literal["outcome", "regressor", "coefficient", "elasticity", "derived"]
ScaleDirection = Literal["internal", "business"]

_SCALE_TAG = "_pathmc_scale_units"


@dataclass(frozen=True)
class ScaleContext:
    """Alignment metadata for one conversion at the scaling boundary.

    ``term`` identifies the semantic variable or coefficient term. ``data``
    supplies panel keys and row weights. ``scan_info`` describes the
    time-major layout emitted by the scan compiler. ``scalar`` controls the
    only ambiguous case: reducing heterogeneous per-unit factors to one
    number.
    """

    term: str | "Term" | None = None
    outcome: str | None = None
    data: nw.DataFrame | None = None
    scan_info: Any | None = None
    scalar: Literal["uniform", "mean"] = "uniform"
    columns: tuple[str, ...] | None = None


class _ConvertedArray(np.ndarray):
    """NumPy array subclass carrying conversion state."""

    def __array_finalize__(self, source: Any) -> None:
        """Preserve unit provenance through NumPy views and ufunc results."""
        if source is None:
            return
        state = getattr(source, _SCALE_TAG, None)
        if isinstance(state, Mapping):
            setattr(self, _SCALE_TAG, dict(state))


class _ConvertedFloat(float):
    """Float subclass that retains conversion state through scalar arithmetic."""

    def _tag_result(self, value: Any) -> Any:
        if value is NotImplemented:
            return value
        if not isinstance(value, (int, float, np.integer, np.floating)):
            return value
        tagged = _ConvertedFloat(value)
        state = _conversion_state(self)
        if state:
            setattr(tagged, _SCALE_TAG, state)
        return tagged

    def _binary(self, other: Any, operation: Any) -> Any:
        other_value = float(other) if isinstance(other, _ConvertedFloat) else other
        return self._tag_result(operation(float(self), other_value))

    def _reverse_binary(self, other: Any, operation: Any) -> Any:
        other_value = float(other) if isinstance(other, _ConvertedFloat) else other
        return self._tag_result(operation(other_value, float(self)))

    def __add__(self, other: Any) -> Any:
        return self._binary(other, float.__add__)

    def __radd__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__add__)

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, float.__sub__)

    def __rsub__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__sub__)

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, float.__mul__)

    def __rmul__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__mul__)

    def __truediv__(self, other: Any) -> Any:
        return self._binary(other, float.__truediv__)

    def __rtruediv__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__truediv__)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary(other, float.__floordiv__)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__floordiv__)

    def __mod__(self, other: Any) -> Any:
        return self._binary(other, float.__mod__)

    def __rmod__(self, other: Any) -> Any:
        return self._reverse_binary(other, float.__mod__)

    def __pow__(self, other: float, modulo: None = None) -> Any:
        return self._binary(other, float.__pow__)

    def __rpow__(self, other: float, modulo: None = None) -> Any:
        return self._reverse_binary(other, float.__pow__)

    def __neg__(self) -> _ConvertedFloat:
        return self._tag_result(float.__neg__(self))

    def __pos__(self) -> _ConvertedFloat:
        return self._tag_result(float.__pos__(self))

    def __abs__(self) -> _ConvertedFloat:
        return self._tag_result(float.__abs__(self))


ScaleDims = ScaleContext | Mapping[str, Any] | IntoFrame | None


def _coerce_context(dims: ScaleDims) -> ScaleContext:
    """Normalize the public ``dims=`` argument into conversion metadata."""
    if dims is None:
        return ScaleContext()
    if isinstance(dims, ScaleContext):
        return dims
    if isinstance(dims, Mapping):
        allowed = {"term", "outcome", "data", "scan_info", "scalar", "columns"}
        extra = set(dims) - allowed
        if extra:
            raise ValueError(
                f"Unknown scaling dimension context keys {sorted(extra)}. "
                f"Valid keys are {sorted(allowed)}."
            )
        return ScaleContext(**dims)
    if isinstance(dims, nw.DataFrame):
        return ScaleContext(data=dims)
    try:
        return ScaleContext(data=nw.from_native(dims, eager_only=True))
    except TypeError as exc:
        raise TypeError(
            "dims= must be a data frame, a scaling context mapping, or None; "
            f"got {type(dims).__name__}."
        ) from exc


def _term_key(term: Any) -> str:
    """Stable tag key for a raw variable or parsed coefficient term."""
    if term is None:
        return "<none>"
    if isinstance(term, str):
        return term
    return repr(term)


def _factor_fingerprint(factor: Any) -> str:
    """Return a stable signature for one fully resolved conversion factor."""
    values = np.asarray(factor, dtype=np.float64)
    digest = blake2b(digest_size=16)
    digest.update(repr(values.shape).encode())
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _state_key(term: Any, factor: Any) -> str:
    """Identify conversion semantics by term and resolved factor, not identity."""
    return f"{_term_key(term)}:{_factor_fingerprint(factor)}"


def _conversion_state(obj: Any) -> dict[str, ScaleDirection]:
    """Return a defensive copy of conversion tags carried by *obj*."""
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, Mapping):
        state = attrs.get(_SCALE_TAG, {})
        if isinstance(state, Mapping):
            return dict(state)
    state = getattr(obj, _SCALE_TAG, {})
    return dict(state) if isinstance(state, Mapping) else {}


def _tag_conversion(
    obj: Any,
    *,
    key: str,
    direction: ScaleDirection,
) -> Any:
    """Attach conversion state without changing the object's public shape."""
    state = _conversion_state(obj)
    state[key] = direction
    attrs = getattr(obj, "attrs", None)
    if isinstance(attrs, dict):
        attrs[_SCALE_TAG] = state
        return obj
    if isinstance(obj, np.ndarray):
        tagged = obj.view(_ConvertedArray)
        setattr(tagged, _SCALE_TAG, state)
        return tagged
    if np.isscalar(obj):
        tagged_float = _ConvertedFloat(float(cast(Any, obj)))
        setattr(tagged_float, _SCALE_TAG, state)
        return tagged_float
    try:
        setattr(obj, _SCALE_TAG, state)
    except (AttributeError, TypeError):
        pass
    return obj


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
    """Fitted scale factors keyed by unit, as returned by ``model.fitted_scaling``.

    Each entry maps a scaled data column to ``(dims, table)`` where
    *dims* names the panel unit columns the table is keyed over (empty
    for a single global scale) and *table* maps unit-key tuples to
    divisors. Factors are materialized per data row on demand so the
    same object can be reused on reordered or subset frames.

    Pass a fitted instance back as ``scaling=`` to
    :func:`pathmc.simulate` (or :func:`pathmc.model`) to reuse the exact
    estimation-time scales instead of refitting from data.
    """

    #: column -> (dims, {unit_key_tuple: divisor})
    factors: dict[str, tuple[tuple[str, ...], dict[tuple[str, ...], float]]] = field(
        default_factory=dict
    )
    #: Semantic roles fitted for each column. Empty on legacy/manual objects.
    roles: dict[str, frozenset[Literal["outcome", "regressor"]]] = field(
        default_factory=dict
    )

    def __bool__(self) -> bool:
        """Whether this object contains any fitted factors."""
        return bool(self.factors)

    @property
    def columns(self) -> tuple[str, ...]:
        """Columns with fitted scale factors, in stable order."""
        return tuple(sorted(self.factors))

    def has_factor(self, column: str) -> bool:
        """Return whether *column* has a fitted scale factor."""
        return column in self.factors

    def columns_present_in(self, columns: Any) -> tuple[str, ...]:
        """Return fitted columns present in a frame-like column collection."""
        available = set(columns)
        return tuple(column for column in self.columns if column in available)

    def required_dimensions(self, columns: Any) -> tuple[str, ...]:
        """Return grouping dimensions needed to convert the given columns."""
        required: set[str] = set()
        for column in self.columns_present_in(columns):
            dims, _table = self.factors[column]
            required.update(dims)
        return tuple(sorted(required))

    def _per_row(self, df: nw.DataFrame, column: str) -> np.ndarray:
        """Expand unit-keyed divisors to one value per row of *df*."""
        dims, table = self.factors[column]
        keys = _row_keys(df, dims)
        missing = sorted({k for k in keys if k not in table})
        if missing:
            fitted = sorted(table)[:5]
            raise KeyError(
                f"Scaling factors for column {column!r} have no entry for "
                f"unit(s) {missing[:5]}. The factors were fitted on units "
                f"{fitted}...; pass a frame whose {dims or 'global'} values are "
                "a subset of the estimation-time units, or refit the scaling."
            )
        return np.array([table[k] for k in keys])

    def _scalar_factor(
        self,
        column: str,
        context: ScaleContext,
    ) -> float:
        """Resolve one factor, rejecting lossy heterogeneous reductions."""
        if column not in self.factors:
            return 1.0
        _dims, table = self.factors[column]
        unique = {float(value) for value in table.values()}
        if len(unique) == 1:
            return next(iter(unique))
        if context.scalar == "mean":
            if context.data is None:
                raise ValueError(
                    f"Converting {column!r} to one number with heterogeneous "
                    "scaling factors requires dims={'data': frame, 'scalar': "
                    "'mean'} so the reduction can use fitted row weights."
                )
            return float(np.mean(self._per_row(context.data, column)))
        raise ValueError(
            f"Cannot convert scalar {column!r}: fitted factors differ across "
            "units. Pass row-aligned values, or explicitly request the "
            "data-weighted approximation with dims={'data': frame, "
            "'scalar': 'mean'}."
        )

    def _coefficient_factor(self, term: Any, context: ScaleContext) -> float:
        """Internal-to-business multiplier for one regression coefficient."""
        if context.outcome is None:
            raise ValueError(
                "Coefficient conversion requires dims={'outcome': ..., "
                "'term': ..., 'data': ...}."
            )
        mean_context = replace(context, scalar="mean")
        outcome_factor = self._scalar_factor(context.outcome, mean_context)
        if term is None:
            return outcome_factor
        if isinstance(term, str):
            if term == "Intercept":
                return outcome_factor
            return outcome_factor / self._scalar_factor(term, mean_context)
        if getattr(term, "variable", None) == "Intercept":
            return outcome_factor
        if getattr(term, "basis", None) is not None:
            return outcome_factor
        interaction = getattr(term, "interaction_of", None)
        if interaction is not None:
            factor = outcome_factor
            for part in interaction:
                factor /= self._scalar_factor(part, mean_context)
            return factor
        transform = getattr(term, "transform", None)
        if transform is not None:
            from pathmc.transforms import get_transform

            current = transform
            while current is not None and not isinstance(current, str):
                transform_impl = get_transform(current.name)
                homogeneous = transform_impl.homogeneous
                if homogeneous is None:
                    raise ValueError(
                        f"Cannot rescale a coefficient on transform "
                        f"{current.name!r} to business units: the transform "
                        "does not declare whether it is homogeneous in its "
                        "input. Set its homogeneous attribute to True or False."
                    )
                if not homogeneous:
                    return outcome_factor
                current = current.input_expr
        predictor = getattr(term, "lag_of", None) or getattr(term, "variable", None)
        if not isinstance(predictor, str):
            return outcome_factor
        return outcome_factor / self._scalar_factor(predictor, mean_context)

    def _factor_for(
        self,
        obj: Any,
        *,
        kind: ScaleKind,
        context: ScaleContext,
    ) -> Any:
        """Resolve the aligned internal-to-business multiplier for *obj*."""
        if kind in ("elasticity", "derived"):
            return 1.0
        if kind == "coefficient":
            return self._coefficient_factor(context.term, context)

        column = context.term
        if column is None:
            column = getattr(obj, "name", None)
        if not isinstance(column, str):
            raise ValueError(
                f"{kind} conversion needs a named Series/DataArray or "
                "dims={'term': 'column_name', ...}."
            )
        if column not in self.factors:
            return 1.0

        import xarray as xr

        if isinstance(obj, xr.DataArray):
            factor_dims, table = self.factors[column]
            if factor_dims and all(dim in obj.dims for dim in factor_dims):
                coords = []
                for dim in factor_dims:
                    if dim not in obj.coords:
                        break
                    coords.append([str(value) for value in obj.coords[dim].values])
                else:
                    shape = tuple(len(values) for values in coords)
                    values = np.empty(shape, dtype=float)
                    for index in np.ndindex(*shape):
                        key = tuple(
                            coords[axis][index[axis]] for axis in range(len(shape))
                        )
                        try:
                            values[index] = table[key]
                        except KeyError as exc:
                            raise KeyError(
                                f"Scaling factors for column {column!r} have no "
                                f"entry for xarray coordinate {key}."
                            ) from exc
                    return xr.DataArray(
                        values,
                        dims=factor_dims,
                        coords={dim: obj.coords[dim] for dim in factor_dims},
                    )

        if context.data is not None:
            per_row = self._per_row(context.data, column)
            if context.scan_info is not None:
                scan = context.scan_info
                matrix = per_row[scan.sort_idx].reshape(scan.n_units, scan.n_times).T
                if isinstance(obj, xr.DataArray):
                    obs_dims = [
                        str(dim) for dim in obj.dims if dim not in ("chain", "draw")
                    ]
                    actual = tuple(int(obj.sizes[dim]) for dim in obs_dims)
                    if len(obs_dims) != 2 or actual != matrix.shape:
                        raise ValueError(
                            f"Cannot align scan scaling factors for {column!r}: "
                            f"expected observation shape {matrix.shape} in "
                            f"(time, unit) order, got {actual}. Pass values with "
                            "explicit time and unit dimensions in that order."
                        )
                    time_dim, unit_dim = obs_dims
                    return xr.DataArray(
                        matrix,
                        dims=[time_dim, unit_dim],
                        coords={
                            time_dim: obj.coords[time_dim],
                            unit_dim: obj.coords[unit_dim],
                        },
                    )
                arr = np.asarray(obj)
                if arr.shape == matrix.shape:
                    return matrix
                if arr.ndim > 2 and arr.shape[-2:] == matrix.shape:
                    return matrix.reshape((1,) * (arr.ndim - 2) + matrix.shape)
                raise ValueError(
                    f"Cannot align scan scaling factors for {column!r}: expected "
                    f"shape {matrix.shape}, or leading sample dimensions followed "
                    f"by {matrix.shape}; got {arr.shape}. Do not reshape factors "
                    "based only on an equal element count."
                )
            if isinstance(obj, xr.DataArray):
                obs_dims = [
                    str(dim) for dim in obj.dims if dim not in ("chain", "draw")
                ]
                matching = [dim for dim in obs_dims if obj.sizes[dim] == len(per_row)]
                if len(matching) == 1:
                    dim = matching[0]
                    coordinate_values = (
                        obj.coords[dim]
                        if dim in obj.coords
                        else np.arange(len(per_row))
                    )
                    return xr.DataArray(
                        per_row,
                        dims=[dim],
                        coords={dim: coordinate_values},
                    )
                if not obs_dims:
                    return self._scalar_factor(column, context)
                raise ValueError(
                    f"Cannot align row scaling factors for {column!r}: expected "
                    f"exactly one observation dimension of length {len(per_row)}, "
                    f"got dimensions {tuple(obj.dims)} with shape {obj.shape}."
                )
            arr = np.asarray(obj)
            if arr.shape == per_row.shape:
                return per_row
            if arr.ndim > 1 and arr.shape[-1] == len(per_row):
                return per_row.reshape((1,) * (arr.ndim - 1) + (len(per_row),))
            if arr.ndim == 0:
                return self._scalar_factor(column, context)
            raise ValueError(
                f"Cannot align row scaling factors for {column!r}: expected "
                f"shape ({len(per_row)},), or leading sample dimensions ending "
                f"in {len(per_row)}; got {arr.shape}. Do not reshape factors "
                "based only on an equal element count."
            )

        return self._scalar_factor(column, context)

    def _convert_value(
        self,
        obj: Any,
        *,
        kind: ScaleKind,
        context: ScaleContext,
        direction: ScaleDirection,
    ) -> Any:
        """Convert one non-frame value and tag the result."""
        factor = self._factor_for(obj, kind=kind, context=context)
        key = _state_key(context.term or getattr(obj, "name", None), factor)
        if _conversion_state(obj).get(key) == direction:
            return obj
        result = obj / factor if direction == "internal" else obj * factor
        original_name = getattr(obj, "name", None)
        if original_name is not None and getattr(result, "name", None) is None:
            result = result.rename(original_name)
        return _tag_conversion(result, key=key, direction=direction)

    def _convert_frame(
        self,
        obj: Any,
        *,
        kind: ScaleKind,
        context: ScaleContext,
        direction: ScaleDirection,
    ) -> Any:
        """Convert selected numeric columns of a pandas or Narwhals frame."""
        columns = context.columns
        if columns is None:
            if self.roles:
                columns = tuple(
                    column
                    for column, roles in self.roles.items()
                    if kind in roles and column in obj.columns
                )
            else:
                present = tuple(sorted(set(self.factors) & set(obj.columns)))
                if present:
                    raise ValueError(
                        "Frame conversion with manually constructed "
                        "ScalingFactors requires semantic roles. Pass roles= "
                        "when constructing ScalingFactors, or select columns "
                        "explicitly with dims=ScaleContext(columns=(...))."
                    )
                columns = ()

        if isinstance(obj, nw.DataFrame):
            state = _conversion_state(obj)
            new_columns: list[nw.Series] = []
            for column in columns:
                if column not in self.factors or column not in obj.columns:
                    continue
                values = np.asarray(obj[column].to_numpy(), dtype=float)
                factor = self._per_row(obj, column)
                key = _state_key(column, factor)
                if state.get(key) == direction:
                    continue
                converted = (
                    values / factor if direction == "internal" else values * factor
                )
                new_columns.append(
                    nw.new_series(column, converted, backend=obj.implementation)
                )
                state[key] = direction
            if not new_columns:
                return obj
            nw_result = obj.with_columns(new_columns)
            for key, tagged_direction in state.items():
                nw_result = _tag_conversion(
                    nw_result,
                    key=key,
                    direction=tagged_direction,
                )
            return nw_result

        result = obj.copy()
        state = _conversion_state(result)
        for column in columns:
            if column not in self.factors or column not in result.columns:
                continue
            frame = nw.from_native(result, eager_only=True)
            factor = self._per_row(frame, column)
            key = _state_key(column, factor)
            if state.get(key) == direction:
                continue
            values = np.asarray(result[column], dtype=float)
            result[column] = (
                values / factor if direction == "internal" else values * factor
            )
            state[key] = direction
        attrs = getattr(result, "attrs", None)
        if isinstance(attrs, dict):
            attrs[_SCALE_TAG] = state
        return result

    def _convert(
        self,
        obj: Any,
        *,
        kind: ScaleKind,
        dims: ScaleDims,
        direction: ScaleDirection,
    ) -> Any:
        """Shared implementation for both public conversion directions."""
        if kind not in ("outcome", "regressor", "coefficient", "elasticity", "derived"):
            raise ValueError(
                f"Unknown scaling kind {kind!r}. Expected outcome, regressor, "
                "coefficient, elasticity, or derived."
            )
        context = _coerce_context(dims)

        import pandas as pd
        import xarray as xr

        if isinstance(obj, (pd.DataFrame, nw.DataFrame)):
            return self._convert_frame(
                obj, kind=kind, context=context, direction=direction
            )
        if isinstance(obj, xr.Dataset):
            result = obj.copy()
            for name in result.data_vars:
                term_context = replace(context, term=str(name))
                result[name] = self._convert_value(
                    result[name],
                    kind=kind,
                    context=term_context,
                    direction=direction,
                )
            return result
        if isinstance(obj, pd.Series) and context.term is None:
            context = replace(context, term=str(obj.name))
        return self._convert_value(obj, kind=kind, context=context, direction=direction)

    def to_internal(self, obj: Any, *, kind: ScaleKind, dims: ScaleDims = None) -> Any:
        """Convert business-unit values to the model's internal scale.

        ``kind`` describes the value's semantic role rather than its Python
        container. ``dims`` may be a data frame for row alignment or a mapping
        with ``term``, ``outcome``, ``data``, ``scan_info``, ``scalar``, and
        ``columns`` entries. Converted pandas/xarray objects, NumPy arrays, and
        scalars carry a private unit tag, making repeated conversion a no-op.
        """
        return self._convert(obj, kind=kind, dims=dims, direction="internal")

    def to_business(self, obj: Any, *, kind: ScaleKind, dims: ScaleDims = None) -> Any:
        """Convert internal-scale values to user-facing business units.

        See :meth:`to_internal` for the semantic kinds and alignment context.
        """
        return self._convert(obj, kind=kind, dims=dims, direction="business")

    def transform(self, df: nw.DataFrame) -> nw.DataFrame:
        """Divide the fitted columns of *df* by their scale factors.

        Columns with fitted factors but no matching *df* column (e.g.
        simulated outcomes) are skipped.
        """
        return self.to_internal(
            df,
            kind="regressor",
            dims=ScaleContext(columns=tuple(self.factors)),
        )

    def inverse_transform_column(
        self, values: np.ndarray, column: str, df: nw.DataFrame
    ) -> np.ndarray:
        """Multiply *values* by the fitted factor of *column* (inverse transform)."""
        return np.asarray(
            self.to_business(
                np.asarray(values, dtype=float),
                kind="outcome",
                dims=ScaleContext(term=column, data=df),
            )
        )

    def mean_factor(self, column: str, df: nw.DataFrame) -> float:
        """Data-weighted mean divisor for *column* (1.0 when not scaled).

        This is a row-weighted average of the per-unit divisors, used
        when a single number is required (a labeled coefficient, a
        scalar fill). It is an approximation: a ratio of means is not
        the mean of ratios, so when outcome and predictor factors are
        not proportional across units the resulting coefficient is not
        any one unit's coefficient.
        """
        return self._scalar_factor(
            column, ScaleContext(term=column, data=df, scalar="mean")
        )

    def coefficient_to_business(
        self,
        predictor: str | None,
        outcome: str,
        df: nw.DataFrame,
    ) -> float:
        """Scale a *linear* coefficient from internal to business units.

        Applies ``mean_factor(outcome) / mean_factor(predictor)``. That
        ratio is only correct when the coefficient multiplies the raw
        column (or a homogeneous transform of it, such as adstock).
        Saturating transforms, interactions, and HSGP terms are handled by
        :meth:`to_business` with ``kind="coefficient"`` and a parsed term.
        """
        return float(
            self.to_business(
                1.0,
                kind="coefficient",
                dims=ScaleContext(term=predictor, outcome=outcome, data=df),
            )
        )

    def unscale_xarray(self, column: str, da: Any, df: nw.DataFrame) -> Any:
        """Map internal-scale *da* to business units along observation dims."""
        return self.to_business(
            da,
            kind="outcome",
            dims=ScaleContext(term=column, data=df, scalar="mean"),
        )


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
        if isinstance(key, str) and _SEPARATOR in key and len(dims) > 1:
            normed = tuple(key.split(_SEPARATOR))
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


def _require_positive_finite(
    table: dict[tuple[str, ...], float],
    *,
    role: str,
    method: str,
    where: str,
) -> None:
    """Raise unless every divisor in *table* is positive and finite."""
    if not table:
        return
    arr = np.array(list(table.values()))
    bad = ~(np.isfinite(arr) & (arr > 0))
    if bad.any():
        raise ValueError(
            f"Scaling.{role} method {method!r} produced non-positive or "
            f"non-finite divisor(s) {np.unique(arr[bad])[:5].tolist()} for "
            f"{where}. Every scale factor must be positive and finite."
        )


def _group_stat_table(
    values: np.ndarray, keys: list[tuple[str, ...]], method: str
) -> dict[tuple[str, ...], float]:
    """Group max/mean statistics keyed by unit."""
    stats: dict[tuple[str, ...], float] = {}
    buckets: dict[tuple[str, ...], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)
    for key, rows in buckets.items():
        subset = values[rows]
        stats[key] = float(np.max(subset) if method == "max" else np.mean(subset))
    return stats


def _ensure_grid_covers_keys(
    table: dict[tuple[str, ...], float],
    row_keys: list[tuple[str, ...]],
) -> None:
    """Raise if any observed unit key is absent from a scaling grid."""
    missing = sorted({key for key in row_keys if key not in table})
    if missing:
        shown = ", ".join(repr(m) for m in missing[:5])
        more = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise KeyError(
            f"Scaling grid is missing entries for unit(s) {shown}{more}. "
            f"Grid covers {len(table)} unit(s); every observed combination "
            "of the configured dims must have an entry."
        )


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
) -> dict[str, tuple[tuple[str, ...], dict[tuple[str, ...], float]]]:
    """Fit unit-keyed divisors for one role's columns."""
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

    keys = _row_keys(df, dims)

    if method in ("max", "mean"):
        factors: dict[str, tuple[tuple[str, ...], dict[tuple[str, ...], float]]] = {}
        for col in columns:
            table = _group_stat_table(df[col].to_numpy().astype(float), keys, method)
            _require_positive_finite(
                table, role=role, method=method, where=f"column {col!r}"
            )
            factors[col] = (dims, table)
        return factors

    if method == "fixed":
        value = cfg["value"]
        if np.isscalar(value) and not hasattr(value, "dims"):
            factor = float(value)  # type: ignore[arg-type]
            table = {(): factor}
            _require_positive_finite(
                table, role=role, method=method, where="fixed value"
            )
            return {col: ((), table) for col in columns}
        table = _as_grid(value, dims)
        _ensure_grid_covers_keys(table, keys)
        _require_positive_finite(table, role=role, method=method, where="fixed grid")
        return {col: (dims, table) for col in columns}

    # method == "divide"
    if not dims:
        raise ValueError(
            'Scaling method "divide" requires "dims" naming the panel unit '
            "columns the external grid is keyed by."
        )
    table = _as_grid(cfg["by"], dims)
    _ensure_grid_covers_keys(table, keys)
    _require_positive_finite(table, role=role, method=method, where="divide grid")
    return {col: (dims, table) for col in columns}


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
        existing_roles = {column: set(value) for column, value in scaling.roles.items()}
        for column in target_columns or set():
            if column in scaling.factors:
                existing_roles.setdefault(column, set()).add("outcome")
        for column in channel_columns or set():
            if column in scaling.factors:
                existing_roles.setdefault(column, set()).add("regressor")
        resolved_roles = {
            column: frozenset(value) for column, value in existing_roles.items()
        }
        if resolved_roles == scaling.roles:
            return scaling
        return replace(scaling, roles=resolved_roles)
    if not isinstance(scaling, Scaling):
        raise TypeError(
            "scaling= accepts pathmc.Scaling or a fitted "
            f"pathmc.ScalingFactors, got {type(scaling).__name__}."
        )

    factors: dict[str, tuple[tuple[str, ...], dict[tuple[str, ...], float]]] = {}
    fitted_roles: dict[str, set[Literal["outcome", "regressor"]]] = {}
    for role, candidates in (
        ("target", target_columns or set()),
        ("channel", channel_columns or set()),
    ):
        cfg = _validate_spec(role, getattr(scaling, role))
        if cfg:
            fitted = _fit_role(
                role,
                cfg,
                df,
                panel_info,
                candidates,
                has_data=role in roles_with_data,
            )
            factors.update(fitted)
            semantic_role: Literal["outcome", "regressor"] = (
                "outcome" if role == "target" else "regressor"
            )
            for column in fitted:
                fitted_roles.setdefault(column, set()).add(semantic_role)
    return ScalingFactors(
        factors=factors,
        roles={column: frozenset(value) for column, value in fitted_roles.items()},
    )


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
