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
"""Shared causal query layer: predictions, comparisons, and slopes."""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.axes
import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from narwhals.stable.v1.typing import IntoFrame

from pathmc.idata import DEFAULT_HDI_PROB
from pathmc.idata import hdi as compute_hdi
from pathmc.reprs import ReprSpec, ResultReprMixin
from pathmc.simulate import (
    EstimandResult,
    _DrawStorageMixin,
    _stack_sample,
    run_do_pymc,
)

if TYPE_CHECKING:
    from pathmc._model import PathModel

__all__ = ["InterpretResult", "datagrid"]

_COMPARISON_ESTIMANDS = {"diff": "ATE", "ratio": "ratio", "lift": "lift"}


def datagrid(data: IntoFrame, **cols: list[float] | list[int]) -> pd.DataFrame:
    """Build a covariate grid as the cartesian product of supplied columns.

    Columns not listed in ``cols`` are held constant: numeric columns use
    the column mean; non-numeric columns use the mode.

    Parameters
    ----------
    data : IntoFrame
        Reference data frame defining the available columns.
    **cols
        Column names mapped to lists of values to cross.

    Returns
    -------
    pd.DataFrame
        One row per grid point.
    """
    nw_df = nw.from_native(data, eager_only=True)
    native = nw_df.to_pandas()

    keys = list(cols.keys())
    rows: list[dict[str, Any]] = []
    value_lists = [list(cols[k]) for k in keys]
    for combo in itertools.product(*value_lists):
        row = dict(zip(keys, combo, strict=True))
        for col in native.columns:
            if col in row:
                continue
            series = native[col]
            if pd.api.types.is_numeric_dtype(series):
                row[col] = float(series.mean())
            else:
                row[col] = series.mode().iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


class InterpretResult(_DrawStorageMixin, ResultReprMixin):
    """Unit-preserving posterior draws for predictions, contrasts, or slopes.

    Internal storage is an :class:`xarray.Dataset` with dims
    ``("chain", "draw", "unit")`` exposed as :attr:`dataset`.

    Parameters
    ----------
    ds : xr.Dataset
        Labelled draws for the outcome variable.
    outcome : str
        Outcome variable name.
    quantity : str
        ``"prediction"``, ``"comparison"``, or ``"slope"``.
    variable : str or None
        Varied variable name, if any.
    estimator : str
        Estimator backend label.
    causal : bool
        Whether the result supports a causal interpretation.
    interventional : bool
        Whether graph surgery / ``do()`` was used.
    identifiable : bool or None
        Backdoor identifiability when a treatment/outcome pair is defined.
    """

    def __init__(
        self,
        *,
        ds: xr.Dataset,
        outcome: str,
        quantity: str,
        variable: str | None = None,
        estimator: str = "structural",
        causal: bool = False,
        interventional: bool = False,
        identifiable: bool | None = None,
    ) -> None:
        self._ds = ds
        self._default_var = outcome
        self._quantity = quantity
        self._variable = variable
        self._estimator = estimator
        self._causal = causal
        self._interventional = interventional
        self._identifiable = identifiable
        self._ds.attrs.update({
            "estimator": estimator,
            "interventional": interventional,
            "identifiable": identifiable,
            "causal": causal,
            "quantity": quantity,
            "variable": variable,
        })

    @property
    def outcome(self) -> str:
        """The outcome variable this result targets."""
        return self._default_var

    @property
    def quantity(self) -> str:
        """Query kind: ``prediction``, ``comparison``, or ``slope``."""
        return self._quantity

    @property
    def variable(self) -> str | None:
        """Varied variable name, if any."""
        return self._variable

    @property
    def estimator(self) -> str:
        """Estimator backend that produced this result."""
        return self._estimator

    @property
    def causal(self) -> bool:
        """Whether the result supports a causal interpretation."""
        return self._causal

    @property
    def interventional(self) -> bool:
        """Whether graph surgery / ``do()`` was used."""
        return self._interventional

    @property
    def identifiable(self) -> bool | None:
        """Backdoor identifiability of the treatment-outcome pair, if defined."""
        return self._identifiable

    def draws(self, var: str | None = None) -> np.ndarray:
        """Return flat posterior draws, defaulting to the outcome variable."""
        key = self._default_var if var is None else var
        return _stack_sample(self._ds[key])

    def mean(self, var: str | None = None) -> float:
        """Return the posterior mean, defaulting to the outcome variable."""
        return float(np.mean(self.draws(var)))

    def hdi(self, var: str | None = None, prob: float = DEFAULT_HDI_PROB) -> np.ndarray:
        """Return the highest-density interval, defaulting to the outcome."""
        return compute_hdi(self.draws(var), prob=prob)

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        *,
        var: str | None = None,
        bins: int | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot the marginal posterior of the outcome (mean over ``unit`` first).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. Creates a new figure when ``None``.
        var : str or None
            Variable to plot. Defaults to the outcome.
        bins : int or None
            Histogram bin count.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the histogram.
        """
        import matplotlib.pyplot as plt

        key = self._default_var if var is None else var
        da = self._ds[key]
        if "unit" in da.dims:
            da = da.mean("unit")
        draws = _stack_sample(da)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        else:
            from typing import cast

            fig = cast("matplotlib.figure.Figure", ax.get_figure())

        ax.hist(draws, bins=bins, density=True, alpha=0.7, edgecolor="k")
        ax.set_xlabel(key)
        ax.set_ylabel("density")
        ax.set_title(f"{self._quantity}: {key}")
        return fig

    def _repr_compact(self) -> str:
        mean = self.mean()
        lo, hi = self.hdi()
        return (
            f"InterpretResult({self._quantity}, {self._default_var})  "
            f"mean={mean:.2f}  hdi=[{lo:.2f}, {hi:.2f}]"
        )

    def _repr_spec(self) -> ReprSpec:
        mean = self.mean()
        lo, hi = self.hdi()
        return ReprSpec(
            title=f"{self._quantity.title()} — {self._default_var}",
            rows=[
                ["Mean", f"{mean:.2f}"],
                ["HDI", f"[{lo:.2f}, {hi:.2f}]"],
                ["Quantity", self._quantity],
                ["Interventional", str(self._interventional)],
            ],
            footer="Methods: .mean() .hdi() .draws() .plot() .dataset",
        )


def _panel_not_implemented(method: str) -> None:
    raise NotImplementedError(
        f"{method}() is not yet supported for panel models. "
        "Use a cross-sectional model or do(simulate_over='time') instead."
    )


def _dag_nodes(model: PathModel) -> set[str]:
    return (
        set(model._graph_info.endogenous)
        | set(model._graph_info.exogenous)
        | set(model._graph_info.latent)
    )


def _identifiable_flag(
    model: PathModel, treatment: str | None, outcome: str
) -> bool | None:
    if treatment is None:
        return None
    nodes = _dag_nodes(model)
    if treatment not in nodes or outcome not in nodes:
        return None
    return model.is_identifiable(treatment, outcome)


def _validate_conditional(conditional: dict[str, Any] | None) -> dict[str, float]:
    if not conditional:
        return {}
    validated: dict[str, float] = {}
    for key, val in conditional.items():
        if isinstance(val, (list, tuple, np.ndarray)):
            raise TypeError(
                f"conditional values must be scalars; got {type(val).__name__} "
                f"for '{key}'. Use predictions(newdata=datagrid(...)) for grids."
            )
        validated[key] = float(val)
    return validated


@contextmanager
def _temporary_query_data(model: PathModel, data: nw.DataFrame):
    """Temporarily point exogenous ``pm.Data`` nodes at *data*."""
    gen_model = model._gen_model
    if gen_model is None:
        yield
        return

    updates: dict[str, np.ndarray] = {}
    for col in data.columns:
        if col not in gen_model.named_vars:
            continue
        arr = np.asarray(data[col].to_numpy())
        if arr.ndim != 1:
            continue
        updates[col] = arr.astype(float, copy=False)

    if not updates:
        yield
        return

    assert model._data is not None
    previous = {
        name: np.asarray(model._data[name].to_numpy(), dtype=float) for name in updates
    }
    pm.set_data(updates, model=gen_model)
    try:
        yield
    finally:
        pm.set_data(previous, model=gen_model)


def _unit_prediction(
    model: PathModel,
    data: nw.DataFrame,
    outcome: str,
    set_dict: dict[str, float | np.ndarray],
    *,
    swap_data: bool = False,
) -> xr.DataArray:
    """Unit-level response-mean draws for one outcome under ``set_dict``."""
    assert model._gen_model is not None
    assert model._idata is not None

    def _run() -> xr.DataArray:
        result = run_do_pymc(
            gen_model=model._gen_model,
            graph_info=model._graph_info,
            idata=model._idata,
            data=data,
            set=set_dict,
            kind="mean",
            families=model._families,
            average_units=False,
        )
        if outcome not in result.dataset:
            raise KeyError(
                f"Outcome '{outcome}' not found in do() output. "
                f"Available: {sorted(result.dataset.data_vars)}"
            )
        return result.dataset[outcome]

    if swap_data:
        with _temporary_query_data(model, data):
            return _run()
    return _run()


def _wrap_interpret(
    da: xr.DataArray,
    outcome: str,
    quantity: str,
    variable: str | None,
    *,
    interventional: bool,
    identifiable: bool | None,
) -> InterpretResult:
    causal = interventional
    return InterpretResult(
        ds=xr.Dataset({outcome: da}),
        outcome=outcome,
        quantity=quantity,
        variable=variable,
        interventional=interventional,
        causal=causal,
        identifiable=identifiable,
    )


def _wrap_estimand(
    da: xr.DataArray,
    outcome: str,
    treatment: str,
    estimand: str,
    *,
    identifiable: bool | None,
) -> EstimandResult:
    return EstimandResult(
        ds=xr.Dataset({outcome: da}),
        outcome=outcome,
        treatment=treatment,
        estimand=estimand,
        interventional=True,
        causal=True,
        identifiable=identifiable,
    )


def _apply_comparison(
    hi: xr.DataArray, lo: xr.DataArray, comparison: Literal["diff", "ratio", "lift"]
) -> xr.DataArray:
    if comparison == "diff":
        return hi - lo
    if comparison == "ratio":
        return hi / lo
    if comparison == "lift":
        return (hi - lo) / lo
    raise ValueError(
        f"Unknown comparison {comparison!r}. Expected 'diff', 'ratio', or 'lift'."
    )


def _apply_slope(
    hi: xr.DataArray,
    lo: xr.DataArray,
    x: np.ndarray,
    eps: float,
    slope: Literal["dydx", "eyex", "eydx", "dyex"],
) -> xr.DataArray:
    dydx = (hi - lo) / eps
    x_da = xr.DataArray(x, dims=["unit"], coords={"unit": np.arange(len(x))})
    if slope == "dydx":
        return dydx
    if slope == "eyex":
        return dydx * (x_da / lo)
    if slope == "eydx":
        return dydx / lo
    if slope == "dyex":
        return dydx * x_da
    raise ValueError(
        f"Unknown slope {slope!r}. Expected 'dydx', 'eyex', 'eydx', or 'dyex'."
    )


def _to_frame(model: PathModel, newdata: IntoFrame | None) -> tuple[nw.DataFrame, bool]:
    if newdata is None:
        assert model._data is not None
        return model._data, False
    return nw.from_native(newdata, eager_only=True), True


def predictions(
    model: PathModel,
    outcome: str,
    *,
    set: dict[str, float | np.ndarray] | None = None,
    newdata: IntoFrame | None = None,
) -> InterpretResult:
    """Response-mean predictions on the fitted frame or a covariate grid."""
    model._require_fitted("predictions")
    if model._panel_info is not None:
        _panel_not_implemented("predictions")

    set_dict = {} if set is None else dict(set)
    interventional = bool(set_dict)
    data, swap_data = _to_frame(model, newdata)
    variable = next(iter(set_dict)) if len(set_dict) == 1 else None
    identifiable = _identifiable_flag(model, variable, outcome)

    da = _unit_prediction(model, data, outcome, set_dict, swap_data=swap_data)
    return _wrap_interpret(
        da,
        outcome,
        "prediction",
        variable,
        interventional=interventional,
        identifiable=identifiable,
    )


def comparisons(
    model: PathModel,
    outcome: str,
    variable: str,
    *,
    contrast: tuple[float, float] = (0.0, 1.0),
    comparison: Literal["diff", "ratio", "lift"] = "diff",
    conditional: dict[str, float] | None = None,
    average_by: Literal["all"] | None = "all",
) -> EstimandResult | InterpretResult:
    """Draw-wise interventional contrasts between two intervention values."""
    model._require_fitted("comparisons")
    if model._panel_info is not None:
        _panel_not_implemented("comparisons")

    cond = _validate_conditional(conditional)
    lo, hi = contrast
    set_lo: dict[str, float | np.ndarray] = {variable: lo, **cond}
    set_hi: dict[str, float | np.ndarray] = {variable: hi, **cond}
    data, _swap = _to_frame(model, None)
    identifiable = _identifiable_flag(model, variable, outcome)

    lo_da = _unit_prediction(model, data, outcome, set_lo)
    hi_da = _unit_prediction(model, data, outcome, set_hi)
    contrast_da = _apply_comparison(hi_da, lo_da, comparison)

    if average_by == "all":
        return _wrap_estimand(
            contrast_da.mean("unit"),
            outcome,
            variable,
            _COMPARISON_ESTIMANDS[comparison],
            identifiable=identifiable,
        )

    return _wrap_interpret(
        contrast_da,
        outcome,
        "comparison",
        variable,
        interventional=True,
        identifiable=identifiable,
    )


def slopes(
    model: PathModel,
    outcome: str,
    wrt: str,
    *,
    slope: Literal["dydx", "eyex", "eydx", "dyex"] = "dydx",
    eps: float = 1e-4,
    conditional: dict[str, float] | None = None,
    average_by: Literal["all"] | None = "all",
) -> EstimandResult | InterpretResult:
    """Finite-difference interventional slopes using per-row covariate values."""
    model._require_fitted("slopes")
    if model._panel_info is not None:
        _panel_not_implemented("slopes")

    cond = _validate_conditional(conditional)
    data, _swap = _to_frame(model, None)
    x = np.asarray(data[wrt].to_numpy(), dtype=float)
    set_lo: dict[str, float | np.ndarray] = {wrt: x, **cond}
    set_hi: dict[str, float | np.ndarray] = {wrt: x + eps, **cond}
    identifiable = _identifiable_flag(model, wrt, outcome)

    lo_da = _unit_prediction(model, data, outcome, set_lo)
    hi_da = _unit_prediction(model, data, outcome, set_hi)
    slope_da = _apply_slope(hi_da, lo_da, x, eps, slope)

    if average_by == "all":
        return _wrap_estimand(
            slope_da.mean("unit"),
            outcome,
            wrt,
            slope,
            identifiable=identifiable,
        )

    return _wrap_interpret(
        slope_da,
        outcome,
        "slope",
        wrt,
        interventional=True,
        identifiable=identifiable,
    )
