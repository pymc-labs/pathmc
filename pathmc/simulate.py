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
"""do() operator: interventional simulation via posterior propagation.

Cross-sectional do() uses PyMC-native graph surgery: ``pm.do()`` on
the generative model + ``pm.sample_posterior_predictive()`` for
kind="predictive", or ``pm.do()`` + ``compute_deterministics`` for
kind="mean".

Panel do() for models with temporal dependencies (adstock, lags) uses
a scan-compiled generative model — the same ``pm.do()`` mechanism
handles temporal propagation natively.
"""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors

from pathmc.graph import GraphInfo
from pathmc.idata import DEFAULT_HDI_PROB
from pathmc.idata import hdi as compute_hdi
from pathmc.idata import hdi_label
from pathmc.idata import posterior
from pathmc.panel import PanelInfo
from pathmc.reprs import ReprSpec, ResultReprMixin

__all__ = ["DoResult", "EstimandResult"]


# ---------------------------------------------------------------------------
# xarray storage helpers
# ---------------------------------------------------------------------------


def _unit_dims(da: xr.DataArray) -> list[str]:
    """Return unit-like dim names on *da* (shared ``unit`` or per-var ``unit_*``)."""
    return [d for d in da.dims if d == "unit" or d.startswith("unit_")]


def _stack_sample(da: xr.DataArray) -> np.ndarray:
    """Flatten ``("chain", "draw")`` (and unit) into a 1-D ``sample`` axis.

    Returns a ``(n_samples,)`` numpy array, preserving the historical public
    contract of ``draws()``. A ``unit`` dim (present for ``kind="predictive"``
    results) is flattened into the sample axis — matching the prior
    ``raw.flatten()`` behaviour. A ``time`` dim is averaged over first (the
    cross-sectional view of a panel var).
    """
    dims = list(da.dims)
    if "time" in dims:
        da = da.mean(dim="time")
    stack_dims = [d for d in ("chain", "draw") if d in da.dims] + _unit_dims(da)
    if not stack_dims:
        return np.asarray(da.values).ravel()
    stacked = da.stack(sample=stack_dims)
    return stacked.transpose("sample").values


def _stack_sample_time(da: xr.DataArray) -> np.ndarray:
    """Flatten ``("chain", "draw")`` within each time step.

    Returns a ``(n_times, n_samples)`` numpy array for a panel per-time var.
    Panel producers average over units before storage, so no ``unit`` dim is
    expected here.
    """
    extra = [
        d
        for d in da.dims
        if d not in ("chain", "draw", "time")
        and d != "unit"
        and not d.startswith("unit_")
    ]
    if extra:
        da = da.mean(dim=extra)
    stack_dims = [d for d in ("chain", "draw") if d in da.dims]
    if not stack_dims:
        return np.asarray(da.values)
    stacked = da.stack(sample=stack_dims)
    return stacked.transpose("time", "sample").values


def _build_dataset(
    values: dict[str, np.ndarray],
    values_by_time: dict[str, np.ndarray] | None,
    time_index: np.ndarray | None,
    *,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_units: int | None = None,
    n_units_per_var: dict[str, int] | None = None,
) -> xr.Dataset:
    """Build the internal ``("chain", "draw"[, "unit"][, "time"])`` Dataset.

    ``values`` arrays are flat ``(n_samples,)`` where ``n_samples`` is
    ``chain * draw`` for mean-path results or ``chain * draw * unit`` for
    predictive; they are reshaped back to named dims when ``n_chains``/
    ``n_draws`` are supplied and the length matches. Arrays of unrecognised
    length fall back to a per-variable ``"sample_<var>"`` dim so legacy
    callers (e.g. test fixtures) keep working without colliding.
    """
    data_vars: dict[str, xr.DataArray] = {}

    def _chain_draw_coords(nc: int, nd: int) -> dict[str, np.ndarray]:
        return {"chain": np.arange(nc), "draw": np.arange(nd)}

    if values:
        first = next(iter(values.values()))
        n_samples = first.shape[0]
        if n_chains is None:
            n_chains = 1
        if n_draws is None:
            n_draws = n_samples // n_chains

        # Vars that also have a per-time entry are stored from values_by_time
        # (with a time dim); the time-averaged view is derived by _stack_sample.
        time_var_names = set(values_by_time or {})
        predictive_lengths: list[int] = []
        for var, arr in values.items():
            if var in time_var_names:
                continue
            var_n_units = (n_units_per_var or {}).get(var, n_units)
            if (
                var_n_units is not None
                and arr.shape[0] == n_chains * n_draws * var_n_units
            ):
                predictive_lengths.append(var_n_units)
        heterogeneous_units = len(set(predictive_lengths)) > 1

        for var, arr in values.items():
            if var in time_var_names:
                continue  # stored from values_by_time below
            arr = np.asarray(arr)
            length = arr.shape[0]
            coords = _chain_draw_coords(n_chains, n_draws)
            var_n_units = (n_units_per_var or {}).get(var, n_units)
            if length == n_chains * n_draws:
                data_vars[var] = xr.DataArray(
                    arr.reshape(n_chains, n_draws),
                    dims=("chain", "draw"),
                    coords=coords,
                )
            elif var_n_units is not None and length == n_chains * n_draws * var_n_units:
                unit_dim = f"unit_{var}" if heterogeneous_units else "unit"
                data_vars[var] = xr.DataArray(
                    arr.reshape(n_chains, n_draws, var_n_units),
                    dims=("chain", "draw", unit_dim),
                    coords={**coords, unit_dim: np.arange(var_n_units)},
                )
            else:
                # Legacy/unknown shape: per-var sample dim avoids collisions.
                sample_dim = f"sample_{var}"
                data_vars[var] = xr.DataArray(arr, dims=(sample_dim,))

    if values_by_time:
        n_times = next(iter(values_by_time.values())).shape[0]
        time_coord = (
            np.asarray(time_index) if time_index is not None else np.arange(n_times)
        )
        for var, arr in values_by_time.items():
            arr = np.asarray(arr)  # (n_times, n_samples)
            if n_chains is None:
                n_chains = 1
            if n_draws is None:
                n_draws = arr.shape[1] // n_chains
            coords = {
                "time": time_coord,
                **_chain_draw_coords(n_chains, n_draws),
            }
            data_vars[var] = xr.DataArray(
                arr.reshape(n_times, n_chains, n_draws),
                dims=("time", "chain", "draw"),
                coords=coords,
            )

    return xr.Dataset(data_vars)


def _infer_build_kwargs(ds: xr.Dataset) -> dict[str, Any]:
    """Recover ``_build_dataset`` kwargs from an internal result Dataset."""
    kwargs: dict[str, Any] = {}
    if "chain" in ds.dims:
        kwargs["n_chains"] = int(ds.sizes["chain"])
    if "draw" in ds.dims:
        kwargs["n_draws"] = int(ds.sizes["draw"])
    n_units_per_var: dict[str, int] = {}
    for var in ds.data_vars:
        da = ds[var]
        unit_dims = _unit_dims(da)
        if unit_dims:
            n_units_per_var[str(var)] = int(da.sizes[unit_dims[0]])
    if n_units_per_var:
        kwargs["n_units_per_var"] = n_units_per_var
    return kwargs


def _subtract_stores(lhs: xr.Dataset, rhs: xr.Dataset) -> xr.Dataset:
    """Subtract two result Datasets on their stacked sample axes.

    Positional numpy subtraction preserves the historical contrast contract:
    operands with the same flat draws but different internal dim layouts
    (e.g. legacy ``sample_<var>`` vs labelled ``chain``/``draw``) still pair
    element-wise. Raises ``ValueError`` on length mismatch rather than
    xarray's silent coordinate inner-join.
    """
    common = [str(v) for v in lhs.data_vars if v in rhs.data_vars]
    values: dict[str, np.ndarray] = {}
    values_by_time: dict[str, np.ndarray] | None = None
    time_index: np.ndarray | None = None

    for var in common:
        l_da = lhs[var]
        r_da = rhs[var]
        if "time" in l_da.dims:
            l_bt = _stack_sample_time(l_da)
            r_bt = _stack_sample_time(r_da)
            if l_bt.shape != r_bt.shape:
                raise ValueError(
                    f"Cannot subtract: variable '{var}' has incompatible per-time "
                    f"draw shapes {l_bt.shape} vs {r_bt.shape}."
                )
            diff_bt = l_bt - r_bt
            if values_by_time is None:
                values_by_time = {}
            values_by_time[var] = diff_bt
            values[var] = diff_bt.mean(axis=0)
            if time_index is None and "time" in l_da.coords:
                time_index = np.asarray(l_da["time"].values)
        else:
            l_flat = _stack_sample(l_da)
            r_flat = _stack_sample(r_da)
            if l_flat.shape != r_flat.shape:
                raise ValueError(
                    f"Cannot subtract: variable '{var}' has incompatible draw "
                    f"counts ({l_flat.shape[0]} vs {r_flat.shape[0]})."
                )
            values[var] = l_flat - r_flat

    return _build_dataset(
        values,
        values_by_time,
        time_index,
        **_infer_build_kwargs(lhs),
    )


class DoResult(ResultReprMixin):
    """Container for propagated posterior draws under an intervention.

    Supports ``.mean(var)``, ``.hdi(var)``, and contrast arithmetic
    via subtraction (``scenario - baseline``).

    For panel ``do(simulate_over="time")``, the result also stores
    per-time-step draws accessible via :meth:`by_time`.

    Internal storage is an :class:`xarray.Dataset` exposed as :attr:`dataset`
    with named dims ``("chain", "draw")`` for cross-sectional draws and an
    additional ``"time"`` dim for panel per-time draws. The ``_values`` and
    ``_values_by_time`` attributes are backward-compatible dict views
    (stacked over ``chain``/``draw`` into a ``"sample"`` axis) that
    preserve the historical numpy-array public contract.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Posterior draws for each variable, shape ``(n_samples,)`` where
        ``n_samples`` is ``chain * draw`` for mean-path results or
        ``chain * draw * unit`` for predictive. Used to build the internal
        Dataset when ``ds`` is not supplied.
    values_by_time : dict[str, np.ndarray] | None
        Per-time-step draws, shape ``(n_times, n_samples)``.
        Available only for panel do() results.
    time_index : array-like | None
        Time labels corresponding to the ``time`` dim.
    ds : xr.Dataset | None
        Pre-built internal Dataset with dims ``("chain", "draw")`` and,
        optionally, a ``"time"`` dim. When supplied, takes precedence over
        the dict arguments (which are used only by legacy callers).
    n_chains, n_draws : int | None
        Chain/draw sizes used to reconstruct the ``chain``/``draw`` dims
        when building the Dataset from flat ``values`` dicts. If omitted,
        inferred from the first variable's length (treating the array as
        a flattened ``chain * draw`` sample vector with ``chain=1``).
    n_units, n_units_per_var : int | dict[str, int] | None
        Unit counts for predictive ``values`` arrays whose length is
        ``chain * draw * unit``. ``n_units_per_var`` overrides ``n_units``
        per variable when predictive vars have heterogeneous unit counts.
    """

    def __init__(
        self,
        values: dict[str, np.ndarray] | None = None,
        values_by_time: dict[str, np.ndarray] | None = None,
        time_index: np.ndarray | None = None,
        *,
        ds: xr.Dataset | None = None,
        n_chains: int | None = None,
        n_draws: int | None = None,
        n_units: int | None = None,
        n_units_per_var: dict[str, int] | None = None,
    ) -> None:
        if ds is not None:
            self._ds = ds
        else:
            self._ds = _build_dataset(
                values or {},
                values_by_time,
                time_index,
                n_chains=n_chains,
                n_draws=n_draws,
                n_units=n_units,
                n_units_per_var=n_units_per_var,
            )

    @property
    def dataset(self) -> xr.Dataset:
        """Labeled posterior draws as an :class:`xarray.Dataset`.

        Variables are stored with dims ``("chain", "draw")`` for mean-path
        results, plus ``"unit"`` for ``kind="predictive"`` and ``"time"`` for
        panel per-time results. Mutating this object affects the result
        in place (and any :class:`EstimandResult` built via
        :meth:`EstimandResult.from_contrast` that shares it).

        For a flat ``(n_samples,)`` numpy view, use :meth:`draws` instead.
        """
        return self._ds

    def _draw(self, var: str) -> np.ndarray:
        """Flat ``(n_samples,)`` draws for one variable (O(1) stack)."""
        return _stack_sample(self._ds[var])

    def _draw_by_time(self, var: str) -> np.ndarray:
        """Per-time ``(n_times, n_samples)`` draws for one panel variable."""
        return _stack_sample_time(self._ds[var])

    # -- backward-compatible dict views (backed by the xarray store) --------

    @property
    def _values(self) -> dict[str, np.ndarray]:
        """Flat ``(n_samples,)`` draws per variable (chain+draw stacked).

        Variables with a ``time`` dim (panel per-time vars) are averaged over
        time first, matching the historical ``values[var] = by_time.mean(axis=0)``
        contract.
        """
        out: dict[str, np.ndarray] = {}
        for var in self._ds.data_vars:
            out[str(var)] = _stack_sample(self._ds[var])
        return out

    @property
    def _values_by_time(self) -> dict[str, np.ndarray] | None:
        """Per-time ``(n_times, n_samples)`` draws, or None if cross-sectional."""
        by_time_vars = [v for v in self._ds.data_vars if "time" in self._ds[v].dims]
        if not by_time_vars:
            return None
        return {str(var): _stack_sample_time(self._ds[var]) for var in by_time_vars}

    @property
    def _time_index(self) -> np.ndarray | None:
        """Time coord of the internal Dataset, or None."""
        if "time" in self._ds.coords:
            return np.asarray(self._ds["time"].values)
        return None

    def draws(self, var: str) -> np.ndarray:
        """Return raw posterior draws for *var* under this intervention.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        np.ndarray
            1-D array of posterior draws, shape ``(n_samples,)``.
        """
        return self._draw(var)

    def mean(self, var: str) -> float:
        """Return the posterior mean of *var* under this intervention."""
        return float(np.mean(self._draw(var)))

    def hdi(self, var: str, prob: float = DEFAULT_HDI_PROB) -> np.ndarray:
        """Return the highest-density interval for *var*.

        Parameters
        ----------
        var : str
            Variable name.
        prob : float
            Probability mass of the interval (default 0.94).

        Returns
        -------
        np.ndarray
            Array of ``[lower, upper]``.
        """
        return compute_hdi(self._draw(var), prob=prob)

    def by_time(self, var: str) -> np.ndarray:
        """Return per-time-step posterior draws, shape ``(n_times, n_samples)``.

        Only available for panel ``do(simulate_over="time")`` results.
        The time axis corresponds to :attr:`time_index`.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        np.ndarray
            Shape ``(n_times, n_samples)``.

        Raises
        ------
        ValueError
            If per-time data is not available (cross-sectional do).
        """
        if self._values_by_time is None:
            raise ValueError(
                "Per-time data not available. "
                "Use do(simulate_over='time') to get per-time results."
            )
        return self._draw_by_time(var)

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        return self._time_index

    def __sub__(self, other: DoResult) -> DoResult:
        """Element-wise contrast between two DoResults."""
        return DoResult(ds=_subtract_stores(self._ds, other._ds))

    def _repr_compact(self) -> str:
        values = self._values
        if not values:
            return "DoResult(empty)"
        n_samples = len(next(iter(values.values())))
        n_vars = len(values)
        return f"DoResult({n_samples} draws, {n_vars} variables)"

    def _repr_spec(self) -> ReprSpec:
        values = self._values
        if not values:
            return ReprSpec(title="DoResult (empty)", rows=[])
        n_samples = len(next(iter(values.values())))
        n_vars = len(values)
        rows = []
        for var, draws in values.items():
            mean = float(np.mean(draws))
            lo, hi = compute_hdi(draws)
            rows.append([var, f"{mean:.2f}", f"[{lo:.2f}, {hi:.2f}]"])
        return ReprSpec(
            title=f"DoResult — {n_samples} draws, {n_vars} variables",
            rows=rows,
            columns=["variable", "mean", hdi_label()],
            footer="Methods: .draws() .mean() .hdi() .by_time() .dataset",
        )


class EstimandResult(ResultReprMixin):
    """Posterior draws for a causal estimand (ATE, CATE, ATT, ATU).

    Unlike :class:`DoResult`, which describes the whole system under an
    intervention, an :class:`EstimandResult` knows which outcome was asked
    about, so its accessors default to that variable and no longer require
    the outcome to be re-specified.

    The same per-variable draws are retained, so ``mean(other_var)`` still
    works for any variable in the contrast.

    Internal storage mirrors :class:`DoResult`: an :class:`xarray.Dataset`
    exposed as :attr:`dataset` with dims ``("chain", "draw")`` (and
    ``"time"`` for panel estimands). The ``_values`` / ``_values_by_time``
    attributes are backward-compatible dict views.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Contrast draws for each variable, shape ``(n_samples,)`` where
        ``n_samples`` is ``chain * draw`` or ``chain * draw * unit``.
    outcome : str
        The outcome variable; the default target of all accessors.
    treatment : str
        The treatment variable that was intervened on.
    estimand : str
        Estimand label, e.g. ``"ATE"``, ``"CATE"``, ``"ATT"``, ``"ATU"``.
    values_by_time : dict[str, np.ndarray] | None
        Per-time-step contrast draws, shape ``(n_times, n_samples)``.
    time_index : array-like | None
        Time labels for the ``time`` dim.
    ds : xr.Dataset | None
        Pre-built internal Dataset (takes precedence over the dicts).
    n_chains, n_draws : int | None
        Chain/draw sizes for reconstructing dims from flat arrays.
    """

    def __init__(
        self,
        values: dict[str, np.ndarray] | None = None,
        outcome: str | None = None,
        treatment: str | None = None,
        estimand: str | None = None,
        values_by_time: dict[str, np.ndarray] | None = None,
        time_index: np.ndarray | None = None,
        *,
        ds: xr.Dataset | None = None,
        n_chains: int | None = None,
        n_draws: int | None = None,
    ) -> None:
        if ds is not None:
            self._ds = ds
        else:
            self._ds = _build_dataset(
                values or {},
                values_by_time,
                time_index,
                n_chains=n_chains,
                n_draws=n_draws,
            )
        # outcome/treatment/estimand are required metadata; keep them as
        # plain attributes (not derived from the Dataset).
        self._default_var: str = outcome if outcome is not None else ""
        self._treatment: str = treatment if treatment is not None else ""
        self._estimand: str = estimand if estimand is not None else ""

    @property
    def dataset(self) -> xr.Dataset:
        """Labeled contrast draws as an :class:`xarray.Dataset`.

        See :attr:`DoResult.dataset` for dim conventions and aliasing notes.
        """
        return self._ds

    def _draw(self, var: str) -> np.ndarray:
        """Flat ``(n_samples,)`` contrast draws for one variable."""
        return _stack_sample(self._ds[var])

    def _draw_by_time(self, var: str) -> np.ndarray:
        """Per-time ``(n_times, n_samples)`` contrast draws for one variable."""
        return _stack_sample_time(self._ds[var])

    @classmethod
    def from_contrast(
        cls,
        contrast: DoResult,
        outcome: str,
        treatment: str,
        estimand: str,
    ) -> EstimandResult:
        """Wrap a :class:`DoResult` contrast as a focused estimand result."""
        return cls(
            ds=contrast.dataset,
            outcome=outcome,
            treatment=treatment,
            estimand=estimand,
        )

    @property
    def outcome(self) -> str:
        """The outcome variable this estimand targets."""
        return self._default_var

    @property
    def treatment(self) -> str:
        """The treatment variable that was intervened on."""
        return self._treatment

    # -- backward-compatible dict views (backed by the xarray store) --------

    @property
    def _values(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for var in self._ds.data_vars:
            out[str(var)] = _stack_sample(self._ds[var])
        return out

    @property
    def _values_by_time(self) -> dict[str, np.ndarray] | None:
        by_time_vars = [v for v in self._ds.data_vars if "time" in self._ds[v].dims]
        if not by_time_vars:
            return None
        return {str(var): _stack_sample_time(self._ds[var]) for var in by_time_vars}

    @property
    def _time_index(self) -> np.ndarray | None:
        if "time" in self._ds.coords:
            return np.asarray(self._ds["time"].values)
        return None

    def _resolve(self, var: str | None) -> str:
        key = self._default_var if var is None else var
        available = [str(v) for v in self._ds.data_vars]
        if key not in available:
            raise KeyError(
                f"Unknown variable '{key}'. Available variables: {sorted(available)}"
            )
        return key

    def draws(self, var: str | None = None) -> np.ndarray:
        """Return raw contrast draws, defaulting to the outcome variable.

        Parameters
        ----------
        var : str | None
            Variable name. Defaults to the outcome.

        Returns
        -------
        np.ndarray
            1-D array of posterior draws, shape ``(n_samples,)``.
        """
        return self._draw(self._resolve(var))

    def mean(self, var: str | None = None) -> float:
        """Return the posterior mean, defaulting to the outcome variable."""
        return float(np.mean(self._draw(self._resolve(var))))

    def hdi(self, var: str | None = None, prob: float = DEFAULT_HDI_PROB) -> np.ndarray:
        """Return the highest-density interval, defaulting to the outcome.

        Parameters
        ----------
        var : str | None
            Variable name. Defaults to the outcome.
        prob : float
            Probability mass of the interval (default 0.94).

        Returns
        -------
        np.ndarray
            Array of ``[lower, upper]``.
        """
        return compute_hdi(self._draw(self._resolve(var)), prob=prob)

    def prob(self, expr: str, var: str | None = None) -> float:
        """Return the posterior probability that the estimand satisfies *expr*.

        *expr* is a comparison applied to the (outcome) estimand draws, e.g.
        ``"> 0"`` returns ``P(estimand > 0)``.

        Parameters
        ----------
        expr : str
            A comparison such as ``"> 0"``, ``">= 1"``, or ``"< -0.5"``.
        var : str | None
            Variable name. Defaults to the outcome.

        Returns
        -------
        float
            Fraction of draws satisfying *expr*.
        """
        draws = self._draw(self._resolve(var))
        namespace: dict[str, Any] = {"x": draws, "np": np, "__builtins__": {}}
        try:
            mask = eval(f"x {expr}", namespace)  # noqa: S307
        except (SyntaxError, NameError) as err:
            raise ValueError(
                f"Could not parse prob() expression '{expr}'. "
                f"Pass a comparison applied to the estimand, e.g. '> 0'."
            ) from err
        return float(np.mean(mask))

    def summary(self, prob: float = DEFAULT_HDI_PROB) -> pd.DataFrame:
        """Return a one-row tidy summary of the estimand.

        Columns: ``outcome``, ``treatment``, ``mean``, ``sd``, ``hdi_3%``,
        ``hdi_97%``, ``p(>0)``. The index holds the estimand label.

        Parameters
        ----------
        prob : float
            Probability mass of the reported HDI (default 0.94).

        Returns
        -------
        pd.DataFrame
            Single-row summary indexed by the estimand label.
        """
        draws = self._draw(self._default_var)
        lo, hi = compute_hdi(draws, prob=prob)
        lower_pct = (1.0 - prob) / 2.0 * 100.0
        upper_pct = (1.0 + prob) / 2.0 * 100.0
        row = {
            "outcome": self._default_var,
            "treatment": self._treatment,
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws)),
            f"hdi_{lower_pct:.0f}%": float(lo),
            f"hdi_{upper_pct:.0f}%": float(hi),
            "p(>0)": float(np.mean(draws > 0)),
        }
        return pd.DataFrame([row], index=pd.Index([self._estimand], name="estimand"))

    def by_time(self, var: str | None = None) -> np.ndarray:
        """Return per-time-step contrast draws, shape ``(n_times, n_samples)``.

        Only available for panel ``simulate_over="time"`` estimands.

        Parameters
        ----------
        var : str | None
            Variable name. Defaults to the outcome.

        Returns
        -------
        np.ndarray
            Shape ``(n_times, n_samples)``.

        Raises
        ------
        ValueError
            If per-time data is not available (cross-sectional estimand).
        """
        if self._values_by_time is None:
            raise ValueError(
                "Per-time data not available. "
                "Use simulate_over='time' to get per-time results."
            )
        return self._draw_by_time(self._resolve(var))

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        return self._time_index

    def __float__(self) -> float:
        """Posterior mean of the estimand (outcome variable)."""
        return self.mean()

    def __sub__(self, other: EstimandResult) -> EstimandResult:
        """Element-wise contrast between two estimands, preserving the outcome."""
        return EstimandResult(
            ds=_subtract_stores(self._ds, other._ds),
            outcome=self._default_var,
            treatment=self._treatment,
            estimand=self._estimand,
        )

    def _repr_compact(self) -> str:
        draws = self._draw(self._default_var)
        mean = float(np.mean(draws))
        lo, hi = compute_hdi(draws)
        label = hdi_label()
        return (
            f"{self._estimand}: {self._treatment}→{self._default_var}  "
            f"mean={mean:.2f}  {label}=[{lo:.2f}, {hi:.2f}]"
        )

    def _repr_spec(self) -> ReprSpec:
        draws = self._draw(self._default_var)
        mean = float(np.mean(draws))
        lo, hi = compute_hdi(draws)
        p_gt_0 = float(np.mean(draws > 0))
        n_samples = len(draws)
        label = hdi_label()
        return ReprSpec(
            title=f"{self._estimand} of {self._treatment} on {self._default_var}",
            rows=[
                ["Mean", f"{mean:.2f}"],
                [label, f"[{lo:.2f}, {hi:.2f}]"],
                ["P(> 0)", f"{p_gt_0:.2f}"],
                ["Draws", str(n_samples)],
            ],
            footer="Methods: .hdi() .prob() .summary() .draws() .by_time() .dataset",
        )


def _apply_inverse_link(mu_vals: Any, family: str) -> Any:
    """Map linear-predictor values back to the response scale."""
    if family == "bernoulli":
        if isinstance(mu_vals, np.ndarray):
            return 1.0 / (1.0 + np.exp(-mu_vals))
        return 1.0 / (1.0 + pt.exp(-mu_vals))
    if family in ("poisson", "negbinomial"):
        if isinstance(mu_vals, np.ndarray):
            return np.exp(mu_vals)
        return pt.exp(mu_vals)
    return mu_vals


def _replace_graph(expr: Any, replacements: dict[Any, Any]) -> Any:
    """Clone *expr* with only the replacements that appear in its graph."""
    if not replacements:
        return expr
    result = expr
    for var, replacement in replacements.items():
        graph_vars = set(ancestors([result]))
        if var in graph_vars:
            result = graph_replace(result, replace={var: replacement}, strict=False)
    return result


def _float_descendants_of(source_var: Any, exprs: list[Any]) -> list[Any]:
    """Find float graph nodes that directly cast or transform *source_var*."""
    descendants: list[Any] = []
    seen: set[Any] = set()
    for expr in exprs:
        for node_var in ancestors([expr]):
            owner = getattr(node_var, "owner", None)
            dtype = getattr(node_var, "dtype", "")
            if (
                owner is not None
                and source_var in owner.inputs
                and str(dtype).startswith("float")
                and node_var not in seen
            ):
                descendants.append(node_var)
                seen.add(node_var)
    return descendants


def _exogenous_fill(values: np.ndarray) -> float:
    """Mean used to fill an exogenous variable for empirical integration.

    Skips NaN/null entries (matching the historical pandas ``skipna=True``
    behavior) and maps an all-missing column to ``nan`` without emitting the
    ``RuntimeWarning`` that ``np.nanmean`` raises on an empty/all-NaN slice.
    Both the full-column and subgroup-slice paths share this policy so they
    cannot diverge on missing-value handling.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def run_do_pymc(
    gen_model: pm.Model,
    graph_info: GraphInfo,
    idata: xr.DataTree,
    data: nw.DataFrame,
    set: dict[str, float | np.ndarray] | None = None,
    kind: str = "mean",
    families: dict[str, str] | None = None,
    subgroup_indices: np.ndarray | None = None,
) -> DoResult:
    """Run the do-operator using PyMC-native graph surgery.

    For ``kind="predictive"``: uses ``pm.do()`` on the generative model
    followed by ``pm.sample_posterior_predictive()`` to forward-sample
    through the causal chain with residual noise.

    For ``kind="mean"``: uses ``pm.do()`` for interventions, then computes
    cloned mu Deterministics with upstream endogenous variables replaced by
    their expected values for noise-free mean propagation.

    Parameters
    ----------
    gen_model : pm.Model
        The generative PyMC model (endogenous vars are free RVs).
    graph_info : GraphInfo
        DAG with topological order and node classification.
    idata : xarray.DataTree
        Posterior samples from ``pm.sample()``.
    data : nw.DataFrame
        Observed data (used for sizing intervention arrays).
    set : dict[str, float] | None
        Variables to intervene on, with their fixed values.
    kind : str
        ``"mean"`` for deterministic propagation, ``"predictive"`` to
        include residual noise at each step.
    families : dict[str, str] | None
        Per-variable distribution families (e.g. ``{"Y": "bernoulli"}``).
        Used to apply the inverse link function for ``kind="mean"``.
    subgroup_indices : np.ndarray | None
        Row indices for subgroup-aware empirical integration. When
        provided, endogenous variable draws are averaged over only
        these rows (e.g., the treated subgroup for ATT). ``None``
        (default) uses all rows.

    Returns
    -------
    DoResult
        Propagated posterior draws for every endogenous variable.
    """

    if set is None:
        set = {}
    if families is None:
        families = {}

    N = len(data)
    latent = graph_info.latent

    free_rv_names = {rv.name for rv in gen_model.free_RVs}
    det_names_set = {d.name for d in gen_model.deterministics}

    block_vars = {
        var
        for var in graph_info.topological_order
        if var in graph_info.endogenous
        and var not in free_rv_names
        and var not in latent
        and f"mu_{var}" in det_names_set
    }

    replacements: dict[str, Any] = {}
    for var, val in set.items():
        key = f"mu_{var}" if (var in latent or var in block_vars) else var
        arr = np.full(N, val)
        target_dtype = gen_model[key].dtype
        replacements[key] = arr.astype(target_dtype)

    if kind == "mean":
        do_model = pm.do(gen_model, replacements)

        mean_det_names: dict[str, str] = {}
        expr_replacements: dict[Any, Any] = {}
        mu_source_exprs = [
            do_model[f"mu_{var}"] * 1
            for var in graph_info.topological_order
            if var in graph_info.endogenous
            and var not in set
            and f"mu_{var}" in do_model.named_vars
        ]
        with do_model:
            for var in graph_info.topological_order:
                if var in graph_info.endogenous and var not in set:
                    mu_name = f"mu_{var}"
                    mu_expr = _replace_graph(do_model[mu_name] * 1, expr_replacements)
                    mean_name = f"pathmc_mean_{mu_name}"
                    pm.Deterministic(mean_name, mu_expr)
                    mean_det_names[var] = mean_name

                    response_expr = _apply_inverse_link(mu_expr, families.get(var, ""))
                    if var in do_model.named_vars:
                        model_var = do_model[var]
                        if model_var.dtype.startswith("float"):
                            expr_replacements[model_var] = response_expr
                        else:
                            for key in _float_descendants_of(
                                model_var, mu_source_exprs
                            ):
                                expr_replacements[key] = response_expr
                    elif mu_name in do_model.named_vars:
                        expr_replacements[do_model[mu_name]] = response_expr

        posterior_ds = posterior(idata)
        missing_rv_names = [
            rv.name for rv in do_model.free_RVs if rv.name not in posterior_ds
        ]
        if missing_rv_names:
            n_chains = posterior_ds.sizes["chain"]
            n_draws = posterior_ds.sizes["draw"]
            dummy_vars: dict[str, xr.DataArray] = {}
            for name in missing_rv_names:
                rv = do_model[name]
                var_shape = tuple(s for s in rv.type.shape if s is not None) or (N,)
                dummy = np.zeros((n_chains, n_draws, *var_shape), dtype=rv.dtype)
                dims = ["chain", "draw"] + [
                    f"{name}_dim_{i}" for i in range(len(var_shape))
                ]
                dummy_vars[name] = xr.DataArray(dummy, dims=dims)
            posterior_ds = posterior_ds.assign(dummy_vars)

        det = pm.compute_deterministics(
            posterior_ds,
            model=do_model,
            var_names=list(mean_det_names.values()),
            progressbar=False,
        )

        post = posterior(idata)
        n_samples = post.sizes["chain"] * post.sizes["draw"]
        values: dict[str, np.ndarray] = {}
        for var in graph_info.topological_order:
            if var in set:
                values[var] = np.full(n_samples, set[var])
            elif var in graph_info.exogenous:
                if var in data.columns:
                    col = data[var].to_numpy()
                    if subgroup_indices is not None:
                        col = col[subgroup_indices]
                    values[var] = np.full(n_samples, _exogenous_fill(col))
                else:
                    values[var] = np.zeros(n_samples)
            else:
                mu_raw = det[mean_det_names[var]].to_numpy()
                if subgroup_indices is not None and mu_raw.ndim >= 3:
                    mu_raw = mu_raw[:, :, subgroup_indices]
                # Map to the response scale per unit, then average over units
                # within each posterior draw (g-computation standardization).
                # The inverse link must be applied before averaging because
                # E[g^{-1}(mu)] != g^{-1}(E[mu]) for non-identity links. This
                # yields one value per draw (n_samples,), not per draw-unit pair.
                resp = _apply_inverse_link(mu_raw, families.get(var, ""))
                if resp.ndim >= 3:
                    values[var] = resp.reshape(n_samples, -1).mean(axis=1)
                else:
                    values[var] = resp.reshape(n_samples)

        return DoResult(
            values=values, n_chains=post.sizes["chain"], n_draws=post.sizes["draw"]
        )

    do_model = pm.do(gen_model, replacements)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Could not extract data from symbolic observation"
        )
        with do_model:
            ppc = pm.sample_posterior_predictive(idata, progressbar=False)

    extra_det_names = [
        f"mu_{var}"
        for var in graph_info.topological_order
        if var not in set and (var in latent or var in block_vars)
    ]
    if extra_det_names:
        posterior_ds = posterior(idata)
        missing_rv_names = [
            rv.name for rv in do_model.free_RVs if rv.name not in posterior_ds
        ]
        if missing_rv_names:
            n_chains = posterior_ds.sizes["chain"]
            n_draws = posterior_ds.sizes["draw"]
            fill_vars: dict[str, xr.DataArray] = {}
            for name in missing_rv_names:
                rv = do_model[name]
                var_shape = tuple(s for s in rv.type.shape if s is not None) or (N,)
                dummy = np.zeros((n_chains, n_draws, *var_shape), dtype=rv.dtype)
                dims = ["chain", "draw"] + [
                    f"{name}_dim_{i}" for i in range(len(var_shape))
                ]
                fill_vars[name] = xr.DataArray(dummy, dims=dims)
            posterior_ds = posterior_ds.assign(fill_vars)

        extra_det = pm.compute_deterministics(
            posterior_ds,
            model=do_model,
            var_names=extra_det_names,
            progressbar=False,
        )
    else:
        extra_det = None

    post = posterior(idata)
    n_samples = post.sizes["chain"] * post.sizes["draw"]
    values = {}
    # Track per-variable unit counts so the Dataset builder can label the
    # unit dim for predictive (flattened chain*draw*N) arrays.
    n_units_per_var: dict[str, int] = {}
    for var in graph_info.topological_order:
        if var in set:
            values[var] = np.full(n_samples, set[var])
        elif var in graph_info.exogenous:
            if var in data.columns:
                col = data[var].to_numpy()
                if subgroup_indices is not None:
                    col = col[subgroup_indices]
                values[var] = np.full(n_samples, _exogenous_fill(col))
            else:
                values[var] = np.zeros(n_samples)
        elif var in ppc.posterior_predictive:
            raw = ppc.posterior_predictive[var].to_numpy()
            if subgroup_indices is not None and raw.ndim >= 3:
                raw = raw[:, :, subgroup_indices]
            n_units_per_var[var] = raw.shape[-1] if raw.ndim >= 3 else 1
            values[var] = raw.flatten()
        elif f"mu_{var}" in ppc.posterior_predictive:
            raw = ppc.posterior_predictive[f"mu_{var}"].to_numpy()
            if subgroup_indices is not None and raw.ndim >= 3:
                raw = raw[:, :, subgroup_indices]
            n_units_per_var[var] = raw.shape[-1] if raw.ndim >= 3 else 1
            values[var] = raw.flatten()
        elif extra_det is not None and f"mu_{var}" in extra_det:
            raw = extra_det[f"mu_{var}"].to_numpy()
            if subgroup_indices is not None and raw.ndim >= 3:
                raw = raw[:, :, subgroup_indices]
            n_units_per_var[var] = raw.shape[-1] if raw.ndim >= 3 else 1
            values[var] = raw.flatten()

    # Predictive arrays flatten chain*draw*N into one axis; pass per-var unit
    # counts so the builder can label a unit dim where lengths match.
    return DoResult(
        values=values,
        n_chains=post.sizes["chain"],
        n_draws=post.sizes["draw"],
        n_units_per_var=n_units_per_var or None,
    )


def run_do_panel_unified(
    gen_model: pm.Model,
    graph_info: GraphInfo,
    idata: xr.DataTree,
    panel_info: PanelInfo,
    scan_info: Any,
    set: dict[str, float | np.ndarray] | None = None,
    kind: str = "mean",
    families: dict[str, str] | None = None,
) -> DoResult:
    """Run the do-operator on a scan-compiled panel model.

    Uses the same ``pm.do()`` approach as cross-sectional, but handles
    the ``(n_times, n_units)`` output shape of the scan model and
    produces per-time-step results in :class:`DoResult`.

    Parameters
    ----------
    gen_model : pm.Model
        The scan-compiled generative model.
    graph_info : GraphInfo
        DAG with topological order and node classification.
    idata : xarray.DataTree
        Posterior samples.
    panel_info : PanelInfo
        Panel metadata.
    scan_info : PanelScanInfo
        Scan compilation metadata (sort indices, dimensions).
    set : dict[str, float | np.ndarray] | None
        Intervention values. Arrays of shape ``(n_times,)`` for
        time-varying interventions.
    kind : str
        ``"mean"`` or ``"predictive"``.
    families : dict[str, str] | None
        Per-variable distribution families.
    """
    if set is None:
        set = {}
    if families is None:
        families = {}

    n_times = scan_info.n_times
    n_units = scan_info.n_units
    latent = graph_info.latent

    replacements: dict[str, Any] = {}
    for var, val in set.items():
        if isinstance(val, np.ndarray):
            mat = np.broadcast_to(val[:, None], (n_times, n_units)).copy()
        else:
            mat = np.full((n_times, n_units), val)
        key = f"mu_{var}" if var in latent else var
        target_dtype = gen_model[key].dtype
        replacements[key] = mat.astype(target_dtype)

    if kind == "mean":
        for var in graph_info.topological_order:
            if var in graph_info.endogenous and var not in set and var not in latent:
                replacements[var] = gen_model[f"mu_{var}"] * 1

        stochastic_latent = {
            v for v in latent if families.get(v, "gaussian") == "latent_normal"
        }

        do_model = pm.do(gen_model, replacements)
        det_names = []
        for var in graph_info.topological_order:
            if var in graph_info.endogenous:
                if var in stochastic_latent:
                    det_names.append(var)
                else:
                    det_names.append(f"mu_{var}")
        det = pm.compute_deterministics(
            posterior(idata),
            model=do_model,
            var_names=det_names,
            progressbar=False,
        )

        post = posterior(idata)
        n_samples = post.sizes["chain"] * post.sizes["draw"]

        values: dict[str, np.ndarray] = {}
        values_by_time: dict[str, np.ndarray] = {}

        for var in graph_info.topological_order:
            if var in set:
                val = set[var]
                scalar = float(np.mean(val)) if isinstance(val, np.ndarray) else val
                values[var] = np.full(n_samples, scalar)
                if isinstance(val, np.ndarray):
                    values_by_time[var] = np.broadcast_to(
                        val[:, None], (n_times, n_samples)
                    ).copy()
                else:
                    values_by_time[var] = np.full((n_times, n_samples), val)
            elif var in graph_info.exogenous:
                values[var] = np.zeros(n_samples)
            elif var in graph_info.endogenous:
                det_key = var if var in stochastic_latent else f"mu_{var}"
                mu_raw = det[det_key].to_numpy()
                mu_flat = mu_raw.reshape(-1, n_times, n_units)
                by_time = mu_flat.mean(axis=2).T
                values_by_time[var] = by_time
                values[var] = by_time.mean(axis=0)

        time_idx = (
            np.array(scan_info.time_values)
            if scan_info.time_values
            else np.arange(n_times)
        )
        return DoResult(
            values=values,
            values_by_time=values_by_time,
            time_index=time_idx,
            n_chains=post.sizes["chain"],
            n_draws=post.sizes["draw"],
        )

    # kind == "predictive"
    do_model = pm.do(gen_model, replacements)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Could not extract data from symbolic observation"
        )
        with do_model:
            ppc = pm.sample_posterior_predictive(idata, progressbar=False)

    stochastic_latent = {
        v for v in latent if families.get(v, "gaussian") == "latent_normal"
    }
    latent_det_names = []
    for var in graph_info.topological_order:
        if var in latent and var not in set:
            latent_det_names.append(var if var in stochastic_latent else f"mu_{var}")
    if latent_det_names:
        latent_det = pm.compute_deterministics(
            posterior(idata),
            model=do_model,
            var_names=latent_det_names,
            progressbar=False,
        )
    else:
        latent_det = None

    post = posterior(idata)
    n_samples = post.sizes["chain"] * post.sizes["draw"]

    values = {}
    values_by_time = {}
    for var in graph_info.topological_order:
        if var in set:
            val = set[var]
            scalar = float(np.mean(val)) if isinstance(val, np.ndarray) else val
            values[var] = np.full(n_samples, scalar)
        elif var in graph_info.exogenous:
            values[var] = np.zeros(n_samples)
        elif var in ppc.posterior_predictive:
            raw = ppc.posterior_predictive[var].to_numpy()
            flat = raw.reshape(-1, n_times, n_units)
            by_time = flat.mean(axis=2).T
            values_by_time[var] = by_time
            values[var] = by_time.mean(axis=0)
        elif latent_det is not None:
            det_key = var if var in stochastic_latent else f"mu_{var}"
            if det_key in latent_det:
                mu_raw = latent_det[det_key].to_numpy()
                mu_flat = mu_raw.reshape(-1, n_times, n_units)
                by_time = mu_flat.mean(axis=2).T
                values_by_time[var] = by_time
                values[var] = by_time.mean(axis=0)

    time_idx = (
        np.array(scan_info.time_values) if scan_info.time_values else np.arange(n_times)
    )
    return DoResult(
        values=values,
        values_by_time=values_by_time,
        time_index=time_idx,
        n_chains=post.sizes["chain"],
        n_draws=post.sizes["draw"],
    )
