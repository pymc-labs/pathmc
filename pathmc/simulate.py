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
    stack_dims = [d for d in ("chain", "draw", "unit") if d in da.dims]
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
    extra = [d for d in da.dims if d not in ("chain", "draw", "time", "unit")]
    if extra:
        da = da.mean(dim=extra)
    stack_dims = [d for d in ("chain", "draw") if d in da.dims]
    if not stack_dims:
        return np.asarray(da.values)
    stacked = da.stack(sample=stack_dims)
    return stacked.transpose("time", "sample").values


def _chain_draw_coords(n_chains: int, n_draws: int) -> dict[str, np.ndarray]:
    """Integer ``chain`` / ``draw`` coordinate labels."""
    return {"chain": np.arange(n_chains), "draw": np.arange(n_draws)}


def _extra_dims(da: xr.DataArray) -> list[str]:
    """Dims other than ``chain`` / ``draw`` on a PyMC output array."""
    return [d for d in da.dims if d not in ("chain", "draw")]


def _select_subgroup(
    da: xr.DataArray, subgroup_indices: np.ndarray | None
) -> xr.DataArray:
    """Restrict observation rows to *subgroup_indices* (first extra dim)."""
    if subgroup_indices is None:
        return da
    extra = _extra_dims(da)
    if extra:
        return da.isel({extra[0]: subgroup_indices})
    return da


def _const_chain_draw(
    fill: float | np.ndarray, n_chains: int, n_draws: int
) -> xr.DataArray:
    """Constant ``(chain, draw)`` array with integer coords."""
    scalar = float(np.mean(fill)) if isinstance(fill, np.ndarray) else float(fill)
    coords = _chain_draw_coords(n_chains, n_draws)
    return xr.DataArray(
        np.broadcast_to(scalar, (n_chains, n_draws)),
        dims=("chain", "draw"),
        coords=coords,
    )


def _const_time_chain_draw(
    val: float | np.ndarray,
    n_times: int,
    n_chains: int,
    n_draws: int,
    time_index: np.ndarray,
) -> xr.DataArray:
    """Constant or time-varying ``(time, chain, draw)`` array."""
    coords = {"time": np.asarray(time_index), **_chain_draw_coords(n_chains, n_draws)}
    if isinstance(val, np.ndarray):
        time_da = xr.DataArray(val, dims=("time",), coords={"time": coords["time"]})
        ones = xr.DataArray(
            np.ones((n_chains, n_draws)),
            dims=("chain", "draw"),
            coords=_chain_draw_coords(n_chains, n_draws),
        )
        return (time_da * ones).transpose("time", "chain", "draw")
    return xr.DataArray(
        np.broadcast_to(val, (n_times, n_chains, n_draws)),
        dims=("time", "chain", "draw"),
        coords=coords,
    )


def _mean_to_chain_draw(da: xr.DataArray, family: str) -> xr.DataArray:
    """Collapse observation dims to ``(chain, draw)`` on the response scale."""
    da = _apply_inverse_link(da, family)
    extra = _extra_dims(da)
    if extra:
        da = da.mean(dim=extra)
    return da.transpose("chain", "draw")


def _predictive_to_chain_draw(
    da: xr.DataArray, subgroup_indices: np.ndarray | None
) -> xr.DataArray:
    """Label predictive draws as ``(chain, draw)`` or ``(chain, draw, unit)``."""
    da = _select_subgroup(da, subgroup_indices)
    extra = _extra_dims(da)
    if len(extra) == 1 and da.sizes[extra[0]] > 1:
        return da.rename({extra[0]: "unit"}).transpose("chain", "draw", "unit")
    if extra:
        da = da.mean(dim=extra)
    return da.transpose("chain", "draw")


def _panel_to_time_chain_draw(
    da: xr.DataArray,
    time_index: np.ndarray,
) -> xr.DataArray:
    """Average over units and return ``(time, chain, draw)``."""
    extra = _extra_dims(da)
    if len(extra) >= 2:
        time_dim, unit_dim = extra[0], extra[1]
        out = da.mean(dim=unit_dim).transpose(time_dim, "chain", "draw")
        if time_dim != "time":
            out = out.rename({time_dim: "time"})
        return out.assign_coords(time=np.asarray(time_index))
    if len(extra) == 1:
        time_dim = extra[0]
        out = da.transpose(time_dim, "chain", "draw")
        if time_dim != "time":
            out = out.rename({time_dim: "time"})
        return out.assign_coords(time=np.asarray(time_index))
    raise ValueError("Panel output missing time/unit dims on deterministics")


def _fill_missing_posterior_rvs(
    posterior_ds: xr.Dataset,
    do_model: pm.Model,
    n_obs: int,
) -> xr.Dataset:
    """Add zero-filled dummy RVs missing from the posterior Dataset."""
    missing = [rv.name for rv in do_model.free_RVs if rv.name not in posterior_ds]
    if not missing:
        return posterior_ds
    n_chains = posterior_ds.sizes["chain"]
    n_draws = posterior_ds.sizes["draw"]
    fill_vars: dict[str, xr.DataArray] = {}
    for name in missing:
        rv = do_model[name]
        var_shape = tuple(s for s in rv.type.shape if s is not None) or (n_obs,)
        dummy = np.zeros((n_chains, n_draws, *var_shape), dtype=rv.dtype)
        dims = ["chain", "draw"] + [f"{name}_dim_{i}" for i in range(len(var_shape))]
        fill_vars[name] = xr.DataArray(dummy, dims=dims)
    return posterior_ds.assign(fill_vars)


def _predictive_source(
    ppc: Any,
    extra_det: xr.Dataset | None,
    var: str,
) -> xr.DataArray | None:
    """Return labelled predictive draws for *var*, if present."""
    if var in ppc.posterior_predictive:
        return ppc.posterior_predictive[var]
    if f"mu_{var}" in ppc.posterior_predictive:
        return ppc.posterior_predictive[f"mu_{var}"]
    if extra_det is not None and f"mu_{var}" in extra_det:
        return extra_det[f"mu_{var}"]
    return None


def _sample_count(da: xr.DataArray) -> int:
    """Posterior sample count matching :func:`_stack_sample` flattening."""
    if "time" in da.dims:
        da = da.mean(dim="time")
    n = da.sizes["chain"] * da.sizes["draw"]
    if "unit" in da.dims:
        n *= da.sizes["unit"]
    return n


def _has_by_time(ds: xr.Dataset) -> bool:
    """True when any variable carries a ``time`` dim."""
    return any("time" in ds[v].dims for v in ds.data_vars)


class _DrawStorageMixin:
    """Shared labelled xarray storage for draw-backed result types."""

    _ds: xr.Dataset

    @property
    def dataset(self) -> xr.Dataset:
        """Labeled draws as an :class:`xarray.Dataset`.

        Variables use dims ``("chain", "draw")`` for cross-sectional results,
        plus ``"unit"`` for ``kind="predictive"`` and ``"time"`` for panel
        per-time results. For a flat ``(n_samples,)`` numpy view, use
        :meth:`draws` instead.
        """
        return self._ds

    def _draw(self, var: str) -> np.ndarray:
        """Flat ``(n_samples,)`` draws for one variable."""
        return _stack_sample(self._ds[var])

    def _draw_by_time(self, var: str) -> np.ndarray:
        """Per-time ``(n_times, n_samples)`` draws for one panel variable."""
        return _stack_sample_time(self._ds[var])

    @property
    def _time_index(self) -> np.ndarray | None:
        """Time coord of the internal Dataset, or None."""
        if "time" in self._ds.coords:
            return np.asarray(self._ds["time"].values)
        return None

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        return self._time_index

    def _by_time_draws(self, var: str) -> np.ndarray:
        """Per-time draws, raising when no ``time`` dim is stored."""
        if not _has_by_time(self._ds):
            raise ValueError(
                "Per-time data not available. "
                "Use do(simulate_over='time') to get per-time results."
            )
        return self._draw_by_time(var)


class DoResult(_DrawStorageMixin, ResultReprMixin):
    """Container for propagated posterior draws under an intervention.

    Supports ``.mean(var)``, ``.hdi(var)``, and contrast arithmetic
    via subtraction (``scenario - baseline``).

    For panel ``do(simulate_over="time")``, the result also stores
    per-time-step draws accessible via :meth:`by_time`.

    Internal storage is an :class:`xarray.Dataset` exposed as :attr:`dataset`
    with named dims ``("chain", "draw")`` for cross-sectional draws and an
    additional ``"time"`` dim for panel per-time draws. Public accessors such
    as :meth:`draws` flatten ``chain``/``draw`` (and ``unit`` when present)
    into a 1-D numpy sample vector.

    Parameters
    ----------
    ds : xr.Dataset
        Labelled posterior draws with dims ``("chain", "draw")`` and,
        optionally, ``"unit"`` or ``"time"``.
    """

    def __init__(self, *, ds: xr.Dataset) -> None:
        self._ds = ds

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
        return self._by_time_draws(var)

    def __sub__(self, other: DoResult) -> DoResult:
        """Element-wise contrast between two DoResults."""
        common = [str(v) for v in self._ds.data_vars if v in other._ds.data_vars]
        return DoResult(ds=self._ds[common] - other._ds[common])

    def _repr_compact(self) -> str:
        if not self._ds.data_vars:
            return "DoResult(empty)"
        vars_list = [str(v) for v in self._ds.data_vars]
        n_samples = _sample_count(self._ds[vars_list[0]])
        n_vars = len(vars_list)
        return f"DoResult({n_samples} draws, {n_vars} variables)"

    def _repr_spec(self) -> ReprSpec:
        if not self._ds.data_vars:
            return ReprSpec(title="DoResult (empty)", rows=[])
        vars_list = [str(v) for v in self._ds.data_vars]
        n_samples = _sample_count(self._ds[vars_list[0]])
        n_vars = len(vars_list)
        rows = []
        for var in vars_list:
            draws = self._draw(var)
            mean = float(np.mean(draws))
            lo, hi = compute_hdi(draws)
            rows.append([var, f"{mean:.2f}", f"[{lo:.2f}, {hi:.2f}]"])
        return ReprSpec(
            title=f"DoResult — {n_samples} draws, {n_vars} variables",
            rows=rows,
            columns=["variable", "mean", hdi_label()],
            footer="Methods: .draws() .mean() .hdi() .by_time() .dataset",
        )


class EstimandResult(_DrawStorageMixin, ResultReprMixin):
    """Posterior draws for a causal estimand (ATE, CATE, ATT, ATU).

    Unlike :class:`DoResult`, which describes the whole system under an
    intervention, an :class:`EstimandResult` knows which outcome was asked
    about, so its accessors default to that variable and no longer require
    the outcome to be re-specified.

    The same per-variable draws are retained, so ``mean(other_var)`` still
    works for any variable in the contrast.

    Internal storage mirrors :class:`DoResult`: an :class:`xarray.Dataset`
    exposed as :attr:`dataset` with dims ``("chain", "draw")`` (and
    ``"time"`` for panel estimands).

    Parameters
    ----------
    ds : xr.Dataset
        Labelled contrast draws.
    outcome : str
        The outcome variable; the default target of all accessors.
    treatment : str
        The treatment variable that was intervened on.
    estimand : str
        Estimand label, e.g. ``"ATE"``, ``"CATE"``, ``"ATT"``, ``"ATU"``.
    """

    def __init__(
        self,
        *,
        ds: xr.Dataset,
        outcome: str,
        treatment: str,
        estimand: str,
    ) -> None:
        self._ds = ds
        self._default_var = outcome
        self._treatment = treatment
        self._estimand = estimand

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
        return self._by_time_draws(self._resolve(var))

    def __float__(self) -> float:
        """Posterior mean of the estimand (outcome variable)."""
        return self.mean()

    def __sub__(self, other: EstimandResult) -> EstimandResult:
        """Element-wise contrast between two estimands, preserving the outcome."""
        common = [str(v) for v in self._ds.data_vars if v in other._ds.data_vars]
        return EstimandResult(
            ds=self._ds[common] - other._ds[common],
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
        n_samples = _sample_count(self._ds[self._default_var])
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
        if isinstance(mu_vals, xr.DataArray):
            return xr.apply_ufunc(
                lambda x: 1.0 / (1.0 + np.exp(-x)),
                mu_vals,
                dask="parallelized",
            )
        if isinstance(mu_vals, np.ndarray):
            return 1.0 / (1.0 + np.exp(-mu_vals))
        return 1.0 / (1.0 + pt.exp(-mu_vals))
    if family in ("poisson", "negbinomial"):
        if isinstance(mu_vals, xr.DataArray):
            return xr.apply_ufunc(np.exp, mu_vals, dask="parallelized")
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
        posterior_ds = _fill_missing_posterior_rvs(posterior_ds, do_model, N)

        det = pm.compute_deterministics(
            posterior_ds,
            model=do_model,
            var_names=list(mean_det_names.values()),
            progressbar=False,
        )

        post = posterior(idata)
        n_chains = post.sizes["chain"]
        n_draws = post.sizes["draw"]
        data_vars: dict[str, xr.DataArray] = {}
        for var in graph_info.topological_order:
            if var in set:
                data_vars[var] = _const_chain_draw(set[var], n_chains, n_draws)
            elif var in graph_info.exogenous:
                if var in data.columns:
                    col = data[var].to_numpy()
                    if subgroup_indices is not None:
                        col = col[subgroup_indices]
                    fill = _exogenous_fill(col)
                else:
                    fill = 0.0
                data_vars[var] = _const_chain_draw(fill, n_chains, n_draws)
            else:
                mu_da = _select_subgroup(det[mean_det_names[var]], subgroup_indices)
                data_vars[var] = _mean_to_chain_draw(mu_da, families.get(var, ""))

        return DoResult(ds=xr.Dataset(data_vars))

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
        posterior_ds = _fill_missing_posterior_rvs(posterior(idata), do_model, N)

        extra_det = pm.compute_deterministics(
            posterior_ds,
            model=do_model,
            var_names=extra_det_names,
            progressbar=False,
        )
    else:
        extra_det = None

    post = posterior(idata)
    n_chains = post.sizes["chain"]
    n_draws = post.sizes["draw"]
    predictive_vars: dict[str, xr.DataArray] = {}
    for var in graph_info.topological_order:
        if var in set:
            predictive_vars[var] = _const_chain_draw(set[var], n_chains, n_draws)
        elif var in graph_info.exogenous:
            if var in data.columns:
                col = data[var].to_numpy()
                if subgroup_indices is not None:
                    col = col[subgroup_indices]
                fill = _exogenous_fill(col)
            else:
                fill = 0.0
            predictive_vars[var] = _const_chain_draw(fill, n_chains, n_draws)
        elif (src := _predictive_source(ppc, extra_det, var)) is not None:
            predictive_vars[var] = _predictive_to_chain_draw(src, subgroup_indices)

    return DoResult(ds=xr.Dataset(predictive_vars))


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
        Panel metadata (reserved for future time-label wiring).
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
    del panel_info  # reserved for future time-label wiring from panel metadata

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
        n_chains = post.sizes["chain"]
        n_draws = post.sizes["draw"]
        time_idx = (
            np.array(scan_info.time_values)
            if scan_info.time_values
            else np.arange(n_times)
        )
        data_vars: dict[str, xr.DataArray] = {}

        for var in graph_info.topological_order:
            if var in set:
                data_vars[var] = _const_time_chain_draw(
                    set[var], n_times, n_chains, n_draws, time_idx
                )
            elif var in graph_info.exogenous:
                data_vars[var] = _const_chain_draw(0.0, n_chains, n_draws)
            elif var in graph_info.endogenous:
                det_key = var if var in stochastic_latent else f"mu_{var}"
                data_vars[var] = _panel_to_time_chain_draw(det[det_key], time_idx)

        return DoResult(ds=xr.Dataset(data_vars))

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
    n_chains = post.sizes["chain"]
    n_draws = post.sizes["draw"]
    time_idx = (
        np.array(scan_info.time_values) if scan_info.time_values else np.arange(n_times)
    )
    predictive_vars: dict[str, xr.DataArray] = {}

    for var in graph_info.topological_order:
        if var in set:
            predictive_vars[var] = _const_time_chain_draw(
                set[var], n_times, n_chains, n_draws, time_idx
            )
        elif var in graph_info.exogenous:
            predictive_vars[var] = _const_chain_draw(0.0, n_chains, n_draws)
        elif var in ppc.posterior_predictive:
            predictive_vars[var] = _panel_to_time_chain_draw(
                ppc.posterior_predictive[var], time_idx
            )
        elif latent_det is not None:
            det_key = var if var in stochastic_latent else f"mu_{var}"
            if det_key in latent_det:
                predictive_vars[var] = _panel_to_time_chain_draw(
                    latent_det[det_key], time_idx
                )

    return DoResult(ds=xr.Dataset(predictive_vars))
