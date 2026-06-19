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
from pathmc.idata import posterior
from pathmc.panel import PanelInfo

__all__ = ["DoResult", "EstimandResult"]


class DoResult:
    """Container for propagated posterior draws under an intervention.

    Supports ``.mean(var)``, ``.hdi(var)``, and contrast arithmetic
    via subtraction (``scenario - baseline``).

    For panel ``do(simulate_over="time")``, the result also stores
    per-time-step draws accessible via :meth:`by_time`.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Posterior draws for each variable, shape ``(n_samples,)``.
    values_by_time : dict[str, np.ndarray] | None
        Per-time-step draws, shape ``(n_times, n_samples)``.
        Available only for panel do() results.
    time_index : array-like | None
        Time labels corresponding to the first axis of *values_by_time*.
    """

    def __init__(
        self,
        values: dict[str, np.ndarray],
        values_by_time: dict[str, np.ndarray] | None = None,
        time_index: np.ndarray | None = None,
    ) -> None:
        self._values = values
        self._values_by_time = values_by_time
        self._time_index = time_index

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
        return self._values[var]

    def mean(self, var: str) -> float:
        """Return the posterior mean of *var* under this intervention."""
        return float(np.mean(self._values[var]))

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
        return compute_hdi(self._values[var], prob=prob)

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
        return self._values_by_time[var]

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        return self._time_index

    def __sub__(self, other: DoResult) -> DoResult:
        """Element-wise contrast between two DoResults."""
        new_values: dict[str, np.ndarray] = {}
        for var in self._values:
            if var in other._values:
                new_values[var] = self._values[var] - other._values[var]

        new_by_time: dict[str, np.ndarray] | None = None
        if self._values_by_time is not None and other._values_by_time is not None:
            new_by_time = {}
            for var in self._values_by_time:
                if var in other._values_by_time:
                    new_by_time[var] = (
                        self._values_by_time[var] - other._values_by_time[var]
                    )

        return DoResult(
            values=new_values,
            values_by_time=new_by_time,
            time_index=self._time_index,
        )

    def __repr__(self) -> str:
        """Tabular summary of the propagated draws for every variable."""
        if not self._values:
            return "DoResult (empty)"
        n_samples = len(next(iter(self._values.values())))
        n_vars = len(self._values)
        name_w = max((len(v) for v in self._values), default=8)
        name_w = max(name_w, 8)
        header = f"DoResult ({n_samples} draws, {n_vars} variables)"
        col_head = f"  {'':<{name_w}}  {'mean':>8}   94% HDI"
        lines = [header, col_head]
        for var, draws in self._values.items():
            mean = float(np.mean(draws))
            lo, hi = compute_hdi(draws)
            lines.append(f"  {var:<{name_w}}  {mean:>8.2f}   [{lo:.2f}, {hi:.2f}]")
        return "\n".join(lines)


class EstimandResult:
    """Posterior draws for a causal estimand (ATE, CATE, ATT, ATU).

    Unlike :class:`DoResult`, which describes the whole system under an
    intervention, an :class:`EstimandResult` knows which outcome was asked
    about, so its accessors default to that variable and no longer require
    the outcome to be re-specified.

    The same per-variable draws are retained, so ``mean(other_var)`` still
    works for any variable in the contrast.

    Parameters
    ----------
    values : dict[str, np.ndarray]
        Contrast draws for each variable, shape ``(n_samples,)``.
    outcome : str
        The outcome variable; the default target of all accessors.
    treatment : str
        The treatment variable that was intervened on.
    estimand : str
        Estimand label, e.g. ``"ATE"``, ``"CATE"``, ``"ATT"``, ``"ATU"``.
    values_by_time : dict[str, np.ndarray] | None
        Per-time-step contrast draws, shape ``(n_times, n_samples)``.
    time_index : array-like | None
        Time labels for the first axis of *values_by_time*.
    """

    def __init__(
        self,
        values: dict[str, np.ndarray],
        outcome: str,
        treatment: str,
        estimand: str,
        values_by_time: dict[str, np.ndarray] | None = None,
        time_index: np.ndarray | None = None,
    ) -> None:
        self._values = values
        self._default_var = outcome
        self._treatment = treatment
        self._estimand = estimand
        self._values_by_time = values_by_time
        self._time_index = time_index

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
            values=contrast._values,
            outcome=outcome,
            treatment=treatment,
            estimand=estimand,
            values_by_time=contrast._values_by_time,
            time_index=contrast._time_index,
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
        return self._default_var if var is None else var

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
        return self._values[self._resolve(var)]

    def mean(self, var: str | None = None) -> float:
        """Return the posterior mean, defaulting to the outcome variable."""
        return float(np.mean(self._values[self._resolve(var)]))

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
        return compute_hdi(self._values[self._resolve(var)], prob=prob)

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
        draws = self._values[self._resolve(var)]
        namespace: dict[str, Any] = {"x": draws, "np": np, "__builtins__": {}}
        try:
            mask = eval(f"x {expr}", namespace)  # noqa: S307
        except SyntaxError as err:
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
        draws = self._values[self._default_var]
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
        return self._values_by_time[self._resolve(var)]

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        return self._time_index

    def __float__(self) -> float:
        """Posterior mean of the estimand (outcome variable)."""
        return self.mean()

    def __sub__(self, other: EstimandResult) -> EstimandResult:
        """Element-wise contrast between two estimands, preserving the outcome."""
        new_values: dict[str, np.ndarray] = {
            var: self._values[var] - other._values[var]
            for var in self._values
            if var in other._values
        }
        return EstimandResult(
            values=new_values,
            outcome=self._default_var,
            treatment=self._treatment,
            estimand=self._estimand,
        )

    def __repr__(self) -> str:
        """Notebook-friendly summary focused on the estimand."""
        draws = self._values[self._default_var]
        mean = float(np.mean(draws))
        lo, hi = compute_hdi(draws)
        p_gt_0 = float(np.mean(draws > 0))
        n_samples = len(draws)
        return (
            f"{self._estimand} of {self._treatment} on {self._default_var}\n"
            f"  Mean:    {mean:.2f}\n"
            f"  94% HDI: [{lo:.2f}, {hi:.2f}]\n"
            f"  P(> 0):  {p_gt_0:.2f}\n"
            f"  Draws:   {n_samples}"
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
                mu_vals = mu_raw.flatten()
                values[var] = _apply_inverse_link(mu_vals, families.get(var, ""))

        return DoResult(values=values)

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
            values[var] = raw.flatten()
        elif f"mu_{var}" in ppc.posterior_predictive:
            raw = ppc.posterior_predictive[f"mu_{var}"].to_numpy()
            if subgroup_indices is not None and raw.ndim >= 3:
                raw = raw[:, :, subgroup_indices]
            values[var] = raw.flatten()
        elif extra_det is not None and f"mu_{var}" in extra_det:
            raw = extra_det[f"mu_{var}"].to_numpy()
            if subgroup_indices is not None and raw.ndim >= 3:
                raw = raw[:, :, subgroup_indices]
            values[var] = raw.flatten()

    return DoResult(values=values)


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
            values=values, values_by_time=values_by_time, time_index=time_idx
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
    return DoResult(values=values, values_by_time=values_by_time, time_index=time_idx)
