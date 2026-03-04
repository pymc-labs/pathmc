"""do() operator: interventional simulation via posterior propagation.

Cross-sectional do() uses PyMC-native graph surgery: ``pm.do()`` on
the generative model + ``pm.sample_posterior_predictive()`` for
kind="predictive", or ``pm.do()`` + ``compute_deterministics`` for
kind="mean".

Panel do() supports three engines:

- **numpy** (default): time-forward NumPy propagation with manual adstock state.
- **batched**: builds a single-timestep pm.Model and applies ``pm.do()``
  + ``compute_deterministics`` per time step in a Python loop.
- **scan**: encodes the full time-forward loop as a ``pytensor.scan``
  inside a pm.Model; a single ``pm.do()`` + ``compute_deterministics``
  evaluates the entire trajectory.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from pathmc.graph import GraphInfo
from pathmc.panel import PanelInfo
from pathmc.parse import Spec, TransformCall
from pathmc.transforms import get_transform


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

    def mean(self, var: str) -> float:
        """Return the posterior mean of *var* under this intervention."""
        return float(np.mean(self._values[var]))

    def hdi(self, var: str, prob: float = 0.94) -> np.ndarray:
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
        return az.hdi(self._values[var], hdi_prob=prob)

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


def run_do_pymc(
    gen_model: pm.Model,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data: pd.DataFrame,
    set: dict[str, float] | None = None,
    kind: str = "mean",
) -> DoResult:
    """Run the do-operator using PyMC-native graph surgery.

    For ``kind="predictive"``: uses ``pm.do()`` on the generative model
    followed by ``pm.sample_posterior_predictive()`` to forward-sample
    through the causal chain with residual noise.

    For ``kind="mean"``: uses ``pm.do()`` with the anonymous tensor trick
    (replacing free endogenous RVs with their mu Deterministics) followed
    by ``compute_deterministics`` for noise-free mean propagation.

    Parameters
    ----------
    gen_model : pm.Model
        The generative PyMC model (endogenous vars are free RVs).
    graph_info : GraphInfo
        DAG with topological order and node classification.
    idata : az.InferenceData
        Posterior samples from ``pm.sample()``.
    data : pd.DataFrame
        Observed data (used for sizing intervention arrays).
    set : dict[str, float] | None
        Variables to intervene on, with their fixed values.
    kind : str
        ``"mean"`` for deterministic propagation, ``"predictive"`` to
        include residual noise at each step.

    Returns
    -------
    DoResult
        Propagated posterior draws for every endogenous variable.
    """
    if set is None:
        set = {}

    N = len(data)
    latent = graph_info.latent

    replacements: dict[str, Any] = {}
    for var, val in set.items():
        key = f"mu_{var}" if var in latent else var
        replacements[key] = np.full(N, val)

    if kind == "mean":
        for var in graph_info.topological_order:
            if var in graph_info.endogenous and var not in set and var not in latent:
                replacements[var] = gen_model[f"mu_{var}"] * 1

        do_model = pm.do(gen_model, replacements)
        det_names = [
            f"mu_{var}"
            for var in graph_info.topological_order
            if var in graph_info.endogenous
        ]
        det = pm.compute_deterministics(
            idata.posterior, model=do_model, var_names=det_names, progressbar=False
        )

        stacked = idata.posterior.stack(sample=("chain", "draw"))
        n_samples = stacked.sizes["sample"]
        values: dict[str, np.ndarray] = {}
        for var in graph_info.topological_order:
            if var in set:
                values[var] = np.full(n_samples, set[var])
            elif var in graph_info.exogenous:
                if var in data.columns:
                    values[var] = np.full(n_samples, float(data[var].mean()))
                else:
                    values[var] = np.zeros(n_samples)
            else:
                mu_vals = det[f"mu_{var}"].values.flatten()
                values[var] = mu_vals

        return DoResult(values=values)

    do_model = pm.do(gen_model, replacements)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Could not extract data from symbolic observation"
        )
        with do_model:
            ppc = pm.sample_posterior_predictive(idata, progressbar=False)

    # Latent vars are pm.Deterministic nodes — PPC won't forward-sample them.
    # Compute them explicitly so they appear in the result.
    latent_det_names = [
        f"mu_{var}"
        for var in graph_info.topological_order
        if var in latent and var not in set
    ]
    if latent_det_names:
        latent_det = pm.compute_deterministics(
            idata.posterior,
            model=do_model,
            var_names=latent_det_names,
            progressbar=False,
        )
    else:
        latent_det = None

    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_samples = stacked.sizes["sample"]
    values = {}
    for var in graph_info.topological_order:
        if var in set:
            values[var] = np.full(n_samples, set[var])
        elif var in graph_info.exogenous:
            if var in data.columns:
                values[var] = np.full(n_samples, float(data[var].mean()))
            else:
                values[var] = np.zeros(n_samples)
        elif var in ppc.posterior_predictive:
            values[var] = ppc.posterior_predictive[var].values.flatten()
        elif f"mu_{var}" in ppc.posterior_predictive:
            values[var] = ppc.posterior_predictive[f"mu_{var}"].values.flatten()
        elif latent_det is not None and f"mu_{var}" in latent_det:
            values[var] = latent_det[f"mu_{var}"].values.flatten()

    return DoResult(values=values)


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable inverse-logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _apply_link(linear: np.ndarray, family: str) -> np.ndarray:
    """Apply the inverse link function for mean propagation."""
    if family == "bernoulli":
        return _expit(linear)
    if family in ("poisson", "negbinomial"):
        return np.exp(linear)
    return linear


def _add_residual_noise(
    linear: np.ndarray,
    var: str,
    family: str,
    stacked: object,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from the residual distribution for predictive propagation."""
    if family == "bernoulli":
        probs = _expit(linear)
        return rng.binomial(1, probs).astype(float)

    if family == "poisson":
        mu = np.exp(np.clip(linear, -20, 20))
        return rng.poisson(mu).astype(float)

    if family == "negbinomial":
        mu = np.exp(np.clip(linear, -20, 20))
        alpha = stacked[f"alpha_disp_{var}"].values
        p = alpha / (alpha + mu)
        return rng.negative_binomial(alpha.astype(int).clip(1), p).astype(float)

    if family == "studentt":
        sigma_arr = stacked[f"sigma_{var}"].values
        nu_arr = stacked[f"nu_{var}"].values
        return linear + rng.standard_t(df=np.clip(nu_arr, 2, 1000)) * sigma_arr

    sigma_arr = stacked[f"sigma_{var}"].values
    return linear + rng.normal(0, sigma_arr)


def _build_transform_map(spec: Spec) -> dict[str, TransformCall]:
    """Map variable names to their TransformCall for all transform terms."""
    tmap: dict[str, TransformCall] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            if term.transform is not None:
                tmap[term.variable] = term.transform
    return tmap


def _resolve_set_value(val: float | np.ndarray, t_idx: int) -> float:
    """Return the intervention value at time step *t_idx*.

    Scalars are returned unchanged; arrays are indexed.
    """
    if isinstance(val, np.ndarray):
        return float(val[t_idx])
    return float(val)


def run_panel_do(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data: pd.DataFrame,
    design_columns: dict[str, list[str]],
    panel_info: PanelInfo,
    set: dict[str, float | np.ndarray] | None = None,
    families: dict[str, str] | None = None,
    kind: str = "mean",
    init_from: str = "observed",
    rng: np.random.Generator | None = None,
) -> DoResult:
    """Time-forward panel do() — propagate interventions through time.

    Iterates through time steps within each unit, using simulated values
    for lagged dependencies at each step.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    graph_info : GraphInfo
        DAG with topological order.
    idata : az.InferenceData
        Posterior samples.
    data : pd.DataFrame
        Observed panel data.
    design_columns : dict[str, list[str]]
        Column names of each design matrix.
    panel_info : PanelInfo
        Panel metadata.
    set : dict[str, float | np.ndarray] | None
        Variables to fix at specific values. Array values of shape
        ``(n_times,)`` specify per-time-step interventions.
    families : dict[str, str] | None
        Per-variable distribution families.
    kind : str
        ``"mean"`` or ``"predictive"``.
    init_from : str
        ``"observed"`` to use observed data for initial conditions.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    DoResult
        Propagated draws with per-time-step data in ``by_time()``.
    """

    if set is None:
        set = {}
    if families is None:
        families = {}
    if rng is None:
        rng = np.random.default_rng()

    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_samples = stacked.sizes["sample"]

    transform_map = _build_transform_map(spec)

    unit_col = panel_info.unit
    time_col = panel_info.time

    data_sorted = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    units = panel_info.unit_labels
    time_values = sorted(data_sorted[time_col].unique())
    n_units = len(units)
    n_times = len(time_values)

    has_alpha = f"alpha_{graph_info.topological_order[-1]}" in stacked

    all_vars = dict.fromkeys(graph_info.topological_order)
    for intervened_var in set:
        all_vars[intervened_var] = None

    all_values: dict[str, np.ndarray] = {
        var: np.zeros((n_units, n_times, n_samples)) for var in all_vars
    }

    adstock_state: dict[str, np.ndarray] = {}

    for u_idx, unit in enumerate(units):
        unit_mask = data_sorted[unit_col] == unit
        unit_data = data_sorted[unit_mask].sort_values(time_col).reset_index(drop=True)

        for col_name in transform_map:
            adstock_state[col_name] = np.zeros(n_samples)

        for t_idx, _time_val in enumerate(time_values):
            for intervened_var, intervened_val in set.items():
                if intervened_var not in graph_info.topological_order:
                    all_values[intervened_var][u_idx, t_idx, :] = _resolve_set_value(
                        intervened_val, t_idx
                    )

            for var in graph_info.topological_order:
                if var in set:
                    all_values[var][u_idx, t_idx, :] = _resolve_set_value(
                        set[var], t_idx
                    )
                elif var in graph_info.exogenous:
                    is_lag = _parse_lag(var)
                    if is_lag is not None:
                        base_var, lag_k = is_lag
                        src_t = t_idx - lag_k
                        if src_t >= 0 and base_var in all_values:
                            all_values[var][u_idx, t_idx, :] = all_values[base_var][
                                u_idx, src_t, :
                            ]
                        elif t_idx < len(unit_data) and init_from == "observed":
                            all_values[var][u_idx, t_idx, :] = float(
                                unit_data.iloc[t_idx].get(var, 0.0)
                            )
                        else:
                            all_values[var][u_idx, t_idx, :] = 0.0
                    elif t_idx < len(unit_data):
                        all_values[var][u_idx, t_idx, :] = float(
                            unit_data.iloc[t_idx].get(var, 0.0)
                        )
                    else:
                        all_values[var][u_idx, t_idx, :] = 0.0
                else:
                    beta_arr = stacked[f"beta_{var}"]
                    cols = design_columns[var]
                    linear = np.zeros(n_samples)

                    for col in cols:
                        coef = beta_arr.sel({f"{var}_predictors": col}).values
                        if col == "Intercept":
                            linear = linear + coef
                        elif col in transform_map:
                            parent_val = _get_panel_col_value(
                                col,
                                u_idx,
                                t_idx,
                                all_values,
                                set,
                                unit_data,
                                init_from,
                                n_samples,
                            )
                            transformed = _apply_panel_transform(
                                transform_map[col],
                                parent_val,
                                adstock_state,
                                col,
                                stacked,
                            )
                            linear = linear + coef * transformed
                        else:
                            parent_val = _get_panel_col_value(
                                col,
                                u_idx,
                                t_idx,
                                all_values,
                                set,
                                unit_data,
                                init_from,
                                n_samples,
                            )
                            linear = linear + coef * parent_val

                    if has_alpha and f"alpha_{var}" in stacked:
                        alpha_arr = stacked[f"alpha_{var}"]
                        alpha_unit = alpha_arr.sel(unit=unit).values
                        linear = linear + alpha_unit

                    for col in cols:
                        if col == "Intercept":
                            continue
                        slope_name = f"slope_{var}_{col}"
                        if slope_name in stacked:
                            slope_arr = stacked[slope_name]
                            slope_unit = slope_arr.sel(unit=unit).values
                            parent_val = _get_panel_col_value(
                                col,
                                u_idx,
                                t_idx,
                                all_values,
                                set,
                                unit_data,
                                init_from,
                                n_samples,
                            )
                            linear = linear + slope_unit * parent_val

                    family = families.get(var, "gaussian")
                    is_latent = var in graph_info.latent
                    if kind == "predictive" and not is_latent:
                        all_values[var][u_idx, t_idx, :] = _add_residual_noise(
                            linear, var, family, stacked, n_samples, rng
                        )
                    else:
                        all_values[var][u_idx, t_idx, :] = _apply_link(linear, family)

    result_values: dict[str, np.ndarray] = {}
    by_time: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        result_values[var] = all_values[var].mean(axis=(0, 1))
        by_time[var] = all_values[var].mean(axis=0)  # (n_times, n_samples)

    return DoResult(
        values=result_values,
        values_by_time=by_time,
        time_index=np.array(time_values),
    )


def _get_panel_col_value(
    col: str,
    u_idx: int,
    t_idx: int,
    all_values: dict[str, np.ndarray],
    set_dict: dict[str, float | np.ndarray],
    unit_data: pd.DataFrame,
    init_from: str,
    n_samples: int,
) -> np.ndarray:
    """Resolve a column value for the panel do() inner loop."""
    is_lag = _parse_lag(col)
    if is_lag is not None:
        base_var, lag_k = is_lag
        src_t = t_idx - lag_k
        if src_t >= 0 and base_var in all_values:
            return all_values[base_var][u_idx, src_t, :]
        if init_from == "observed" and t_idx < len(unit_data):
            return np.full(n_samples, float(unit_data.iloc[t_idx].get(col, 0.0)))
        return np.zeros(n_samples)

    if col in all_values:
        return all_values[col][u_idx, t_idx, :]
    if col in set_dict:
        return np.full(n_samples, _resolve_set_value(set_dict[col], t_idx))
    if t_idx < len(unit_data):
        return np.full(n_samples, float(unit_data.iloc[t_idx].get(col, 0.0)))
    return np.zeros(n_samples)


def _apply_panel_transform(
    tc: TransformCall,
    input_val: np.ndarray,
    adstock_state: dict[str, np.ndarray],
    col_key: str,
    stacked: object,
) -> np.ndarray:
    """Apply transform in panel do(), tracking adstock state across time steps."""
    if isinstance(tc.input_expr, TransformCall):
        input_val = _apply_panel_transform(
            tc.input_expr, input_val, adstock_state, col_key + "_inner", stacked
        )

    transform = get_transform(tc.name)
    params = {key: stacked[name].values for key, name in tc.params.items()}

    if transform.name == "adstock":
        decay = params["decay"]
        prev = adstock_state.get(col_key, np.zeros_like(input_val))
        result = input_val + decay * prev
        adstock_state[col_key] = result
        return result

    return transform.apply_numpy(input_val, params)


def _parse_lag(col_name: str) -> tuple[str, int] | None:
    """Parse a lag column name like 'sales_lag1' -> ('sales', 1)."""
    m = re.match(r"^(.+)_lag(\d+)$", col_name)
    if m:
        return m.group(1), int(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Shared infrastructure for batched / scan engines
# ---------------------------------------------------------------------------


def _has_random_intercepts(pooling: str | dict | None) -> bool:
    """Whether pooling spec requests random intercepts."""
    if pooling == "partial":
        return True
    if isinstance(pooling, dict):
        return pooling.get("intercept", False)
    return False


def _get_slope_vars(pooling: str | dict | None) -> list[str]:
    """Extract variables that should get random slopes."""
    if isinstance(pooling, dict):
        return list(pooling.get("slopes", []))
    return []


def _has_adstock(tc: TransformCall) -> bool:
    """Return True if the transform chain contains an adstock step."""
    if tc.name == "adstock":
        return True
    if isinstance(tc.input_expr, TransformCall):
        return _has_adstock(tc.input_expr)
    return False


def _get_adstock_input(tc: TransformCall) -> str:
    """Return the leaf input variable of a transform chain."""
    if isinstance(tc.input_expr, TransformCall):
        return _get_adstock_input(tc.input_expr)
    return tc.input_expr


def _prepare_panel_data(
    data: pd.DataFrame,
    panel_info: PanelInfo,
) -> tuple[int, int, list, list, pd.DataFrame]:
    """Sort panel data and return (n_units, n_times, units, time_values, sorted_df)."""
    unit_col = panel_info.unit
    time_col = panel_info.time
    data_sorted = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    units = panel_info.unit_labels
    time_values = sorted(data_sorted[time_col].unique())
    return len(units), len(time_values), units, time_values, data_sorted


def _reshape_to_panel(
    data_sorted: pd.DataFrame,
    column: str,
    n_units: int,
    n_times: int,
) -> np.ndarray:
    """Reshape a column from flat sorted data to (n_times, n_units)."""
    return data_sorted[column].values.reshape(n_units, n_times).T


def _apply_single_step_transform_pt(
    tc: TransformCall,
    input_tensor: Any,
    adstock_state: Any,
    param_rvs: dict[str, Any],
) -> tuple[Any, Any]:
    """Apply one time step of a (possibly nested) transform chain in PyTensor.

    Returns (transformed_value, updated_adstock_state).
    For non-adstock transforms, adstock_state is returned unchanged.
    """
    import pytensor.tensor as pt

    if isinstance(tc.input_expr, TransformCall):
        input_tensor, adstock_state = _apply_single_step_transform_pt(
            tc.input_expr, input_tensor, adstock_state, param_rvs
        )

    if tc.name == "adstock":
        decay = param_rvs[tc.params["decay"]]
        result = input_tensor + decay * adstock_state
        return result, result

    if tc.name == "logistic_saturation":
        lam = param_rvs[tc.params["lam"]]
        return 1.0 - pt.exp(-lam * input_tensor), adstock_state

    transform = get_transform(tc.name)
    params = {key: param_rvs[name] for key, name in tc.params.items()}
    return transform.apply_pymc(input_tensor, params), adstock_state


# ---------------------------------------------------------------------------
# Panel do() engine: batched pm.do() (one step model, Python time loop)
# ---------------------------------------------------------------------------


def _build_step_model(
    spec: Spec,
    graph_info: GraphInfo,
    design_columns: dict[str, list[str]],
    families: dict[str, str],
    n_units: int,
    unit_labels: list,
    pooling: str | dict | None,
    latent: set[str],
) -> pm.Model:
    """Build a pm.Model that computes one time step for all units.

    Parameter RVs are named identically to the estimation model so that
    ``compute_deterministics`` can pull values from the posterior by name.
    """
    import pytensor.tensor as pt

    from pathmc.compile import get_predictor_columns

    transform_map = _build_transform_map(spec)
    has_ri = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)
    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        coords[f"{reg.lhs}_predictors"] = get_predictor_columns(reg)
    if has_ri:
        coords["unit"] = unit_labels

    with pm.Model(coords=coords) as step_model:
        # --- transform parameter priors (same names as estimation model) ---
        tparam_rvs: dict[str, Any] = {}
        for reg in spec.regressions:
            for term in reg.terms:
                if term.transform is not None:
                    _emit_tc_priors(term.transform, tparam_rvs)

        # --- Data nodes for exogenous inputs at this time step ---
        data_nodes: dict[str, Any] = {}
        for var in graph_info.topological_order:
            if var in graph_info.exogenous:
                data_nodes[var] = pm.Data(var, np.zeros(n_units))

        # --- Data nodes for adstock carry-over state ---
        adstock_data: dict[str, Any] = {}
        for col, tc in transform_map.items():
            if _has_adstock(tc):
                adstock_data[col] = pm.Data(f"_adstock_state_{col}", np.zeros(n_units))

        # --- compile each endogenous variable ---
        endogenous_rvs: dict[str, Any] = {}
        for var in endogenous_order:
            reg = reg_by_lhs[var]
            family = families.get(var, "gaussian")
            cols = design_columns[var]

            beta = pm.Normal(f"beta_{var}", mu=0, sigma=10, dims=f"{var}_predictors")

            mu = pt.zeros(n_units)
            adstock_updates: dict[str, Any] = {}

            for i, col in enumerate(cols):
                coef = beta[i]
                if col == "Intercept":
                    mu = mu + coef
                elif col in transform_map:
                    tc = transform_map[col]
                    inp = _get_adstock_input(tc)
                    raw_input = (
                        endogenous_rvs[inp]
                        if inp in endogenous_rvs
                        else data_nodes.get(inp, pt.zeros(n_units))
                    )
                    prev_state = adstock_data.get(col, pt.zeros(n_units))
                    transformed, new_state = _apply_single_step_transform_pt(
                        tc, raw_input, prev_state, tparam_rvs
                    )
                    adstock_updates[col] = new_state
                    mu = mu + coef * transformed
                elif col in endogenous_rvs:
                    mu = mu + coef * endogenous_rvs[col]
                elif col in data_nodes:
                    mu = mu + coef * data_nodes[col]

            if has_ri:
                mu_a = pm.Normal(f"mu_alpha_{var}", mu=0, sigma=10)
                sig_a = pm.HalfNormal(f"sigma_alpha_{var}", sigma=1)
                alpha = pm.Normal(f"alpha_{var}", mu=mu_a, sigma=sig_a, dims="unit")
                mu = mu + alpha

            term_variables = {t.variable for t in reg.terms}
            for svar in slope_vars:
                if svar not in term_variables:
                    continue
                mu_s = pm.Normal(f"mu_slope_{var}_{svar}", mu=0, sigma=10)
                sig_s = pm.HalfNormal(f"sigma_slope_{var}_{svar}", sigma=1)
                slope = pm.Normal(
                    f"slope_{var}_{svar}", mu=mu_s, sigma=sig_s, dims="unit"
                )
                x_node = data_nodes.get(svar, endogenous_rvs.get(svar))
                if x_node is not None:
                    mu = mu + slope * x_node

            for col, state in adstock_updates.items():
                pm.Deterministic(f"_adstock_out_{col}", state)

            mu_det = pm.Deterministic(f"mu_{var}", mu)
            rv = _emit_step_rv(var, mu_det, family, latent)
            endogenous_rvs[var] = rv

    return step_model


def _emit_tc_priors(tc: TransformCall, emitted: dict[str, Any]) -> None:
    """Recursively emit priors for transform params (same names as estimation)."""
    if isinstance(tc.input_expr, TransformCall):
        _emit_tc_priors(tc.input_expr, emitted)
    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in emitted:
            pspec = transform.param_specs[param_key]
            emitted[param_name] = transform.emit_prior(param_name, pspec)


def _emit_step_rv(var: str, mu: Any, family: str, latent: set[str]) -> Any:
    """Emit a free RV for an endogenous variable in the step model."""
    if var in latent:
        return mu
    if family == "bernoulli":
        return pm.Bernoulli(var, logit_p=mu)
    if family == "poisson":
        return pm.Poisson(var, mu=pm.math.exp(mu))
    if family == "negbinomial":
        alpha_disp = pm.HalfNormal(f"alpha_disp_{var}", sigma=1)
        return pm.NegativeBinomial(var, mu=pm.math.exp(mu), alpha=alpha_disp)
    if family == "studentt":
        sigma = pm.HalfNormal(f"sigma_{var}", sigma=1)
        nu = pm.Gamma(f"nu_{var}", alpha=2, beta=0.1)
        return pm.StudentT(var, nu=nu, mu=mu, sigma=sigma)
    sigma = pm.HalfNormal(f"sigma_{var}", sigma=1)
    return pm.Normal(var, mu=mu, sigma=sigma)


def run_panel_do_batched(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data: pd.DataFrame,
    design_columns: dict[str, list[str]],
    panel_info: PanelInfo,
    set: dict[str, float | np.ndarray] | None = None,
    families: dict[str, str] | None = None,
    kind: str = "mean",
    init_from: str = "observed",
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    rng: np.random.Generator | None = None,
) -> DoResult:
    """Time-forward panel do() using batched pm.do() per time step.

    Builds a single-timestep pm.Model and applies ``pm.do()`` +
    ``compute_deterministics`` at each time step. Lagged dependencies
    and adstock state are resolved from simulated values at previous steps.

    .. note::

       ``pm.Data`` nodes are shared across posterior draws in
       ``compute_deterministics``, so temporal carry-over state (lag
       values, adstock accumulation) is averaged over draws before
       being passed to the next time step.  This is a mean-field
       approximation.  For well-identified models the error is
       negligible, but for precision-sensitive work the **scan** engine
       (per-draw correct via ``pytensor.scan``) is preferred.
    """
    if set is None:
        set = {}
    if families is None:
        families = {}
    if latent is None:
        latent = set()
    if rng is None:
        rng = np.random.default_rng()

    n_units, n_times, units, time_values, data_sorted = _prepare_panel_data(
        data, panel_info
    )
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_samples = stacked.sizes["sample"]

    transform_map = _build_transform_map(spec)
    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    step_model = _build_step_model(
        spec,
        graph_info,
        design_columns,
        families,
        n_units,
        units,
        pooling,
        latent,
    )

    # --- exogenous data matrices (n_times, n_units) ---
    exog_matrices: dict[str, np.ndarray] = {}
    for var in graph_info.exogenous:
        if var in data.columns:
            exog_matrices[var] = _reshape_to_panel(data_sorted, var, n_units, n_times)

    # --- storage ---
    all_values: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        all_values[var] = np.zeros((n_units, n_times, n_samples))
    for iv in set:
        if iv not in all_values:
            all_values[iv] = np.zeros((n_units, n_times, n_samples))

    adstock_state_np: dict[str, np.ndarray] = {
        col: np.zeros((n_units, n_samples))
        for col, tc in transform_map.items()
        if _has_adstock(tc)
    }

    det_names = [f"mu_{v}" for v in endogenous_order]
    adstock_det_names = [
        f"_adstock_out_{col}" for col, tc in transform_map.items() if _has_adstock(tc)
    ]

    for t_idx in range(n_times):
        replacements: dict[str, Any] = {}

        # Exogenous values (data or intervention)
        for var in graph_info.exogenous:
            if var in set:
                v = _resolve_set_value(set[var], t_idx)
                replacements[var] = np.full(n_units, v)
                all_values[var][:, t_idx, :] = v
            else:
                lag = _parse_lag(var)
                if lag is not None:
                    base_var, lag_k = lag
                    src_t = t_idx - lag_k
                    if src_t >= 0 and base_var in all_values:
                        val = all_values[base_var][:, src_t, :].mean(axis=1)
                    elif init_from == "observed" and var in exog_matrices:
                        val = exog_matrices[var][t_idx]
                    else:
                        val = np.zeros(n_units)
                    replacements[var] = val.astype(float)
                elif var in exog_matrices:
                    replacements[var] = exog_matrices[var][t_idx]
                else:
                    replacements[var] = np.zeros(n_units)

        # Adstock carry state
        for col in adstock_state_np:
            replacements[f"_adstock_state_{col}"] = (
                adstock_state_np[col].mean(axis=1).astype(float)
            )

        # Intervened endogenous
        for iv, iv_val in set.items():
            if iv in graph_info.endogenous:
                all_values[iv][:, t_idx, :] = _resolve_set_value(iv_val, t_idx)

        # Replace endogenous RVs with their mu deterministics so that
        # compute_deterministics only needs parameters that exist in
        # the posterior (endogenous RVs were observed during estimation).
        for var in endogenous_order:
            if var not in set and var not in latent:
                replacements[var] = step_model[f"mu_{var}"] * 1

        do_model = pm.do(step_model, replacements)
        all_det_names = det_names + adstock_det_names
        det = pm.compute_deterministics(
            idata.posterior,
            model=do_model,
            var_names=all_det_names,
            progressbar=False,
        )

        for var in endogenous_order:
            if var in set:
                continue
            mu_vals = det[f"mu_{var}"].values
            flat = mu_vals.reshape(-1, n_units)  # (n_samples, n_units)

            if kind == "predictive" and var not in latent:
                family = families.get(var, "gaussian")
                for u_idx in range(n_units):
                    all_values[var][u_idx, t_idx, :] = _add_residual_noise(
                        flat[:, u_idx], var, family, stacked, n_samples, rng
                    )
            else:
                for u_idx in range(n_units):
                    all_values[var][u_idx, t_idx, :] = flat[:, u_idx]

        # Update adstock state from deterministics
        for col in adstock_state_np:
            det_key = f"_adstock_out_{col}"
            if det_key in det:
                vals = det[det_key].values.reshape(-1, n_units)
                adstock_state_np[col] = vals.T  # (n_units, n_samples)

    result_values: dict[str, np.ndarray] = {}
    by_time: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        result_values[var] = all_values[var].mean(axis=(0, 1))
        by_time[var] = all_values[var].mean(axis=0)  # (n_times, n_samples)

    return DoResult(
        values=result_values,
        values_by_time=by_time,
        time_index=np.array(time_values),
    )


# ---------------------------------------------------------------------------
# Panel do() engine: pytensor.scan
# ---------------------------------------------------------------------------


def _build_scan_model(
    spec: Spec,
    graph_info: GraphInfo,
    design_columns: dict[str, list[str]],
    families: dict[str, str],
    panel_info: PanelInfo,
    data_sorted: pd.DataFrame,
    n_units: int,
    n_times: int,
    units: list,
    pooling: str | dict | None,
    latent: set[str],
    set_dict: dict[str, float | np.ndarray],
    init_from: str,
) -> pm.Model:
    """Build a pm.Model with pytensor.scan for panel time-forward simulation.

    The scan step function computes one time step for all units and all
    endogenous variables. Parameter RVs are named identically to the
    estimation model for posterior matching.
    """
    import pytensor
    import pytensor.tensor as pt

    from pathmc.compile import get_predictor_columns

    transform_map = _build_transform_map(spec)
    has_ri = _has_random_intercepts(pooling)
    slope_vars = _get_slope_vars(pooling)
    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    # Identify exogenous variables that are NOT lags
    pure_exog = [
        v
        for v in graph_info.topological_order
        if v in graph_info.exogenous
        and _parse_lag(v) is None
        and v in data_sorted.columns
    ]
    # Identify lag columns
    lag_cols: dict[str, tuple[str, int]] = {}
    for v in graph_info.topological_order:
        if v in graph_info.exogenous:
            parsed = _parse_lag(v)
            if parsed is not None:
                lag_cols[v] = parsed

    # Identify adstock columns
    adstock_cols = [col for col, tc in transform_map.items() if _has_adstock(tc)]

    coords: dict[str, Any] = {}
    for reg in spec.regressions:
        coords[f"{reg.lhs}_predictors"] = get_predictor_columns(reg)
    if has_ri:
        coords["unit"] = units

    with pm.Model(coords=coords) as scan_model:
        # --- transform parameter priors ---
        tparam_rvs: dict[str, Any] = {}
        for reg in spec.regressions:
            for term in reg.terms:
                if term.transform is not None:
                    _emit_tc_priors(term.transform, tparam_rvs)

        # --- regression parameter priors ---
        beta_rvs: dict[str, Any] = {}
        sigma_rvs: dict[str, Any] = {}
        for var in endogenous_order:
            family = families.get(var, "gaussian")
            beta_rvs[var] = pm.Normal(
                f"beta_{var}", mu=0, sigma=10, dims=f"{var}_predictors"
            )
            if var not in latent:
                if family in ("gaussian", "studentt"):
                    sigma_rvs[var] = pm.HalfNormal(f"sigma_{var}", sigma=1)
                if family == "studentt":
                    pm.Gamma(f"nu_{var}", alpha=2, beta=0.1)
                if family == "negbinomial":
                    pm.HalfNormal(f"alpha_disp_{var}", sigma=1)

        # --- random effects ---
        alpha_rvs: dict[str, Any] = {}
        slope_rvs: dict[str, dict[str, Any]] = {}
        if has_ri:
            for var in endogenous_order:
                mu_a = pm.Normal(f"mu_alpha_{var}", mu=0, sigma=10)
                sig_a = pm.HalfNormal(f"sigma_alpha_{var}", sigma=1)
                alpha_rvs[var] = pm.Normal(
                    f"alpha_{var}", mu=mu_a, sigma=sig_a, dims="unit"
                )
        for var in endogenous_order:
            reg = reg_by_lhs[var]
            term_variables = {t.variable for t in reg.terms}
            slope_rvs[var] = {}
            for svar in slope_vars:
                if svar in term_variables:
                    mu_s = pm.Normal(f"mu_slope_{var}_{svar}", mu=0, sigma=10)
                    sig_s = pm.HalfNormal(f"sigma_slope_{var}_{svar}", sigma=1)
                    slope_rvs[var][svar] = pm.Normal(
                        f"slope_{var}_{svar}", mu=mu_s, sigma=sig_s, dims="unit"
                    )

        # --- exogenous data as pm.Data (n_times, n_units) ---
        exog_data_nodes: dict[str, Any] = {}
        for var in pure_exog:
            if var in set_dict:
                v = set_dict[var]
                if isinstance(v, np.ndarray):
                    mat = np.broadcast_to(v[:, None], (n_times, n_units)).copy()
                else:
                    mat = np.full((n_times, n_units), v)
            else:
                mat = _reshape_to_panel(data_sorted, var, n_units, n_times)
            exog_data_nodes[var] = pm.Data(f"scan_{var}", mat.astype(float))

        # --- define scan ---
        # Carry: one tensor per endogenous var + one per adstock column
        # Sequences: one tensor per pure exogenous var (n_times, n_units)

        # Initial values for carry — use observed lag data when available
        init_endo: dict[str, np.ndarray] = {}
        for var in endogenous_order:
            init_endo[var] = np.zeros(n_units)
        if init_from == "observed":
            for lag_col, (base_var, _lag_k) in lag_cols.items():
                if base_var in init_endo and lag_col in data_sorted.columns:
                    mat = _reshape_to_panel(data_sorted, lag_col, n_units, n_times)
                    init_endo[base_var] = mat[0].astype("float64")

        init_adstock: dict[str, np.ndarray] = {}
        for col in adstock_cols:
            init_adstock[col] = np.zeros(n_units)

        # Build ordered lists for scan signature
        endo_keys = list(endogenous_order)
        adstock_keys = sorted(adstock_cols)
        exog_keys = sorted(exog_data_nodes.keys())

        sequences = [exog_data_nodes[k] for k in exog_keys]

        outputs_info = [
            pt.as_tensor_variable(init_endo[k].astype("float64")) for k in endo_keys
        ] + [
            pt.as_tensor_variable(init_adstock[k].astype("float64"))
            for k in adstock_keys
        ]

        # Non-sequences: all parameters
        non_seq_list = []
        non_seq_names = []
        for var in endo_keys:
            non_seq_list.append(beta_rvs[var])
            non_seq_names.append(f"beta_{var}")
        for name, rv in tparam_rvs.items():
            non_seq_list.append(rv)
            non_seq_names.append(name)
        for var in endo_keys:
            if var in alpha_rvs:
                non_seq_list.append(alpha_rvs[var])
                non_seq_names.append(f"alpha_{var}")
        for var in endo_keys:
            for svar, srv in slope_rvs.get(var, {}).items():
                non_seq_list.append(srv)
                non_seq_names.append(f"slope_{var}_{svar}")

        n_seq = len(sequences)
        n_carry = len(endo_keys) + len(adstock_keys)

        def step_fn(*args: Any) -> list[Any]:
            seq_args = args[:n_seq]
            carry_args = args[n_seq : n_seq + n_carry]
            ns_args = args[n_seq + n_carry :]

            exog_t = {k: seq_args[i] for i, k in enumerate(exog_keys)}
            prev_endo = {k: carry_args[i] for i, k in enumerate(endo_keys)}
            prev_adstock = {
                k: carry_args[len(endo_keys) + i] for i, k in enumerate(adstock_keys)
            }

            ns_map: dict[str, Any] = {}
            for i, name in enumerate(non_seq_names):
                ns_map[name] = ns_args[i]

            new_endo: dict[str, Any] = {}
            new_adstock = dict(prev_adstock)

            for var in endo_keys:
                cols = design_columns[var]
                beta = ns_map[f"beta_{var}"]
                mu = pt.zeros(n_units)

                for ci, col in enumerate(cols):
                    coef = beta[ci]
                    if col == "Intercept":
                        mu = mu + coef
                    elif col in transform_map:
                        tc = transform_map[col]
                        inp_name = _get_adstock_input(tc)
                        if inp_name in new_endo:
                            raw = new_endo[inp_name]
                        elif inp_name in exog_t:
                            raw = exog_t[inp_name]
                        else:
                            raw = pt.zeros(n_units)
                        prev_st = prev_adstock.get(col, pt.zeros(n_units))
                        transformed, new_st = _apply_single_step_transform_pt(
                            tc, raw, prev_st, ns_map
                        )
                        new_adstock[col] = new_st
                        mu = mu + coef * transformed
                    elif col in new_endo:
                        mu = mu + coef * new_endo[col]
                    elif col in exog_t:
                        mu = mu + coef * exog_t[col]
                    else:
                        lag = _parse_lag(col)
                        if lag is not None:
                            base_var, _lag_k = lag
                            if base_var in prev_endo:
                                mu = mu + coef * prev_endo[base_var]

                if f"alpha_{var}" in ns_map:
                    mu = mu + ns_map[f"alpha_{var}"]

                for svar in slope_vars:
                    skey = f"slope_{var}_{svar}"
                    if skey in ns_map:
                        if svar in exog_t:
                            x_val = exog_t[svar]
                        elif svar in new_endo:
                            x_val = new_endo[svar]
                        else:
                            x_val = pt.zeros(n_units)
                        mu = mu + ns_map[skey] * x_val

                family = families.get(var, "gaussian")
                if var in latent:
                    new_endo[var] = mu
                elif family == "bernoulli":
                    new_endo[var] = 1.0 / (1.0 + pt.exp(-mu))
                elif family in ("poisson", "negbinomial"):
                    new_endo[var] = pt.exp(pt.clip(mu, -20, 20))
                else:
                    new_endo[var] = mu

            out = [new_endo[k] for k in endo_keys]
            out += [new_adstock[k] for k in adstock_keys]
            return out

        results, _updates = pytensor.scan(
            fn=step_fn,
            sequences=sequences,
            outputs_info=outputs_info,
            non_sequences=non_seq_list,
            strict=True,
        )

        if not isinstance(results, list):
            results = [results]

        for i, var in enumerate(endo_keys):
            pm.Deterministic(f"mu_{var}_scan", results[i])

    return scan_model


def run_panel_do_scan(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data: pd.DataFrame,
    design_columns: dict[str, list[str]],
    panel_info: PanelInfo,
    set: dict[str, float | np.ndarray] | None = None,
    families: dict[str, str] | None = None,
    kind: str = "mean",
    init_from: str = "observed",
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    rng: np.random.Generator | None = None,
) -> DoResult:
    """Time-forward panel do() using pytensor.scan.

    Builds a pm.Model encoding the full time-forward loop as a
    ``pytensor.scan``.  A single ``pm.do()`` replaces exogenous data
    with intervention values, then ``compute_deterministics`` evaluates
    the trajectory using posterior parameter draws.

    For ``kind="predictive"``, post-hoc noise is added independently
    at each (unit, time) point after the mean trajectory is computed.
    """
    if set is None:
        set = {}
    if families is None:
        families = {}
    if latent is None:
        latent = set()
    if rng is None:
        rng = np.random.default_rng()

    n_units, n_times, units, time_values, data_sorted = _prepare_panel_data(
        data, panel_info
    )
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_samples = stacked.sizes["sample"]

    endogenous_order = [
        v for v in graph_info.topological_order if v in graph_info.endogenous
    ]

    scan_model = _build_scan_model(
        spec,
        graph_info,
        design_columns,
        families,
        panel_info,
        data_sorted,
        n_units,
        n_times,
        units,
        pooling,
        latent,
        set,
        init_from,
    )

    # Apply pm.do() for interventions on exogenous data
    replacements: dict[str, Any] = {}
    for var, val in set.items():
        scan_key = f"scan_{var}"
        if scan_key in {v.name for v in scan_model.data_vars}:
            if isinstance(val, np.ndarray):
                replacements[scan_key] = np.broadcast_to(
                    val[:, None], (n_times, n_units)
                ).copy()
            else:
                replacements[scan_key] = np.full((n_times, n_units), val)

    if replacements:
        do_model = pm.do(scan_model, replacements)
    else:
        do_model = scan_model

    det_names = [f"mu_{v}_scan" for v in endogenous_order]
    det = pm.compute_deterministics(
        idata.posterior, model=do_model, var_names=det_names, progressbar=False
    )

    # Extract results — scan output shape is (chain, draw, n_times, n_units)
    all_values: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        all_values[var] = np.zeros((n_units, n_times, n_samples))

    for var in graph_info.topological_order:
        if var in set:
            v = set[var]
            if isinstance(v, np.ndarray):
                for t in range(n_times):
                    all_values[var][:, t, :] = v[t]
            else:
                all_values[var][:, :, :] = v
        elif var in graph_info.exogenous:
            if var in data.columns:
                mat = _reshape_to_panel(data_sorted, var, n_units, n_times)
                for u in range(n_units):
                    for t in range(n_times):
                        all_values[var][u, t, :] = mat[t, u]
            else:
                all_values[var][:, :, :] = 0.0
        else:
            det_key = f"mu_{var}_scan"
            vals = det[det_key].values  # (chain, draw, n_times, n_units)
            flat = vals.reshape(-1, n_times, n_units)  # (n_samples, T, U)
            for u in range(n_units):
                for t in range(n_times):
                    mu_slice = flat[:, t, u]
                    if kind == "predictive" and var not in latent:
                        family = families.get(var, "gaussian")
                        all_values[var][u, t, :] = _add_residual_noise(
                            mu_slice, var, family, stacked, n_samples, rng
                        )
                    else:
                        all_values[var][u, t, :] = mu_slice

    result_values: dict[str, np.ndarray] = {}
    by_time: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        result_values[var] = all_values[var].mean(axis=(0, 1))
        by_time[var] = all_values[var].mean(axis=0)  # (n_times, n_samples)

    return DoResult(
        values=result_values,
        values_by_time=by_time,
        time_index=np.array(time_values),
    )
