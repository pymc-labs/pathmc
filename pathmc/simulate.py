"""do() operator: interventional simulation via posterior propagation.

Implements Pearl's do-operator for cross-sectional path models by
propagating posterior coefficient draws through the DAG in topological
order, skipping the structural equation for any intervened variable.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from pathmc.graph import GraphInfo
from pathmc.panel import PanelInfo
from pathmc.parse import Spec, TransformCall
from pathmc.transforms import get_transform


class DoResult:
    """Container for propagated posterior draws under an intervention.

    Supports ``.mean(var)``, ``.hdi(var)``, and contrast arithmetic
    via subtraction (``scenario - baseline``).
    """

    def __init__(self, values: dict[str, np.ndarray]) -> None:
        self._values = values

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

    def __sub__(self, other: DoResult) -> DoResult:
        """Element-wise contrast between two DoResults."""
        new_values: dict[str, np.ndarray] = {}
        for var in self._values:
            if var in other._values:
                new_values[var] = self._values[var] - other._values[var]
        return DoResult(values=new_values)


def _expit(x: np.ndarray) -> np.ndarray:
    """Numerically stable inverse-logit (sigmoid)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def run_do(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data_means: dict[str, float],
    design_columns: dict[str, list[str]],
    set: dict[str, float] | None = None,
    families: dict[str, str] | None = None,
    kind: str = "mean",
    rng: np.random.Generator | None = None,
    panel_info: PanelInfo | None = None,
) -> DoResult:
    """Propagate posterior draws through the DAG under an intervention.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    graph_info : GraphInfo
        DAG with topological order.
    idata : az.InferenceData
        Posterior samples from ``pm.sample()``.
    data_means : dict[str, float]
        Mean of each observed variable in the data (used for non-intervened
        exogenous variables).
    design_columns : dict[str, list[str]]
        Column names of each design matrix, keyed by endogenous variable.
    set : dict[str, float] | None
        Variables to intervene on, with their fixed values.
    families : dict[str, str] | None
        Per-variable distribution families (default ``"gaussian"``).
    kind : str
        ``"mean"`` for deterministic propagation, ``"predictive"`` to add
        residual noise at each step.
    rng : np.random.Generator | None
        Random number generator for predictive sampling.

    Returns
    -------
    DoResult
        Propagated posterior draws for every variable in the DAG.
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
    values: dict[str, np.ndarray] = {}

    has_panel_intercepts = (
        panel_info is not None
        and f"mu_alpha_{graph_info.topological_order[-1]}" in stacked
    )

    for var in graph_info.topological_order:
        if var in set:
            values[var] = np.full(n_samples, set[var])
        elif var in graph_info.exogenous:
            values[var] = np.full(n_samples, data_means[var])
        else:
            beta_arr = stacked[f"beta_{var}"]
            cols = design_columns[var]
            linear = np.zeros(n_samples)

            for col in cols:
                coef = beta_arr.sel({f"{var}_predictors": col}).values
                if col == "Intercept":
                    linear = linear + coef
                elif col in transform_map:
                    transformed = _apply_transform_chain_numpy(
                        transform_map[col], values[col], stacked
                    )
                    linear = linear + coef * transformed
                else:
                    linear = linear + coef * values[col]

            if has_panel_intercepts and f"alpha_{var}" in stacked:
                alpha_arr = stacked[f"alpha_{var}"]
                alpha_mean = alpha_arr.mean(dim="unit").values
                linear = linear + alpha_mean

            family = families.get(var, "gaussian")

            if kind == "predictive":
                values[var] = _add_residual_noise(
                    linear, var, family, stacked, n_samples, rng
                )
            else:
                values[var] = _apply_link(linear, family)

    return DoResult(values=values)


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


def _apply_transform_chain_numpy(
    tc: TransformCall,
    input_val: np.ndarray,
    stacked: object,
) -> np.ndarray:
    """Apply a (possibly nested) transform chain using numpy.

    For cross-sectional do(), input_val is a constant broadcast to n_samples.
    The transform is applied pointwise, using posterior draws of the parameters.
    """
    if isinstance(tc.input_expr, TransformCall):
        input_val = _apply_transform_chain_numpy(tc.input_expr, input_val, stacked)

    transform = get_transform(tc.name)
    params = {key: stacked[name].values for key, name in tc.params.items()}

    if transform.name == "adstock":
        return input_val
    return transform.apply_numpy(input_val, params)


def run_panel_do(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data: pd.DataFrame,
    design_columns: dict[str, list[str]],
    panel_info: PanelInfo,
    set: dict[str, float] | None = None,
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
    set : dict[str, float] | None
        Variables to fix at specific values.
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
        Propagated draws averaged over units and time steps.
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
                    all_values[intervened_var][u_idx, t_idx, :] = intervened_val

            for var in graph_info.topological_order:
                if var in set:
                    all_values[var][u_idx, t_idx, :] = set[var]
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

                    family = families.get(var, "gaussian")
                    if kind == "predictive":
                        all_values[var][u_idx, t_idx, :] = _add_residual_noise(
                            linear, var, family, stacked, n_samples, rng
                        )
                    else:
                        all_values[var][u_idx, t_idx, :] = _apply_link(linear, family)

    result_values: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        result_values[var] = all_values[var].mean(axis=(0, 1))

    return DoResult(values=result_values)


def _get_panel_col_value(
    col: str,
    u_idx: int,
    t_idx: int,
    all_values: dict[str, np.ndarray],
    set_dict: dict[str, float],
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
        return np.full(n_samples, set_dict[col])
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
    import re

    m = re.match(r"^(.+)_lag(\d+)$", col_name)
    if m:
        return m.group(1), int(m.group(2))
    return None
