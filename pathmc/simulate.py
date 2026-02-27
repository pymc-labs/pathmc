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
from pathmc.parse import Spec


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
            elif family == "bernoulli":
                values[var] = _expit(linear)
            else:
                values[var] = linear

    return DoResult(values=values)


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

    sigma_arr = stacked[f"sigma_{var}"].values
    return linear + rng.normal(0, sigma_arr)


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

    for u_idx, unit in enumerate(units):
        unit_mask = data_sorted[unit_col] == unit
        unit_data = data_sorted[unit_mask].sort_values(time_col).reset_index(drop=True)

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
                        else:
                            is_lag = _parse_lag(col)
                            if is_lag is not None:
                                base_var, lag_k = is_lag
                                src_t = t_idx - lag_k
                                if src_t >= 0 and base_var in all_values:
                                    parent_val = all_values[base_var][u_idx, src_t, :]
                                elif init_from == "observed" and t_idx < len(unit_data):
                                    parent_val = np.full(
                                        n_samples,
                                        float(unit_data.iloc[t_idx].get(col, 0.0)),
                                    )
                                else:
                                    parent_val = np.zeros(n_samples)
                                linear = linear + coef * parent_val
                            elif col in all_values:
                                linear = (
                                    linear + coef * all_values[col][u_idx, t_idx, :]
                                )
                            elif col in set:
                                linear = linear + coef * set[col]
                            else:
                                if t_idx < len(unit_data):
                                    val = float(unit_data.iloc[t_idx].get(col, 0.0))
                                else:
                                    val = 0.0
                                linear = linear + coef * val

                    if has_alpha and f"alpha_{var}" in stacked:
                        alpha_arr = stacked[f"alpha_{var}"]
                        unit_label = unit
                        alpha_unit = alpha_arr.sel(unit=unit_label).values
                        linear = linear + alpha_unit

                    family = families.get(var, "gaussian")
                    if kind == "predictive":
                        all_values[var][u_idx, t_idx, :] = _add_residual_noise(
                            linear, var, family, stacked, n_samples, rng
                        )
                    elif family == "bernoulli":
                        all_values[var][u_idx, t_idx, :] = _expit(linear)
                    else:
                        all_values[var][u_idx, t_idx, :] = linear

    result_values: dict[str, np.ndarray] = {}
    for var in graph_info.topological_order:
        result_values[var] = all_values[var].mean(axis=(0, 1))

    return DoResult(values=result_values)


def _parse_lag(col_name: str) -> tuple[str, int] | None:
    """Parse a lag column name like 'sales_lag1' -> ('sales', 1)."""
    import re

    m = re.match(r"^(.+)_lag(\d+)$", col_name)
    if m:
        return m.group(1), int(m.group(2))
    return None
