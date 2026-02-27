"""do() operator: interventional simulation via posterior propagation.

Implements Pearl's do-operator for cross-sectional path models by
propagating posterior coefficient draws through the DAG in topological
order, skipping the structural equation for any intervened variable.
"""

from __future__ import annotations

import arviz as az
import numpy as np

from pathmc.graph import GraphInfo
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


def run_do(
    spec: Spec,
    graph_info: GraphInfo,
    idata: az.InferenceData,
    data_means: dict[str, float],
    design_columns: dict[str, list[str]],
    set: dict[str, float] | None = None,
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

    Returns
    -------
    DoResult
        Propagated posterior draws for every variable in the DAG.
    """
    if set is None:
        set = {}

    stacked = idata.posterior.stack(sample=("chain", "draw"))
    n_samples = stacked.sizes["sample"]

    values: dict[str, np.ndarray] = {}

    for var in graph_info.topological_order:
        if var in set:
            values[var] = np.full(n_samples, set[var])
        elif var in graph_info.exogenous:
            values[var] = np.full(n_samples, data_means[var])
        else:
            beta_arr = stacked[f"beta_{var}"]
            cols = design_columns[var]
            result = np.zeros(n_samples)

            for col in cols:
                coef = beta_arr.sel({f"{var}_predictors": col}).values
                if col == "Intercept":
                    result = result + coef
                else:
                    result = result + coef * values[col]

            values[var] = result

    return DoResult(values=values)
