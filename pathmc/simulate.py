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
