"""PathModel: the primary user-facing object returned by fit()."""

from __future__ import annotations

from typing import Any

import arviz as az
import graphviz
import pandas as pd
import pymc as pm

from pathmc.compile import build_design_matrix, compile_to_pymc
from pathmc.graph import GraphInfo, build_graph
from pathmc.introspect import (
    EquationList,
    PriorTable,
    build_dag_viz,
    build_equations,
    build_priors,
)
from pathmc.parse import Spec, parse_spec
from pathmc.simulate import DoResult, run_do


class PathModel:
    """A compiled Bayesian path model.

    Created by :func:`pathmc.fit`. Holds the parsed specification, graph
    structure, design matrices, and the compiled PyMC model.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    graph_info : GraphInfo
        DAG with topological order and node classification.
    data : pd.DataFrame
        Observed data used to build design matrices.
    """

    def __init__(
        self,
        spec: Spec,
        graph_info: GraphInfo,
        data: pd.DataFrame,
        families: dict[str, str] | None = None,
    ) -> None:
        self._spec = spec
        self._graph_info = graph_info
        self._data = data

        self._design_matrices: dict[str, pd.DataFrame] = {}
        for reg in spec.regressions:
            self._design_matrices[reg.lhs] = build_design_matrix(reg, data)

        self._pymc_model: pm.Model = compile_to_pymc(
            spec, data, self._design_matrices, families=families
        )
        self._idata: az.InferenceData | None = None

    @property
    def pymc_model(self) -> pm.Model:
        """The compiled PyMC model."""
        return self._pymc_model

    def design(self, var: str) -> pd.DataFrame:
        """Return the design matrix for an endogenous variable's equation.

        Parameters
        ----------
        var : str
            Name of the endogenous (LHS) variable.

        Returns
        -------
        pd.DataFrame
            Design matrix with named columns.

        Raises
        ------
        KeyError
            If *var* is not an endogenous variable in the model.
        """
        if var not in self._design_matrices:
            available = ", ".join(sorted(self._design_matrices))
            raise KeyError(
                f"No equation for '{var}'. Available endogenous variables: {available}"
            )
        return self._design_matrices[var]

    def graph(self) -> graphviz.Digraph:
        """Return a graphviz DAG of the structural model.

        Works before sampling. Exogenous nodes are drawn as boxes,
        endogenous nodes as ellipses. Labeled coefficients appear on edges.
        """
        return build_dag_viz(self._spec, self._graph_info)

    def equations(self) -> EquationList:
        """Return human-readable structural equations.

        Works before sampling.
        """
        return build_equations(self._spec)

    def priors(self) -> PriorTable:
        """Return a summary of prior distributions for all parameters.

        Works before sampling.
        """
        return build_priors(self._spec)

    def summary(self) -> pd.DataFrame:
        """Return a posterior summary table.

        Returns
        -------
        pd.DataFrame
            ArviZ summary of all model parameters.

        Raises
        ------
        RuntimeError
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .summary()."
            )
        return az.summary(self._idata)

    def sample(self, **kwargs: Any) -> az.InferenceData:
        """Run MCMC sampling and store the resulting InferenceData.

        Parameters
        ----------
        **kwargs
            Passed directly to ``pm.sample()`` (e.g. ``draws``, ``tune``,
            ``chains``, ``random_seed``).

        Returns
        -------
        az.InferenceData
            Posterior samples.
        """
        with self._pymc_model:
            self._idata = pm.sample(**kwargs)
        return self._idata

    def do(
        self,
        set: dict[str, float] | None = None,
        shift: dict[str, float] | None = None,
        kind: str = "mean",
    ) -> DoResult:
        """Simulate an intervention using the do-operator.

        Propagates posterior coefficient draws through the DAG in
        topological order, skipping the structural equation for any
        variable in *set*.

        Parameters
        ----------
        set : dict[str, float] | None
            Variables to fix at specific values (hard intervention).
        shift : dict[str, float] | None
            Reserved for soft interventions (not yet implemented).
        kind : str
            Propagation mode. Currently only ``"mean"`` is supported.

        Returns
        -------
        DoResult
            Container with ``.mean(var)``, ``.hdi(var)``, and
            contrast arithmetic via ``__sub__``.

        Raises
        ------
        RuntimeError
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .do()."
            )

        data_means = {col: float(self._data[col].mean()) for col in self._data.columns}
        design_columns = {
            var: list(dm.columns) for var, dm in self._design_matrices.items()
        }

        return run_do(
            spec=self._spec,
            graph_info=self._graph_info,
            idata=self._idata,
            data_means=data_means,
            design_columns=design_columns,
            set=set,
        )


def fit(
    spec_string: str,
    data: pd.DataFrame,
    families: dict[str, str] | None = None,
    **kwargs: Any,
) -> PathModel:
    """Parse a specification and compile a Bayesian path model.

    Parameters
    ----------
    spec_string : str
        Model specification in the pathmc DSL.
    data : pd.DataFrame
        Observed data.
    families : dict[str, str] | None
        Per-variable distribution families (default ``"gaussian"``).
    **kwargs
        Reserved for future options.

    Returns
    -------
    PathModel
        Compiled model ready for sampling and introspection.
    """
    spec = parse_spec(spec_string)
    graph_info = build_graph(spec)
    return PathModel(spec=spec, graph_info=graph_info, data=data, families=families)
