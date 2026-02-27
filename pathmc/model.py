"""PathModel: the primary user-facing object returned by fit()."""

from __future__ import annotations

import pandas as pd
import pymc as pm

import graphviz

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
    ) -> None:
        self._spec = spec
        self._graph_info = graph_info
        self._data = data

        self._design_matrices: dict[str, pd.DataFrame] = {}
        for reg in spec.regressions:
            self._design_matrices[reg.lhs] = build_design_matrix(reg, data)

        self._pymc_model: pm.Model = compile_to_pymc(spec, data, self._design_matrices)

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


def fit(spec_string: str, data: pd.DataFrame, **kwargs) -> PathModel:
    """Parse a specification and compile a Bayesian path model.

    Parameters
    ----------
    spec_string : str
        Model specification in the pathmc DSL.
    data : pd.DataFrame
        Observed data.
    **kwargs
        Reserved for future options (e.g. custom priors, families).

    Returns
    -------
    PathModel
        Compiled model ready for sampling and introspection.
    """
    spec = parse_spec(spec_string)
    graph_info = build_graph(spec)
    return PathModel(spec=spec, graph_info=graph_info, data=data)
