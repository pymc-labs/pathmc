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
"""Graph builder for pathmc structural models.

Converts a parsed Spec into a directed acyclic graph (DAG) with
topological ordering, node classification, and residual covariance blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from pathmc.exceptions import CycleError
from pathmc.parse import Spec

__all__: list[str] = []


@dataclass
class GraphInfo:
    """DAG representation of a structural model.

    Attributes
    ----------
    topological_order : list[str]
        Nodes in a valid topological sort of the directed graph.
    exogenous : set[str]
        Nodes with no parents (root causes).
    endogenous : set[str]
        Nodes with at least one parent (determined by other variables).
    residual_blocks : list[set[str]]
        Connected components of ``~~`` residual covariance edges.
    latent : set[str]
        Endogenous variables declared as latent (no observed data column).
    """

    topological_order: list[str]
    exogenous: set[str]
    endogenous: set[str]
    residual_blocks: list[set[str]] = field(default_factory=list)
    latent: set[str] = field(default_factory=set)
    _dag: nx.DiGraph = field(repr=False, default_factory=nx.DiGraph)

    def has_edge(self, source: str, target: str) -> bool:
        """Return True if a directed edge exists from *source* to *target*."""
        return self._dag.has_edge(source, target)

    @property
    def temporal_edges(self) -> list[tuple[str, str]]:
        """Temporal edges as ``(source, lag_node)`` pairs."""
        return [
            (u, v) for u, v, d in self._dag.edges(data=True) if d.get("temporal", False)
        ]

    @property
    def contemporaneous_dag(self) -> nx.DiGraph:
        """Subgraph with only contemporaneous (non-temporal) edges.

        Preserves all nodes from the full DAG so that isolated lag
        variables remain visible for classification and ordering.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self._dag.nodes)
        g.add_edges_from(
            (u, v) for u, v, d in self._dag.edges(data=True) if not d.get("temporal")
        )
        return g


def build_graph(spec: Spec, latent: set[str] | None = None) -> GraphInfo:
    """Build a DAG from a parsed specification.

    Parameters
    ----------
    spec : Spec
        Parsed model specification (from ``parse_spec``).
    latent : set[str] | None
        Variables to treat as latent (unobserved deterministic mediators).
        Must be endogenous (appear as LHS of a regression).

    Returns
    -------
    GraphInfo
        Graph with topological order, node classification, and residual blocks.

    Raises
    ------
    CycleError
        If the directed edges form a cycle.
    ValueError
        If a latent variable is not endogenous.
    """
    if latent is None:
        latent = set()

    dag = nx.DiGraph()

    for reg in spec.regressions:
        dag.add_node(reg.lhs)
        for term in reg.terms:
            if term.interaction_of is not None:
                for var in term.interaction_of:
                    dag.add_node(var)
                    dag.add_edge(var, reg.lhs)
            else:
                dag.add_node(term.variable)
                dag.add_edge(term.variable, reg.lhs)

    for reg in spec.regressions:
        for term in reg.terms:
            if term.lag_of is not None:
                dag.add_edge(term.lag_of, term.variable, temporal=True)

    contemp_dag = nx.DiGraph()
    contemp_dag.add_nodes_from(dag.nodes)
    contemp_dag.add_edges_from(
        (u, v) for u, v, d in dag.edges(data=True) if not d.get("temporal")
    )

    if not nx.is_directed_acyclic_graph(contemp_dag):
        cycles = list(nx.simple_cycles(contemp_dag))
        cycle_str = " -> ".join(cycles[0] + [cycles[0][0]]) if cycles else "unknown"
        raise CycleError(
            f"Cycle detected: {cycle_str}. "
            "The structural model must be a directed acyclic graph (DAG). "
            "Remove or reverse an edge to break the cycle."
        )

    topological_order = list(nx.topological_sort(contemp_dag))

    endogenous = {reg.lhs for reg in spec.regressions}
    exogenous = set(dag.nodes) - endogenous

    for var in latent:
        if var not in endogenous:
            raise ValueError(
                f"Latent variable '{var}' is not endogenous "
                f"(no regression equation). Only variables on the left-hand "
                f"side of a regression can be declared latent."
            )

    residual_blocks = _build_residual_blocks(spec)

    block_vars = set().union(*residual_blocks) if residual_blocks else set()
    latent_in_blocks = latent & block_vars
    if latent_in_blocks:
        sorted_vars = ", ".join(f"'{v}'" for v in sorted(latent_in_blocks))
        raise ValueError(
            f"Latent variable(s) {sorted_vars} cannot appear in a residual "
            f"covariance block (~~). Latent variables have no observed data, "
            f"so residual covariance is undefined. Remove them from ~~ or "
            f"from the latent list."
        )

    return GraphInfo(
        topological_order=topological_order,
        exogenous=exogenous,
        endogenous=endogenous,
        residual_blocks=residual_blocks,
        latent=latent,
        _dag=dag,
    )


def _build_residual_blocks(spec: Spec) -> list[set[str]]:
    """Find connected components from ``~~`` declarations."""
    if not spec.residual_covs:
        return []

    ug = nx.Graph()
    for rc in spec.residual_covs:
        ug.add_edge(rc.var1, rc.var2)

    return [comp for comp in nx.connected_components(ug)]
