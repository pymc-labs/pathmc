"""Graph builder for pathmc structural models.

Converts a parsed Spec into a directed acyclic graph (DAG) with
topological ordering, node classification, and residual covariance blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from pathmc.exceptions import CycleError
from pathmc.parse import Spec


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
    """

    topological_order: list[str]
    exogenous: set[str]
    endogenous: set[str]
    residual_blocks: list[set[str]] = field(default_factory=list)
    _dag: nx.DiGraph = field(repr=False, default_factory=nx.DiGraph)

    def has_edge(self, source: str, target: str) -> bool:
        """Return True if a directed edge exists from *source* to *target*."""
        return self._dag.has_edge(source, target)


def build_graph(spec: Spec) -> GraphInfo:
    """Build a DAG from a parsed specification.

    Parameters
    ----------
    spec : Spec
        Parsed model specification (from ``parse_spec``).

    Returns
    -------
    GraphInfo
        Graph with topological order, node classification, and residual blocks.

    Raises
    ------
    CycleError
        If the directed edges form a cycle.
    """
    dag = nx.DiGraph()

    for reg in spec.regressions:
        dag.add_node(reg.lhs)
        for term in reg.terms:
            dag.add_node(term.variable)
            dag.add_edge(term.variable, reg.lhs)

    if not nx.is_directed_acyclic_graph(dag):
        cycles = list(nx.simple_cycles(dag))
        cycle_str = " -> ".join(cycles[0] + [cycles[0][0]]) if cycles else "unknown"
        raise CycleError(
            f"Cycle detected: {cycle_str}. "
            "The structural model must be a directed acyclic graph (DAG). "
            "Remove or reverse an edge to break the cycle."
        )

    topological_order = list(nx.topological_sort(dag))

    endogenous = {reg.lhs for reg in spec.regressions}
    exogenous = set(dag.nodes) - endogenous

    residual_blocks = _build_residual_blocks(spec)

    return GraphInfo(
        topological_order=topological_order,
        exogenous=exogenous,
        endogenous=endogenous,
        residual_blocks=residual_blocks,
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
