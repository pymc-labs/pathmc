"""Causal identification helpers for pathmc structural models.

Provides backdoor adjustment set computation, collider warnings,
and identifiability checks using the DAG stored in GraphInfo.
"""

from __future__ import annotations

from itertools import combinations

import networkx as nx

from pathmc.graph import GraphInfo


def adjustment_sets(
    graph_info: GraphInfo,
    treatment: str,
    outcome: str,
) -> list[set[str]]:
    """Find all valid backdoor adjustment sets for the causal effect
    of *treatment* on *outcome*.

    A valid backdoor set Z satisfies:
    1. No node in Z is a descendant of treatment.
    2. Z blocks all backdoor paths (non-causal paths) from treatment
       to outcome.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.
    treatment : str
        Treatment variable name.
    outcome : str
        Outcome variable name.

    Returns
    -------
    list[set[str]]
        All valid minimal adjustment sets, sorted by size then
        alphabetically. Empty list if no valid set exists or if
        the effect is already identified without adjustment.
    """
    dag = graph_info._dag

    if treatment not in dag.nodes:
        raise ValueError(
            f"Treatment '{treatment}' not in DAG. Available nodes: {sorted(dag.nodes)}"
        )
    if outcome not in dag.nodes:
        raise ValueError(
            f"Outcome '{outcome}' not in DAG. Available nodes: {sorted(dag.nodes)}"
        )

    descendants = nx.descendants(dag, treatment)
    candidates = set(dag.nodes) - {treatment, outcome} - descendants

    mutilated = dag.copy()
    mutilated.remove_edges_from(list(dag.in_edges(treatment)))

    valid_sets: list[set[str]] = []

    for size in range(len(candidates) + 1):
        for subset in combinations(sorted(candidates), size):
            z = set(subset)
            if _blocks_all_backdoor_paths(dag, mutilated, treatment, outcome, z):
                if _is_minimal(z, valid_sets):
                    valid_sets.append(z)

    valid_sets.sort(key=lambda s: (len(s), sorted(s)))
    return valid_sets


def is_identifiable(
    graph_info: GraphInfo,
    treatment: str,
    outcome: str,
) -> bool:
    """Check whether the causal effect of *treatment* on *outcome*
    is identifiable via the backdoor criterion.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.
    treatment : str
        Treatment variable name.
    outcome : str
        Outcome variable name.

    Returns
    -------
    bool
        True if at least one valid backdoor adjustment set exists
        (including the empty set, meaning no adjustment is needed).
    """
    return len(adjustment_sets(graph_info, treatment, outcome)) > 0


def collider_warnings(
    graph_info: GraphInfo,
    adjustment_vars: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Check if any variable in the adjustment set is a collider
    on a path between treatment and outcome.

    Conditioning on a collider opens a spurious path and introduces
    bias. This function warns about such variables.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.
    adjustment_vars : set[str]
        Proposed adjustment set.
    treatment : str
        Treatment variable name.
    outcome : str
        Outcome variable name.

    Returns
    -------
    list[str]
        Human-readable warning strings. Empty if no issues found.
    """
    dag = graph_info._dag
    warnings_list: list[str] = []

    for var in adjustment_vars:
        if var not in dag.nodes:
            continue
        parents = list(dag.predecessors(var))
        if len(parents) >= 2:
            treatment_ancestor = var in nx.descendants(
                dag, treatment
            ) or treatment in nx.ancestors(dag, var)
            outcome_ancestor = var in nx.descendants(
                dag, outcome
            ) or outcome in nx.ancestors(dag, var)
            if treatment_ancestor or outcome_ancestor:
                if _is_collider_on_path(dag, var, treatment, outcome):
                    warnings_list.append(
                        f"'{var}' is a collider between '{treatment}' and "
                        f"'{outcome}'. Conditioning on it may open a spurious "
                        f"path and introduce bias."
                    )

    return warnings_list


def _blocks_all_backdoor_paths(
    original_dag: nx.DiGraph,
    mutilated_dag: nx.DiGraph,
    treatment: str,
    outcome: str,
    z: set[str],
) -> bool:
    """Check if Z blocks all backdoor paths from treatment to outcome.

    A backdoor path is any path in the undirected skeleton of the
    original DAG that starts with an arrow INTO the treatment node.
    """
    undirected = original_dag.to_undirected()

    for path in nx.all_simple_paths(undirected, treatment, outcome):
        if len(path) < 2:
            continue

        # A backdoor path has an arrow into treatment from the next node
        next_node = path[1]
        is_backdoor = original_dag.has_edge(next_node, treatment)

        if not is_backdoor:
            continue

        if len(path) < 3:
            # Two-node backdoor path: treatment <- outcome (direct edge into
            # treatment from outcome). This would mean outcome causes treatment,
            # which is unusual. Blocked only if impossible.
            if not _path_blocked_by(original_dag, path, z):
                return False
        else:
            if not _path_blocked_by(original_dag, path, z):
                return False
    return True


def _path_blocked_by(
    dag: nx.DiGraph,
    path: list[str],
    z: set[str],
) -> bool:
    """Check if a path is blocked by conditioning set Z using d-separation rules.

    A path is blocked if any intermediate node satisfies:
    - It's a non-collider (chain or fork) and is in Z, OR
    - It's a collider and neither it nor any descendant is in Z.
    """
    for i in range(1, len(path) - 1):
        prev_node, node, next_node = path[i - 1], path[i], path[i + 1]

        into_from_prev = dag.has_edge(prev_node, node)
        into_from_next = dag.has_edge(next_node, node)

        is_collider = into_from_prev and into_from_next

        if is_collider:
            desc = nx.descendants(dag, node) | {node}
            if not (desc & z):
                return True
        else:
            if node in z:
                return True

    return False


def _is_collider_on_path(
    dag: nx.DiGraph,
    var: str,
    treatment: str,
    outcome: str,
) -> bool:
    """Check if var is a collider on any undirected path between treatment
    and outcome."""
    undirected = dag.to_undirected()
    try:
        for path in nx.all_simple_paths(undirected, treatment, outcome):
            if var not in path:
                continue
            idx = path.index(var)
            if idx == 0 or idx == len(path) - 1:
                continue
            prev_node = path[idx - 1]
            next_node = path[idx + 1]
            if dag.has_edge(prev_node, var) and dag.has_edge(next_node, var):
                return True
    except nx.NetworkXError:
        pass
    return False


def _is_minimal(candidate: set[str], existing: list[set[str]]) -> bool:
    """A set is minimal if no existing valid set is a proper subset of it."""
    for s in existing:
        if s < candidate:
            return False
    return True
