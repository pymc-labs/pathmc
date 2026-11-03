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
"""Independent gate tests for variance-optimal graphical adjustment sets."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

import pathmc
from pathmc.graph import GraphInfo, build_graph
from pathmc.identify import optimal_adjustment_set
from pathmc.parse import parse_spec


def _graph_info(
    nodes: list[str],
    edges: list[tuple[str, str]],
    *,
    latent: set[str] | None = None,
    temporal_edges: set[tuple[str, str]] | None = None,
) -> GraphInfo:
    """Build ``GraphInfo`` directly so oracle graphs need not use the DSL."""
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    temporal_edges = temporal_edges or set()
    for source, target in edges:
        dag.add_edge(source, target, temporal=(source, target) in temporal_edges)
    endogenous = {target for _, target in edges}
    return GraphInfo(
        topological_order=list(nx.topological_sort(dag)),
        exogenous=set(nodes) - endogenous,
        endogenous=endogenous,
        residual_blocks=[],
        latent=set() if latent is None else set(latent),
        _dag=dag,
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("Y ~ X", set()),
        ("X ~ Z\nY ~ X + Z", {"Z"}),
        ("M ~ X\nY ~ M", set()),
        ("X ~ I\nY ~ X + P", {"P"}),
        ("X ~ Z1\nZ2 ~ Z1\nY ~ X + Z2", {"Z2"}),
        ("M ~ X + Z\nY ~ M", {"Z"}),
        (
            "M1 ~ Z1 + X\nM2 ~ Z2\nY ~ M1 + M2\nX ~ Z2",
            {"M2", "Z1"},
        ),
        ("M ~ X + Z\nY ~ M + W\nD ~ M\nQ ~ Y\nS ~ X", {"Z", "W"}),
    ],
)
def test_hand_derived_o_sets(spec, expected):
    graph = build_graph(parse_spec(spec))
    assert optimal_adjustment_set(graph, "X", "Y") == expected


def test_empty_o_set_is_a_fresh_successful_result():
    graph = build_graph(parse_spec("Y ~ X"))
    first = optimal_adjustment_set(graph, "X", "Y")
    second = optimal_adjustment_set(graph, "X", "Y")
    first.add("mutated")
    assert second == set()


def test_latent_nodes_elsewhere_do_not_disable_observed_o_set():
    graph = build_graph(parse_spec("U ~ W\nX ~ U\nZ ~ U\nY ~ X + Z"), latent={"U"})
    assert optimal_adjustment_set(graph, "X", "Y") == {"Z"}


def test_collider_and_irrelevant_components_are_not_selected():
    graph = build_graph(parse_spec("M ~ A + B\nX ~ A\nY ~ X + B\nD ~ C"))
    assert optimal_adjustment_set(graph, "X", "Y") == {"B"}


def test_multiple_causal_paths_include_only_outcome_precision_parent():
    graph = build_graph(parse_spec("M1 ~ X\nM2 ~ X\nY ~ M1 + M2 + P"))
    assert optimal_adjustment_set(graph, "X", "Y") == {"P"}


def test_equation_insertion_order_does_not_change_result():
    forward = build_graph(parse_spec("X ~ Z1\nZ2 ~ Z1\nY ~ X + Z2"))
    reverse = build_graph(parse_spec("Y ~ Z2 + X\nZ2 ~ Z1\nX ~ Z1"))
    assert optimal_adjustment_set(forward, "X", "Y") == {"Z2"}
    assert optimal_adjustment_set(reverse, "X", "Y") == {"Z2"}


def test_call_does_not_mutate_graph_info():
    graph = build_graph(parse_spec("M ~ X + Z\nY ~ M\nA ~ B\nC ~ D\nA ~~ C"))
    before_nodes = list(graph._dag.nodes(data=True))
    before_edges = list(graph._dag.edges(data=True))
    before_latent = set(graph.latent)
    before_blocks = deepcopy(graph.residual_blocks)

    assert optimal_adjustment_set(graph, "X", "Y") == {"Z"}

    assert list(graph._dag.nodes(data=True)) == before_nodes
    assert list(graph._dag.edges(data=True)) == before_edges
    assert graph.latent == before_latent
    assert graph.residual_blocks == before_blocks


@pytest.mark.parametrize(
    ("treatment", "outcome", "message"),
    [
        ("missing", "Y", "Treatment 'missing' not in DAG"),
        ("X", "missing", "Outcome 'missing' not in DAG"),
        ("X", "X", "distinct treatment and outcome"),
    ],
)
def test_endpoint_validation(treatment, outcome, message):
    graph = build_graph(parse_spec("Y ~ X"))
    with pytest.raises(ValueError, match=message):
        optimal_adjustment_set(graph, treatment, outcome)


@pytest.mark.parametrize("endpoint", ["treatment", "outcome"])
def test_latent_endpoint_is_rejected(endpoint):
    graph = build_graph(parse_spec("M ~ X\nY ~ M"), latent={"M"})
    treatment, outcome = ("M", "Y") if endpoint == "treatment" else ("X", "M")
    with pytest.raises(ValueError, match=rf"{endpoint.capitalize()} 'M'.*latent"):
        optimal_adjustment_set(graph, treatment, outcome)


@pytest.mark.parametrize("spec", ["X ~ Z\nY ~ Z", "Y ~ Z\nX ~ W"])
def test_no_directed_path_reports_structural_zero(spec):
    graph = build_graph(parse_spec(spec))
    with pytest.raises(ValueError, match="no directed causal path.*zero total effect"):
        optimal_adjustment_set(graph, "X", "Y")


def test_no_path_boundary_does_not_change_existing_minimal_set_behavior():
    model = pathmc.model("X ~ Z\nY ~ Z")
    assert model.adjustment_sets("X", "Y") == [{"Z"}]


def test_latent_required_o_set_is_distinct_from_nonidentification():
    graph = build_graph(parse_spec("U ~ W\nX ~ Z\nY ~ X + U"), latent={"U"})
    with pytest.raises(ValueError, match=r"requires.*\['U'\].*explicit valid set"):
        optimal_adjustment_set(graph, "X", "Y")


def test_residual_confounding_without_observed_set_reports_nonidentification():
    graph = build_graph(parse_spec("X ~ Z\nM ~ X\nY ~ M\nX ~~ Y"))
    with pytest.raises(ValueError, match="No observed backdoor adjustment set"):
        optimal_adjustment_set(graph, "X", "Y")


def test_residual_required_o_set_names_declaration_not_synthetic_node():
    graph = build_graph(parse_spec("_u_resid_0 ~ W\nM ~ X\nY ~ M\nM ~~ Y"))
    with pytest.raises(ValueError, match=r"M ~~ Y.*explicit valid set") as exc_info:
        optimal_adjustment_set(graph, "X", "Y")
    assert "_u_resid_0_" not in str(exc_info.value)


def test_latent_mediator_and_unrelated_residual_block_are_supported_data_free():
    model = pathmc.model(
        "M ~ X\nY ~ M + P\nA ~ B\nC ~ D\nA ~~ C",
        latent=["M"],
    )
    assert model.optimal_adjustment_set("X", "Y") == {"P"}


def test_temporal_edges_are_not_treated_as_contemporaneous_causal_paths():
    graph = _graph_info(
        ["X", "lag_X", "Y"],
        [("X", "lag_X"), ("lag_X", "Y")],
        temporal_edges={("X", "lag_X")},
    )
    with pytest.raises(ValueError, match="no directed causal path"):
        optimal_adjustment_set(graph, "X", "Y")


def _reference_o_set(dag: nx.DiGraph, treatment: str, outcome: str) -> set[str]:
    """Small-graph oracle using explicit directed-path enumeration."""
    paths = list(nx.all_simple_paths(dag, treatment, outcome))
    causal = {node for path in paths for node in path[1:]}
    forbidden = causal | {treatment}
    for node in causal:
        forbidden.update(nx.descendants(dag, node))
    parents = {parent for node in causal for parent in dag.predecessors(node)}
    return parents - forbidden


def _is_backdoor_valid(
    dag: nx.DiGraph, treatment: str, outcome: str, adjustment: set[str]
) -> bool:
    """Independent validity oracle using NetworkX d-separation directly."""
    if adjustment & nx.descendants(dag, treatment):
        return False
    mutilated = dag.copy()
    mutilated.remove_edges_from(list(dag.out_edges(treatment)))
    return nx.is_d_separator(mutilated, {treatment}, {outcome}, adjustment)


def test_small_dags_match_directed_path_and_d_separation_oracles():
    nodes = ["A", "X", "B", "P", "Y"]
    possible = [edge for edge in combinations(nodes, 2) if edge != ("X", "Y")]
    for mask in range(1 << len(possible)):
        edges = [("X", "Y")]
        edges.extend(edge for bit, edge in enumerate(possible) if mask & (1 << bit))
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        dag.add_edges_from(edges)
        graph = _graph_info(nodes, edges)

        expected = _reference_o_set(dag, "X", "Y")
        actual = optimal_adjustment_set(graph, "X", "Y")

        assert actual == expected, f"edges={edges}"
        assert _is_backdoor_valid(dag, "X", "Y", actual), f"edges={edges}"


def _population_covariance(
    nodes: list[str],
    coefficients: dict[tuple[str, str], float],
) -> np.ndarray:
    """Return covariance for ``v = B v + epsilon`` with unit noise."""
    positions = {node: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    for (source, target), coefficient in coefficients.items():
        matrix[positions[target], positions[source]] = coefficient
    transform = np.linalg.inv(np.eye(len(nodes)) - matrix)
    return transform @ transform.T


def _conditional_variance(
    covariance: np.ndarray,
    positions: dict[str, int],
    target: str,
    given: list[str],
) -> float:
    target_index = positions[target]
    if not given:
        return float(covariance[target_index, target_index])
    given_indices = [positions[node] for node in given]
    cross = covariance[target_index, given_indices]
    conditioned = covariance[np.ix_(given_indices, given_indices)]
    return float(
        covariance[target_index, target_index]
        - cross @ np.linalg.solve(conditioned, cross.T)
    )


def _ols_asymptotic_variance(
    covariance: np.ndarray,
    nodes: list[str],
    treatment: str,
    outcome: str,
    adjustment: set[str],
) -> float:
    positions = {node: index for index, node in enumerate(nodes)}
    ordered = sorted(adjustment)
    numerator = _conditional_variance(
        covariance, positions, outcome, [treatment, *ordered]
    )
    denominator = _conditional_variance(covariance, positions, treatment, ordered)
    return numerator / denominator


@pytest.mark.parametrize(
    ("nodes", "coefficients"),
    [
        (
            ["I", "X", "P", "Y"],
            {("I", "X"): 0.8, ("X", "Y"): 0.5, ("P", "Y"): 1.5},
        ),
        (
            ["Z", "X", "P", "Y"],
            {
                ("Z", "X"): 0.7,
                ("Z", "Y"): 0.4,
                ("X", "Y"): 0.6,
                ("P", "Y"): 1.2,
            },
        ),
        (
            ["X", "Z", "M", "Y"],
            {("X", "M"): 0.8, ("Z", "M"): 0.6, ("M", "Y"): 0.9},
        ),
    ],
)
def test_o_set_minimizes_population_ols_variance(nodes, coefficients):
    edges = list(coefficients)
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(edges)
    graph = _graph_info(nodes, edges)
    covariance = _population_covariance(nodes, coefficients)
    optimal = optimal_adjustment_set(graph, "X", "Y")
    optimal_variance = _ols_asymptotic_variance(covariance, nodes, "X", "Y", optimal)

    candidates = set(nodes) - {"X", "Y"} - nx.descendants(dag, "X")
    for size in range(len(candidates) + 1):
        for subset in combinations(sorted(candidates), size):
            adjustment = set(subset)
            if not _is_backdoor_valid(dag, "X", "Y", adjustment):
                continue
            candidate_variance = _ols_asymptotic_variance(
                covariance, nodes, "X", "Y", adjustment
            )
            assert optimal_variance <= candidate_variance + 1e-10


def test_precision_predictor_has_hand_computed_five_to_one_gain():
    nodes = ["X", "P", "Y"]
    coefficients = {("X", "Y"): 1.0, ("P", "Y"): 2.0}
    covariance = _population_covariance(nodes, coefficients)
    assert _ols_asymptotic_variance(covariance, nodes, "X", "Y", set()) == (
        pytest.approx(5.0)
    )
    assert _ols_asymptotic_variance(covariance, nodes, "X", "Y", {"P"}) == (
        pytest.approx(1.0)
    )
