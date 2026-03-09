"""M2 gate tests: graph building and validation.

These tests verify that a parsed Spec is correctly converted to a DAG
with topological ordering, node classification, and cycle detection.

All tests are fast (no data, no PyMC, no sampling).
"""

import pytest

from pathmc.parse import parse_spec
from pathmc.graph import build_graph

from conftest import (
    MEDIATION_SPEC,
    FORK_SPEC,
    COLLIDER_SPEC,
    PARALLEL_MEDIATORS_SPEC,
    CYCLIC_SPEC,
)


class TestTopologicalOrder:
    def test_mediation_order(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        order = info.topological_order
        assert order.index("X") < order.index("M")
        assert order.index("M") < order.index("Y")

    def test_fork_order(self):
        spec = parse_spec(FORK_SPEC)
        info = build_graph(spec)
        order = info.topological_order
        assert order.index("Z") < order.index("X")
        assert order.index("Z") < order.index("Y")

    def test_collider_order(self):
        spec = parse_spec(COLLIDER_SPEC)
        info = build_graph(spec)
        order = info.topological_order
        assert order.index("X") < order.index("C")
        assert order.index("Y") < order.index("C")

    def test_all_nodes_in_order(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        info = build_graph(spec)
        order_set = set(info.topological_order)
        assert order_set == {"T", "M1", "M2", "Y"}


class TestNodeClassification:
    def test_exogenous_nodes(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert "X" in info.exogenous

    def test_endogenous_nodes(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert "M" in info.endogenous
        assert "Y" in info.endogenous

    def test_exogenous_endogenous_disjoint(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        info = build_graph(spec)
        assert info.exogenous.isdisjoint(info.endogenous)

    def test_exogenous_union_endogenous_is_all_nodes(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        info = build_graph(spec)
        assert info.exogenous | info.endogenous == set(info.topological_order)


class TestEdges:
    def test_directed_edges_present(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert info.has_edge("X", "M")
        assert info.has_edge("M", "Y")
        assert info.has_edge("X", "Y")

    def test_no_reverse_edges(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert not info.has_edge("Y", "X")
        assert not info.has_edge("M", "X")
        assert not info.has_edge("Y", "M")

    def test_collider_edges(self):
        spec = parse_spec(COLLIDER_SPEC)
        info = build_graph(spec)
        assert info.has_edge("X", "C")
        assert info.has_edge("Y", "C")
        assert not info.has_edge("C", "X")


class TestResidualBlocks:
    def test_residual_block_identified(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        info = build_graph(spec)
        assert any({"M1", "M2"} <= block for block in info.residual_blocks)

    def test_no_blocks_when_no_residual_cov(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert len(info.residual_blocks) == 0


class TestCycleDetection:
    def test_cycle_raises_error(self):
        spec = parse_spec(CYCLIC_SPEC)
        with pytest.raises(Exception, match="(?i)cycl"):
            build_graph(spec)

    def test_acyclic_spec_does_not_raise(self):
        spec = parse_spec(PARALLEL_MEDIATORS_SPEC)
        build_graph(spec)  # should not raise


class TestTemporalEdges:
    """Temporal edges for lag() syntax (#16)."""

    def test_lag_creates_temporal_edge(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        info = build_graph(spec)
        assert ("sales", "lag(sales)") in info.temporal_edges

    def test_lag_no_contemporaneous_cycle(self):
        """sales ~ lag(sales) should not raise CycleError."""
        spec = parse_spec("sales ~ lag(sales)")
        info = build_graph(spec)
        assert "lag(sales)" in info.exogenous
        assert "sales" in info.endogenous

    def test_lag_topological_order_unchanged(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        info = build_graph(spec)
        order = info.topological_order
        assert order.index("lag(sales)") < order.index("sales")
        assert order.index("spend") < order.index("sales")

    def test_lag_exogenous_classification_unchanged(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        info = build_graph(spec)
        assert "lag(sales)" in info.exogenous
        assert "spend" in info.exogenous
        assert "sales" in info.endogenous

    def test_contemporaneous_dag_excludes_temporal(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        info = build_graph(spec)
        cdag = info.contemporaneous_dag
        assert not cdag.has_edge("sales", "lag(sales)")
        assert cdag.has_edge("lag(sales)", "sales")
        assert cdag.has_edge("spend", "sales")

    def test_contemporaneous_dag_preserves_all_nodes(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        info = build_graph(spec)
        cdag = info.contemporaneous_dag
        assert set(cdag.nodes) == set(info._dag.nodes)

    def test_no_temporal_edges_without_lags(self):
        spec = parse_spec(MEDIATION_SPEC)
        info = build_graph(spec)
        assert info.temporal_edges == []

    def test_multiple_lag_terms(self):
        spec = parse_spec("Y ~ lag(X)\nX ~ lag(X)")
        info = build_graph(spec)
        assert ("X", "lag(X)") in info.temporal_edges
        assert len(info.temporal_edges) == 1
