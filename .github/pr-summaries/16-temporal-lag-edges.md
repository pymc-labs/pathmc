# PR: Graph layer models temporal lag edges

Closes #16

## Issue Summary

When using `lag(var)` syntax, the graph builder treated `lag(sales)` as a completely separate exogenous node with no visible relationship to `sales`. The DAG had no temporal dimension.

## Root Cause

`build_graph()` only added contemporaneous edges from regression terms. It had no concept of temporal relationships, so `lag(sales)` appeared as an isolated root node even though it represents `sales` at a previous time step.

## Solution

Add temporal edges (with `temporal=True` attribute) from base variables to their lag nodes (e.g., `sales → lag(sales)`). Use a contemporaneous-only subgraph for cycle detection and topological sorting so the `sales → lag(sales) → sales` path doesn't trigger a false cycle error. Render temporal edges distinctly in the DAG visualization.

## Changes Made

- `pathmc/graph.py`: Add temporal edges from `Term.lag_of`, filter to contemporaneous subgraph for cycle detection and topological sort, add `temporal_edges` property and `contemporaneous_dag` property to `GraphInfo`
- `pathmc/identify.py`: Replace `graph_info._dag` with `graph_info.contemporaneous_dag` in all four public functions (`adjustment_sets`, `frontdoor_identifiable`, `collider_warnings`, `implied_independences`) so temporal edges don't interfere with d-separation queries
- `pathmc/introspect.py`: Style lag nodes as gray dashed boxes; render temporal edges as gray dashed arrows with `t−1` label and `constraint="false"` to avoid distorting layout
- `tests/test_graph.py`: 8 new tests for temporal edge creation, cycle avoidance, topological order, node classification, and contemporaneous DAG filtering
- `tests/test_identification.py`: 3 new regression tests verifying identification results are unchanged for models with lag terms

## Testing

- [x] Existing tests pass (351 fast tests)
- [x] New tests added (11 tests)
- [x] Linter clean

## Notes

- `compile.py`, `simulate.py`, `effects.py`, and `model.py` are unaffected — they consume `topological_order`, `exogenous`, and `endogenous` which are derived from the contemporaneous subgraph
- Panel identification (reasoning about lag-mediated confounding across time) is future work; temporal edges are metadata annotations for now
