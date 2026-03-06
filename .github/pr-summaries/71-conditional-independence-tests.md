# PR: Enumerate and test implied conditional independences from the DAG

Closes #71

## Issue Summary

Add the ability to enumerate all conditional independence implications of a DAG and automatically test them against observed data using partial correlation tests. This lets users assess whether their proposed DAG is consistent with the data before (or after) fitting the model.

## Root Cause

pathmc had no mechanism for checking DAG-data compatibility. Users could specify a DAG and fit it, but had no way to see whether the data contradicted the structural assumptions — specifically, whether pairs of variables that the DAG says should be conditionally independent actually are.

## Solution

Implemented the **basis set** approach (Shipley, 2000): for each pair of non-adjacent nodes in the DAG, compute the conditioning set Z = pa(X) ∪ pa(Y) \ {X, Y}, verify d-separation via NetworkX, and test the implied independence against data using partial correlation (regression residuals + Pearson r test).

Key design choices:
- **No new dependencies**: uses NetworkX's `d_separated()` for graph queries and scipy (already a PyMC transitive dependency) for statistical tests
- **Works before sampling**: only requires the graph structure and observed data, not the posterior
- **Rich result object**: `ImplicationTestResult` with `__repr__`, `_repr_html_()` for Jupyter, `.violations`, `.to_dataframe()`

## Changes Made

- `pathmc/identify.py`: Added `ConditionalIndependence` dataclass, `ImplicationTestResult` dataclass with rich display, `implied_independences()` function, `test_implications()` function, and `_partial_correlation_test()` helper
- `pathmc/model.py`: Added `PathModel.implied_independences()` and `PathModel.test_implications()` methods, imported new symbols from identify
- `docs/comparison.qmd`: Added "Implied independence tests" row to the feature comparison table

## Testing

- [x] All 283 existing tests pass (non-slow)
- [x] Smoke-tested against multiple DAG topologies (chain, fork, collider, diamond)
- [x] Verified violation detection with misspecified DAGs
- [x] Verified empty-case handling (fully connected DAGs with no missing edges)
- [x] Lint clean (ruff check + format)

## Comparison with other packages

This feature is analogous to R's `dagitty::impliedConditionalIndependencies()` with its "missing.edge" / "basis.set" type. In Python, `pgmpy` has some conditional independence testing but is not focused on the SEM/path-analysis workflow. Neither DoWhy, CausalPy, semopy, EconML, nor Bambi offer this capability — pathmc is now the only Python package in its comparison set with built-in DAG implication testing.

## Notes

- The partial correlation test assumes continuous data with linear relationships. Future work could add non-parametric tests (e.g., kernel-based CI tests) or tests appropriate for binary/count data.
- Follow-up issues will be created for other DAG scrutiny methods identified during brainstorming (residual correlation diagnostics, posterior predictive d-separation checks, vanishing tetrads).
