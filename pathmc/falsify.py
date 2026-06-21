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
"""Whole-graph DAG falsification via a permutation-based test.

Ports the permutation test of Eulig et al. (2023), as implemented in
dowhy's ``gcm.falsify_graph``, to pathmc. Rather than checking implied
conditional independences one missing edge at a time (see
:mod:`pathmc.identify`), this module grades the *entire* DAG against a
baseline of randomly relabeled ("rewired") competitor graphs.

Two questions are answered:

1. **Does the graph break its promises?** The Local Markov Condition
   (LMC) test counts how many implied parental conditional independences
   the data actually violate.
2. **Is the graph informative at all?** A node-permutation baseline
   (the test of permutation/adjacency, tPA) checks whether the proposed
   DAG fits the data meaningfully better than random rewirings, and
   whether its arrow directions are even distinguishable from the pile of
   permutations (those lying in the same Markov equivalence class).

An informative DAG that beats the permuted baseline on LMC violations is
the positive case: *not contradicted and testable*. A non-informative
DAG (many permutations share its Markov equivalence class) is also *not
rejected*, but that verdict is vacuous — interpret it as *not
falsifiable* rather than as evidence for the graph.

Reference
---------
Eulig, E., Mastakouri, A. A., Blöbaum, P., Hardt, M., & Janzing, D.
(2023). Toward Falsifying Causal Graphs Using a Permutation-Based Test.
https://arxiv.org/abs/2305.09565
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING

import narwhals.stable.v1 as nw
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from pathmc.graph import GraphInfo
from pathmc.reprs import ResultReprMixin

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["FalsificationResult", "falsify_graph"]

# Graphs with at most this many nodes have few enough distinct node
# relabelings (7! = 5040) to enumerate exactly when requested. Larger
# graphs always use random sampling, since n! grows explosively and a
# full enumeration would exhaust time and memory.
_MAX_EXACT_NODES = 7

# Hard upper bound on an *explicit* sampled-permutation baseline for larger
# graphs. A request above this almost certainly reflects a mistake (e.g. a
# typo) and would run for an impractically long time, so it is rejected
# rather than allowed to hang.
_MAX_SAMPLED_PERMUTATIONS = 1_000_000

# Cap applied when n_permutations is *defaulted* from a tiny
# significance_level (round(1 / significance_level) can be enormous). We
# never auto-run more than this; users wanting finer resolution set
# n_permutations explicitly. Matches dowhy's practical guidance of a few
# hundred to a few thousand permutations.
_DEFAULT_PERMUTATION_CAP = 1_000


@dataclass(repr=False)
class FalsificationResult(ResultReprMixin):
    """Result of a permutation-based DAG falsification test.

    Produced by :func:`falsify_graph`. The verdict follows Eulig et al.
    (2023): the DAG is *falsifiable* (informative) when few node
    permutations share its Markov equivalence class, and *falsified*
    (rejected) when, despite being informative, it does not violate
    fewer Local Markov Conditions than the permuted baseline.

    Parameters
    ----------
    given_lmc_violations : int
        Number of Local Markov Condition violations of the given DAG.
    n_lmc_tests : int
        Number of LMC (parental conditional independence) tests run on
        the given DAG. Tests requiring unavailable data are not counted.
    given_lmc_violation_fraction : float
        ``given_lmc_violations / n_lmc_tests`` (0 if no tests ran).
    perm_lmc_violation_fractions : np.ndarray
        Fraction of LMC violations for each permuted DAG, shape
        ``(n_permutations,)``.
    perm_tpa_violation_fractions : np.ndarray
        Fraction of parental d-separation (tPA) violations for each
        permuted DAG relative to the given DAG, shape
        ``(n_permutations,)``.
    p_value_lmc : float
        Fraction of permutations whose LMC violation fraction is less
        than or equal to the given DAG's. Small values mean the DAG
        beats the random baseline.
    p_value_tpa : float
        Fraction of permutations lying in the Markov equivalence class
        of the given DAG (zero tPA violations). Small values mean the
        DAG is informative / falsifiable.
    n_permutations : int
        Number of permuted DAGs evaluated.
    n_in_mec : int
        Number of permutations sharing the given DAG's Markov equivalence
        class.
    significance_level : float
        Significance level for the permutation-based verdict.
    significance_ci : float
        Significance level used for each conditional independence test.
    local_violations : pd.DataFrame
        One row per LMC test on the given DAG with columns ``node``,
        ``non_descendant``, ``conditioning_set``, ``p_value``, and
        ``violation``.
    """

    given_lmc_violations: int
    n_lmc_tests: int
    given_lmc_violation_fraction: float
    perm_lmc_violation_fractions: np.ndarray
    perm_tpa_violation_fractions: np.ndarray
    p_value_lmc: float
    p_value_tpa: float
    n_permutations: int
    n_in_mec: int
    significance_level: float
    significance_ci: float
    local_violations: pd.DataFrame

    @property
    def can_evaluate(self) -> bool:
        """Whether the verdict is well-defined.

        ``False`` when the DAG implies no testable parental conditional
        independences (e.g. a fully connected graph), in which case both
        :attr:`falsifiable` and :attr:`falsified` are ``None``.
        """
        return self.n_lmc_tests > 0 and self.n_permutations > 0

    @property
    def falsifiable(self) -> bool | None:
        """Whether the DAG is informative enough to be falsified.

        ``True`` when the fraction of permutations in the given DAG's
        Markov equivalence class is at most :attr:`significance_level`.
        ``None`` when :attr:`can_evaluate` is ``False``.
        """
        if not self.can_evaluate:
            return None
        return self.p_value_tpa <= self.significance_level

    @property
    def falsified(self) -> bool | None:
        """Whether the data falsify (reject) the DAG.

        ``True`` only when the DAG is strictly informative
        (``p_value_tpa < significance_level``) and its LMC violations are
        *not* clearly better than the permuted baseline (``p_value_lmc``
        exceeds :attr:`significance_level`). ``None`` when
        :attr:`can_evaluate` is ``False``.

        The strict ``<`` on the informativeness side mirrors Eulig et al.
        (2023) / dowhy, where a DAG sitting exactly at the boundary
        (``p_value_tpa == significance_level``) is not rejected.
        """
        if not self.can_evaluate:
            return None
        return bool(
            self.p_value_lmc > self.significance_level
            and self.p_value_tpa < self.significance_level
        )

    @property
    def violations(self) -> pd.DataFrame:
        """Subset of ``local_violations`` where the LMC test was violated."""
        if "violation" not in self.local_violations.columns:
            return self.local_violations.copy()
        mask = self.local_violations["violation"].astype(bool)
        return self.local_violations[mask].copy()

    def _summary_lines(self) -> list[str]:
        if not self.can_evaluate:
            return [
                "DAG falsification: cannot be evaluated.",
                "The DAG implies no testable parental conditional "
                "independences (it may be fully connected). Remove an edge "
                "to create a falsifiable prediction.",
            ]
        informative = "" if self.falsifiable else " not"
        decision = "reject" if self.falsified else "do not reject"
        beats = (1.0 - self.p_value_lmc) * 100.0
        return [
            f"DAG falsification (significance level = {self.significance_level}):",
            f"  The DAG is{informative} informative: {self.n_in_mec} / "
            f"{self.n_permutations} permutations lie in its Markov "
            f"equivalence class (p = {self.p_value_tpa:.3f}).",
            f"  The DAG violates {self.given_lmc_violations}/{self.n_lmc_tests} "
            f"LMCs and beats {beats:.1f}% of the permuted DAGs "
            f"(p = {self.p_value_lmc:.3f}).",
            f"  Verdict: we {decision} the DAG.",
        ]

    def _repr_compact(self) -> str:
        if not self.can_evaluate:
            return f"FalsificationResult(not_evaluable, {self.n_permutations} permutations)"
        if self.falsified:
            verdict = "falsified"
        elif not self.falsifiable:
            verdict = "not_falsifiable"
        else:
            verdict = "not_rejected"
        return (
            f"FalsificationResult({verdict}, "
            f"p_lmc={self.p_value_lmc:.3f}, "
            f"{self.n_permutations} permutations)"
        )

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if not self.can_evaluate:
            return (
                "<h4>DAG Falsification</h4>"
                '<p><span style="color: gray;">Cannot be evaluated — the DAG '
                "implies no testable parental conditional independences.</span></p>"
            )

        if self.falsified:
            verdict = (
                '<span style="color: red; font-weight: bold;">✗ Rejected</span> '
                "— the data falsify this DAG."
            )
        elif not self.falsifiable:
            verdict = (
                '<span style="color: orange; font-weight: bold;">⚠ Not '
                "informative</span> — too many permutations share its Markov "
                "equivalence class to falsify it."
            )
        else:
            verdict = (
                '<span style="color: green; font-weight: bold;">✓ Not '
                "rejected</span> — the DAG is informative and beats the "
                "permuted baseline."
            )

        beats = (1.0 - self.p_value_lmc) * 100.0
        rows = [
            ("Informative (falsifiable)", "Yes" if self.falsifiable else "No"),
            (
                "Permutations in Markov equivalence class",
                f"{self.n_in_mec} / {self.n_permutations} (p = {self.p_value_tpa:.3f})",
            ),
            (
                "LMC violations (given DAG)",
                f"{self.given_lmc_violations} / {self.n_lmc_tests}",
            ),
            ("Beats permuted baseline", f"{beats:.1f}% (p = {self.p_value_lmc:.3f})"),
        ]
        body = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
        )
        return (
            f"<h4>DAG Falsification (α = {self.significance_level})</h4>"
            f"<p>{verdict}</p>"
            f"<table><tbody>{body}</tbody></table>"
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        bins: int | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot histograms of permuted-baseline violation fractions.

        Shows the distribution of LMC violation fractions (blue) and tPA
        d-separation violation fractions (orange) across permuted DAGs,
        with dashed vertical lines marking the given DAG's values. A given
        DAG far to the left of the LMC histogram beats the baseline.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. Creates a new figure if ``None``.
        bins : int | None
            Number of histogram bins. Defaults to an automatic choice.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the histogram.

        Raises
        ------
        RuntimeError
            If the result cannot be evaluated (no LMC tests).
        """
        import matplotlib.pyplot as plt

        if not self.can_evaluate:
            raise RuntimeError(
                "Cannot plot a falsification result that cannot be evaluated. "
                "The DAG implies no testable conditional independences."
            )

        if bins is not None and bins < 1:
            raise ValueError(f"bins must be a positive integer or None, got {bins}.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            from typing import cast

            fig = cast("matplotlib.figure.Figure", ax.get_figure())

        ax.hist(
            [self.perm_lmc_violation_fractions, self.perm_tpa_violation_fractions],
            bins=bins,
            alpha=0.5,
            color=["tab:blue", "tab:orange"],
            edgecolor="k",
            label=[
                "LMC violations (permuted DAGs)",
                "tPA d-sep violations (permuted DAGs)",
            ],
        )
        ylim = ax.get_ylim()[1]
        ax.plot(
            [self.given_lmc_violation_fraction] * 2,
            [0, ylim],
            "--",
            color="tab:blue",
            label="LMC violations (given DAG)",
        )
        ax.plot(
            [0.0, 0.0],
            [0, ylim],
            "--",
            color="tab:orange",
            label="tPA violations (given DAG)",
        )
        ax.set_ylim((0, ylim))
        ax.set_xlabel("Fraction of violations")
        ax.set_ylabel("# permuted DAGs")
        ax.set_title(
            f"DAG falsification "
            f"(p_LMC = {self.p_value_lmc:.3f}, p_tPA = {self.p_value_tpa:.3f})"
        )
        ax.legend(loc="best", fontsize="small")
        return fig


class _PartialCorrelationTester:
    """Memoized partial-correlation conditional independence tester.

    Holds a numeric matrix extracted once from the data and caches each
    ``X ⊥⊥ Y | Z`` p-value. Independence is symmetric in ``X`` and ``Y``,
    so the cache key normalizes their order, mirroring dowhy's
    ``_PValuesMemory`` and avoiding recomputation across the many
    permuted graphs that re-test the same triples.
    """

    def __init__(self, data: nw.DataFrame, variables: list[str]) -> None:
        columns = set(data.columns)
        numeric: list[str] = []
        arrays: list[np.ndarray] = []
        for var in variables:
            if var not in columns:
                continue
            try:
                col = data.select(var).to_numpy().astype(float).ravel()
            except (ValueError, TypeError):
                # Non-numeric (string/categorical) columns cannot enter a
                # partial-correlation test; treat them as unavailable so the
                # affected triples are skipped, exactly like a missing column.
                continue
            numeric.append(var)
            arrays.append(col)
        if arrays:
            matrix = np.column_stack(arrays)
        else:
            matrix = np.empty((0, 0), dtype=float)
        self._matrix = matrix
        self._col_idx = {name: i for i, name in enumerate(numeric)}
        self._cache: dict[tuple[frozenset[str], frozenset[str]], float | None] = {}

    def p_value(self, x: str, y: str, z_vars: tuple[str, ...]) -> float | None:
        """Return the CI test p-value, or ``None`` if it cannot be run.

        ``None`` signals a skipped test: a required variable has no data
        column, or there are too few complete observations.
        """
        key = (frozenset((x, y)), frozenset(z_vars))
        if key in self._cache:
            return self._cache[key]
        result = self._compute(x, y, z_vars)
        self._cache[key] = result
        return result

    def _compute(self, x: str, y: str, z_vars: tuple[str, ...]) -> float | None:
        needed = [x, y, *z_vars]
        if any(v not in self._col_idx for v in needed):
            return None

        cols = [self._col_idx[v] for v in needed]
        arr = self._matrix[:, cols]
        arr = arr[~np.isnan(arr).any(axis=1)]
        n = arr.shape[0]
        k = len(z_vars)
        if n < 3:
            return None

        x_vals = arr[:, 0]
        y_vals = arr[:, 1]

        if k == 0:
            if np.std(x_vals) == 0.0 or np.std(y_vals) == 0.0:
                return None
            r, p = stats.pearsonr(x_vals, y_vals)
            return float(p)

        z_with_intercept = np.column_stack([np.ones(n), arr[:, 2:]])
        # Effective rank handles constant or collinear conditioning columns:
        # a constant conditioner collapses into the intercept, so the test
        # reduces to the marginal one and the degrees of freedom must reflect
        # the true number of independent predictors, not len(z_vars).
        rank = int(np.linalg.matrix_rank(z_with_intercept))
        df = n - rank - 1
        if df <= 0:
            return None

        beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x_vals, rcond=None)
        resid_x = x_vals - z_with_intercept @ beta_x
        beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y_vals, rcond=None)
        resid_y = y_vals - z_with_intercept @ beta_y

        if np.std(resid_x) == 0.0 or np.std(resid_y) == 0.0:
            return None

        r = float(np.corrcoef(resid_x, resid_y)[0, 1])
        if np.isnan(r):
            return None
        if r * r >= 1.0:
            return 0.0

        t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r * r)
        return float(2.0 * stats.t.sf(np.abs(t_stat), df))


def _parental_triples(
    dag: nx.DiGraph,
    include_unconditional: bool,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Enumerate ``(node, non_descendant, parents)`` triples for LMC tests.

    For each node ``X`` with parents ``Z = pa(X)``, every non-descendant
    ``Y`` (excluding ``X`` itself and its parents) yields the testable
    implication ``X ⊥⊥ Y | Z``. Root nodes (no parents) are included only
    when ``include_unconditional`` is ``True``.
    """
    triples: list[tuple[str, str, tuple[str, ...]]] = []
    all_nodes = set(dag.nodes)
    for node in dag.nodes:
        parents = tuple(sorted(dag.predecessors(node)))
        excluded = nx.descendants(dag, node) | {node} | set(parents)
        non_descendants = sorted(all_nodes - excluded)
        if (parents or include_unconditional) and non_descendants:
            for non_desc in non_descendants:
                triples.append((node, non_desc, parents))
    return triples


def _validate_lmc(
    dag: nx.DiGraph,
    tester: _PartialCorrelationTester,
    significance_ci: float,
    include_unconditional: bool,
) -> tuple[int, int, list[dict]]:
    """Count Local Markov Condition violations of *dag* against the data.

    Returns ``(n_tests, n_violations, local)`` where ``local`` holds one
    record per executed test. Triples whose CI test cannot be run (missing
    data, too few observations) are skipped and not counted.
    """
    n_tests = 0
    n_violations = 0
    local: list[dict] = []
    for node, non_desc, parents in _parental_triples(dag, include_unconditional):
        p_value = tester.p_value(node, non_desc, parents)
        if p_value is None:
            continue
        n_tests += 1
        violation = p_value <= significance_ci
        if violation:
            n_violations += 1
        local.append({
            "node": node,
            "non_descendant": non_desc,
            "conditioning_set": ", ".join(parents),
            "p_value": p_value,
            "violation": violation,
        })
    return n_tests, n_violations, local


def _validate_tpa(
    permuted_dag: nx.DiGraph,
    reference_dag: nx.DiGraph,
    include_unconditional: bool,
) -> tuple[int, int]:
    """Count parental d-separations of *permuted_dag* violated in the reference.

    For each parental triple ``(X, Y, Z)`` implied by *permuted_dag*,
    checks whether ``X ⊥⊥ Y | Z`` holds (d-separation) in *reference_dag*.
    Zero violations means the two graphs share a Markov equivalence class.
    Returns ``(n_tests, n_violations)``.
    """
    n_tests = 0
    n_violations = 0
    for node, non_desc, parents in _parental_triples(
        permuted_dag, include_unconditional
    ):
        n_tests += 1
        if not nx.is_d_separator(reference_dag, {node}, {non_desc}, set(parents)):
            n_violations += 1
    return n_tests, n_violations


def _permuted_dags(
    dag: nx.DiGraph,
    n_permutations: int,
    rng: np.random.Generator,
):
    """Yield node-relabeled copies of *dag* preserving its structure.

    For small graphs (at most :data:`_MAX_EXACT_NODES` nodes), when
    *n_permutations* covers every distinct relabeling, all of them are
    enumerated exactly and deterministically. Otherwise *n_permutations*
    random relabelings are drawn (sampling with replacement). The identity
    relabeling is allowed, matching dowhy's baseline.
    """
    nodes = list(dag.nodes)
    n = len(nodes)
    max_perms = math.factorial(n) if n <= _MAX_EXACT_NODES else None

    if max_perms is not None and n_permutations >= max_perms:
        for exact_ordering in permutations(nodes):
            mapping = {nodes[i]: exact_ordering[i] for i in range(n)}
            yield nx.relabel_nodes(dag, mapping, copy=True)
        return

    node_array = np.asarray(nodes, dtype=object)
    for _ in range(n_permutations):
        random_ordering = list(rng.permutation(node_array))
        mapping = {nodes[i]: random_ordering[i] for i in range(n)}
        yield nx.relabel_nodes(dag, mapping, copy=True)


def falsify_graph(
    graph_info: GraphInfo,
    data: nw.DataFrame,
    *,
    n_permutations: int | None = None,
    significance_level: float = 0.05,
    significance_ci: float = 0.05,
    include_unconditional: bool = True,
    random_seed: int | None = None,
) -> FalsificationResult:
    """Falsify a whole DAG against data via a node-permutation test.

    Implements the permutation test of Eulig et al. (2023). The given
    DAG's count of Local Markov Condition (LMC) violations is compared to
    a baseline of randomly relabeled competitor graphs. The DAG is
    *informative* when few permutations share its Markov equivalence
    class. An informative DAG that violates fewer LMCs than the permuted
    baseline is the positive case: *not contradicted and testable*. A
    non-informative DAG is also *not rejected*, but that verdict is
    vacuous — interpret it as *not falsifiable* rather than as evidence
    for the graph.

    Conditional independence is tested with partial correlation, the same
    linear-Gaussian methodology used by
    :func:`pathmc.identify.test_implications`. Because the test is linear,
    purely nonlinear dependencies are not detected, so a "not rejected"
    verdict is only as strong as the linear-Gaussian assumption. The test
    uses observed data directly and works before sampling.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model. Only contemporaneous (non-temporal)
        directed edges are used; temporal ``lag(...)`` terms appear as
        ordinary contemporaneous nodes, so falsification targets
        cross-sectional (observed-variable) structure. Residual
        covariances (``~~``) are not supported and raise ``ValueError``.
    data : nw.DataFrame
        Observed data. Variables without a usable numeric column (latent
        nodes, or non-numeric columns) are skipped in CI tests but still
        participate in the d-separation oracle and node permutations.
    n_permutations : int | None
        Number of permuted DAGs in the baseline. Defaults to
        ``round(1 / significance_level)`` (20 at the default level). For
        small graphs (at most 7 nodes), if this meets or exceeds the
        number of distinct node relabelings (``n!``), all of them are
        enumerated exactly; otherwise random relabelings are sampled.
    significance_level : float
        Significance level for the permutation-based verdict
        (default 0.05).
    significance_ci : float
        Significance level for each conditional independence test
        (default 0.05).
    include_unconditional : bool
        Whether to also test the unconditional independences implied by
        root nodes (default ``True``).
    random_seed : int | None
        Seed for the permutation sampler, for reproducible results.

    Returns
    -------
    FalsificationResult
        Verdict (``.falsified``, ``.falsifiable``), permutation p-values,
        per-test local violations, and a ``.plot()`` helper.

    Raises
    ------
    ValueError
        If *significance_level* or *significance_ci* is not in ``(0, 1)``;
        if *n_permutations* is not a positive integer (or, for a large
        graph, exceeds the sampling cap); or if the model declares a
        residual covariance (``~~``), which is unsupported.
    """
    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            f"significance_level must be in (0, 1), got {significance_level}. "
            f"Use a value such as 0.05."
        )
    if not 0.0 < significance_ci < 1.0:
        raise ValueError(
            f"significance_ci must be in (0, 1), got {significance_ci}. "
            f"Use a value such as 0.05."
        )

    user_set_permutations = n_permutations is not None
    if n_permutations is None:
        # Cap the defaulted count so a tiny significance_level cannot imply
        # an astronomically large (and slow) auto-run.
        n_permutations = min(
            int(round(1.0 / significance_level)), _DEFAULT_PERMUTATION_CAP
        )
    if not isinstance(n_permutations, (int, np.integer)) or isinstance(
        n_permutations, bool
    ):
        raise ValueError(
            f"n_permutations must be a positive integer or None, got "
            f"{n_permutations!r} of type {type(n_permutations).__name__}."
        )
    if n_permutations < 1:
        raise ValueError(
            f"n_permutations must be a positive integer, got {n_permutations}."
        )

    if graph_info.residual_blocks:
        raise ValueError(
            "falsify_graph does not support residual covariances (~~). "
            "These encode unobserved confounding (a bidirected/ADMG edge), "
            "which the permutation test — like dowhy's gcm.falsify_graph — "
            "does not model. Falsify the directed structure by building a "
            "model without the ~~ terms, or use test_implications() / "
            "sensitivity() to reason about the confounded pairs."
        )

    dag = graph_info.contemporaneous_dag

    if (
        user_set_permutations
        and dag.number_of_nodes() > _MAX_EXACT_NODES
        and n_permutations > _MAX_SAMPLED_PERMUTATIONS
    ):
        raise ValueError(
            f"n_permutations={n_permutations} is too large for a graph "
            f"with {dag.number_of_nodes()} nodes (the baseline is "
            f"sampled, not enumerated). Use at most "
            f"{_MAX_SAMPLED_PERMUTATIONS}; a few hundred to a few "
            f"thousand permutations is typically plenty."
        )

    nodes = sorted(dag.nodes)
    rng = np.random.default_rng(random_seed)
    tester = _PartialCorrelationTester(data, nodes)

    given_n_tests, given_violations, local = _validate_lmc(
        dag, tester, significance_ci, include_unconditional
    )
    given_fraction = given_violations / given_n_tests if given_n_tests else 0.0

    perm_lmc_fractions: list[float] = []
    perm_tpa_fractions: list[float] = []
    n_in_mec = 0

    for permuted in _permuted_dags(dag, n_permutations, rng):
        lmc_tests, lmc_viol, _ = _validate_lmc(
            permuted, tester, significance_ci, include_unconditional
        )
        tpa_tests, tpa_viol = _validate_tpa(permuted, dag, include_unconditional)
        perm_lmc_fractions.append(lmc_viol / lmc_tests if lmc_tests else 0.0)
        perm_tpa_fractions.append(tpa_viol / tpa_tests if tpa_tests else 0.0)
        if tpa_viol == 0:
            n_in_mec += 1

    perm_lmc = np.asarray(perm_lmc_fractions, dtype=float)
    perm_tpa = np.asarray(perm_tpa_fractions, dtype=float)
    n_evaluated = len(perm_lmc)

    if n_evaluated > 0:
        p_value_lmc = float(np.mean(perm_lmc <= given_fraction))
        p_value_tpa = float(np.mean(perm_tpa <= 0.0))
    else:
        p_value_lmc = 1.0
        p_value_tpa = 1.0

    local_df = (
        pd.DataFrame(local)
        if local
        else pd.DataFrame(
            columns=[
                "node",
                "non_descendant",
                "conditioning_set",
                "p_value",
                "violation",
            ]
        )
    )

    return FalsificationResult(
        given_lmc_violations=given_violations,
        n_lmc_tests=given_n_tests,
        given_lmc_violation_fraction=given_fraction,
        perm_lmc_violation_fractions=perm_lmc,
        perm_tpa_violation_fractions=perm_tpa,
        p_value_lmc=p_value_lmc,
        p_value_tpa=p_value_tpa,
        n_permutations=n_evaluated,
        n_in_mec=n_in_mec,
        significance_level=significance_level,
        significance_ci=significance_ci,
        local_violations=local_df,
    )
