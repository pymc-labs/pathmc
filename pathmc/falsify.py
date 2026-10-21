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
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Any

import narwhals.stable.v1 as nw
import networkx as nx
import numpy as np
import pandas as pd

from pathmc._ci import _PartialCorrelationTester
from pathmc._ci_report import (
    collapse_statement,
    n_rank_collapsed,
    warn_conditioning_rank_collapse,
    written_counts,
)
from pathmc._types import BinsLike, validate_bins
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
        Local Markov Condition violations of the independence that was
        written down. A rank-collapsed row is not counted.
    n_lmc_tests : int
        Parental tests that actually tested the written parent set.
        Rank-collapsed rows and tests that could not be run are excluded.
        Zero means the DAG cannot be evaluated on the requested
        independences.
    given_lmc_violation_fraction : float
        ``given_lmc_violations / n_lmc_tests`` (0 if no written tests ran).
    perm_lmc_violation_fractions : np.ndarray
        Written-test LMC violation fraction for each permuted DAG that
        produced a written parental test. A permutation whose tests all
        rank-collapsed is omitted, not stored as 0. The length can be
        shorter than ``n_permutations``.
    perm_tpa_violation_fractions : np.ndarray
        Fraction of parental d-separation (tPA) violations for each
        permuted DAG relative to the given DAG, shape
        ``(n_permutations,)``.
    p_value_lmc : float
        Fraction of the scored permutations whose written-test violation
        fraction is at most the given DAG's. Both sides exclude
        rank-collapsed rows, and a permutation with no written parental
        test is not in this fraction. Small values mean the DAG beats
        the random baseline on the independences that were actually
        tested. NaN when permutations ran but none produced a written
        parental test.
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
        ``non_descendant``, ``conditioning_set``, ``requested_k``,
        ``effective_k``, ``rank_collapsed``, ``df``, ``p_value``, and
        ``violation``. ``requested_k`` is the number of conditioning
        columns that entered the test. ``effective_k`` is
        ``rank([1, Z]) - 1``. ``rank_collapsed`` is true when that
        executed test used fewer columns than requested, or a named
        conditioner contributed no columns. A collapsed row
        stays in this frame for inspection, but it is excluded from
        ``given_lmc_violations``, ``n_lmc_tests``, ``violations``, and
        the permutation fraction. A permuted DAG that produced no
        written parental test is omitted from that fraction, not scored
        as zero violations. If every drawn permutation is omitted,
        :attr:`falsified` is unset.
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

        ``False`` when no written parental test ran: the DAG implies none,
        or every requested parent set rank-collapsed. Both
        :attr:`falsifiable` and :attr:`falsified` are then ``None``.
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
        :attr:`can_evaluate` is ``False``, and when permutations ran but
        none produced a written parental test. The latter is not a pass
        and not a fail of the LMC comparison.

        The strict ``<`` on the informativeness side mirrors Eulig et al.
        (2023) / dowhy, where a DAG sitting exactly at the boundary
        (``p_value_tpa == significance_level``) is not rejected.
        """
        if not self.can_evaluate or self._lmc_not_compared():
            return None
        return bool(
            self.p_value_lmc > self.significance_level
            and self.p_value_tpa < self.significance_level
        )

    @property
    def violations(self) -> pd.DataFrame:
        """Violated rows that tested the parent set that was written down."""
        frame = self.local_violations
        if "violation" not in frame.columns:
            return frame.copy()
        mask = frame["violation"].astype(bool)
        if "rank_collapsed" in frame.columns:
            mask &= ~frame["rank_collapsed"].fillna(False).astype(bool)
        return frame.loc[mask].copy()

    def _only_rank_collapsed(self) -> bool:
        """Whether every executed parental test collapsed.

        ``can_evaluate`` is then false, but the DAG did imply parental
        tests. Those rows are not a saturated graph and not violations.
        """
        return self.n_lmc_tests == 0 and n_rank_collapsed(self.local_violations) > 0

    def _lmc_not_compared(self) -> bool:
        """Whether permutations ran but none produced a written test.

        Those draws are omitted from the LMC fraction. Publishing a
        rejection decision would treat "not tested" as a pass or a fail.
        """
        return self.n_permutations > 0 and len(self.perm_lmc_violation_fractions) == 0

    def _summary_lines(self) -> list[str]:
        n_collapsed = n_rank_collapsed(self.local_violations)
        if not self.can_evaluate:
            if self._only_rank_collapsed():
                return [
                    "DAG falsification: cannot be evaluated.",
                    "Every requested parental independence rank-collapsed, "
                    "so no written parent set was actually tested. Those "
                    "rows are not violations. Drop the redundant parent "
                    "and re-test.",
                ]
            return [
                "DAG falsification: cannot be evaluated.",
                "The DAG implies no testable parental conditional "
                "independences (it may be fully connected). Remove an edge "
                "to create a falsifiable prediction.",
            ]
        informative = "" if self.falsifiable else " not"
        decision = "reject" if self.falsified else "do not reject"
        n_scored = len(self.perm_lmc_violation_fractions)
        omitted = self.n_permutations - n_scored
        if n_scored == 0:
            comparison = (
                "No permuted DAG produced a written parental test, so the "
                "permutation fraction was not computed."
            )
        elif omitted:
            beats = (1.0 - self.p_value_lmc) * 100.0
            comparison = (
                f"The permutation fraction on those requested tests beats "
                f"{beats:.1f}% of the {n_scored} permuted DAGs that produced "
                f"a written-test fraction (p = {self.p_value_lmc:.3f}). "
                f"{omitted} permuted DAGs produced no written parental test "
                f"and are not in that fraction."
            )
        else:
            beats = (1.0 - self.p_value_lmc) * 100.0
            comparison = (
                f"beats {beats:.1f}% of the permuted DAGs (p = {self.p_value_lmc:.3f})."
            )
        lines = [
            f"DAG falsification (significance level = {self.significance_level}):",
            f"  The DAG is{informative} informative: {self.n_in_mec} / "
            f"{self.n_permutations} permutations lie in its Markov "
            f"equivalence class (p = {self.p_value_tpa:.3f}).",
        ]
        if n_collapsed:
            noun = "test" if n_collapsed == 1 else "tests"
            if n_scored == 0 or omitted:
                lines.append(
                    f"  The DAG violates {self.given_lmc_violations} of "
                    f"{self.n_lmc_tests} requested parental independences; "
                    f"{n_collapsed} {noun} rank-collapsed. {comparison}"
                )
            else:
                lines.append(
                    f"  The DAG violates {self.given_lmc_violations} of "
                    f"{self.n_lmc_tests} requested parental independences; "
                    f"{n_collapsed} {noun} rank-collapsed. The permutation "
                    f"fraction on those requested tests {comparison}"
                )
            lines.append(
                f"  {n_collapsed} parental {noun} rank-collapsed. Those "
                f"p-values are not the requested parental independences."
            )
            if self._lmc_not_compared():
                lines.append("  Verdict: the LMC comparison was not run.")
            else:
                lines.append(
                    f"  Verdict: we {decision} the DAG on the tests that ran; "
                    "rank-collapsed rows did not test the requested parent sets."
                )
        else:
            if n_scored == 0 or omitted:
                lines.append(
                    f"  The DAG violates {self.given_lmc_violations}/"
                    f"{self.n_lmc_tests} LMCs. {comparison}"
                )
            else:
                lines.append(
                    f"  The DAG violates {self.given_lmc_violations}/"
                    f"{self.n_lmc_tests} LMCs and {comparison}"
                )
            if self._lmc_not_compared():
                lines.append("  Verdict: the LMC comparison was not run.")
            else:
                lines.append(f"  Verdict: we {decision} the DAG.")
        return lines

    def _repr_compact(self) -> str:
        if not self.can_evaluate:
            collapsed = ", rank-collapsed" if self._only_rank_collapsed() else ""
            return (
                f"FalsificationResult(not_evaluable{collapsed}, "
                f"{self.n_permutations} permutations)"
            )
        if self._lmc_not_compared():
            verdict = "not_compared"
        elif self.falsified:
            verdict = "falsified"
        elif not self.falsifiable:
            verdict = "not_falsifiable"
        else:
            verdict = "not_rejected"
        collapse = ", rank-collapsed" if n_rank_collapsed(self.local_violations) else ""
        p_lmc = (
            "n/a"
            if len(self.perm_lmc_violation_fractions) == 0
            else f"{self.p_value_lmc:.3f}"
        )
        return (
            f"FalsificationResult({verdict}{collapse}, "
            f"p_lmc={p_lmc}, "
            f"{self.n_permutations} permutations)"
        )

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if not self.can_evaluate:
            if self._only_rank_collapsed():
                return (
                    "<h4>DAG Falsification</h4>"
                    '<p><span style="color: gray;">Cannot be evaluated — '
                    "every requested parental independence rank-collapsed, "
                    "so no written parent set was actually tested. Those "
                    "rows are not violations. Drop the redundant parent "
                    "and re-test.</span></p>"
                )
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
        elif self._lmc_not_compared():
            verdict = (
                '<span style="color: orange; font-weight: bold;">⚠ Not '
                "compared</span> — no permuted DAG produced a written "
                "parental test, so the LMC comparison was not run."
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
        n_collapsed = n_rank_collapsed(self.local_violations)
        if n_collapsed:
            noun = "test" if n_collapsed == 1 else "tests"
            verdict += (
                f" {n_collapsed} rank-collapsed {noun} did not test the "
                f"requested parent sets."
            )

        n_scored = len(self.perm_lmc_violation_fractions)
        omitted = self.n_permutations - n_scored
        if n_scored == 0:
            beats_cell = "not computed"
        elif omitted:
            beats = (1.0 - self.p_value_lmc) * 100.0
            beats_cell = (
                f"{beats:.1f}% of {n_scored} scored "
                f"(p = {self.p_value_lmc:.3f}; {omitted} omitted)"
            )
        else:
            beats = (1.0 - self.p_value_lmc) * 100.0
            beats_cell = f"{beats:.1f}% (p = {self.p_value_lmc:.3f})"
        if n_collapsed:
            lmc = (
                f"{self.given_lmc_violations} / {self.n_lmc_tests} requested "
                f"({n_collapsed} rank-collapsed)"
            )
        else:
            lmc = f"{self.given_lmc_violations} / {self.n_lmc_tests}"
        rows = [
            ("Informative (falsifiable)", "Yes" if self.falsifiable else "No"),
            (
                "Permutations in Markov equivalence class",
                f"{self.n_in_mec} / {self.n_permutations} (p = {self.p_value_tpa:.3f})",
            ),
            ("LMC violations (given DAG)", lmc),
            ("Beats permuted baseline", beats_cell),
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
        bins: BinsLike | None = None,
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
        bins : BinsLike | None
            Passed through to ``matplotlib.axes.Axes.hist``: a positive
            integer bin count, a binning strategy name such as
            ``"auto"``, or a sequence of bin edges. Defaults to an
            automatic choice.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the histogram.

        Raises
        ------
        RuntimeError
            If the result cannot be evaluated: no written parental test
            ran, either because the DAG implies none or because every
            requested parent set rank-collapsed.
        ValueError
            If *bins* is neither one of the accepted types nor, for an
            integer bin count, positive. Strategy names and bin edges are
            validated by matplotlib, which reports them more precisely.
        """
        import matplotlib.pyplot as plt

        if not self.can_evaluate:
            if self._only_rank_collapsed():
                raise RuntimeError(
                    "Cannot plot a falsification result that cannot be "
                    "evaluated. Every requested parental independence "
                    "rank-collapsed, so no written parent set was actually "
                    "tested."
                )
            raise RuntimeError(
                "Cannot plot a falsification result that cannot be evaluated. "
                "The DAG implies no testable conditional independences."
            )

        hist_bins: Any = validate_bins(bins)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            from typing import cast

            fig = cast("matplotlib.figure.Figure", ax.get_figure())

        lmc_fractions = self.perm_lmc_violation_fractions
        plot_bins = (
            hist_bins.tolist() if isinstance(hist_bins, np.ndarray) else hist_bins
        )
        if len(lmc_fractions) == 0:
            ax.hist(
                self.perm_tpa_violation_fractions,
                bins=plot_bins,
                alpha=0.5,
                color="tab:orange",
                edgecolor="k",
                label="tPA d-sep violations (permuted DAGs)",
            )
        else:
            ax.hist(
                [lmc_fractions, self.perm_tpa_violation_fractions],
                bins=plot_bins,
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
        p_lmc = "n/a" if len(lmc_fractions) == 0 else f"{self.p_value_lmc:.3f}"
        ax.set_title(
            f"DAG falsification (p_LMC = {p_lmc}, p_tPA = {self.p_value_tpa:.3f})"
        )
        ax.legend(loc="best", fontsize="small")
        return fig


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
    """Count written Local Markov Condition violations of *dag*.

    Returns ``(n_tests, n_violations, local)`` where ``local`` holds one
    record per executed test, including rank-collapsed rows. The returned
    counts exclude those rows: a collapsed parent set did not test the
    independence that was written down. Triples whose CI test cannot be
    run are skipped and not counted.
    """
    local: list[dict] = []
    for node, non_desc, parents in _parental_triples(dag, include_unconditional):
        result = tester.ci_result(node, non_desc, parents)
        if result is None or result.skip_reason is not None or result.p is None:
            continue
        dropped = tester.dropped_conditioners(parents)
        local.append({
            "node": node,
            "non_descendant": non_desc,
            "conditioning_set": ", ".join(parents),
            "requested_k": result.requested_k,
            "effective_k": result.effective_k,
            "rank_collapsed": result.rank_collapsed or bool(dropped),
            "df": result.df,
            "p_value": result.p,
            "violation": result.p <= significance_ci,
        })
    frame = pd.DataFrame(local) if local else pd.DataFrame(columns=_LOCAL_COLUMNS)
    n_tests, n_violations = written_counts(frame, "violation")
    return n_tests, n_violations, local


def _falsify_collapse_labels(
    frame: pd.DataFrame,
    tester: _PartialCorrelationTester,
) -> list[str]:
    """Labels for executed LMC rows whose parent set collapsed."""
    if frame.empty or "rank_collapsed" not in frame.columns:
        return []
    collapsed = frame.loc[frame["rank_collapsed"].fillna(False).astype(bool)]
    labels: list[str] = []
    for _, row in collapsed.iterrows():
        conditioning = str(row["conditioning_set"])
        names = [part for part in conditioning.split(", ") if part]
        labels.append(
            collapse_statement(
                str(row["node"]),
                str(row["non_descendant"]),
                conditioning,
                int(row["requested_k"]),
                int(row["effective_k"]),
                tester.dropped_conditioners(names),
            )
        )
    return labels


_LOCAL_COLUMNS = [
    "node",
    "non_descendant",
    "conditioning_set",
    "requested_k",
    "effective_k",
    "rank_collapsed",
    "df",
    "p_value",
    "violation",
]


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
    categorical_vars: Iterable[str] | None = None,
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
    uses observed data directly and works before sampling. Each local
    violation row reports ``requested_k``, ``effective_k``,
    ``rank_collapsed``, and ``df`` from that test.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model. Only contemporaneous (non-temporal)
        directed edges are used; temporal ``lag(...)`` terms appear as
        ordinary contemporaneous nodes, so falsification targets
        cross-sectional (observed-variable) structure. Residual
        covariances (``~~``) are not supported and raise ``ValueError``.
    data : nw.DataFrame
        Observed data. Categorical columns (string, pandas categorical, or
        named in *categorical_vars*) enter the CI tests as indicator
        blocks. Variables without a data column (latent nodes) are skipped
        in CI tests but still participate in the d-separation oracle and
        node permutations.
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
    categorical_vars : Iterable[str] | None
        Variables to dummy-encode even though their column is numeric
        (integer-coded labels declared with ``C()``).

    Returns
    -------
    FalsificationResult
        Verdict (``.falsified``, ``.falsifiable``), permutation p-values,
        per-test local violations, and a ``.plot()`` helper.

    Warns
    -----
    UserWarning
        When an LMC test on the *given* DAG has ``rank_collapsed`` set.
        That includes a constant, duplicated, or collinear parent, and a
        named parent that contributed no columns. The test still runs on
        the columns that entered; that p-value is not evidence about the
        full parental independence. Permuted DAGs are not warned about
        individually. Drop the redundant parent and re-run.

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
    tester = _PartialCorrelationTester(data, nodes, categorical_vars)

    given_n_tests, given_violations, local = _validate_lmc(
        dag,
        tester,
        significance_ci,
        include_unconditional,
    )
    local_df = pd.DataFrame(local) if local else pd.DataFrame(columns=_LOCAL_COLUMNS)
    warn_conditioning_rank_collapse(_falsify_collapse_labels(local_df, tester))
    given_fraction = given_violations / given_n_tests if given_n_tests else 0.0

    perm_lmc_fractions: list[float] = []
    perm_tpa_fractions: list[float] = []
    n_in_mec = 0

    for permuted in _permuted_dags(dag, n_permutations, rng):
        lmc_tests, lmc_viol, _ = _validate_lmc(
            permuted, tester, significance_ci, include_unconditional
        )
        tpa_tests, tpa_viol = _validate_tpa(permuted, dag, include_unconditional)
        # An empty written set is not zero violations. Scoring it as 0
        # would publish "not tested" as a perfect LMC baseline.
        if lmc_tests:
            perm_lmc_fractions.append(lmc_viol / lmc_tests)
        perm_tpa_fractions.append(tpa_viol / tpa_tests if tpa_tests else 0.0)
        if tpa_viol == 0:
            n_in_mec += 1

    perm_lmc = np.asarray(perm_lmc_fractions, dtype=float)
    perm_tpa = np.asarray(perm_tpa_fractions, dtype=float)
    n_evaluated = len(perm_tpa)

    if n_evaluated == 0:
        p_value_lmc = 1.0
        p_value_tpa = 1.0
    else:
        p_value_tpa = float(np.mean(perm_tpa <= 0.0))
        if len(perm_lmc) == 0:
            p_value_lmc = float("nan")
        else:
            p_value_lmc = float(np.mean(perm_lmc <= given_fraction))

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
