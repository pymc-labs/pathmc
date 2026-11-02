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
"""Causal identification helpers for pathmc structural models.

Provides backdoor adjustment set computation, front-door criterion checks,
collider warnings, identifiability checks, and implied conditional
independence enumeration and testing using the DAG stored in GraphInfo.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import narwhals.stable.v1 as nw
import networkx as nx
import numpy as np
import pandas as pd

from pathmc._ci import named_column_ci
from pathmc._ci_report import (
    collapse_statement,
    n_rank_collapsed,
    rank_collapse_note,
    row_is_rank_collapsed,
    violation_rows,
    warn_conditioning_rank_collapse,
    written_counts,
)
from pathmc.graph import GraphInfo
from pathmc.reprs import ResultReprMixin

__all__ = ["ImplicationTestResult"]


def _require_nodes(dag: nx.DiGraph, **named: str) -> None:
    """Raise ``ValueError`` if any named node is absent from *dag*.

    Keyword keys are human-readable roles (e.g. ``treatment="X"``); the
    role is capitalized in the error message, which lists the available
    nodes.
    """
    for role, name in named.items():
        if name not in dag.nodes:
            raise ValueError(
                f"{role.capitalize()} '{name}' not in DAG. "
                f"Available nodes: {sorted(dag.nodes)}"
            )


def _with_residual_confounders(
    graph_info: GraphInfo,
) -> tuple[nx.DiGraph, set[str]]:
    """Return the contemporaneous DAG with ``~~`` blocks made explicit.

    A residual covariance block (``X ~~ Y``) declares that the two
    equations share an *unobserved* common cause. The DAG stored in
    :class:`~pathmc.graph.GraphInfo` has no node for that cause, so
    graphical criteria that only read directed edges are blind to it and
    will happily report an effect as identifiable when the user has
    explicitly declared endogeneity.

    This is the standard latent projection: each residual block gets one
    synthetic latent node with an edge to every block member. Because the
    node is latent it can never enter an adjustment set, so the backdoor
    path it opens is unblockable, which is the correct answer.

    Returns
    -------
    tuple[nx.DiGraph, set[str]]
        The augmented DAG and the latent set including the synthetic
        confounders.
    """
    dag = graph_info.contemporaneous_dag
    latent = set(graph_info.latent)

    if not graph_info.residual_blocks:
        return dag, latent

    dag = dag.copy()
    used_names = set(dag.nodes)
    for i, block in enumerate(graph_info.residual_blocks):
        if len(block) < 2:
            continue
        u = _fresh_latent_name(i, used_names)
        used_names.add(u)
        dag.add_node(u)
        latent.add(u)
        for var in sorted(block):
            dag.add_edge(u, var)

    return dag, latent


def _fresh_latent_name(index: int, used_names: set[str]) -> str:
    """Return a synthetic latent name that collides with no real node.

    Real variables are user-supplied and may themselves be named
    ``_u_resid_<i>`` by coincidence; if that name is already taken, keep
    appending an underscore until it is not.
    """
    name = f"_u_resid_{index}"
    while name in used_names:
        name += "_"
    return name


def _mutilate_outgoing(dag: nx.DiGraph, treatment: str) -> nx.DiGraph:
    """Drop arrows out of *treatment*, leaving only backdoor paths.

    Pearl's backdoor criterion is d-separation in this graph. Removing
    incoming arrows instead would delete the backdoor paths themselves.
    """
    mutilated = dag.copy()
    mutilated.remove_edges_from(list(dag.out_edges(treatment)))
    return mutilated


def _backdoor_separated(
    dag: nx.DiGraph, treatment: str, outcome: str, z: set[str]
) -> bool:
    """Return whether *z* blocks every backdoor path from treatment to outcome.

    ``nx.is_d_separator`` is polynomial in the graph size. Enumerating
    simple paths is exponential in the number of paths, which made
    identification unusable around 16–20 nodes.

    Equal endpoints are not a backdoor query: NetworkX rejects a separator
    whose sets are not disjoint, and a variable is not a cause of itself.
    """
    if treatment == outcome:
        return False
    return nx.is_d_separator(
        _mutilate_outgoing(dag, treatment), {treatment}, {outcome}, z
    )


def _minimal_adjustment_candidates(
    dag: nx.DiGraph, treatment: str, outcome: str, latent: set[str]
) -> set[str]:
    """Observed ancestors that can appear in a *minimal* adjustment set.

    Minimal separators never contain non-ancestors of the query pair
    (van der Zander, Liskiewicz & Textor 2019). A non-ancestor can still
    pad a non-minimal set; ``is_valid_adjustment_set`` accepts those.
    """
    ancestors = nx.ancestors(dag, treatment) | nx.ancestors(dag, outcome)
    descendants = nx.descendants(dag, treatment)
    return ancestors - descendants - latent - {treatment, outcome}


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

    .. note::

        This function reasons about the DAG structure declared in the
        model specification. It cannot detect omitted variables, missing
        edges, or other forms of misspecification. Use
        ``test_implications()`` to check whether the DAG's structural
        assumptions are consistent with observed data.

        A ``~~`` residual covariance block is a declaration of an
        unobserved common cause of its members, so it is expanded into
        a latent confounder node before the criterion is applied: no
        adjustment set can block the backdoor path it opens.

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
        alphabetically. ``[set()]`` means no covariates are required.
        ``[]`` means the effect is not identifiable.
    """
    _require_nodes(graph_info.contemporaneous_dag, treatment=treatment, outcome=outcome)

    dag, latent = _with_residual_confounders(graph_info)

    candidates = _minimal_adjustment_candidates(dag, treatment, outcome, latent)
    ordered = sorted(candidates)

    valid_sets: list[set[str]] = []
    for size in range(len(ordered) + 1):
        for subset in combinations(ordered, size):
            z = set(subset)
            if any(existing < z for existing in valid_sets):
                continue
            if _backdoor_separated(dag, treatment, outcome, z):
                valid_sets.append(z)
    valid_sets.sort(key=lambda s: (len(s), sorted(s)))
    return valid_sets


def is_valid_adjustment_set(
    graph_info: GraphInfo,
    treatment: str,
    outcome: str,
    z: set[str],
) -> bool:
    """Check whether *z* is a valid backdoor adjustment set.

    A set is valid when every member is an observed DAG node, excludes
    treatment and outcome, contains no treatment descendants or latents,
    and blocks all backdoor paths. A member may be a collider on some
    treatment–outcome path when that path stays blocked elsewhere. The
    set need not be minimal. Non-minimal sets are accepted when they
    still block all backdoor paths. ``collider_warnings()`` lists
    colliders in *z* as a heuristic and does not decide validity.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.
    treatment : str
        Treatment variable name.
    outcome : str
        Outcome variable name.
    z : set[str]
        Proposed adjustment set.

    Returns
    -------
    bool
        ``True`` when *z* is valid.

    Raises
    ------
    ValueError
        With a single actionable reason when *z* is invalid.
    """
    _require_nodes(graph_info.contemporaneous_dag, treatment=treatment, outcome=outcome)

    dag, latent = _with_residual_confounders(graph_info)

    for var in sorted(z):
        if var not in dag.nodes:
            raise ValueError(
                f"Adjustment variable '{var}' is not a node in the DAG. "
                f"Available nodes: {sorted(graph_info.contemporaneous_dag.nodes)}"
            )

    if treatment in z:
        raise ValueError(
            f"Adjustment set cannot include the treatment '{treatment}'. "
            f"Remove '{treatment}' from adjustment_set=."
        )

    if outcome in z:
        raise ValueError(
            f"Adjustment set cannot include the outcome '{outcome}'. "
            f"Remove '{outcome}' from adjustment_set=."
        )

    latent_in_z = sorted(z & latent)
    if latent_in_z:
        raise ValueError(
            f"Adjustment set contains unobserved (latent) variable(s): "
            f"{latent_in_z}. Latent nodes cannot be conditioned on — "
            f"remove them from adjustment_set=."
        )

    descendants = nx.descendants(dag, treatment)
    descendants_in_z = sorted(z & descendants)
    if descendants_in_z:
        raise ValueError(
            f"Adjustment set contains descendant(s) of '{treatment}': "
            f"{descendants_in_z}. Descendants of the treatment cannot be "
            f"in an adjustment set — remove them from adjustment_set=."
        )

    if not _backdoor_separated(dag, treatment, outcome, z):
        raise ValueError(
            f"Adjustment set {sorted(z)} does not block all backdoor paths "
            f"from '{treatment}' to '{outcome}'. Call adjustment_sets() on "
            f"the structural model to see valid sets, or pass a different "
            f"adjustment_set=."
        )

    return True


def is_identifiable(
    graph_info: GraphInfo,
    treatment: str,
    outcome: str,
) -> bool:
    """Check whether the causal effect of *treatment* on *outcome*
    is identifiable via the backdoor criterion.

    .. note::

        This function reasons about the DAG structure declared in the
        model specification. It cannot detect omitted variables, missing
        edges, or other forms of misspecification. Use
        ``test_implications()`` to check whether the DAG's structural
        assumptions are consistent with observed data.

        A ``~~`` residual covariance block is a declaration of an
        unobserved common cause of its members, so it is expanded into
        a latent confounder node before the criterion is applied: no
        adjustment set can block the backdoor path it opens.

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
    _require_nodes(graph_info.contemporaneous_dag, treatment=treatment, outcome=outcome)
    dag, latent = _with_residual_confounders(graph_info)
    # The full observed-ancestor set is valid iff any adjustment set exists.
    canonical = _minimal_adjustment_candidates(dag, treatment, outcome, latent)
    return _backdoor_separated(dag, treatment, outcome, canonical)


def frontdoor_identifiable(
    graph_info: GraphInfo,
    treatment: str,
    mediator: str,
    outcome: str,
) -> tuple[bool, str]:
    """Check whether the front-door criterion identifies the causal effect
    of *treatment* on *outcome* through *mediator*.

    The front-door criterion (Pearl, 2009) requires three conditions:

    1. *mediator* intercepts all directed paths from *treatment* to *outcome*.
    2. There is no unblocked backdoor path from *treatment* to *mediator*.
    3. All backdoor paths from *mediator* to *outcome* are blocked by
       conditioning on *treatment*.

    This function checks the criterion on the DAG as represented in
    *graph_info*. If the estimation spec includes adjustment variables
    that create edges absent from the true causal DAG (e.g. including
    treatment in the outcome equation to block a backdoor), build a
    separate ``GraphInfo`` from the causal structure for this check.

    .. note::

        This function reasons about the DAG structure declared in the
        model specification. It cannot detect omitted variables, missing
        edges, or other forms of misspecification. Use
        ``test_implications()`` to check whether the DAG's structural
        assumptions are consistent with observed data.

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.
    treatment : str
        Treatment variable name.
    mediator : str
        Mediator variable name.
    outcome : str
        Outcome variable name.

    Returns
    -------
    tuple[bool, str]
        ``(identifiable, message)`` where *message* explains the result
        or describes which condition fails.
    """
    _require_nodes(
        graph_info.contemporaneous_dag,
        treatment=treatment,
        mediator=mediator,
        outcome=outcome,
    )

    dag, _ = _with_residual_confounders(graph_info)

    if treatment == mediator or mediator == outcome or treatment == outcome:
        raise ValueError(
            "Treatment, mediator, and outcome must be three distinct "
            f"variables. Got treatment='{treatment}', mediator='{mediator}', "
            f"outcome='{outcome}'."
        )

    directed_paths = list(nx.all_simple_paths(dag, treatment, outcome))
    if not directed_paths:
        return (
            False,
            f"No directed path from '{treatment}' to '{outcome}' exists "
            f"in the DAG. The front-door criterion requires the mediator "
            f"to carry the causal effect.",
        )

    for path in directed_paths:
        if mediator not in path:
            bypass = " \u2192 ".join(path)
            return (
                False,
                f"Condition 1 fails: directed path {bypass} does not pass "
                f"through '{mediator}'. The front-door criterion requires "
                f"all directed paths to be fully mediated.",
            )

    if not _blocks_all_backdoor_paths(dag, dag, treatment, mediator, set()):
        return (
            False,
            f"Condition 2 fails: there is an unblocked backdoor path from "
            f"'{treatment}' to '{mediator}'. The front-door criterion "
            f"requires the treatment\u2013mediator relationship to be "
            f"unconfounded.",
        )

    if not _blocks_all_backdoor_paths(dag, dag, mediator, outcome, {treatment}):
        return (
            False,
            f"Condition 3 fails: not all backdoor paths from '{mediator}' "
            f"to '{outcome}' are blocked by conditioning on '{treatment}'. "
            f"The front-door criterion requires treatment to block all "
            f"mediator\u2013outcome confounding.",
        )

    return (
        True,
        f"Front-door criterion satisfied: the causal effect of "
        f"'{treatment}' on '{outcome}' is identified through "
        f"'{mediator}'.",
    )


def collider_warnings(
    graph_info: GraphInfo,
    adjustment_vars: set[str],
    treatment: str,
    outcome: str,
) -> list[str]:
    """Warn when an adjustment variable is a collider on a path
    between treatment and outcome.

    This is a structural heuristic. It flags the collider even when
    the rest of the set, or another unconditioned collider, keeps that
    path blocked. ``is_valid_adjustment_set()`` decides whether the
    set satisfies the backdoor criterion.

    .. note::

        This function reasons about the DAG structure declared in the
        model specification. It cannot detect omitted variables, missing
        edges, or other forms of misspecification. Use
        ``test_implications()`` to check whether the DAG's structural
        assumptions are consistent with observed data.

        A ``~~`` residual covariance block is a declaration of an
        unobserved common cause of its members, so it is expanded into
        a latent confounder node before colliders are searched for: a
        variable that only has one parent in the declared DAG can still
        become a collider once the synthetic confounder is added.

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
    dag, latent = _with_residual_confounders(graph_info)
    warnings_list: list[str] = []

    for var in adjustment_vars:
        if var not in dag.nodes:
            continue
        if var in latent:
            warnings_list.append(
                f"'{var}' is an unobserved (latent) variable and cannot be "
                f"conditioned on. Remove it from the adjustment set."
            )
            continue
        parents = list(dag.predecessors(var))
        if len(parents) >= 2:
            if _is_collider_on_path(dag, var, treatment, outcome):
                warnings_list.append(
                    f"'{var}' is a collider between '{treatment}' and "
                    f"'{outcome}'. Conditioning on it may open a spurious "
                    f"path and introduce bias."
                )

    return warnings_list


@dataclass(frozen=True)
class ConditionalIndependence:
    """An implied conditional independence statement from a DAG.

    Represents the testable implication X ⊥⊥ Y | Z, meaning X and Y
    are conditionally independent given the conditioning set Z.

    Parameters
    ----------
    x : str
        First variable.
    y : str
        Second variable.
    conditioning_set : frozenset[str]
        Variables to condition on. Empty frozenset for marginal independence.
    """

    x: str
    y: str
    conditioning_set: frozenset[str]

    def __str__(self) -> str:
        if self.conditioning_set:
            cond = ", ".join(sorted(self.conditioning_set))
            return f"{self.x} ⊥⊥ {self.y} | {{{cond}}}"
        return f"{self.x} ⊥⊥ {self.y}"

    def __repr__(self) -> str:
        return str(self)


@dataclass(repr=False)
class ImplicationTestResult(ResultReprMixin):
    """Results of testing implied conditional independences against data.

    Each row corresponds to one implied independence statement from the
    DAG, tested via partial correlation. A low p-value indicates that the
    data are inconsistent with the independence — suggesting the DAG may
    be missing an edge.

    Parameters
    ----------
    results : pd.DataFrame
        One row per independence test with columns: ``x``, ``y``,
        ``conditioning_set``, ``requested_k``, ``effective_k``,
        ``rank_collapsed``, ``df``, ``partial_corr``, ``p_value``,
        ``n_obs``, ``significant``. ``requested_k`` is the number of
        conditioning columns that entered the test. That equals the number
        of named conditioners when each is a single column; a categorical
        conditioner contributes its indicator columns. ``effective_k`` is
        ``rank([1, Z]) - 1``. For a single-column test ``df`` is
        ``n_obs - effective_k - 2``; a block test reports Rao's denominator
        degrees of freedom. ``effective_k`` and ``df`` are missing when the
        test was skipped. ``rank_collapsed`` is true when an executed test
        used fewer conditioning columns than requested, or a named
        conditioner contributed no columns. False on a skip.
        ``significant`` is false on a skip and on a collapsed row: the
        p-value is still the reduced test, but the column matches
        ``n_violations``. A collapsed row did not test the independence
        that was requested.
    alpha : float
        Significance level used for the ``significant`` column.
    """

    results: pd.DataFrame
    alpha: float

    @property
    def n_tests(self) -> int:
        """Total rows, including skips and rank-collapsed tests."""
        return len(self.results)

    @property
    def n_violations(self) -> int:
        """Significant rows that tested the independence that was written down.

        A rank-collapsed row is not a violation of that independence.
        ``significant`` is false on those rows, so this equals
        ``results["significant"].sum()`` for frames this module builds.
        """
        if self.results.empty or "significant" not in self.results.columns:
            return 0
        return int(self.results["significant"].astype(bool).sum())

    @property
    def violations(self) -> pd.DataFrame:
        """Significant rows that tested the independence that was written down."""
        return violation_rows(self.results, "significant")

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full results as a DataFrame."""
        return self.results.copy()

    def _repr_compact(self) -> str:
        written_tests, written_violations = written_counts(self.results, "significant")
        collapsed = n_rank_collapsed(self.results)
        collapse = f", {collapsed} rank-collapsed" if collapsed else ""
        return (
            f"ImplicationTestResult({written_tests} tests, "
            f"{written_violations} violations{collapse}, α={self.alpha})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        n_collapsed = n_rank_collapsed(self.results)
        written_tests, written_violations = written_counts(self.results, "significant")
        if n_collapsed:
            noun = "set" if n_collapsed == 1 else "sets"
            collapse = (
                f'<span style="color: orange; font-weight: bold;">⚠ '
                f"{n_collapsed} conditioning {noun} rank-collapsed.</span> "
                "Those rows did not test the independence you requested."
            )
        else:
            collapse = ""
        if written_violations == 0 and not n_collapsed:
            status = (
                '<span style="color: green; font-weight: bold;">✓ All '
                "implied independences are consistent with the data.</span>"
            )
        elif written_violations == 0:
            status = collapse
        else:
            status = (
                f'<span style="color: red; font-weight: bold;">✗ '
                f"{written_violations} of {written_tests} implied "
                f"independences violated.</span>"
            )
            if collapse:
                status = f"{status} {collapse}"

        header = f"<h4>DAG Implication Tests (α = {self.alpha})</h4><p>{status}</p>"

        rows = []
        for _, row in self.results.iterrows():
            cond = row["conditioning_set"]
            cond_str = f" | {{{cond}}}" if cond else ""
            statement = f"{row['x']} ⊥⊥ {row['y']}{cond_str}"
            collapsed = row_is_rank_collapsed(row)

            if collapsed or pd.isna(row["p_value"]):
                style = ""
                sig_mark = "—"
            elif row["significant"]:
                style = ' style="background-color: #ffe0e0;"'
                sig_mark = "✗"
            else:
                style = ""
                sig_mark = "✓"

            r_str = (
                f"{row['partial_corr']:.3f}"
                if not pd.isna(row["partial_corr"])
                else "—"
            )
            p_str = f"{row['p_value']:.4f}" if not pd.isna(row["p_value"]) else "—"
            statement += rank_collapse_note(row)

            rows.append(
                f"<tr{style}>"
                f"<td>{statement}</td>"
                f"<td>{r_str}</td>"
                f"<td>{p_str}</td>"
                f"<td>{sig_mark}</td>"
                f"</tr>"
            )

        table = (
            "<table>"
            "<thead><tr>"
            "<th>Independence</th>"
            "<th>Partial r</th>"
            "<th>p-value</th>"
            "<th>Pass</th>"
            "</tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody>"
            "</table>"
        )

        return header + table


def implied_independences(
    graph_info: GraphInfo,
) -> list[ConditionalIndependence]:
    """Enumerate conditional independences implied by the DAG.

    For each pair of non-adjacent nodes (X, Y), computes the conditioning
    set Z = pa(X) ∪ pa(Y) \\ {X, Y} and verifies d-separation. This is
    the *basis set* approach (Shipley, 2000): one testable implication per
    missing edge.

    A ``~~`` residual covariance block is a declaration of an unobserved
    common cause of its members, so d-separation is checked on the DAG
    with that confounder made explicit: two variables joined only by a
    declared ``~~`` edge (directly, or through a chain of blocks) are
    d-connected through the synthetic latent and are correctly never
    reported as independent, even though neither the candidate pair nor
    the conditioning set can include the latent itself (it has no
    observed data column to test against).

    Parameters
    ----------
    graph_info : GraphInfo
        DAG from the structural model.

    Returns
    -------
    list[ConditionalIndependence]
        Implied independence statements, sorted by (x, y) alphabetically.
    """
    dag = graph_info.contemporaneous_dag
    dag_aug, _ = _with_residual_confounders(graph_info)
    nodes = sorted(dag.nodes)
    result: list[ConditionalIndependence] = []

    for i, x in enumerate(nodes):
        for y in nodes[i + 1 :]:
            if dag.has_edge(x, y) or dag.has_edge(y, x):
                continue

            parents_x = set(dag.predecessors(x))
            parents_y = set(dag.predecessors(y))
            conditioning = (parents_x | parents_y) - {x, y}

            if nx.is_d_separator(dag_aug, {x}, {y}, conditioning):
                result.append(
                    ConditionalIndependence(
                        x=x,
                        y=y,
                        conditioning_set=frozenset(conditioning),
                    )
                )

    return result


def test_implications(
    independences: list[ConditionalIndependence],
    data: nw.DataFrame,
    alpha: float = 0.05,
    *,
    categorical_vars: Iterable[str] | None = None,
) -> ImplicationTestResult:
    """Test implied conditional independences against observed data.

    Uses partial correlation to test each independence statement. For
    an independence X ⊥⊥ Y | Z, regresses both X and Y on Z, then
    tests whether the correlation between residuals is significantly
    different from zero.

    A categorical variable enters as its indicator columns: as a
    conditioner this adjusts for group membership, and as X or Y the test
    becomes the linear multivariate generalisation (Wilks' lambda with
    Rao's F approximation, the partial F-test when the other side is a
    single column). The reported ``partial_corr`` is then the largest
    canonical correlation, which is non-negative.

    A significant result (p < alpha) on a row that is not rank-collapsed
    indicates a *violation*: the data show an association that the DAG
    says should not exist, suggesting a missing edge or incorrect
    structure. A collapsed row keeps the reduced-test p-value and sets
    ``significant`` false, so it is not that violation.

    Parameters
    ----------
    independences : list[ConditionalIndependence]
        Independence statements to test (from :func:`implied_independences`).
    data : nw.DataFrame
        Observed data. A statement that names a column absent from
        *data* is skipped. Declared latents and other unobserved
        members do not abort the remaining observable tests.
    alpha : float
        Significance level for flagging violations (default 0.05).
    categorical_vars : Iterable[str] | None
        Variables to dummy-encode even though their column is numeric
        (integer-coded labels declared with ``C()``). String and pandas
        categorical columns are always encoded.

    Returns
    -------
    ImplicationTestResult
        Test results with partial correlations, p-values, ``requested_k``,
        ``effective_k``, ``rank_collapsed``, and residual degrees of
        freedom (``df``).

    Warns
    -----
    UserWarning
        When an executed row has ``rank_collapsed`` set: a constant,
        duplicated, or collinear conditioner, or a named conditioner that
        contributed no columns. The test still runs on the columns that
        entered; the p-value is not evidence about the full independence.
        Drop the redundant variable and re-test.

    """
    rows: list[dict] = []
    collapse_labels: list[str] = []

    for ci in independences:
        all_vars = {ci.x, ci.y} | set(ci.conditioning_set)
        missing = all_vars - set(data.columns)
        if missing:
            # Unobserved members have no column to test; skip so a later
            # observable violation is still returned.
            continue

        z_vars = sorted(ci.conditioning_set)
        result = named_column_ci(
            data,
            ci.x,
            ci.y,
            z_vars,
            categorical_vars=categorical_vars,
        )
        conditioning = ", ".join(z_vars)
        collapsed = result.rank_collapsed
        if result.skip_reason is not None:
            r = np.nan
            p = np.nan
            df_value = np.nan
            effective_k = np.nan
        else:
            assert result.r is not None and result.p is not None
            assert result.df is not None and result.effective_k is not None
            r = result.r
            p = result.p
            df_value = result.df
            effective_k = result.effective_k
            if collapsed:
                collapse_labels.append(
                    collapse_statement(
                        ci.x,
                        ci.y,
                        conditioning,
                        result.requested_k,
                        int(effective_k),
                        result.dropped_conditioners,
                    )
                )

        rows.append({
            "x": ci.x,
            "y": ci.y,
            "conditioning_set": conditioning,
            "requested_k": result.requested_k,
            "effective_k": effective_k,
            "rank_collapsed": collapsed,
            "df": df_value,
            "partial_corr": r,
            "p_value": p,
            "n_obs": result.n,
            "significant": (
                result.skip_reason is None
                and result.p is not None
                and result.p < alpha
                and not collapsed
            ),
        })

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "x",
                "y",
                "conditioning_set",
                "requested_k",
                "effective_k",
                "rank_collapsed",
                "df",
                "partial_corr",
                "p_value",
                "n_obs",
                "significant",
            ]
        )
    warn_conditioning_rank_collapse(collapse_labels)

    return ImplicationTestResult(results=df, alpha=alpha)


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
