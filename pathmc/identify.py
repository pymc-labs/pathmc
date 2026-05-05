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

from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

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
    outcome : str
        Outcome variable name.

    Returns
    -------
    list[set[str]]
        All valid minimal adjustment sets, sorted by size then
        alphabetically. Empty list if no valid set exists or if
        the effect is already identified without adjustment.
    """
    dag = graph_info.contemporaneous_dag
    latent = graph_info.latent

    if treatment not in dag.nodes:
        raise ValueError(
            f"Treatment '{treatment}' not in DAG. Available nodes: {sorted(dag.nodes)}"
        )
    if outcome not in dag.nodes:
        raise ValueError(
            f"Outcome '{outcome}' not in DAG. Available nodes: {sorted(dag.nodes)}"
        )

    descendants = nx.descendants(dag, treatment)
    candidates = set(dag.nodes) - {treatment, outcome} - descendants - latent

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
    outcome : str
        Outcome variable name.

    Returns
    -------
    bool
        True if at least one valid backdoor adjustment set exists
        (including the empty set, meaning no adjustment is needed).
    """
    return len(adjustment_sets(graph_info, treatment, outcome)) > 0


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
    dag = graph_info.contemporaneous_dag

    for name, role in [
        (treatment, "Treatment"),
        (mediator, "Mediator"),
        (outcome, "Outcome"),
    ]:
        if name not in dag.nodes:
            raise ValueError(
                f"{role} '{name}' not in DAG. Available nodes: {sorted(dag.nodes)}"
            )

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
    """Check if any variable in the adjustment set is a collider
    on a path between treatment and outcome.

    Conditioning on a collider opens a spurious path and introduces
    bias. This function warns about such variables.

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
    dag = graph_info.contemporaneous_dag
    latent = graph_info.latent
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


@dataclass
class ImplicationTestResult:
    """Results of testing implied conditional independences against data.

    Each row corresponds to one implied independence statement from the
    DAG, tested via partial correlation. A low p-value indicates that the
    data are inconsistent with the independence — suggesting the DAG may
    be missing an edge.

    Parameters
    ----------
    results : pd.DataFrame
        One row per independence test with columns: ``x``, ``y``,
        ``conditioning_set``, ``partial_corr``, ``p_value``, ``significant``.
    alpha : float
        Significance level used for the ``significant`` column.
    """

    results: pd.DataFrame
    alpha: float

    @property
    def n_tests(self) -> int:
        """Total number of implied independences tested."""
        return len(self.results)

    @property
    def n_violations(self) -> int:
        """Number of independence violations (significant partial correlations)."""
        return int(self.results["significant"].sum())

    @property
    def violations(self) -> pd.DataFrame:
        """Subset of results where the independence is violated."""
        return self.results[self.results["significant"]].copy()

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full results as a DataFrame."""
        return self.results.copy()

    def __repr__(self) -> str:
        lines = [
            f"ImplicationTestResult: {self.n_tests} tests, "
            f"{self.n_violations} violations (α = {self.alpha})",
            "",
        ]
        if self.n_violations == 0:
            lines.append("All implied independences are consistent with the data.")
        else:
            lines.append("Violated independences (data suggests a missing edge):")
            for _, row in self.violations.iterrows():
                cond = row["conditioning_set"]
                cond_str = f" | {{{cond}}}" if cond else ""
                lines.append(
                    f"  {row['x']} ⊥⊥ {row['y']}{cond_str}  "
                    f"r = {row['partial_corr']:.3f}, p = {row['p_value']:.4f}"
                )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if self.n_violations == 0:
            status = (
                '<span style="color: green; font-weight: bold;">✓ All '
                "implied independences are consistent with the data.</span>"
            )
        else:
            status = (
                f'<span style="color: red; font-weight: bold;">✗ '
                f"{self.n_violations} of {self.n_tests} implied "
                f"independences violated.</span>"
            )

        header = f"<h4>DAG Implication Tests (α = {self.alpha})</h4><p>{status}</p>"

        rows = []
        for _, row in self.results.iterrows():
            cond = row["conditioning_set"]
            cond_str = f" | {{{cond}}}" if cond else ""
            statement = f"{row['x']} ⊥⊥ {row['y']}{cond_str}"

            if pd.isna(row["p_value"]):
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
    nodes = sorted(dag.nodes)
    result: list[ConditionalIndependence] = []

    for i, x in enumerate(nodes):
        for y in nodes[i + 1 :]:
            if dag.has_edge(x, y) or dag.has_edge(y, x):
                continue

            parents_x = set(dag.predecessors(x))
            parents_y = set(dag.predecessors(y))
            conditioning = (parents_x | parents_y) - {x, y}

            if nx.is_d_separator(dag, {x}, {y}, conditioning):
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
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> ImplicationTestResult:
    """Test implied conditional independences against observed data.

    Uses partial correlation to test each independence statement. For
    an independence X ⊥⊥ Y | Z, regresses both X and Y on Z, then
    tests whether the correlation between residuals is significantly
    different from zero.

    A significant result (p < alpha) indicates a *violation*: the data
    show an association that the DAG says should not exist, suggesting
    a missing edge or incorrect structure.

    Parameters
    ----------
    independences : list[ConditionalIndependence]
        Independence statements to test (from :func:`implied_independences`).
    data : pd.DataFrame
        Observed data. Must contain columns for all variables referenced
        in the independence statements.
    alpha : float
        Significance level for flagging violations (default 0.05).

    Returns
    -------
    ImplicationTestResult
        Test results with partial correlations and p-values.

    Raises
    ------
    ValueError
        If required columns are missing from *data*.
    """
    rows: list[dict] = []

    for ci in independences:
        all_vars = {ci.x, ci.y} | set(ci.conditioning_set)
        missing = all_vars - set(data.columns)
        if missing:
            raise ValueError(
                f"Columns {sorted(missing)} required by independence "
                f"'{ci}' are not in the data. Available columns: "
                f"{sorted(data.columns)}"
            )

        r, p, n = _partial_correlation_test(
            data, ci.x, ci.y, sorted(ci.conditioning_set)
        )

        cond_str = ", ".join(sorted(ci.conditioning_set))
        rows.append(
            {
                "x": ci.x,
                "y": ci.y,
                "conditioning_set": cond_str,
                "partial_corr": r,
                "p_value": p,
                "n_obs": n,
                "significant": (not np.isnan(p)) and p < alpha,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=[
                "x",
                "y",
                "conditioning_set",
                "partial_corr",
                "p_value",
                "n_obs",
                "significant",
            ]
        )
    return ImplicationTestResult(results=df, alpha=alpha)


def _partial_correlation_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    z_vars: list[str],
) -> tuple[float, float, int]:
    """Test conditional independence via partial correlation.

    Returns (partial_r, p_value, n_obs). If there are insufficient
    observations for the test, returns (nan, nan, n_obs).
    """
    cols = [x, y, *z_vars]
    sub = data[cols].dropna()
    n = len(sub)
    k = len(z_vars)

    if n < k + 3:
        return np.nan, np.nan, n

    x_vals = sub[x].to_numpy(dtype=float)
    y_vals = sub[y].to_numpy(dtype=float)

    if not z_vars:
        r, p = stats.pearsonr(x_vals, y_vals)
        return float(r), float(p), n

    z_mat = sub[z_vars].to_numpy(dtype=float)
    z_with_intercept = np.column_stack([np.ones(n), z_mat])

    beta_x, _, _, _ = np.linalg.lstsq(z_with_intercept, x_vals, rcond=None)
    resid_x = x_vals - z_with_intercept @ beta_x

    beta_y, _, _, _ = np.linalg.lstsq(z_with_intercept, y_vals, rcond=None)
    resid_y = y_vals - z_with_intercept @ beta_y

    r = np.corrcoef(resid_x, resid_y)[0, 1]
    df = n - k - 2
    t_stat = r * np.sqrt(df) / np.sqrt(1.0 - r**2)
    p = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return float(r), float(p), n


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
