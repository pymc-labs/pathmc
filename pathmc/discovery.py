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
"""Causal-discovery front end for pathmc.

pathmc is otherwise *DAG-in*: the user hands it a graph and pathmc
estimates, identifies, and falsifies against it. This module adds a
*discovery* step — learning candidate structure from data — via
:class:`TBFPC`, a target-oriented variant of the PC algorithm that uses
Bayes factors (a ΔBIC approximation) as its conditional-independence test.

Discovery rarely pins down a single DAG. :class:`TBFPC` returns a *CPDAG*
(a partially oriented graph standing for a whole Markov equivalence class);
:meth:`TBFPC.get_all_cdags_from_cpdag` enumerates the acyclic orientations
in that class, and :func:`pathmc.same_markov_equivalence_class` checks two
graphs for equivalence. Each enumerated DAG can then be turned into a model
with :func:`pathmc.dag_to_spec` / :class:`pathmc.BuildModelFromDAG`.
"""

from __future__ import annotations

import itertools as it
import warnings
from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt

__all__ = ["TBFPC", "TestResult"]

_TARGET_EDGE_RULES = ("any", "conservative", "fullS")


class TestResult(TypedDict):
    """Conditional-independence test statistics recorded during fitting.

    One entry is stored per ``(x, y, conditioning_set)`` test in
    :attr:`TBFPC.test_results`. ``forced`` is present (and ``True``) only
    for edges fixed by ``required_edges``, where no test was run.
    """

    bic0: float
    bic1: float
    delta_bic: float
    logBF10: float
    BF10: float
    independent: bool
    conditioning_set: list[str]
    forced: NotRequired[bool]


EMPTY_CONDITION_SET: frozenset[str] = frozenset()


class TBFPC:
    r"""Target-first Bayes Factor PC (TBF-PC) causal-discovery algorithm.

    A target-oriented variant of the Peter–Clark (PC) algorithm that uses
    Bayes factors (via a ΔBIC approximation) as the conditional-independence
    test.

    For each conditional-independence test of the form

    .. math::

        H_0 : Y \perp X \mid S
        \quad \text{vs.} \quad
        H_1 : Y \not\!\perp X \mid S

    two linear models are compared:

    .. math::

        M_0 : Y \sim S
        \\
        M_1 : Y \sim S + X

    where :math:`S` is a conditioning set of variables.

    The Bayesian Information Criterion (BIC) is defined as

    .. math::

        \mathrm{BIC}(M) = n \log\!\left(\frac{\mathrm{RSS}}{n}\right)
                          + k \log(n),

    with residual sum of squares :math:`\mathrm{RSS}`, sample size :math:`n`,
    and number of parameters :math:`k`. The Bayes factor is approximated by

    .. math::

        \log \mathrm{BF}_{10} \approx -\tfrac{1}{2}
        \left[ \mathrm{BIC}(M_1) - \mathrm{BIC}(M_0) \right].

    Independence is declared when :math:`\mathrm{BF}_{10} < \tau`, where
    :math:`\tau` is set via ``bf_thresh``.

    Target Edge Rules
    -----------------
    Different rules govern how driver → target edges are retained:

    - ``"any"``: keep :math:`X \to Y` unless **any** conditioning set renders
      :math:`X \perp Y \mid S`.
    - ``"conservative"``: keep :math:`X \to Y` if **at least one**
      conditioning set shows dependence.
    - ``"fullS"``: test only with the **full set** of other drivers as
      :math:`S`.

    Parameters
    ----------
    target : str
        Name of the outcome variable used to orient the search. Must be
        present in the data passed to :meth:`fit`.
    target_edge_rule : {"any", "conservative", "fullS"}
        Rule controlling which driver → target edges are retained.
    bf_thresh : float
        Positive Bayes-factor threshold for the conditional-independence
        tests.
    max_conditioning_set_size : int
        Largest conditioning set ``|S|`` searched in the ``"any"`` and
        ``"conservative"`` target phases and in the driver-skeleton phase
        (default 3). This bounds the combinatorial cost of the search; a pair
        that is only separable by a larger conditioning set will not be
        separated, so its edge is retained. The ``"fullS"`` target rule
        ignores this and always conditions on the full set of other drivers.
        It also controls the separating sets used by the v-structure
        orientation, so an overly small value can leave colliders undetected.
    forbidden_edges : Sequence[tuple[str, str]] | None
        Node pairs that must never be connected in the learned graph
        (background knowledge / orientation constraints). Symmetric: an
        entry ``(u, v)`` also forbids ``v—u``.
    required_edges : Sequence[tuple[str, str]] | None
        Directed ``(u, v)`` pairs that must appear as ``u -> v`` in the
        learned graph.

    Examples
    --------
    Basic usage with the full conditioning set::

        import numpy as np, pandas as pd
        from pathmc import TBFPC

        rng = np.random.default_rng(7)
        n = 2000
        C = rng.gamma(2, 1, n)
        A = 0.7 * C + rng.gamma(2, 1, n)
        D = 0.5 * C + rng.gamma(2, 1, n)
        B = 0.8 * A + rng.gamma(2, 1, n)
        Y = 0.9 * B + 0.6 * D + 0.7 * C + rng.gamma(2, 1, n)

        df = pd.DataFrame({"A": A, "B": B, "C": C, "D": D, "Y": Y})
        df = (df - df.mean()) / df.std()  # recommended scaling

        model = TBFPC(target="Y", target_edge_rule="fullS")
        model.fit(df, drivers=["A", "B", "C", "D"])
        print(model.to_digraph())

    Background knowledge — forbid an edge, force another::

        model = TBFPC(
            target="Y",
            forbidden_edges=[("A", "C")],
            required_edges=[("B", "Y")],
        )
        model.fit(df, drivers=["A", "B", "C", "D"])

    References
    ----------
    - Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*.
      MIT Press. [PC algorithm]
    - Spirtes & Glymour (1991). "An Algorithm for Fast Recovery of Sparse
      Causal Graphs."
    - Kass, R. & Raftery, A. (1995). "Bayes Factors."
    """

    def __init__(
        self,
        target: str,
        *,
        target_edge_rule: Literal["any", "conservative", "fullS"] = "any",
        bf_thresh: float = 1.0,
        max_conditioning_set_size: int = 3,
        forbidden_edges: Sequence[tuple[str, str]] | None = None,
        required_edges: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        warnings.warn(
            "TBFPC is experimental and its API may change; use with caution.",
            UserWarning,
            stacklevel=2,
        )

        if not isinstance(target, str) or not target:
            raise ValueError(
                "target must be a non-empty string naming the outcome variable. "
                f"Got {target!r}."
            )
        if target_edge_rule not in _TARGET_EDGE_RULES:
            raise ValueError(
                f"Unknown target_edge_rule {target_edge_rule!r}. "
                f"Choose from {', '.join(repr(r) for r in _TARGET_EDGE_RULES)}."
            )
        bf_thresh = float(bf_thresh)
        if not bf_thresh > 0.0:
            raise ValueError(
                f"bf_thresh must be a positive Bayes-factor threshold, got "
                f"{bf_thresh}. Use a value > 0 (1.0 is a neutral default)."
            )
        if not isinstance(max_conditioning_set_size, int) or isinstance(
            max_conditioning_set_size, bool
        ):
            raise TypeError(
                "max_conditioning_set_size must be an int, got "
                f"{type(max_conditioning_set_size).__name__}."
            )
        if max_conditioning_set_size < 0:
            raise ValueError(
                "max_conditioning_set_size must be a non-negative int, got "
                f"{max_conditioning_set_size}."
            )

        self.target = target
        self.target_edge_rule = target_edge_rule
        self.bf_thresh = bf_thresh
        self.max_conditioning_set_size = max_conditioning_set_size
        self.forbidden_edges = self._coerce_edges(forbidden_edges, "forbidden_edges")
        self.required_edges = self._coerce_edges(required_edges, "required_edges")

        conflicts = [
            (u, v)
            for (u, v) in self.required_edges
            if (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges
        ]
        if conflicts:
            conflict_str = ", ".join(f"{u}->{v}" for u, v in conflicts)
            raise ValueError(
                f"Required edges conflict with forbidden edges: {conflict_str}. "
                "Remove the pair from one of the two lists."
            )

        # Internal state
        self.sep_sets: dict[tuple[str, str], set[str]] = {}
        self._adj_directed: set[tuple[str, str]] = set()
        self._adj_undirected: set[tuple[str, str]] = set()
        self.nodes_: list[str] = []
        self.test_results: dict[tuple[str, str, frozenset[str]], TestResult] = {}

        # Shared response vector for symbolic BIC computation. Initialized with
        # a placeholder; updated with the actual response during fitting.
        self.y_sh = pytensor.shared(np.zeros(1, dtype="float64"), name="y_sh")
        self._bic_fn = self._build_symbolic_bic_fn()

    @staticmethod
    def _coerce_edges(
        edges: Sequence[tuple[str, str]] | None, arg_name: str
    ) -> set[tuple[str, str]]:
        """Validate and normalize an edge sequence into a set of 2-tuples."""
        result: set[tuple[str, str]] = set()
        for edge in edges or []:
            if (
                not isinstance(edge, tuple | list)
                or len(edge) != 2
                or not all(isinstance(node, str) for node in edge)
            ):
                raise ValueError(
                    f"{arg_name} must contain (u, v) string pairs, got {edge!r}."
                )
            result.add((edge[0], edge[1]))
        return result

    @staticmethod
    def _parse_cpdag_dot(
        dot: str,
    ) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]:
        """Parse a CPDAG DOT block into nodes, directed and undirected edges.

        Delegates to :func:`pathmc.cpdag._parse_dot` so every DOT consumer
        shares one grammar (quoted ids, chained edges, attribute brackets,
        quote-aware comments, ``dir=none`` undirected edges). Undirected edges
        are returned here as sorted ``(u, v)`` tuples for the orientation
        bookkeeping below.
        """
        from pathmc.cpdag import _parse_dot

        nodes, directed, undirected_fs = _parse_dot(dot)
        undirected = {tuple(sorted(edge)) for edge in undirected_fs}
        return nodes, directed, undirected  # type: ignore[return-value]

    @staticmethod
    def _is_acyclic(
        nodes: set[str], edges: list[tuple[str, str]] | set[tuple[str, str]]
    ) -> bool:
        """DFS cycle check over ``nodes`` and ``edges``."""
        adj: dict[str, list[str]] = {u: [] for u in nodes}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, [])
        state = {u: 0 for u in nodes}  # 0=unseen, 1=visiting, 2=done

        def dfs(u: str) -> bool:
            state[u] = 1
            for w in adj[u]:
                if state[w] == 1:
                    return False
                if state[w] == 0 and not dfs(w):
                    return False
            state[u] = 2
            return True

        return all(state[u] or dfs(u) for u in nodes)

    def _dot_from_edges(
        self, nodes: set[str], edges: list[tuple[str, str]] | set[tuple[str, str]]
    ) -> str:
        """Render a fully directed graph to DOT; highlight the target."""
        lines = ["digraph G {", "  node [shape=ellipse];"]
        for n in sorted(nodes):
            if hasattr(self, "target") and n == self.target:
                lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
            else:
                lines.append(f'  "{n}";')
        for u, v in sorted(edges):
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines)

    def _key(self, u: str, v: str) -> tuple[str, str]:
        """Return a sorted 2-tuple key for the undirected pair ``{u, v}``."""
        return (u, v) if u <= v else (v, u)

    def _set_sep(self, u: str, v: str, S: Sequence[str]) -> None:
        """Record the separation set ``S`` for the node pair ``(u, v)``."""
        self.sep_sets[self._key(u, v)] = set(S)

    def _has_forbidden(self, u: str, v: str) -> bool:
        """Return ``True`` if edge ``u—v`` is forbidden in either direction."""
        return (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges

    def _is_required(self, u: str, v: str) -> bool:
        """Return ``True`` if the directed edge ``u -> v`` is required."""
        return (u, v) in self.required_edges

    def _add_directed(self, u: str, v: str) -> None:
        """Add ``u -> v`` if not forbidden; drop the undirected edge if present."""
        if not self._has_forbidden(u, v):
            self._adj_undirected.discard(self._key(u, v))
            self._adj_directed.add((u, v))

    def _add_undirected(self, u: str, v: str) -> None:
        """Add ``u -- v`` if allowed and not already directed or required."""
        if (
            not self._has_forbidden(u, v)
            and (u, v) not in self._adj_directed
            and (v, u) not in self._adj_directed
            and not self._is_required(u, v)
            and not self._is_required(v, u)
        ):
            self._adj_undirected.add(self._key(u, v))

    def _remove_all(self, u: str, v: str) -> None:
        """Remove any edge (directed or undirected) between ``u`` and ``v``."""
        if self._is_required(u, v) or self._is_required(v, u):
            return
        self._adj_undirected.discard(self._key(u, v))
        self._adj_directed.discard((u, v))
        self._adj_directed.discard((v, u))

    def _enforce_required_edges(self) -> None:
        """Force required edges to appear as directed adjacencies."""
        for u, v in self.required_edges:
            self._adj_undirected.discard(self._key(u, v))
            self._adj_directed.discard((v, u))
            self._adj_directed.add((u, v))
            self.test_results[(u, v, EMPTY_CONDITION_SET)] = {
                "bic0": float("nan"),
                "bic1": float("nan"),
                "delta_bic": float("nan"),
                "logBF10": float("nan"),
                "BF10": float("nan"),
                "independent": False,
                "conditioning_set": [],
                "forced": True,
            }

    def _validate_required_nodes(self, drivers: Sequence[str]) -> None:
        """Ensure required edges reference known nodes."""
        allowed = set(drivers) | {self.target}
        missing: set[str] = set()
        for u, v in self.required_edges:
            if u not in allowed:
                missing.add(u)
            if v not in allowed:
                missing.add(v)
        if missing:
            raise ValueError(
                "Required edges reference unknown nodes: "
                + ", ".join(sorted(missing))
                + ". Every node must be the target or one of the drivers."
            )

    def _build_symbolic_bic_fn(self):
        """Build a BIC callable using a fast solver with a pseudoinverse fallback."""
        X = pt.matrix("X")
        n = pt.iscalar("n")

        xtx = pt.dot(X.T, X)
        xty = pt.dot(X.T, self.y_sh)

        beta_solve = pt.linalg.solve(xtx, xty)
        resid_solve = self.y_sh - pt.dot(X, beta_solve)
        rss_solve = pt.sum(resid_solve**2)

        beta_pinv = pt.linalg.pinv(X) @ self.y_sh
        resid_pinv = self.y_sh - pt.dot(X, beta_pinv)
        rss_pinv = pt.sum(resid_pinv**2)

        k = X.shape[1]

        nf = pt.cast(n, "float64")
        rss_solve_safe = pt.maximum(rss_solve, np.finfo("float64").tiny)
        rss_pinv_safe = pt.maximum(rss_pinv, np.finfo("float64").tiny)

        bic_solve = nf * pt.log(rss_solve_safe / nf) + k * pt.log(nf)
        bic_pinv = nf * pt.log(rss_pinv_safe / nf) + k * pt.log(nf)

        bic_solve_fn = pytensor.function(
            [X, n], [bic_solve, rss_solve], on_unused_input="ignore", mode="FAST_RUN"
        )
        bic_pinv_fn = pytensor.function(
            [X, n], bic_pinv, on_unused_input="ignore", mode="FAST_RUN"
        )

        def bic_fn(X_val: np.ndarray, n_val: int) -> float:
            try:
                bic_value, rss_value = bic_solve_fn(X_val, n_val)
                if np.isfinite(rss_value) and rss_value > np.finfo("float64").tiny:
                    return float(bic_value)
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                pass
            return float(bic_pinv_fn(X_val, n_val))

        return bic_fn

    def _ci_independent(
        self, df: pd.DataFrame, x: str, y: str, cond: Sequence[str]
    ) -> bool:
        """Return ``True`` if ΔBIC indicates ``x ⟂ y | cond``."""
        if self._has_forbidden(x, y):
            return True
        if self._is_required(x, y) or self._is_required(y, x):
            self.test_results[(x, y, frozenset(cond))] = TestResult(
                bic0=float("nan"),
                bic1=float("nan"),
                delta_bic=float("nan"),
                logBF10=float("nan"),
                BF10=float("nan"),
                independent=False,
                conditioning_set=list(cond),
                forced=True,
            )
            return False

        n = len(df)
        self.y_sh.set_value(df[y].to_numpy().astype("float64"))

        if len(cond) == 0:
            X0 = np.ones((n, 1))
        else:
            X0 = np.column_stack([np.ones(n), df[list(cond)].to_numpy()])
        X1 = np.column_stack([X0, df[x].to_numpy()])

        bic0 = float(self._bic_fn(X0, n))
        bic1 = float(self._bic_fn(X1, n))

        delta_bic = bic1 - bic0
        logBF10 = -0.5 * delta_bic
        # Decide in log space: an overwhelmingly dependent pair has a huge
        # logBF10 whose exp() overflows float64. The comparison below is
        # exactly equivalent to BF10 < bf_thresh but never overflows; BF10
        # itself is still reported (as inf when it overflows).
        independence = bool(logBF10 < np.log(self.bf_thresh))
        with np.errstate(over="ignore"):
            BF10 = np.exp(logBF10)
        result: TestResult = {
            "bic0": bic0,
            "bic1": bic1,
            "delta_bic": delta_bic,
            "logBF10": logBF10,
            "BF10": BF10,
            "independent": independence,
            "conditioning_set": list(cond),
        }
        self.test_results[(x, y, frozenset(cond))] = result

        return independence

    def _test_target_edges(self, df: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 1: test driver → target edges per ``target_edge_rule``."""
        for xi in drivers:
            neighbor_sets = [d for d in drivers if d != xi]
            max_k = min(self.max_conditioning_set_size, len(neighbor_sets))
            all_sets = [
                tuple(S)
                for k in range(max_k + 1)
                for S in it.combinations(neighbor_sets, k)
            ]

            if self.target_edge_rule == "any":
                keep = True
                for S in all_sets:
                    if self._ci_independent(df, xi, self.target, S):
                        self._set_sep(xi, self.target, S)
                        keep = False
                        break
                if keep:
                    self._add_directed(xi, self.target)
                else:
                    self._remove_all(xi, self.target)

            elif self.target_edge_rule == "conservative":
                indep_all = True
                for S in all_sets:
                    if not self._ci_independent(df, xi, self.target, S):
                        indep_all = False
                    else:
                        self._set_sep(xi, self.target, S)
                if indep_all:
                    self._remove_all(xi, self.target)
                else:
                    self._add_directed(xi, self.target)

            elif self.target_edge_rule == "fullS":
                S = tuple(neighbor_sets)
                if self._ci_independent(df, xi, self.target, S):
                    self._set_sep(xi, self.target, S)
                    self._remove_all(xi, self.target)
                else:
                    self._add_directed(xi, self.target)

    def _test_driver_skeleton(self, df: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 2: build the undirected driver skeleton via pairwise CI tests."""
        for xi, xj in it.combinations(drivers, 2):
            others = [d for d in drivers if d not in (xi, xj)]
            max_k = min(self.max_conditioning_set_size, len(others))
            dependent = True
            sep_rec = False
            for k in range(max_k + 1):
                for S in it.combinations(others, k):
                    if self._ci_independent(df, xi, xj, S):
                        self._set_sep(xi, xj, S)
                        dependent = False
                        sep_rec = True
                        break
                if sep_rec:
                    break
            if dependent:
                self._add_undirected(xi, xj)
            else:
                self._remove_all(xi, xj)

    def fit(self, df: pd.DataFrame, drivers: Sequence[str]) -> TBFPC:
        """Fit the TBF-PC procedure to *df*.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset containing the target column and every candidate driver.
            Standardizing the columns (zero mean, unit variance) is
            recommended so the ΔBIC test is well scaled.
        drivers : Sequence[str]
            Column names to treat as potential drivers of the target.

        Returns
        -------
        TBFPC
            The fitted instance (``self``) with internal adjacency structures
            populated.

        Examples
        --------
        ::

            model = TBFPC(target="Y", target_edge_rule="fullS")
            model.fit(df, drivers=["A", "B", "C"])
        """
        self._validate_required_nodes(drivers)

        self.sep_sets.clear()
        self._adj_directed.clear()
        self._adj_undirected.clear()
        self.test_results.clear()

        self._enforce_required_edges()
        self._test_target_edges(df, drivers)
        self._test_driver_skeleton(df, drivers)
        self._enforce_required_edges()

        self.nodes_ = [*list(drivers), self.target]
        self._orient_cpdag()
        return self

    def _orient_cpdag(self) -> None:
        """Turn the partially-directed result into a proper CPDAG.

        After the skeleton phase every retained driver edge points into the
        target and all driver--driver edges are undirected. This step orients
        the edges whose direction is *compelled* (identical across every
        member of the Markov equivalence class), exactly as the PC algorithm
        does:

        1. **Unshielded colliders.** For each unshielded driver triple
           ``x -- z -- y`` (``x`` and ``y`` non-adjacent), orient
           ``x -> z <- y`` when ``z`` is absent from the recorded separating
           set of ``x`` and ``y``. Colliders into the target are already
           oriented by the target-first phase, so only driver--driver edges
           are considered here.
        2. **Meek rules R1-R3.** Propagate the forced orientations to a
           fixpoint without creating a cycle or a new collider.

        Genuinely reversible edges stay undirected, so the result may be a
        single DAG or a multi-member CPDAG; :meth:`get_all_cdags_from_cpdag`
        enumerates whichever class survives. The orientation relies on the
        separating sets, so a too-small ``max_conditioning_set_size`` can
        leave a collider undetected.
        """
        nodes = set(self.nodes_)

        def adjacent(a: str, b: str) -> bool:
            return (
                (a, b) in self._adj_directed
                or (b, a) in self._adj_directed
                or self._key(a, b) in self._adj_undirected
            )

        def is_dir(a: str, b: str) -> bool:
            return (a, b) in self._adj_directed

        def orient(a: str, b: str) -> None:
            """Orient an undirected ``a -- b`` as ``a -> b`` (else a no-op)."""
            if self._key(a, b) in self._adj_undirected:
                self._adj_undirected.discard(self._key(a, b))
                self._adj_directed.add((a, b))

        def meek_compels(x: str, y: str) -> bool:
            """Return ``True`` if Meek's rules force ``x -- y`` to ``x -> y``."""
            # R1: c -> x, c not adjacent to y => x -> y (avoid a new collider).
            for c in nodes:
                if c not in (x, y) and is_dir(c, x) and not adjacent(c, y):
                    return True
            # R2: x -> c -> y => x -> y (avoid a cycle).
            for c in nodes:
                if c not in (x, y) and is_dir(x, c) and is_dir(c, y):
                    return True
            # R3: x -- c, x -- d, c -> y, d -> y, c,d non-adjacent => x -> y.
            undir = [
                c
                for c in nodes
                if c not in (x, y) and self._key(x, c) in self._adj_undirected
            ]
            for c, d in it.combinations(undir, 2):
                if is_dir(c, y) and is_dir(d, y) and not adjacent(c, d):
                    return True
            return False

        undirected_neighbors: dict[str, set[str]] = {n: set() for n in nodes}
        for u, v in self._adj_undirected:
            undirected_neighbors[u].add(v)
            undirected_neighbors[v].add(u)

        colliders: set[tuple[str, str]] = set()
        for z in nodes:
            for x, y in it.combinations(sorted(undirected_neighbors[z]), 2):
                if adjacent(x, y):
                    continue  # shielded triple
                sep = self.sep_sets.get(self._key(x, y))
                if sep is not None and z not in sep:
                    colliders.add((x, z))
                    colliders.add((y, z))
        for a, b in sorted(colliders):
            orient(a, b)

        changed = True
        while changed:
            changed = False
            for a, b in sorted(self._adj_undirected):
                if meek_compels(a, b):
                    orient(a, b)
                    changed = True
                    break
                if meek_compels(b, a):
                    orient(b, a)
                    changed = True
                    break

    def get_directed_edges(self) -> list[tuple[str, str]]:
        """Return the directed edges learned by the algorithm.

        Returns
        -------
        list[tuple[str, str]]
            Sorted list of ``(u, v)`` oriented edges.
        """
        return sorted(self._adj_directed)

    def get_undirected_edges(self) -> list[tuple[str, str]]:
        """Return the undirected edges remaining after orientation.

        Returns
        -------
        list[tuple[str, str]]
            Sorted list of ``(u, v)`` pairs for unresolved adjacencies.
        """
        return sorted(self._adj_undirected)

    def get_test_results(self, x: str, y: str) -> list[TestResult]:
        """Return ΔBIC diagnostics for the unordered pair ``{x, y}``.

        Parameters
        ----------
        x, y : str
            The two variables in the pair (order does not matter).

        Returns
        -------
        list[TestResult]
            One entry per conditioning set tested, each holding ``bic0``,
            ``bic1``, ``delta_bic``, ``logBF10``, ``BF10``, ``independent``,
            and the ``conditioning_set`` used.
        """
        return [v for (xi, yi, _), v in self.test_results.items() if {xi, yi} == {x, y}]

    def summary(self) -> str:
        """Render a text summary of the learned graph and the CI-test count.

        Returns
        -------
        str
            Multiline string listing directed edges, undirected edges, and
            the number of conditional-independence tests executed.
        """
        lines = ["=== Directed edges ==="]
        for u, v in self.get_directed_edges():
            suffix = " [required]" if self._is_required(u, v) else ""
            lines.append(f"{u} -> {v}{suffix}")
        lines.append("=== Undirected edges ===")
        for u, v in self.get_undirected_edges():
            lines.append(f"{u} -- {v}")
        lines.append("=== Number of CI tests run ===")
        lines.append(str(len(self.test_results)))
        return "\n".join(lines)

    def to_digraph(self) -> str:
        """Return the learned CPDAG encoded in DOT format.

        Directed edges render as ``u -> v``; undirected (unoriented) edges as
        ``u -> v [style=dashed, dir=none]``; required edges are highlighted,
        and the target node is filled.

        Returns
        -------
        str
            DOT string compatible with Graphviz rendering utilities and with
            :func:`pathmc.same_markov_equivalence_class`.
        """
        lines = ["digraph G {", "  node [shape=ellipse];"]
        for n in self.nodes_:
            if n == self.target:
                lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
            else:
                lines.append(f'  "{n}";')
        for u, v in self.get_directed_edges():
            attrs = " [color=darkgreen, penwidth=2]" if self._is_required(u, v) else ""
            lines.append(f'  "{u}" -> "{v}"{attrs};')
        for u, v in self.get_undirected_edges():
            lines.append(f'  "{u}" -> "{v}" [style=dashed, dir=none];')
        lines.append("}")
        return "\n".join(lines)

    def get_all_cdags_from_cpdag(self, dot_cpdag: str | None = None) -> list[str]:
        """Enumerate the member DAGs of the CPDAG's Markov equivalence class.

        This is what makes the discovery output a *set* of equally plausible
        graphs rather than one arbitrary DAG: every undirected edge is
        oriented both ways, then an orientation is kept only if it (a) stays
        acyclic and (b) introduces no *new* v-structure (unshielded collider)
        beyond those already compelled by the CPDAG's directed edges. Those
        two filters together are exactly the membership test for the Markov
        equivalence class, so every returned DAG satisfies
        :func:`pathmc.same_markov_equivalence_class` against the input CPDAG.
        Downstream model averaging fits each returned DAG and pools the effect
        posteriors.

        Because :meth:`fit` already orients the compelled edges
        (v-structures + Meek rules), a CPDAG with no reversible edges
        collapses to a single DAG, while genuinely reversible structure
        (chains/forks, cliques) yields several members.

        Parameters
        ----------
        dot_cpdag : str | None
            If provided, parse the CPDAG from this DOT string (undirected
            edges encoded as ``[style=dashed, dir=none]``). If ``None``, use
            this model's current CPDAG from :meth:`get_directed_edges` and
            :meth:`get_undirected_edges`.

        Returns
        -------
        list[str]
            DOT strings, each a fully oriented DAG (no dashed edges) and a
            member of the same Markov equivalence class as the CPDAG.
        """
        from pathmc.cpdag import _skeleton, _v_structures

        nodes, fixed_dir, undirected = (
            self._parse_cpdag_dot(dot_cpdag)
            if dot_cpdag is not None
            else (
                set(self.nodes_),
                set(self.get_directed_edges()),
                set(self.get_undirected_edges()),
            )
        )

        if not undirected:
            edges = sorted(fixed_dir)
            if self._is_acyclic(nodes, edges):
                return [self._dot_from_edges(nodes, edges)]
            return []

        und = sorted({self._key(u, v) for (u, v) in undirected})
        skeleton = _skeleton(set(fixed_dir), {frozenset(e) for e in und})
        baseline_v = _v_structures(set(fixed_dir), skeleton)

        cdags: list[str] = []
        for mask in it.product((0, 1), repeat=len(und)):
            oriented = list(fixed_dir)
            oriented.extend(
                (u, v) if b == 0 else (v, u)
                for b, (u, v) in zip(mask, und, strict=False)
            )
            if not self._is_acyclic(nodes, oriented):
                continue
            if _v_structures(set(oriented), skeleton) != baseline_v:
                # Orienting an undirected edge created a collider absent from
                # the CPDAG, so this DAG lies in a different equivalence class.
                continue
            cdags.append(self._dot_from_edges(nodes, oriented))
        return cdags
