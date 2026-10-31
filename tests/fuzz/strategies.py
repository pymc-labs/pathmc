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
"""Grammar strategies for random-but-valid pathmc specs (issue #414).

The generator stays inside constraints the compiler accepts:

* ``lag+adstock`` is co-occurring terms (``lag(x) + adstock(x, decay=...)``),
  never ``lag(adstock(...))``, which is a parse error.
* Random slopes are ``pooling={"intercept": True, "slopes": [...]}`` on a
  raw column that also appears as a plain term. The string ``"slopes"`` is
  not a pooling mode.
* ``lag()``, partial pooling, and slopes require ``panel=``. Without a
  panel the compiler either raises (lag) or silently drops the hierarchical
  terms (pooling).
* Contemporaneous self-edges (``y ~ y``, ``y ~ adstock(y, ...)``, ``y ~ y:x``)
  are cycles. ``lag(y)`` is temporal and allowed, including count families.
* Random slopes name an exogenous plain column (``x0``). An endogenous slope
  is not a data column; the non-scan compiler raises ``KeyError``.
* The compiling budget does not draw a count outcome on the scan path.
  ``dlogp`` for that class takes minutes (a count predictor, a count outcome
  coupled to a Gaussian mediator, and a count outcome with slopes all did).
  Naming a narrower subclass kept missing the next one. ``lag`` of a count
  is still compiled by the covering stratum and the frozen corpus. The
  grammar still emits the class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hypothesis import assume, strategies as st

FAMILIES: tuple[str, ...] = ("gaussian", "bernoulli", "poisson", "negbinomial")
COUNT_FAMILIES = frozenset({"poisson", "negbinomial"})
TERM_KINDS: tuple[str, ...] = (
    "plain",
    "lag",
    "adstock",
    "logistic_saturation",
    "interaction",
    "lag+adstock",
)
# Single-parent edges only. Interaction needs two plain names and is added
# separately. ``plain`` is first so Hypothesis shrinks toward it.
_SINGLE_PARENT_KINDS: tuple[str, ...] = (
    "plain",
    "logistic_saturation",
    "adstock",
    "lag",
    "lag+adstock",
)
_TEMPORAL_KINDS = frozenset({"lag", "adstock", "lag+adstock"})
POOLING_KINDS: tuple[str, ...] = ("none", "partial", "slopes")

# Issue #414: n_units in {1, 2, 5}, n_time in {3, 10, 30}, plus the
# adversarial pair (1, 2). Ordered small-first so shrinking prefers it.
PANEL_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (2, 3),
    (5, 3),
    (1, 10),
    (2, 10),
    (5, 10),
    (1, 30),
    (2, 30),
    (5, 30),
)
# Cross-section has no time axis. n=1 is a constant predictor, which the
# issue says to filter, so the smallest draw is 2.
CROSS_SECTION_N: tuple[int, ...] = (2, 5, 10, 30)
# The per-commit mixed budget compiles each draw, including ``dlogp``. Large
# panels make that session take tens of minutes, so it draws from this subset.
# Nightly and the grammar test still use the full menu above.
FAST_PANEL_SHAPES: tuple[tuple[int, int], ...] = ((1, 2), (2, 3))
FAST_CROSS_SECTION_N: tuple[int, ...] = (2, 5)

_PANEL_KEYS = (("unit", "unit"), ("time", "time"))


@dataclass(frozen=True)
class Edge:
    """One rendered term, or a co-occurring lag+adstock pair, into ``child``."""

    child: str
    kind: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Stratum:
    """One covering cell the fast session must compile."""

    id: str
    family: str
    term: str
    pooling: str
    scan: bool


# Each required family, term kind, pooling mode, and scan bit appears at
# least once. The fast session fails if a stratum stops carrying its tag.
COVERING_STRATA: tuple[Stratum, ...] = (
    Stratum("family-gaussian", "gaussian", "plain", "none", False),
    Stratum("family-bernoulli", "bernoulli", "plain", "none", False),
    Stratum("family-poisson", "poisson", "plain", "none", False),
    Stratum("family-negbinomial", "negbinomial", "plain", "none", False),
    Stratum("term-lag", "gaussian", "lag", "none", True),
    Stratum("term-adstock", "gaussian", "adstock", "none", True),
    Stratum("term-logistic", "gaussian", "logistic_saturation", "none", False),
    Stratum("term-interaction", "gaussian", "interaction", "none", False),
    Stratum("term-lag-adstock", "gaussian", "lag+adstock", "none", True),
    Stratum("pooling-partial", "gaussian", "plain", "partial", False),
    Stratum("pooling-slopes", "gaussian", "plain", "slopes", False),
    Stratum("scan-poisson-lag-exog", "poisson", "lag", "none", True),
)


@dataclass(frozen=True)
class FuzzCase:
    """One fully specified model the oracle can rebuild from ``seed`` alone.

    The frame is not stored. :func:`realize_frame` rebuilds it, so a printed
    case is a complete reproducer and Hypothesis can shrink the draws that
    produced it.
    """

    seed: int
    spec: str
    families: tuple[tuple[str, str], ...]
    panel: tuple[tuple[str, str], ...] | None
    pooling_kind: str
    slope_vars: tuple[str, ...]
    layout: str
    n_units: int
    n_time: int
    n_obs: int
    columns: tuple[str, ...]
    predictors: tuple[str, ...]
    term_kinds: tuple[str, ...]
    intercept: bool

    def families_dict(self) -> dict[str, str]:
        """Family map accepted by :func:`pathmc.model`."""
        return dict(self.families)

    def panel_dict(self) -> dict[str, str] | None:
        """Panel map accepted by :func:`pathmc.model`, or ``None``."""
        if self.panel is None:
            return None
        return dict(self.panel)

    def pooling(self) -> str | dict[str, object] | None:
        """Pooling argument accepted by :func:`pathmc.model`."""
        if self.pooling_kind == "none":
            return None
        if self.pooling_kind == "partial":
            return "partial"
        return {"intercept": True, "slopes": list(self.slope_vars)}

    def expects_scan(self) -> bool:
        """Whether this spec should take the scan-panel compiler."""
        temporal = bool(set(self.term_kinds) & _TEMPORAL_KINDS)
        return self.layout == "panel" and temporal

    def count_scan_predictor(self) -> bool:
        """Count outcome on the scan path.

        The compiling budget skips the whole class. A narrower predicate
        missed the next multi-minute graph.
        """
        if not self.expects_scan():
            return False
        return any(family in COUNT_FAMILIES for _, family in self.families)

    def plain_terms(self) -> set[str]:
        """Raw column names that appear as plain terms, not inside a call.

        ``str.replace("0 +", "")`` would also eat the ``0 +`` inside ``x0 +``,
        so the intercept prefix is stripped only from the start of the RHS.
        """
        found: set[str] = set()
        for line in self.spec.splitlines():
            rhs = line.split("~", 1)[1].strip()
            if rhs.startswith("0 +"):
                rhs = rhs[3:].strip()
            for part in rhs.split("+"):
                token = part.strip()
                if token.isidentifier():
                    found.add(token)
        return found

    def __repr__(self) -> str:
        return (
            f"FuzzCase(seed={self.seed}, spec={self.spec!r}, "
            f"families={self.families_dict()}, panel={self.panel_dict()}, "
            f"pooling={self.pooling()!r}, layout={self.layout}, "
            f"n_units={self.n_units}, n_time={self.n_time}, "
            f"n_obs={self.n_obs}, term_kinds={self.term_kinds})"
        )


class Coverage:
    """Session record of compile cells the fuzzer actually built.

    A run that only exercises trivial ``y ~ x`` Gaussians must fail loudly.
    Required cells are the issue's families, term kinds, pooling modes, and
    both the scan and non-scan compilers.
    """

    def __init__(self) -> None:
        self.families: set[str] = set()
        self.terms: set[str] = set()
        self.pooling: set[str] = set()
        self.scan: set[bool] = set()
        self.n = 0

    def record(self, case: FuzzCase, *, scan: bool) -> None:
        """Record one successfully compiled case."""
        self.n += 1
        self.families.update(family for _, family in case.families)
        self.terms.update(case.term_kinds)
        self.pooling.add(case.pooling_kind)
        self.scan.add(scan)

    def missing(self) -> dict[str, list[str]]:
        """Names of required cells this session has not compiled."""
        gaps: dict[str, list[str]] = {}
        absent_families = sorted(set(FAMILIES) - self.families)
        absent_terms = sorted(set(TERM_KINDS) - self.terms)
        absent_pooling = sorted(set(POOLING_KINDS) - self.pooling)
        absent_scan = [
            str(flag).lower() for flag in (False, True) if flag not in self.scan
        ]
        if absent_families:
            gaps["family"] = absent_families
        if absent_terms:
            gaps["term"] = absent_terms
        if absent_pooling:
            gaps["pooling"] = absent_pooling
        if absent_scan:
            gaps["scan"] = absent_scan
        return gaps

    def assert_session_covered(self) -> None:
        """Raise if any required compile cell was never hit."""
        gaps = self.missing()
        if not gaps:
            return
        detail = ", ".join(
            f"{name} never hit {values}" for name, values in gaps.items()
        )
        raise AssertionError(
            "Fuzz session coverage is incomplete. A run that only exercises "
            f"trivial `y ~ x` Gaussians is not enough. {detail}. "
            f"Recorded {self.n} case(s); families={sorted(self.families)}, "
            f"terms={sorted(self.terms)}, pooling={sorted(self.pooling)}, "
            f"scan={sorted(self.scan)}."
        )


def _terms(edge: Edge) -> list[str]:
    parent = edge.inputs[0]
    child = edge.child
    if edge.kind == "plain":
        return [parent]
    if edge.kind == "lag":
        return [f"lag({parent})"]
    if edge.kind == "adstock":
        return [f"adstock({parent}, decay=decay_{child}_{parent})"]
    if edge.kind == "logistic_saturation":
        return [f"logistic_saturation({parent}, lam=lam_{child}_{parent})"]
    if edge.kind == "interaction":
        return [":".join(edge.inputs)]
    if edge.kind == "lag+adstock":
        # Co-occurring terms. Nested lag(adstock(...)) is a parse error.
        return [
            f"lag({parent})",
            f"adstock({parent}, decay=decay_{child}_{parent})",
        ]
    raise ValueError(f"unknown term kind {edge.kind!r}")


def render_spec(edges: list[Edge], intercept: dict[str, bool]) -> str:
    """Render equations in child order, one endogenous node per line."""
    children: list[str] = []
    for edge in edges:
        if edge.child not in children:
            children.append(edge.child)
    lines: list[str] = []
    for child in children:
        terms: list[str] = []
        for edge in edges:
            if edge.child == child:
                terms.extend(_terms(edge))
        if not terms:
            raise ValueError(
                f"empty design for '{child}'. Add at least one predictor term."
            )
        rhs = " + ".join(terms)
        if not intercept[child]:
            rhs = f"0 + {rhs}"
        lines.append(f"{child} ~ {rhs}")
    if not lines:
        raise ValueError("empty design: the spec has no equations.")
    return "\n".join(lines)


def slow_count_scan(
    edges: list[Edge],
    families: dict[str, str],
    layout: str,
) -> bool:
    """Count outcome on the scan path.

    ``dlogp`` for this class takes minutes. A count predictor, a count
    outcome coupled to a Gaussian mediator (seed 4119011), and a count
    outcome with exogenous slopes (seed 44270) were all slow. The compiling
    budget skips the class. The covering stratum and the frozen corpus still
    compile the known cases.
    """
    temporal = any(edge.kind in _TEMPORAL_KINDS for edge in edges)
    if layout != "panel" or not temporal:
        return False
    return any(family in COUNT_FAMILIES for family in families.values())


def _legal_kinds(parent: str, child: str) -> tuple[str, ...]:
    if parent == child:
        # lag(y) is temporal. adstock(y) and lag+adstock (which renders
        # adstock(y)) are contemporaneous self-edges and a structural cycle.
        return ("lag",)
    return _SINGLE_PARENT_KINDS


def _col_kind(name: str, families: dict[str, str]) -> str:
    family = families.get(name, "gaussian")
    if family == "bernoulli":
        return "bernoulli"
    if family in COUNT_FAMILIES:
        return "count"
    return "continuous"


def _series(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    """Length-``n`` column with at least two distinct values when ``n >= 2``."""
    if n < 2:
        raise ValueError(
            f"refusing a length-{n} column; a single row is a constant "
            "predictor. Use n_obs >= 2 or n_time >= 2."
        )
    if kind == "continuous":
        return np.linspace(-1.0, 1.0, n) + rng.normal(0.0, 0.05, size=n)
    if kind == "bernoulli":
        return np.resize(np.array([0, 1], dtype=int), n).copy()
    if kind == "count":
        return np.resize(np.array([0, 1, 2, 3], dtype=int), n).copy()
    raise ValueError(f"unknown column kind {kind!r}")


def realize_frame(case: FuzzCase) -> pd.DataFrame:
    """Rebuild the case frame from ``seed`` and the stored shape.

    Predictors are not constant. Outcomes match the family the compiler
    will cast: 0/1 for Bernoulli, non-negative ints for counts.
    """
    rng = np.random.default_rng(case.seed)
    families = case.families_dict()
    if case.layout == "panel":
        blocks: list[pd.DataFrame] = []
        for unit in range(case.n_units):
            block: dict[str, np.ndarray] = {
                "unit": np.full(case.n_time, unit, dtype=int),
                "time": np.arange(case.n_time, dtype=int),
            }
            for column in case.columns:
                block[column] = _series(rng, case.n_time, _col_kind(column, families))
            blocks.append(pd.DataFrame(block))
        frame = pd.concat(blocks, ignore_index=True)
    else:
        block = {
            column: _series(rng, case.n_obs, _col_kind(column, families))
            for column in case.columns
        }
        frame = pd.DataFrame(block)
    degenerate = [column for column in case.predictors if frame[column].nunique() < 2]
    if degenerate:
        raise ValueError(
            f"degenerate spec: predictor(s) {degenerate} are constant. "
            "Filter this draw or widen the column."
        )
    return frame


def _assemble(
    *,
    seed: int,
    edges: list[Edge],
    families: dict[str, str],
    pooling_kind: str,
    slope_vars: tuple[str, ...],
    layout: str,
    n_units: int,
    n_time: int,
    n_obs: int,
    intercept: dict[str, bool],
) -> FuzzCase:
    spec = render_spec(edges, intercept)
    if "lag(adstock" in spec or "lag(logistic_saturation" in spec:
        raise ValueError(f"nested lag is not a legal term: {spec}")
    columns = tuple(
        sorted({name for edge in edges for name in (*edge.inputs, edge.child)})
    )
    predictors = tuple(sorted({name for edge in edges for name in edge.inputs}))
    # One intercept flag for the repr. Mixed models may differ per equation;
    # the spec is the source of truth, and this is true only if every
    # equation keeps the formula intercept.
    return FuzzCase(
        seed=seed,
        spec=spec,
        families=tuple(sorted(families.items())),
        panel=_PANEL_KEYS if layout == "panel" else None,
        pooling_kind=pooling_kind,
        slope_vars=slope_vars,
        layout=layout,
        n_units=n_units,
        n_time=n_time,
        n_obs=n_obs,
        columns=columns,
        predictors=predictors,
        term_kinds=tuple(sorted({edge.kind for edge in edges})),
        intercept=all(intercept.values()),
    )


def _shape(
    draw: st.DrawFn,
    layout: str,
    panel_shapes: tuple[tuple[int, int], ...],
    cross_section_n: tuple[int, ...],
) -> tuple[str, int, int, int]:
    if layout == "panel":
        n_units, n_time = draw(st.sampled_from(panel_shapes))
        return layout, n_units, n_time, n_units * n_time
    n_obs = draw(st.sampled_from(cross_section_n))
    return layout, 0, 0, n_obs


@st.composite
def stratum_cases(draw: st.DrawFn, stratum: Stratum) -> FuzzCase:
    """One example that carries ``stratum``'s family, term, pooling, and scan bit.

    The structure, intercept, and shape are fixed so one example cannot miss
    the cell or land on a 5×30 panel. The seed still shrinks. The mixed
    strategy and the frozen corpus cover the shape menu.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    # Fixed small shapes. The mixed strategy and the frozen corpus cover the
    # shape menu, including the adversarial pair (1, 2). A covering cell that
    # drew 5x30 would make the per-commit session slow without adding a tag.
    needs_panel = (
        stratum.scan or stratum.pooling != "none" or stratum.term in _TEMPORAL_KINDS
    )
    if needs_panel:
        layout, n_units, n_time, n_obs = "panel", 2, 3, 6
    else:
        layout, n_units, n_time, n_obs = "cross_section", 0, 0, 8
    intercept_on = True
    if stratum.term == "interaction":
        edges = [Edge("x2", "interaction", ("x0", "x1"))]
        families = {"x2": stratum.family}
        intercept = {"x2": intercept_on}
    else:
        edges = [Edge("x1", stratum.term, ("x0",))]
        if stratum.pooling == "slopes" and stratum.term != "plain":
            edges.append(Edge("x1", "plain", ("x0",)))
        families = {"x1": stratum.family}
        intercept = {"x1": intercept_on}
    slope_vars = ("x0",) if stratum.pooling == "slopes" else ()
    case = _assemble(
        seed=seed,
        edges=edges,
        families=families,
        pooling_kind=stratum.pooling,
        slope_vars=slope_vars,
        layout=layout,
        n_units=n_units,
        n_time=n_time,
        n_obs=n_obs,
        intercept=intercept,
    )
    if stratum.family not in case.families_dict().values():
        raise ValueError(f"{stratum.id} dropped family {stratum.family}")
    if stratum.term not in case.term_kinds:
        raise ValueError(f"{stratum.id} dropped term {stratum.term}")
    if case.pooling_kind != stratum.pooling or case.expects_scan() != stratum.scan:
        raise ValueError(
            f"{stratum.id} built pooling={case.pooling_kind} "
            f"scan={case.expects_scan()}, expected {stratum.pooling} "
            f"scan={stratum.scan}"
        )
    return case


def _endogenous(draw: st.DrawFn, names: list[str]) -> list[str]:
    """Nodes with an equation. The root stays exogenous; the last node does not."""
    chosen = [names[-1]]
    for name in names[1:-1]:
        if draw(st.booleans()):
            chosen.append(name)
    return sorted(chosen, key=lambda name: int(name[1:]))


@st.composite
def fuzz_cases(
    draw: st.DrawFn,
    panel_shapes: tuple[tuple[int, int], ...] = PANEL_SHAPES,
    cross_section_n: tuple[int, ...] = CROSS_SECTION_N,
    for_compile: bool = False,
) -> FuzzCase:
    """Random acyclic spec with 2–5 nodes and at least one equation.

    Every endogenous equation includes an edge from ``x0`` (always
    exogenous), so the design stays non-empty.
    """
    seed = draw(st.integers(0, 2**31 - 1))
    n_nodes = draw(st.integers(2, 5))
    names = [f"x{i}" for i in range(n_nodes)]
    endogenous = _endogenous(draw, names)
    families = {node: draw(st.sampled_from(FAMILIES)) for node in endogenous}
    pooling_kind = draw(st.sampled_from(POOLING_KINDS))
    layout = draw(st.sampled_from(("cross_section", "panel")))
    if pooling_kind != "none":
        layout = "panel"

    edges: list[Edge] = []
    for child in endogenous:
        parents = names[: int(child[1:])]
        # parents[0] is x0. Always keep it so the equation has a legal term.
        chosen = [parents[0]]
        for parent in parents[1:]:
            if draw(st.booleans()):
                chosen.append(parent)
        for parent in chosen:
            kinds = _legal_kinds(parent, child)
            kind = draw(st.sampled_from(kinds))
            if kind in {"lag", "lag+adstock"}:
                layout = "panel"
            edges.append(Edge(child, kind, (parent,)))
        self_kinds = _legal_kinds(child, child)
        if self_kinds and draw(st.booleans()):
            layout = "panel"
            edges.append(Edge(child, draw(st.sampled_from(self_kinds)), (child,)))

    candidates = [child for child in endogenous if int(child[1:]) >= 2]
    if candidates and draw(st.booleans()):
        child = draw(st.sampled_from(candidates))
        earlier = names[: int(child[1:])]
        left = draw(st.sampled_from(earlier))
        rest = [name for name in earlier if name != left]
        right = draw(st.sampled_from(rest))
        edges.append(Edge(child, "interaction", tuple(sorted((left, right)))))

    if pooling_kind == "slopes" and not any(
        edge.kind == "plain" and edge.inputs == ("x0",) for edge in edges
    ):
        edges.append(Edge(endogenous[-1], "plain", ("x0",)))
    # x0 is always exogenous. An endogenous plain parent is not a data column.
    slope_vars = ("x0",) if pooling_kind == "slopes" else ()
    if for_compile and slow_count_scan(edges, families, layout):
        assume(False)
    intercept = {child: draw(st.booleans()) for child in endogenous}
    layout, n_units, n_time, n_obs = _shape(draw, layout, panel_shapes, cross_section_n)
    return _assemble(
        seed=seed,
        edges=edges,
        families=families,
        pooling_kind=pooling_kind,
        slope_vars=slope_vars,
        layout=layout,
        n_units=n_units,
        n_time=n_time,
        n_obs=n_obs,
        intercept=intercept,
    )
