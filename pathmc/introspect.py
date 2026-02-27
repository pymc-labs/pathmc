"""Introspection helpers for PathModel.

Provides human-readable views of model structure: DAG visualisation,
structural equations, and prior specifications. All work before sampling.
"""

from __future__ import annotations


import graphviz

from pathmc.graph import GraphInfo
from pathmc.parse import Spec


class EquationList:
    """Human-readable structural equations.

    Provides a readable ``__str__`` listing each regression equation
    in the model.
    """

    def __init__(self, equations: list[str]) -> None:
        self._equations = equations

    def __str__(self) -> str:
        return "\n".join(self._equations)

    def __repr__(self) -> str:
        return f"EquationList({self._equations!r})"

    def __iter__(self):
        return iter(self._equations)

    def __len__(self) -> int:
        return len(self._equations)


class PriorTable:
    """Summary of prior distributions for all model parameters.

    Provides a readable ``__str__`` table of priors keyed by parameter name.
    """

    def __init__(self, entries: dict[str, str]) -> None:
        self._entries = entries

    def __str__(self) -> str:
        lines = []
        for name, prior_str in self._entries.items():
            lines.append(f"  {name}: {prior_str}")
        return "Priors:\n" + "\n".join(lines)

    def __repr__(self) -> str:
        return f"PriorTable({self._entries!r})"


def build_dag_viz(spec: Spec, graph_info: GraphInfo) -> graphviz.Digraph:
    """Build a graphviz Digraph from the model's DAG.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    graph_info : GraphInfo
        Graph with edge information.

    Returns
    -------
    graphviz.Digraph
        Renderable DAG for notebook display.
    """
    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="LR")

    for node in graph_info.topological_order:
        shape = "ellipse" if node in graph_info.endogenous else "box"
        dot.node(node, shape=shape)

    for reg in spec.regressions:
        for term in reg.terms:
            label = term.label if term.label else ""
            dot.edge(term.variable, reg.lhs, label=label)

    for rc in spec.residual_covs:
        dot.edge(rc.var1, rc.var2, style="dashed", dir="both", label="~~")

    return dot


def build_equations(spec: Spec) -> EquationList:
    """Build human-readable equations from the parsed specification.

    Parameters
    ----------
    spec : Spec
        Parsed specification.

    Returns
    -------
    EquationList
        Iterable, printable list of equation strings.
    """
    lines: list[str] = []
    for reg in spec.regressions:
        terms = []
        if reg.has_intercept:
            terms.append("1")
        for t in reg.terms:
            if t.label:
                terms.append(f"{t.label}*{t.variable}")
            else:
                terms.append(t.variable)
        lines.append(f"{reg.lhs} ~ {' + '.join(terms)}")
    return EquationList(lines)


def build_priors(spec: Spec) -> PriorTable:
    """Build a prior summary table from the model specification.

    Parameters
    ----------
    spec : Spec
        Parsed specification.

    Returns
    -------
    PriorTable
        Printable summary of all parameter priors.
    """
    entries: dict[str, str] = {}
    for reg in spec.regressions:
        entries[f"beta_{reg.lhs}"] = "Normal(0, 10)"
        entries[f"sigma_{reg.lhs}"] = "HalfNormal(1)"
    return PriorTable(entries)
