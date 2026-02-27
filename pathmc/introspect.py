"""Introspection helpers for PathModel.

Provides human-readable views of model structure: DAG visualisation,
structural equations, and prior specifications. All work before sampling.
"""

from __future__ import annotations


import graphviz

from pathmc.graph import GraphInfo
from pathmc.parse import Spec, TransformCall


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
            edge_label = term.label or ""
            if term.transform is not None:
                edge_label = _format_transform(term.transform)
                if term.label:
                    edge_label = f"{term.label}*{edge_label}"
            dot.edge(term.variable, reg.lhs, label=edge_label)

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
            term_str = _format_term(t)
            terms.append(term_str)
        lines.append(f"{reg.lhs} ~ {' + '.join(terms)}")
    return EquationList(lines)


def _format_term(t: object) -> str:
    """Format a term for equation display, including transform expressions."""
    if t.transform is not None:
        expr = _format_transform(t.transform)
        if t.label:
            return f"{t.label}*{expr}"
        return expr
    if t.label:
        return f"{t.label}*{t.variable}"
    return t.variable


def _format_transform(tc: TransformCall) -> str:
    """Recursively format a TransformCall as a string."""
    if isinstance(tc.input_expr, TransformCall):
        input_str = _format_transform(tc.input_expr)
    else:
        input_str = tc.input_expr
    param_strs = [f"{k}={v}" for k, v in tc.params.items()]
    all_args = [input_str] + param_strs
    return f"{tc.name}({', '.join(all_args)})"


def build_priors(
    spec: Spec,
    families: dict[str, str] | None = None,
    pooling: str | dict | None = None,
) -> PriorTable:
    """Build a prior summary table from the model specification.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    families : dict[str, str] | None
        Per-variable distribution families.
    pooling : str | dict | None
        Pooling specification for panel models.

    Returns
    -------
    PriorTable
        Printable summary of all parameter priors.
    """
    if families is None:
        families = {}

    has_intercepts = pooling == "partial" or (
        isinstance(pooling, dict) and pooling.get("intercept", False)
    )
    slope_vars: list[str] = []
    if isinstance(pooling, dict):
        slope_vars = list(pooling.get("slopes", []))

    entries: dict[str, str] = {}
    seen_transform_params: set[str] = set()
    for reg in spec.regressions:
        entries[f"beta_{reg.lhs}"] = "Normal(0, 10)"
        family = families.get(reg.lhs, "gaussian")
        if family not in ("bernoulli", "poisson", "negbinomial"):
            entries[f"sigma_{reg.lhs}"] = "HalfNormal(1)"
        if family == "negbinomial":
            entries[f"alpha_disp_{reg.lhs}"] = "HalfNormal(1)"
        if family == "studentt":
            entries[f"nu_{reg.lhs}"] = "Gamma(2, 0.1)"
        if has_intercepts:
            entries[f"mu_alpha_{reg.lhs}"] = "Normal(0, 10)"
            entries[f"sigma_alpha_{reg.lhs}"] = "HalfNormal(1)"
            entries[f"alpha_{reg.lhs}"] = "Normal(mu_alpha, sigma_alpha)"
        for svar in slope_vars:
            term_variables = {t.variable for t in reg.terms}
            if svar in term_variables:
                entries[f"mu_slope_{reg.lhs}_{svar}"] = "Normal(0, 10)"
                entries[f"sigma_slope_{reg.lhs}_{svar}"] = "HalfNormal(1)"
                entries[f"slope_{reg.lhs}_{svar}"] = "Normal(mu_slope, sigma_slope)"
        for term in reg.terms:
            if term.transform is not None:
                _collect_transform_priors(
                    term.transform, entries, seen_transform_params
                )
    return PriorTable(entries)


def _collect_transform_priors(
    tc: TransformCall,
    entries: dict[str, str],
    seen: set[str],
) -> None:
    """Recursively add transform parameter priors to the entries dict."""
    from pathmc.transforms import get_transform

    if isinstance(tc.input_expr, TransformCall):
        _collect_transform_priors(tc.input_expr, entries, seen)

    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in seen:
            seen.add(param_name)
            pspec = transform.param_specs[param_key]
            entries[param_name] = pspec.default_prior
