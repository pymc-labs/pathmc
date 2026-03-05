"""Introspection helpers for PathModel.

Provides human-readable views of model structure: DAG visualisation,
structural equations, and prior specifications. All work before sampling.
"""

from __future__ import annotations

import re

import graphviz

from pathmc.graph import GraphInfo
from pathmc.parse import Spec, Term, TransformCall

_GREEK = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "zeta": r"\zeta",
    "eta": r"\eta",
    "theta": r"\theta",
    "iota": r"\iota",
    "kappa": r"\kappa",
    "lambda": r"\lambda",
    "lam": r"\lambda",
    "mu": r"\mu",
    "nu": r"\nu",
    "xi": r"\xi",
    "pi": r"\pi",
    "rho": r"\rho",
    "sigma": r"\sigma",
    "tau": r"\tau",
    "upsilon": r"\upsilon",
    "phi": r"\phi",
    "chi": r"\chi",
    "psi": r"\psi",
    "omega": r"\omega",
}

_GREEK_PATTERN = re.compile(
    r"^(" + "|".join(sorted(_GREEK.keys(), key=len, reverse=True)) + r")(?:_(.+))?$"
)


def _latexify_name(name: str) -> str:
    """Convert a parameter name to LaTeX, recognising Greek prefixes.

    Examples: ``theta_tv`` -> ``\\theta_{tv}``, ``b_tv`` -> ``b_{tv}``,
    ``sigma`` -> ``\\sigma``.
    """
    m = _GREEK_PATTERN.match(name)
    if m:
        base = _GREEK[m.group(1)]
        suffix = m.group(2)
        if suffix:
            return rf"{base}_{{{suffix}}}"
        return base
    if "_" in name:
        head, tail = name.split("_", 1)
        return rf"{head}_{{{tail}}}"
    return name


class EquationList:
    """Human-readable structural equations.

    Provides a readable ``__str__`` for plain text and ``_repr_latex_``
    for rich LaTeX rendering in Jupyter / Quarto notebooks.
    """

    def __init__(self, equations: list[str], latex_lines: list[str]) -> None:
        self._equations = equations
        self._latex_lines = latex_lines

    def __str__(self) -> str:
        return "\n".join(self._equations)

    def __repr__(self) -> str:
        return f"EquationList({self._equations!r})"

    def _repr_latex_(self) -> str:
        """LaTeX rendering for Jupyter/Quarto display."""
        sep = " \\\\\n"
        inner = sep.join(self._latex_lines)
        return f"$$\n\\begin{{aligned}}\n{inner}\n\\end{{aligned}}\n$$"

    def __iter__(self):
        return iter(self._equations)

    def __len__(self) -> int:
        return len(self._equations)


class PriorTable:
    """Summary of prior distributions for all model parameters.

    Provides a readable ``__str__`` table and ``_repr_latex_`` for rich
    LaTeX rendering in Jupyter / Quarto notebooks.
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

    def _repr_latex_(self) -> str:
        """LaTeX rendering for Jupyter/Quarto display."""
        lines: list[str] = []
        for name, prior_str in self._entries.items():
            lhs = _latexify_name(name)
            rhs = _latexify_prior(prior_str)
            lines.append(rf"{lhs} &\sim {rhs}")
        sep = " \\\\\n"
        inner = sep.join(lines)
        return f"$$\n\\begin{{aligned}}\n{inner}\n\\end{{aligned}}\n$$"


def _latexify_prior(prior_str: str) -> str:
    r"""Convert a prior string like ``Normal(0, 10)`` to LaTeX.

    Returns e.g. ``\text{Normal}(0,\, 10)``.
    """
    m = re.match(r"(\w+)\((.+)\)$", prior_str)
    if not m:
        return rf"\text{{{prior_str}}}"
    dist_name = m.group(1)
    args = m.group(2)
    args_latex = r",\, ".join(a.strip() for a in args.split(","))
    return rf"\text{{{dist_name}}}({args_latex})"


def build_dag_viz(spec: Spec, graph_info: GraphInfo) -> graphviz.Digraph:
    """Build a graphviz Digraph from the model's DAG.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    graph_info : GraphInfo
        Graph with edge information and latent set.

    Returns
    -------
    graphviz.Digraph
        Renderable DAG for notebook display. Latent nodes are drawn
        with dashed borders to distinguish them from observed nodes.
    """
    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="LR")

    for node in graph_info.topological_order:
        if node in graph_info.latent:
            dot.node(node, shape="ellipse", style="dashed")
        elif node in graph_info.endogenous:
            dot.node(node, shape="ellipse")
        else:
            dot.node(node, shape="box")

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


def build_equations(spec: Spec, latent: set[str] | None = None) -> EquationList:
    """Build human-readable equations from the parsed specification.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    latent : set[str] | None
        Latent variable names. Equations for these are annotated with
        ``[deterministic]`` to indicate no likelihood.

    Returns
    -------
    EquationList
        Iterable, printable list of equation strings with LaTeX rendering.
    """
    if latent is None:
        latent = set()

    lines: list[str] = []
    latex_lines: list[str] = []
    for reg in spec.regressions:
        terms: list[str] = []
        latex_terms: list[str] = []
        if reg.has_intercept:
            terms.append("1")
            latex_terms.append(rf"\beta_{{0,\,{reg.lhs}}}")
        for t in reg.terms:
            terms.append(_format_term(t))
            latex_terms.append(_format_term_latex(t))

        suffix = " [deterministic]" if reg.lhs in latent else ""
        lines.append(f"{reg.lhs} ~ {' + '.join(terms)}{suffix}")

        if reg.lhs in latent:
            latex_lines.extend(
                _build_equation_latex(reg.lhs, latex_terms, deterministic=True)
            )
        else:
            latex_lines.extend(_build_equation_latex(reg.lhs, latex_terms))
    for dp in spec.defined_params:
        lines.append(f"{dp.name} := {dp.expression}")
        latex_lines.append(
            rf"{_latexify_name(dp.name)} &\equiv {_latexify_expression(dp.expression)}"
        )
    return EquationList(lines, latex_lines)


def _format_term(t: Term) -> str:
    """Format a term for equation display, including transform expressions."""
    if t.transform is not None:
        expr = _format_transform(t.transform)
        if t.label:
            return f"{t.label}*{expr}"
        return expr
    if t.label:
        return f"{t.label}*{t.variable}"
    return t.variable


_MULTILINE_THRESHOLD = 4


def _build_equation_latex(
    lhs: str,
    latex_terms: list[str],
    deterministic: bool = False,
) -> list[str]:
    """Build LaTeX lines for one equation, splitting long ones across lines.

    Short equations (fewer than ``_MULTILINE_THRESHOLD`` terms) stay on a
    single line. Longer ones put each term on its own line, aligned on ``+``.
    Deterministic equations omit the error term.
    """
    lhs_latex = rf"\mathrm{{{lhs}}}"

    if deterministic:
        if len(latex_terms) < _MULTILINE_THRESHOLD:
            rhs = " + ".join(latex_terms)
            return [rf"{lhs_latex} &\equiv {rhs}"]

        result: list[str] = []
        for i, term in enumerate(latex_terms):
            if i == 0:
                result.append(rf"{lhs_latex} &\equiv {term}")
            else:
                result.append(rf"&\quad + {term}")
        return result

    error = rf"\varepsilon_{{{lhs}}}"

    if len(latex_terms) < _MULTILINE_THRESHOLD:
        rhs = " + ".join(latex_terms) + f" + {error}"
        return [rf"{lhs_latex} &= {rhs}"]

    result = []
    for i, term in enumerate(latex_terms):
        if i == 0:
            result.append(rf"{lhs_latex} &= {term}")
        else:
            result.append(rf"&\quad + {term}")
    result[-1] += f" + {error}"
    return result


def _format_term_latex(t: Term) -> str:
    """Format a term as LaTeX, including transform expressions."""
    if t.transform is not None:
        expr = _format_transform_latex(t.transform)
        if t.label:
            return rf"{_latexify_name(t.label)} \cdot {expr}"
        return expr
    if t.label:
        return rf"{_latexify_name(t.label)} \cdot \mathrm{{{t.variable}}}"
    return rf"\mathrm{{{t.variable}}}"


def _format_transform(tc: TransformCall) -> str:
    """Recursively format a TransformCall as a string."""
    if isinstance(tc.input_expr, TransformCall):
        input_str = _format_transform(tc.input_expr)
    else:
        input_str = tc.input_expr
    param_strs = [f"{k}={v}" for k, v in tc.params.items()]
    all_args = [input_str] + param_strs
    return f"{tc.name}({', '.join(all_args)})"


def _format_transform_latex(tc: TransformCall) -> str:
    """Recursively format a TransformCall as LaTeX."""
    if isinstance(tc.input_expr, TransformCall):
        input_str = _format_transform_latex(tc.input_expr)
    else:
        input_str = rf"\mathrm{{{tc.input_expr}}}"
    param_strs = [rf"{_latexify_name(v)}" for v in tc.params.values()]
    all_args = [input_str] + param_strs
    name_latex = rf"\operatorname{{{tc.name}}}"
    joined = r",\, ".join(all_args)
    return rf"{name_latex}({joined})"


def _latexify_expression(expr: str) -> str:
    """Convert a defined-parameter expression like ``a*b`` to LaTeX."""
    tokens = re.split(r"(\*|\+|-|/)", expr)
    parts: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token == "*":
            parts.append(r" \cdot ")
        elif token in ("+", "-", "/"):
            parts.append(f" {token} ")
        else:
            parts.append(_latexify_name(token))
    return "".join(parts)


def build_priors(
    spec: Spec,
    families: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
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
    latent : set[str] | None
        Latent variables (no sigma/likelihood priors emitted).

    Returns
    -------
    PriorTable
        Printable summary of all parameter priors.
    """
    if families is None:
        families = {}
    if latent is None:
        latent = set()

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

        if reg.lhs not in latent:
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
