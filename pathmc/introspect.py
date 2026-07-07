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
"""Introspection helpers for PathModel.

Provides human-readable views of model structure: DAG visualisation,
structural equations, and prior specifications. All work before sampling.
"""

from __future__ import annotations

import re

import graphviz

from pathmc.graph import GraphInfo
from pathmc.parse import Spec, Term, TransformCall

__all__: list[str] = []

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


class ModelEquations:
    """Combined structural equations and prior specifications.

    Provides a unified ``__str__`` for plain text and ``_repr_latex_``
    for rich LaTeX rendering in Jupyter / Quarto notebooks.
    """

    def __init__(self, equations: EquationList, priors: PriorTable) -> None:
        self._equations = equations
        self._priors = priors

    def __str__(self) -> str:
        return f"{self._priors}\n\n{self._equations}"

    def __repr__(self) -> str:
        return f"ModelEquations(equations={self._equations!r}, priors={self._priors!r})"

    def _repr_latex_(self) -> str:
        """LaTeX rendering combining priors and structural equations."""
        eq_sep = " \\\\\n"

        prior_lines: list[str] = []
        for name, prior_str in self._priors._entries.items():
            lhs = _latexify_name(name)
            rhs = _latexify_prior(prior_str)
            prior_lines.append(rf"{lhs} &\sim {rhs}")
        pr_inner = eq_sep.join(prior_lines)

        eq_inner = eq_sep.join(self._equations._latex_lines)

        return (
            f"$$\n\\begin{{aligned}}\n"
            f"{pr_inner}\n"
            f"\\\\[6pt]\n"
            f"{eq_inner}\n"
            f"\\end{{aligned}}\n$$"
        )


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


def build_dag_viz(
    spec: Spec,
    graph_info: GraphInfo,
    families: dict[str, str] | None = None,
) -> graphviz.Digraph:
    """Build a graphviz Digraph from the model's DAG.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    graph_info : GraphInfo
        Graph with edge information and latent set.
    families : dict[str, str] | None
        Per-variable distribution families. Stochastic latent nodes
        (``"latent_normal"``) are drawn with dashed + bold borders;
        deterministic latent nodes use dashed borders only.

    Returns
    -------
    graphviz.Digraph
        Renderable DAG for notebook display. Latent nodes are drawn
        with dashed borders to distinguish them from observed nodes.
    """
    if families is None:
        families = {}

    dot = graphviz.Digraph(format="svg")
    dot.attr(rankdir="LR")

    lag_nodes = {v for _, v in graph_info.temporal_edges}

    for node in graph_info.topological_order:
        if node in lag_nodes:
            continue
        elif node in graph_info.latent:
            if families.get(node) == "latent_normal":
                dot.node(node, shape="ellipse", style="dashed,bold")
            else:
                dot.node(node, shape="ellipse", style="dashed")
        elif node in graph_info.endogenous:
            dot.node(node, shape="ellipse")
        else:
            dot.node(node, shape="box")

    drawn_edges: set[tuple[str, str]] = set()
    for reg in spec.regressions:
        for term in reg.terms:
            if term.interaction_of is not None:
                for var in term.interaction_of:
                    if (var, reg.lhs) not in drawn_edges:
                        dot.edge(var, reg.lhs)
                        drawn_edges.add((var, reg.lhs))
                continue
            if term.fixed_value is not None:
                fv = (
                    int(term.fixed_value)
                    if term.fixed_value == int(term.fixed_value)
                    else term.fixed_value
                )
                edge_label = str(fv)
            else:
                edge_label = term.label or ""
            if term.transform is not None:
                edge_label = _format_transform(term.transform)
                if term.label:
                    edge_label = f"{term.label}*{edge_label}"
            if term.lag_of is not None:
                lag_label = f"{edge_label} (t\u22121)" if edge_label else "t\u22121"
                dot.edge(term.lag_of, reg.lhs, label=lag_label, style="dashed")
            else:
                dot.edge(term.variable, reg.lhs, label=edge_label)
            drawn_edges.add((term.variable, reg.lhs))

    for rc in spec.residual_covs:
        dot.edge(rc.var1, rc.var2, style="dashed", dir="both", label="~~")

    return dot


def build_equations(
    spec: Spec,
    latent: set[str] | None = None,
    families: dict[str, str] | None = None,
) -> EquationList:
    """Build human-readable equations from the parsed specification.

    Parameters
    ----------
    spec : Spec
        Parsed specification.
    latent : set[str] | None
        Latent variable names. Deterministic latent equations are
        annotated ``[deterministic]``; stochastic latent (family
        ``latent_normal``) are annotated ``[stochastic]``.
    families : dict[str, str] | None
        Per-variable distribution families, used to distinguish
        stochastic latent nodes (``"latent_normal"``) from
        deterministic latent nodes.

    Returns
    -------
    EquationList
        Iterable, printable list of equation strings with LaTeX rendering.
    """
    if latent is None:
        latent = set()
    if families is None:
        families = {}

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

        is_stochastic_latent = (
            reg.lhs in latent and families.get(reg.lhs) == "latent_normal"
        )
        is_deterministic_latent = reg.lhs in latent and not is_stochastic_latent
        family = families.get(reg.lhs, "gaussian")

        if is_deterministic_latent:
            lines.append(f"{reg.lhs} = {' + '.join(terms)} [deterministic]")
            latex_lines.extend(
                _build_equation_latex(reg.lhs, latex_terms, deterministic=True)
            )
        else:
            lines.append(f"mu_{reg.lhs} = {' + '.join(terms)}")
            likelihood = _likelihood_text(reg.lhs, family)
            if is_stochastic_latent:
                likelihood += " [stochastic]"
            lines.append(likelihood)
            latex_lines.extend(
                _build_equation_latex(reg.lhs, latex_terms, family=family)
            )
    for dp in spec.defined_params:
        lines.append(f"{dp.name} := {dp.expression}")
        latex_lines.append(
            rf"{_latexify_name(dp.name)} &\equiv {_latexify_expression(dp.expression)}"
        )
    return EquationList(lines, latex_lines)


def _format_term(t: Term) -> str:
    """Format a term for equation display, including transform expressions."""
    prefix = ""
    if t.fixed_value is not None:
        fv = (
            int(t.fixed_value) if t.fixed_value == int(t.fixed_value) else t.fixed_value
        )
        prefix = f"{fv}*"
    elif t.label:
        prefix = f"{t.label}*"

    if t.hsgp is not None:
        return f"f_hsgp({t.hsgp.variable})"
    if t.transform is not None:
        return f"{prefix}{_format_transform(t.transform)}"
    if t.interaction_of is not None:
        return f"{prefix}{' × '.join(t.interaction_of)}"
    return f"{prefix}{t.variable}" if prefix else t.variable


_MULTILINE_THRESHOLD = 4


def _likelihood_latex(lhs: str, family: str) -> str:
    """Build the distributional likelihood line in LaTeX."""
    lhs_latex = rf"\mathrm{{{lhs}}}"
    mu = rf"\mu_{{{lhs}}}"
    sigma = rf"\sigma_{{{lhs}}}"

    if family == "bernoulli":
        return rf"{lhs_latex} &\sim \text{{Bernoulli}}(\text{{logit}}^{{-1}}({mu}))"
    if family == "poisson":
        return rf"{lhs_latex} &\sim \text{{Poisson}}(\exp({mu}))"
    if family == "negbinomial":
        alpha = rf"\alpha_{{\text{{disp}},\,{lhs}}}"
        return rf"{lhs_latex} &\sim \text{{NegBinomial}}(\exp({mu}),\, {alpha})"
    if family == "studentt":
        nu = rf"\nu_{{{lhs}}}"
        return rf"{lhs_latex} &\sim \text{{StudentT}}({nu},\, {mu},\, {sigma})"
    return rf"{lhs_latex} &\sim \text{{Normal}}({mu},\, {sigma})"


def _likelihood_text(lhs: str, family: str) -> str:
    """Build the distributional likelihood line in plain text."""
    mu = f"mu_{lhs}"
    sigma = f"sigma_{lhs}"

    if family == "bernoulli":
        return f"{lhs} ~ Bernoulli(logit_inv({mu}))"
    if family == "poisson":
        return f"{lhs} ~ Poisson(exp({mu}))"
    if family == "negbinomial":
        return f"{lhs} ~ NegBinomial(exp({mu}), alpha_disp_{lhs})"
    if family == "studentt":
        return f"{lhs} ~ StudentT(nu_{lhs}, {mu}, {sigma})"
    return f"{lhs} ~ Normal({mu}, {sigma})"


def _build_equation_latex(
    lhs: str,
    latex_terms: list[str],
    deterministic: bool = False,
    family: str = "gaussian",
) -> list[str]:
    r"""Build LaTeX lines for one equation using :math:`\mu` + likelihood form.

    Observed and stochastic equations produce a :math:`\mu` definition line
    (linear predictor) and a distributional likelihood line.
    Deterministic equations produce a single :math:`\equiv` line.
    Long linear predictors are split across multiple lines.
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

    mu_lhs = rf"\mu_{{{lhs}}}"

    if len(latex_terms) < _MULTILINE_THRESHOLD:
        rhs = " + ".join(latex_terms)
        result = [rf"{mu_lhs} &= {rhs}"]
    else:
        result = []
        for i, term in enumerate(latex_terms):
            if i == 0:
                result.append(rf"{mu_lhs} &= {term}")
            else:
                result.append(rf"&\quad + {term}")

    result.append(_likelihood_latex(lhs, family))
    return result


def _format_term_latex(t: Term) -> str:
    """Format a term as LaTeX, including transform expressions."""
    if t.fixed_value is not None:
        fv = (
            int(t.fixed_value) if t.fixed_value == int(t.fixed_value) else t.fixed_value
        )
        prefix = rf"{fv} \cdot "
    elif t.label:
        prefix = rf"{_latexify_name(t.label)} \cdot "
    else:
        prefix = ""

    if t.hsgp is not None:
        return rf"f_{{\mathrm{{hsgp}}}}(\mathrm{{{t.hsgp.variable}}})"
    if t.transform is not None:
        return f"{prefix}{_format_transform_latex(t.transform)}"
    if t.interaction_of is not None:
        parts = [rf"\mathrm{{{v}}}" for v in t.interaction_of]
        expr = r" \times ".join(parts)
        return f"{prefix}{expr}"
    var_expr = rf"\mathrm{{{t.variable}}}"
    return f"{prefix}{var_expr}" if prefix else var_expr


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


def _prior_to_str(prior: object) -> str:
    """Format a ``Prior`` object as a compact string for display."""
    dist = getattr(prior, "distribution", "?")
    params = getattr(prior, "parameters", {})
    parts = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{dist}({parts})"


def build_priors(
    spec: Spec,
    families: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    prior_config: dict[str, object] | None = None,
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
    prior_config : dict[str, Prior] | None
        Resolved prior configuration. If provided, entries are
        formatted from the actual ``Prior`` objects.

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

    from pathmc.compile import get_free_predictor_columns

    def _entry(key: str, default_str: str) -> str:
        if prior_config and key in prior_config:
            return _prior_to_str(prior_config[key])
        return default_str

    entries: dict[str, str] = {}
    seen_transform_params: set[str] = set()
    for reg in spec.regressions:
        if get_free_predictor_columns(reg):
            entries[f"beta_{reg.lhs}"] = _entry(f"beta_{reg.lhs}", "Normal(0, 10)")

        family = families.get(reg.lhs, "gaussian")
        if reg.lhs in latent:
            if family == "latent_normal":
                entries[f"sigma_{reg.lhs}"] = _entry(
                    f"sigma_{reg.lhs}", "HalfNormal(1)"
                )
        else:
            if family not in ("bernoulli", "poisson", "negbinomial"):
                entries[f"sigma_{reg.lhs}"] = _entry(
                    f"sigma_{reg.lhs}", "HalfNormal(1)"
                )
            if family == "negbinomial":
                entries[f"alpha_disp_{reg.lhs}"] = _entry(
                    f"alpha_disp_{reg.lhs}", "HalfNormal(1)"
                )
            if family == "studentt":
                entries[f"nu_{reg.lhs}"] = _entry(f"nu_{reg.lhs}", "Gamma(2, 0.1)")

        if has_intercepts:
            entries[f"mu_alpha_{reg.lhs}"] = _entry(
                f"mu_alpha_{reg.lhs}", "Normal(0, 10)"
            )
            entries[f"sigma_alpha_{reg.lhs}"] = _entry(
                f"sigma_alpha_{reg.lhs}", "HalfNormal(1)"
            )
            entries[f"alpha_{reg.lhs}"] = "Normal(mu_alpha, sigma_alpha)"
        for svar in slope_vars:
            term_variables = {t.variable for t in reg.terms}
            if svar in term_variables:
                entries[f"mu_slope_{reg.lhs}_{svar}"] = _entry(
                    f"mu_slope_{reg.lhs}_{svar}", "Normal(0, 10)"
                )
                entries[f"sigma_slope_{reg.lhs}_{svar}"] = _entry(
                    f"sigma_slope_{reg.lhs}_{svar}", "HalfNormal(1)"
                )
                entries[f"slope_{reg.lhs}_{svar}"] = "Normal(mu_slope, sigma_slope)"
        for term in reg.terms:
            if term.transform is not None:
                _collect_transform_priors(
                    term.transform, entries, seen_transform_params, prior_config
                )
            if term.hsgp is not None:
                var = term.hsgp.variable
                entries[f"ell_{reg.lhs}_{var}"] = _entry(
                    f"ell_{reg.lhs}_{var}", "InverseGamma(3, 1)"
                )
                entries[f"eta_{reg.lhs}_{var}"] = _entry(
                    f"eta_{reg.lhs}_{var}", "HalfNormal(1)"
                )
                # beta_hsgp is only a tunable prior in the non-centered
                # parametrization; in centered mode beta uses the data-derived
                # sqrt_psd scale, so it is intentionally not listed to match
                # what default_priors registers (tune ell/eta instead).
                if not term.hsgp.centered:
                    entries[f"beta_hsgp_{reg.lhs}_{var}"] = _entry(
                        f"beta_hsgp_{reg.lhs}_{var}", "Normal(0, 1)"
                    )

    if spec.residual_covs:
        import networkx as nx

        ug = nx.Graph()
        for rc in spec.residual_covs:
            ug.add_edge(rc.var1, rc.var2)
        for component in nx.connected_components(ug):
            block_name = "_".join(sorted(component))
            entries[f"chol_{block_name}"] = (
                "LKJCholeskyCov(eta=2, sd_dist=HalfNormal(1))"
            )

    return PriorTable(entries)


def _collect_transform_priors(
    tc: TransformCall,
    entries: dict[str, str],
    seen: set[str],
    prior_config: dict[str, object] | None = None,
) -> None:
    """Recursively add transform parameter priors to the entries dict."""
    from pathmc.transforms import get_transform

    if isinstance(tc.input_expr, TransformCall):
        _collect_transform_priors(tc.input_expr, entries, seen, prior_config)

    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in seen:
            seen.add(param_name)
            if prior_config and param_name in prior_config:
                entries[param_name] = _prior_to_str(prior_config[param_name])
            else:
                pspec = transform.param_specs[param_key]
                entries[param_name] = pspec.default_prior
