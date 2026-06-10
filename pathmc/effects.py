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
"""Labeled coefficient extraction and defined parameter evaluation.

Provides the logic behind ``PathModel.effects_summary()`` and
``PathModel.effect(path)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd

from pathmc.parse import Spec


def _has_labeled_terms(spec: Spec) -> bool:
    """Check whether any regression term has a user-supplied label."""
    return any(term.label is not None for reg in spec.regressions for term in reg.terms)


@dataclass
class EffectResult:
    """Posterior draws for a labeled or path-based effect."""

    name: str
    draws: np.ndarray

    @property
    def mean(self) -> float:
        """Posterior mean of the effect."""
        return float(np.mean(self.draws))

    @property
    def sd(self) -> float:
        """Posterior standard deviation of the effect."""
        return float(np.std(self.draws))

    def hdi(self, prob: float = 0.94) -> np.ndarray:
        """Highest density interval for the effect."""
        return az.hdi(self.draws, prob=prob)

    def __repr__(self) -> str:
        lo, hi = self.hdi()
        return (
            f"EffectResult('{self.name}', "
            f"mean={self.mean:.4f}, sd={self.sd:.4f}, "
            f"94% HDI=[{lo:.4f}, {hi:.4f}])"
        )


def extract_labeled_draws(
    spec: Spec,
    idata: az.InferenceData,
) -> dict[str, np.ndarray]:
    """Extract posterior draws for all labeled coefficients.

    Parameters
    ----------
    spec : Spec
        Parsed model specification with labeled terms.
    idata : az.InferenceData
        Posterior samples from MCMC.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from label name to flattened posterior draws.
    """
    labeled_draws: dict[str, np.ndarray] = {}

    for reg in spec.regressions:
        beta_name = f"beta_{reg.lhs}"
        coord_name = f"{reg.lhs}_predictors"

        for term in reg.terms:
            if term.label is not None:
                draws = (
                    idata
                    .posterior[beta_name]  # type: ignore[attr-defined]
                    .sel({coord_name: term.variable})
                    .values.flatten()
                )
                labeled_draws[term.label] = draws

    return labeled_draws


def evaluate_defined_params(
    spec: Spec,
    labeled_draws: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Evaluate ``:=`` expressions over posterior draws.

    Parameters
    ----------
    spec : Spec
        Parsed model specification containing defined parameter expressions.
    labeled_draws : dict[str, np.ndarray]
        Posterior draws for labeled coefficients.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from defined parameter name to computed draws.
    """
    defined_draws: dict[str, np.ndarray] = {}

    for dp in spec.defined_params:
        namespace: dict = {k: v for k, v in labeled_draws.items()}
        namespace.update(defined_draws)
        namespace["__builtins__"] = {}

        draws = eval(dp.expression, namespace)  # noqa: S307
        defined_draws[dp.name] = np.asarray(draws)

    return defined_draws


def build_effects_summary(
    spec: Spec,
    idata: az.InferenceData,
) -> pd.DataFrame:
    """Build a summary DataFrame of labeled coefficients and defined parameters.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    idata : az.InferenceData
        Posterior samples.

    Returns
    -------
    pd.DataFrame
        Summary with mean, sd, hdi_3%, hdi_97% for each effect.
    """
    labeled_draws = extract_labeled_draws(spec, idata)
    defined_draws = evaluate_defined_params(spec, labeled_draws)

    all_draws = {**labeled_draws, **defined_draws}

    rows = []
    for name, draws in all_draws.items():
        hdi = az.hdi(draws, prob=0.94)
        rows.append({
            "name": name,
            "mean": float(np.mean(draws)),
            "sd": float(np.std(draws)),
            "hdi_3%": float(hdi[0]),
            "hdi_97%": float(hdi[1]),
        })

    if not rows:
        return pd.DataFrame(columns=["mean", "sd", "hdi_3%", "hdi_97%"]).rename_axis(
            "name"
        )
    return pd.DataFrame(rows).set_index("name")


def build_standardized_effects(
    spec: Spec,
    idata: az.InferenceData,
    data: pd.DataFrame,
    latent: set[str] | None = None,
) -> pd.DataFrame:
    """Compute stdyx-standardized coefficients from posterior draws.

    For each labeled coefficient on edge X -> Y, computes::

        stdyx = coef * sd(X) / sd(Y)

    This gives the expected change in Y (in SD units) per SD change in X.
    Edges involving latent variables (no observed SD) are skipped.

    Parameters
    ----------
    spec : Spec
        Parsed model specification with labeled terms.
    idata : az.InferenceData
        Posterior samples from MCMC.
    data : pd.DataFrame
        Observed data used to compute variable standard deviations.
    latent : set[str] | None
        Latent variable names (skipped for standardization).

    Returns
    -------
    pd.DataFrame
        Summary with columns: mean, sd, hdi_3%, hdi_97% of the
        standardized coefficient. Index is the label name.
    """
    if latent is None:
        latent = set()

    labeled_draws = extract_labeled_draws(spec, idata)

    rows = []
    for reg in spec.regressions:
        lhs = reg.lhs
        if lhs in latent or lhs not in data.columns:
            continue
        sd_y = float(data[lhs].std())
        if sd_y == 0:
            continue

        for term in reg.terms:
            if term.label is None or term.label not in labeled_draws:
                continue

            if term.interaction_of is not None:
                continue

            var = term.variable
            if var in latent or var not in data.columns:
                continue
            sd_x = float(data[var].std())
            if sd_x == 0:
                continue

            raw_draws = labeled_draws[term.label]
            std_draws = raw_draws * sd_x / sd_y
            hdi = az.hdi(std_draws, prob=0.94)
            rows.append({
                "name": term.label,
                "predictor": var,
                "outcome": lhs,
                "mean": float(np.mean(std_draws)),
                "sd": float(np.std(std_draws)),
                "hdi_3%": float(hdi[0]),
                "hdi_97%": float(hdi[1]),
            })

    if not rows:
        return pd.DataFrame(
            columns=["predictor", "outcome", "mean", "sd", "hdi_3%", "hdi_97%"]
        ).rename_axis("name")
    return pd.DataFrame(rows).set_index("name")


def compute_path_effect(
    path: str,
    spec: Spec,
    idata: az.InferenceData,
) -> EffectResult:
    """Compute the effect along a specified causal path.

    Parameters
    ----------
    path : str
        Path string like ``"X -> M -> Y"`` specifying the causal pathway.
    spec : Spec
        Parsed model specification.
    idata : az.InferenceData
        Posterior samples.

    Returns
    -------
    EffectResult
        Posterior draws for the path-specific effect.

    Raises
    ------
    ValueError
        If a node in the path is not endogenous or an edge does not exist.
    """
    nodes = [n.strip() for n in path.split("->")]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    reg_by_lhs = {r.lhs: r for r in spec.regressions}
    labeled_draws = extract_labeled_draws(spec, idata)

    edge_draws: list[np.ndarray] = []
    for source, target in edges:
        if target not in reg_by_lhs:
            raise ValueError(
                f"Variable '{target}' is not endogenous (no regression equation). "
                f"Cannot compute path effect through it."
            )
        reg = reg_by_lhs[target]
        matched_term = None
        for t in reg.terms:
            if t.variable == source:
                matched_term = t
                break

        if matched_term is None:
            raise ValueError(
                f"No direct edge from '{source}' to '{target}' in the model. "
                f"Check the path specification."
            )

        if matched_term.label is not None and matched_term.label in labeled_draws:
            draws = labeled_draws[matched_term.label]
        else:
            beta_name = f"beta_{target}"
            coord_name = f"{target}_predictors"
            draws = (
                idata.posterior[beta_name].sel({coord_name: source}).values.flatten()  # type: ignore[attr-defined]
            )

        edge_draws.append(draws)

    result_draws = edge_draws[0]
    for d in edge_draws[1:]:
        result_draws = result_draws * d

    return EffectResult(name=path, draws=result_draws)
