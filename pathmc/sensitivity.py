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
"""Sensitivity analysis for unmeasured confounding.

Implements a parametric approach to assess how robust a causal effect
estimate is to potential unmeasured confounding, following the tradition
of Rosenbaum (2002) and VanderWeele & Ding (2017).

The core idea: hypothesize an unmeasured confounder U that affects both
the treatment (with strength γ) and outcome (with strength δ). The
product γ × δ represents the confounding bias in the estimated ATE.
Sweeping over a grid of (γ, δ) values reveals how large confounding
would need to be to overturn the causal conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import arviz as az
import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure


@dataclass
class SensitivityResult:
    """Result of unmeasured confounding sensitivity analysis.

    Contains the observed ATE posterior draws and adjusted ATE values
    for a grid of hypothetical confounder strengths (γ, δ).

    The confounding model assumes a latent U with:

    - Effect γ on the treatment variable
    - Effect δ on the outcome variable
    - Confounding bias in the ATE = γ × δ

    Parameters
    ----------
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    observed_ate_draws : np.ndarray
        Posterior draws of the unadjusted ATE, shape ``(n_draws,)``.
    gamma_values : np.ndarray
        1D grid of γ (confounder → treatment) values.
    delta_values : np.ndarray
        1D grid of δ (confounder → outcome) values.
    adjusted_ate_mean : np.ndarray
        Mean bias-adjusted ATE at each grid point,
        shape ``(n_gamma, n_delta)``.
    prob_sign_change : np.ndarray
        Posterior probability that the ATE sign flips at each grid
        point, shape ``(n_gamma, n_delta)``.
    """

    outcome: str
    treatment: str
    observed_ate_draws: np.ndarray
    gamma_values: np.ndarray
    delta_values: np.ndarray
    adjusted_ate_mean: np.ndarray
    prob_sign_change: np.ndarray

    @property
    def observed_ate(self) -> float:
        """Posterior mean of the unadjusted ATE."""
        return float(np.mean(self.observed_ate_draws))

    @property
    def observed_ate_hdi(self) -> np.ndarray:
        """94% highest density interval of the unadjusted ATE."""
        return az.hdi(self.observed_ate_draws, prob=0.94)

    @property
    def tipping_point(self) -> float:
        """The γ × δ product that reduces the posterior mean ATE to zero.

        A confounder whose effects on treatment and outcome multiply to
        this value would exactly nullify the observed average treatment
        effect. The tipping point equals the observed mean ATE because
        adjusted ATE = observed ATE − γ × δ.
        """
        return self.observed_ate

    def __repr__(self) -> str:
        hdi = self.observed_ate_hdi
        tp = abs(self.tipping_point)
        sym = float(np.sqrt(tp)) if tp > 0 else 0.0
        return (
            f"SensitivityResult(treatment='{self.treatment}' → "
            f"outcome='{self.outcome}')\n"
            f"  Observed ATE: {self.observed_ate:.4f} "
            f"[{hdi[0]:.4f}, {hdi[1]:.4f}] (94% HDI)\n"
            f"  Tipping point: γ × δ = {self.tipping_point:.4f}\n"
            f"  (e.g., γ = {sym:.4f}, δ = {sym:.4f} would nullify the effect)"
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "RdBu_r",
        n_levels: int = 20,
    ) -> matplotlib.figure.Figure:
        """Contour plot of adjusted ATE vs. confounder strength.

        Shows how the estimated ATE changes as a function of the
        hypothetical confounder's effect on treatment (γ) and
        outcome (δ). A black contour line marks the tipping boundary
        where the adjusted ATE crosses zero.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. Creates a new figure if None.
        cmap : str
            Matplotlib colormap name (default ``"RdBu_r"``).
        n_levels : int
            Number of filled contour levels.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the contour plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = cast("matplotlib.figure.Figure", ax.get_figure())

        G, D = np.meshgrid(self.gamma_values, self.delta_values, indexing="ij")

        vmax = float(np.max(np.abs(self.adjusted_ate_mean)))
        cf = ax.contourf(
            G,
            D,
            self.adjusted_ate_mean,
            levels=n_levels,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(cf, ax=ax, label="Adjusted ATE")

        ate_min = float(np.min(self.adjusted_ate_mean))
        ate_max = float(np.max(self.adjusted_ate_mean))
        if ate_min < 0 < ate_max:
            cs = ax.contour(
                G,
                D,
                self.adjusted_ate_mean,
                levels=[0.0],
                colors="black",
                linewidths=2,
            )
            ax.clabel(cs, fmt="ATE=0", fontsize=10)

        g_lo, g_hi = self.gamma_values[0], self.gamma_values[-1]
        d_lo, d_hi = self.delta_values[0], self.delta_values[-1]
        if g_lo <= 0 <= g_hi and d_lo <= 0 <= d_hi:
            ax.plot(
                0,
                0,
                "k*",
                markersize=15,
                zorder=5,
                label=f"Observed ATE = {self.observed_ate:.3f}",
            )
            ax.legend(loc="best")

        ax.set_xlabel("γ (confounder → treatment)")
        ax.set_ylabel("δ (confounder → outcome)")
        ax.set_title(
            f"Sensitivity: {self.treatment} → {self.outcome}\n"
            f"Robustness to unmeasured confounding"
        )

        return fig


def compute_sensitivity(
    observed_ate_draws: np.ndarray,
    outcome: str,
    treatment: str,
    gamma_range: tuple[float, float],
    delta_range: tuple[float, float],
    n_grid: int,
) -> SensitivityResult:
    """Compute sensitivity analysis over a grid of confounder strengths.

    For each (γ, δ) pair on the grid, the adjusted ATE is::

        adjusted ATE = observed ATE − γ × δ

    where γ × δ is the confounding bias from an unmeasured confounder U
    with effect γ on the treatment and δ on the outcome. This follows
    the omitted variable bias framework: a latent common cause inflates
    the observed treatment effect by the product of its effects on
    treatment and outcome.

    Parameters
    ----------
    observed_ate_draws : np.ndarray
        1D array of posterior ATE draws.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    gamma_range : tuple[float, float]
        ``(min, max)`` range for γ values.
    delta_range : tuple[float, float]
        ``(min, max)`` range for δ values.
    n_grid : int
        Number of grid points per dimension.

    Returns
    -------
    SensitivityResult
        Sensitivity analysis results.
    """
    gamma_vals = np.linspace(gamma_range[0], gamma_range[1], n_grid)
    delta_vals = np.linspace(delta_range[0], delta_range[1], n_grid)

    G, D = np.meshgrid(gamma_vals, delta_vals, indexing="ij")
    bias_grid = G * D

    observed_mean = float(np.mean(observed_ate_draws))
    adjusted_mean = observed_mean - bias_grid

    sorted_draws = np.sort(observed_ate_draws)
    n_draws = len(sorted_draws)
    flat_bias = bias_grid.ravel()

    if abs(observed_mean) < 1e-15:
        prob_sign_flat = np.full(flat_bias.shape, 0.5, dtype=float)
    elif observed_mean > 0:
        idx = np.searchsorted(sorted_draws, flat_bias).astype(float)
        prob_sign_flat = idx / n_draws
    else:
        idx = np.searchsorted(sorted_draws, flat_bias, side="right").astype(float)
        prob_sign_flat = 1.0 - idx / n_draws

    prob_sign = prob_sign_flat.reshape(G.shape)

    return SensitivityResult(
        outcome=outcome,
        treatment=treatment,
        observed_ate_draws=observed_ate_draws,
        gamma_values=gamma_vals,
        delta_values=delta_vals,
        adjusted_ate_mean=adjusted_mean,
        prob_sign_change=prob_sign,
    )
