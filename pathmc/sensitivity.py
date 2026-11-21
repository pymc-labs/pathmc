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

Applies hypothetical additive bias shifts γ × δ to fitted ATE draws.
In a linear omitted-variable-bias identity, δ is the outcome coefficient
of an unmeasured U conditional on treatment and observed covariates,
and γ is the treatment coefficient in the auxiliary regression of U
on treatment and those covariates. γ is not generally U's forward
causal effect on treatment; unit variance U alone does not equate them.

The supplied product must be calibrated to the reported intervention
contrast. This module neither estimates that calibration nor rescales
the bias for non-unit contrasts. The linear coefficient identity is exact;
shifting Bayesian draws is not an exact refit and does not propagate
uncertainty in the sensitivity parameters. Nonlinear causal effects need
an appropriate bias model before this shift has a causal interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from pathmc.idata import hdi, hdi_label
from pathmc.reprs import ReprSpec, ResultReprMixin

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["SensitivityResult"]


@dataclass(repr=False)
class SensitivityResult(ResultReprMixin):
    """Result of unmeasured confounding sensitivity analysis.

    Contains the observed ATE posterior draws and adjusted ATE values
    for a grid of hypothetical confounder strengths (γ, δ).

    γ and δ specify a hypothetical fixed bias product on the ATE's
    contrast scale. For an exact linear bias interpretation, use the
    auxiliary-regression calibration described in this module, not an
    unqualified product of forward causal effects. Shifted draws do not
    include uncertainty about these sensitivity parameters.

    Parameters
    ----------
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name.
    observed_ate_draws : np.ndarray
        Posterior draws of the unadjusted ATE, shape ``(n_draws,)``.
    gamma_values : np.ndarray
        1D grid of γ sensitivity values, calibrated to the bias scale.
    delta_values : np.ndarray
        1D grid of δ sensitivity values, calibrated to the bias scale.
    adjusted_ate_mean : np.ndarray
        Mean bias-adjusted ATE at each grid point,
        shape ``(n_gamma, n_delta)``.
    prob_sign_change : np.ndarray
        Fraction of draws whose bias-adjusted ATE does not have the
        observed mean's sign, at each grid point. An adjusted value of
        exactly 0 counts as a change. Shape ``(n_gamma, n_delta)``.
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
        """Highest density interval of the unadjusted ATE at the default probability mass."""
        return hdi(self.observed_ate_draws)

    @property
    def tipping_point(self) -> float:
        """The γ × δ product that reduces the posterior mean ATE to zero.

        A hypothetical fixed bias equal to this value exactly nullifies
        the fitted posterior mean under adjusted ATE = observed ATE − γ × δ.
        This numerical threshold does not identify a confounder or imply
        that the shifted posterior is an exact adjusted posterior.
        """
        return self.observed_ate

    def _repr_compact(self) -> str:
        tp = self.tipping_point
        return (
            f"SensitivityResult({self.treatment}→{self.outcome}, "
            f"ATE={self.observed_ate:.4f}, tipping_point={tp:.4f})"
        )

    def _repr_spec(self) -> ReprSpec:
        hdi_vals = self.observed_ate_hdi
        gamma, delta = _symmetric_nullifying_pair(self.tipping_point)
        # adjusted ATE = observed ATE − γ × δ. A same-sign pair for a
        # negative tipping point is off by about 2|ATE|; a correctly
        # signed pair is off by at most a few ulps. Only the latter is 0.
        adjusted = self.observed_ate - gamma * delta
        scale = abs(self.tipping_point)
        nullifies = abs(adjusted) <= 8 * np.spacing(scale if scale else 1.0)
        claim = " would nullify the effect" if nullifies else ""
        label = hdi_label()
        return ReprSpec(
            title=f"Sensitivity Analysis — {self.treatment} → {self.outcome}",
            rows=[
                [
                    "Observed ATE",
                    f"{self.observed_ate:.4f} [{hdi_vals[0]:.4f}, {hdi_vals[1]:.4f}] ({label})",
                ],
                ["Tipping point", f"γ × δ = {self.tipping_point:.4f}"],
                [
                    "Symmetric example",
                    f"γ = {gamma:.4f}, δ = {delta:.4f}{claim}",
                ],
            ],
            footer="Methods: .plot() &nbsp;·&nbsp; Attributes: .tipping_point .observed_ate_hdi",
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        cmap: str = "RdBu_r",
        n_levels: int = 20,
    ) -> matplotlib.figure.Figure:
        """Contour plot of adjusted ATE vs. confounder strength.

        Shows how the estimated ATE changes as a function of the
        supplied sensitivity parameters γ and δ. The arrow labels are
        causal intuition; a linear bias interpretation requires the
        calibration described in this module. A black contour line marks
        the tipping boundary
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


def _symmetric_nullifying_pair(tipping_point: float) -> tuple[float, float]:
    """Return equal-magnitude (γ, δ) whose product is ``tipping_point``.

    A positive tipping point keeps both coordinates positive, so the
    displayed line is unchanged. A negative tipping point puts the sign
    on δ so the product has the tipping point's sign. Division by γ
    keeps that product within a few ulps, which is what the summary
    treats as a nullifying pair.
    """
    if tipping_point == 0.0:
        return 0.0, 0.0
    gamma = float(np.sqrt(abs(tipping_point)))
    delta = float(tipping_point / gamma)
    return gamma, delta


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

    γ × δ is a user-specified fixed bias on the reported ATE contrast
    scale. An exact linear coefficient interpretation requires the
    auxiliary-regression calibration described in this module. No
    variance normalization or intervention-contrast rescaling is applied,
    and no uncertainty about γ or δ is propagated.

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

    # Fraction of draws whose adjusted ATE (draw − bias) does not have
    # the observed mean's sign. Exactly 0 counts as a change. The mean's
    # sign is used as-is: a near-zero mean is not replaced by 0.5.
    if observed_mean > 0.0:
        # adjusted <= 0 iff draw <= bias
        idx = np.searchsorted(sorted_draws, flat_bias, side="right")
        prob_sign_flat = idx / n_draws
    elif observed_mean < 0.0:
        # adjusted >= 0 iff draw >= bias
        idx = np.searchsorted(sorted_draws, flat_bias, side="left")
        prob_sign_flat = 1.0 - idx / n_draws
    else:
        prob_sign_flat = np.ones(flat_bias.shape, dtype=float)

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
