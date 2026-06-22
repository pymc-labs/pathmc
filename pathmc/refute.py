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
"""Bayesian placebo-treatment refutation of an estimated effect.

Ports dowhy's ``placebo_treatment_refuter`` and upgrades it to a fully
Bayesian form. dowhy replaces the treatment with a permuted ("placebo")
copy, re-estimates a *point* effect, repeats this ``num_simulations``
times, and checks whether the resulting empirical distribution straddles
zero. Each pathmc re-fit instead yields a full posterior, so the spread
of placebo estimates carries two distinct uncertainties: the
within-permutation posterior uncertainty of a single fit, and the
between-permutation variation in where that posterior lands.

These are pooled with a hierarchical normal-normal random-effects model
(the calibration model of Bugaev & Trujillo, "Bayesian Design Analysis
for Time-Series Quasi-Experiments"): each placebo permutation is a "fold"
contributing a posterior mean ``m_j`` and posterior SD ``s_j`` of the
average treatment effect, and the model decomposes their dispersion into
a systematic placebo bias ``mu_null`` (which a valid pipeline keeps near
zero) and a structural volatility ``tau_het``. The null predictive
``theta_new ~ Normal(mu_null, tau_het)`` therefore centers near zero only
when the pipeline is sound (its center is the placebo bias ``mu_null``);
the real (observed) effect is then scored against it via a calibrated
``z`` statistic and tail probability — the post-estimation calibration
step dowhy lacks.

Reference
---------
Pearl/dowhy refutation API: https://www.pywhy.org/dowhy/. Hierarchical
calibration model: Bugaev, A., & Trujillo, C., "Bayesian Design Analysis
for Time-Series Quasi-Experiments: A Placebo-Calibrated Framework".
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pymc as pm

from pathmc.idata import hdi, posterior
from pathmc.reprs import ResultReprMixin

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from pathmc._model import PathModel

__all__ = ["PlaceboRefutationResult", "refute_placebo"]

# Floor for placebo-fold posterior SDs. A degenerate s_j = 0 would make the
# within-fold likelihood sigma zero and break the hierarchical fit, so it is
# clamped to a tiny positive value (matching the calibration reference code).
_MIN_FOLD_SD = 1e-6

# A placebo fold whose posterior SD exceeds this multiple of the median fold SD
# is flagged as a likely non-converged outlier that can dominate the calibration.
_SD_OUTLIER_FACTOR = 10.0


@dataclass(repr=False)
class PlaceboRefutationResult(ResultReprMixin):
    """Result of a Bayesian placebo-treatment refutation.

    Produced by :func:`refute_placebo` / :meth:`pathmc.PathModel.refute_placebo`.
    Holds the real effect's posterior, the per-permutation placebo
    summaries, the hierarchical null model's posterior, and the calibrated
    comparison of the real effect against the placebo null.

    The pipeline *passes* the placebo test when the systematic placebo
    bias ``mu_null`` credibly includes zero (:attr:`passes_placebo`): a
    sound estimator should attribute no effect to a treatment whose link
    to the outcome has been severed by permutation. The real effect
    *survives* the null when its calibrated tail probability is small
    (:attr:`effect_survives`).

    Parameters
    ----------
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable name (the permuted/placebo variable).
    observed_ate_draws : np.ndarray
        Posterior draws of the real ATE from the fitted model, shape
        ``(n_draws,)``.
    fold_means : np.ndarray
        Placebo-fold posterior means ``m_j``, shape ``(n_permutations,)``.
    fold_sds : np.ndarray
        Placebo-fold posterior SDs ``s_j`` (floored at ``1e-6``), shape
        ``(n_permutations,)``.
    mu_null_draws : np.ndarray
        Posterior draws of the systematic placebo bias ``mu_null``.
    tau_het_draws : np.ndarray
        Posterior draws of the structural volatility ``tau_het``.
    theta_new_draws : np.ndarray
        Draws from the null predictive distribution
        ``theta_new ~ Normal(mu_null, tau_het)``.
    z_cal : float
        Calibrated z statistic of the real ATE against the placebo null,
        ``(observed_ate - null_mean) / sqrt(null_sd**2 + mean(s_j**2))``,
        where ``null_mean`` and ``null_sd`` summarize the null predictive
        distribution ``theta_new`` (see :attr:`null_mean`, :attr:`null_sd`).
        ``null_sd`` (the predictive SD) is used rather than the bare
        ``tau_het`` parameter because it also absorbs posterior uncertainty
        in ``mu_null``, matching the calibration reference implementation.
    p_tail : float
        One-sided (directional) calibrated tail probability: the fraction
        of observable-null predictive draws at least as extreme, *in the
        direction of the observed effect*, as the observed ATE. The tail
        side is selected *post hoc* from ``sign(observed_ate - null_mean)``
        (effectively the sign of the observed effect, since the null
        straddles zero) — it is not a pre-registered side, so do not read
        it as a fixed one-sided hypothesis test. Floored at
        ``1 / (n_draws + 1)`` to reflect the Monte Carlo resolution rather
        than reporting an exact zero.
    n_permutations : int
        Number of placebo permutations (folds).
    significance_level : float
        Threshold for the :attr:`effect_survives` verdict (default 0.05).
    random_seed : int | None
        Seed used for the permutations and hierarchical fit.
    """

    outcome: str
    treatment: str
    observed_ate_draws: np.ndarray
    fold_means: np.ndarray
    fold_sds: np.ndarray
    mu_null_draws: np.ndarray
    tau_het_draws: np.ndarray
    theta_new_draws: np.ndarray
    z_cal: float
    p_tail: float
    n_permutations: int
    significance_level: float = 0.05
    random_seed: int | None = None

    @property
    def observed_ate(self) -> float:
        """Posterior mean of the real (observed) ATE."""
        return float(np.mean(self.observed_ate_draws))

    @property
    def observed_ate_hdi(self) -> np.ndarray:
        """94% highest density interval of the real ATE."""
        return hdi(self.observed_ate_draws)

    @property
    def estimated_effect(self) -> float:
        """The originally estimated ATE (dowhy's "estimated effect").

        Alias of :attr:`observed_ate`, named to mirror dowhy's
        ``CausalRefutation.estimated_effect``.
        """
        return self.observed_ate

    @property
    def new_effect(self) -> float:
        """The pooled placebo effect (dowhy's "new effect").

        The systematic placebo bias ``mu_null`` pooled across permutations
        — the effect the pipeline still reports once the treatment-outcome
        link is severed. A sound pipeline keeps this near zero. This is the
        Bayesian analogue of dowhy's ``new_effect = mean(placebo estimates)``,
        but pooled hierarchically and equipped with :attr:`mu_null_hdi`.
        """
        return self.mu_null

    @property
    def mu_null(self) -> float:
        """Posterior mean of the systematic placebo bias ``mu_null``."""
        return float(np.mean(self.mu_null_draws))

    @property
    def mu_null_hdi(self) -> np.ndarray:
        """94% highest density interval of the placebo bias ``mu_null``."""
        return hdi(self.mu_null_draws)

    @property
    def tau_het(self) -> float:
        """Posterior mean of the structural volatility ``tau_het``."""
        return float(np.mean(self.tau_het_draws))

    @property
    def null_mean(self) -> float:
        """Mean of the null predictive distribution (placebo effect center)."""
        return float(np.mean(self.theta_new_draws))

    @property
    def null_sd(self) -> float:
        """SD of the null predictive distribution (total placebo volatility)."""
        return float(np.std(self.theta_new_draws))

    @property
    def sigma_pred(self) -> float:
        """Predictive-null SD used in the ``z_cal`` denominator.

        ``sqrt(null_sd**2 + mean(s_j**2))``: the null-predictive spread
        convolved with the mean within-fold estimation *variance*. The mean
        of squares (not the square of the mean) is used so that ``z_cal``
        is coherent with the bootstrap ``p_tail``, which draws
        ``Normal(0, s_j)`` noise and thus has variance ``mean(s_j**2)``.
        """
        mean_within_var = float(np.mean(np.asarray(self.fold_sds, dtype=float) ** 2))
        return float(np.sqrt(self.null_sd**2 + mean_within_var))

    @property
    def passes_placebo(self) -> bool:
        """Whether the placebo null straddles zero.

        ``True`` when the 94% HDI of ``mu_null`` contains zero, i.e. the
        pipeline reports no systematic effect for a permuted treatment.
        A ``False`` verdict flags a pipeline that manufactures effects
        from noise.

        .. note::

            This is a pure interval rule with no region of practical
            equivalence (ROPE), so a *negligibly small but very precisely
            estimated* placebo bias can fail. Read it together with
            :attr:`mu_null` (the bias magnitude): a ``mu_null`` that is
            tiny relative to the observed effect is practically benign even
            when its HDI excludes zero.
        """
        lo, hi = self.mu_null_hdi
        return bool(lo <= 0.0 <= hi)

    @property
    def effect_survives(self) -> bool:
        """Whether the real effect is distinguishable from the placebo null.

        ``True`` when the calibrated tail probability :attr:`p_tail` is
        below :attr:`significance_level`: the observed effect is too
        extreme to be explained by placebo (structural) noise alone.
        """
        return bool(self.p_tail < self.significance_level)

    def _repr_compact(self) -> str:
        placebo = "PASS" if self.passes_placebo else "FAIL"
        survives = "survives" if self.effect_survives else "does-not-survive"
        return (
            f"PlaceboRefutationResult({self.treatment}→{self.outcome}: "
            f"placebo={placebo}, μ_null={self.mu_null:.3f}, "
            f"ATE={self.observed_ate:.3f}, p_tail={self.p_tail:.3f} [{survives}])"
        )

    def summary(self) -> str:
        """Return a dowhy-style textual summary of the refutation.

        Mirrors dowhy's ``CausalRefutation`` output
        (``Estimated effect`` / ``New effect`` / ``p value``), with the
        Bayesian additions of a credible interval on the placebo effect and
        a pass/fail placebo verdict.

        Returns
        -------
        str
            A multi-line summary string.
        """
        mu_lo, mu_hi = self.mu_null_hdi
        placebo = "PASS" if self.passes_placebo else "FAIL"
        return (
            f"Refute: Use a Placebo Treatment "
            f"({self.treatment} -> {self.outcome})\n"
            f"Estimated effect: {self.estimated_effect:.4f}\n"
            f"New effect: {self.new_effect:.4f} [{mu_lo:.4f}, {mu_hi:.4f}] "
            f"(pooled placebo effect; ~0 if sound) [{placebo}]\n"
            f"p value: {self.p_tail:.4f}"
        )

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks."""
        if self.passes_placebo:
            placebo_verdict = (
                '<span style="color: green; font-weight: bold;">✓ Pass</span> '
                "— the placebo null straddles zero."
            )
        else:
            placebo_verdict = (
                '<span style="color: red; font-weight: bold;">✗ Fail</span> '
                "— the pipeline reports a non-zero effect for a permuted treatment."
            )
        if self.effect_survives:
            effect_verdict = (
                '<span style="color: green; font-weight: bold;">✓ Survives</span> '
                "— the observed effect is too extreme for placebo noise."
            )
        else:
            effect_verdict = (
                '<span style="color: orange; font-weight: bold;">⚠ Not '
                "distinguishable</span> — the observed effect is consistent "
                "with placebo noise."
            )

        mu_lo, mu_hi = self.mu_null_hdi
        ate_lo, ate_hi = self.observed_ate_hdi
        rows = [
            ("Placebo test", placebo_verdict),
            ("Real effect", effect_verdict),
            (
                "Placebo bias μ_null",
                f"{self.mu_null:.4f} [{mu_lo:.4f}, {mu_hi:.4f}]",
            ),
            ("Structural volatility τ_het", f"{self.tau_het:.4f}"),
            (
                "Observed ATE",
                f"{self.observed_ate:.4f} [{ate_lo:.4f}, {ate_hi:.4f}]",
            ),
            ("Calibration", f"z = {self.z_cal:.3f}, p_tail = {self.p_tail:.4f}"),
            ("Permutations", str(self.n_permutations)),
        ]
        body = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
        )
        return (
            f"<h4>Placebo Refutation: {self.treatment} → {self.outcome} "
            f"(α = {self.significance_level})</h4>"
            f"<table><tbody>{body}</tbody></table>"
        )

    def plot(
        self,
        ax: matplotlib.axes.Axes | None = None,
        kind: str = "comparison",
        bins: int = 50,
    ) -> matplotlib.figure.Figure:
        """Plot the refutation result.

        Parameters
        ----------
        ax : matplotlib.axes.Axes | None
            Axes to plot on. Creates a new figure if ``None``.
        kind : {"comparison", "null"}
            Which view to draw (default ``"comparison"``):

            - ``"comparison"`` — the real (observed) effect and the pooled
              placebo effect side by side, each as a point with its 94% HDI,
              against a zero reference. This is the visual analogue of
              dowhy's "Estimated effect vs New effect": a sound pipeline
              shows the placebo row sitting on zero and the observed row away
              from it.
            - ``"null"`` — the placebo null-predictive distribution
              (histogram) with the observed ATE marked.
        bins : int
            Number of histogram bins, used only when ``kind="null"``
            (default 50).

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the plot.

        Raises
        ------
        ValueError
            If *kind* is unknown or *bins* is not positive.
        """
        import matplotlib.pyplot as plt

        if bins < 1:
            raise ValueError(f"bins must be a positive integer, got {bins}.")
        if kind not in ("comparison", "null"):
            raise ValueError(
                f"Unknown kind={kind!r}. Choose from 'comparison' or 'null'."
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            from typing import cast

            fig = cast("matplotlib.figure.Figure", ax.get_figure())

        if kind == "comparison":
            self._plot_comparison(ax)
        else:
            self._plot_null(ax, bins)
        return fig

    def _plot_comparison(self, ax: matplotlib.axes.Axes) -> None:
        """Draw observed vs pooled-placebo effect with HDIs (dowhy-style)."""
        ate_lo, ate_hi = self.observed_ate_hdi
        mu_lo, mu_hi = self.mu_null_hdi
        points = [self.observed_ate, self.new_effect]
        lowers = [self.observed_ate - ate_lo, self.new_effect - mu_lo]
        uppers = [ate_hi - self.observed_ate, mu_hi - self.new_effect]
        y = [1, 0]
        colors = ["tab:red", "tab:gray"]
        for yi, pt, lo, hi, c in zip(y, points, lowers, uppers, colors):
            ax.errorbar(
                pt,
                yi,
                xerr=[[lo], [hi]],
                fmt="o",
                color=c,
                capsize=5,
                markersize=8,
                linewidth=2,
            )
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1.5, label="zero")
        ax.set_yticks(y)
        ax.set_yticklabels([
            "Estimated effect\n(observed)",
            "New effect\n(placebo)",
        ])
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel("Effect (94% HDI)")
        survives = "survives" if self.effect_survives else "does not survive"
        ax.set_title(
            f"Placebo refutation: {self.treatment} → {self.outcome}\n"
            f"p value = {self.p_tail:.4f} → effect {survives} the placebo null"
        )
        ax.legend(loc="best", fontsize="small")

    def _plot_null(self, ax: matplotlib.axes.Axes, bins: int) -> None:
        """Draw the placebo null-predictive distribution vs the observed ATE."""
        ax.hist(
            self.theta_new_draws,
            bins=bins,
            density=True,
            alpha=0.6,
            color="tab:gray",
            edgecolor="none",
            label="Placebo null predictive",
        )
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1.5, label="zero")
        ax.axvline(
            self.observed_ate,
            color="tab:red",
            linestyle="--",
            linewidth=2,
            label=f"Observed ATE = {self.observed_ate:.3f}",
        )
        ax.set_xlabel("Effect")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Placebo refutation: {self.treatment} → {self.outcome}\n"
            f"p_tail = {self.p_tail:.4f}, z_cal = {self.z_cal:.3f}"
        )
        ax.legend(loc="best", fontsize="small")


def _permute_and_refit(
    model: PathModel,
    outcome: str,
    treatment: str,
    values: tuple[float, float],
    seed: int,
    sample_kwargs: dict[str, Any],
) -> tuple[float, float]:
    """Refit on a permuted-treatment dataset and summarize the placebo ATE.

    Returns ``(m_j, s_j)``: the posterior mean and SD of the average
    treatment effect under one placebo permutation.
    """
    clone = model._refit_permuted(treatment, seed, sample_kwargs)
    # The observed-ATE call in refute_placebo already surfaces any
    # out-of-range extrapolation warning once; silence the identical per-fold
    # repeats here to avoid n_permutations duplicate warnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*outside.*observed data range.*")
        placebo_ate = clone.ate(outcome, treatment, values=values)
    draws = np.asarray(placebo_ate.draws(outcome), dtype=float)
    return float(np.mean(draws)), float(np.std(draws))


def _fit_hierarchical_null(
    fold_means: np.ndarray,
    fold_sds: np.ndarray,
    sample_kwargs: dict[str, Any],
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the hierarchical normal-normal null model over placebo folds.

    Each fold mean ``m_j`` is a noisy observation of a latent placebo
    error ``theta_j ~ Normal(mu_null, tau_het)`` with known within-fold
    SD ``s_j``. Uses a non-centered parameterization. Returns flattened
    posterior draws of ``(mu_null, tau_het, theta_new)`` where
    ``theta_new`` is the null predictive for an unseen fold.

    The weakly-informative prior scale is the *total* marginal SD of the
    fold means, ``sqrt(var(m_j) + mean(s_j**2))`` (between-fold plus
    within-fold variance), rather than the between-fold SD alone. The
    latter collapses toward zero precisely when the placebo folds agree on
    a systematic bias — the clearest signature of a broken pipeline — which
    would shrink ``mu_null`` toward zero and mask the bias. The total scale
    stays informative in that case while reducing to the between-fold scale
    when within-fold noise is negligible.
    """
    fold_means = np.asarray(fold_means, dtype=float)
    fold_sds = np.asarray(fold_sds, dtype=float)
    if fold_means.shape != fold_sds.shape or fold_means.ndim != 1:
        raise ValueError(
            f"fold_means and fold_sds must be 1-D arrays of equal length, got "
            f"shapes {fold_means.shape} and {fold_sds.shape}."
        )
    if not (np.all(np.isfinite(fold_means)) and np.all(np.isfinite(fold_sds))):
        raise RuntimeError(
            "Hierarchical null received non-finite placebo fold summaries "
            "(m_j or s_j). A placebo re-fit most likely failed to converge. "
            "Increase draws/tune or target_accept via sample_kwargs, or inspect "
            "the model specification."
        )
    # Enforce the within-fold SD floor here too (not just in the caller) so a
    # zero/negative s_j never reaches the likelihood as sigma <= 0.
    fold_sds = np.maximum(fold_sds, _MIN_FOLD_SD)

    n_folds = len(fold_means)
    between_var = float(np.var(fold_means))
    within_var = float(np.mean(fold_sds**2))
    sigma_hat = float(np.sqrt(between_var + within_var))
    if not sigma_hat > 0.0:
        sigma_hat = 1.0

    hier_kwargs = {**sample_kwargs, "random_seed": random_seed}
    hier_kwargs.setdefault("progressbar", False)
    if sys.platform == "darwin":
        hier_kwargs.setdefault("mp_ctx", "forkserver")

    # Prior scale = 2 * sigma_hat. The factor 2 follows Roever (2021)'s
    # weakly-informative heterogeneity prior for random-effects meta-analysis
    # (HalfNormal scale = 2x the empirical SD): wide enough not to over-shrink
    # tau_het at small J, while still scaling to the data. The same scale is
    # used for mu_null. This constant does real work at small J -- see the
    # docstring note that the null spread is prior-dominated until ~8+ folds.
    prior_scale = 2.0 * sigma_hat
    with pm.Model(coords={"fold": np.arange(n_folds)}):
        obs_sd = pm.Data("obs_sd", fold_sds, dims="fold")
        mu = pm.Normal("mu_null", mu=0.0, sigma=prior_scale)
        tau = pm.HalfNormal("tau_het", sigma=prior_scale)
        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="fold")
        theta = pm.Deterministic("theta", mu + tau * z, dims="fold")
        pm.Normal("lik", mu=theta, sigma=obs_sd, observed=fold_means, dims="fold")
        idata = pm.sample(**hier_kwargs)
        pm.Normal("theta_new", mu=mu, sigma=tau)
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["theta_new"],
            random_seed=random_seed,
            progressbar=False,
        )

    post = posterior(idata)
    mu_draws = np.asarray(post["mu_null"].to_numpy()).flatten()
    tau_draws = np.asarray(post["tau_het"].to_numpy()).flatten()
    theta_new = np.asarray(
        ppc["posterior_predictive"].dataset["theta_new"].to_numpy()
    ).flatten()
    return mu_draws, tau_draws, theta_new


def refute_placebo(
    model: PathModel,
    outcome: str,
    treatment: str,
    *,
    values: tuple[float, float] = (0.0, 1.0),
    n_permutations: int = 4,
    significance_level: float = 0.05,
    sample_kwargs: dict[str, Any] | None = None,
    random_seed: int | None = None,
) -> PlaceboRefutationResult:
    """Refute an estimated effect by replacing the treatment with a placebo.

    Permutes the *treatment* column ``n_permutations`` times, re-fitting
    the model and re-estimating the ATE each time. Because the treatment's
    link to the outcome is severed by permutation, a sound pipeline should
    report no effect. The per-permutation posterior summaries ``(m_j, s_j)``
    are pooled through a hierarchical normal-normal random-effects model
    that separates the systematic placebo bias ``mu_null`` (which should
    straddle zero) from the structural volatility ``tau_het``. The real
    effect is then calibrated against the resulting null predictive
    distribution.

    Each permutation triggers a full MCMC re-fit, so cost scales linearly
    with *n_permutations*.

    .. note::

        The hierarchical null places a Gaussian likelihood on the per-fold
        ATE summaries, so the calibration is best trusted for response-scale
        effects that are approximately Gaussian. For bounded effects (e.g. a
        probability difference under a Bernoulli outcome) it is a normal
        approximation and ``mu_null`` may drift from zero near the bounds.

        When *treatment* is itself an endogenous (regressed) variable,
        permuting its observed column severs *all* of its structural
        relationships, not only its link to the outcome; the ``do()``-based
        ATE remains well defined, but the null then reflects "treatment
        decoupled from everything".

    Parameters
    ----------
    model : PathModel
        A fitted, data-bound model created via :func:`pathmc.model`.
    outcome : str
        Outcome variable name.
    treatment : str
        Treatment variable to replace with a placebo. Must be an observed
        (non-latent) data column.
    values : tuple[float, float]
        ``(lo, hi)`` intervention values for the ATE contrast, matching
        :meth:`pathmc.PathModel.ate` (default ``(0.0, 1.0)``).
    n_permutations : int
        Number of placebo permutations / folds (default 4). Four is a
        floor: with so few folds the between-fold volatility ``tau_het`` is
        a variance component estimated from a handful of points and is
        prior-dominated, so both verdicts lean on the prior scale. Use 8 or
        more folds when you need a data-driven null spread; each fold is a
        full MCMC re-fit, so cost scales linearly.
    significance_level : float
        Threshold for the :attr:`PlaceboRefutationResult.effect_survives`
        verdict (default 0.05).
    sample_kwargs : dict | None
        Keyword arguments forwarded to ``.fit()`` for each placebo re-fit
        and to the hierarchical null model's sampler (e.g. ``draws``,
        ``tune``, ``chains``, ``target_accept``). Defaults to PyMC's
        sampler defaults.
    random_seed : int | None
        Seed for the permutations and the hierarchical fit, for
        reproducible results.

    Returns
    -------
    PlaceboRefutationResult
        Placebo null summary, verdicts (``.passes_placebo``,
        ``.effect_survives``), calibration (``.z_cal``, ``.p_tail``), and a
        ``.plot()`` helper.

    Raises
    ------
    ValueError
        If *treatment* or *outcome* is not in the model; if they are equal;
        if *treatment* is latent, lacks a data column, or is constant; if
        *outcome* is latent; if *values* has equal endpoints; if
        *n_permutations* < 2; or if *significance_level* is not in
        ``(0, 1)``.
    NotImplementedError
        If the model is a panel model (not yet supported).
    RuntimeError
        If the model was created without data, called before ``.fit()``, or
        not created via :func:`pathmc.model` (no construction record to
        re-fit from).
    """
    model._require_fitted("refute_placebo")

    if not 0.0 < significance_level < 1.0:
        raise ValueError(
            f"significance_level must be in (0, 1), got {significance_level}. "
            f"Use a value such as 0.05."
        )
    if (
        not isinstance(n_permutations, (int, np.integer))
        or isinstance(n_permutations, bool)
        or n_permutations < 2
    ):
        raise ValueError(
            f"n_permutations must be an integer >= 2, got {n_permutations!r}. "
            f"At least two placebo folds are needed to identify the "
            f"between-fold volatility τ_het; 4 or more is recommended."
        )
    if len(values) != 2:
        raise ValueError(
            f"values must be a (lo, hi) pair, got {len(values)} entries: "
            f"{values}. Use e.g. values=(0.0, 1.0)."
        )
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(
            f"values must be finite numbers, got {values}. Use e.g. values=(0.0, 1.0)."
        )
    if float(values[0]) == float(values[1]):
        raise ValueError(
            f"values must have distinct (lo, hi) endpoints, got {values}. "
            f"Use e.g. values=(0.0, 1.0) so the ATE contrast is non-trivial."
        )

    if treatment == outcome:
        raise ValueError(
            f"treatment and outcome must differ, but both are '{treatment}'. "
            f"Pass the treatment and outcome variables you want to compare."
        )

    all_vars = model._graph_info.exogenous | model._graph_info.endogenous
    if treatment not in all_vars:
        raise ValueError(
            f"Treatment '{treatment}' not in model. "
            f"Available variables: {sorted(all_vars)}"
        )
    if outcome not in all_vars:
        raise ValueError(
            f"Outcome '{outcome}' not in model. Available variables: {sorted(all_vars)}"
        )
    if treatment in model._latent:
        raise ValueError(
            f"Treatment '{treatment}' is latent and has no data column to "
            f"permute. The placebo refuter requires an observed treatment."
        )
    if outcome in model._latent:
        raise ValueError(
            f"Outcome '{outcome}' is latent and has no data column. The "
            f"placebo refuter requires an observed outcome to estimate an ATE."
        )
    if model._panel_info is not None:
        raise NotImplementedError(
            "refute_placebo() is not yet supported for panel models. "
            "Permuting the treatment would break the unit/time structure."
        )

    assert model._data is not None
    if treatment not in model._data.columns:
        raise ValueError(
            f"Treatment '{treatment}' has no column in the data and cannot "
            f"be permuted. Available columns: {sorted(model._data.columns)}"
        )
    treat_col = np.asarray(model._data[treatment].to_numpy(), dtype=float)
    if np.unique(treat_col[~np.isnan(treat_col)]).size < 2:
        raise ValueError(
            f"Treatment '{treatment}' is constant, so permuting it is a no-op "
            f"and the placebo test is meaningless. Provide a treatment that "
            f"varies across observations."
        )
    if model._construction is None:
        raise RuntimeError(
            "refute_placebo() requires a model created via pathmc.model(...), "
            "which records the spec and settings needed to re-fit on permuted "
            "data."
        )

    # compute_log_likelihood is owned by the refuter (refits skip it); strip it
    # so a user-supplied value cannot collide with the explicit keyword in
    # clone.fit() nor leak into the hierarchical pm.sample() call.
    sample_kwargs = {
        k: v for k, v in (sample_kwargs or {}).items() if k != "compute_log_likelihood"
    }
    rng = np.random.default_rng(random_seed)

    observed = model.ate(outcome, treatment, values=values)
    observed_draws = np.asarray(observed.draws(outcome), dtype=float)
    if observed_draws.size == 0:
        raise RuntimeError(
            "The fitted posterior has no draws for the ATE. Re-fit the model "
            "with draws >= 1 before calling refute_placebo()."
        )
    if not np.all(np.isfinite(observed_draws)):
        raise RuntimeError(
            "The observed ATE posterior contains non-finite values. The fitted "
            "model likely failed to converge; re-fit before calling "
            "refute_placebo()."
        )

    perm_seeds = rng.integers(0, 2**32 - 1, size=n_permutations)
    fold_means = np.empty(n_permutations, dtype=float)
    fold_sds = np.empty(n_permutations, dtype=float)
    for j in range(n_permutations):
        m_j, s_j = _permute_and_refit(
            model, outcome, treatment, values, int(perm_seeds[j]), sample_kwargs
        )
        fold_means[j] = m_j
        fold_sds[j] = s_j
    fold_sds = np.where(fold_sds < _MIN_FOLD_SD, _MIN_FOLD_SD, fold_sds)

    # A single placebo fold with a posterior SD far larger than the others is a
    # symptom of a non-converged re-fit. Such a fold dominates the equal-weighted
    # calibration (mean(s_j**2) and the bootstrap), which can inflate p_tail and
    # mask a genuine effect (a false "does not survive"). Warn rather than fail,
    # since the verdict direction is conservative.
    sorted_sds = np.sort(fold_sds)
    max_sd = float(sorted_sds[-1])
    rest_median = float(np.median(sorted_sds[:-1]))
    if rest_median > 0.0 and max_sd > _SD_OUTLIER_FACTOR * rest_median:
        warnings.warn(
            f"A placebo fold has a posterior SD (max s_j = {max_sd:.3g}) far "
            f"larger than the others (median of the rest = {rest_median:.3g}). "
            f"This fold likely did not converge and can dominate the calibration "
            f"(z_cal / p_tail), risking a false 'effect does not survive'. "
            f"Increase draws/tune or target_accept via sample_kwargs, or raise "
            f"n_permutations.",
            UserWarning,
            stacklevel=2,
        )

    hier_seed = int(rng.integers(0, 2**32 - 1))
    mu_null_draws, tau_het_draws, theta_new_draws = _fit_hierarchical_null(
        fold_means, fold_sds, sample_kwargs, hier_seed
    )

    # These summarize the null PREDICTIVE distribution (theta_new), not the
    # mu_null / tau_het parameter posteriors; kept distinct on purpose.
    null_mean = float(np.mean(theta_new_draws))
    null_sd = float(np.std(theta_new_draws))
    mean_within_var = float(np.mean(fold_sds**2))
    sigma_pred = float(np.sqrt(null_sd**2 + mean_within_var))
    observed_mean = float(np.mean(observed_draws))
    z_cal = (observed_mean - null_mean) / sigma_pred if sigma_pred > 0.0 else 0.0

    noise_rng = np.random.default_rng(hier_seed)
    noise_sd = noise_rng.choice(fold_sds, size=theta_new_draws.size)
    pred_null = theta_new_draws + noise_rng.normal(0.0, noise_sd)
    if observed_mean >= null_mean:
        p_tail = float(np.mean(pred_null >= observed_mean))
    else:
        p_tail = float(np.mean(pred_null <= observed_mean))
    # Floor at the Monte Carlo resolution: an exact 0.0 overstates certainty
    # when no predictive draw happens to exceed the observed effect.
    p_tail = max(p_tail, 1.0 / (pred_null.size + 1))

    return PlaceboRefutationResult(
        outcome=outcome,
        treatment=treatment,
        observed_ate_draws=observed_draws,
        fold_means=fold_means,
        fold_sds=fold_sds,
        mu_null_draws=mu_null_draws,
        tau_het_draws=tau_het_draws,
        theta_new_draws=theta_new_draws,
        z_cal=z_cal,
        p_tail=p_tail,
        n_permutations=n_permutations,
        significance_level=significance_level,
        random_seed=random_seed,
    )
