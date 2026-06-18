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
"""Tests for the Bayesian placebo refuter."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import pathmc
from pathmc.refute import PlaceboRefutationResult


def _make_result(
    *,
    mu_null_draws: np.ndarray,
    observed_loc: float = 0.8,
    p_tail: float = 0.001,
    n_permutations: int = 4,
    significance_level: float = 0.05,
) -> PlaceboRefutationResult:
    """Build a PlaceboRefutationResult from hand-set posteriors.

    Lets verdict logic be tested without running MCMC.
    """
    rng = np.random.default_rng(0)
    observed = rng.normal(loc=observed_loc, scale=0.1, size=2000)
    theta_new = rng.normal(loc=float(np.mean(mu_null_draws)), scale=0.2, size=2000)
    tau_het_draws = np.abs(rng.normal(loc=0.2, scale=0.05, size=mu_null_draws.size))
    fold_means = rng.normal(loc=0.0, scale=0.1, size=n_permutations)
    fold_sds = np.abs(rng.normal(loc=0.15, scale=0.02, size=n_permutations))
    return PlaceboRefutationResult(
        outcome="Y",
        treatment="T",
        observed_ate_draws=observed,
        fold_means=fold_means,
        fold_sds=fold_sds,
        mu_null_draws=mu_null_draws,
        tau_het_draws=tau_het_draws,
        theta_new_draws=theta_new,
        z_cal=5.0,
        p_tail=p_tail,
        n_permutations=n_permutations,
        significance_level=significance_level,
    )


class TestPlaceboRefutationResult:
    """Verdict logic and display of the result dataclass."""

    def test_passes_placebo_when_null_straddles_zero(self):
        mu = np.random.default_rng(1).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        assert result.passes_placebo is True

    def test_fails_placebo_when_null_excludes_zero(self):
        mu = np.random.default_rng(2).normal(loc=5.0, scale=0.1, size=2000)
        result = _make_result(mu_null_draws=mu)
        assert result.passes_placebo is False

    def test_effect_survives_with_small_p_tail(self):
        mu = np.random.default_rng(3).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu, p_tail=0.001)
        assert result.effect_survives is True

    def test_effect_does_not_survive_with_large_p_tail(self):
        mu = np.random.default_rng(4).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu, p_tail=0.5)
        assert result.effect_survives is False

    def test_significance_level_controls_survival(self):
        mu = np.random.default_rng(5).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu, p_tail=0.04, significance_level=0.01)
        assert result.effect_survives is False
        result2 = _make_result(mu_null_draws=mu, p_tail=0.04, significance_level=0.10)
        assert result2.effect_survives is True

    def test_observed_ate_and_hdi(self):
        mu = np.random.default_rng(6).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu, observed_loc=0.8)
        assert 0.7 < result.observed_ate < 0.9
        hdi = result.observed_ate_hdi
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]
        assert hdi[0] < result.observed_ate < hdi[1]

    def test_mu_null_hdi_ordering(self):
        mu = np.random.default_rng(7).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        lo, hi = result.mu_null_hdi
        assert lo < hi

    def test_summary_properties(self):
        mu = np.random.default_rng(8).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        assert np.isfinite(result.mu_null)
        assert np.isfinite(result.tau_het)
        assert np.isfinite(result.null_mean)
        assert result.null_sd >= 0

    def test_sigma_pred_and_repr_reconstructable(self):
        mu = np.random.default_rng(15).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        # sigma_pred combines the null predictive SD and the mean within-fold
        # variance (mean of squares), matching the p_tail bootstrap.
        expected = np.sqrt(result.null_sd**2 + np.mean(result.fold_sds**2))
        assert np.isclose(result.sigma_pred, expected)
        assert result.sigma_pred > 0
        assert "σ_pred" in repr(result)

    def test_repr_contains_key_info(self):
        mu = np.random.default_rng(9).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        r = repr(result)
        assert "treatment='T'" in r
        assert "outcome='Y'" in r
        assert "Placebo test" in r
        assert "Observed ATE" in r
        assert "Calibration" in r

    def test_repr_html_pass_and_fail(self):
        mu_pass = np.random.default_rng(10).normal(loc=0.0, scale=1.0, size=2000)
        html_pass = _make_result(mu_null_draws=mu_pass)._repr_html_()
        assert "Pass" in html_pass
        mu_fail = np.random.default_rng(11).normal(loc=5.0, scale=0.1, size=2000)
        html_fail = _make_result(mu_null_draws=mu_fail)._repr_html_()
        assert "Fail" in html_fail

    def test_plot_returns_figure(self):
        import matplotlib.figure

        mu = np.random.default_rng(12).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        fig = result.plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_with_existing_axes(self):
        import matplotlib.pyplot as plt

        mu = np.random.default_rng(13).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        fig, ax = plt.subplots()
        returned = result.plot(ax=ax)
        assert returned is fig

    def test_plot_bad_bins_raises(self):
        mu = np.random.default_rng(14).normal(loc=0.0, scale=1.0, size=2000)
        result = _make_result(mu_null_draws=mu)
        with pytest.raises(ValueError, match="bins"):
            result.plot(bins=0)


class TestRefutePlaceboErrorHandling:
    """Validation and guard conditions."""

    def _model(self):
        df = pd.DataFrame({
            "T": np.random.default_rng(0).normal(size=50),
            "X": np.random.default_rng(1).normal(size=50),
            "Y": np.random.default_rng(2).normal(size=50),
        })
        return pathmc.model("Y ~ T + X", data=df)

    def test_no_samples_raises(self):
        model = self._model()
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.refute_placebo("Y", "T")

    def test_requires_data(self):
        model = pathmc.model("Y ~ T + X")
        with pytest.raises(RuntimeError, match="requires data"):
            model.refute_placebo("Y", "T")

    def test_bad_n_permutations_raises(self):
        model = self._model()
        model._idata = True  # bypass sampling check
        with pytest.raises(ValueError, match="n_permutations"):
            model.refute_placebo("Y", "T", n_permutations=1)

    def test_n_permutations_bool_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="n_permutations"):
            model.refute_placebo("Y", "T", n_permutations=True)

    def test_bad_significance_level_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="significance_level"):
            model.refute_placebo("Y", "T", significance_level=1.5)

    def test_unknown_treatment_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="not in model"):
            model.refute_placebo("Y", "UNKNOWN")

    def test_unknown_outcome_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="not in model"):
            model.refute_placebo("UNKNOWN", "T")

    def test_treatment_equals_outcome_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="must differ"):
            model.refute_placebo("Y", "Y")

    def test_equal_values_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="distinct"):
            model.refute_placebo("Y", "T", values=(1.0, 1.0))

    def test_non_finite_values_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="finite"):
            model.refute_placebo("Y", "T", values=(0.0, np.inf))

    def test_wrong_length_values_raises(self):
        model = self._model()
        model._idata = True
        with pytest.raises(ValueError, match="lo, hi"):
            model.refute_placebo("Y", "T", values=(0.0, 1.0, 2.0))

    def test_constant_treatment_raises(self):
        df = pd.DataFrame({
            "T": np.ones(50),
            "X": np.random.default_rng(1).normal(size=50),
            "Y": np.random.default_rng(2).normal(size=50),
        })
        model = pathmc.model("Y ~ T + X", data=df)
        model._idata = True
        with pytest.raises(ValueError, match="constant"):
            model.refute_placebo("Y", "T")

    def test_latent_treatment_raises(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(1).normal(size=50),
            "Y": np.random.default_rng(2).normal(size=50),
        })
        model = pathmc.model("M ~ X\nY ~ M", latent=["M"], data=df)
        model._idata = True
        with pytest.raises(ValueError, match="latent"):
            model.refute_placebo("Y", "M")

    def test_latent_outcome_raises(self):
        df = pd.DataFrame({
            "X": np.random.default_rng(1).normal(size=50),
            "Y": np.random.default_rng(2).normal(size=50),
        })
        model = pathmc.model("M ~ X\nY ~ M", latent=["M"], data=df)
        model._idata = True
        with pytest.raises(ValueError, match="latent"):
            model.refute_placebo("M", "X")


class TestHierarchicalNullGuards:
    """Fast guards in _fit_hierarchical_null (no sampling)."""

    def test_non_finite_fold_summaries_raise(self):
        from pathmc.refute import _fit_hierarchical_null

        with pytest.raises(RuntimeError, match="non-finite"):
            _fit_hierarchical_null(
                np.array([0.1, np.nan, 0.2]), np.array([0.1, 0.1, 0.1]), {}, 0
            )

    def test_mismatched_shapes_raise(self):
        from pathmc.refute import _fit_hierarchical_null

        with pytest.raises(ValueError, match="equal length"):
            _fit_hierarchical_null(np.array([0.1, 0.2]), np.array([0.1]), {}, 0)


@pytest.mark.slow
class TestHierarchicalNull:
    """The hierarchical null model's empirical-Bayes prior must not collapse."""

    def test_detects_consistent_placebo_bias(self):
        # Folds agree on a clear +0.12 bias with moderate within-fold noise.
        # A between-fold-only prior scale would shrink mu_null toward zero
        # (false PASS); the total-variance scale must keep it near the bias.
        from pathmc.refute import _fit_hierarchical_null

        fold_means = np.array([0.12, 0.13, 0.11, 0.12, 0.125, 0.118])
        fold_sds = np.array([0.08] * 6)
        mu_draws, _, _ = _fit_hierarchical_null(fold_means, fold_sds, {}, 0)
        assert np.mean(mu_draws) > 0.05

    def test_symmetric_folds_straddle_zero(self):
        from pathmc.idata import hdi
        from pathmc.refute import _fit_hierarchical_null

        fold_means = np.array([0.3, -0.25, 0.15, -0.2])
        fold_sds = np.array([0.2] * 4)
        mu_draws, _, _ = _fit_hierarchical_null(fold_means, fold_sds, {}, 1)
        lo, hi = hdi(mu_draws)
        assert lo <= 0.0 <= hi

    def test_returns_flat_arrays(self):
        from pathmc.refute import _fit_hierarchical_null

        fold_means = np.array([0.1, -0.1, 0.0])
        fold_sds = np.array([0.1, 0.1, 0.1])
        mu_draws, tau_draws, theta_new = _fit_hierarchical_null(
            fold_means, fold_sds, {}, 2
        )
        assert mu_draws.ndim == 1
        assert tau_draws.ndim == 1
        assert theta_new.ndim == 1
        assert np.all(tau_draws >= 0.0)

    def test_zero_fold_sds_are_floored(self):
        # Zero within-fold SDs must be floored internally so the likelihood
        # never sees sigma <= 0 (no sampler initialization failure).
        from pathmc.refute import _fit_hierarchical_null

        fold_means = np.array([0.1, -0.1, 0.05])
        fold_sds = np.array([0.0, 0.0, 0.0])
        mu_draws, _, _ = _fit_hierarchical_null(fold_means, fold_sds, {}, 3)
        assert np.all(np.isfinite(mu_draws))


@pytest.mark.slow
class TestRefutePlaceboKwargs:
    """compute_log_likelihood in sample_kwargs must not crash the refuter."""

    def test_compute_log_likelihood_in_sample_kwargs_is_stripped(self):
        rng = np.random.default_rng(7)
        n = 80
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = 0.7 * T + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"T": T, "Y": Y})
        model = pathmc.model("Y ~ T", data=df)
        model.fit(random_seed=7)
        # Should not raise a double-keyword TypeError.
        result = model.refute_placebo(
            "Y",
            "T",
            n_permutations=2,
            sample_kwargs={"compute_log_likelihood": True},
            random_seed=7,
        )
        assert isinstance(result, PlaceboRefutationResult)

    def test_warns_on_heterogeneous_fold_sds(self, monkeypatch):
        import pathmc.refute as refute

        rng = np.random.default_rng(7)
        n = 80
        T = rng.binomial(1, 0.5, size=n).astype(float)
        Y = 0.7 * T + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"T": T, "Y": Y})
        model = pathmc.model("Y ~ T", data=df)
        model.fit(random_seed=7)

        # One fold's posterior SD is far larger than the others (simulated
        # non-convergence) -> the refuter should warn.
        sds = iter([0.1, 5.0])
        monkeypatch.setattr(
            refute, "_permute_and_refit", lambda *a, **k: (0.05, next(sds))
        )
        with pytest.warns(UserWarning, match="did not converge"):
            model.refute_placebo("Y", "T", n_permutations=2, random_seed=1)


@pytest.mark.slow
class TestRefutePlaceboIntegration:
    """End-to-end placebo refutation on a fitted model."""

    @pytest.fixture(scope="class")
    def strong_effect_model(self):
        rng = np.random.default_rng(42)
        n = 200
        T = rng.binomial(1, 0.5, size=n).astype(float)
        X = rng.normal(size=n)
        Y = 0.8 * T + 0.5 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"T": T, "X": X, "Y": Y})
        model = pathmc.model("Y ~ T + X", data=df)
        model.fit(random_seed=42)
        return model

    def test_returns_result(self, strong_effect_model):
        result = strong_effect_model.refute_placebo(
            "Y", "T", n_permutations=3, random_seed=42
        )
        assert isinstance(result, PlaceboRefutationResult)
        assert result.n_permutations == 3
        assert len(result.fold_means) == 3
        assert len(result.fold_sds) == 3

    def test_placebo_effects_near_zero(self, strong_effect_model):
        result = strong_effect_model.refute_placebo(
            "Y", "T", n_permutations=3, random_seed=42
        )
        # Placebo fold means should be far smaller than the real effect.
        assert np.max(np.abs(result.fold_means)) < result.observed_ate

    def test_real_effect_survives_placebo_null(self, strong_effect_model):
        result = strong_effect_model.refute_placebo(
            "Y", "T", n_permutations=3, random_seed=42
        )
        assert result.observed_ate > 0
        assert result.effect_survives is True
        assert result.mu_null_hdi[0] < result.mu_null_hdi[1]

    def test_plot_integration(self, strong_effect_model):
        import matplotlib.figure

        result = strong_effect_model.refute_placebo(
            "Y", "T", n_permutations=3, random_seed=42
        )
        fig = result.plot()
        assert isinstance(fig, matplotlib.figure.Figure)
