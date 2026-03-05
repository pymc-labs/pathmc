"""Tests for sensitivity analysis (unmeasured confounding)."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import pathmc
from pathmc.sensitivity import SensitivityResult, compute_sensitivity


@pytest.fixture(scope="module")
def fork_model():
    """A simple fork model: Z -> X, Z -> Y, X -> Y.

    True causal effect of X on Y is ~0.4.
    """
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.normal(size=n)
    X = 0.5 * Z + rng.normal(scale=0.5, size=n)
    Y = 0.4 * X + 0.6 * Z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    model = pathmc.fit("X ~ Z\nY ~ X + Z", data=df)
    model.sample(draws=300, tune=300, chains=2, random_seed=42)
    return model


class TestComputeSensitivity:
    """Test the lower-level compute_sensitivity function."""

    def test_adjusted_ate_at_origin_equals_observed(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        assert abs(result.adjusted_ate_mean[0, 0] - result.observed_ate) < 1e-10

    def test_adjusted_ate_decreases_with_positive_bias(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        assert result.adjusted_ate_mean[-1, -1] < result.adjusted_ate_mean[0, 0]

    def test_tipping_point_equals_observed_mean(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        assert abs(result.tipping_point - result.observed_ate) < 1e-10

    def test_prob_sign_change_at_origin_is_low(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        assert result.prob_sign_change[0, 0] < 0.01

    def test_prob_sign_change_increases_with_bias(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 2.0),
            delta_range=(0.0, 2.0),
            n_grid=10,
        )
        assert result.prob_sign_change[-1, -1] > result.prob_sign_change[0, 0]

    def test_grid_shape(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=15,
        )
        assert result.adjusted_ate_mean.shape == (15, 15)
        assert result.prob_sign_change.shape == (15, 15)
        assert len(result.gamma_values) == 15
        assert len(result.delta_values) == 15

    def test_negative_observed_ate(self):
        draws = np.random.default_rng(0).normal(loc=-0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        assert result.observed_ate < 0
        assert result.adjusted_ate_mean[0, 0] < 0
        assert result.prob_sign_change[0, 0] < 0.01


class TestSensitivityResult:
    """Test the SensitivityResult dataclass."""

    def test_repr_contains_key_info(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=5,
        )
        r = repr(result)
        assert "treatment='X'" in r
        assert "outcome='Y'" in r
        assert "Observed ATE" in r
        assert "Tipping point" in r

    def test_observed_ate_hdi(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=5,
        )
        hdi = result.observed_ate_hdi
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]
        assert hdi[0] < result.observed_ate < hdi[1]

    def test_plot_returns_figure(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        import matplotlib.figure

        fig = result.plot()
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_with_existing_axes(self):
        import matplotlib.pyplot as plt

        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 1.0),
            delta_range=(0.0, 1.0),
            n_grid=10,
        )
        fig, ax = plt.subplots()
        returned_fig = result.plot(ax=ax)
        assert returned_fig is fig

    def test_plot_with_tipping_line(self):
        draws = np.random.default_rng(0).normal(loc=0.5, scale=0.1, size=1000)
        result = compute_sensitivity(
            draws,
            "Y",
            "X",
            gamma_range=(0.0, 2.0),
            delta_range=(0.0, 2.0),
            n_grid=20,
        )
        fig = result.plot()
        assert fig is not None


@pytest.mark.slow
class TestPathModelSensitivity:
    """Integration tests with a fitted PathModel."""

    def test_returns_sensitivity_result(self, fork_model):
        result = fork_model.sensitivity("Y", "X")
        assert isinstance(result, SensitivityResult)

    def test_observed_ate_is_positive(self, fork_model):
        result = fork_model.sensitivity("Y", "X")
        assert result.observed_ate > 0

    def test_adjusted_ate_at_origin(self, fork_model):
        result = fork_model.sensitivity("Y", "X", n_grid=5)
        assert abs(result.adjusted_ate_mean[0, 0] - result.observed_ate) < 1e-10

    def test_custom_ranges(self, fork_model):
        result = fork_model.sensitivity(
            "Y",
            "X",
            gamma_range=(0.0, 0.5),
            delta_range=(0.0, 0.5),
            n_grid=5,
        )
        assert result.gamma_values[-1] == 0.5
        assert result.delta_values[-1] == 0.5

    def test_custom_values(self, fork_model):
        result = fork_model.sensitivity("Y", "X", values=(-1.0, 2.0))
        assert result.observed_ate > 0

    def test_plot_integration(self, fork_model):
        import matplotlib.figure

        result = fork_model.sensitivity("Y", "X", n_grid=5)
        fig = result.plot()
        assert isinstance(fig, matplotlib.figure.Figure)


class TestSensitivityErrorHandling:
    """Test error conditions."""

    def test_no_samples_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.sensitivity("Y", "X")

    def test_bad_gamma_range_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        model._idata = True  # bypass sampling check
        with pytest.raises(ValueError, match="gamma_range"):
            model.sensitivity("Y", "X", gamma_range=(1.0, 0.0))

    def test_bad_delta_range_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        model._idata = True
        with pytest.raises(ValueError, match="delta_range"):
            model.sensitivity("Y", "X", delta_range=(1.0, 0.0))

    def test_bad_n_grid_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        model._idata = True
        with pytest.raises(ValueError, match="n_grid"):
            model.sensitivity("Y", "X", n_grid=1)

    def test_unknown_treatment_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        model._idata = True
        with pytest.raises(ValueError, match="not in model"):
            model.sensitivity("Y", "UNKNOWN")

    def test_unknown_outcome_raises(self):
        df = pd.DataFrame(
            {
                "X": np.random.normal(size=50),
                "Y": np.random.normal(size=50),
            }
        )
        model = pathmc.fit("Y ~ X", data=df)
        model._idata = True
        with pytest.raises(ValueError, match="not in model"):
            model.sensitivity("UNKNOWN", "X")
