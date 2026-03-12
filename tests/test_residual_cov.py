"""M7 gate tests: residual covariance (~~) compilation.

TestResidualCovCompilation verifies that ~~ produces MvNormal blocks.
TestResidualCovGuards verifies that ~~ is rejected for non-Gaussian outcomes.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc

from conftest import PARALLEL_MEDIATORS_SPEC


class TestResidualCovCompilation:
    def test_residual_cov_compiles(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_correlation_parameter_in_model(self, parallel_mediators_data):
        """The compiled model should contain an LKJ or correlation-related parameter."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        rv_names = {rv.name.lower() for rv in model.pymc_model.free_RVs}
        has_corr_param = any(
            keyword in name
            for name in rv_names
            for keyword in ("chol", "corr", "lkj", "rho")
        )
        assert has_corr_param, (
            f"No correlation parameter found in model. Free RV names: {rv_names}"
        )

    def test_residual_cov_model_has_observed(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        assert len(model.pymc_model.observed_RVs) > 0


class TestResidualCovGuards:
    def test_residual_cov_requires_gaussian(self):
        """~~ between a Bernoulli and a Gaussian outcome should raise."""
        spec = "Y1 ~ X\nY2 ~ X\nY1 ~~ Y2"
        rng = np.random.default_rng(99)
        data = pd.DataFrame(
            {
                "X": rng.normal(size=100),
                "Y1": rng.binomial(1, 0.5, size=100).astype(float),
                "Y2": rng.normal(size=100),
            }
        )
        with pytest.raises(
            Exception, match="(?i)(gaussian|continuous|family|bernoulli|covariance)"
        ):
            pathmc.model(spec, data=data, families={"Y1": "bernoulli"})


class TestBlockVarDeterministics:
    """Block variables should have mu_{var} deterministics and wire through endogenous_rvs."""

    def test_mu_deterministics_exist(self, parallel_mediators_data):
        """Each block variable should have a mu_{var} Deterministic in the model."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        det_names = {d.name for d in model.pymc_model.deterministics}
        assert "mu_M1" in det_names, f"mu_M1 missing from deterministics: {det_names}"
        assert "mu_M2" in det_names, f"mu_M2 missing from deterministics: {det_names}"

    def test_block_vars_not_free_rvs(self, parallel_mediators_data):
        """Block variables should NOT appear as individual free RVs."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        free_rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "M1" not in free_rv_names
        assert "M2" not in free_rv_names

    def test_lkj_prior_in_priors_output(self, parallel_mediators_data):
        """priors() should show the chol_{block} LKJ entry."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        prior_str = str(model.priors())
        assert "chol_M1_M2" in prior_str
        assert "LKJCholeskyCov" in prior_str


@pytest.mark.slow
class TestResidualCovSampling:
    def test_residual_cov_model_samples(self, fitted_parallel_mediators):
        """The model with ~~ should sample without errors."""
        summary = fitted_parallel_mediators.summary()
        assert summary is not None


@pytest.mark.slow
class TestBlockVarDoOperator:
    """do() should propagate through block variables correctly."""

    def test_do_through_block_vars_mean(self, fitted_parallel_mediators):
        """do() on an exogenous variable should propagate through block vars."""
        r0 = fitted_parallel_mediators.do(set={"T": 0.0}, kind="mean")
        r1 = fitted_parallel_mediators.do(set={"T": 1.0}, kind="mean")
        ate = r1.mean("Y") - r0.mean("Y")
        assert ate > 0, f"ATE of T on Y should be positive, got {ate}"

    def test_do_on_block_var_mean(self, fitted_parallel_mediators):
        """do() directly on a block variable should work with kind='mean'."""
        r0 = fitted_parallel_mediators.do(set={"M1": 0.0}, kind="mean")
        r1 = fitted_parallel_mediators.do(set={"M1": 1.0}, kind="mean")
        ate = r1.mean("Y") - r0.mean("Y")
        assert ate > 0, f"ATE of M1 on Y should be positive, got {ate}"

    def test_do_on_block_var_predictive(self, fitted_parallel_mediators):
        """do() directly on a block variable should work with kind='predictive'."""
        r0 = fitted_parallel_mediators.do(set={"M1": 0.0}, kind="predictive")
        r1 = fitted_parallel_mediators.do(set={"M1": 1.0}, kind="predictive")
        ate = r1.mean("Y") - r0.mean("Y")
        assert ate > 0, f"ATE of M1 on Y should be positive, got {ate}"
