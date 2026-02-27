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
        model = pathmc.fit(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_correlation_parameter_in_model(self, parallel_mediators_data):
        """The compiled model should contain an LKJ or correlation-related parameter."""
        model = pathmc.fit(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        rv_names = {rv.name.lower() for rv in model.pymc_model.free_RVs}
        has_corr_param = any(
            keyword in name
            for name in rv_names
            for keyword in ("chol", "corr", "lkj", "rho")
        )
        assert has_corr_param, (
            f"No correlation parameter found in model. "
            f"Free RV names: {rv_names}"
        )

    def test_residual_cov_model_has_observed(self, parallel_mediators_data):
        model = pathmc.fit(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
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
            pathmc.fit(spec, data=data, families={"Y1": "bernoulli"})


@pytest.mark.slow
class TestResidualCovSampling:
    def test_residual_cov_model_samples(self, fitted_parallel_mediators):
        """The model with ~~ should sample without errors."""
        summary = fitted_parallel_mediators.summary()
        assert summary is not None
