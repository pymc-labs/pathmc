"""M23 gate tests: Posterior predictive checks.

Tests verify that .predict() wraps pm.sample_posterior_predictive()
and produces correct InferenceData output for all families.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC


class TestPredictAPI:
    """API surface: .predict() method exists and validates state."""

    def test_predict_method_exists(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        assert hasattr(model, "predict")
        assert callable(model.predict)

    def test_predict_before_sample_raises(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        with pytest.raises(RuntimeError):
            model.predict()


@pytest.mark.slow
class TestGaussianPPC:
    """Posterior predictive for Gaussian models."""

    def test_predict_adds_posterior_predictive(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata = model.predict()
        assert hasattr(idata, "posterior_predictive")

    def test_posterior_predictive_has_observed_vars(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata = model.predict()
        pp = idata.posterior_predictive
        pp_vars = set(pp.data_vars)
        assert "M_obs" in pp_vars or "M" in pp_vars
        assert "Y_obs" in pp_vars or "Y" in pp_vars

    def test_posterior_predictive_shape(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata = model.predict()
        pp = idata.posterior_predictive
        for var_name in pp.data_vars:
            arr = pp[var_name]
            assert arr.shape[-1] == len(mediation_data)


@pytest.mark.slow
class TestBernoulliPPC:
    """Posterior predictive for Bernoulli models."""

    def test_bernoulli_predict(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(0.5 + 1.0 * X)))
        Y = rng.binomial(1, p, size=n).astype(float)
        df = pd.DataFrame({"X": X, "Y": Y})

        model = pathmc.fit("Y ~ X", data=df, families={"Y": "bernoulli"})
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata = model.predict()
        assert hasattr(idata, "posterior_predictive")


@pytest.mark.slow
class TestResidualBlockPPC:
    """Posterior predictive for ~~ (MvNormal) block models."""

    def test_residual_block_predict(self, parallel_mediators_data):
        model = pathmc.fit(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata = model.predict()
        assert hasattr(idata, "posterior_predictive")


@pytest.mark.slow
class TestPredictIdempotent:
    """Calling predict multiple times doesn't break things."""

    def test_predict_twice(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        model.sample(draws=100, tune=100, chains=1, random_seed=42)
        idata1 = model.predict()
        idata2 = model.predict()
        assert hasattr(idata1, "posterior_predictive")
        assert hasattr(idata2, "posterior_predictive")
