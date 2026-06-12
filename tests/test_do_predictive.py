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
"""M12 gate tests: predictive do() sampling.

Tests verify that kind='predictive' adds residual noise,
producing wider HDIs and realistic distributional spread.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc

from conftest import MEDIATION_SPEC


@pytest.fixture(scope="module")
def fitted_simple(mock_pymc_sample_module):
    """Simple Y ~ X model fitted for predictive testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    Y = 0.5 * X + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame({"X": X, "Y": Y})

    model = pathmc.model("Y ~ X", data=data)
    model.fit(draws=50, tune=50, chains=1, random_seed=42)
    return model


@pytest.fixture(scope="module")
def fitted_mediation_for_pred(mock_pymc_sample_module):
    """Mediation model fitted for predictive do() tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=0.5, size=n)
    mediation_data = pd.DataFrame({"X": X, "M": M, "Y": Y})
    model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
    model.fit(draws=50, tune=50, chains=1, random_seed=42)
    return model


@pytest.mark.slow
class TestPredictiveDoAPI:
    def test_predictive_returns_do_result(self, fitted_simple):
        result = fitted_simple.do(set={"X": 1.0}, kind="predictive")
        assert result is not None
        assert hasattr(result, "mean")
        assert hasattr(result, "hdi")

    def test_predictive_contrast_arithmetic(self, fitted_simple):
        r0 = fitted_simple.do(set={"X": 0.0}, kind="predictive")
        r1 = fitted_simple.do(set={"X": 1.0}, kind="predictive")
        contrast = r1 - r0
        assert contrast is not None
        assert contrast.mean("Y") is not None


@pytest.mark.slow
class TestPredictiveVsMean:
    def test_predictive_hdi_wider_than_mean(self, fitted_simple):
        """Predictive HDIs should be wider because they include residual noise."""
        r_mean = fitted_simple.do(set={"X": 1.0}, kind="mean")
        r_pred = fitted_simple.do(set={"X": 1.0}, kind="predictive")

        hdi_mean = r_mean.hdi("Y")
        hdi_pred = r_pred.hdi("Y")

        width_mean = hdi_mean[1] - hdi_mean[0]
        width_pred = hdi_pred[1] - hdi_pred[0]

        assert width_pred > width_mean, (
            f"Predictive HDI width ({width_pred:.3f}) should be wider "
            f"than mean HDI width ({width_mean:.3f})"
        )

    def test_predictive_draws_have_spread(self, fitted_simple):
        """Predictive draws should not collapse to a point."""
        result = fitted_simple.do(set={"X": 1.0}, kind="predictive")
        hdi = result.hdi("Y")
        width = hdi[1] - hdi[0]
        assert width > 0.1, f"Predictive draws too narrow: HDI width = {width:.3f}"


@pytest.mark.slow
class TestPredictiveBernoulli:
    def test_bernoulli_predictive_draws_are_binary(self, mock_pymc_sample):
        """Predictive draws for Bernoulli outcomes should be 0 or 1."""
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(0.2 + 1.5 * X)))
        Y = rng.binomial(1, p, size=n).astype(float)
        data = pd.DataFrame({"X": X, "Y": Y})

        model = pathmc.model("Y ~ X", data=data, families={"Y": "bernoulli"})
        model.fit(draws=50, tune=50, chains=1, random_seed=42)

        result = model.do(set={"X": 1.0}, kind="predictive")
        mean_val = result.mean("Y")
        assert 0 <= mean_val <= 1, f"Bernoulli mean should be in [0,1], got {mean_val}"


@pytest.mark.slow
class TestPredictiveMediation:
    def test_mediation_predictive(self, fitted_mediation_for_pred):
        """Predictive do() works through multi-step DAG."""
        r0 = fitted_mediation_for_pred.do(set={"X": 0.0}, kind="predictive")
        r1 = fitted_mediation_for_pred.do(set={"X": 1.0}, kind="predictive")
        ate = r1 - r0
        assert ate.mean("Y") is not None
