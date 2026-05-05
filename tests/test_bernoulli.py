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
"""M11 gate tests: Bernoulli-logit family support.

Tests verify that binary outcomes compile to Bernoulli likelihoods
with a logit link, and that do() returns probabilities in (0, 1).
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def binary_data():
    """X -> Y (binary) with logistic link. True coefficient ~1.5."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.2 + 1.5 * X)))
    Y = rng.binomial(1, p, size=n).astype(float)
    return pd.DataFrame({"X": X, "Y": Y})


@pytest.fixture
def mixed_data():
    """X -> M (continuous) -> Y (binary)."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * M)))
    Y = rng.binomial(1, p, size=n).astype(float)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


class TestBernoulliCompilation:
    def test_bernoulli_compiles(self, binary_data):
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        assert model.pymc_model is not None

    def test_observed_rv_is_bernoulli(self, binary_data):
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        rv_names = [rv.name for rv in model.pymc_model.observed_RVs]
        assert "Y" in rv_names

    def test_no_sigma_for_bernoulli(self, binary_data):
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        free_names = [rv.name for rv in model.pymc_model.free_RVs]
        assert "sigma_Y" not in free_names
        assert "beta_Y" in free_names

    def test_residual_cov_with_bernoulli_raises(self, binary_data):
        spec = "Y ~ X\nY ~~ X"
        with pytest.raises(ValueError, match="(?i)gaussian"):
            pathmc.model(spec, data=binary_data, families={"Y": "bernoulli"})


class TestBernoulliMixed:
    def test_mixed_model_compiles(self, mixed_data):
        spec = "M ~ a*X\nY ~ b*M"
        model = pathmc.model(spec, data=mixed_data, families={"Y": "bernoulli"})
        free_names = [rv.name for rv in model.pymc_model.free_RVs]
        assert "sigma_M" in free_names
        assert "sigma_Y" not in free_names


@pytest.mark.slow
class TestBernoulliSampling:
    def test_bernoulli_samples(self, binary_data):
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        model.fit(draws=100, tune=100, chains=1, random_seed=42)
        summary = model.summary()
        assert summary is not None
        assert len(summary) > 0

    def test_do_returns_probabilities(self, binary_data):
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        model.fit(draws=100, tune=100, chains=1, random_seed=42)

        result = model.do(set={"X": 1.0})
        y_mean = result.mean("Y")
        assert 0 < y_mean < 1, f"Expected probability in (0,1), got {y_mean}"

    def test_do_on_bernoulli_treatment(self, binary_data):
        """do() on a Bernoulli variable itself must cast float values to int."""
        rng = np.random.default_rng(99)
        n = 200
        Z = rng.normal(size=n)
        T = rng.binomial(1, 1 / (1 + np.exp(-Z))).astype(float)
        Y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * T + 0.3 * Z)))).astype(float)
        df = pd.DataFrame({"Z": Z, "T": T, "Y": Y})

        model = pathmc.model(
            "T ~ Z\nY ~ T + Z",
            data=df,
            families={"T": "bernoulli", "Y": "bernoulli"},
        )
        model.fit(draws=100, tune=100, chains=1, random_seed=42)
        ate = model.ate("Y", "T", values=(0.0, 1.0))
        assert np.isfinite(ate.mean("Y"))

    def test_do_higher_x_higher_prob(self, binary_data):
        """Positive coefficient means higher X should give higher P(Y=1)."""
        model = pathmc.model("Y ~ X", data=binary_data, families={"Y": "bernoulli"})
        model.fit(draws=200, tune=200, chains=1, random_seed=42)

        r_low = model.do(set={"X": -1.0})
        r_high = model.do(set={"X": 1.0})
        assert r_high.mean("Y") > r_low.mean("Y")


@pytest.mark.slow
class TestBernoulliEffects:
    def test_effects_summary_with_mixed(self, mixed_data):
        spec = "M ~ a*X\nY ~ b*M"
        model = pathmc.model(spec, data=mixed_data, families={"Y": "bernoulli"})
        model.fit(draws=100, tune=100, chains=1, random_seed=42)
        effects = model.effects_summary()
        assert "a" in str(effects)
        assert "b" in str(effects)
