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
"""M22 gate tests: Additional families — Poisson, NegBinomial, StudentT.

Tests verify that new likelihood families compile, sample, and interact
correctly with do(), residual covariance guards, and introspection.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture(scope="module")
def poisson_data():
    """X -> Y (count) via Poisson with log link. True coeff ~0.3."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    mu = np.exp(1.0 + 0.3 * X)
    Y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"X": X, "Y": Y})


@pytest.fixture(scope="module")
def negbin_data():
    """X -> Y (overdispersed count) via NegBinomial."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    mu = np.exp(1.5 + 0.4 * X)
    alpha = 3.0
    p = alpha / (alpha + mu)
    Y = rng.negative_binomial(alpha, p).astype(float)
    return pd.DataFrame({"X": X, "Y": Y})


@pytest.fixture(scope="module")
def studentt_data():
    """X -> Y with heavy-tailed noise (StudentT)."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    Y = 0.5 + 0.8 * X + rng.standard_t(df=4, size=n) * 0.5
    return pd.DataFrame({"X": X, "Y": Y})


@pytest.fixture(scope="module")
def fitted_poisson(poisson_data):
    model = pathmc.model("Y ~ X", data=poisson_data, families={"Y": "poisson"})
    model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="module")
def fitted_negbin(negbin_data):
    model = pathmc.model("Y ~ X", data=negbin_data, families={"Y": "negbinomial"})
    model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="module")
def fitted_studentt(studentt_data):
    model = pathmc.model("Y ~ X", data=studentt_data, families={"Y": "studentt"})
    model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


class TestPoissonCompilation:
    """Poisson family compiles correctly."""

    def test_poisson_compiles(self, poisson_data):
        model = pathmc.model("Y ~ X", data=poisson_data, families={"Y": "poisson"})
        assert model.pymc_model is not None

    def test_no_sigma_for_poisson(self, poisson_data):
        model = pathmc.model("Y ~ X", data=poisson_data, families={"Y": "poisson"})
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "sigma_Y" not in free_names

    def test_observed_rv_present(self, poisson_data):
        model = pathmc.model("Y ~ X", data=poisson_data, families={"Y": "poisson"})
        obs_names = {rv.name for rv in model.pymc_model.observed_RVs}
        assert "Y" in obs_names

    def test_residual_cov_with_poisson_raises(self, poisson_data):
        poisson_data = poisson_data.copy()
        poisson_data["Z"] = np.random.default_rng(99).normal(size=len(poisson_data))
        spec = "Y ~ X\nZ ~ X\nY ~~ Z"
        with pytest.raises(ValueError, match="(?i)gaussian"):
            pathmc.model(spec, data=poisson_data, families={"Y": "poisson"})


class TestNegBinomialCompilation:
    """NegBinomial family compiles with dispersion parameter."""

    def test_negbin_compiles(self, negbin_data):
        model = pathmc.model("Y ~ X", data=negbin_data, families={"Y": "negbinomial"})
        assert model.pymc_model is not None

    def test_dispersion_param_exists(self, negbin_data):
        model = pathmc.model("Y ~ X", data=negbin_data, families={"Y": "negbinomial"})
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert any("alpha" in name and "Y" in name for name in free_names) or any(
            "disp" in name for name in free_names
        )

    def test_residual_cov_with_negbin_raises(self, negbin_data):
        negbin_data = negbin_data.copy()
        negbin_data["Z"] = np.random.default_rng(99).normal(size=len(negbin_data))
        spec = "Y ~ X\nZ ~ X\nY ~~ Z"
        with pytest.raises(ValueError, match="(?i)gaussian"):
            pathmc.model(spec, data=negbin_data, families={"Y": "negbinomial"})


class TestStudentTCompilation:
    """StudentT family compiles with nu parameter."""

    def test_studentt_compiles(self, studentt_data):
        model = pathmc.model("Y ~ X", data=studentt_data, families={"Y": "studentt"})
        assert model.pymc_model is not None

    def test_nu_param_exists(self, studentt_data):
        model = pathmc.model("Y ~ X", data=studentt_data, families={"Y": "studentt"})
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert any("nu" in name for name in free_names)

    def test_sigma_exists_for_studentt(self, studentt_data):
        model = pathmc.model("Y ~ X", data=studentt_data, families={"Y": "studentt"})
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "sigma_Y" in free_names


@pytest.mark.slow
class TestPoissonSampling:
    """Poisson model samples and produces sensible do() results."""

    def test_poisson_samples(self, fitted_poisson):
        summary = fitted_poisson.summary()
        assert summary is not None
        assert len(summary) > 0

    def test_poisson_do_positive_counts(self, fitted_poisson):
        """do(kind='mean') should return positive values (exp of linear predictor)."""
        result = fitted_poisson.do(set={"X": 1.0})
        assert result.mean("Y") > 0

    def test_poisson_do_higher_x_higher_count(self, fitted_poisson):
        r_low = fitted_poisson.do(set={"X": -1.0})
        r_high = fitted_poisson.do(set={"X": 1.0})
        assert r_high.mean("Y") > r_low.mean("Y")


@pytest.mark.slow
class TestNegBinSampling:
    """NegBinomial model samples."""

    def test_negbin_samples(self, fitted_negbin):
        summary = fitted_negbin.summary()
        assert summary is not None

    def test_negbin_do_positive(self, fitted_negbin):
        result = fitted_negbin.do(set={"X": 1.0})
        assert result.mean("Y") > 0


@pytest.mark.slow
class TestStudentTSampling:
    """StudentT model samples."""

    def test_studentt_samples(self, fitted_studentt):
        summary = fitted_studentt.summary()
        assert summary is not None

    def test_studentt_do_works(self, fitted_studentt):
        r0 = fitted_studentt.do(set={"X": 0.0})
        r1 = fitted_studentt.do(set={"X": 1.0})
        ate = r1 - r0
        assert np.isfinite(ate.mean("Y"))


@pytest.mark.slow
class TestPredictiveNewFamilies:
    """do(kind='predictive') works for new families."""

    def test_poisson_predictive(self, fitted_poisson):
        result = fitted_poisson.do(set={"X": 1.0}, kind="predictive")
        assert np.isfinite(result.mean("Y"))

    def test_studentt_predictive(self, fitted_studentt):
        result = fitted_studentt.do(set={"X": 1.0}, kind="predictive")
        assert np.isfinite(result.mean("Y"))
