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
"""Tests for issue #149: effect() and standardized() guards for non-Gaussian models."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.effects import _guard_non_gaussian_path


class TestGuardNonGaussianPath:
    """Unit tests for the guard helper (no MCMC needed)."""

    def test_no_families_passes(self):
        _guard_non_gaussian_path(["X", "M", "Y"], None, "X -> M -> Y")

    def test_all_gaussian_passes(self):
        families = {"M": "gaussian", "Y": "gaussian"}
        _guard_non_gaussian_path(["X", "M", "Y"], families, "X -> M -> Y")

    def test_bernoulli_raises(self):
        families = {"M": "bernoulli"}
        with pytest.raises(NotImplementedError, match="non-Gaussian"):
            _guard_non_gaussian_path(["X", "M", "Y"], families, "X -> M -> Y")

    def test_poisson_single_edge_passes(self):
        """Single-edge path: one coefficient, no cross-scale product."""
        families = {"Y": "poisson"}
        _guard_non_gaussian_path(["X", "Y"], families, "X -> Y")

    def test_negbinomial_single_edge_passes(self):
        families = {"Y": "negbinomial"}
        _guard_non_gaussian_path(["X", "Y"], families, "X -> Y")

    def test_multi_hop_poisson_target_raises(self):
        families = {"Y": "poisson"}
        with pytest.raises(NotImplementedError, match="non-Gaussian"):
            _guard_non_gaussian_path(["X", "M", "Y"], families, "X -> M -> Y")

    def test_message_suggests_do(self):
        families = {"M": "bernoulli"}
        with pytest.raises(NotImplementedError, match="do\\(\\)-based simulation"):
            _guard_non_gaussian_path(["X", "M", "Y"], families, "X -> M -> Y")

    def test_exogenous_non_gaussian_passes(self):
        """Source-node family does not affect the target regression coefficient."""
        families = {"X": "poisson"}
        _guard_non_gaussian_path(["X", "Y"], families, "X -> Y")

    def test_bernoulli_mediator_single_edge_passes(self):
        """M -> Y with bernoulli M: coefficient is on Y's Gaussian scale."""
        families = {"M": "bernoulli", "Y": "gaussian"}
        _guard_non_gaussian_path(["M", "Y"], families, "M -> Y")


class TestEffectNonGaussianIntegration:
    """Integration tests using PathModel with mock sampling."""

    def test_effect_raises_for_bernoulli_mediator(self, mock_pymc_sample):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        M = (X > 0).astype(float)
        Y = M + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model(
            "M ~ X\nY ~ M + X",
            data=df,
            families={"M": "bernoulli"},
        )
        m.fit(random_seed=42)

        with pytest.raises(NotImplementedError, match="non-Gaussian"):
            m.effect("X -> M -> Y")

    def test_effect_ok_for_bernoulli_mediator_direct_path(self, mock_pymc_sample):
        """Single-edge M -> Y is valid even when M is Bernoulli."""
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        M = (X > 0).astype(float)
        Y = M + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model(
            "M ~ X\nY ~ M + X",
            data=df,
            families={"M": "bernoulli"},
        )
        m.fit(random_seed=42)
        result = m.effect("M -> Y")
        assert result is not None

    def test_effect_ok_for_gaussian(self, mock_pymc_sample):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        M = 0.5 * X + rng.normal(scale=0.5, size=n)
        Y = 0.8 * M + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model("M ~ X\nY ~ M", data=df)
        m.fit(random_seed=42)
        result = m.effect("X -> M -> Y")
        assert result is not None

    def test_standardized_skips_bernoulli_with_warning(self, mock_pymc_sample):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        M = (X > 0).astype(float)
        Y = M + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model(
            "M ~ a*X\nY ~ b*M + c*X",
            data=df,
            families={"M": "bernoulli"},
        )
        m.fit(random_seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = m.standardized()

        warning_msgs = [str(x.message) for x in w]
        assert any("non-Gaussian" in msg for msg in warning_msgs)
        assert "a" not in result.index
        assert {"b", "c"} & set(result.index)

    def test_standardized_ok_for_gaussian(self, mock_pymc_sample):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        M = 0.5 * X + rng.normal(scale=0.5, size=n)
        Y = 0.8 * M + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model("M ~ a*X\nY ~ b*M", data=df)
        m.fit(random_seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = m.standardized()

        non_gaussian_warnings = [x for x in w if "non-Gaussian" in str(x.message)]
        assert len(non_gaussian_warnings) == 0
        assert "a" in result.index
