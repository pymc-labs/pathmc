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
"""Tests for stochastic latent nodes and sparse measurement equations."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.graph import build_graph
from pathmc.identify import adjustment_sets, collider_warnings, is_identifiable
from pathmc.parse import parse_spec


# ---------------------------------------------------------------------------
# Compilation — stochastic latent (latent_normal)
# ---------------------------------------------------------------------------


class TestStochasticLatentCompilation:
    """latent_normal family emits process noise for latent nodes."""

    @pytest.fixture
    def chain_data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=n)
        Y = 0.5 * X + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"X": X, "Y": Y})

    def test_compiles_with_latent_normal(self, chain_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=chain_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        assert model.pymc_model is not None

    def test_latent_normal_has_sigma(self, chain_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=chain_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        free_names = [rv.name for rv in model.pymc_model.free_RVs]
        assert "sigma_M" in free_names
        assert "sigma_Y" in free_names

    def test_latent_normal_is_free_rv(self, chain_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=chain_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "M" in free_names

    def test_latent_normal_not_observed(self, chain_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=chain_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        obs_names = {rv.name for rv in model.pymc_model.observed_RVs}
        assert "M" not in obs_names
        assert "Y" in obs_names


# ---------------------------------------------------------------------------
# Compilation — sparse measurement equations
# ---------------------------------------------------------------------------


class TestSparseMeasurement:
    """Endogenous variables with NaN data compile as masked likelihoods."""

    @pytest.fixture
    def sparse_data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=n)
        M_true = 0.5 * X
        Y = 0.8 * M_true + rng.normal(scale=0.5, size=n)
        M_obs = np.full(n, np.nan)
        observed_idx = rng.choice(n, size=20, replace=False)
        M_obs[observed_idx] = M_true[observed_idx] + rng.normal(
            scale=0.2, size=len(observed_idx)
        )
        return pd.DataFrame({"X": X, "Y": Y, "M_obs": M_obs})

    def test_sparse_compiles(self, sparse_data):
        model = pathmc.model(
            "M ~ X\nY ~ M\nM_obs ~ 0 + 1*M",
            data=sparse_data,
            latent=["M"],
        )
        assert model.pymc_model is not None

    def test_sparse_observed_is_masked(self, sparse_data):
        model = pathmc.model(
            "M ~ X\nY ~ M\nM_obs ~ 0 + 1*M",
            data=sparse_data,
            latent=["M"],
        )
        obs_names = {rv.name for rv in model.pymc_model.observed_RVs}
        has_m_obs = any("M_obs" in name for name in obs_names)
        assert has_m_obs, f"Expected M_obs-related observed RV, got {obs_names}"

    def test_sparse_with_latent_normal(self, sparse_data):
        """Stochastic latent + sparse measurement together."""
        model = pathmc.model(
            "M ~ X\nY ~ M\nM_obs ~ 0 + 1*M",
            data=sparse_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        free_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "sigma_M" in free_names
        assert "M" in free_names
        obs_names = {rv.name for rv in model.pymc_model.observed_RVs}
        has_m_obs = any("M_obs" in name for name in obs_names)
        assert has_m_obs, f"Expected M_obs-related observed RV, got {obs_names}"


# ---------------------------------------------------------------------------
# Introspection — stochastic latent
# ---------------------------------------------------------------------------


class TestStochasticLatentIntrospection:
    """Introspection distinguishes stochastic from deterministic latent."""

    @pytest.fixture
    def stochastic_model(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=n)
        Y = 0.5 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "Y": Y})
        return pathmc.model(
            "M ~ X\nY ~ M",
            data=df,
            latent=["M"],
            families={"M": "latent_normal"},
        )

    def test_equations_annotate_stochastic(self, stochastic_model):
        eqs = stochastic_model.equations()
        text = str(eqs)
        assert "stochastic" in text.lower()
        assert "deterministic" not in text.lower()

    def test_priors_include_sigma_for_latent_normal(self, stochastic_model):
        priors = stochastic_model.priors()
        text = str(priors)
        assert "sigma_M" in text


# ---------------------------------------------------------------------------
# Identification — latent-aware adjustment sets
# ---------------------------------------------------------------------------


class TestLatentIdentification:
    """adjustment_sets excludes latent nodes from candidate sets."""

    def test_adjustment_set_excludes_latent(self):
        spec = parse_spec("M ~ X\nY ~ M + X")
        gi = build_graph(spec, latent={"M"})
        sets = adjustment_sets(gi, "X", "Y")
        for s in sets:
            assert "M" not in s, f"Latent node M should not appear in {s}"

    def test_identifiable_without_latent_set(self):
        """X -> M (latent) -> Y with X -> Y: the empty set is valid."""
        spec = parse_spec("M ~ X\nY ~ M + X")
        gi = build_graph(spec, latent={"M"})
        assert is_identifiable(gi, "X", "Y")

    def test_collider_warnings_for_latent(self):
        spec = parse_spec("M ~ X\nY ~ M")
        gi = build_graph(spec, latent={"M"})
        warnings = collider_warnings(gi, {"M"}, "X", "Y")
        assert any("latent" in w.lower() or "unobserved" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# do() through stochastic latent — requires sampling
# ---------------------------------------------------------------------------


class TestStochasticLatentDo:
    """do() propagates through stochastic latent mediators."""

    @pytest.fixture(scope="class")
    def fitted_stochastic_model(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=n)
        M_true = 0.8 * X + rng.normal(scale=0.3, size=n)
        Y = 0.6 * M_true + rng.normal(scale=0.5, size=n)
        M_obs = np.full(n, np.nan)
        obs_idx = rng.choice(n, size=int(0.3 * n), replace=False)
        M_obs[obs_idx] = M_true[obs_idx] + rng.normal(scale=0.2, size=len(obs_idx))
        df = pd.DataFrame({"X": X, "Y": Y, "M_obs": M_obs})

        model = pathmc.model(
            "M ~ X\nY ~ M\nM_obs ~ 0 + 1*M",
            data=df,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        model.fit(draws=200, tune=200, chains=2, random_seed=42)
        return model

    @pytest.mark.slow
    def test_ate_through_stochastic_latent(self, fitted_stochastic_model):
        ate = fitted_stochastic_model.ate("Y", "X")
        assert ate.mean("Y") > 0.3

    @pytest.mark.slow
    def test_do_mean_returns_values(self, fitted_stochastic_model):
        result = fitted_stochastic_model.do(set={"X": 1.0}, kind="mean")
        assert result.mean("Y") is not None
        assert result.mean("M") is not None

    @pytest.mark.slow
    def test_do_predictive_returns_values(self, fitted_stochastic_model):
        result = fitted_stochastic_model.do(set={"X": 1.0}, kind="predictive")
        assert result.mean("Y") is not None

    @pytest.mark.slow
    def test_hdi_has_width(self, fitted_stochastic_model):
        ate = fitted_stochastic_model.ate("Y", "X")
        hdi = ate.hdi("Y")
        assert hdi[1] > hdi[0]


# ---------------------------------------------------------------------------
# simulate() with latent nodes
# ---------------------------------------------------------------------------


class TestSimulateWithLatent:
    """simulate() supports latent nodes."""

    def test_simulate_deterministic_latent(self):
        exog = pd.DataFrame({"X": np.random.default_rng(42).normal(size=100)})
        df = pathmc.simulate(
            "M ~ X\nY ~ M",
            data=exog,
            params={"beta_M": [0.0, 1.0], "beta_Y": [0.0, 0.8], "sigma_Y": 0.5},
            latent=["M"],
            random_seed=42,
        )
        assert "Y" in df.columns
        assert "M" in df.columns
        assert np.corrcoef(df["X"], df["Y"])[0, 1] > 0.3

    def test_simulate_stochastic_latent(self):
        exog = pd.DataFrame({"X": np.random.default_rng(42).normal(size=100)})
        df = pathmc.simulate(
            "M ~ X\nY ~ M",
            data=exog,
            params={
                "beta_M": [0.0, 1.0],
                "sigma_M": 0.3,
                "beta_Y": [0.0, 0.8],
                "sigma_Y": 0.5,
            },
            latent=["M"],
            families={"M": "latent_normal"},
            random_seed=42,
        )
        assert "Y" in df.columns
        assert "M" in df.columns
