"""Tests for latent deterministic mediators."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.graph import build_graph
from pathmc.parse import parse_spec


# ---------------------------------------------------------------------------
# Graph layer
# ---------------------------------------------------------------------------


class TestGraphLatent:
    """build_graph accepts and validates latent variables."""

    def test_latent_stored_on_graph_info(self):
        spec = parse_spec("M ~ X\nY ~ M")
        gi = build_graph(spec, latent={"M"})
        assert gi.latent == {"M"}

    def test_latent_must_be_endogenous(self):
        spec = parse_spec("Y ~ X")
        with pytest.raises(ValueError, match="not endogenous"):
            build_graph(spec, latent={"X"})

    def test_latent_default_empty(self):
        spec = parse_spec("Y ~ X")
        gi = build_graph(spec)
        assert gi.latent == set()

    def test_latent_in_topological_order(self):
        spec = parse_spec("M ~ X\nY ~ M")
        gi = build_graph(spec, latent={"M"})
        order = gi.topological_order
        assert order.index("X") < order.index("M") < order.index("Y")


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class TestLatentCompilation:
    """Latent mediators compile without data columns."""

    @pytest.fixture
    def chain_data(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=n)
        Y = 0.5 * X + rng.normal(scale=0.5, size=n)
        return pd.DataFrame({"X": X, "Y": Y})

    def test_compiles_with_latent(self, chain_data):
        model = pathmc.fit("M ~ X\nY ~ M", data=chain_data, latent=["M"])
        assert model.pymc_model is not None

    def test_latent_has_no_sigma(self, chain_data):
        model = pathmc.fit("M ~ X\nY ~ M", data=chain_data, latent=["M"])
        free_names = [rv.name for rv in model.pymc_model.free_RVs]
        assert "sigma_M" not in free_names
        assert "sigma_Y" in free_names

    def test_latent_not_observed(self, chain_data):
        model = pathmc.fit("M ~ X\nY ~ M", data=chain_data, latent=["M"])
        obs_names = {rv.name for rv in model.pymc_model.observed_RVs}
        assert "M" not in obs_names
        assert "Y" in obs_names

    def test_mu_deterministics_exist(self, chain_data):
        model = pathmc.fit("M ~ X\nY ~ M", data=chain_data, latent=["M"])
        det_names = {d.name for d in model.pymc_model.deterministics}
        assert "mu_M" in det_names
        assert "mu_Y" in det_names

    def test_non_latent_endogenous_must_be_in_data(self):
        df = pd.DataFrame({"X": [1, 2, 3]})
        with pytest.raises(ValueError, match="not found in data"):
            pathmc.fit("Y ~ X", data=df)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


class TestLatentIntrospection:
    """Introspection methods reflect latent status."""

    @pytest.fixture
    def latent_model(self):
        rng = np.random.default_rng(42)
        n = 100
        X = rng.normal(size=n)
        Y = 0.5 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "Y": Y})
        return pathmc.fit("M ~ X\nY ~ M", data=df, latent=["M"])

    def test_equations_annotate_latent(self, latent_model):
        eqs = latent_model.equations()
        text = str(eqs)
        assert "deterministic" in text.lower()

    def test_graph_renders(self, latent_model):
        dot = latent_model.graph()
        assert dot is not None
        src = dot.source
        assert "M" in src

    def test_priors_skip_sigma_for_latent(self, latent_model):
        priors = latent_model.priors()
        text = str(priors)
        assert "sigma_M" not in text
        assert "sigma_Y" in text


# ---------------------------------------------------------------------------
# do() with latent mediators
# ---------------------------------------------------------------------------


class TestLatentDo:
    """do() propagates through latent mediators correctly."""

    @pytest.fixture(scope="class")
    def fitted_latent_model(self):
        rng = np.random.default_rng(42)
        n = 300
        X = rng.normal(size=n)
        Y = 0.8 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "Y": Y})

        model = pathmc.fit(
            "M ~ X\nY ~ M",
            data=df,
            latent=["M"],
        )
        model.sample(draws=200, tune=200, chains=2, random_seed=42)
        return model

    @pytest.mark.slow
    def test_ate_through_latent(self, fitted_latent_model):
        ate = fitted_latent_model.ate("Y", "X")
        assert ate.mean("Y") > 0.3

    @pytest.mark.slow
    def test_do_mean_returns_values(self, fitted_latent_model):
        result = fitted_latent_model.do(set={"X": 1.0}, kind="mean")
        assert result.mean("Y") is not None
        assert result.mean("M") is not None

    @pytest.mark.slow
    def test_do_predictive_returns_values(self, fitted_latent_model):
        result = fitted_latent_model.do(set={"X": 1.0}, kind="predictive")
        y_mean = result.mean("Y")
        assert y_mean is not None

    @pytest.mark.slow
    def test_hdi_has_width(self, fitted_latent_model):
        ate = fitted_latent_model.ate("Y", "X")
        hdi = ate.hdi("Y")
        assert hdi[1] > hdi[0]
