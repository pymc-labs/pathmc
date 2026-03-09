"""Tests for custom prior support (issue #79)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

import pathmc
from pathmc.model import fit
from pathmc.parse import parse_spec
from pathmc.priors import (
    _ensure_dims,
    default_priors,
    merge_priors,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 100
    X = rng.normal(size=n)
    Z = rng.normal(size=n)
    M = 0.5 * X + rng.normal(size=n) * 0.3
    Y = 0.3 * X + 0.4 * M + rng.normal(size=n) * 0.5
    return pd.DataFrame({"X": X, "Z": Z, "M": M, "Y": Y})


@pytest.fixture()
def mediation_spec() -> str:
    return """
    M ~ X
    Y ~ X + M
    """


@pytest.fixture()
def panel_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    units = ["A", "B", "C"]
    times = list(range(10))
    rows = []
    for u in units:
        for t in times:
            rows.append({"unit": u, "time": t, "X": rng.normal(), "Y": rng.normal()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# default_priors
# ---------------------------------------------------------------------------


class TestDefaultPriors:
    def test_simple_regression_keys(self, simple_data: pd.DataFrame) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec)
        assert "beta_Y" in p
        assert "sigma_Y" in p
        assert isinstance(p["beta_Y"], Prior)
        assert isinstance(p["sigma_Y"], Prior)

    def test_default_beta_is_normal_0_10(self, simple_data: pd.DataFrame) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec)
        assert p["beta_Y"].distribution == "Normal"
        assert p["beta_Y"].parameters["mu"] == 0
        assert p["beta_Y"].parameters["sigma"] == 10

    def test_default_sigma_is_halfnormal_1(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec)
        assert p["sigma_Y"].distribution == "HalfNormal"
        assert p["sigma_Y"].parameters["sigma"] == 1

    def test_mediation_keys(self) -> None:
        spec = parse_spec("M ~ X\nY ~ X + M")
        p = default_priors(spec)
        assert "beta_M" in p
        assert "sigma_M" in p
        assert "beta_Y" in p
        assert "sigma_Y" in p

    def test_studentt_includes_nu(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, families={"Y": "studentt"})
        assert "nu_Y" in p
        assert p["nu_Y"].distribution == "Gamma"

    def test_negbinomial_includes_alpha_disp(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, families={"Y": "negbinomial"})
        assert "alpha_disp_Y" in p
        assert "sigma_Y" not in p

    def test_bernoulli_no_sigma(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, families={"Y": "bernoulli"})
        assert "sigma_Y" not in p

    def test_pooling_partial_includes_random_intercept(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, pooling="partial")
        assert "mu_alpha_Y" in p
        assert "sigma_alpha_Y" in p

    def test_pooling_slopes_includes_random_slopes(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, pooling={"intercept": True, "slopes": ["X"]})
        assert "mu_slope_Y_X" in p
        assert "sigma_slope_Y_X" in p

    def test_latent_normal_includes_sigma(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(
            spec,
            families={"Y": "latent_normal"},
            latent={"Y"},
        )
        assert "sigma_Y" in p

    def test_pure_latent_no_sigma(self) -> None:
        spec = parse_spec("Y ~ X")
        p = default_priors(spec, latent={"Y"})
        assert "sigma_Y" not in p


# ---------------------------------------------------------------------------
# merge_priors
# ---------------------------------------------------------------------------


class TestMergePriors:
    def test_no_overrides_returns_defaults(self) -> None:
        spec = parse_spec("Y ~ X")
        defaults = default_priors(spec)
        merged = merge_priors(defaults, None)
        assert merged == defaults

    def test_override_replaces_single_key(self) -> None:
        spec = parse_spec("Y ~ X")
        defaults = default_priors(spec)
        custom = Prior("Normal", mu=0, sigma=2)
        merged = merge_priors(defaults, {"beta_Y": custom})
        assert merged["beta_Y"] is custom
        assert merged["sigma_Y"] is defaults["sigma_Y"]

    def test_unknown_key_raises(self) -> None:
        spec = parse_spec("Y ~ X")
        defaults = default_priors(spec)
        with pytest.raises(ValueError, match="Unknown prior key"):
            merge_priors(defaults, {"nonexistent": Prior("Normal")})


# ---------------------------------------------------------------------------
# _ensure_dims
# ---------------------------------------------------------------------------


class TestEnsureDims:
    def test_none_dims_passthrough(self) -> None:
        p = Prior("Normal", mu=0, sigma=1)
        result = _ensure_dims(p, None)
        assert result is p

    def test_sets_dims_when_empty(self) -> None:
        p = Prior("Normal", mu=0, sigma=1)
        result = _ensure_dims(p, "Y_predictors")
        assert result.dims == ("Y_predictors",)
        assert result is not p

    def test_matching_dims_passthrough(self) -> None:
        p = Prior("Normal", mu=0, sigma=1, dims="Y_predictors")
        result = _ensure_dims(p, "Y_predictors")
        assert result is p

    def test_conflicting_dims_raises(self) -> None:
        p = Prior("Normal", mu=0, sigma=1, dims="wrong")
        with pytest.raises(ValueError, match="model requires"):
            _ensure_dims(p, "Y_predictors")


# ---------------------------------------------------------------------------
# fit() with priors
# ---------------------------------------------------------------------------


class TestFitWithPriors:
    def test_fit_default_priors_compiles(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        assert m.pymc_model is not None

    def test_fit_custom_beta_compiles(self, simple_data: pd.DataFrame) -> None:
        m = fit(
            "Y ~ X",
            simple_data,
            priors={"beta_Y": Prior("Normal", mu=0, sigma=2)},
        )
        assert m.pymc_model is not None

    def test_fit_custom_sigma_compiles(self, simple_data: pd.DataFrame) -> None:
        m = fit(
            "Y ~ X",
            simple_data,
            priors={"sigma_Y": Prior("Exponential", lam=1)},
        )
        assert m.pymc_model is not None

    def test_fit_invalid_key_raises(self, simple_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown prior key"):
            fit(
                "Y ~ X",
                simple_data,
                priors={"nonexistent": Prior("Normal")},
            )

    def test_custom_prior_reflected_in_priors_table(
        self, simple_data: pd.DataFrame
    ) -> None:
        m = fit(
            "Y ~ X",
            simple_data,
            priors={"beta_Y": Prior("Normal", mu=0, sigma=2)},
        )
        table_str = str(m.priors())
        assert "sigma=2" in table_str

    def test_mediation_custom_priors(
        self,
        simple_data: pd.DataFrame,
        mediation_spec: str,
    ) -> None:
        m = fit(
            mediation_spec,
            simple_data,
            priors={
                "beta_M": Prior("Normal", mu=0, sigma=5),
                "beta_Y": Prior("Normal", mu=0, sigma=5),
            },
        )
        table_str = str(m.priors())
        assert "sigma=5" in table_str


# ---------------------------------------------------------------------------
# set_priors
# ---------------------------------------------------------------------------


class TestSetPriors:
    def test_set_priors_recompiles(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        old_model = m.pymc_model
        m.set_priors({"beta_Y": Prior("Normal", mu=0, sigma=2)})
        assert m.pymc_model is not old_model

    def test_set_priors_merges(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        m.set_priors({"beta_Y": Prior("Normal", mu=0, sigma=2)})
        table_str = str(m.priors())
        assert "sigma=2" in table_str
        assert "sigma_Y" in table_str

    def test_set_priors_invalidates_idata(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        m._idata = "placeholder"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m.set_priors({"beta_Y": Prior("Normal", mu=0, sigma=2)})
            assert len(w) == 1
            assert "discarded" in str(w[0].message)
        assert m._idata is None

    def test_set_priors_invalid_key_raises(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        with pytest.raises(ValueError, match="Unknown prior key"):
            m.set_priors({"bad_key": Prior("Normal")})

    def test_multiple_set_priors_calls(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        m.set_priors({"beta_Y": Prior("Normal", mu=0, sigma=2)})
        m.set_priors({"sigma_Y": Prior("Exponential", lam=0.5)})
        table_str = str(m.priors())
        assert "sigma=2" in table_str
        assert "lam=0.5" in table_str


# ---------------------------------------------------------------------------
# sample_prior_predictive
# ---------------------------------------------------------------------------


class TestSamplePriorPredictive:
    def test_returns_inference_data(self, simple_data: pd.DataFrame) -> None:
        import arviz as az

        m = fit("Y ~ X", simple_data)
        ppc = m.sample_prior_predictive(draws=5, random_seed=42)
        assert isinstance(ppc, az.InferenceData)
        assert "prior" in ppc.groups()

    def test_prior_predictive_has_model_vars(self, simple_data: pd.DataFrame) -> None:
        m = fit("Y ~ X", simple_data)
        ppc = m.sample_prior_predictive(draws=5, random_seed=42)
        assert "beta_Y" in ppc.prior.data_vars
        assert "sigma_Y" in ppc.prior.data_vars

    def test_prior_has_outcome_vars(self, simple_data: pd.DataFrame) -> None:
        """Outcome variables live in the prior group (free RVs in the generative model)."""
        m = fit("Y ~ X", simple_data)
        ppc = m.sample_prior_predictive(draws=5, random_seed=42)
        assert "Y" in ppc.prior.data_vars


# ---------------------------------------------------------------------------
# Prior from pathmc namespace
# ---------------------------------------------------------------------------


class TestPriorReexport:
    def test_prior_importable_from_pathmc(self) -> None:
        assert pathmc.Prior is Prior

    def test_prior_in_all(self) -> None:
        assert "Prior" in pathmc.__all__


# ---------------------------------------------------------------------------
# PyMC model structure with custom priors
# ---------------------------------------------------------------------------


class TestModelStructure:
    def test_custom_beta_distribution_in_model(self, simple_data: pd.DataFrame) -> None:
        """Custom beta prior should use the specified distribution."""
        m = fit(
            "Y ~ X",
            simple_data,
            priors={"beta_Y": Prior("Laplace", mu=0, b=1)},
        )
        var_names = [v.name for v in m._gen_model.free_RVs]
        assert "beta_Y" in var_names

    def test_custom_sigma_distribution_in_model(
        self, simple_data: pd.DataFrame
    ) -> None:
        """Custom sigma prior should use the specified distribution."""
        m = fit(
            "Y ~ X",
            simple_data,
            priors={"sigma_Y": Prior("Exponential", lam=1)},
        )
        var_names = [v.name for v in m._gen_model.free_RVs]
        assert "sigma_Y" in var_names

    def test_hierarchical_beta_prior(self, simple_data: pd.DataFrame) -> None:
        """Hierarchical Prior creates parent variables."""
        m = fit(
            "Y ~ X",
            simple_data,
            priors={
                "beta_Y": Prior(
                    "Normal",
                    mu=Prior("Normal", mu=0, sigma=1),
                    sigma=Prior("HalfNormal", sigma=1),
                ),
            },
        )
        var_names = [v.name for v in m._gen_model.free_RVs]
        assert "beta_Y" in var_names
        assert "beta_Y_mu" in var_names
        assert "beta_Y_sigma" in var_names
