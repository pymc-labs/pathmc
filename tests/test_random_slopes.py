"""Gate tests for M15: Random slopes."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def panel_data():
    """Panel: 3 regions, 20 weeks, Y ~ X with varying slopes."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    true_slopes = {"A": 0.3, "B": 0.8, "C": 1.5}
    rows = []
    for region in regions:
        for week in range(1, 21):
            x = rng.normal()
            y = 2.0 + true_slopes[region] * x + rng.normal(scale=0.3)
            rows.append({"region": region, "week": week, "X": x, "Y": y})
    return pd.DataFrame(rows)


class TestRandomSlopeCompilation:
    """Random slope RVs are created in the PyMC model."""

    def test_slope_rvs_exist(self, panel_data):
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "slope_Y_X" in rv_names
        assert "mu_slope_Y_X" in rv_names
        assert "sigma_slope_Y_X" in rv_names

    def test_slope_has_unit_dim(self, panel_data):
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "slope_Y_X" in rv_names
        coords = model.pymc_model.coords
        assert "unit" in coords

    def test_intercept_also_random(self, panel_data):
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "alpha_Y" in rv_names

    def test_fixed_coefficients_unchanged(self, panel_data):
        """Beta for X still exists as a fixed effect."""
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "beta_Y" in rv_names

    def test_slope_only_for_specified_vars(self, panel_data):
        """Only 'X' gets a random slope, not other predictors."""
        panel_data = panel_data.copy()
        rng = np.random.default_rng(99)
        panel_data["Z"] = rng.normal(size=len(panel_data))

        model = pathmc.model(
            "Y ~ X + Z",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "slope_Y_X" in rv_names
        assert "slope_Y_Z" not in rv_names


class TestRandomSlopePriors:
    """Priors table includes slope parameters."""

    def test_priors_include_slope_params(self, panel_data):
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        prior_str = repr(model.priors())
        assert "mu_slope_Y_X" in prior_str
        assert "sigma_slope_Y_X" in prior_str
        assert "slope_Y_X" in prior_str


@pytest.mark.slow
class TestRandomSlopeSampling:
    """Random slope model samples without error."""

    def test_sampling_completes(self, panel_data):
        model = pathmc.model(
            "Y ~ X",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling={"intercept": True, "slopes": ["X"]},
        )
        idata = model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        assert idata is not None
        summary = model.summary()
        assert any("slope_Y_X" in idx for idx in summary.index)
