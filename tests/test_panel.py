"""Gate tests for M14: Panel-aware fit() + random intercepts."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def panel_data():
    """Panel data: 3 regions, 20 weeks, Y ~ X with region-level intercepts."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 20
    true_intercepts = {"A": 1.0, "B": 3.0, "C": 5.0}
    rows = []
    for region in regions:
        for week in range(1, n_weeks + 1):
            x = rng.normal()
            y = true_intercepts[region] + 0.5 * x + rng.normal(scale=0.3)
            rows.append({"region": region, "week": week, "X": x, "Y": y})
    return pd.DataFrame(rows)


@pytest.fixture()
def simple_spec():
    return "Y ~ X"


class TestPanelCompilation:
    """Panel model compiles correctly with random intercepts."""

    def test_panel_model_compiles(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        assert model.pymc_model is not None

    def test_random_intercept_rvs_exist(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "alpha_Y" in rv_names
        assert "mu_alpha_Y" in rv_names
        assert "sigma_alpha_Y" in rv_names

    def test_unit_coord_set(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        coords = model.pymc_model.coords
        assert "unit" in coords
        assert set(coords["unit"]) == {"A", "B", "C"}

    def test_beta_still_exists(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "beta_Y" in rv_names


class TestPanelIntrospection:
    """Priors include group-level parameters."""

    def test_priors_include_group_params(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        prior_table = model.priors()
        prior_str = repr(prior_table)
        assert "mu_alpha_Y" in prior_str
        assert "sigma_alpha_Y" in prior_str
        assert "alpha_Y" in prior_str


class TestCrossSectionalUnchanged:
    """Cross-sectional behavior unchanged when panel=None."""

    def test_no_panel_no_alpha(self, panel_data, simple_spec):
        model = pathmc.model(simple_spec, data=panel_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "alpha_Y" not in rv_names

    def test_no_panel_no_unit_coord(self, panel_data, simple_spec):
        model = pathmc.model(simple_spec, data=panel_data)
        assert "unit" not in model.pymc_model.coords


@pytest.mark.slow
class TestPanelSampling:
    """Panel model samples correctly."""

    def test_sampling_completes(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        idata = model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        assert idata is not None

    def test_summary_includes_alpha(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        summary = model.summary()
        assert any("alpha_Y" in idx for idx in summary.index)

    def test_do_works_with_panel(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r0 = model.do(set={"X": 0.0})
        r1 = model.do(set={"X": 1.0})
        ate = r1 - r0
        assert 0.0 < ate.mean("Y") < 2.0


class TestMultipleEndogenous:
    """Panel with multiple endogenous variables gets intercepts for each."""

    def test_multi_endo_intercepts(self, panel_data):
        panel_data = panel_data.copy()
        rng = np.random.default_rng(99)
        panel_data["M"] = 0.3 * panel_data["X"] + rng.normal(size=len(panel_data))

        spec = """
        M ~ X
        Y ~ M + X
        """
        model = pathmc.model(
            spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "alpha_M" in rv_names
        assert "alpha_Y" in rv_names
