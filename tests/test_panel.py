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
"""Gate tests for M14: Panel-aware fit() + random intercepts."""

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v1 as nw

import pathmc
import pathmc.panel


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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

    @pytest.fixture(scope="class")
    def fitted_panel(self, panel_data, simple_spec):
        model = pathmc.model(
            simple_spec,
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        idata = model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        return model, idata

    def test_sampling_completes(self, fitted_panel):
        _, idata = fitted_panel
        assert idata is not None

    def test_summary_includes_alpha(self, fitted_panel):
        model, _ = fitted_panel
        summary = model.summary()
        assert any("alpha_Y" in idx for idx in summary.index)

    def test_do_works_with_panel(self, fitted_panel):
        model, _ = fitted_panel
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


@pytest.fixture(scope="module")
def geo_brand_data():
    """2 geos x 2 brands x 10 weeks rectangular panel."""
    rng = np.random.default_rng(0)
    rows = []
    for geo in ["North", "South"]:
        for brand in ["Acme", "Bolt"]:
            for week in range(10):
                rows.append({
                    "geo": geo,
                    "brand": brand,
                    "week": week,
                    "tv": rng.uniform(5.0, 30.0),
                })
    return pd.DataFrame(rows)


class TestMultiDimensionalPanel:
    """panel={'unit': [cols], 'time': col} builds a composite unit key."""

    def test_composite_unit_labels(self, geo_brand_data):
        info, df_out = pathmc.panel.build_panel_info(
            nw.from_native(geo_brand_data, eager_only=True),
            {"unit": ["geo", "brand"], "time": "week"},
        )
        assert info.is_multi_dim
        assert info.unit == "geo|brand"
        assert info.unit_columns == ("geo", "brand")
        assert info.unit_labels == [
            "North|Acme",
            "North|Bolt",
            "South|Acme",
            "South|Bolt",
        ]
        assert "geo|brand" in df_out.columns

    def test_ragged_panel_rejected(self, geo_brand_data):
        bad = geo_brand_data[
            ~((geo_brand_data["geo"] == "South") & (geo_brand_data["week"] == 9))
        ]
        with pytest.raises(ValueError, match="not rectangular"):
            pathmc.simulate(
                "sales ~ tv",
                data=bad,
                params={"beta_sales": [1.0, 2.0], "sigma_sales": 1.0},
                panel={"unit": ["geo", "brand"], "time": "week"},
            )

    def test_simulate_multi_dim_honors_cell_intercepts(self, geo_brand_data):
        out = pathmc.simulate(
            "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)",
            data=geo_brand_data,
            params={
                "beta_sales": [2.5],
                "theta_tv": 0.7,
                "lam_tv": 0.3,
                "sigma_sales": 1e-8,
                "alpha_sales": [46.0, 51.0, 56.0, 61.0],
                "mu_alpha_sales": 53.0,
                "sigma_alpha_sales": 5.0,
            },
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling="partial",
            random_seed=42,
        )
        out_nw = nw.from_native(out, eager_only=True)
        means = (
            out_nw
            .group_by(["geo", "brand"])
            .agg(nw.col("sales").mean().alias("mean_sales"))
            .to_native()
            .set_index(["geo", "brand"])["mean_sales"]
        )
        got = means.to_dict()
        truth = {
            ("North", "Acme"): 46.0,
            ("North", "Bolt"): 51.0,
            ("South", "Acme"): 56.0,
            ("South", "Bolt"): 61.0,
        }
        for cell, intercept in truth.items():
            # zero-noise run: mean = intercept + saturation lift (> 0)
            assert got[cell] > intercept
            assert got[cell] < intercept + 4.0

    def test_single_column_panel_unchanged(self, geo_brand_data):
        wide = geo_brand_data.assign(
            cell=geo_brand_data["geo"] + "|" + geo_brand_data["brand"]
        ).drop(columns=["geo", "brand"])
        info, _ = pathmc.panel.build_panel_info(
            nw.from_native(wide, eager_only=True),
            {"unit": "cell", "time": "week"},
        )
        assert not info.is_multi_dim
        assert info.unit == "cell"
        assert len(info.unit_labels) == 4
