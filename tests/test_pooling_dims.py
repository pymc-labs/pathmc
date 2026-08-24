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
"""Structured hierarchical pooling across panel dimensions (pooling['by_var']).

Covers the ``by_var`` grammar added for issue #431:

- ``{"var": {"coefficient": dims}}`` emits a per-cell coefficient pooled
  toward ``mu_{var}_{dim}`` / scaled by ``sigma_{var}_{dim}``, where *dims*
  name panel['unit'] columns to pool over.
- ``{"param": "none"}`` forces unpooled per-cell parameters (coefficients
  and transform parameters alike).
"""

import warnings

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture(scope="module")
def geo_brand_data():
    """2 geos x 2 brands x 12 weeks rectangular panel."""
    rng = np.random.default_rng(7)
    rows = []
    for geo in ["North", "South"]:
        for brand in ["Acme", "Bolt"]:
            for week in range(12):
                rows.append({
                    "geo": geo,
                    "brand": brand,
                    "week": week,
                    "tv": rng.uniform(2.0, 20.0),
                    "radio": rng.uniform(0.5, 8.0),
                })
    return pd.DataFrame(rows)


POOL_BY_GEO = {
    "intercept": True,
    "by_var": {"tv": {"coefficient": ("geo",)}},
}


class TestByVarCoefficientRVs:
    """Pooled coefficients get per-cell RVs plus dim-subset hyperpriors."""

    @pytest.fixture(scope="class")
    def model(self, geo_brand_data):
        return pathmc.model(
            "sales ~ 0 + tv + radio",
            data=geo_brand_data.assign(
                sales=np.random.default_rng(0).normal(size=len(geo_brand_data))
            ),
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling=POOL_BY_GEO,
        )

    def test_rv_names_and_shapes(self, model):
        shapes = {
            rv.name: tuple(int(d) for d in rv.shape.eval())
            for rv in model.pymc_model.free_RVs
        }
        # Per-cell coefficient over the full composite grid...
        assert shapes["beta_tv"] == (4,)
        # ...pooled toward a geo-level mean, not a cell-level one.
        assert shapes["mu_tv_geo"] == (2,)
        assert shapes["sigma_tv_geo"] == ()
        assert "mu_tv" not in shapes

    def test_flat_beta_excludes_pooled_predictor(self, model):
        shapes = {rv.name for rv in model.pymc_model.free_RVs}
        # 'tv' is handled by by_var; only 'radio' stays in the flat beta.
        assert "beta_sales" in shapes
        coords = model.pymc_model.coords
        assert list(coords["sales_predictors"]) == ["radio"]

    def test_dim_coord_registered(self, model):
        assert set(model.pymc_model.coords["geo"]) == {"North", "South"}

    def test_priors_table_lists_hyperpriors(self, model):
        prior_str = repr(model.priors())
        assert "mu_tv_geo" in prior_str
        assert "sigma_tv_geo" in prior_str
        assert "beta_tv" in prior_str

    def test_user_can_override_hyperpriors(self, geo_brand_data):
        from pymc_extras.prior import Prior

        model = pathmc.model(
            "sales ~ 0 + tv",
            data=geo_brand_data.assign(
                sales=np.random.default_rng(1).normal(size=len(geo_brand_data))
            ),
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling=POOL_BY_GEO,
            priors={
                "mu_tv_geo": Prior("Normal", mu=1.0, sigma=2.0, dims=("geo",)),
            },
        )
        assert model.pymc_model is not None


class TestNoneCoefficient:
    """``"none"`` on a predictor forces unpooled per-cell coefficients."""

    def test_none_coefficient_rvs(self, geo_brand_data):
        model = pathmc.model(
            "sales ~ 0 + tv",
            data=geo_brand_data.assign(
                sales=np.random.default_rng(2).normal(size=len(geo_brand_data))
            ),
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling={"by_var": {"tv": "none"}},
        )
        shapes = {
            rv.name: tuple(int(d) for d in rv.shape.eval())
            for rv in model.pymc_model.free_RVs
        }
        assert shapes["beta_tv"] == (4,)
        assert "mu_tv_geo" not in shapes
        assert "sigma_tv_geo" not in shapes
        assert "mu_tv" not in shapes


class TestScanPanelByVar:
    """The scan path (temporal deps) supports the same grammar."""

    @pytest.fixture(scope="class")
    def template_and_model(self, geo_brand_data):
        pooling = {
            "intercept": True,
            "by_var": {
                # 'radio' is untransformed; 'tv' is a transform leaf and
                # cannot be coefficient-pooled (see TestValidationErrors).
                "radio": {"coefficient": ("geo",)},
                "theta_tv": "none",
            },
        }
        spec = (
            "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta_tv), "
            "lam=lam_tv) + radio"
        )
        template = pathmc.simulate_params_template(
            spec,
            data=geo_brand_data,
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling=pooling,
        )
        model = pathmc.model(
            spec,
            data=geo_brand_data.assign(
                sales=np.random.default_rng(3).normal(size=len(geo_brand_data))
            ),
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling=pooling,
        )
        return template, model

    def test_template_surfaces_new_rvs(self, template_and_model):
        template, _ = template_and_model
        assert template["mu_radio_geo"]["shape"] == (2,)
        assert template["beta_radio"]["shape"] == (4,)
        assert template["theta_tv"]["shape"] == (4,)
        assert "mu_radio" not in template

    def test_scan_model_rvs(self, template_and_model):
        _, model = template_and_model
        names = {rv.name for rv in model.pymc_model.free_RVs}
        assert {"beta_radio", "mu_radio_geo", "sigma_radio_geo", "theta_tv"} <= names


class TestValidationErrors:
    """Malformed by_var configs raise errors naming the valid options."""

    def _df(self, geo_brand_data):
        return geo_brand_data.assign(
            sales=np.random.default_rng(5).normal(size=len(geo_brand_data))
        )

    def test_unknown_dim_name(self, geo_brand_data):
        with pytest.raises(ValueError, match="channel"):
            pathmc.model(
                "sales ~ 0 + tv",
                data=self._df(geo_brand_data),
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": {"coefficient": ("channel",)}}},
            )

    def test_unknown_var_name(self, geo_brand_data):
        with pytest.raises(ValueError, match="'tv_spend'"):
            pathmc.model(
                "sales ~ 0 + tv",
                data=self._df(geo_brand_data),
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv_spend": {"coefficient": ("geo",)}}},
            )

    def test_bad_entry_grammar(self, geo_brand_data):
        with pytest.raises(ValueError, match='"none"'):
            pathmc.model(
                "sales ~ 0 + tv",
                data=self._df(geo_brand_data),
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": "partial"}},
            )

    def test_by_var_requires_panel(self, geo_brand_data):
        with pytest.raises(ValueError, match="panel"):
            pathmc.simulate(
                "Y ~ X",
                data=pd.DataFrame({"X": np.arange(6.0)}),
                params={"beta_Y": [1.0], "sigma_Y": 1.0},
                pooling={"by_var": {"X": {"coefficient": ("geo",)}}},
            )

    def test_multi_equation_predictor_rejected(self, geo_brand_data):
        df = self._df(geo_brand_data)
        df["M"] = 0.1 * df["tv"]
        with pytest.raises(ValueError, match="multiple equations"):
            pathmc.model(
                "M ~ 0 + tv\nsales ~ M + tv",
                data=df,
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": {"coefficient": ("geo",)}}},
            )

    def test_transform_leaf_coefficient_rejected(self, geo_brand_data):
        with pytest.raises(ValueError, match="transformed term"):
            pathmc.model(
                "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)",
                data=self._df(geo_brand_data),
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": {"coefficient": ("geo",)}}},
            )

    def test_transform_leaf_none_coefficient_rejected(self, geo_brand_data):
        with pytest.raises(ValueError, match="transformed term"):
            pathmc.model(
                "sales ~ 0 + adstock(tv, decay=theta_tv)",
                data=self._df(geo_brand_data),
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": "none"}},
            )

    def test_multi_equation_none_entry_rejected(self, geo_brand_data):
        df = self._df(geo_brand_data)
        df["M"] = 0.1 * df["tv"]
        with pytest.raises(ValueError, match="multiple equations"):
            pathmc.model(
                "M ~ 0 + tv\nsales ~ M + tv",
                data=df,
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": "none"}},
            )

    def test_dim_less_override_still_per_cell(self, geo_brand_data):
        """A dim-less user Prior override cannot collapse 'none' to a scalar."""
        from pymc_extras.prior import Prior

        model = pathmc.model(
            "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)",
            data=self._df(geo_brand_data),
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling={"by_var": {"theta_tv": "none"}},
            priors={"theta_tv": Prior("Beta", alpha=2, beta=2)},
        )
        shapes = {
            rv.name: tuple(int(d) for d in rv.shape.eval())
            for rv in model.pymc_model.free_RVs
        }
        assert shapes["theta_tv"] == (4,)


class TestTransformSensitivity:
    """The mu surface responds to per-cell transform parameters (no silent zeroing)."""

    def test_mu_responds_to_theta_and_lam(self, geo_brand_data):
        spec = (
            "sales ~ 0 + logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)"
        )
        panel = {"unit": ["geo", "brand"], "time": "week"}
        pooling = {"intercept": True, "by_var": {"theta_tv": "none"}}
        base = {
            "beta_sales": [2.0],
            "mu_alpha_sales": 50.0,
            "sigma_alpha_sales": 1e-8,
            "alpha_sales": [50.0, 50.0, 50.0, 50.0],
            "lam_tv": 0.5,
            "sigma_sales": 1e-8,
        }

        def run(theta):
            return pathmc.simulate(
                spec,
                data=geo_brand_data,
                params={**base, "theta_tv": theta},
                panel=panel,
                pooling=pooling,
                random_seed=7,
            )

        out_low = run([0.05, 0.05, 0.05, 0.05])
        out_high = run([0.9, 0.9, 0.9, 0.9])
        low = nw.from_native(out_low, eager_only=True)["sales"].to_numpy().sum()
        high = nw.from_native(out_high, eager_only=True)["sales"].to_numpy().sum()
        # Higher decay accumulates more adstock -> higher sales.
        assert high > low


class TestWarningConditionalityWithByVar:
    """The double-counting-intercept warning stays tied to random intercepts."""

    def setup_method(self):
        rng = np.random.default_rng(11)
        self.df = geo_frame(rng)

    def test_by_var_alone_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            pathmc.model(
                "sales ~ tv",
                data=self.df,
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={"by_var": {"tv": {"coefficient": ("geo",)}}},
            )

    def test_by_var_with_intercept_auto_drops(self):
        """Partial pooling auto-drops the redundant formula intercept (no
        warning since the intercept-wiring fix); by_var entries still
        compile alongside the dropped-intercept design."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = pathmc.model(
                "sales ~ tv",
                data=self.df,
                panel={"unit": ["geo", "brand"], "time": "week"},
                pooling={
                    "intercept": True,
                    "by_var": {"tv": {"coefficient": ("geo",)}},
                },
            )
        names = {rv.name for rv in model.pymc_model.free_RVs}
        # by_var machinery intact alongside the auto-dropped intercept
        assert {"beta_tv", "mu_tv_geo", "sigma_tv_geo"} <= names


def geo_frame(rng):
    rows = []
    for geo in ["North", "South"]:
        for brand in ["Acme", "Bolt"]:
            for week in range(8):
                rows.append({
                    "geo": geo,
                    "brand": brand,
                    "week": week,
                    "tv": rng.uniform(2.0, 20.0),
                    "sales": rng.normal(),
                })
    return pd.DataFrame(rows)


@pytest.mark.slow
class TestParameterRecovery:
    """Synthetic 2-geo x 2-brand MMM: recover geo-level coefficient means."""

    @pytest.fixture(scope="class")
    def recovered(self):
        rng = np.random.default_rng(42)
        true_alpha = {
            ("North", "Acme"): 40.0,
            ("North", "Bolt"): 44.0,
            ("South", "Acme"): 50.0,
            ("South", "Bolt"): 54.0,
        }
        true_beta = {
            ("North", "Acme"): 1.9,
            ("North", "Bolt"): 1.1,  # geo mean 1.5
            ("South", "Acme"): 3.4,
            ("South", "Bolt"): 2.6,  # geo mean 3.0
        }
        rows = []
        for geo in ["North", "South"]:
            for brand in ["Acme", "Bolt"]:
                for _week in range(25):
                    tv = rng.uniform(2.0, 10.0)
                    sales = (
                        true_alpha[(geo, brand)]
                        + true_beta[(geo, brand)] * tv
                        + rng.normal(scale=0.5)
                    )
                    rows.append({
                        "geo": geo,
                        "brand": brand,
                        "week": len(rows) % 25,
                        "tv": tv,
                        "sales": sales,
                    })
        df = pd.DataFrame(rows)

        model = pathmc.model(
            "sales ~ 0 + tv",
            data=df,
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling={
                "intercept": True,
                "by_var": {"tv": {"coefficient": ("geo",)}},
            },
        )
        idata = model.fit(draws=500, tune=500, chains=2, cores=1, random_seed=42)
        return idata, true_beta

    def test_geo_means_recovered(self, recovered):
        idata, _ = recovered
        post = idata.posterior["mu_tv_geo"].mean(dim=("chain", "draw")).values
        # Sorted geo levels are ['North', 'South'].
        assert abs(post[0] - 1.5) < 0.3
        assert abs(post[1] - 3.0) < 0.3
        assert post[1] > post[0]

    def test_cell_coefficients_shrink_toward_geo_means(self, recovered):
        idata, true_beta = recovered
        post = idata.posterior["beta_tv"].mean(dim=("chain", "draw")).values
        # Cell order matches sorted unit labels.
        truth = np.array([
            true_beta[("North", "Acme")],
            true_beta[("North", "Bolt")],
            true_beta[("South", "Acme")],
            true_beta[("South", "Bolt")],
        ])
        assert np.all(np.abs(post - truth) < 0.75)
