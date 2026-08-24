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
"""Tests for pathmc.simulate() — data generation from known parameters."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def exog_df():
    """Simple exogenous DataFrame with one predictor."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"X": rng.normal(size=200)})


@pytest.fixture
def exog_df_two():
    """Exogenous DataFrame with two predictors."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"X1": rng.normal(size=200), "X2": rng.normal(size=200)})


class TestSimulateBasic:
    """Core simulate() behaviour — shapes, columns, reproducibility."""

    def test_returns_dataframe_with_endogenous_columns(self, exog_df):
        df = pathmc.simulate(
            "Y ~ X",
            data=exog_df,
            params={"beta_Y": [2.0, 0.5], "sigma_Y": 1.0},
            random_seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "Y" in df.columns
        assert "X" in df.columns
        assert len(df) == len(exog_df)

    def test_exogenous_columns_unchanged(self, exog_df):
        df = pathmc.simulate(
            "Y ~ X",
            data=exog_df,
            params={"beta_Y": [0.0, 1.0], "sigma_Y": 0.5},
            random_seed=42,
        )
        pd.testing.assert_series_equal(df["X"], exog_df["X"])

    def test_reproducible_with_seed(self, exog_df):
        params = {"beta_Y": [1.0, 0.5], "sigma_Y": 1.0}
        df1 = pathmc.simulate("Y ~ X", data=exog_df, params=params, random_seed=7)
        df2 = pathmc.simulate("Y ~ X", data=exog_df, params=params, random_seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_give_different_data(self, exog_df):
        params = {"beta_Y": [1.0, 0.5], "sigma_Y": 1.0}
        df1 = pathmc.simulate("Y ~ X", data=exog_df, params=params, random_seed=1)
        df2 = pathmc.simulate("Y ~ X", data=exog_df, params=params, random_seed=2)
        assert not np.allclose(df1["Y"].values, df2["Y"].values)


class TestSimulateMultiEquation:
    """Multi-equation (mediation) models."""

    def test_mediation_columns(self, exog_df):
        spec = "M ~ X\nY ~ M + X"
        params = {
            "beta_M": [0.0, 0.8],
            "sigma_M": 0.5,
            "beta_Y": [1.0, 0.5, 0.3],
            "sigma_Y": 1.0,
        }
        df = pathmc.simulate(spec, data=exog_df, params=params, random_seed=42)
        assert "M" in df.columns
        assert "Y" in df.columns
        assert "X" in df.columns

    def test_mediation_chain_consistency(self, exog_df):
        """M depends on X, Y depends on M — values should be correlated."""
        spec = "M ~ X\nY ~ M"
        params = {
            "beta_M": [0.0, 5.0],
            "sigma_M": 0.01,
            "beta_Y": [0.0, 5.0],
            "sigma_Y": 0.01,
        }
        df = pathmc.simulate(spec, data=exog_df, params=params, random_seed=42)
        corr = df[["X", "M", "Y"]].corr()
        assert corr.loc["X", "M"] > 0.9
        assert corr.loc["M", "Y"] > 0.9
        assert corr.loc["X", "Y"] > 0.8


class TestSimulateFamilies:
    """Non-Gaussian families."""

    def test_bernoulli(self, exog_df):
        df = pathmc.simulate(
            "Y ~ X",
            data=exog_df,
            params={"beta_Y": [0.0, 1.0]},
            families={"Y": "bernoulli"},
            random_seed=42,
        )
        assert set(df["Y"].unique()).issubset({0, 1})

    def test_poisson(self, exog_df):
        df = pathmc.simulate(
            "Y ~ X",
            data=exog_df,
            params={"beta_Y": [1.0, 0.2]},
            families={"Y": "poisson"},
            random_seed=42,
        )
        assert (df["Y"] >= 0).all()
        assert df["Y"].dtype in (np.int64, np.int32, int)


class TestSimulateValidation:
    """Error handling and validation."""

    def test_missing_params_raises(self, exog_df):
        with pytest.raises(ValueError, match="Missing parameter values"):
            pathmc.simulate(
                "Y ~ X",
                data=exog_df,
                params={"beta_Y": [1.0, 0.5]},
            )

    def test_extra_params_warns(self, exog_df):
        with pytest.warns(UserWarning, match="Ignoring unknown parameter"):
            pathmc.simulate(
                "Y ~ X",
                data=exog_df,
                params={
                    "beta_Y": [1.0, 0.5],
                    "sigma_Y": 1.0,
                    "bogus_param": 99.0,
                },
                random_seed=42,
            )

    def test_residual_cov_round_trip(self, exog_df):
        """Correlated residuals: empirical corr of block members matches the
        target encoded in ``chol_Y1_Y2``, and both columns are simulated."""
        exog = exog_df.copy()
        exog["Y1"] = 0.0
        exog["Y2"] = 0.0
        params = {
            "beta_Y1": [0.0, 0.5],
            "beta_Y2": [0.0, -0.5],
            # L = [[1.0, 0.0], [0.8, 0.6]] -> corr 0.8, unit variances.
            "chol_Y1_Y2": [1.0, 0.8, 0.6],
        }
        df = pathmc.simulate(
            "Y1 ~ X\nY2 ~ X\nY1 ~~ Y2",
            data=exog,
            params=params,
            random_seed=42,
        )
        assert {"Y1", "Y2"} <= set(df.columns)
        # Residualize out the mean structure so the shared X term does not
        # contaminate the empirical residual correlation.
        x = df["X"].to_numpy()
        design = np.column_stack([np.ones_like(x), x])
        resid = np.empty_like(df[["Y1", "Y2"]].to_numpy())
        for j, col in enumerate(["Y1", "Y2"]):
            coef, *_ = np.linalg.lstsq(design, df[col].to_numpy(), rcond=None)
            resid[:, j] = df[col].to_numpy() - design @ coef
        emp_corr = np.corrcoef(resid[:, 0], resid[:, 1])[0, 1]
        assert emp_corr == pytest.approx(0.8, abs=0.05)

    def test_endogenous_in_data_ignored(self, exog_df):
        """If the user passes Y in data, it should be ignored."""
        exog_with_y = exog_df.copy()
        exog_with_y["Y"] = 999.0
        df = pathmc.simulate(
            "Y ~ X",
            data=exog_with_y,
            params={"beta_Y": [0.0, 1.0], "sigma_Y": 0.1},
            random_seed=42,
        )
        assert not np.allclose(df["Y"].values, 999.0)


class TestSimulateNoIntercept:
    """Models without intercept."""

    def test_no_intercept(self, exog_df):
        df = pathmc.simulate(
            "Y ~ 0 + X",
            data=exog_df,
            params={"beta_Y": [0.5], "sigma_Y": 1.0},
            random_seed=42,
        )
        assert "Y" in df.columns
        assert len(df) == len(exog_df)


@pytest.fixture
def panel_exog():
    """Panel exogenous data: 3 regions x 25 weeks of TV spend."""
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East"]
    rows = []
    for region in regions:
        for week in range(1, 26):
            rows.append({"region": region, "week": week, "tv": rng.uniform(5, 30)})
    return pd.DataFrame(rows)


class TestSimulatePanel:
    """simulate() with panel= — scan forward pass in time order per unit."""

    def test_lag_without_panel_raises(self):
        with pytest.raises(ValueError, match="panel"):
            pathmc.simulate(
                "Y ~ X + lag(Y)",
                data=pd.DataFrame({"X": [1.0, 2.0]}),
                params={"beta_Y": [0.0, 1.0], "sigma_Y": 1.0},
            )

    def test_lag_cold_start_exact_ar(self):
        """With sigma=0 the recursion is deterministic: Y_t = b0 + b1*Y_{t-1}.

        Cold start at zero means Y_0 = b0 exactly, and each unit's
        recursion is independent.
        """
        rng = np.random.default_rng(0)
        df = pd.DataFrame([
            {"unit": f"u{u}", "time": t, "X": rng.normal()}
            for u in range(3)
            for t in range(20)
        ])
        b0, b1 = 0.8, 0.5
        out = pathmc.simulate(
            "Y ~ X + lag(Y)",
            data=df,
            params={"beta_Y": [b0, 0.0, b1], "sigma_Y": 0.0},
            panel={"unit": "unit", "time": "time"},
            random_seed=1,
        )
        for _, sub in out.groupby("unit"):
            y = sub.sort_values("time")["Y"].to_numpy()
            expected = [b0]
            for _ in range(1, len(y)):
                expected.append(b0 + b1 * expected[-1])
            np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_existing_endogenous_column_poisoned_and_ignored(self):
        """Supplied endogenous values must not seed the temporal carry."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame([
            {"unit": f"u{u}", "time": t, "X": rng.normal(), "Y": 999.0}
            for u in range(3)
            for t in range(20)
        ])
        b0, b1 = 3.0, 0.5
        out = pathmc.simulate(
            "Y ~ X + lag(Y)",
            data=df,
            params={"beta_Y": [b0, 0.0, b1], "sigma_Y": 0.0},
            panel={"unit": "unit", "time": "time"},
            random_seed=1,
        )
        assert not (out["Y"] == 999.0).any()
        for _, sub in out.groupby("unit"):
            y = sub.sort_values("time")["Y"].to_numpy()
            expected = [b0]
            for _ in range(1, len(y)):
                expected.append(b0 + b1 * expected[-1])
            np.testing.assert_allclose(y, expected, atol=1e-10)

    def test_mmm_transform_moments_match_numpy_reference(self, panel_exog):
        """Adstock + saturation DGP moments reproduce a NumPy reference."""
        params = {
            "beta_sales": [2.5],  # intercept auto-dropped under partial pooling
            "theta_tv": 0.7,
            "lam_tv": 0.3,
            "alpha_sales": [55.0, 50.0, 60.0],  # sorted units: East,North,South
            "mu_alpha_sales": 0.0,
            "sigma_alpha_sales": 1.0,
            "sigma_sales": 0.5,
        }
        spec = (
            "sales ~ b_tv*logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)"
        )
        out = pathmc.simulate(
            spec,
            data=panel_exog,
            params=params,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
            random_seed=123,
        )
        assert list(out.columns) == ["region", "week", "tv", "sales"]

        # no fixed intercept under partial pooling (auto-dropped): the
        # per-unit alpha IS the intercept
        intercept_map = {"East": 55.0, "North": 50.0, "South": 60.0}
        for region in ["North", "South", "East"]:
            sub = panel_exog[panel_exog.region == region].sort_values("week")
            adstocked = 0.0
            saturated = []
            for tv in sub["tv"]:
                adstocked = tv + 0.7 * adstocked
                saturated.append(1 - np.exp(-0.3 * adstocked))
            ref_centered = (2.5 * np.array(saturated)).mean()
            sim = out[out.region == region].sort_values("week")
            sim_centered = (sim["sales"].to_numpy() - intercept_map[region]).mean()
            # sd=0.5 over 25 obs -> se ~= 0.1; allow 3.5 se.
            assert abs(sim_centered - ref_centered) < 0.35, region

        means = out.groupby("region")["sales"].mean()
        assert means["South"] > means["North"] > 50  # alphas differ as supplied

    def test_reproducible_with_seed(self, panel_exog):
        kwargs = dict(
            spec_string="sales ~ b_tv*logistic_saturation(adstock(tv, "
            "decay=theta_tv), lam=lam_tv)",
            data=panel_exog,
            params={
                "beta_sales": [2.5],
                "theta_tv": 0.7,
                "lam_tv": 0.3,
                "alpha_sales": [55.0, 50.0, 60.0],
                "mu_alpha_sales": 0.0,
                "sigma_alpha_sales": 1.0,
                "sigma_sales": 0.5,
            },
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        first = pathmc.simulate(**kwargs, random_seed=99)["sales"].to_numpy()
        second = pathmc.simulate(**kwargs, random_seed=99)["sales"].to_numpy()
        np.testing.assert_array_equal(first, second)

        gen_a = pathmc.simulate(**kwargs, random_seed=np.random.default_rng(7))[
            "sales"
        ].to_numpy()
        gen_b = pathmc.simulate(**kwargs, random_seed=np.random.default_rng(7))[
            "sales"
        ].to_numpy()
        np.testing.assert_array_equal(gen_a, gen_b)
        assert not np.allclose(first, gen_a)

    def test_row_order_preserved_for_unsorted_input(self):
        rng = np.random.default_rng(5)
        rows = [
            {"unit": f"u{u}", "time": t, "X": rng.normal()}
            for u in range(3)
            for t in range(12)
        ]
        interleaved = pd.DataFrame(rows).sample(frac=1.0, random_state=0)
        sorted_df = interleaved.sort_values(["unit", "time"]).reset_index(drop=True)
        params = {"beta_Y": [0.5, 0.2, 0.4], "sigma_Y": 0.8}
        out_i = pathmc.simulate(
            "Y ~ X + lag(Y)",
            data=interleaved,
            params=params,
            panel={"unit": "unit", "time": "time"},
            random_seed=11,
        )
        out_s = pathmc.simulate(
            "Y ~ X + lag(Y)",
            data=sorted_df,
            params=params,
            panel={"unit": "unit", "time": "time"},
            random_seed=11,
        )
        merged = interleaved.reset_index(drop=True)[["unit", "time"]].copy()
        merged["Y_interleaved"] = out_i["Y"].to_numpy()
        merged = merged.merge(
            sorted_df[["unit", "time"]].assign(Y_sorted=out_s["Y"].to_numpy()),
            on=["unit", "time"],
        )
        np.testing.assert_allclose(merged["Y_interleaved"], merged["Y_sorted"])

    def test_stochastic_latent_in_scan_model(self, panel_exog):
        out = pathmc.simulate(
            "M ~ adstock(tv, decay=theta)\nsales ~ M",
            data=panel_exog,
            params={
                "beta_M": [1.0, 0.8],
                "sigma_M": 0.4,
                "init_M": 0.0,
                "theta": 0.6,
                "beta_sales": [0.5, 1.2],
                "sigma_sales": 0.3,
            },
            latent=["M"],
            families={"M": "latent_normal"},
            panel={"unit": "region", "time": "week"},
            random_seed=4,
        )
        assert list(out.columns) == ["region", "week", "tv", "M", "sales"]

    def test_param_shape_mismatch_raises(self, panel_exog):
        with pytest.raises(ValueError, match="beta_sales"):
            pathmc.simulate(
                "sales ~ b_tv*logistic_saturation(adstock(tv, decay=theta_tv), "
                "lam=lam_tv)",
                data=panel_exog,
                params={
                    "beta_sales": [2.5, 1.0],  # model requires just (b_tv,)
                    "theta_tv": 0.7,
                    "lam_tv": 0.3,
                    "alpha_sales": [55.0, 50.0, 60.0],
                    "mu_alpha_sales": 0.0,
                    "sigma_alpha_sales": 1.0,
                    "sigma_sales": 0.5,
                },
                panel={"unit": "region", "time": "week"},
                pooling="partial",
            )


class TestSimulatePanelTransforms:
    """Transform chains (adstock + logistic_saturation) through simulate() on panels.

    The NumPy reference DGP uses pathmc's exact saturation kernel
    ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))`` (== ``tanh(lam*x/2)``).
    Specs are intercept-free under ``pooling="partial"`` so
    ``mu_alpha_<lhs>`` serves as the effective intercept (avoids the
    known partial-pooling non-identifiability warning).
    """

    SPEC = (
        "sales ~ 0 + b_tv*logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)"
    )

    @staticmethod
    def _reference(tv, theta, lam):
        """NumPy reference: geometric adstock recursion + pathmc saturation."""
        adstocked = 0.0
        saturated = []
        for x in tv:
            adstocked = x + theta * adstocked
            saturated.append(
                (1 - np.exp(-lam * adstocked)) / (1 + np.exp(-lam * adstocked))
            )
        return np.asarray(saturated)

    def _panel_params(self, alphas, sigma=0.0):
        return {
            "beta_sales": [2.5],
            "theta_tv": 0.7,
            "lam_tv": 0.3,
            "alpha_sales": list(alphas),
            "mu_alpha_sales": 55.0,
            "sigma_alpha_sales": 2.0,
            "sigma_sales": sigma,
        }

    def _simulate(self, df, params, seed):
        return pathmc.simulate(
            self.SPEC,
            data=df,
            params=params,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
            random_seed=seed,
        )

    def test_zero_noise_series_matches_numpy_reference(self, panel_exog):
        """With sigma=0 the scan recursion must reproduce the reference loop."""
        intercepts = {"North": 50.0, "South": 60.0, "East": 55.0}
        # sorted unit labels: East, North, South
        params = self._panel_params([55.0, 50.0, 60.0])
        out = self._simulate(panel_exog, params, seed=5)
        assert list(out.columns) == ["region", "week", "tv", "sales"]
        for region, intercept in intercepts.items():
            tv = (
                panel_exog[panel_exog.region == region]
                .sort_values("week")["tv"]
                .to_numpy()
            )
            expected = intercept + 2.5 * self._reference(tv, theta=0.7, lam=0.3)
            actual = out[out.region == region].sort_values("week")["sales"].to_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_nested_chain_params_honored(self, panel_exog):
        """Varying theta_tv / lam_tv changes output and matches its own reference."""
        tv = (
            panel_exog[panel_exog.region == "North"]
            .sort_values("week")["tv"]
            .to_numpy()
        )
        base = {"theta_tv": 0.7, "lam_tv": 0.3}

        def run(theta, lam):
            params = {**self._panel_params([55.0, 50.0, 60.0]), **base}
            params["theta_tv"], params["lam_tv"] = theta, lam
            return self._simulate(panel_exog, params, seed=5)

        out_base = run(0.7, 0.3)
        out_theta = run(0.4, 0.3)
        out_lam = run(0.7, 0.6)

        north_base = (
            out_base[out_base.region == "North"].sort_values("week")["sales"]
        ).to_numpy()
        north_theta = (
            out_theta[out_theta.region == "North"].sort_values("week")["sales"]
        ).to_numpy()
        north_lam = (
            out_lam[out_lam.region == "North"].sort_values("week")["sales"]
        ).to_numpy()
        assert not np.allclose(north_theta, north_base)
        assert not np.allclose(north_lam, north_base)

        expected_theta = 50.0 + 2.5 * self._reference(tv, theta=0.4, lam=0.3)
        np.testing.assert_allclose(north_theta, expected_theta, atol=1e-8)
        expected_lam = 50.0 + 2.5 * self._reference(tv, theta=0.7, lam=0.6)
        np.testing.assert_allclose(north_lam, expected_lam, atol=1e-8)

    def test_noisy_moments_within_sampling_error(self, panel_exog):
        """With sigma>0, per-unit demeaned sales track the reference mean."""
        params = self._panel_params([55.0, 50.0, 60.0], sigma=0.5)
        out = self._simulate(panel_exog, params, seed=17)
        intercepts = {"East": 55.0, "North": 50.0, "South": 60.0}
        for region, intercept in intercepts.items():
            tv = (
                panel_exog[panel_exog.region == region]
                .sort_values("week")["tv"]
                .to_numpy()
            )
            ref_mean = (2.5 * self._reference(tv, 0.7, 0.3)).mean()
            sim = out[out.region == region].sort_values("week")
            centered = (sim["sales"].to_numpy() - intercept).mean()
            # se ~= 0.5/sqrt(25) = 0.1; allow 4 se
            assert abs(centered - ref_mean) < 0.4, region

    def test_multi_dim_panel_geo_brand(self):
        """Adstock recurses within each geo|brand composite unit (#430)."""
        rng = np.random.default_rng(8)
        rows = [
            {"geo": g, "brand": b, "week": w, "tv": rng.uniform(5, 30)}
            for g in ["North", "South"]
            for b in ["Acme", "Zen"]
            for w in range(1, 21)
        ]
        df = pd.DataFrame(rows)
        params = self._panel_params([51.0, 52.0, 53.0, 54.0])
        out = pathmc.simulate(
            self.SPEC,
            data=df,
            params=params,
            panel={"unit": ["geo", "brand"], "time": "week"},
            pooling="partial",
            random_seed=9,
        )
        # composite unit key column added by build_panel_info
        assert list(out.columns) == [
            "geo",
            "brand",
            "week",
            "tv",
            "geo|brand",
            "sales",
        ]
        intercepts = {
            ("North", "Acme"): 51.0,
            ("North", "Zen"): 52.0,
            ("South", "Acme"): 53.0,
            ("South", "Zen"): 54.0,
        }
        for (geo, brand), intercept in intercepts.items():
            tv = (
                df[(df.geo == geo) & (df.brand == brand)]
                .sort_values("week")["tv"]
                .to_numpy()
            )
            expected = intercept + 2.5 * self._reference(tv, theta=0.7, lam=0.3)
            mask = (out.geo == geo) & (out.brand == brand)
            actual = out[mask].sort_values("week")["sales"].to_numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-8)

    @pytest.mark.slow
    def test_recover_transform_params(self):
        """Fit the same transform spec; posterior means land near truth.

        Pooled (no random intercepts) so the sampler only has to explore
        ``beta_sales``, ``theta_tv``, ``lam_tv``, ``sigma_sales``. The
        conftest autouse fixture forces every ``pm.sample()`` call down
        to 50 draws / 50 tune / 1 chain, which is too short for HDI-level
        precision on a scan model — so recovery is asserted on posterior
        means with tolerances ~5x the observed seed-to-seed spread, and
        the DGP uses low noise (sigma=0.25) to keep the likelihood sharp.
        """
        import arviz as az

        rng = np.random.default_rng(11)
        n_units, n_times = 8, 60
        exog = pd.DataFrame([
            {"unit": f"u{u}", "time": t, "tv": rng.uniform(1, 6)}
            # spend in the responsive saturation regime: adstocked tv
            # stays where tanh(lam*x/2) still varies with the params
            for u in range(n_units)
            for t in range(n_times)
        ])
        params = {
            "beta_sales": [2.5],
            "theta_tv": 0.7,
            "lam_tv": 0.3,
            "sigma_sales": 0.25,
        }
        sim = pathmc.simulate(
            self.SPEC,
            data=exog,
            params=params,
            panel={"unit": "unit", "time": "time"},
            random_seed=2024,
        )
        model = pathmc.model(
            self.SPEC,
            data=sim,
            panel={"unit": "unit", "time": "time"},
        )
        idata = model.fit(random_seed=42)
        summary = az.summary(
            idata, var_names=["beta_sales", "theta_tv", "lam_tv"], round_to="none"
        )
        means = {"b_tv": summary.loc["beta_sales[tv]", "mean"]}
        means.update({v: summary.loc[v, "mean"] for v in ["theta_tv", "lam_tv"]})
        truth = {"b_tv": 2.5, "theta_tv": 0.7, "lam_tv": 0.3}
        for name, value in truth.items():
            assert abs(means[name] - value) < 0.2, (
                f"{name}: mean {means[name]:.3f} vs truth {value}"
            )


@pytest.mark.slow
class TestSimulatePanelRecovery:
    """Simulate-and-recover through the full panel pipeline."""

    def test_lag_ar_coefficient_recovered(self):
        rng = np.random.default_rng(3)
        n_units, n_times = 8, 40
        exog = pd.DataFrame([
            {"unit": f"u{u}", "time": t} for u in range(n_units) for t in range(n_times)
        ])
        truth = {"intercept": 0.5, "phi": 0.5, "sigma": 0.4}
        sim = pathmc.simulate(
            "Y ~ lag(Y)",
            data=exog,
            params={
                "beta_Y": [truth["intercept"], truth["phi"]],
                "sigma_Y": truth["sigma"],
            },
            panel={"unit": "unit", "time": "time"},
            random_seed=2024,
        )
        assert not sim["Y"].isna().any()

        import arviz as az

        model = pathmc.model(
            "Y ~ lag(Y)",
            data=sim,
            panel={"unit": "unit", "time": "time"},
        )
        idata = model.fit(draws=400, tune=500, chains=2, cores=1, random_seed=42)
        summary = az.summary(idata, var_names=["beta_Y"], round_to="none")
        phi_mean = summary["mean"].iloc[1]
        hdi = az.hdi(idata.posterior["beta_Y"].isel({"Y_predictors": 1}))
        assert abs(phi_mean - truth["phi"]) < 0.15, phi_mean
        assert (
            float(hdi.sel(ci_bound="lower"))
            < truth["phi"]
            < float(hdi.sel(ci_bound="upper"))
        )


class TestSimulateFunnelDGP:
    """Upper-funnel funnel DGP (#433): x2/x3 <- x1, x4 <- x2 + x3,

    and y through adstock + logistic saturation of the mid-funnel x4.
    Runs on a panel so the adstock recursion carries state within units.
    """

    SPEC = """
    x2 ~ x1
    x3 ~ x1
    x4 ~ x2 + x3
    y ~ 0 + b*logistic_saturation(adstock(x4, decay=theta), lam=lam) + ctrl
    """

    PARAMS = {
        "beta_x2": [0.0, 0.8],
        "sigma_x2": 0.5,
        "beta_x3": [0.0, 0.6],
        "sigma_x3": 0.5,
        "beta_x4": [0.0, 1.0, 0.9],
        "sigma_x4": 0.5,
        "theta": 0.7,
        "lam": 0.5,
        "beta_y": [1.2, 0.3],
        "sigma_y": 0.0,
    }

    @staticmethod
    def _exog(n_units=2, n_times=40, seed=42):
        rng = np.random.default_rng(seed)
        return pd.DataFrame([
            {"unit": f"u{u}", "time": t, "x1": rng.normal(), "ctrl": rng.normal()}
            for u in range(n_units)
            for t in range(1, n_times + 1)
        ])

    def _simulate(self, exog, params=None, seed=7):
        return pathmc.simulate(
            self.SPEC,
            data=exog,
            params=params or self.PARAMS,
            panel={"unit": "unit", "time": "time"},
            random_seed=seed,
        )

    @staticmethod
    def _reference(x4, theta, lam):
        """NumPy adstock recursion + pathmc saturation kernel (tanh form)."""
        adstocked = 0.0
        out = []
        for v in x4:
            adstocked = v + theta * adstocked
            out.append((1 - np.exp(-lam * adstocked)) / (1 + np.exp(-lam * adstocked)))
        return np.asarray(out)

    def test_funnel_propagates_through_chain(self):
        out = self._simulate(self._exog())
        assert list(out.columns) == [
            "unit",
            "time",
            "x1",
            "ctrl",
            "x2",
            "x3",
            "x4",
            "y",
        ]
        corr = out[["x1", "x2", "x3", "x4"]].corr()
        assert corr.loc["x1", "x2"] > 0.5
        assert corr.loc["x1", "x3"] > 0.5
        assert corr.loc["x2", "x4"] > 0.3
        assert corr.loc["x3", "x4"] > 0.3

    def test_outcome_matches_numpy_reference_with_zero_noise(self):
        """With sigma_y=0, y must equal b*saturation(adstock(x4)) + c*ctrl.

        Descendant equations consume the *mean structure* of upstream
        endogenous variables (mirroring estimation), so the reference
        propagates x1 through exact coefficient chains instead of the
        noisy realized x2/x3/x4 columns.
        """
        exog = self._exog()
        out = self._simulate(exog)
        for unit in sorted(exog["unit"].unique()):
            rows = (
                out[out.unit == unit]
                .sort_values("time")[["x1", "ctrl", "y"]]
                .reset_index(drop=True)
            )
            x1 = rows["x1"].to_numpy()
            x4_mu = (
                self.PARAMS["beta_x4"][1] * self.PARAMS["beta_x2"][1] * x1
                + self.PARAMS["beta_x4"][2] * self.PARAMS["beta_x3"][1] * x1
            )
            expected = (
                self.PARAMS["beta_y"][0]
                * self._reference(
                    x4_mu,
                    theta=self.PARAMS["theta"],
                    lam=self.PARAMS["lam"],
                )
                + self.PARAMS["beta_y"][1] * rows["ctrl"].to_numpy()
            )
            np.testing.assert_allclose(rows["y"].to_numpy(), expected, atol=1e-8)
