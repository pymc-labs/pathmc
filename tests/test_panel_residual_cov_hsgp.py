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
"""Phase 3 of #434: ``~~`` and ``hsgp()`` inside panel models.

Covers the two combinations that used to be rejected (or, for
scan-compiled ``~~``, silently mis-compiled): a residual-covariance block
in a scan-compiled panel model, and an HSGP smooth in a panel model with
or without temporal dependence.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import pathmc

PANEL = {"unit": "geo", "time": "week"}


def _panel_frame(n_units: int = 4, n_times: int = 15, seed: int = 0) -> pd.DataFrame:
    """Rectangular panel with a distinct exogenous value per cell."""
    rng = np.random.default_rng(seed)
    n = n_units * n_times
    df = pd.DataFrame({
        "geo": np.repeat([f"g{u}" for u in range(n_units)], n_times),
        "week": np.tile(np.arange(n_times), n_units),
        "x": rng.normal(size=n),
    })
    df["M1"] = 0.5 * df["x"] + rng.normal(0, 0.4, n)
    df["M2"] = -0.3 * df["x"] + rng.normal(0, 0.4, n)
    df["y"] = df["M1"] + df["M2"] + rng.normal(0, 0.3, n)
    return df


# --- scan-compiled panel + residual covariance ------------------------------


class TestScanPanelResidualCov:
    """``~~`` in a panel model whose temporal dependence forces a scan."""

    SPEC = "M1 ~ x\nM2 ~ x\ny ~ M1 + M2 + lag(y)\nM1 ~~ M2"

    def test_block_members_are_observed(self):
        """Regression: ``~~`` used to silently drop M1/M2 from the likelihood.

        Before Phase 3 the scan compiler ignored residual blocks while
        ``PathModel._compile`` still skipped block members when attaching
        observations, so M1 and M2 became *unobserved free RVs* and their
        data never entered the logp -- a silently wrong model, not an error.
        """
        model = pathmc.model(self.SPEC, _panel_frame(), panel=PANEL)
        pm_model = model._pymc_model
        free = {rv.name for rv in pm_model.free_RVs}
        observed = {rv.name for rv in pm_model.observed_RVs}

        assert "M1" not in free and "M2" not in free
        assert "M1_M2_obs" in observed
        assert "chol_M1_M2" in free

    def test_no_unused_per_member_sigma(self):
        """Block members take their scale from the block covariance.

        A leftover ``sigma_{member}`` would be a free RV with no likelihood
        attached, sampled purely from its prior. The cross-sectional path
        omits it; the scan path must match.
        """
        model = pathmc.model(self.SPEC, _panel_frame(), panel=PANEL)
        free = {rv.name for rv in model._pymc_model.free_RVs}
        assert "sigma_M1" not in free
        assert "sigma_M2" not in free
        assert "sigma_y" in free

    def test_block_logp_matches_numpy_oracle(self):
        """The joint term pairs each cell's mu with that same cell's data.

        Oracle is built in the *original* dataframe row order and summed,
        so it is invariant to the compiler's internal (unit, time) sort.
        A mis-ravelled mu would pair cell i's mean with cell j's outcome
        and shift the total.
        """
        df = _panel_frame()
        model = pathmc.model(self.SPEC, df, panel=PANEL)
        pm_model = model._pymc_model
        point = pm_model.initial_point()

        stds, corr, beta_m1, beta_m2 = pm_model.compile_fn(
            [
                pm_model["chol_M1_M2_stds"],
                pm_model["chol_M1_M2_corr"],
                pm_model["beta_M1"],
                pm_model["beta_M2"],
            ]
        )(point)
        cov = np.diag(stds) @ np.asarray(corr) @ np.diag(stds)

        design = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
        mu = np.column_stack([design @ beta_m1, design @ beta_m2])
        outcomes = df[["M1", "M2"]].to_numpy()
        expected = stats.multivariate_normal(cov=cov).logpdf(outcomes - mu).sum()

        actual = pm_model.compile_logp(vars=[pm_model["M1_M2_obs"]])(point)
        assert float(actual) == pytest.approx(float(expected), rel=1e-8)

    @pytest.mark.parametrize(
        "spec, match",
        [
            ("M1 ~ x + lag(M1)\nM2 ~ x\ny ~ M1\nM1 ~~ M2", "feeds a lag"),
            ("M1 ~ x\nM2 ~ x\ny ~ M1 + M2 + lag(y)\nM1 ~~ M2", "has missing values"),
        ],
        ids=["lag-member", "missing-member"],
    )
    def test_unsupported_block_members_raise(self, spec, match):
        df = _panel_frame()
        if match == "has missing values":
            df.loc[3, "M1"] = np.nan
        with pytest.raises(NotImplementedError, match=match):
            pathmc.model(spec, df, panel=PANEL)

    def test_simulate_recovers_residual_correlation(self):
        """Simulated block residuals carry the correlation encoded in chol.

        Residuals are recovered by OLS *outside* pathmc, so a broken
        generative path cannot hide behind the code that produced it.
        """
        exog = _panel_frame(n_units=25, n_times=200, seed=7)[["geo", "week", "x"]]
        df = pathmc.simulate(
            "M1 ~ x\nM2 ~ x\ny ~ M1 + M2 + lag(y)\nM1 ~~ M2",
            data=exog,
            params={
                "beta_M1": [0.0, 0.5],
                "beta_M2": [0.0, -0.5],
                "beta_y": [0.0, 1.0, 1.0, 0.2],
                "sigma_y": 0.1,
                "chol_M1_M2": [1.0, 0.8, 0.6],
            },
            panel=PANEL,
            random_seed=42,
        )

        design = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
        resid = np.column_stack([
            df[col].to_numpy()
            - design @ np.linalg.lstsq(design, df[col].to_numpy(), rcond=None)[0]
            for col in ("M1", "M2")
        ])
        # chol [1, 0.8, 0.6] -> corr = 0.8 / sqrt(0.8**2 + 0.6**2) = 0.8
        assert np.corrcoef(resid[:, 0], resid[:, 1])[0, 1] == pytest.approx(
            0.8, abs=0.03
        )

    def test_simulate_descendant_reads_realized_draw(self):
        """A descendant of a block member sees the noisy draw, not ``mu``.

        Mirrors the cross-sectional contract (#469): regressing the
        descendant on the realized member must recover the structural
        coefficient. Wiring ``mu_M1`` in instead attenuates the slope.
        """
        exog = _panel_frame(n_units=20, n_times=150, seed=3)[["geo", "week", "x"]]
        df = pathmc.simulate(
            "M1 ~ x\nM2 ~ x\nz ~ M1\ny ~ z + lag(y)\nM1 ~~ M2",
            data=exog,
            params={
                "beta_M1": [0.0, 0.5],
                "beta_M2": [0.0, -0.5],
                "beta_z": [0.0, 3.0],
                "sigma_z": 0.05,
                "beta_y": [0.0, 1.0, 0.2],
                "sigma_y": 0.1,
                "chol_M1_M2": [1.0, 0.8, 0.6],
            },
            panel=PANEL,
            random_seed=11,
        )
        design = np.column_stack([np.ones(len(df)), df["M1"].to_numpy()])
        slope = np.linalg.lstsq(design, df["z"].to_numpy(), rcond=None)[0][1]
        assert slope == pytest.approx(3.0, abs=0.05)


# --- panel + HSGP -----------------------------------------------------------


class TestPanelHSGP:
    """``hsgp()`` smooths in panel models, with and without a scan."""

    def test_flat_panel_matches_cross_sectional(self):
        """Without temporal deps the panel smooth is the cross-sectional one.

        The basis is pointwise in its input, so adding ``panel=`` must not
        change the linear predictor for the same rows and parameters.
        """
        df = _panel_frame()
        spec = "y ~ x + hsgp(week, m=6, c=1.5)"

        panel_model = pathmc.model(spec, df, panel=PANEL)._pymc_model
        xsec_model = pathmc.model(spec, df)._pymc_model

        point = xsec_model.initial_point()
        mu_xsec = xsec_model.compile_fn([xsec_model["mu_y"]])(point)
        mu_panel = panel_model.compile_fn([panel_model["mu_y"]])(point)

        np.testing.assert_allclose(
            np.asarray(mu_panel).ravel(), np.asarray(mu_xsec).ravel(), rtol=1e-10
        )

    def test_scan_panel_mu_matches_numpy_oracle(self):
        """The pre-built smooth is aligned to the right (time, unit) cells.

        The smooth is assembled once outside the scan and fed in as a
        sequence; if the reshape were transposed, ``mu_y`` would pick up
        another cell's smooth value. Checked at t=0, where the lag carry is
        seeded from data and the oracle needs no recursion.
        """
        df = _panel_frame()
        model = pathmc.model(
            "y ~ x + lag(y) + hsgp(week, m=6, c=1.5)", df, panel=PANEL
        )
        pm_model = model._pymc_model
        point = pm_model.initial_point()

        mu_y, f_smooth, beta_y = pm_model.compile_fn(
            [pm_model["mu_y"], pm_model["f_y_week"], pm_model["beta_y"]]
        )(point)
        mu_y = np.asarray(mu_y)
        f_smooth = np.asarray(f_smooth)

        scan_info = pm_model._pathmc_panel_scan
        assert f_smooth.shape == (scan_info.n_times, scan_info.n_units)

        # Sorted (unit, time) view of the inputs; row 0 is t=0 across units.
        x_panel = (
            df["x"].to_numpy()[scan_info.sort_idx].reshape(scan_info.n_units, -1).T
        )
        y_panel = (
            df["y"].to_numpy()[scan_info.sort_idx].reshape(scan_info.n_units, -1).T
        )
        intercept, beta_x, beta_lag = beta_y[0], beta_y[1], beta_y[2]
        expected_t0 = (
            intercept + beta_x * x_panel[0] + beta_lag * y_panel[0] + f_smooth[0]
        )
        np.testing.assert_allclose(mu_y[0], expected_t0, rtol=1e-10)

    def test_scan_panel_smooth_varies_over_time_not_units(self):
        """A smooth over the time column is constant within a timestep.

        Cheap transpose detector that does not depend on any parameter
        values: ``week`` is identical across units at a given ``t``, so the
        correctly-oriented smooth has zero variance along the unit axis.
        """
        model = pathmc.model(
            "y ~ x + lag(y) + hsgp(week, m=6, c=1.5)", _panel_frame(), panel=PANEL
        )
        pm_model = model._pymc_model
        f_smooth = np.asarray(
            pm_model.compile_fn([pm_model["f_y_week"]])(pm_model.initial_point())
        )
        np.testing.assert_allclose(f_smooth.std(axis=1), 0.0, atol=1e-12)
        assert f_smooth.std(axis=0).max() > 0.0

    def test_simulate_scan_panel_hsgp_reproduces_smooth(self):
        """Zero-noise simulation returns the smooth the model reports."""
        df = _panel_frame(n_units=3, n_times=20, seed=5)[["geo", "week", "x"]]
        rng = np.random.default_rng(0)
        params = {
            "beta_y": [0.0, 0.0, 0.0],
            "sigma_y": 0.0,
            "ell_y_week": 2.0,
            "eta_y_week": 1.0,
            "beta_hsgp_y_week": rng.normal(size=6),
        }
        out = pathmc.simulate(
            "y ~ x + lag(y) + hsgp(week, m=6, c=1.5)",
            data=df,
            params=params,
            panel=PANEL,
            random_seed=0,
        )
        # With every coefficient zeroed and no noise, y is exactly the smooth,
        # which depends only on week -- identical across geos.
        by_week = out.pivot_table(index="week", columns="geo", values="y")
        np.testing.assert_allclose(
            by_week.to_numpy().std(axis=1), 0.0, atol=1e-10
        )
        assert by_week.to_numpy().std(axis=0).max() > 0.0

    def test_hsgp_in_residual_block_still_rejected(self):
        """Phase 3 does not lift the HSGP-on-a-block-member restriction."""
        with pytest.raises(NotImplementedError, match="residual-covariance block"):
            pathmc.model(
                "M1 ~ x + hsgp(week, m=6, c=1.5)\nM2 ~ x\ny ~ M1 + M2\nM1 ~~ M2",
                _panel_frame(),
                panel=PANEL,
            )
