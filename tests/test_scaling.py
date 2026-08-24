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
"""Tests for pathmc.scaling — heterogeneous unit scaling (issue #432)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc.scaling import Scaling, ScalingFactors, fit_scaling

SPEC = "Y ~ X"

POPULATIONS = {"big": 100.0, "small": 10.0}
TARGET_SCALES = {"big": 500.0, "small": 50.0}

PANEL = {"unit": "geo", "time": "week"}
MULTI_PANEL = {"unit": ["geo", "brand"], "time": "week"}


def _nw(df: pd.DataFrame):
    import narwhals.stable.v1 as nw_

    return nw_.from_native(df, eager_only=True)


def _panel_info(panel: dict):
    """Build PanelInfo through the public entry point."""
    from pathmc.panel import build_panel_info

    if isinstance(panel["unit"], str):
        base = make_panel()
    else:
        base = make_multi_dim_panel()
    info, _ = build_panel_info(_nw(base), panel)
    return info


def make_panel() -> pd.DataFrame:
    """Heterogeneous panel: columns differ ~10x across geos."""
    rng = np.random.default_rng(7)
    frames = []
    for geo, pop in POPULATIONS.items():
        n_weeks = 12
        frames.append(
            pd.DataFrame({
                "geo": geo,
                "week": np.arange(n_weeks),
                "X": rng.normal(size=n_weeks) * pop,
                "Y": rng.normal(size=n_weeks) * pop * 0.1 + 2.0 * pop,
            })
        )
    return pd.concat(frames, ignore_index=True)


def make_multi_dim_panel() -> pd.DataFrame:
    """Cartesian geo × brand panel for composite-key scaling."""
    rng = np.random.default_rng(11)
    frames = []
    for geo, pop in POPULATIONS.items():
        for brand in ("acme", "bolt"):
            frames.append(
                pd.DataFrame({
                    "geo": geo,
                    "brand": brand,
                    "week": np.arange(6),
                    "spend": rng.uniform(size=6) * pop,
                })
            )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Factor math
# ---------------------------------------------------------------------------


class TestFactorMath:
    def test_max_global(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "max"}),
            _nw(raw),
            target_columns={"Y"},
        )
        np.testing.assert_allclose(factors.factors["Y"], raw["Y"].max())

    def test_mean_per_group(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "mean", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            target_columns={"Y"},
        )
        expected = raw.groupby("geo")["Y"].transform("mean").to_numpy()
        np.testing.assert_allclose(factors.factors["Y"], expected)

    def test_max_per_group_matches_manual_prescale(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "max", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            target_columns={"Y"},
        )
        scaled = factors.transform(_nw(raw))
        group_max = raw.groupby("geo")["Y"].transform("max").to_numpy()
        np.testing.assert_allclose(
            scaled["Y"].to_numpy(), raw["Y"].to_numpy() / group_max
        )

    def test_fixed_constant(self):
        factors = fit_scaling(
            Scaling(channel={"method": "fixed", "value": 25.0}),
            _nw(make_panel()),
            channel_columns={"X"},
        )
        np.testing.assert_allclose(factors.factors["X"], 25.0)

    def test_fixed_grid_dict(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(
                target={
                    "method": "fixed",
                    "value": TARGET_SCALES,
                    "dims": ("geo",),
                }
            ),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            target_columns={"Y"},
        )
        np.testing.assert_allclose(
            factors.factors["Y"], raw["geo"].map(TARGET_SCALES).to_numpy()
        )

    def test_divide_by_xarray_coords_align(self):
        """DataArray coords align with the panel unit dims they name."""
        pop = xr.DataArray(
            [100.0, 10.0],
            coords={"geo": ["big", "small"]},
            dims=["geo"],
        )
        raw = make_panel()
        factors = fit_scaling(
            Scaling(channel={"method": "divide", "by": pop, "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            channel_columns={"X"},
        )
        np.testing.assert_allclose(
            factors.factors["X"], raw["geo"].map(POPULATIONS).to_numpy()
        )

    def test_divide_multi_dim_composite_alignment(self):
        """Multi-D panels: coords align per dim regardless of declared order."""
        raw = make_multi_dim_panel()
        pops = xr.DataArray(
            [[100.0, 10.0], [100.0, 10.0]],
            coords={"brand": ["acme", "bolt"], "geo": ["big", "small"]},
            dims=["brand", "geo"],
        )
        factors = fit_scaling(
            Scaling(
                channel={
                    "method": "divide",
                    "by": pops,
                    "dims": ("geo", "brand"),
                }
            ),
            _nw(raw),
            panel_info=_panel_info(MULTI_PANEL),
            channel_columns={"spend"},
        )
        np.testing.assert_allclose(
            factors.factors["spend"], raw["geo"].map(POPULATIONS).to_numpy()
        )

    def test_divide_dict_tuple_and_joined_keys(self):
        raw = make_multi_dim_panel()
        info = _panel_info(MULTI_PANEL)
        grid_tuple = {
            ("big", "acme"): 8.0,
            ("big", "bolt"): 2.0,
            ("small", "acme"): 16.0,
            ("small", "bolt"): 4.0,
        }
        grid_joined = {
            "big|acme": 8.0,
            "big|bolt": 2.0,
            "small|acme": 16.0,
            "small|bolt": 4.0,
        }
        f_t = fit_scaling(
            Scaling(
                channel={
                    "method": "divide",
                    "by": grid_tuple,
                    "dims": ("geo", "brand"),
                }
            ),
            _nw(raw),
            panel_info=info,
            channel_columns={"spend"},
        )
        f_j = fit_scaling(
            Scaling(
                channel={
                    "method": "divide",
                    "by": grid_joined,
                    "dims": ("geo", "brand"),
                }
            ),
            _nw(raw),
            panel_info=info,
            channel_columns={"spend"},
        )
        lookup = {
            ("big", "acme"): 8.0,
            ("small", "bolt"): 4.0,
            ("big", "bolt"): 2.0,
            ("small", "acme"): 16.0,
        }
        expected = [lookup[k] for k in zip(raw["geo"], raw["brand"])]
        np.testing.assert_allclose(f_t.factors["spend"], expected)
        np.testing.assert_allclose(f_j.factors["spend"], expected)

    def test_transform_inverse_round_trip(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "max"}),
            _nw(raw),
            target_columns={"Y"},
        )
        scaled = factors.transform(_nw(raw))
        recovered = factors.inverse_transform_column(scaled["Y"].to_numpy(), "Y")
        np.testing.assert_allclose(recovered, raw["Y"].to_numpy())


# ---------------------------------------------------------------------------
# Wiring into model(): estimation happens in scaled units
# ---------------------------------------------------------------------------


class TestModelWiring:
    def test_fitted_scaling_stored_and_data_scaled(self):
        raw = make_panel()
        m = pathmc.model(SPEC, data=raw, scaling=Scaling(target={"method": "max"}))
        assert m._data is not None
        assert m.fitted_scaling is not None
        np.testing.assert_allclose(
            m._data["Y"].to_numpy(),
            raw["Y"].to_numpy() / raw["Y"].max(),
        )
        # A target-only spec leaves exogenous columns untouched.
        np.testing.assert_allclose(m._data["X"].to_numpy(), raw["X"].to_numpy())

    def test_scaled_model_equivalent_to_manual_prescale(self):
        """Compiled model matches hand-prescaling the data exactly."""
        raw = make_panel()
        pop = xr.DataArray(
            [100.0, 10.0], coords={"geo": ["big", "small"]}, dims=["geo"]
        )
        scaling = Scaling(channel={"method": "divide", "by": pop, "dims": ("geo",)})
        auto = pathmc.model(SPEC, data=raw, panel=PANEL, scaling=scaling)

        manual_df = raw.copy()
        manual_df["X"] = manual_df["X"] / manual_df["geo"].map(POPULATIONS)
        manual = pathmc.model(SPEC, data=manual_df, panel=PANEL)

        assert auto._data is not None
        assert manual._data is not None
        np.testing.assert_allclose(
            auto._data["X"].to_numpy(), manual._data["X"].to_numpy()
        )
        for lhs in auto._design_matrices:
            np.testing.assert_allclose(
                auto._design_matrices[lhs].to_pandas().to_numpy(),
                manual._design_matrices[lhs].to_pandas().to_numpy(),
            )
        # Identical structure and data => identical pointwise log-probability.
        auto_lp = auto.pymc_model.point_logps()
        manual_lp = manual.pymc_model.point_logps()
        assert set(auto_lp) == set(manual_lp)
        for key, val in auto_lp.items():
            assert float(val) == pytest.approx(float(manual_lp[key]))

    def test_no_scaling_leaves_data_untouched(self):
        raw = make_panel()
        m = pathmc.model(SPEC, data=raw)
        assert m.fitted_scaling is None
        assert m._data is not None
        np.testing.assert_allclose(m._data["X"].to_numpy(), raw["X"].to_numpy())
        np.testing.assert_allclose(m._data["Y"].to_numpy(), raw["Y"].to_numpy())

    def test_target_and_channel_roles_scale_distinct_columns(self):
        rng = np.random.default_rng(3)
        raw = make_panel()
        raw["Y"] = rng.normal(size=len(raw)) + 100
        scaling = Scaling(
            target={"method": "max"},
            channel={"method": "fixed", "value": 2.0},
        )
        m = pathmc.model(SPEC, data=raw, scaling=scaling)
        assert m._data is not None
        np.testing.assert_allclose(
            m._data["Y"].to_numpy(),
            raw["Y"].to_numpy() / raw["Y"].max(),
        )
        np.testing.assert_allclose(m._data["X"].to_numpy(), raw["X"].to_numpy() / 2.0)

    def test_refit_permuted_reaches_model_construction(self, monkeypatch):
        """_refit_permuted must find every _construction key it reads.

        Regression for review round 1: the scaling wiring dropped the
        panel/pooling/latent keys from the recorded construction dict,
        so every refute_placebo() on a panel model raised KeyError.
        """
        raw = make_panel()
        m = pathmc.model(
            SPEC,
            data=raw,
            panel=PANEL,
            scaling=Scaling(target={"method": "max"}),
        )
        # Skip the MCMC pass; the point is reaching model construction.
        monkeypatch.setattr(pathmc.PathModel, "fit", lambda self, **kw: self)
        clone = m._refit_permuted(treatment="X", seed=0, sample_kwargs={})
        assert isinstance(clone, pathmc.PathModel)
        assert clone._panel_info is not None
        assert clone.fitted_scaling is None  # refits in already-scaled units


# ---------------------------------------------------------------------------
# simulate(): inverse scaling lands outputs in business units
# ---------------------------------------------------------------------------


class TestSimulateRoundTrip:
    PARAMS: dict[str, Any] = {
        # intercept + X effect, expressed in scaled-space units
        "beta_Y": np.array([5.0, 0.8]),
        "sigma_Y": 1e-6,
    }

    def test_outputs_in_business_units_when_scaling_passed(self):
        raw = make_panel()
        scaling = Scaling(
            target={
                "method": "fixed",
                "value": TARGET_SCALES,
                "dims": ("geo",),
            },
            channel={
                "method": "divide",
                "by": POPULATIONS,
                "dims": ("geo",),
            },
        )
        out = pathmc.simulate(
            SPEC, data=raw, params=self.PARAMS, panel=PANEL, scaling=scaling
        )
        t = raw["geo"].map(TARGET_SCALES).to_numpy()
        p = raw["geo"].map(POPULATIONS).to_numpy()
        beta = self.PARAMS["beta_Y"]
        expected = (beta[0] + beta[1] * raw["X"] / p) * t
        np.testing.assert_allclose(out["Y"].to_numpy(), expected, atol=1e-2)

    def test_without_scaling_params_stay_in_raw_units(self):
        raw = make_panel()
        out = pathmc.simulate(SPEC, data=raw, params=self.PARAMS, panel=PANEL)
        beta = self.PARAMS["beta_Y"]
        expected = beta[0] + beta[1] * raw["X"]
        np.testing.assert_allclose(out["Y"].to_numpy(), expected, atol=1e-2)

    def test_exogenous_columns_not_rescaled_in_output(self):
        raw = make_panel()
        scaling = Scaling(
            channel={"method": "divide", "by": POPULATIONS, "dims": ("geo",)}
        )
        out = pathmc.simulate(
            SPEC, data=raw, params=self.PARAMS, panel=PANEL, scaling=scaling
        )
        np.testing.assert_allclose(out["X"].to_numpy(), raw["X"].to_numpy())

    def test_prefitted_factors_object_accepted(self):
        raw = make_panel()
        factors = ScalingFactors(factors={"Y": np.full(len(raw), 10.0)})
        out = pathmc.simulate(
            SPEC, data=raw, params=self.PARAMS, panel=PANEL, scaling=factors
        )
        beta = self.PARAMS["beta_Y"]
        expected = (beta[0] + beta[1] * raw["X"]) * 10.0
        np.testing.assert_allclose(out["Y"].to_numpy(), expected, atol=1e-2)

    def test_template_accepts_scaling_and_ignores_shapes(self):
        raw = make_panel()
        base = pathmc.simulate_params_template(SPEC, data=raw, panel=PANEL)
        with_scaling = pathmc.simulate_params_template(
            SPEC, data=raw, panel=PANEL, scaling=Scaling(target={"method": "max"})
        )
        assert with_scaling == base

    def test_template_rejects_invalid_scaling(self):
        with pytest.raises(ValueError, match="Unknown"):
            pathmc.simulate_params_template(
                SPEC,
                data=make_panel(),
                scaling=Scaling(target={"method": "median"}),
            )

    def test_target_max_mean_rejected_in_simulate(self):
        raw = make_panel()
        with pytest.raises(ValueError, match="simulating"):
            pathmc.simulate(
                SPEC,
                data=raw,
                params=self.PARAMS,
                panel=PANEL,
                scaling=Scaling(target={"method": "max"}),
            )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown Scaling.target method"):
            pathmc.model(
                SPEC,
                data=make_panel(),
                scaling=Scaling(target={"method": "median"}),
            )

    def test_fixed_requires_value(self):
        with pytest.raises(ValueError, match='"fixed" requires'):
            pathmc.model(
                SPEC,
                data=make_panel(),
                scaling=Scaling(target={"method": "fixed"}),
            )

    def test_divide_requires_by(self):
        with pytest.raises(ValueError, match='"divide" requires'):
            pathmc.model(
                SPEC,
                data=make_panel(),
                scaling=Scaling(target={"method": "divide"}),
            )

    def test_dims_must_be_panel_unit_columns(self):
        with pytest.raises(ValueError, match="not panel unit columns"):
            pathmc.model(
                SPEC,
                data=make_panel(),
                panel=PANEL,
                scaling=Scaling(
                    channel={
                        "method": "divide",
                        "by": POPULATIONS,
                        "dims": ("region",),
                    }
                ),
            )

    def test_dims_require_panel(self):
        with pytest.raises(ValueError, match="require a panel model"):
            pathmc.model(
                SPEC,
                data=make_panel(),
                scaling=Scaling(
                    channel={
                        "method": "divide",
                        "by": POPULATIONS,
                        "dims": ("geo",),
                    }
                ),
            )

    def test_missing_grid_key(self):
        partial = {"big": 100.0}  # no "small" entry
        with pytest.raises(KeyError, match="missing entries"):
            pathmc.model(
                SPEC,
                data=make_panel(),
                panel=PANEL,
                scaling=Scaling(
                    channel={
                        "method": "divide",
                        "by": partial,
                        "dims": ("geo",),
                    }
                ),
            )

    def test_invalid_scaling_type_rejected(self):
        with pytest.raises(TypeError, match="accepts pathmc.Scaling"):
            pathmc.model(
                SPEC,
                data=make_panel(),
                scaling="max",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Slow MCMC tests: estimation equivalence and recovery
# ---------------------------------------------------------------------------


def make_business_unit_data(seed: int = 42) -> pd.DataFrame:
    """DGP with a common scaled-space effect, heterogeneous business units."""
    rng = np.random.default_rng(seed)
    beta, alpha = 0.8, 1.2
    frames = []
    for geo, pop in POPULATIONS.items():
        n_weeks = 20
        xd = rng.normal(size=n_weeks)
        yd = alpha + beta * xd + 0.1 * rng.normal(size=n_weeks)
        frames.append(
            pd.DataFrame({
                "geo": geo,
                "week": np.arange(n_weeks),
                "X": xd * pop,
                "Y": yd * TARGET_SCALES[geo],
            })
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.slow
class TestEstimationEquivalence:
    @pytest.fixture(scope="class")
    def fitted_pair(self):
        raw = make_business_unit_data()
        pop = xr.DataArray(
            [100.0, 10.0], coords={"geo": ["big", "small"]}, dims=["geo"]
        )
        scaling = Scaling(
            target={
                "method": "fixed",
                "value": TARGET_SCALES,
                "dims": ("geo",),
            },
            channel={"method": "divide", "by": pop, "dims": ("geo",)},
        )
        auto = pathmc.model(SPEC, data=raw, panel=PANEL, scaling=scaling)
        auto.fit(draws=300, tune=300, chains=1, random_seed=42)

        manual_df = raw.copy()
        manual_df["X"] = manual_df["X"] / manual_df["geo"].map(POPULATIONS)
        manual_df["Y"] = manual_df["Y"] / manual_df["geo"].map(TARGET_SCALES)
        manual = pathmc.model(SPEC, data=manual_df, panel=PANEL)
        manual.fit(draws=300, tune=300, chains=1, random_seed=42)
        return auto, manual

    def test_posteriors_match_manual_prescale(self, fitted_pair):
        auto, manual = fitted_pair
        auto_sum = auto.summary()
        manual_sum = manual.summary()
        shared = sorted(set(auto_sum.index) & set(manual_sum.index))
        assert shared, "no shared parameter names between fits"
        for param in shared:
            np.testing.assert_allclose(
                auto_sum.loc[param, "mean"],
                manual_sum.loc[param, "mean"],
                rtol=0.15,
                err_msg=param,
            )

    def test_recovers_common_scaled_effect(self, fitted_pair):
        auto, _ = fitted_pair
        summary = auto.summary()
        beta_rows = summary[summary.index.str.startswith("beta_Y")]
        effect_row = beta_rows[~beta_rows.index.str.contains("Intercept")]
        assert effect_row["mean"].iloc[0] == pytest.approx(0.8, abs=0.05)
