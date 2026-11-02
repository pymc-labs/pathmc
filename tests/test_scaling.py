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
    info = build_panel_info(_nw(base), panel)
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
        nw_df = _nw(raw)
        np.testing.assert_allclose(factors._per_row(nw_df, "Y"), raw["Y"].max())

    def test_mean_per_group(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "mean", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            target_columns={"Y"},
        )
        nw_df = _nw(raw)
        expected = raw.groupby("geo")["Y"].transform("mean").to_numpy()
        np.testing.assert_allclose(factors._per_row(nw_df, "Y"), expected)

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
        nw_df = _nw(make_panel())
        np.testing.assert_allclose(factors._per_row(nw_df, "X"), 25.0)

    def test_fixed_constant_with_dims_reuses_on_unseen_units(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(channel={"method": "fixed", "value": 25.0, "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            channel_columns={"X"},
        )
        future = raw.assign(geo="new")
        np.testing.assert_allclose(factors._per_row(_nw(future), "X"), 25.0)

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
        nw_df = _nw(raw)
        np.testing.assert_allclose(
            factors._per_row(nw_df, "Y"), raw["geo"].map(TARGET_SCALES).to_numpy()
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
        nw_df = _nw(raw)
        np.testing.assert_allclose(
            factors._per_row(nw_df, "X"), raw["geo"].map(POPULATIONS).to_numpy()
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
        nw_df = _nw(raw)
        np.testing.assert_allclose(
            factors._per_row(nw_df, "spend"), raw["geo"].map(POPULATIONS).to_numpy()
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
        nw_df = _nw(raw)
        expected = [lookup[k] for k in zip(raw["geo"], raw["brand"])]
        np.testing.assert_allclose(f_t._per_row(nw_df, "spend"), expected)
        np.testing.assert_allclose(f_j._per_row(nw_df, "spend"), expected)

    def test_reordered_frame_uses_unit_keys_not_position(self):
        """Factors keyed by unit survive row reordering."""
        raw = make_panel()
        factors = fit_scaling(
            Scaling(channel={"method": "max", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            channel_columns={"X"},
        )
        reversed_native = raw.iloc[::-1].reset_index(drop=True)
        scaled = factors.transform(_nw(reversed_native))
        divisor = reversed_native.groupby("geo")["X"].transform("max").to_numpy()
        np.testing.assert_allclose(
            scaled["X"].to_numpy(), reversed_native["X"] / divisor
        )

    def test_subset_frame_uses_matching_unit_keys(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(channel={"method": "max", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            channel_columns={"X"},
        )
        subset = raw[raw["geo"] == "big"].reset_index(drop=True)
        scaled = factors.transform(_nw(subset))
        divisor = subset.groupby("geo")["X"].transform("max").to_numpy()
        np.testing.assert_allclose(scaled["X"].to_numpy(), subset["X"] / divisor)

    def test_unseen_unit_raises_key_error(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(channel={"method": "max", "dims": ("geo",)}),
            _nw(raw),
            panel_info=_panel_info(PANEL),
            channel_columns={"X"},
        )
        extra = pd.concat(
            [
                raw,
                pd.DataFrame({
                    "geo": ["unknown"],
                    "week": [99],
                    "X": [1.0],
                    "Y": [1.0],
                }),
            ],
            ignore_index=True,
        )
        with pytest.raises(KeyError, match="no entry"):
            factors.transform(_nw(extra))

    def test_transform_inverse_round_trip(self):
        raw = make_panel()
        factors = fit_scaling(
            Scaling(target={"method": "max"}),
            _nw(raw),
            target_columns={"Y"},
        )
        nw_df = _nw(raw)
        scaled = factors.transform(nw_df)
        recovered = factors.inverse_transform_column(scaled["Y"].to_numpy(), "Y", nw_df)
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

    def test_lag_only_channel_is_divided_before_compile(self):
        """A column used only as lag(X) is still that channel."""
        raw = pd.DataFrame({
            "geo": ["a", "a", "b", "b"],
            "week": [0, 1, 0, 1],
            "X": [8.0, 4.0, 2.0, 6.0],
            "Z": [10.0, 20.0, 30.0, 40.0],
            "Y": [1.0, 1.0, 1.0, 1.0],
        })
        scaling = Scaling(
            target={"method": "fixed", "value": 10.0},
            channel={"method": "fixed", "value": 2.0},
        )
        fitted = pathmc.model(
            "Y ~ lag(X) + Z",
            data=raw,
            panel=PANEL,
            scaling=scaling,
        )
        assert fitted._data is not None
        np.testing.assert_allclose(fitted._data["X"].to_numpy(), raw["X"] / 2.0)
        np.testing.assert_allclose(fitted._data["Z"].to_numpy(), raw["Z"] / 2.0)

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

    def test_lag_only_channel_divided_before_lag(self):
        """Divide a lag-only channel before the lag, then restore the target.

        The cited probe is channel factor 4 and target factor 10. A second
        series with different factors rejects a special case for those
        numbers, and a contemporaneous companion column must still be divided.
        """
        df = pd.DataFrame({
            "geo": ["a"] * 4,
            "week": [0, 1, 2, 3],
            "X": [4.0, 2.0, 6.0, 0.0],
        })
        scale = Scaling(
            target={"method": "fixed", "value": 10.0},
            channel={"method": "fixed", "value": 4.0},
        )
        params = {"beta_Y": [0.0, 3.0], "sigma_Y": 0.0}
        lag_only = pathmc.simulate(
            "Y ~ lag(X)",
            data=df,
            params=params,
            panel=PANEL,
            scaling=scale,
            random_seed=1,
        )
        # First row repeats, then the lag sees X / 4, then * 10.
        np.testing.assert_allclose(lag_only["Y"].to_numpy(), [30.0, 30.0, 15.0, 45.0])

        other = pd.DataFrame({
            "geo": ["a", "a", "a"],
            "week": [0, 1, 2],
            "X": [6.0, 3.0, 9.0],
            "Z": [8.0, 4.0, 2.0],
        })
        other_scale = Scaling(
            target={"method": "fixed", "value": 5.0},
            channel={"method": "fixed", "value": 2.0},
        )
        mixed = pathmc.simulate(
            "Y ~ lag(X) + Z",
            data=other,
            params={"beta_Y": [0.0, 1.0, 1.0], "sigma_Y": 0.0},
            panel=PANEL,
            scaling=other_scale,
            random_seed=1,
        )
        # lag(X/2) repeats 3 then shifts; Z/2 is contemporaneous; * 5.
        np.testing.assert_allclose(mixed["Y"].to_numpy(), [35.0, 25.0, 12.5])
        np.testing.assert_allclose(mixed["X"].to_numpy(), other["X"].to_numpy())
        np.testing.assert_allclose(mixed["Z"].to_numpy(), other["Z"].to_numpy())

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
        factors = ScalingFactors(
            factors={"Y": (("geo",), {("big",): 10.0, ("small",): 10.0})}
        )
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


class TestSimulateLatentBusinessUnits:
    """Latent generated columns are inverse-scaled with the other outcomes."""

    def test_latent_mediator_matches_observed_business_units(self):
        """A latent mediator uses the target factor, not internal units.

        Nearby to the reported probe (channel 4 / target 10 / unit slopes):
        different factors and slopes, so a special case for those inputs fails.
        """
        data = pd.DataFrame({"X": [1.0, 0.0, 3.0, 2.0]})
        scaling = Scaling(
            target={"method": "fixed", "value": 5.0},
            channel={"method": "fixed", "value": 2.0},
        )
        spec = "M ~ 0 + X\nY ~ 0 + M"
        params = {"beta_M": [2.0], "beta_Y": [3.0], "sigma_Y": 0.0}
        observed = pathmc.simulate(
            spec,
            data=data,
            params={**params, "sigma_M": 0.0},
            scaling=scaling,
            random_seed=0,
        )
        latent = pathmc.simulate(
            spec,
            data=data,
            params=params,
            scaling=scaling,
            latent=["M"],
            random_seed=0,
        )
        # Internal M is 2 * (X / 2) = X; business M is 5 * X.
        # Internal Y is 3 * M_internal; business Y is 15 * X.
        expected_m = np.array([5.0, 0.0, 15.0, 10.0])
        expected_y = np.array([15.0, 0.0, 45.0, 30.0])
        np.testing.assert_allclose(observed["M"].to_numpy(), expected_m)
        np.testing.assert_allclose(observed["Y"].to_numpy(), expected_y)
        np.testing.assert_allclose(latent["M"].to_numpy(), expected_m)
        np.testing.assert_allclose(latent["Y"].to_numpy(), expected_y)
        np.testing.assert_allclose(latent["M"].to_numpy(), observed["M"].to_numpy())

    def test_only_latent_endogenous_column_is_inverse_scaled(self):
        """Dropping every latent from target columns leaves this column unscaled."""
        data = pd.DataFrame({"X": [4.0, 8.0, 0.0]})
        out = pathmc.simulate(
            "L ~ 0 + X",
            data=data,
            params={"beta_L": [1.0]},
            scaling=Scaling(
                target={"method": "fixed", "value": 10.0},
                channel={"method": "fixed", "value": 4.0},
            ),
            latent=["L"],
            random_seed=1,
        )
        np.testing.assert_allclose(out["L"].to_numpy(), [10.0, 20.0, 0.0])

    def test_stochastic_latent_draw_is_inverse_scaled(self):
        """A zero-noise latent_normal draw is still a generated endogenous column."""
        data = pd.DataFrame({"X": [2.0, 6.0]})
        out = pathmc.simulate(
            "M ~ 0 + X\nY ~ 0 + M",
            data=data,
            params={
                "beta_M": [1.0],
                "sigma_M": 0.0,
                "beta_Y": [1.0],
                "sigma_Y": 0.0,
            },
            families={"M": "latent_normal"},
            scaling=Scaling(
                target={"method": "fixed", "value": 3.0},
                channel={"method": "fixed", "value": 2.0},
            ),
            latent=["M"],
            random_seed=2,
        )
        # Internal M = X / 2; business M = 1.5 * X. Y uses that internal M.
        np.testing.assert_allclose(out["M"].to_numpy(), [3.0, 9.0])
        np.testing.assert_allclose(out["Y"].to_numpy(), [3.0, 9.0])


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

    def test_fixed_negative_value_rejected(self):
        with pytest.raises(ValueError, match="non-positive"):
            fit_scaling(
                Scaling(channel={"method": "fixed", "value": -2.0}),
                _nw(make_panel()),
                channel_columns={"X"},
            )

    def test_divide_grid_nan_rejected(self):
        with pytest.raises(ValueError, match="non-positive"):
            fit_scaling(
                Scaling(
                    channel={
                        "method": "divide",
                        "by": {"big": 100.0, "small": float("nan")},
                        "dims": ("geo",),
                    }
                ),
                _nw(make_panel()),
                panel_info=_panel_info(PANEL),
                channel_columns={"X"},
            )

    def test_divide_grid_inf_rejected(self):
        with pytest.raises(ValueError, match="non-positive"):
            fit_scaling(
                Scaling(
                    channel={
                        "method": "divide",
                        "by": {"big": float("inf"), "small": 10.0},
                        "dims": ("geo",),
                    }
                ),
                _nw(make_panel()),
                panel_info=_panel_info(PANEL),
                channel_columns={"X"},
            )


# ---------------------------------------------------------------------------
# do() interventions are in business units
# ---------------------------------------------------------------------------


def _beta_mean(model: pathmc.PathModel, lhs: str, term: str) -> float:
    """Posterior mean of ``beta_{lhs}``'s coefficient whose index contains *term*."""
    summary = model.summary()
    rows = summary[summary.index.str.startswith(f"beta_{lhs}")]
    hit = rows[rows.index.str.contains(term, regex=False)]
    if hit.empty:
        raise AssertionError(
            f"no beta_{lhs} row matching {term!r} in {list(rows.index)}"
        )
    return float(hit["mean"].iloc[0])


class TestDoBusinessUnits:
    def test_global_channel_scale_set_is_business_units(self, mock_pymc_sample):
        """do(set={"tv": 500}) means 500 raw units, not 500 scaled units (G2)."""
        rng = np.random.default_rng(0)
        tv = rng.uniform(0, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 2.0 * (tv / 1000.0) + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ tv", data=df, scaling=Scaling(channel={"method": "max"})
        )
        m.fit()
        assert m.fitted_scaling is not None
        factor = next(iter(m.fitted_scaling.factors["tv"][1].values()))
        raw = 500.0
        got = float(m.do(set={"tv": raw}, kind="mean").mean("sales"))
        intercept = _beta_mean(m, "sales", "Intercept")
        slope = _beta_mean(m, "sales", "tv")
        expected = intercept + slope * (raw / factor)
        assert got == pytest.approx(expected, rel=1e-5)
        unscaled_wrong = intercept + slope * raw
        assert abs(got - expected) < abs(got - unscaled_wrong)

    def test_per_geo_channel_scale_set_is_business_units(self, mock_pymc_sample):
        """Per-unit factors: the same raw spend is a different scaled value per geo."""
        raw = make_panel()
        scaling = Scaling(
            channel={"method": "divide", "by": POPULATIONS, "dims": ("geo",)}
        )
        m = pathmc.model(SPEC, data=raw, panel=PANEL, scaling=scaling)
        m.fit()
        spend = 5.0
        got = float(m.do(set={"X": spend}, kind="mean").mean("Y"))
        intercept = _beta_mean(m, "Y", "Intercept")
        slope = _beta_mean(m, "Y", "X")
        scaled_mean = float(np.mean([spend / POPULATIONS[g] for g in raw["geo"]]))
        expected = intercept + slope * scaled_mean
        assert got == pytest.approx(expected, rel=1e-5)

    def test_scan_panel_do_scales_set(self, mock_pymc_sample):
        """Time-forward do() divides channel interventions before the scan."""
        rng = np.random.default_rng(1)
        n_units, n_times = 3, 8
        df = pd.DataFrame([
            {
                "geo": f"g{u}",
                "week": t,
                "tv": rng.uniform(10, 100),
                "sales": rng.normal(),
            }
            for u in range(n_units)
            for t in range(n_times)
        ])
        factor = 10.0
        m = pathmc.model(
            "sales ~ lag(sales) + tv",
            data=df,
            panel=PANEL,
            scaling=Scaling(channel={"method": "fixed", "value": factor}),
        )
        m.fit()
        spend = 50.0
        result = m.do(set={"tv": spend}, simulate_over="time", kind="mean")
        intercept = _beta_mean(m, "sales", "Intercept")
        slope = _beta_mean(m, "sales", "tv")
        # Cold start is not a clean intercept + beta*tv identity (lag carry
        # and unit-averaging add a small residual), so check the scale
        # convention: 50 raw must be closer to tv=5 internal than to tv=50.
        first = float(result.by_time("sales")[0].mean())
        scaled = intercept + slope * (spend / factor)
        unscaled = intercept + slope * spend
        assert abs(first - scaled) < abs(first - unscaled)

    def test_do_output_target_in_business_units(self, mock_pymc_sample):
        """do() outcomes are returned in business units when the target is scaled."""
        rng = np.random.default_rng(2)
        tv = rng.uniform(10, 100, 60)
        sales = 3.0 * tv + rng.normal(scale=1.0, size=60)
        df = pd.DataFrame({"tv": tv, "sales": sales})
        m = pathmc.model(
            "sales ~ tv",
            data=df,
            scaling=Scaling(target={"method": "max"}),
        )
        m.fit()
        assert m.fitted_scaling is not None
        target_factor = next(iter(m.fitted_scaling.factors["sales"][1].values()))
        raw_tv = 50.0
        got = float(m.do(set={"tv": raw_tv}, kind="mean").mean("sales"))
        intercept = _beta_mean(m, "sales", "Intercept")
        slope = _beta_mean(m, "sales", "tv")
        expected = (intercept + slope * raw_tv) * target_factor
        assert got == pytest.approx(expected, rel=1e-4)
        scaled_mean = intercept + slope * raw_tv
        assert abs(got - expected) < abs(got - scaled_mean)

    def test_effects_summary_slope_in_business_units(self, mock_pymc_sample):
        """effects_summary() rescales channel coefficients to business units."""
        rng = np.random.default_rng(3)
        tv = rng.uniform(10, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 2.0 * (tv / 1000.0) + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ b*tv", data=df, scaling=Scaling(channel={"method": "max"})
        )
        m.fit()
        factor = next(iter(m.fitted_scaling.factors["tv"][1].values()))
        summary = m.effects_summary()
        post = m._idata["posterior"]["beta_sales"].mean(("chain", "draw")).values
        beta_scaled = float(post[1])
        beta_business = beta_scaled / factor
        assert summary.loc["b", "mean"] == pytest.approx(beta_business, rel=1e-5)

    def test_saturation_coef_not_divided_by_channel_factor(self, mock_pymc_sample):
        """S1: logistic_saturation is unitless; only the outcome factor applies."""
        rng = np.random.default_rng(4)
        tv = rng.uniform(10, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 5.0 * np.tanh(tv / 1000.0) + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ b*logistic_saturation(tv, lam=lam_tv)",
            data=df,
            scaling=Scaling(channel={"method": "max"}),
        )
        m.fit()
        factor = next(iter(m.fitted_scaling.factors["tv"][1].values()))
        internal = _label_mean(m, "b")
        reported = m.effects_summary().loc["b", "mean"]
        assert reported == pytest.approx(internal, rel=1e-5)
        assert abs(reported - internal) < abs(reported - internal / factor)

    def test_interaction_coef_divides_by_product_of_factors(self, mock_pymc_sample):
        """S1: tv:radio coefficient is per product of the two raw columns."""
        rng = np.random.default_rng(5)
        tv = rng.uniform(10, 1000, 80)
        radio = rng.uniform(10, 500, 80)
        df = pd.DataFrame({
            "tv": tv,
            "radio": radio,
            "sales": 1.0 + 2e-6 * tv * radio + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ b*tv:radio",
            data=df,
            scaling=Scaling(channel={"method": "max"}),
        )
        m.fit()
        f_tv = next(iter(m.fitted_scaling.factors["tv"][1].values()))
        f_radio = next(iter(m.fitted_scaling.factors["radio"][1].values()))
        internal = _label_mean(m, "b")
        reported = m.effects_summary().loc["b", "mean"]
        expected = internal / (f_tv * f_radio)
        assert reported == pytest.approx(expected, rel=1e-5)

    def test_defined_param_matches_product_of_rescaled_labels(self, mock_pymc_sample):
        """N1: indirect := a*b uses business-unit a and b, not mixed units."""
        rng = np.random.default_rng(6)
        x = rng.uniform(10, 1000, 80)
        m_obs = 0.5 * x + rng.normal(scale=1.0, size=80)
        y = 0.4 * m_obs + rng.normal(scale=1.0, size=80)
        df = pd.DataFrame({"X": x, "M": m_obs, "Y": y})
        model = pathmc.model(
            "M ~ a*X\nY ~ b*M\nindirect := a*b",
            data=df,
            scaling=Scaling(
                target={"method": "max"},
                channel={"method": "max"},
            ),
        )
        model.fit()
        from pathmc.effects import extract_labeled_draws, _labeled_coef_business_scale

        draws = extract_labeled_draws(model._spec, model._idata)
        sa = _labeled_coef_business_scale(
            model._spec, "a", model.fitted_scaling, model._data
        )
        sb = _labeled_coef_business_scale(
            model._spec, "b", model.fitted_scaling, model._data
        )
        expected = float(np.mean(draws["a"] * sa * draws["b"] * sb))
        reported = model.effects_summary().loc["indirect", "mean"]
        assert sa * sb != pytest.approx(1.0, rel=0.2)
        assert reported == pytest.approx(expected, rel=1e-5)
        internal_product = float(np.mean(draws["a"] * draws["b"]))
        assert reported != pytest.approx(internal_product, rel=0.05)

    def test_predict_scan_panel_unscales_per_unit(self, mock_pymc_sample):
        """S2: scan-panel predict() multiplies by per-unit factors, not the mean."""
        import pymc as pm

        from pathmc._model import _observed_carry

        rng = np.random.default_rng(8)
        pops = {"big": 1000.0, "small": 10.0}
        frames = []
        for geo, pop in pops.items():
            tv = rng.uniform(1.0, 10.0, 12)
            frames.append(
                pd.DataFrame({
                    "geo": geo,
                    "week": np.arange(12),
                    "tv": tv,
                    "sales": 2.0 * tv * pop + rng.normal(scale=0.1 * pop, size=12),
                })
            )
        df = pd.concat(frames, ignore_index=True)
        m = pathmc.model(
            "sales ~ lag(sales) + tv",
            data=df,
            panel=PANEL,
            scaling=Scaling(
                target={"method": "divide", "by": pops, "dims": ("geo",)},
                channel={"method": "fixed", "value": 1.0},
            ),
        )
        m.fit()
        assert m._gen_model is not None
        scan_info = m._gen_model._pathmc_panel_scan
        with m._pymc_model, _observed_carry(m._pymc_model, True):
            raw = pm.sample_posterior_predictive(
                m._idata, random_seed=0, extend_inferencedata=False, progressbar=False
            )
        internal = raw.posterior_predictive["sales"]
        per_row = m.fitted_scaling._per_row(m._data, "sales")
        factor_mat = (
            per_row[scan_info.sort_idx].reshape(scan_info.n_units, scan_info.n_times).T
        )
        expected = internal.values * factor_mat
        idata = m.predict(random_seed=0, progressbar=False)
        got = idata.posterior_predictive["sales"].values
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
        mean_f = m.fitted_scaling.mean_factor("sales", m._data)
        assert np.max(np.abs(got - internal.values * mean_f)) > 1.0

    def test_slopes_use_business_unit_endpoints(self, mock_pymc_sample):
        """S3: slopes() finite-differences at business-unit x, not x/F²."""
        rng = np.random.default_rng(9)
        tv = rng.uniform(50, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 5.0 * np.tanh(tv / 400.0) + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ b*logistic_saturation(tv, lam=lam_tv)",
            data=df,
            scaling=Scaling(channel={"method": "max"}),
        )
        m.fit()
        eps = 1e-4
        got = float(m.slopes("sales", "tv", eps=eps).mean())
        lo = float(m.do(set={"tv": tv}, kind="mean").mean("sales"))
        hi = float(m.do(set={"tv": tv + eps}, kind="mean").mean("sales"))
        expected = (hi - lo) / eps
        assert got == pytest.approx(expected, rel=1e-3)

    def test_slopes_eyex_uses_business_x_over_business_y(self, mock_pymc_sample):
        """S3: eyex = dydx * (x_business / mu_business), not mixed units."""
        import xarray as xr

        from pathmc.interpret import _column_in_business_units, _unit_prediction

        rng = np.random.default_rng(10)
        tv = rng.uniform(100, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 2.0 * tv + 50.0 + rng.normal(scale=1.0, size=80),
        })
        m = pathmc.model(
            "sales ~ tv", data=df, scaling=Scaling(channel={"method": "max"})
        )
        m.fit()
        x = _column_in_business_units(m, m._data, "tv")
        lo = _unit_prediction(m, m._data, "sales", {"tv": x})
        dydx = m.slopes("sales", "tv", average_by=None).dataset["sales"]
        eyex = m.slopes("sales", "tv", slope="eyex", average_by=None).dataset["sales"]
        x_da = xr.DataArray(x, dims=["unit"], coords={"unit": dydx.coords["unit"]})
        expected = dydx * (x_da / lo)
        np.testing.assert_allclose(
            eyex.mean(("chain", "draw")).values,
            expected.mean(("chain", "draw")).values,
            rtol=1e-4,
            atol=1e-6,
        )
        x_scaled = np.asarray(m._data["tv"].to_numpy(), dtype=float)
        wrong = dydx * (
            xr.DataArray(x_scaled, dims=["unit"], coords={"unit": dydx.coords["unit"]})
            / lo
        )
        err_right = np.max(
            np.abs(
                eyex.mean(("chain", "draw")).values
                - expected.mean(("chain", "draw")).values
            )
        )
        err_wrong = np.max(
            np.abs(
                eyex.mean(("chain", "draw")).values
                - wrong.mean(("chain", "draw")).values
            )
        )
        assert err_right < err_wrong

    def test_predictions_newdata_matches_do_set(self, mock_pymc_sample):
        """N2: predictions(newdata=) treats the grid as business units, like do(set=)."""
        rng = np.random.default_rng(11)
        tv = rng.uniform(10, 1000, 80)
        df = pd.DataFrame({
            "tv": tv,
            "sales": 2.0 * (tv / 1000.0) + 0.01 * rng.normal(size=80),
        })
        m = pathmc.model(
            "sales ~ tv", data=df, scaling=Scaling(channel={"method": "max"})
        )
        m.fit()
        raw = 500.0
        pred = m.predictions("sales", newdata=pd.DataFrame({"tv": [raw]}))
        got = float(pred.dataset["sales"].mean(("chain", "draw", "unit")))
        expected = float(m.do(set={"tv": raw}, kind="mean").mean("sales"))
        assert got == pytest.approx(expected, rel=1e-4)

    def test_predict_observed_data_in_business_units(self, mock_pymc_sample):
        """N3: observed_data is inverse-transformed; a second predict() is idempotent."""
        rng = np.random.default_rng(12)
        tv = rng.uniform(10, 100, 60)
        sales = 3.0 * tv + rng.normal(scale=1.0, size=60)
        df = pd.DataFrame({"tv": tv, "sales": sales})
        m = pathmc.model(
            "sales ~ tv", data=df, scaling=Scaling(target={"method": "max"})
        )
        m.fit()
        idata = m.predict()
        obs = float(idata.observed_data["sales"].mean())
        assert obs == pytest.approx(float(df["sales"].mean()), rel=1e-6)
        scaled_mean = float(np.mean(m._data["sales"].to_numpy()))
        assert abs(obs - float(df["sales"].mean())) < abs(obs - scaled_mean)
        idata2 = m.predict()
        obs2 = float(idata2.observed_data["sales"].mean())
        assert obs2 == pytest.approx(obs, rel=1e-12)

    def test_predict_replacement_data_keeps_observed_data_in_business_units(
        self, mock_pymc_sample
    ):
        rng = np.random.default_rng(13)
        tv = rng.uniform(10, 100, 60)
        sales = 3.0 * tv + rng.normal(scale=1.0, size=60)
        df = pd.DataFrame({"tv": tv, "sales": sales})
        model = pathmc.model(
            "sales ~ tv",
            data=df,
            scaling=Scaling(
                target={"method": "max"},
                channel={"method": "max"},
            ),
        )
        model.fit()

        replacement = pd.DataFrame({"tv": np.full(len(df), 50.0)})
        idata = model.predict(data=replacement)

        np.testing.assert_allclose(
            idata.observed_data["sales"].values,
            df["sales"].to_numpy(),
            rtol=1e-6,
        )


def _label_mean(model: pathmc.PathModel, label: str) -> float:
    """Posterior mean of a labeled coefficient on the *internal* scale."""
    from pathmc.effects import extract_labeled_draws

    return float(np.mean(extract_labeled_draws(model._spec, model._idata)[label]))


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
