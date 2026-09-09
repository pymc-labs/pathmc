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
