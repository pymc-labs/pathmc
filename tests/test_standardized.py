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
"""Gate tests for M29: Standardized effects (stdyx)."""

import arviz as az
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc import Scaling


@pytest.fixture(scope="module")
def mediation_model(mock_pymc_sample_module):
    """Mediation model with labeled coefficients."""
    rng = np.random.default_rng(42)
    n = 400
    X = rng.normal(loc=5, scale=2, size=n)
    M = 0.5 * X + rng.normal(scale=1.0, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"X": X, "M": M, "Y": Y})

    model = pathmc.model("M ~ a*X\nY ~ b*M + c*X", data=df)
    model.fit(draws=50, tune=50, chains=2, random_seed=42)
    return model, df


class TestStandardizedEffects:
    def test_returns_dataframe(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_labels(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        assert "a" in result.index
        assert "b" in result.index
        assert "c" in result.index

    def test_has_expected_columns(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        assert "mean" in result.columns
        assert "sd" in result.columns
        assert "hdi_3%" in result.columns
        assert "hdi_97%" in result.columns

    def test_has_predictor_outcome_columns(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        assert "predictor" in result.columns
        assert "outcome" in result.columns

    def test_stdyx_matches_manual(self, mediation_model):
        model, df = mediation_model
        result = model.standardized()

        # Manual computation for 'a' (X -> M)
        sd_x = float(df["X"].std())
        sd_m = float(df["M"].std())
        idata = model._idata
        a_draws = idata.posterior["beta_M"].sel(M_predictors="X").values.flatten()
        expected_stdyx = float(np.mean(a_draws * sd_x / sd_m))
        actual_stdyx = result.loc["a", "mean"]
        assert abs(actual_stdyx - expected_stdyx) < 1e-6

    def test_values_are_finite(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        assert result["mean"].notna().all()
        assert result["sd"].notna().all()

    def test_hdi_ordered(self, mediation_model):
        model, _ = mediation_model
        result = model.standardized()
        for _, row in result.iterrows():
            assert row["hdi_3%"] < row["hdi_97%"]


class TestStandardizedBeforeSample:
    def test_raises_before_sample(self):
        df = pd.DataFrame({
            "X": np.random.normal(size=50),
            "Y": np.random.normal(size=50),
        })
        model = pathmc.model("Y ~ a*X", data=df)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.standardized()


def _pin_internal_beta(model, value: float) -> None:
    order = [str(v) for v in model.pymc_model.coords["Y_predictors"]]
    model._idata = az.from_dict(
        {"posterior": {"beta_Y": np.full((1, 4, len(order)), value)}},
        coords={"Y_predictors": order},
        dims={"beta_Y": ["Y_predictors"]},
    )


class TestHeterogeneousScalingStdyx:
    """stdyx must use the business-unit coefficient and observed SDs.

    Uniform factors cancel, so the internal ratio matches. Heterogeneous
    factors do not: returning ``internal beta * sd(internal X) / sd(internal Y)``
    is the failure.
    """

    @pytest.mark.parametrize(
        ("y_factors", "x_factors", "x_pattern"),
        [
            ({"large": 10.0, "small": 100.0}, None, (1.0, 2.0, 3.0)),
            (
                {"north": 4.0, "south": 25.0, "west": 2.0},
                {"north": 8.0, "south": 3.0, "west": 12.0},
                (1.5, 2.5, 6.0, 9.0),
            ),
            (
                None,
                {"large": 5.0, "small": 40.0},
                (2.0, 3.0, 5.0, 8.0),
            ),
        ],
    )
    def test_matches_business_beta_times_observed_sds(
        self, y_factors, x_factors, x_pattern
    ):
        geos = list((y_factors or x_factors).keys())
        rows = []
        for geo in geos:
            y_factor = 1.0 if y_factors is None else y_factors[geo]
            for time, xv in enumerate(x_pattern):
                rows.append({
                    "geo": geo,
                    "time": time,
                    "X": xv + 0.25 * time,
                    "Y": y_factor * (xv + 0.5 * time),
                })
        df = pd.DataFrame(rows)
        scaling_kwargs = {}
        if y_factors is not None:
            scaling_kwargs["target"] = {
                "method": "divide",
                "by": y_factors,
                "dims": ("geo",),
            }
        if x_factors is not None:
            scaling_kwargs["channel"] = {
                "method": "divide",
                "by": x_factors,
                "dims": ("geo",),
            }
        model = pathmc.model(
            "Y ~ 0 + b*X",
            data=df,
            panel={"unit": "geo", "time": "time"},
            scaling=Scaling(**scaling_kwargs),
        )
        _pin_internal_beta(model, 2.0)

        beta = float(model.effects_summary().loc["b", "mean"])
        got = float(model.standardized().loc["b", "mean"])
        oracle = beta * float(df["X"].std()) / float(df["Y"].std())
        internal = model._data
        internal_ratio = 2.0 * float(internal["X"].std()) / float(internal["Y"].std())

        assert got == pytest.approx(oracle, rel=1e-6, abs=1e-8)
        assert internal_ratio != pytest.approx(oracle, rel=1e-3)
        assert got != pytest.approx(internal_ratio, rel=1e-3)
