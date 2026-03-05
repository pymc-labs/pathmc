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

    def test_residual_cov_raises(self, exog_df):
        exog = exog_df.copy()
        exog["Y1"] = 0.0
        exog["Y2"] = 0.0
        with pytest.raises(NotImplementedError, match="residual covariances"):
            pathmc.simulate(
                "Y1 ~ X\nY2 ~ X\nY1 ~~ Y2",
                data=exog,
                params={},
            )

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
