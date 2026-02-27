"""Gate tests for M13: add_lags() utility."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def panel_df():
    """Simple two-unit, four-period panel."""
    return pd.DataFrame(
        {
            "region": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "week": [1, 2, 3, 4, 1, 2, 3, 4],
            "sales": [10, 20, 30, 40, 50, 60, 70, 80],
            "spend": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


class TestLagValues:
    """Lag column names and values are correct."""

    def test_single_var_single_lag(self, panel_df):
        result = pathmc.add_lags(
            panel_df,
            variables=["sales"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        assert "sales_lag1" in result.columns

        a_rows = result[result["region"] == "A"]
        assert np.isnan(a_rows.iloc[0]["sales_lag1"])
        assert a_rows.iloc[1]["sales_lag1"] == 10
        assert a_rows.iloc[2]["sales_lag1"] == 20
        assert a_rows.iloc[3]["sales_lag1"] == 30

    def test_lag_does_not_cross_units(self, panel_df):
        result = pathmc.add_lags(
            panel_df,
            variables=["sales"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        b_rows = result[result["region"] == "B"]
        assert np.isnan(b_rows.iloc[0]["sales_lag1"])
        assert b_rows.iloc[1]["sales_lag1"] == 50

    def test_multiple_lags(self, panel_df):
        result = pathmc.add_lags(
            panel_df,
            variables=["sales"],
            lags=[1, 2],
            panel={"unit": "region", "time": "week"},
        )
        assert "sales_lag1" in result.columns
        assert "sales_lag2" in result.columns

        a_rows = result[result["region"] == "A"]
        assert np.isnan(a_rows.iloc[0]["sales_lag2"])
        assert np.isnan(a_rows.iloc[1]["sales_lag2"])
        assert a_rows.iloc[2]["sales_lag2"] == 10

    def test_multiple_variables(self, panel_df):
        result = pathmc.add_lags(
            panel_df,
            variables=["sales", "spend"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        assert "sales_lag1" in result.columns
        assert "spend_lag1" in result.columns

    def test_integer_lags_shorthand(self, panel_df):
        """lags=2 should create lag1 and lag2."""
        result = pathmc.add_lags(
            panel_df,
            variables=["sales"],
            lags=2,
            panel={"unit": "region", "time": "week"},
        )
        assert "sales_lag1" in result.columns
        assert "sales_lag2" in result.columns


class TestSorting:
    """Output is sorted by unit then time."""

    def test_unsorted_input_is_sorted(self):
        df = pd.DataFrame(
            {
                "region": ["B", "A", "B", "A"],
                "week": [2, 1, 1, 2],
                "sales": [60, 10, 50, 20],
            }
        )
        result = pathmc.add_lags(
            df,
            variables=["sales"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        assert list(result["region"]) == ["A", "A", "B", "B"]
        assert list(result["week"]) == [1, 2, 1, 2]


class TestValidation:
    """Errors on invalid inputs."""

    def test_missing_unit_key(self, panel_df):
        with pytest.raises(KeyError, match="unit"):
            pathmc.add_lags(
                panel_df,
                variables=["sales"],
                lags=[1],
                panel={"time": "week"},
            )

    def test_missing_time_key(self, panel_df):
        with pytest.raises(KeyError, match="time"):
            pathmc.add_lags(
                panel_df,
                variables=["sales"],
                lags=[1],
                panel={"unit": "region"},
            )

    def test_missing_unit_column(self, panel_df):
        with pytest.raises(KeyError, match="country"):
            pathmc.add_lags(
                panel_df,
                variables=["sales"],
                lags=[1],
                panel={"unit": "country", "time": "week"},
            )

    def test_missing_variable_column(self, panel_df):
        with pytest.raises(KeyError, match="revenue"):
            pathmc.add_lags(
                panel_df,
                variables=["revenue"],
                lags=[1],
                panel={"unit": "region", "time": "week"},
            )


class TestEdgeCases:
    """Edge cases."""

    def test_single_unit(self):
        df = pd.DataFrame(
            {"region": ["A", "A", "A"], "week": [1, 2, 3], "sales": [10, 20, 30]}
        )
        result = pathmc.add_lags(
            df,
            variables=["sales"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        assert len(result) == 3
        assert np.isnan(result.iloc[0]["sales_lag1"])
        assert result.iloc[1]["sales_lag1"] == 10

    def test_original_columns_preserved(self, panel_df):
        result = pathmc.add_lags(
            panel_df,
            variables=["sales"],
            lags=[1],
            panel={"unit": "region", "time": "week"},
        )
        for col in ["region", "week", "sales", "spend"]:
            assert col in result.columns
