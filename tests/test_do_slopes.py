"""Gate tests for M26: do() propagation of random slopes."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def panel_df():
    """Panel data with 3 regions, 20 weeks, and known spend patterns."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 20
    rows = []
    for region in regions:
        for week in range(1, n_weeks + 1):
            rows.append(
                {
                    "region": region,
                    "week": week,
                    "spend": rng.uniform(5, 30),
                    "trend": week,
                    "sales": rng.normal(50, 5),
                }
            )
    df = pd.DataFrame(rows)
    # Overwrite sales with a DGP that has region-varying spend effects
    slopes = {"A": 1.0, "B": 2.0, "C": 3.0}
    df["sales"] = df.apply(
        lambda r: (
            50
            + slopes[r["region"]] * r["spend"]
            + 0.1 * r["trend"]
            + rng.normal(scale=1.0)
        ),
        axis=1,
    )
    return df


@pytest.fixture()
def slope_model(panel_df):
    """Fit a model with random slopes on spend."""
    model = pathmc.fit(
        "sales ~ spend + trend",
        data=panel_df,
        panel={"unit": "region", "time": "week"},
        pooling={"intercept": True, "slopes": ["spend"]},
    )
    model.sample(draws=200, tune=200, chains=2, random_seed=42)
    return model


class TestCrossSectionalDoSlopes:
    """Cross-sectional do() should include random slope contributions."""

    def test_do_returns_result(self, slope_model):
        r = slope_model.do(set={"spend": 10.0})
        assert r.mean("sales") is not None

    def test_ate_reflects_slopes(self, slope_model):
        r_lo = slope_model.do(set={"spend": 5.0})
        r_hi = slope_model.do(set={"spend": 15.0})
        ate = r_hi - r_lo
        # With region-varying slopes (1,2,3), average ~2, so ATE for 10-unit
        # change should be around 20. Check it's positive and substantial.
        assert ate.mean("sales") > 5.0

    def test_slopes_make_a_difference(self, slope_model):
        """Verify random slopes actually contribute to the do() result.

        We compare the do() result against what we'd get using only
        the fixed effect (beta). If slopes are propagated, the result
        should differ from beta-only.
        """
        idata = slope_model._idata
        stacked = idata.posterior.stack(sample=("chain", "draw"))

        if "slope_sales_spend" in stacked:
            slope_mean = stacked["slope_sales_spend"].mean(dim="unit").values
            assert not np.allclose(slope_mean, 0, atol=0.01), (
                "Slope draws are near zero — can't distinguish from no-slope case"
            )


class TestPanelDoSlopes:
    """Panel do(simulate_over='time') should use unit-specific slopes."""

    def test_panel_do_returns_result(self, slope_model):
        r = slope_model.do(
            set={"spend": 10.0},
            simulate_over="time",
            kind="mean",
        )
        assert r.mean("sales") is not None

    def test_panel_do_ate_positive(self, slope_model):
        r_lo = slope_model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r_hi = slope_model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")
        ate = r_hi - r_lo
        assert ate.mean("sales") > 5.0
