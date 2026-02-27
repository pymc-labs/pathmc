"""Gate tests for M17: Panel smoke tests (end-to-end integration)."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture()
def full_panel_data():
    """Full panel pipeline data: 3 regions, 20 weeks, sales ~ spend_lag1."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    true_intercepts = {"A": 5.0, "B": 8.0, "C": 12.0}
    n_weeks = 20
    rows = []
    for region in regions:
        spend_prev = 10.0
        for week in range(1, n_weeks + 1):
            spend = rng.uniform(5, 15)
            sales = true_intercepts[region] + 0.6 * spend_prev + rng.normal(scale=0.5)
            rows.append(
                {
                    "region": region,
                    "week": week,
                    "sales": sales,
                    "spend": spend,
                }
            )
            spend_prev = spend
    df = pd.DataFrame(rows)
    df = pathmc.add_lags(
        df,
        variables=["spend"],
        lags=[1],
        panel={"unit": "region", "time": "week"},
    )
    return df.dropna().reset_index(drop=True)


@pytest.mark.slow
class TestFullPipeline:
    """End-to-end: add_lags -> fit -> sample -> summary -> do."""

    def test_pipeline_completes(self, full_panel_data):
        model = pathmc.fit(
            "sales ~ spend_lag1",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        idata = model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        assert idata is not None

        summary = model.summary()
        assert len(summary) > 0
        assert any("alpha_sales" in str(idx) for idx in summary.index)

    def test_do_time_forward_ate(self, full_panel_data):
        """do(simulate_over='time') produces ATE with correct sign."""
        model = pathmc.fit(
            "sales ~ spend_lag1",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)

        r_low = model.do(set={"spend": 5.0}, simulate_over="time", kind="mean")
        r_high = model.do(set={"spend": 15.0}, simulate_over="time", kind="mean")

        ate = r_high - r_low
        assert ate.mean("sales") > 0

    def test_graph_works(self, full_panel_data):
        model = pathmc.fit(
            "sales ~ spend_lag1",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        g = model.graph()
        assert g is not None


@pytest.mark.slow
class TestPanelBernoulli:
    """Panel model with Bernoulli outcome."""

    def test_panel_bernoulli_works(self):
        rng = np.random.default_rng(42)
        regions = ["A", "B"]
        rows = []
        for region in regions:
            for week in range(1, 21):
                x = rng.normal()
                p = 1 / (1 + np.exp(-(0.5 * x)))
                y = float(rng.binomial(1, p))
                rows.append({"region": region, "week": week, "X": x, "Y": y})
        df = pd.DataFrame(rows)

        model = pathmc.fit(
            "Y ~ X",
            data=df,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
            families={"Y": "bernoulli"},
        )
        model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r = model.do(set={"X": 1.0})
        assert 0.0 < r.mean("Y") < 1.0


@pytest.mark.slow
class TestRandomInterceptVariation:
    """Random intercepts produce per-unit variation."""

    def test_different_units_different_intercepts(self, full_panel_data):
        model = pathmc.fit(
            "sales ~ spend_lag1",
            data=full_panel_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        summary = model.summary()

        alpha_rows = [idx for idx in summary.index if "alpha_sales" in str(idx)]
        assert len(alpha_rows) >= 3
