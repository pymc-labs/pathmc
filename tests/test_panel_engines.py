"""Cross-engine comparison tests for panel do().

Verifies that the three panel do() engines (numpy, batched, scan)
produce consistent results on models with known DGPs.

Engine properties:
- **numpy** and **scan** propagate per-draw temporal state and should
  agree within floating-point tolerance.
- **batched** averages temporal carry-over across draws (mean-field
  approximation imposed by ``pm.Data``).  It agrees closely with the
  other two engines but may show small deviations for models with
  strong temporal dynamics and wide posteriors.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


# ---------------------------------------------------------------------------
# Fixtures — each fits one model, reused across the class
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def simple_panel():
    """No temporal state: sales ~ spend.  All engines must agree exactly."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["A", "B", "C"]:
        for week in range(1, 16):
            spend = rng.uniform(10, 50)
            sales = 50 + 0.5 * spend + rng.normal(0, 2)
            rows.append(
                {"region": region, "week": week, "spend": spend, "sales": sales}
            )
    df = pd.DataFrame(rows)
    model = pathmc.fit(
        "sales ~ spend",
        data=df,
        panel={"unit": "region", "time": "week"},
        pooling="partial",
    )
    model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="class")
def lag_panel():
    """Temporal state via lag: sales ~ spend + sales_lag1."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["A", "B", "C"]:
        for week in range(1, 16):
            spend = rng.uniform(10, 50)
            sales = 50 + 0.5 * spend + rng.normal(0, 2)
            rows.append(
                {"region": region, "week": week, "spend": spend, "sales": sales}
            )
    df = pd.DataFrame(rows)
    df = pathmc.add_lags(
        df,
        variables=["sales"],
        lags=1,
        panel={"unit": "region", "time": "week"},
    )
    df = df.dropna().reset_index(drop=True)
    model = pathmc.fit(
        "sales ~ spend + sales_lag1",
        data=df,
        panel={"unit": "region", "time": "week"},
        pooling="partial",
    )
    model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


@pytest.fixture(scope="class")
def adstock_panel():
    """Temporal state via adstock + saturation transform chain."""
    rng = np.random.default_rng(42)
    rows = []
    for region in ["A", "B", "C"]:
        adstocked = 0.0
        for week in range(1, 16):
            tv = rng.uniform(10, 50)
            adstocked = tv + 0.7 * adstocked
            sat = 1 - np.exp(-0.3 * adstocked)
            sales = 50 + 8.0 * sat + 0.1 * week + rng.normal(0, 1.5)
            rows.append(
                {
                    "region": region,
                    "week": week,
                    "tv": tv,
                    "trend": week,
                    "sales": sales,
                }
            )
    df = pd.DataFrame(rows)
    model = pathmc.fit(
        "sales ~ b_tv*logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)"
        " + trend",
        data=df,
        panel={"unit": "region", "time": "week"},
        pooling="partial",
    )
    model.sample(draws=200, tune=200, chains=2, cores=1, random_seed=42)
    return model


# ---------------------------------------------------------------------------
# Tests: no temporal state — all engines must agree tightly
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNoTemporalState:
    """sales ~ spend: no lags or adstock.  All engines identical."""

    def test_mean_mode_engines_agree(self, simple_panel):
        results = {}
        for engine in ("numpy", "batched", "scan"):
            results[engine] = simple_panel.do(
                set={"spend": 30.0},
                simulate_over="time",
                kind="mean",
                panel_engine=engine,
            )

        np_mean = results["numpy"].mean("sales")
        bat_mean = results["batched"].mean("sales")
        scan_mean = results["scan"].mean("sales")

        assert np.isfinite(np_mean)
        assert np_mean == pytest.approx(bat_mean, abs=0.05)
        assert np_mean == pytest.approx(scan_mean, abs=0.05)

    def test_predictive_mode_engines_agree(self, simple_panel):
        results = {}
        for engine in ("numpy", "batched", "scan"):
            results[engine] = simple_panel.do(
                set={"spend": 30.0},
                simulate_over="time",
                kind="predictive",
                panel_engine=engine,
            )

        np_mean = results["numpy"].mean("sales")
        bat_mean = results["batched"].mean("sales")
        scan_mean = results["scan"].mean("sales")

        assert np.isfinite(np_mean)
        # Predictive adds independent noise, so wider tolerance
        assert np_mean == pytest.approx(bat_mean, abs=1.0)
        assert np_mean == pytest.approx(scan_mean, abs=1.0)

    def test_ate_sign_consistent(self, simple_panel):
        """All engines agree on the sign and approximate magnitude of the ATE."""
        ates = {}
        for engine in ("numpy", "batched", "scan"):
            ates[engine] = simple_panel.ate(
                "sales",
                "spend",
                values=(10.0, 40.0),
                simulate_over="time",
                panel_engine=engine,
            )

        for engine in ("numpy", "batched", "scan"):
            assert ates[engine].mean("sales") > 0, f"{engine} ATE should be positive"

        np_ate = ates["numpy"].mean("sales")
        assert np_ate == pytest.approx(ates["batched"].mean("sales"), abs=0.1)
        assert np_ate == pytest.approx(ates["scan"].mean("sales"), abs=0.1)


# ---------------------------------------------------------------------------
# Tests: lag model — numpy and scan per-draw correct; batched approximate
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLagModel:
    """sales ~ spend + sales_lag1: temporal feedback through lags."""

    def test_numpy_scan_agree_tightly(self, lag_panel):
        """numpy and scan are both per-draw correct — should match closely."""
        r_np = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="numpy",
        )
        r_scan = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="scan",
        )
        assert r_np.mean("sales") == pytest.approx(r_scan.mean("sales"), abs=0.1)

    def test_batched_close_to_numpy(self, lag_panel):
        """batched uses mean-field carry — should still be close."""
        r_np = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="numpy",
        )
        r_bat = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="batched",
        )
        assert r_np.mean("sales") == pytest.approx(r_bat.mean("sales"), rel=0.05)

    def test_all_engines_ate_positive(self, lag_panel):
        for engine in ("numpy", "batched", "scan"):
            ate = lag_panel.ate(
                "sales",
                "spend",
                values=(10.0, 40.0),
                simulate_over="time",
                panel_engine=engine,
            )
            assert ate.mean("sales") > 0, f"{engine} ATE should be positive"


# ---------------------------------------------------------------------------
# Tests: adstock + saturation — the hardest case
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAdstockModel:
    """Adstock carry-over and logistic saturation.

    Adstock state accumulates across time steps, making per-draw
    propagation important.  numpy and scan are per-draw correct;
    batched uses mean-field carry.
    """

    def test_numpy_scan_agree_tightly(self, adstock_panel):
        r_np = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="numpy",
        )
        r_scan = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="scan",
        )
        assert r_np.mean("sales") == pytest.approx(r_scan.mean("sales"), abs=0.1)

    def test_batched_close_to_numpy(self, adstock_panel):
        r_np = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="numpy",
        )
        r_bat = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="mean",
            panel_engine="batched",
        )
        assert r_np.mean("sales") == pytest.approx(r_bat.mean("sales"), rel=0.05)

    def test_all_engines_ate_positive(self, adstock_panel):
        """Doubling TV spend increases sales for all engines."""
        for engine in ("numpy", "batched", "scan"):
            ate = adstock_panel.ate(
                "sales",
                "tv",
                values=(15.0, 30.0),
                simulate_over="time",
                panel_engine=engine,
            )
            assert ate.mean("sales") > 0, f"{engine} ATE should be positive"

    def test_all_engines_contrasts_agree(self, adstock_panel):
        """ATE magnitude should be consistent across engines.

        Uses a wider spend range (10→50) to produce a large enough
        ATE for meaningful relative comparison.
        """
        ates = {}
        for engine in ("numpy", "batched", "scan"):
            ates[engine] = adstock_panel.ate(
                "sales",
                "tv",
                values=(10.0, 50.0),
                simulate_over="time",
                panel_engine=engine,
            )

        np_ate = ates["numpy"].mean("sales")
        scan_ate = ates["scan"].mean("sales")
        bat_ate = ates["batched"].mean("sales")

        assert np_ate > 0
        assert np_ate == pytest.approx(scan_ate, abs=0.15)
        assert np_ate == pytest.approx(bat_ate, abs=0.15)


# ---------------------------------------------------------------------------
# Tests: known DGP recovery — verify correct magnitude
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestKnownDGP:
    """Verify engines recover the correct causal effect from a known DGP.

    DGP: sales = 50 + 0.5 * spend + noise(sigma=2)
    True ATE(spend 10→40) = 0.5 * 30 = 15.0
    """

    def test_ate_recovers_true_effect(self, simple_panel):
        for engine in ("numpy", "batched", "scan"):
            ate = simple_panel.ate(
                "sales",
                "spend",
                values=(10.0, 40.0),
                simulate_over="time",
                panel_engine=engine,
            )
            estimated = ate.mean("sales")
            assert estimated == pytest.approx(15.0, abs=5.0), (
                f"{engine}: ATE={estimated:.2f}, expected ~15.0"
            )

    def test_hdi_covers_true_effect(self, simple_panel):
        """94% HDI should contain the true ATE for at least one engine."""
        ate = simple_panel.ate(
            "sales",
            "spend",
            values=(10.0, 40.0),
            simulate_over="time",
            panel_engine="numpy",
        )
        hdi = ate.hdi("sales", prob=0.94)
        assert hdi[0] < 15.0 < hdi[1], (
            f"True ATE 15.0 outside 94% HDI [{hdi[0]:.2f}, {hdi[1]:.2f}]"
        )
