"""Panel do() tests.

Tests panel interventions on models with various temporal structures
(no temporal state, lags, adstock). The ``panel_engine`` parameter is
deprecated; all panel do() calls go through the unified scan-compiled
generative model (or cross-sectional fallback for non-temporal models).
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
    """Temporal state via lag() syntax: sales ~ spend + lag(sales)."""
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
        "sales ~ spend + lag(sales)",
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
# Tests: lag model — scan-compiled with lag() syntax
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLagModel:
    """sales ~ spend + lag(sales): temporal feedback through lag() syntax."""

    def test_mean_is_finite(self, lag_panel):
        r = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="mean",
        )
        assert np.isfinite(r.mean("sales"))

    def test_ate_positive(self, lag_panel):
        ate = lag_panel.ate(
            "sales",
            "spend",
            values=(10.0, 40.0),
            simulate_over="time",
        )
        assert ate.mean("sales") > 0, "ATE should be positive"

    def test_hdi_is_valid(self, lag_panel):
        r0 = lag_panel.do(set={"spend": 10.0}, simulate_over="time", kind="mean")
        r1 = lag_panel.do(set={"spend": 40.0}, simulate_over="time", kind="mean")
        contrast = r1 - r0
        hdi = contrast.hdi("sales")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]


# ---------------------------------------------------------------------------
# Tests: adstock + saturation — the hardest case
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAdstockModel:
    """Adstock carry-over and logistic saturation.

    Adstock state accumulates across time steps, making per-draw
    propagation important.
    """

    def test_mean_is_finite(self, adstock_panel):
        r = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="mean",
        )
        assert np.isfinite(r.mean("sales"))

    def test_ate_positive(self, adstock_panel):
        """Doubling TV spend increases sales."""
        ate = adstock_panel.ate(
            "sales",
            "tv",
            values=(15.0, 30.0),
            simulate_over="time",
        )
        assert ate.mean("sales") > 0, "ATE should be positive"

    def test_contrasts_valid(self, adstock_panel):
        """ATE magnitude and HDI are valid."""
        ate = adstock_panel.ate(
            "sales",
            "tv",
            values=(10.0, 50.0),
            simulate_over="time",
        )
        assert ate.mean("sales") > 0
        hdi = ate.hdi("sales")
        assert hdi[0] < hdi[1]


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
        ate = simple_panel.ate(
            "sales",
            "spend",
            values=(10.0, 40.0),
            simulate_over="time",
        )
        estimated = ate.mean("sales")
        assert estimated == pytest.approx(15.0, abs=5.0), (
            f"ATE={estimated:.2f}, expected ~15.0"
        )

    def test_hdi_covers_true_effect(self, simple_panel):
        """94% HDI should contain the true ATE."""
        ate = simple_panel.ate(
            "sales",
            "spend",
            values=(10.0, 40.0),
            simulate_over="time",
        )
        hdi = ate.hdi("sales", prob=0.94)
        assert hdi[0] < 15.0 < hdi[1], (
            f"True ATE 15.0 outside 94% HDI [{hdi[0]:.2f}, {hdi[1]:.2f}]"
        )


# ---------------------------------------------------------------------------
# Tests: predictive mode with temporal state — scan adds noise post-hoc
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPredictiveModeTemporal:
    """Verify predictive mode behaviour when temporal state is present."""

    def test_scan_predictive_on_lag_model(self, lag_panel):
        """Predictive on a scan-compiled lag model should produce valid results."""
        r = lag_panel.do(
            set={"spend": 30.0},
            simulate_over="time",
            kind="predictive",
        )
        assert np.isfinite(r.mean("sales"))

    def test_scan_predictive_on_adstock_model(self, adstock_panel):
        """Predictive on a scan-compiled adstock model should produce valid results."""
        r = adstock_panel.do(
            set={"tv": 30.0},
            simulate_over="time",
            kind="predictive",
        )
        assert np.isfinite(r.mean("sales"))

    def test_scan_mean_no_warning_on_lag_model(self, lag_panel):
        """scan + mean mode should NOT warn even with temporal state."""
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            lag_panel.do(
                set={"spend": 30.0},
                simulate_over="time",
                kind="mean",
            )

        post_hoc_warnings = [
            c
            for c in caught
            if issubclass(c.category, UserWarning) and "post-hoc" in str(c.message)
        ]
        assert len(post_hoc_warnings) == 0


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Validate error messages for malformed inputs."""

    def test_wrong_length_array_raises(self, simple_panel):
        """Array intervention with wrong length should raise ValueError."""
        wrong_length = np.full(5, 30.0)  # panel has 15 time steps
        with pytest.raises(ValueError, match="expected"):
            simple_panel.do(
                set={"spend": wrong_length},
                simulate_over="time",
            )

    def test_unknown_engine_emits_deprecation(self, simple_panel):
        """Unknown panel_engine should emit DeprecationWarning (ignored)."""
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            simple_panel.do(
                set={"spend": 30.0},
                simulate_over="time",
                panel_engine="turbo",
            )
        deprecations = [c for c in caught if issubclass(c.category, DeprecationWarning)]
        assert any("deprecated" in str(c.message) for c in deprecations)

    def test_lag_requires_panel(self):
        """lag() terms without panel= should raise ValueError."""
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n)})
        with pytest.raises(ValueError, match="panel"):
            pathmc.fit("Y ~ lag(X)", data=df)
