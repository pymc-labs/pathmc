"""Tests for lag() DSL syntax (#13)."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.parse import parse_spec


class TestLagParsing:
    """Parser recognizes lag(var) as a structural term."""

    def test_lag_term_parsed(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        reg = spec.regressions[0]
        lag_term = next(t for t in reg.terms if t.lag_of is not None)
        assert lag_term.lag_of == "sales"
        assert lag_term.variable == "lag(sales)"
        assert lag_term.transform is None

    def test_lag_exogenous(self):
        spec = parse_spec("sales ~ lag(spend)")
        reg = spec.regressions[0]
        assert reg.terms[0].lag_of == "spend"
        assert reg.terms[0].variable == "lag(spend)"

    def test_lag_with_label(self):
        spec = parse_spec("sales ~ b_lag*lag(sales)")
        reg = spec.regressions[0]
        assert reg.terms[0].label == "b_lag"
        assert reg.terms[0].lag_of == "sales"

    def test_lag_rejects_params(self):
        with pytest.raises(Exception, match="does not accept parameters"):
            parse_spec("sales ~ lag(sales, k=2)")

    def test_lag_rejects_nested_transform(self):
        with pytest.raises(Exception, match="plain variable name"):
            parse_spec("sales ~ lag(adstock(sales, decay=theta))")

    def test_lag_mixed_with_other_terms(self):
        spec = parse_spec("sales ~ spend + lag(sales) + trend")
        reg = spec.regressions[0]
        variables = [t.variable for t in reg.terms]
        assert "spend" in variables
        assert "lag(sales)" in variables
        assert "trend" in variables

    def test_non_lag_terms_have_no_lag_of(self):
        spec = parse_spec("sales ~ spend + lag(sales)")
        reg = spec.regressions[0]
        spend_term = next(t for t in reg.terms if t.variable == "spend")
        assert spend_term.lag_of is None


class TestLagRequiresPanel:
    """lag() terms require panel= to be set."""

    def test_lag_without_panel_raises(self):
        rng = np.random.default_rng(42)
        n = 50
        df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n)})
        with pytest.raises(ValueError, match="panel"):
            pathmc.model("Y ~ lag(X)", data=df)


class TestLagCompilation:
    """lag() terms compile into scan models correctly."""

    def test_lag_endogenous_compiles(self):
        """Endogenous self-lag: sales ~ spend + lag(sales)."""
        rng = np.random.default_rng(42)
        rows = []
        for region in ["A", "B"]:
            for week in range(1, 11):
                spend = rng.uniform(10, 50)
                sales = 50 + 0.5 * spend + rng.normal(0, 2)
                rows.append(
                    {"region": region, "week": week, "spend": spend, "sales": sales}
                )
        df = pd.DataFrame(rows)
        model = pathmc.model(
            "sales ~ spend + lag(sales)",
            data=df,
            panel={"unit": "region", "time": "week"},
        )
        assert model.pymc_model is not None
        assert hasattr(model._gen_model, "_pathmc_panel_scan")

    def test_lag_exogenous_compiles(self):
        """Exogenous lag: sales ~ lag(spend)."""
        rng = np.random.default_rng(42)
        rows = []
        for region in ["A", "B"]:
            for week in range(1, 11):
                spend = rng.uniform(10, 50)
                sales = 50 + 0.5 * spend + rng.normal(0, 2)
                rows.append(
                    {"region": region, "week": week, "spend": spend, "sales": sales}
                )
        df = pd.DataFrame(rows)
        model = pathmc.model(
            "sales ~ lag(spend)",
            data=df,
            panel={"unit": "region", "time": "week"},
        )
        assert model.pymc_model is not None
        assert hasattr(model._gen_model, "_pathmc_panel_scan")
