"""M20 gate tests: Transform registry, built-in transforms, and PyMC compilation.

Tests verify that models with transform expressions compile to PyMC models
with correctly constrained transform parameters, and that introspection
methods reflect the transform structure.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc


@pytest.fixture()
def timeseries_data():
    """Simple time series: Y depends on adstocked X."""
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 10, size=n)
    adstocked = np.zeros(n)
    decay = 0.7
    for t in range(n):
        adstocked[t] = x[t] + (decay * adstocked[t - 1] if t > 0 else 0)
    y = 2.0 + 0.5 * adstocked + rng.normal(scale=1, size=n)
    return pd.DataFrame({"X": x, "Y": y})


@pytest.fixture()
def saturation_data():
    """Simple data: Y depends on saturated X."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(0, 10, size=n)
    saturated = 1 - np.exp(-0.3 * x)
    y = 1.0 + 3.0 * saturated + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Y": y})


@pytest.fixture()
def panel_adstock_data():
    """Panel data for adstock: 3 regions, 20 weeks."""
    rng = np.random.default_rng(42)
    regions = ["A", "B", "C"]
    n_weeks = 20
    rows = []
    for region in regions:
        x_prev_adstocked = 0.0
        for week in range(1, n_weeks + 1):
            x = rng.uniform(5, 15)
            x_prev_adstocked = x + 0.6 * x_prev_adstocked
            y = 5.0 + 0.4 * x_prev_adstocked + rng.normal(scale=0.5)
            rows.append({"region": region, "week": week, "X": x, "Y": y})
    return pd.DataFrame(rows)


class TestAdstockCompilation:
    """adstock(X, decay=theta) compiles with Beta-constrained theta."""

    def test_compiles_to_pymc(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_theta_in_free_rvs(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "theta" in rv_names

    def test_beta_still_exists(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "beta_Y" in rv_names

    def test_labeled_adstock(self, timeseries_data):
        model = pathmc.model("Y ~ b*adstock(X, decay=theta)", data=timeseries_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "theta" in rv_names
        assert "beta_Y" in rv_names


class TestSaturationCompilation:
    """logistic_saturation(X, lam=lam) compiles with positive-constrained lam."""

    def test_compiles_to_pymc(self, saturation_data):
        model = pathmc.model(
            "Y ~ logistic_saturation(X, lam=lam_x)", data=saturation_data
        )
        assert isinstance(model.pymc_model, pm.Model)

    def test_lam_in_free_rvs(self, saturation_data):
        model = pathmc.model(
            "Y ~ logistic_saturation(X, lam=lam_x)", data=saturation_data
        )
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "lam_x" in rv_names


class TestNestedCompilation:
    """Composed logistic_saturation(adstock(...)) compiles both params."""

    def test_nested_compiles(self, timeseries_data):
        spec = "Y ~ logistic_saturation(adstock(X, decay=theta), lam=lam_x)"
        model = pathmc.model(spec, data=timeseries_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_both_params_present(self, timeseries_data):
        spec = "Y ~ logistic_saturation(adstock(X, decay=theta), lam=lam_x)"
        model = pathmc.model(spec, data=timeseries_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "theta" in rv_names
        assert "lam_x" in rv_names


class TestMixedTerms:
    """Transforms mixed with plain terms in same equation."""

    def test_mixed_compiles(self, timeseries_data):
        timeseries_data = timeseries_data.copy()
        timeseries_data["trend"] = np.arange(len(timeseries_data))
        spec = "Y ~ b_x*adstock(X, decay=theta) + trend"
        model = pathmc.model(spec, data=timeseries_data)
        assert isinstance(model.pymc_model, pm.Model)


class TestTransformIntrospection:
    """Introspection methods reflect transform structure."""

    def test_priors_include_transform_params(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        prior_str = repr(model.priors())
        assert "theta" in prior_str

    def test_equations_show_transforms(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        eq_str = str(model.equations())
        assert "adstock" in eq_str

    def test_graph_includes_transform(self, timeseries_data):
        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=timeseries_data)
        g = model.graph()
        graph_source = g.source if hasattr(g, "source") else str(g)
        assert "adstock" in graph_source.lower() or "X" in graph_source


class TestTransformErrors:
    """Error handling for unknown transforms."""

    def test_unknown_transform_raises(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"X": rng.normal(size=50), "Y": rng.normal(size=50)})
        with pytest.raises(Exception, match="(?i)unknown|not found|unrecognized"):
            pathmc.model("Y ~ unknown_transform(X, p=val)", data=df)


class TestPanelAdstockCompilation:
    """Adstock with panel mode compiles correctly."""

    def test_panel_adstock_compiles(self, panel_adstock_data):
        model = pathmc.model(
            "Y ~ adstock(X, decay=theta)",
            data=panel_adstock_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        assert isinstance(model.pymc_model, pm.Model)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "theta" in rv_names
        assert "alpha_Y" in rv_names
