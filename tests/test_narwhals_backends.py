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
"""Cross-backend tests: the public API must accept pandas *and* polars.

narwhals lets pathmc accept any supported DataFrame at the input boundary
and return the *same* backend for data-returning functions
(``simulate``, ``add_lags``, ``PathModel.design``). Posterior-summary
tables (``summary``/``standardized``) come from arviz and are always
pandas regardless of input backend.
"""

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import pathmc

MEDIATION_SPEC = """\
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""


@pytest.fixture(params=["pandas", "polars"])
def backend(request):
    """The DataFrame backend under test."""
    return request.param


def _frame(backend: str, data: dict):
    """Construct a native DataFrame of the requested backend."""
    return pd.DataFrame(data) if backend == "pandas" else pl.DataFrame(data)


def _native_type(backend: str) -> type:
    return pd.DataFrame if backend == "pandas" else pl.DataFrame


@pytest.fixture
def mediation_dict():
    """X -> M -> Y data as a plain dict (backend-agnostic)."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=0.5, size=n)
    return {"X": X, "M": M, "Y": Y}


PANEL_SPEC = "sales ~ lag(spend)"
PANEL_KW = {"unit": "region", "time": "week"}


@pytest.fixture
def panel_dict():
    """Panel data (region x week) with a lagged-spend -> sales structure.

    Exercises the bespoke narwhals panel translation (``with_row_index``,
    ``sort``, ``.over()``).
    """
    rng = np.random.default_rng(42)
    rows = {"region": [], "week": [], "sales": [], "spend": []}
    for region in ("A", "B", "C"):
        spend_prev = 0.0
        for week in range(1, 16):
            spend = rng.uniform(5, 15)
            rows["region"].append(region)
            rows["week"].append(week)
            rows["sales"].append(5.0 + 0.5 * spend_prev + rng.normal(scale=0.5))
            rows["spend"].append(spend)
            spend_prev = spend
    return rows


@pytest.fixture
def binary_treatment_dict():
    """Binary treatment T -> outcome Y, for att/atu/prob counterfactuals."""
    rng = np.random.default_rng(0)
    n = 200
    T = rng.integers(0, 2, size=n).astype(float)
    Y = 1.5 * T + rng.normal(scale=0.5, size=n)
    return {"T": T, "Y": Y}


class TestModelCompilation:
    def test_model_compiles(self, backend, mediation_dict):
        df = _frame(backend, mediation_dict)
        model = pathmc.model(MEDIATION_SPEC, data=df)
        assert model.pymc_model is not None

    def test_design_preserves_backend(self, backend, mediation_dict):
        df = _frame(backend, mediation_dict)
        model = pathmc.model(MEDIATION_SPEC, data=df)
        dm = model.design("Y")
        assert isinstance(dm, _native_type(backend))
        assert "M" in dm.columns and "X" in dm.columns

    def test_missing_endogenous_column_raises(self, backend):
        df = _frame(backend, {"X": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="not found in data columns"):
            pathmc.model("Y ~ X", data=df)


class TestSimulatePreservesBackend:
    def test_simulate_returns_same_backend(self, backend):
        rng = np.random.default_rng(0)
        df = _frame(backend, {"X": rng.normal(size=100)})
        out = pathmc.simulate(
            "Y ~ X",
            data=df,
            params={"beta_Y": [2.0, 0.5], "sigma_Y": 1.0},
            random_seed=42,
        )
        assert isinstance(out, _native_type(backend))
        assert list(out.columns) == ["X", "Y"]

    def test_simulate_reproducible(self, backend):
        rng = np.random.default_rng(0)
        df = _frame(backend, {"X": rng.normal(size=100)})
        params = {"beta_Y": [1.0, 0.5], "sigma_Y": 1.0}
        out1 = pathmc.simulate("Y ~ X", data=df, params=params, random_seed=7)
        out2 = pathmc.simulate("Y ~ X", data=df, params=params, random_seed=7)
        np.testing.assert_allclose(np.asarray(out1["Y"]), np.asarray(out2["Y"]))


class TestAddLagsPreservesBackend:
    def test_add_lags_returns_same_backend(self, backend):
        data = {
            "region": ["A", "A", "A", "B", "B", "B"],
            "week": [1, 2, 3, 1, 2, 3],
            "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
        df = _frame(backend, data)
        with pytest.warns(DeprecationWarning):
            out = pathmc.add_lags(df, ["X"], 1, {"unit": "region", "time": "week"})
        assert isinstance(out, _native_type(backend))
        assert "X_lag1" in out.columns
        # First row of each unit has a null lag; lag equals previous week's X.
        x_lag1 = np.asarray(out["X_lag1"], dtype=float)
        np.testing.assert_array_equal(x_lag1[[1, 2, 4, 5]], [1.0, 2.0, 4.0, 5.0])


@pytest.mark.slow
class TestEndToEndBothBackends:
    def test_fit_and_summary(self, backend, mediation_dict):
        df = _frame(backend, mediation_dict)
        model = pathmc.model(MEDIATION_SPEC, data=df)
        model.fit(draws=200, tune=200, chains=2, random_seed=42, progressbar=False)
        summary = model.summary()
        # Summaries are arviz artifacts: always pandas, regardless of backend.
        assert isinstance(summary, pd.DataFrame)
        standardized = model.standardized()
        assert isinstance(standardized, pd.DataFrame)

    def test_fit_recovers_parameters_consistently(self, mediation_dict):
        """pandas and polars inputs must yield the same posterior."""
        results = {}
        for backend in ("pandas", "polars"):
            df = _frame(backend, mediation_dict)
            model = pathmc.model(MEDIATION_SPEC, data=df)
            model.fit(draws=200, tune=200, chains=2, random_seed=42, progressbar=False)
            results[backend] = model.summary()["mean"]
        # pandas and polars feed identical numpy arrays into PyMC, so the
        # posteriors should match closely; 1e-6 is robust against any future
        # backend-path float divergence while still catching real drift.
        np.testing.assert_allclose(
            results["pandas"].to_numpy(),
            results["polars"].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


class TestPanelBothBackends:
    """Panel models exercise the bespoke narwhals translation for both backends."""

    def test_panel_model_compiles(self, backend, panel_dict):
        df = _frame(backend, panel_dict)
        model = pathmc.model(PANEL_SPEC, data=df, panel=PANEL_KW, pooling="partial")
        assert model.pymc_model is not None

    @pytest.mark.slow
    def test_panel_fit_and_time_do(self, backend, panel_dict):
        df = _frame(backend, panel_dict)
        model = pathmc.model(PANEL_SPEC, data=df, panel=PANEL_KW, pooling="partial")
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        result = model.do(set={"spend": 10.0}, simulate_over="time", kind="mean")
        assert np.isfinite(result.mean("sales"))


class TestIdentificationBothBackends:
    """test_implications() reads observed data; must accept polars input."""

    def test_implications_runs(self, backend):
        # Chain X -> M -> Y (no direct X -> Y edge) implies X ⊥⊥ Y | M.
        rng = np.random.default_rng(1)
        n = 200
        X = rng.normal(size=n)
        M = 0.7 * X + rng.normal(scale=0.5, size=n)
        Y = 0.6 * M + rng.normal(scale=0.5, size=n)
        df = _frame(backend, {"X": X, "M": M, "Y": Y})
        model = pathmc.model("M ~ X\nY ~ M", data=df)
        result = model.test_implications()
        # Result table is an arviz/pandas artifact regardless of input backend.
        assert isinstance(result.to_dataframe(), pd.DataFrame)


@pytest.mark.slow
class TestCounterfactualsBothBackends:
    """att/atu/prob access self._data; verify they work with polars input."""

    def test_att_atu_prob(self, backend, binary_treatment_dict):
        df = _frame(backend, binary_treatment_dict)
        model = pathmc.model("Y ~ T", data=df)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        att = model.att(outcome="Y", treatment="T")
        atu = model.atu(outcome="Y", treatment="T")
        assert np.isfinite(att.mean("Y"))
        assert np.isfinite(atu.mean("Y"))
        p = model.prob("Y > 0", set={"T": 1.0})
        assert 0.0 <= p <= 1.0


@pytest.mark.slow
class TestMissingDataBothBackends:
    """Regression coverage for the att/atu subgroup-mean NaN handling.

    The subgroup empirical-integration path slices an exogenous covariate
    and fills it with the subgroup mean. That mean must skip nulls/NaN
    (matching the historical pandas ``skipna=True`` behavior) so a single
    missing value does not poison the whole result. An exogenous root
    covariate cannot carry NaN through fitting (only endogenous variables
    get the masked-array path), so we fit on complete data and then inject
    a missing value into the stored covariate before exercising the
    subgroup fill via ``att()``/``atu()``.
    """

    def test_att_atu_skip_missing_covariate(self, backend):
        rng = np.random.default_rng(0)
        n = 200
        T = rng.integers(0, 2, size=n).astype(float)
        X = rng.normal(size=n)
        Y = 1.5 * T + 0.7 * X + rng.normal(scale=0.5, size=n)

        df = _frame(backend, {"T": T, "X": X, "Y": Y})
        model = pathmc.model("Y ~ T + X", data=df)
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)

        treated_rows = np.where(np.isclose(T, 1.0))[0]
        X_missing = X.copy()
        X_missing[treated_rows[0]] = np.nan
        model._data = nw.from_native(_frame(backend, {"T": T, "X": X_missing, "Y": Y}))

        att = model.att(outcome="Y", treatment="T")
        atu = model.atu(outcome="Y", treatment="T")

        # The outcome contrast and the filled covariate must both stay
        # finite: the buggy path propagated the injected NaN into the
        # subgroup fill, turning these into NaN.
        assert np.isfinite(att.mean("Y"))
        assert np.isfinite(atu.mean("Y"))
        assert np.all(np.isfinite(att.draws("X")))
        assert np.all(np.isfinite(atu.draws("X")))
