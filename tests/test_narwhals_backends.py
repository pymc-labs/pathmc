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
        np.testing.assert_allclose(
            results["pandas"].to_numpy(),
            results["polars"].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
