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
"""Gate tests for unit-level counterfactual API (issue #29)."""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.idata import posterior as _posterior


TRUE_A = 0.5
TRUE_B = 0.7
TRUE_C = 0.4

JOE_X = 0.5
JOE_H = 1.0
JOE_Y = 1.5
H_NEW = 2.0
ANALYTICAL_Y_CF = 1.90


@pytest.fixture(scope="module")
def encouragement_model(mock_pymc_sample_module):
    """Encouragement design from docs/examples/02-effect-estimation/counterfactual.qmd."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=n)
    H = TRUE_A * X + rng.normal(size=n)
    Y = TRUE_B * X + TRUE_C * H + rng.normal(size=n)
    df = pd.DataFrame({"X": X, "H": H, "Y": Y})

    model = pathmc.model("H ~ X\nY ~ X + H", data=df)
    model.fit(draws=50, tune=50, chains=2, random_seed=42)

    post = model._idata["posterior"].dataset.copy(deep=True)
    post["beta_H"].loc[{"H_predictors": "Intercept"}] = 0.0
    post["beta_H"].loc[{"H_predictors": "X"}] = TRUE_A
    post["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
    post["beta_Y"].loc[{"Y_predictors": "X"}] = TRUE_B
    post["beta_Y"].loc[{"Y_predictors": "H"}] = TRUE_C
    model._idata["posterior"].dataset = post
    return model


def _manual_counterfactual(
    idata, evidence: dict[str, float], do: dict[str, float]
) -> float:
    """Notebook manual implementation for Joe's counterfactual."""
    post = _posterior(idata)
    a = post["beta_H"].sel(H_predictors="X").values.ravel()
    b = post["beta_Y"].sel(Y_predictors="X").values.ravel()
    c = post["beta_Y"].sel(Y_predictors="H").values.ravel()
    intercept_h = post["beta_H"].sel(H_predictors="Intercept").values.ravel()
    intercept_y = post["beta_Y"].sel(Y_predictors="Intercept").values.ravel()

    u_h = evidence["H"] - intercept_h - a * evidence["X"]
    u_y = evidence["Y"] - intercept_y - b * evidence["X"] - c * evidence["H"]
    y_cf = intercept_y + b * evidence["X"] + c * do["H"] + u_y
    return float(np.mean(y_cf))


class TestCounterfactual:
    def test_joe_counterfactual_matches_analytical(self, encouragement_model):
        result = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": H_NEW},
        )
        assert abs(result.mean("Y") - ANALYTICAL_Y_CF) < 1e-10

    def test_matches_manual_implementation(self, encouragement_model):
        result = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": H_NEW},
        )
        manual = _manual_counterfactual(
            encouragement_model._idata,
            {"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            {"H": H_NEW},
        )
        assert abs(result.mean("Y") - manual) < 1e-10

    def test_differs_from_population_do(self, encouragement_model):
        cf = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": H_NEW},
        )
        pop = encouragement_model.do(set={"H": H_NEW})
        assert cf.mean("Y") > pop.mean("Y")

    def test_hdi(self, encouragement_model):
        result = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": H_NEW},
        )
        interval = result.hdi("Y", prob=0.94)
        assert len(interval) == 2
        assert interval[0] <= interval[1]

    def test_unknown_evidence_raises(self, encouragement_model):
        with pytest.raises(ValueError, match="Unknown variable"):
            encouragement_model.counterfactual(
                evidence={"X": 0.5, "bogus": 1.0},
                do={"H": 2.0},
            )

    def test_do_on_exogenous_raises(self, encouragement_model):
        with pytest.raises(ValueError, match="not endogenous"):
            encouragement_model.counterfactual(
                evidence={"X": 0.5, "H": 1.0, "Y": 1.5},
                do={"X": 1.0},
            )

    def test_empty_do_raises(self, encouragement_model):
        with pytest.raises(ValueError, match="do must contain"):
            encouragement_model.counterfactual(
                evidence={"X": 0.5, "H": 1.0, "Y": 1.5},
                do={},
            )

    def test_partial_evidence_warns(self, encouragement_model):
        with pytest.warns(UserWarning, match="ancestors"):
            encouragement_model.counterfactual(
                evidence={"H": JOE_H, "Y": JOE_Y},
                do={"H": H_NEW},
            )

    def test_requires_fit(self):
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "X": rng.normal(size=n),
            "H": rng.normal(size=n),
            "Y": rng.normal(size=n),
        })
        model = pathmc.model("H ~ X\nY ~ X + H", data=df)
        with pytest.raises(RuntimeError, match="counterfactual"):
            model.counterfactual(evidence={"X": 0.0}, do={"H": 1.0})

    def test_non_gaussian_raises(self, mock_pymc_sample_module):
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(size=n)
        Y = (X + rng.normal(size=n) > 0).astype(int)
        df = pd.DataFrame({"X": X, "Y": Y})
        model = pathmc.model("Y ~ X", data=df, families={"Y": "bernoulli"})
        model.fit(draws=20, tune=20, chains=2, random_seed=0)
        with pytest.raises(ValueError, match="Gaussian"):
            model.counterfactual(evidence={"X": 0.5, "Y": 1.0}, do={"Y": 0.0})
