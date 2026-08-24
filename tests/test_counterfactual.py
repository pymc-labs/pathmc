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

    # Pin draws to the data-generating coefficients: these gate tests verify
    # abduction/prediction arithmetic, not posterior-recovery accuracy.
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

    def test_do_on_exogenous_variable(self, encouragement_model):
        result = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"X": 1.5},
        )
        assert abs(result.mean("Y") - 2.4) < 1e-10

    def test_empty_do_raises(self, encouragement_model):
        with pytest.raises(ValueError, match="do must contain"):
            encouragement_model.counterfactual(
                evidence={"X": 0.5, "H": 1.0, "Y": 1.5},
                do={},
            )

    def test_partial_evidence_raises_by_default(self, encouragement_model):
        with pytest.raises(ValueError, match="must include every model variable"):
            encouragement_model.counterfactual(
                evidence={"H": JOE_H, "Y": JOE_Y},
                do={"H": H_NEW},
            )

    def test_partial_evidence_warns_when_explicit(self, encouragement_model):
        with pytest.warns(UserWarning, match="does not include"):
            encouragement_model.counterfactual(
                evidence={"H": JOE_H, "Y": JOE_Y},
                do={"H": H_NEW},
                allow_partial_evidence=True,
            )

    def test_identity_when_do_matches_observed_value(self, encouragement_model):
        result = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": JOE_H},
        )
        assert abs(result.mean("Y") - JOE_Y) < 1e-10

    def test_same_evidence_counterfactual_contrast(self, encouragement_model):
        evidence = {"X": JOE_X, "H": JOE_H, "Y": JOE_Y}
        counterfactual = encouragement_model.counterfactual(
            evidence=evidence,
            do={"H": H_NEW},
        )
        factual = encouragement_model.counterfactual(evidence=evidence, do={"H": JOE_H})
        contrast = counterfactual - factual
        assert abs(contrast.mean("Y") - TRUE_C) < 1e-10

    def test_warns_on_extrapolation(self, encouragement_model):
        with pytest.warns(UserWarning, match="outside the observed data range"):
            encouragement_model.counterfactual(
                evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
                do={"H": 500.0},
            )

    def test_counterfactual_cannot_contrast_population_do(self, encouragement_model):
        counterfactual = encouragement_model.counterfactual(
            evidence={"X": JOE_X, "H": JOE_H, "Y": JOE_Y},
            do={"H": H_NEW},
        )
        population = encouragement_model.do(set={"H": H_NEW})
        with pytest.raises(ValueError, match="counterfactual result with a do"):
            counterfactual - population

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

    def test_transform_model_raises(self, mock_pymc_sample_module):
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 1, size=100)
        Y = X + rng.normal(size=100)
        model = pathmc.model(
            "Y ~ adstock(X, decay=theta)",
            data=pd.DataFrame({"X": X, "Y": Y}),
        )
        model.fit(draws=20, tune=20, chains=2, random_seed=0)
        with pytest.raises(NotImplementedError, match="transform"):
            model.counterfactual(evidence={"X": 0.5, "Y": 1.0}, do={"X": 0.5})

    def test_lagged_panel_model_raises(self, mock_pymc_sample_module):
        data = pd.DataFrame({
            "region": ["A", "A", "B", "B"],
            "week": [1, 2, 1, 2],
            "X": [0.5, 1.0, 0.5, 1.0],
            "Y": [1.0, 1.5, 1.0, 1.5],
        })
        model = pathmc.model(
            "Y ~ lag(X)",
            data=data,
            panel={"unit": "region", "time": "week"},
        )
        model.fit(draws=20, tune=20, chains=2, random_seed=0)
        with pytest.raises(NotImplementedError, match="panel models"):
            model.counterfactual(evidence={"X": 0.5, "Y": 1.0}, do={"X": 0.0})

    def test_interaction_uses_factual_and_counterfactual_values(
        self, mock_pymc_sample_module
    ):
        rng = np.random.default_rng(0)
        X = rng.normal(size=100)
        Z = rng.normal(size=100)
        Y = X + 0.5 * Z + 0.8 * X * Z + rng.normal(size=100)
        model = pathmc.model(
            "Y ~ X + Z + X:Z", data=pd.DataFrame({"X": X, "Z": Z, "Y": Y})
        )
        model.fit(draws=20, tune=20, chains=2, random_seed=0)

        post = model._idata["posterior"].dataset.copy(deep=True)
        post["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
        post["beta_Y"].loc[{"Y_predictors": "X"}] = 1.0
        post["beta_Y"].loc[{"Y_predictors": "Z"}] = 0.5
        post["beta_Y"].loc[{"Y_predictors": "X:Z"}] = 0.8
        model._idata["posterior"].dataset = post

        result = model.counterfactual(
            evidence={"X": 0.5, "Z": 2.0, "Y": 2.3},
            do={"X": 1.0},
        )
        assert abs(result.mean("Y") - 3.6) < 1e-10


class TestCounterfactualChainAbduction:
    """Independent hand-derived oracle for a 3-hop chain X -> M -> Y -> Z.

    Item 6 of #327 asks whether ``counterfactual()`` performs real
    abduction-action-prediction, or just re-runs the intervention with the
    noise terms reset. The Joe/encouragement fixture above only exercises a
    single downstream hop (do(H) -> Y directly), which can't distinguish
    "abduct once, then plug the intervened value straight into every
    downstream equation" from "abduct once, then *propagate* the
    counterfactual (not the observed) value through every downstream hop".

    This chain does distinguish them: Z's equation depends on Y, and Y's
    *observed* value (0.5) differs from Y's *counterfactual* value under
    do(M=3.0) (2.12). If ``run_counterfactual`` fed the observed Y into Z's
    prediction step instead of the counterfactual Y, Z would come out at the
    observed value (2.0) regardless of the intervention -- the intervention
    would never reach Z at all. The analytic oracle below is hand-derived
    from Pearl's three steps:

        abduction:  u_M = M_obs - a*X_obs
                    u_Y = Y_obs - b*M_obs
                    u_Z = Z_obs - c*Y_obs
        action:     M := 3.0 (do)
        prediction: M_cf = 3.0
                    Y_cf = b*M_cf + u_Y
                    Z_cf = c*Y_cf + u_Z
    """

    TRUE_A = 0.6  # X -> M
    TRUE_B = 0.9  # M -> Y
    TRUE_C = 1.1  # Y -> Z

    EVIDENCE = {"X": 1.0, "M": 1.2, "Y": 0.5, "Z": 2.0}
    DO_M = 3.0

    # Hand-derived oracle values (see docstring above).
    U_M = EVIDENCE["M"] - TRUE_A * EVIDENCE["X"]
    U_Y = EVIDENCE["Y"] - TRUE_B * EVIDENCE["M"]
    U_Z = EVIDENCE["Z"] - TRUE_C * EVIDENCE["Y"]
    M_CF = DO_M
    Y_CF = TRUE_B * M_CF + U_Y
    Z_CF = TRUE_C * Y_CF + U_Z

    # If the implementation instead skipped propagation and fed the
    # *observed* Y into Z's prediction step, Z would come out unchanged from
    # the evidence (no trace of the intervention on M at all).
    Z_IF_ABDUCTION_SKIPPED = EVIDENCE["Z"]

    @pytest.fixture(scope="class")
    def chain_model(self, mock_pymc_sample_module):
        rng = np.random.default_rng(7)
        n = 500
        X = rng.normal(size=n)
        M = self.TRUE_A * X + rng.normal(size=n)
        Y = self.TRUE_B * M + rng.normal(size=n)
        Z = self.TRUE_C * Y + rng.normal(size=n)
        df = pd.DataFrame({"X": X, "M": M, "Y": Y, "Z": Z})

        model = pathmc.model("M ~ X\nY ~ M\nZ ~ Y", data=df)
        model.fit(draws=50, tune=50, chains=2, random_seed=7)

        # Pin draws to the data-generating coefficients: this test is about
        # the abduction/action/prediction arithmetic, not posterior recovery.
        post = model._idata["posterior"].dataset.copy(deep=True)
        post["beta_M"].loc[{"M_predictors": "Intercept"}] = 0.0
        post["beta_M"].loc[{"M_predictors": "X"}] = self.TRUE_A
        post["beta_Y"].loc[{"Y_predictors": "Intercept"}] = 0.0
        post["beta_Y"].loc[{"Y_predictors": "M"}] = self.TRUE_B
        post["beta_Z"].loc[{"Z_predictors": "Intercept"}] = 0.0
        post["beta_Z"].loc[{"Z_predictors": "Y"}] = self.TRUE_C
        model._idata["posterior"].dataset = post
        return model

    def test_pinning_took_effect(self, chain_model):
        """Guard against the idata bracket-path/DataTree-attribute footgun.

        ``model._idata.posterior = ds`` updates the DataTree attribute but
        not the ``idata["posterior"]`` node that ``pathmc.idata.posterior``
        (and therefore ``counterfactual()``) actually reads. Pin -- and
        assert -- via the bracket path, or the rest of this test class would
        silently exercise un-pinned, randomly-sampled coefficients.
        """
        post = _posterior(chain_model._idata)
        assert np.all(post["beta_M"].sel(M_predictors="X").values == self.TRUE_A)
        assert np.all(post["beta_Y"].sel(Y_predictors="M").values == self.TRUE_B)
        assert np.all(post["beta_Z"].sel(Z_predictors="Y").values == self.TRUE_C)

    def test_multi_hop_counterfactual_matches_analytical_oracle(self, chain_model):
        """Z's counterfactual must use the *propagated* Y_cf, not evidence Y."""
        result = chain_model.counterfactual(
            evidence=self.EVIDENCE,
            do={"M": self.DO_M},
        )
        assert abs(result.mean("Y") - self.Y_CF) < 1e-10
        assert abs(result.mean("Z") - self.Z_CF) < 1e-10

    def test_multi_hop_counterfactual_is_not_naive_reintervention(self, chain_model):
        """The intervention on M must actually reach Z through Y.

        A naive implementation that re-ran the intervention on evidence
        values without propagating the counterfactual Y downstream would
        report Z unchanged from its observed value -- this asserts that
        isn't what happens.
        """
        result = chain_model.counterfactual(
            evidence=self.EVIDENCE,
            do={"M": self.DO_M},
        )
        assert abs(result.mean("Z") - self.Z_IF_ABDUCTION_SKIPPED) > 1e-6

    def test_identity_when_do_matches_observed_value(self, chain_model):
        """do(M) at the observed M must reproduce every observed value exactly."""
        result = chain_model.counterfactual(
            evidence=self.EVIDENCE,
            do={"M": self.EVIDENCE["M"]},
        )
        assert abs(result.mean("Y") - self.EVIDENCE["Y"]) < 1e-10
        assert abs(result.mean("Z") - self.EVIDENCE["Z"]) < 1e-10


class TestCounterfactualLatent:
    def test_latent_model_raises(self, mock_pymc_sample_module):
        rng = np.random.default_rng(0)
        n = 100
        X = rng.normal(size=n)
        Y = 0.8 * X + rng.normal(size=n)
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=pd.DataFrame({"X": X, "Y": Y}),
            latent=["M"],
        )
        model.fit(draws=20, tune=20, chains=2, random_seed=0)
        with pytest.raises(NotImplementedError, match="latent variables"):
            model.counterfactual(evidence={"X": 0.5, "Y": 1.0}, do={"M": 0.4})
