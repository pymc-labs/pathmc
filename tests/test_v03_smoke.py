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
"""M24 gate tests: v0.3 integration smoke tests.

End-to-end tests combining transforms, new families, and PPC.
All tests use actual MCMC sampling with minimal draws.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture(scope="module")
def mmm_transform_data():
    """Panel MMM data with known adstock + saturation structure.

    True DGP:
      adstocked_tv = adstock(tv, decay=0.7)
      saturated_tv = 1 - exp(-0.3 * adstocked_tv)
      sales = intercept[region] + 2.5 * saturated_tv + noise
    """
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East"]
    n_weeks = 25
    true_intercepts = {"North": 50, "South": 60, "East": 55}
    rows = []
    for region in regions:
        adstocked = 0.0
        for week in range(1, n_weeks + 1):
            tv = rng.uniform(5, 30)
            adstocked = tv + 0.7 * adstocked
            saturated = 1 - np.exp(-0.3 * adstocked)
            sales = true_intercepts[region] + 2.5 * saturated + rng.normal(scale=0.5)
            rows.append({
                "region": region,
                "week": week,
                "tv": tv,
                "sales": sales,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def poisson_dgp_data():
    """Poisson count data with known DGP: Y ~ Poisson(exp(1 + 0.3*X))."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    Y = rng.poisson(np.exp(1.0 + 0.3 * X)).astype(float)
    return pd.DataFrame({"X": X, "Y": Y})


@pytest.mark.slow
class TestMMMWithTransforms:
    """Full MMM pipeline: adstock + saturation + panel + do() + PPC."""

    @pytest.fixture(scope="class")
    def mmm_model(self, mmm_transform_data):
        spec = (
            "sales ~ b_tv*logistic_saturation(adstock(tv, decay=theta_tv), lam=lam_tv)"
        )
        model = pathmc.model(
            spec,
            data=mmm_transform_data,
            panel={"unit": "region", "time": "week"},
            pooling="partial",
        )
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        return model

    def test_mmm_pipeline(self, mmm_model):
        summary = mmm_model.summary()
        assert summary is not None
        assert len(summary) > 0

    def test_mmm_do_counterfactual(self, mmm_model):
        r_low = mmm_model.do(set={"tv": 5.0}, simulate_over="time", kind="mean")
        r_high = mmm_model.do(set={"tv": 25.0}, simulate_over="time", kind="mean")
        contrast = r_high - r_low
        assert contrast.mean("sales") > 0

    def test_mmm_predict(self, mmm_model):
        idata = mmm_model.predict()
        assert hasattr(idata, "posterior_predictive")


@pytest.mark.slow
class TestPoissonSmoke:
    """Poisson count model end-to-end."""

    def test_poisson_pipeline(self, poisson_dgp_data):
        model = pathmc.model("Y ~ X", data=poisson_dgp_data, families={"Y": "poisson"})
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        r0 = model.do(set={"X": 0.0})
        r1 = model.do(set={"X": 1.0})
        ate = r1 - r0
        assert ate.mean("Y") > 0


@pytest.mark.slow
class TestStudentTSmoke:
    """StudentT model end-to-end."""

    def test_studentt_pipeline(self):
        rng = np.random.default_rng(42)
        n = 150
        X = rng.normal(size=n)
        Y = 1.0 + 0.5 * X + rng.standard_t(df=4, size=n) * 0.5
        df = pd.DataFrame({"X": X, "Y": Y})

        model = pathmc.model("Y ~ X", data=df, families={"Y": "studentt"})
        model.fit(draws=200, tune=200, chains=2, cores=1, random_seed=42)
        summary = model.summary()
        assert any("nu" in str(idx) for idx in summary.index)

        r0 = model.do(set={"X": 0.0})
        r1 = model.do(set={"X": 1.0})
        ate = r1 - r0
        assert 0.0 < ate.mean("Y") < 2.0


@pytest.mark.slow
class TestTransformParameterRecovery:
    """Transform parameters should be in reasonable range for known DGP."""

    def test_adstock_decay_recovered(self):
        """True decay=0.7 — posterior mean should be in (0.3, 0.95)."""
        rng = np.random.default_rng(42)
        n = 80
        x = rng.uniform(0, 10, size=n)
        adstocked = np.zeros(n)
        for t in range(n):
            adstocked[t] = x[t] + (0.7 * adstocked[t - 1] if t > 0 else 0)
        y = 2.0 + 0.5 * adstocked + rng.normal(scale=1, size=n)
        df = pd.DataFrame({"X": x, "Y": y})

        model = pathmc.model("Y ~ adstock(X, decay=theta)", data=df)
        model.fit(draws=500, tune=500, chains=2, cores=1, random_seed=42)
        import arviz as az

        summary = az.summary(model._idata, var_names=["theta"], round_to="none")
        theta_mean = summary["mean"].iloc[0]
        assert 0.3 < theta_mean < 0.95, f"theta_mean={theta_mean}"
