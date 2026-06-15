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
"""Tests for ATT/ATU: canonical subgroup-aware treatment effects.

TestAttAtuAPI contains fast tests verifying the interface surface.
TestAttAtuSemantics contains slow tests verifying estimand correctness.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import pathmc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFOUNDED_SPEC = "Y ~ T + X"
INTERACTION_SPEC = "Y ~ T + X + T:X"


def _make_binary_treatment_data(rng, n=500):
    """Confounded binary treatment with known DGP.

    X -> T (confounding), X -> Y, T -> Y.
    True causal effect of T on Y = 0.5 (constant, no interaction).
    """
    X = rng.normal(size=n)
    T = (rng.uniform(size=n) < expit(0.8 * X)).astype(float)
    Y = 0.5 * T + 0.3 * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "T": T, "Y": Y})


def _make_interaction_data(rng, n=500):
    """Binary treatment with effect modification (interaction).

    T -> Y effect = 0.5 + 0.4*X, so ATT != ATU when E[X|T=1] != E[X|T=0].
    """
    X = rng.normal(size=n)
    T = (rng.uniform(size=n) < expit(0.8 * X)).astype(float)
    Y = 0.5 * T + 0.3 * X + 0.4 * T * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "T": T, "Y": Y})


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_data(rng):
    return _make_binary_treatment_data(rng)


@pytest.fixture
def interaction_data(rng):
    return _make_interaction_data(rng)


@pytest.fixture(scope="module")
def fitted_binary():
    """Binary treatment model fitted with minimal draws."""
    rng = np.random.default_rng(42)
    df = _make_binary_treatment_data(rng)
    model = pathmc.model(CONFOUNDED_SPEC, data=df)
    model.fit(draws=300, tune=300, chains=2, random_seed=42)
    return model


@pytest.fixture(scope="module")
def fitted_interaction():
    """Interaction model fitted with minimal draws."""
    rng = np.random.default_rng(42)
    df = _make_interaction_data(rng)
    model = pathmc.model(INTERACTION_SPEC, data=df)
    model.fit(draws=300, tune=300, chains=2, random_seed=42)
    return model


# ---------------------------------------------------------------------------
# Fast tests: API surface
# ---------------------------------------------------------------------------


class TestAttAtuAPI:
    def test_att_method_exists(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        assert hasattr(model, "att")
        assert callable(model.att)

    def test_atu_method_exists(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        assert hasattr(model, "atu")
        assert callable(model.atu)

    def test_att_before_sampling_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.att("Y", "T")

    def test_atu_before_sampling_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.atu("Y", "T")

    def test_att_no_matching_rows_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        model._idata = True  # bypass sampling check
        with pytest.raises(
            ValueError, match="No observations.*Check the treated_value parameter"
        ):
            model.att("Y", "T", treated_value=999.0)

    def test_atu_no_matching_rows_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        model._idata = True  # bypass sampling check
        with pytest.raises(
            ValueError, match="No observations.*Check the untreated_value parameter"
        ):
            model.atu("Y", "T", untreated_value=999.0)


# ---------------------------------------------------------------------------
# Slow tests: estimand semantics
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAttAtuSemantics:
    """Verify ATT/ATU return finite, reasonable values."""

    def test_att_returns_do_result(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        assert hasattr(att, "mean")
        assert hasattr(att, "hdi")

    def test_atu_returns_do_result(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        assert hasattr(atu, "mean")
        assert hasattr(atu, "hdi")

    def test_att_finite(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        assert np.isfinite(att.mean("Y"))

    def test_atu_finite(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        assert np.isfinite(atu.mean("Y"))

    def test_att_hdi(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        hdi = att.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]

    def test_atu_hdi(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        hdi = atu.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]


@pytest.mark.slow
class TestAttAtuLinearConsistency:
    """In a linear model without interactions, ATT ≈ ATU ≈ ATE."""

    def test_att_close_to_ate(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T")
        att = fitted_binary.att("Y", "T")
        assert abs(att.mean("Y") - ate.mean("Y")) < 0.1

    def test_atu_close_to_ate(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T")
        atu = fitted_binary.atu("Y", "T")
        assert abs(atu.mean("Y") - ate.mean("Y")) < 0.1

    def test_att_close_to_atu(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        atu = fitted_binary.atu("Y", "T")
        assert abs(att.mean("Y") - atu.mean("Y")) < 0.1


@pytest.mark.slow
class TestAttAtuInteraction:
    """With effect modification, ATT and ATU should diverge."""

    def test_att_differs_from_atu(self, fitted_interaction):
        att = fitted_interaction.att("Y", "T")
        atu = fitted_interaction.atu("Y", "T")
        # ATT > ATU because treated units have higher X on average
        # and the interaction coefficient is positive
        assert att.mean("Y") > atu.mean("Y")

    def test_att_positive(self, fitted_interaction):
        att = fitted_interaction.att("Y", "T")
        assert att.mean("Y") > 0

    def test_atu_positive(self, fitted_interaction):
        atu = fitted_interaction.atu("Y", "T")
        assert atu.mean("Y") > 0


@pytest.mark.slow
class TestNonDefaultCoding:
    """Treatment coded as {-1, +1} instead of {0, 1}."""

    @pytest.fixture(scope="class")
    def fitted_alt_coding(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(size=n)
        T = np.where(rng.uniform(size=n) < expit(0.8 * X), 1.0, -1.0)
        Y = 0.5 * T + 0.3 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "T": T, "Y": Y})
        model = pathmc.model("Y ~ T + X", data=df)
        model.fit(draws=300, tune=300, chains=2, random_seed=42)
        return model

    def test_att_with_custom_treated_value(self, fitted_alt_coding):
        att = fitted_alt_coding.att("Y", "T", treated_value=1.0)
        assert np.isfinite(att.mean("Y"))

    def test_atu_with_custom_untreated_value(self, fitted_alt_coding):
        atu = fitted_alt_coding.atu("Y", "T", untreated_value=-1.0)
        assert np.isfinite(atu.mean("Y"))

    def test_custom_values_tuple(self, fitted_alt_coding):
        att = fitted_alt_coding.att("Y", "T", values=(-1.0, 1.0), treated_value=1.0)
        assert att.mean("Y") > 0


@pytest.mark.slow
class TestKindVariants:
    """Both kind='mean' and kind='predictive' work."""

    def test_att_kind_mean(self, fitted_binary):
        att = fitted_binary.att("Y", "T", kind="mean")
        assert np.isfinite(att.mean("Y"))

    def test_att_kind_predictive(self, fitted_binary):
        att = fitted_binary.att("Y", "T", kind="predictive")
        assert np.isfinite(att.mean("Y"))

    def test_atu_kind_mean(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T", kind="mean")
        assert np.isfinite(atu.mean("Y"))

    def test_atu_kind_predictive(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T", kind="predictive")
        assert np.isfinite(atu.mean("Y"))

    def test_mean_and_predictive_similar(self, fitted_binary):
        att_mean = fitted_binary.att("Y", "T", kind="mean")
        att_pred = fitted_binary.att("Y", "T", kind="predictive")
        assert abs(att_mean.mean("Y") - att_pred.mean("Y")) < 0.3


@pytest.mark.slow
class TestExistingBehaviorUnchanged:
    """Existing ate()/cate() behavior is not affected."""

    def test_ate_unchanged(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T", values=(0.0, 1.0))
        r0 = fitted_binary.do(set={"T": 0.0})
        r1 = fitted_binary.do(set={"T": 1.0})
        manual = r1 - r0
        assert abs(ate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_cate_unchanged(self, fitted_binary):
        cate = fitted_binary.cate("Y", "T", values=(0.0, 1.0), condition={"X": 1.0})
        r0 = fitted_binary.do(set={"T": 0.0, "X": 1.0})
        r1 = fitted_binary.do(set={"T": 1.0, "X": 1.0})
        manual = r1 - r0
        assert abs(cate.mean("Y") - manual.mean("Y")) < 1e-10
