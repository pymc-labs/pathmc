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
"""Gate tests for M28: Causal query sugar (ate, cate, prob)."""

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture(scope="module")
def fork_model():
    """A simple fork model: Z -> X, Z -> Y, X -> Y."""
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.normal(size=n)
    X = 0.5 * Z + rng.normal(scale=0.5, size=n)
    Y = 0.4 * X + 0.6 * Z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
    model.fit(draws=300, tune=300, chains=2, random_seed=42)
    return model


class TestATE:
    def test_ate_returns_do_result(self, fork_model):
        ate = fork_model.ate("Y", "X")
        assert hasattr(ate, "mean")
        assert hasattr(ate, "hdi")

    def test_ate_matches_manual_do(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        r0 = fork_model.do(set={"X": 0.0})
        r1 = fork_model.do(set={"X": 1.0})
        manual = r1 - r0
        assert abs(ate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_ate_positive(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        assert ate.mean("Y") > 0

    def test_ate_custom_values(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(-1.0, 2.0))
        # Should be ~3 * 0.4 = 1.2
        assert ate.mean("Y") > 0.5

    def test_ate_hdi(self, fork_model):
        ate = fork_model.ate("Y", "X")
        hdi = ate.hdi("Y", prob=0.94)
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]


class TestCATE:
    def test_cate_returns_do_result(self, fork_model):
        cate = fork_model.cate("Y", "X", condition={"Z": 1.0})
        assert hasattr(cate, "mean")

    def test_cate_matches_manual_do(self, fork_model):
        cate = fork_model.cate("Y", "X", values=(0.0, 1.0), condition={"Z": 1.0})
        r0 = fork_model.do(set={"X": 0.0, "Z": 1.0})
        r1 = fork_model.do(set={"X": 1.0, "Z": 1.0})
        manual = r1 - r0
        assert abs(cate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_cate_without_condition_equals_ate(self, fork_model):
        ate = fork_model.ate("Y", "X", values=(0.0, 1.0))
        cate = fork_model.cate("Y", "X", values=(0.0, 1.0))
        assert abs(ate.mean("Y") - cate.mean("Y")) < 1e-10


class TestProb:
    def test_prob_returns_float(self, fork_model):
        p = fork_model.prob("Y > 0", set={"X": 1.0})
        assert isinstance(p, float)

    def test_prob_between_zero_and_one(self, fork_model):
        p = fork_model.prob("Y > 0", set={"X": 1.0})
        assert 0.0 <= p <= 1.0

    def test_prob_higher_x_higher_prob(self, fork_model):
        p_lo = fork_model.prob("Y > 0", set={"X": -2.0})
        p_hi = fork_model.prob("Y > 0", set={"X": 2.0})
        assert p_hi > p_lo

    def test_prob_always_true(self, fork_model):
        p = fork_model.prob("Y > -1000", set={"X": 0.0})
        assert p == 1.0

    def test_prob_always_false(self, fork_model):
        p = fork_model.prob("Y > 1000", set={"X": 0.0})
        assert p == 0.0
