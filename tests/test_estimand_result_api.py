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
"""Characterization tests for the public API of :class:`EstimandResult`.

These lock in the *current* behavior of methods that were previously
untested: ``summary()``, ``prob()``, ``__float__``, ``__sub__``,
``from_contrast``, default-to-outcome draw resolution, and HDI widening.
They are written against the existing ``dict[str, np.ndarray]`` storage and
must continue to pass after the planned migration to an xarray-backed
internal store (see issue #319).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathmc.simulate import DoResult, EstimandResult


# ---------------------------------------------------------------------------
# Hand-built fixtures (no MCMC) — mirror tests/test_reprs.py style
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N_SAMPLES = 1000
N_CHAINS = 1
N_DRAWS = N_SAMPLES


def _make_draws(loc: float, scale: float = 0.1) -> np.ndarray:
    return RNG.normal(loc=loc, scale=scale, size=N_SAMPLES)


@pytest.fixture
def estimand_result() -> EstimandResult:
    """An ATE of X on Y with positive effect draws."""
    return EstimandResult(
        values={"Y": _make_draws(loc=0.5), "X": _make_draws(loc=1.0)},
        outcome="Y",
        treatment="X",
        estimand="ATE",
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
    )


@pytest.fixture
def estimand_result_negative() -> EstimandResult:
    """An ATE with negative effect draws (for prob('< 0') tests)."""
    return EstimandResult(
        values={"Y": _make_draws(loc=-0.5)},
        outcome="Y",
        treatment="X",
        estimand="ATE",
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
    )


@pytest.fixture
def do_contrast() -> DoResult:
    """A DoResult contrast to exercise from_contrast()."""
    return DoResult(
        values={"Y": _make_draws(loc=0.5), "X": _make_draws(loc=1.0)},
        n_chains=N_CHAINS,
        n_draws=N_DRAWS,
    )


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


class TestSummary:
    """EstimandResult.summary() returns a tidy one-row DataFrame."""

    def test_returns_dataframe(self, estimand_result):
        df = estimand_result.summary()
        assert isinstance(df, pd.DataFrame)

    def test_single_row(self, estimand_result):
        df = estimand_result.summary()
        assert df.shape[0] == 1

    def test_expected_columns(self, estimand_result):
        df = estimand_result.summary()
        assert list(df.columns) == [
            "outcome",
            "treatment",
            "mean",
            "sd",
            "hdi_3%",
            "hdi_97%",
            "p(>0)",
        ]

    def test_index_is_estimand_label(self, estimand_result):
        df = estimand_result.summary()
        assert df.index.name == "estimand"
        assert df.index.tolist() == ["ATE"]

    def test_mean_column_matches_mean_method(self, estimand_result):
        df = estimand_result.summary()
        assert abs(df["mean"].iloc[0] - estimand_result.mean()) < 1e-12

    def test_outcome_and_treatment_columns(self, estimand_result):
        df = estimand_result.summary()
        assert df["outcome"].iloc[0] == "Y"
        assert df["treatment"].iloc[0] == "X"

    def test_p_gt_zero_column(self, estimand_result):
        df = estimand_result.summary()
        expected = float(np.mean(estimand_result.draws() > 0))
        assert abs(df["p(>0)"].iloc[0] - expected) < 1e-12

    def test_hdi_columns_bracket_mean(self, estimand_result):
        df = estimand_result.summary()
        assert df["hdi_3%"].iloc[0] < df["hdi_97%"].iloc[0]


# ---------------------------------------------------------------------------
# prob()
# ---------------------------------------------------------------------------


class TestProb:
    """EstimandResult.prob() parses comparison expressions."""

    def test_returns_float(self, estimand_result):
        assert isinstance(estimand_result.prob("> 0"), float)

    def test_between_zero_and_one(self, estimand_result):
        p = estimand_result.prob("> 0")
        assert 0.0 <= p <= 1.0

    def test_gt_zero_matches_manual(self, estimand_result):
        expected = float(np.mean(estimand_result.draws() > 0))
        assert abs(estimand_result.prob("> 0") - expected) < 1e-12

    def test_gt_zero(self, estimand_result):
        # draws centered at 0.5, so most are > 0
        assert estimand_result.prob("> 0") > 0.9

    def test_lt_zero(self, estimand_result):
        expected = float(np.mean(estimand_result.draws() < 0))
        assert abs(estimand_result.prob("< 0") - expected) < 1e-12

    def test_ge_one(self, estimand_result):
        expected = float(np.mean(estimand_result.draws() >= 1.0))
        assert abs(estimand_result.prob(">= 1") - expected) < 1e-12

    def test_lt_negative_half(self, estimand_result):
        expected = float(np.mean(estimand_result.draws() < -0.5))
        assert abs(estimand_result.prob("< -0.5") - expected) < 1e-12

    def test_always_true(self, estimand_result):
        assert estimand_result.prob("> -1e9") == 1.0

    def test_always_false(self, estimand_result):
        assert estimand_result.prob("> 1e9") == 0.0

    def test_complement(self, estimand_result):
        # P(> 0) + P(< 0) == 1 for continuous draws (no mass exactly at 0)
        total = estimand_result.prob("> 0") + estimand_result.prob("< 0")
        assert abs(total - 1.0) < 1e-9

    def test_malformed_expression_raises(self, estimand_result):
        with pytest.raises(ValueError, match="Could not parse"):
            estimand_result.prob("foo bar")

    def test_syntax_error_raises(self, estimand_result):
        # ">> 0" is valid Python (right-shift) but fails at runtime on floats;
        # the current handler only catches SyntaxError/NameError, so TypeError
        # propagates. Characterizing actual behavior.
        with pytest.raises(TypeError):
            estimand_result.prob(">> 0")

    def test_prob_on_other_var(self, estimand_result):
        # prob() can target a non-outcome variable
        expected = float(np.mean(estimand_result.draws("X") > 0))
        assert abs(estimand_result.prob("> 0", var="X") - expected) < 1e-12


# ---------------------------------------------------------------------------
# __float__()
# ---------------------------------------------------------------------------


class TestFloat:
    """float(estimand_result) returns the posterior mean of the outcome."""

    def test_equals_mean(self, estimand_result):
        assert float(estimand_result) == pytest.approx(estimand_result.mean())

    def test_is_python_float(self, estimand_result):
        assert isinstance(float(estimand_result), float)


# ---------------------------------------------------------------------------
# __sub__()
# ---------------------------------------------------------------------------


class TestSub:
    """Subtracting two EstimandResults preserves the outcome and shapes."""

    def test_draws_shape_preserved(self, estimand_result):
        other = EstimandResult(
            values={"Y": _make_draws(loc=0.1), "X": _make_draws(loc=0.0)},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        diff = estimand_result - other
        assert diff.draws().shape == (N_SAMPLES,)

    def test_mean_is_difference(self, estimand_result):
        other = EstimandResult(
            values={"Y": _make_draws(loc=0.1), "X": _make_draws(loc=0.0)},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        diff = estimand_result - other
        assert abs(diff.mean() - (estimand_result.mean() - other.mean())) < 1e-12

    def test_outcome_preserved(self, estimand_result):
        other = EstimandResult(
            values={"Y": _make_draws(loc=0.1), "X": _make_draws(loc=0.0)},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        diff = estimand_result - other
        assert diff.outcome == "Y"
        assert diff.treatment == "X"


# ---------------------------------------------------------------------------
# from_contrast()
# ---------------------------------------------------------------------------


class TestFromContrast:
    """from_contrast() wraps a DoResult contrast as an EstimandResult."""

    def test_outcome_and_treatment_set(self, do_contrast):
        er = EstimandResult.from_contrast(
            do_contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        assert er.outcome == "Y"
        assert er.treatment == "X"

    def test_draws_match_contrast(self, do_contrast):
        er = EstimandResult.from_contrast(
            do_contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        np.testing.assert_array_equal(er.draws(), do_contrast.draws("Y"))

    def test_mean_matches_contrast(self, do_contrast):
        er = EstimandResult.from_contrast(
            do_contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        assert abs(er.mean() - do_contrast.mean("Y")) < 1e-12

    def test_other_var_accessible(self, do_contrast):
        er = EstimandResult.from_contrast(
            do_contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        np.testing.assert_array_equal(er.draws("X"), do_contrast.draws("X"))


# ---------------------------------------------------------------------------
# draws(var=None) default-to-outcome resolution
# ---------------------------------------------------------------------------


class TestDrawsDefault:
    """draws() with no arg defaults to the outcome variable."""

    def test_default_equals_outcome(self, estimand_result):
        np.testing.assert_array_equal(
            estimand_result.draws(), estimand_result.draws("Y")
        )

    def test_explicit_other_var(self, estimand_result):
        # X is the treatment variable, also present in the contrast
        assert estimand_result.draws("X").shape == (N_SAMPLES,)

    def test_unknown_var_raises(self, estimand_result):
        with pytest.raises(KeyError, match="Unknown variable"):
            estimand_result.draws("nope")


# ---------------------------------------------------------------------------
# hdi(prob=...) widening
# ---------------------------------------------------------------------------


class TestHdiWidening:
    """Higher prob yields a wider HDI."""

    def test_higher_prob_widens_interval(self, estimand_result):
        narrow = estimand_result.hdi(prob=0.5)
        wide = estimand_result.hdi(prob=0.99)
        narrow_width = narrow[1] - narrow[0]
        wide_width = wide[1] - wide[0]
        assert wide_width > narrow_width

    def test_hdi_brackets_mean(self, estimand_result):
        hdi = estimand_result.hdi()
        m = estimand_result.mean()
        assert hdi[0] <= m <= hdi[1]
