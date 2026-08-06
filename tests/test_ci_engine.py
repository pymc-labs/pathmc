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
"""Tests for the shared partial-correlation CI engine (#276 / #278).

The characterization classes pin the behavior of the two pre-existing
implementations — ``identify._partial_correlation_test`` and
``falsify._PartialCorrelationTester`` — on well-conditioned inputs, so the
engine unification can demonstrate "no behavior change for well-conditioned
inputs" against a recorded baseline rather than by algebra alone.
"""

import dataclasses
import warnings

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc._ci import CIResult, partial_correlation_ci
from pathmc.falsify import _PartialCorrelationTester
from pathmc.identify import _partial_correlation_test

# Values recorded from both implementations at aa80e4d (pre-unification).
# The two implementations agree bitwise on these inputs; the pins use a
# relative tolerance only to absorb BLAS variation across platforms.
WELL_CONDITIONED_CASES = {
    # (x, y, conditioners): (partial_r, p_value, n_obs)
    ("X", "Y", ()): (0.7316996316560405, 8.494760568123121e-35, 200),
    ("X", "Y", ("M",)): (0.2526391622765326, 0.00031816897908150133, 200),
    ("X", "Y", ("Z1", "Z2")): (0.47115390801105606, 2.463113409473222e-12, 200),
    ("M", "Z2", ("X",)): (-0.017153803352143768, 0.8099601363017269, 200),
}

NAN_CASE_EXPECTED = (0.24943756765617675, 0.0004081110120031142, 198)

RTOL = 1e-9


def _approx(value, rel=RTOL):
    """Purely relative comparison: pytest.approx's default abs=1e-12 would
    otherwise accept *any* value when pinning p-values of magnitude ~1e-34."""
    return pytest.approx(value, rel=rel, abs=0.0)


def _make_frame() -> pd.DataFrame:
    """Five correlated columns; Z1 opens a non-causal X–Y path given M."""
    rng = np.random.default_rng(42)
    n = 200
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = 0.8 * z1 - 0.4 * z2 + rng.normal(scale=0.7, size=n)
    m = 0.7 * x + rng.normal(scale=0.6, size=n)
    y = 0.6 * m + 0.3 * z1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "M": m, "Y": y, "Z1": z1, "Z2": z2})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _make_frame()


@pytest.fixture(scope="module")
def data(frame) -> nw.DataFrame:
    return nw.from_native(frame, eager_only=True)


@pytest.fixture(scope="module")
def data_with_nans(frame) -> nw.DataFrame:
    df = frame.copy()
    df.loc[3, "X"] = np.nan
    df.loc[10, "M"] = np.nan
    df.loc[10, "Y"] = np.nan
    return nw.from_native(df, eager_only=True)


class TestIdentifyCharacterization:
    """Pin ``identify._partial_correlation_test`` on well-conditioned inputs."""

    @pytest.mark.parametrize(
        "x,y,z,expected",
        [(x, y, z, exp) for (x, y, z), exp in WELL_CONDITIONED_CASES.items()],
        ids=["marginal", "one-conditioner", "two-conditioners", "independent-pair"],
    )
    def test_pinned_values(self, data, x, y, z, expected):
        r, p, n = _partial_correlation_test(data, x, y, list(z))
        exp_r, exp_p, exp_n = expected
        assert r == _approx(exp_r)
        assert p == _approx(exp_p)
        assert n == exp_n

    def test_nan_rows_dropped(self, data_with_nans):
        r, p, n = _partial_correlation_test(data_with_nans, "X", "Y", ["M"])
        exp_r, exp_p, exp_n = NAN_CASE_EXPECTED
        assert r == _approx(exp_r)
        assert p == _approx(exp_p)
        assert n == exp_n


class TestFalsifyCharacterization:
    """Pin ``falsify._PartialCorrelationTester`` on well-conditioned inputs."""

    def test_pinned_values(self, data):
        tester = _PartialCorrelationTester(data, ["X", "M", "Y", "Z1", "Z2"])
        for (x, y, z), (_, exp_p, _) in WELL_CONDITIONED_CASES.items():
            assert tester.p_value(x, y, z) == _approx(exp_p)

    def test_nan_rows_dropped(self, data_with_nans):
        tester = _PartialCorrelationTester(data_with_nans, ["X", "M", "Y"])
        _, exp_p, _ = NAN_CASE_EXPECTED
        assert tester.p_value("X", "Y", ("M",)) == _approx(exp_p)


class TestImplementationParity:
    """The two implementations must agree wherever both are defined.

    This is the lock that prevents the engines from drifting apart again;
    after unification it holds by construction.
    """

    def test_p_values_agree(self, data):
        tester = _PartialCorrelationTester(data, ["X", "M", "Y", "Z1", "Z2"])
        for x, y, z in WELL_CONDITIONED_CASES:
            _, p_identify, _ = _partial_correlation_test(data, x, y, list(z))
            p_falsify = tester.p_value(x, y, z)
            assert p_identify == _approx(p_falsify, rel=1e-12)

    def test_p_values_agree_with_nans(self, data_with_nans):
        _, p_identify, _ = _partial_correlation_test(data_with_nans, "X", "Y", ["M"])
        tester = _PartialCorrelationTester(data_with_nans, ["X", "M", "Y"])
        assert p_identify == _approx(tester.p_value("X", "Y", ("M",)), rel=1e-12)


class TestImplicationsEndToEndCharacterization:
    """Pin ``test_implications()`` output through the public model API.

    The fitted spec omits Z1, which opens an X–Y path given M, so the
    single implied independence X ⊥⊥ Y | M is (correctly) flagged as a
    violation on this data.
    """

    @pytest.fixture(scope="class")
    def result(self, frame):
        model = pathmc.model("M ~ X\nY ~ M", data=frame[["X", "M", "Y"]])
        return model.test_implications()

    def test_schema(self, result):
        assert list(result.results.columns) == [
            "x",
            "y",
            "conditioning_set",
            "partial_corr",
            "p_value",
            "n_obs",
            "significant",
        ]

    def test_pinned_row(self, result):
        row = result.results.iloc[0]
        exp_r, exp_p, exp_n = WELL_CONDITIONED_CASES[("X", "Y", ("M",))]
        assert (row["x"], row["y"], row["conditioning_set"]) == ("X", "Y", "M")
        assert row["partial_corr"] == _approx(exp_r)
        assert row["p_value"] == _approx(exp_p)
        assert row["n_obs"] == exp_n
        assert bool(row["significant"]) is True


# ---------------------------------------------------------------------------
# Unit tests for the shared core: pathmc._ci.partial_correlation_ci
# ---------------------------------------------------------------------------


def _cols(frame, names):
    return frame[list(names)].to_numpy(dtype=float)


class TestCoreWellConditioned:
    """The core reproduces the characterization pins on the same inputs."""

    @pytest.mark.parametrize(
        "x,y,z,expected",
        [(x, y, z, exp) for (x, y, z), exp in WELL_CONDITIONED_CASES.items()],
        ids=["marginal", "one-conditioner", "two-conditioners", "independent-pair"],
    )
    def test_matches_pins(self, frame, x, y, z, expected):
        z_mat = _cols(frame, z) if z else None
        result = partial_correlation_ci(frame[x].to_numpy(), frame[y].to_numpy(), z_mat)
        exp_r, exp_p, exp_n = expected
        assert result.skip_reason is None
        assert result.r == _approx(exp_r)
        assert result.p == _approx(exp_p)
        assert result.n == exp_n
        assert result.df == exp_n - len(z) - 2

    def test_zero_column_z_is_marginal(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        empty_z = np.empty((len(x), 0))
        assert partial_correlation_ci(x, y, empty_z) == partial_correlation_ci(x, y)

    def test_nan_rows_dropped(self, frame):
        x = frame["X"].to_numpy().copy()
        y = frame["Y"].to_numpy().copy()
        z = _cols(frame, ["M"]).copy()
        x[3] = np.nan
        y[10] = np.nan
        z[10, 0] = np.nan
        result = partial_correlation_ci(x, y, z)
        exp_r, exp_p, exp_n = NAN_CASE_EXPECTED
        assert result.r == _approx(exp_r)
        assert result.p == _approx(exp_p)
        assert result.n == exp_n


class TestCoreRankAwareDf:
    """Degrees of freedom follow the effective rank of [1, Z] (#276)."""

    def test_constant_conditioner_equals_marginal(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        marginal = partial_correlation_ci(x, y)
        const = partial_correlation_ci(x, y, np.ones((len(x), 1)))
        assert const.skip_reason is None
        assert const.df == marginal.df
        assert const.p == _approx(marginal.p)

    def test_duplicate_conditioner_idempotent(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z = _cols(frame, ["M"])
        single = partial_correlation_ci(x, y, z)
        doubled = partial_correlation_ci(x, y, np.column_stack([z, z]))
        assert doubled.df == single.df
        assert doubled.p == _approx(single.p)

    def test_collinear_conditioner_counts_once(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z1 = _cols(frame, ["M"])
        collinear = np.column_stack([z1, 2.0 * z1 - 1.0])
        single = partial_correlation_ci(x, y, z1)
        result = partial_correlation_ci(x, y, collinear)
        assert result.df == single.df
        assert result.p == _approx(single.p)

    def test_rank_deficient_z_keeps_small_sample_viable(self):
        # n=5 with three identical conditioners: a nominal-k rule (n < k + 3)
        # would refuse; effective rank 2 leaves df = 2.
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=5), rng.normal(size=5)
        z = np.repeat(rng.normal(size=5)[:, None], 3, axis=1)
        result = partial_correlation_ci(x, y, z)
        assert result.skip_reason is None
        assert result.df == 2
        assert 0.0 <= result.p <= 1.0


class TestCoreSkips:
    """Every degenerate input maps to a named skip, never NaN or a warning."""

    def test_insufficient_observations(self):
        result = partial_correlation_ci(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert result.skip_reason == "insufficient_observations"
        assert result.n == 2
        assert result.r is None and result.p is None and result.df is None

    def test_all_rows_nan(self):
        x = np.array([np.nan, np.nan, np.nan])
        result = partial_correlation_ci(x, np.ones(3))
        assert result.skip_reason == "insufficient_observations"
        assert result.n == 0

    def test_zero_variance_marginal(self):
        rng = np.random.default_rng(0)
        result = partial_correlation_ci(np.ones(50), rng.normal(size=50))
        assert result.skip_reason == "zero_variance"

    def test_nonpositive_df(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=3), rng.normal(size=3)
        z = rng.normal(size=(3, 2))
        result = partial_correlation_ci(x, y, z)
        assert result.skip_reason == "nonpositive_df"

    def test_zero_residual_variance(self):
        # All-zero x gives an exactly-zero lstsq solution and exactly-zero
        # residuals; the guard is exact (matching the historical falsify
        # behavior), so float-noise residuals do not trigger it.
        rng = np.random.default_rng(2)
        z = rng.normal(size=(50, 1))
        result = partial_correlation_ci(np.zeros(50), rng.normal(size=50), z)
        assert result.skip_reason == "zero_residual_variance"

    def test_no_warnings_on_degenerate_input(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(50, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            partial_correlation_ci(np.zeros(50), rng.normal(size=50), z)
            partial_correlation_ci(np.ones(50), rng.normal(size=50))


class TestCorePerfectCorrelation:
    """|r| >= 1 is maximal evidence of dependence: p = 0, not a skip."""

    def test_perfect_residual_correlation(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=50)
        z = rng.normal(size=(50, 1))
        result = partial_correlation_ci(x, x.copy(), z)
        assert result.skip_reason is None
        assert result.p == 0.0
        assert result.r == pytest.approx(1.0)


class TestCIResultContract:
    def test_frozen(self, frame):
        result = partial_correlation_ci(frame["X"].to_numpy(), frame["Y"].to_numpy())
        assert isinstance(result, CIResult)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.p = 0.5


# ---------------------------------------------------------------------------
# identify._partial_correlation_test delegates to the shared core
# ---------------------------------------------------------------------------


class TestIdentifyDelegation:
    """The identify adapter inherits the core's guarantees (#276)."""

    def test_constant_conditioner_equals_marginal(self, frame):
        # The #276 reproducer: conditioning on a constant must be
        # equivalent to the marginal test.
        df = frame.copy()
        df["C"] = 1.0
        data = nw.from_native(df, eager_only=True)
        r_m, p_m, _ = _partial_correlation_test(data, "X", "Y", [])
        r_c, p_c, _ = _partial_correlation_test(data, "X", "Y", ["C"])
        assert r_c == _approx(r_m)
        assert p_c == _approx(p_m)

    def test_rank_deficient_small_sample_returns_result(self):
        # n=5 with three identical conditioners: the nominal-k refusal
        # (n < k + 3) is gone; effective rank leaves df = 2.
        rng = np.random.default_rng(7)
        base = rng.normal(size=5)
        df = pd.DataFrame({
            "X": rng.normal(size=5),
            "Y": rng.normal(size=5),
            "Z1": base,
            "Z2": base,
            "Z3": base,
        })
        data = nw.from_native(df, eager_only=True)
        r, p, n = _partial_correlation_test(data, "X", "Y", ["Z1", "Z2", "Z3"])
        assert not np.isnan(p)
        assert 0.0 <= p <= 1.0
        assert n == 5

    def test_perfect_correlation_p_zero_without_warnings(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=50)
        df = pd.DataFrame({"X": x, "Y": x.copy(), "Z": rng.normal(size=50)})
        data = nw.from_native(df, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r, p, _ = _partial_correlation_test(data, "X", "Y", ["Z"])
        assert p == 0.0
        assert r == pytest.approx(1.0)

    def test_degenerate_input_returns_nan_without_warnings(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({
            "X": np.zeros(50),
            "Y": rng.normal(size=50),
            "Z": rng.normal(size=50),
        })
        data = nw.from_native(df, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r, p, n = _partial_correlation_test(data, "X", "Y", ["Z"])
        assert np.isnan(r) and np.isnan(p)
        assert n == 50
