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
implementations — ``identify._partial_correlation_result`` (a
:class:`~pathmc._ci.CIResult`) and ``falsify._PartialCorrelationTester`` —
on well-conditioned inputs, so the engine unification can demonstrate
"no behavior change for well-conditioned inputs" against a recorded
baseline rather than by algebra alone.
"""

import dataclasses
import warnings
from pathlib import Path

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc._ci import CIResult, partial_correlation_ci
from pathmc.falsify import _PartialCorrelationTester, falsify_graph
from pathmc.graph import build_graph
from pathmc.identify import ConditionalIndependence, _partial_correlation_result
from pathmc.parse import parse_spec


def _warning_is_from_caller(filename: str, caller: Path) -> bool:
    """Whether this warning frame is the caller file.

    A checkout whose path contains ``/pathmc/`` is still the caller when
    the filename is that file. A ``pathmc`` module is not.
    """
    return Path(filename).resolve() == caller.resolve()


def test_caller_attribution_accepts_a_checkout_named_pathmc() -> None:
    caller = Path("/home/runner/work/pathmc/pathmc/tests/test_ci_engine.py")
    assert _warning_is_from_caller(str(caller), caller)
    package = Path("/home/runner/work/pathmc/pathmc/pathmc/_ci_report.py")
    assert not _warning_is_from_caller(str(package), caller)
    other_copy = Path("/tmp/elsewhere/tests/test_ci_engine.py")
    assert not _warning_is_from_caller(str(other_copy), caller)


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
    """Pin the named-column CIResult on well-conditioned inputs."""

    @pytest.mark.parametrize(
        "x,y,z,expected",
        [(x, y, z, exp) for (x, y, z), exp in WELL_CONDITIONED_CASES.items()],
        ids=["marginal", "one-conditioner", "two-conditioners", "independent-pair"],
    )
    def test_pinned_values(self, data, x, y, z, expected):
        result = _partial_correlation_result(data, x, y, list(z))
        exp_r, exp_p, exp_n = expected
        assert isinstance(result, CIResult)
        assert result.r == _approx(exp_r)
        assert result.p == _approx(exp_p)
        assert result.n == exp_n
        assert result.requested_k == len(z)
        assert result.rank_collapsed is False

    def test_nan_rows_dropped(self, data_with_nans):
        result = _partial_correlation_result(data_with_nans, "X", "Y", ["M"])
        exp_r, exp_p, exp_n = NAN_CASE_EXPECTED
        assert result.r == _approx(exp_r)
        assert result.p == _approx(exp_p)
        assert result.n == exp_n
        assert result.rank_collapsed is False


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
            result = _partial_correlation_result(data, x, y, list(z))
            assert result.p == _approx(tester.p_value(x, y, z), rel=1e-12)
            assert result.rank_collapsed is False

    def test_p_values_agree_with_nans(self, data_with_nans):
        result = _partial_correlation_result(data_with_nans, "X", "Y", ["M"])
        tester = _PartialCorrelationTester(data_with_nans, ["X", "M", "Y"])
        assert result.p == _approx(tester.p_value("X", "Y", ("M",)), rel=1e-12)
        assert result.rank_collapsed is False


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
            "requested_k",
            "effective_k",
            "rank_collapsed",
            "df",
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
        assert result.requested_k == len(z)
        assert result.rank_collapsed is False
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
        assert const.requested_k == 1
        assert const.effective_k == 0
        assert const.rank_collapsed is True
        assert marginal.requested_k == 0
        assert marginal.rank_collapsed is False

    def test_duplicate_conditioner_idempotent(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z = _cols(frame, ["M"])
        single = partial_correlation_ci(x, y, z)
        doubled = partial_correlation_ci(x, y, np.column_stack([z, z]))
        assert doubled.df == single.df
        assert doubled.p == _approx(single.p)
        assert doubled.requested_k == 2
        assert doubled.effective_k == 1
        assert doubled.rank_collapsed is True
        assert single.requested_k == 1
        assert single.rank_collapsed is False

    def test_collinear_conditioner_counts_once(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z1 = _cols(frame, ["M"])
        collinear = np.column_stack([z1, 2.0 * z1 - 1.0])
        single = partial_correlation_ci(x, y, z1)
        result = partial_correlation_ci(x, y, collinear)
        assert result.df == single.df
        assert result.p == _approx(single.p)
        assert result.requested_k == 2
        assert result.rank_collapsed is True
        assert single.rank_collapsed is False

    def test_rank_deficient_z_keeps_small_sample_viable(self):
        # n=5 with three identical conditioners: a nominal-k rule (n < k + 3)
        # would refuse; effective rank 2 leaves df = 2.
        rng = np.random.default_rng(7)
        x, y = rng.normal(size=5), rng.normal(size=5)
        z = np.repeat(rng.normal(size=5)[:, None], 3, axis=1)
        result = partial_correlation_ci(x, y, z)
        assert result.skip_reason is None
        assert result.df == 2
        assert result.requested_k == 3
        assert result.effective_k == 1
        assert result.rank_collapsed is True
        assert 0.0 <= result.p <= 1.0


class TestCoreSkips:
    """Every degenerate input maps to a named skip, never NaN or a warning."""

    def test_insufficient_observations(self):
        result = partial_correlation_ci(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert result.skip_reason == "insufficient_observations"
        assert result.n == 2
        assert result.r is None and result.p is None and result.df is None
        assert result.requested_k == 0
        assert result.rank_collapsed is False

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
        assert result.requested_k == 2
        assert result.effective_k is None
        assert result.rank_collapsed is False

    def test_zero_residual_variance(self):
        # All-zero x gives an exactly-zero lstsq solution and exactly-zero
        # residuals; the guard is exact (matching the historical falsify
        # behavior), so float-noise residuals do not trigger it.
        rng = np.random.default_rng(2)
        z = rng.normal(size=(50, 1))
        result = partial_correlation_ci(np.zeros(50), rng.normal(size=50), z)
        assert result.skip_reason == "zero_residual_variance"
        assert result.requested_k == 1
        assert result.rank_collapsed is False

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
# identify._partial_correlation_result returns the shared CIResult
# ---------------------------------------------------------------------------


class TestIdentifyDelegation:
    """The named-column adapter inherits the core's guarantees (#276)."""

    def test_constant_conditioner_equals_marginal(self, frame):
        # The #276 reproducer: conditioning on a constant must be
        # equivalent to the marginal test, and must say the set collapsed.
        df = frame.copy()
        df["C"] = 1.0
        data = nw.from_native(df, eager_only=True)
        marginal = _partial_correlation_result(data, "X", "Y", [])
        const = _partial_correlation_result(data, "X", "Y", ["C"])
        assert isinstance(const, CIResult)
        assert const.r == _approx(marginal.r)
        assert const.p == _approx(marginal.p)
        assert const.requested_k == 1
        assert const.effective_k == 0
        assert const.rank_collapsed is True
        assert marginal.rank_collapsed is False

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
        result = _partial_correlation_result(data, "X", "Y", ["Z1", "Z2", "Z3"])
        assert result.skip_reason is None
        assert result.p is not None and 0.0 <= result.p <= 1.0
        assert result.n == 5
        assert result.requested_k == 3
        assert result.effective_k == 1
        assert result.rank_collapsed is True
        assert result.df == 2

    def test_perfect_correlation_p_zero_without_warnings(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=50)
        df = pd.DataFrame({"X": x, "Y": x.copy(), "Z": rng.normal(size=50)})
        data = nw.from_native(df, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _partial_correlation_result(data, "X", "Y", ["Z"])
        assert result.p == 0.0
        assert result.r == pytest.approx(1.0)
        assert result.rank_collapsed is False
        assert result.requested_k == 1

    def test_degenerate_input_is_a_skip_without_warnings(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({
            "X": np.zeros(50),
            "Y": rng.normal(size=50),
            "Z": rng.normal(size=50),
        })
        data = nw.from_native(df, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _partial_correlation_result(data, "X", "Y", ["Z"])
        assert result.skip_reason == "zero_residual_variance"
        assert result.r is None and result.p is None
        assert result.n == 50
        assert result.requested_k == 1
        assert result.rank_collapsed is False


def _constant_conditioner_frame() -> pd.DataFrame:
    """C is constant, so any set that includes C is rank-deficient.

    The DAG ``M ~ C`` / ``Z ~ X`` implies two independences that condition
    on C: M ⊥⊥ X | C, and M ⊥⊥ Z | {C, X}.
    """
    rng = np.random.default_rng(11)
    n = 50
    x = rng.normal(size=n)
    c = np.ones(n)
    return pd.DataFrame({
        "C": c,
        "X": x,
        "M": 0.4 * c + rng.normal(size=n),
        "Z": 0.7 * x + rng.normal(size=n),
    })


class TestRankCollapseIsVisible:
    """A collapsed conditioning set must not look like the one requested (#376)."""

    def test_core_reports_effective_k_without_warning(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z = _cols(frame, ["M"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            marginal = partial_correlation_ci(x, y)
            const = partial_correlation_ci(x, y, np.ones((len(x), 1)))
            doubled = partial_correlation_ci(x, y, np.column_stack([z, z]))
        assert marginal.effective_k == 0
        assert marginal.requested_k == 0
        assert marginal.rank_collapsed is False
        assert const.effective_k == 0
        assert const.requested_k == 1
        assert const.rank_collapsed is True
        assert const.df == marginal.df == len(x) - 2
        assert doubled.effective_k == 1
        assert doubled.requested_k == 2
        assert doubled.rank_collapsed is True
        assert doubled.df == len(x) - doubled.effective_k - 2

    def test_skip_has_no_effective_k(self):
        result = partial_correlation_ci(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert result.effective_k is None
        assert result.requested_k == 0
        assert result.rank_collapsed is False

    def test_skip_with_conditioners_is_not_rank_collapsed(self):
        # Too few rows to run. requested_k is known, but a skip is not a
        # weaker version of the requested independence.
        result = partial_correlation_ci(
            np.array([1.0, 2.0]),
            np.array([2.0, 1.0]),
            np.ones((2, 1)),
        )
        assert result.skip_reason == "insufficient_observations"
        assert result.requested_k == 1
        assert result.effective_k is None
        assert result.rank_collapsed is False

    def test_named_adapter_reports_collapse_without_warning(self):
        data = nw.from_native(_constant_conditioner_frame(), eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collapsed = _partial_correlation_result(data, "M", "X", ["C"])
            marginal = _partial_correlation_result(data, "M", "X", [])
        assert isinstance(collapsed, CIResult)
        assert collapsed.n == 50
        assert collapsed.rank_collapsed is True
        assert collapsed.requested_k == 1
        assert collapsed.effective_k == 0
        assert collapsed.r == _approx(marginal.r)
        assert marginal.rank_collapsed is False

    def test_implications_frame_and_single_warning(self):
        frame = _constant_conditioner_frame()
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.test_implications()
        collapses = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(collapses) == 1
        assert _warning_is_from_caller(collapses[0].filename, Path(__file__))
        assert "rank collapse" in str(collapses[0].message)
        assert "M ⊥⊥ X | {C}" in str(collapses[0].message)
        assert "M ⊥⊥ Z | {C, X}" in str(collapses[0].message)

        by_pair = {
            (row.x, row.y, row.conditioning_set): row
            for row in result.results.itertuples(index=False)
        }
        only_c = by_pair[("M", "X", "C")]
        assert only_c.requested_k == 1
        assert only_c.effective_k == 0
        assert only_c.rank_collapsed
        assert only_c.df == only_c.n_obs - 2
        both = by_pair[("M", "Z", "C, X")]
        assert both.requested_k == 2
        assert both.effective_k == 1
        assert both.rank_collapsed
        assert both.df == both.n_obs - both.effective_k - 2
        # C is constant, so the marginal C ⊥⊥ X test is skipped outright
        # (zero variance) rather than reported as a rank collapse.
        skipped = by_pair[("C", "X", "")]
        assert skipped.requested_k == 0
        assert not skipped.rank_collapsed
        assert np.isnan(skipped.effective_k)
        assert np.isnan(skipped.p_value)
        html = result._repr_html_()
        assert "All implied independences are consistent" not in html
        assert "rank-collapsed" in html
        assert "did not test" in html
        assert "[effective_k=0]" in html
        assert "[effective_k=1]" in html
        assert "rank-collapsed" in repr(result)

    def test_collapsed_significant_row_is_not_a_violation(self):
        # The only significant row is the collapsed marginal of M ⊥⊥ X | {C}.
        # A wrong display marks that row ✗ and calls it a violated implication.
        rng = np.random.default_rng(4)
        n = 80
        m = rng.normal(size=n)
        frame = pd.DataFrame({
            "C": np.ones(n),
            "M": m,
            "X": m + rng.normal(scale=0.05, size=n),
            "Z": rng.normal(size=n),
        })
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.test_implications()
        significant = result.results[result.results["significant"]]
        assert len(significant) == 1
        row = significant.iloc[0]
        assert row["rank_collapsed"]
        assert {row["x"], row["y"]} == {"M", "X"}
        html = result._repr_html_()
        collapsed_row = next(
            part for part in html.split("<tr") if "M ⊥⊥ X | {C}" in part
        )
        assert "✗" not in collapsed_row
        assert "✓" not in collapsed_row
        assert "—" in collapsed_row
        assert "ffe0e0" not in collapsed_row
        assert "[effective_k=0]" in collapsed_row
        assert "violated" not in html
        assert "implied independences violated" not in html
        assert "0 violations" in repr(result)

    def test_mixed_implication_counts_only_the_written_violation(self):
        rng = np.random.default_rng(1)
        n = 120
        z = rng.normal(size=n)
        x = rng.normal(size=n)
        frame = pd.DataFrame({
            "X": x,
            "Y": z + rng.normal(scale=0.1, size=n),
            "Z": z,
            "W": z + rng.normal(scale=0.1, size=n),
            "M": x + rng.normal(scale=0.1, size=n),
            "C": np.ones(n),
        })
        model = pathmc.model("Y ~ X\nW ~ Z\nM ~ C", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.test_implications()
        collapsed = result.results["rank_collapsed"].fillna(False).astype(bool)
        written = result.results.loc[~collapsed]
        written_violations = int(written["significant"].sum())
        assert collapsed.any()
        assert written_violations >= 1
        assert result.results.loc[collapsed, "significant"].any()
        assert result.n_violations == written_violations
        assert len(result.violations) == written_violations
        assert not result.violations["rank_collapsed"].any()
        html = result._repr_html_()
        assert f"{written_violations} of {len(written)}" in html
        assert f"{written_violations} violations" in repr(result)
        assert f"{len(written)} tests" in repr(result)

    def test_well_conditioned_implications_do_not_warn(self, frame):
        model = pathmc.model("M ~ X\nY ~ M", data=frame[["X", "M", "Y"]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.test_implications()
        row = result.results.iloc[0]
        assert row["requested_k"] == 1
        assert row["effective_k"] == 1
        assert not row["rank_collapsed"]
        assert row["df"] == row["n_obs"] - row["effective_k"] - 2
        html = result._repr_html_()
        assert "effective_k" not in html
        assert "rank-collapsed" not in html
        assert "rank-collapsed" not in repr(result)

    def test_falsify_warns_once_for_the_given_dag(self):
        frame = _constant_conditioner_frame()
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.falsify(n_permutations=24, random_seed=0)
        collapses = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "rank collapse" in str(w.message)
        ]
        assert len(collapses) == 1
        assert _warning_is_from_caller(collapses[0].filename, Path(__file__))
        collapsed = result.local_violations[result.local_violations["rank_collapsed"]]
        assert set(collapsed["conditioning_set"]) == {"C"}
        assert (collapsed["requested_k"] == 1).all()
        assert (collapsed["effective_k"] == 0).all()
        assert (collapsed["df"] == len(frame) - 2).all()
        summary = "\n".join(result._summary_lines())
        assert "rank-collapsed" in summary
        assert "requested parent sets" in summary
        assert "Verdict: we do not reject the DAG." not in summary
        html = result._repr_html_()
        assert "rank-collapsed" in html
        assert "requested parent sets" in html
        assert "LMC violations (given DAG)" in html
        assert "rank-collapsed" in repr(result)

    def test_collapsed_row_does_not_decide_the_lmc_count(self):
        rng = np.random.default_rng(0)
        n = 80
        a = rng.normal(size=n)
        y = rng.normal(size=n)
        frame = pd.DataFrame({
            "A": a,
            "B": 2 * a - 1,
            "M": y + rng.normal(scale=0.05, size=n),
            "X": rng.normal(size=n),
            "Y": y,
        })
        model = pathmc.model("M ~ A + B\nY ~ X", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(n_permutations=8, random_seed=0)
        collapsed = result.local_violations["rank_collapsed"].astype(bool)
        assert collapsed.any()
        scored = result.local_violations.loc[~collapsed]
        scored_viol = int(scored["violation"].sum()) if len(scored) else 0
        scored_tests = len(scored)
        n_collapsed = int(collapsed.sum())
        assert result.given_lmc_violations == scored_viol
        assert result.n_lmc_tests == scored_tests
        assert result.violations["rank_collapsed"].sum() == 0
        assert result.p_value_lmc == pytest.approx(
            float(
                np.mean(
                    result.perm_lmc_violation_fractions
                    <= result.given_lmc_violation_fraction
                )
            )
        )
        text = "\n".join(result._summary_lines())
        noun = "test" if n_collapsed == 1 else "tests"
        assert f"{scored_viol} of {scored_tests} requested" in text
        assert f"{n_collapsed} {noun} rank-collapsed" in text
        assert "permutation fraction" in text
        html = result._repr_html_()
        assert f"{scored_viol} / {scored_tests} requested" in html
        assert f"{n_collapsed} rank-collapsed" in html

    def test_all_collapsed_parent_sets_are_not_evaluable(self):
        rng = np.random.default_rng(2)
        n = 60
        a = rng.normal(size=n)
        frame = pd.DataFrame({
            "A": a,
            "B": 2 * a - 1,
            "X": a + rng.normal(scale=0.2, size=n),
            "Y": a + rng.normal(scale=0.2, size=n),
        })
        model = pathmc.model("B ~ A\nX ~ A + B\nY ~ A + B", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(n_permutations=6, random_seed=1)
        assert result.local_violations["rank_collapsed"].all()
        assert result.n_lmc_tests == 0
        assert result.given_lmc_violations == 0
        assert result.can_evaluate is False
        assert result.falsified is None
        text = "\n".join(result._summary_lines())
        assert "Every requested parental independence rank-collapsed" in text
        assert "fully connected" not in text
        html = result._repr_html_()
        assert "every requested parental independence rank-collapsed" in html
        assert "implies no testable" not in html
        assert "not violations" in html
        compact = repr(result)
        assert "not_evaluable" in compact
        assert "rank-collapsed" in compact
        assert "\n" not in compact
        with pytest.raises(RuntimeError, match="rank-collapsed"):
            result.plot()

    def test_collapsed_only_permutation_is_not_a_perfect_lmc_score(self):
        rng = np.random.default_rng(0)
        n = 200
        a = rng.normal(size=n)
        x = 0.8 * a + rng.normal(scale=0.3, size=n)
        z = 0.8 * a + rng.normal(scale=0.3, size=n)
        y = 0.5 * a + 0.5 * x + 0.5 * z + rng.normal(scale=0.3, size=n)
        w = 0.5 * a + 0.8 * y + rng.normal(scale=0.3, size=n)
        frame = pd.DataFrame({
            "A": a,
            "X": x,
            "Z": z,
            "Y": y,
            "W": w,
            "C": np.ones(n),
        })
        model = pathmc.model(
            "X ~ A\nZ ~ A\nY ~ A + X + Z\nW ~ Y + A\nC ~ A",
            data=frame,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(
                n_permutations=120,
                random_seed=4,
                include_unconditional=False,
            )
        scored = np.asarray(result.perm_lmc_violation_fractions)
        omitted = result.n_permutations - len(scored)
        assert len(result.perm_tpa_violation_fractions) == result.n_permutations
        assert omitted > 0
        assert result.p_value_tpa == pytest.approx(
            result.n_in_mec / result.n_permutations
        )
        scored_p = float(np.mean(scored <= result.given_lmc_violation_fraction))
        inflated = np.concatenate([scored, np.zeros(omitted)])
        inflated_p = float(np.mean(inflated <= result.given_lmc_violation_fraction))
        assert result.p_value_lmc == pytest.approx(scored_p)
        assert inflated_p > result.significance_level
        assert result.p_value_lmc <= result.significance_level
        assert result.falsified is False
        text = "\n".join(result._summary_lines())
        assert "not in that fraction" in text
        assert f"{inflated_p:.3f}" not in text
        html = result._repr_html_()
        assert f"{omitted} omitted" in html
        assert f"{inflated_p:.3f}" not in html
        fig = result.plot()
        assert fig.axes[0].get_title().startswith("DAG falsification")

    def test_unscored_permutation_baseline_is_not_a_pass(self):
        rng = np.random.default_rng(0)
        n = 80
        b = rng.normal(size=n)
        frame = pd.DataFrame({
            "B": b,
            "Y": b + rng.normal(scale=0.2, size=n),
            "Z": b + rng.normal(scale=0.2, size=n),
            "K": np.ones(n),
        })
        model = pathmc.model("Y ~ B\nZ ~ B\nK ~ B", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(
                n_permutations=1,
                random_seed=6,
                include_unconditional=False,
            )
        assert result.can_evaluate is True
        assert result.falsifiable is True
        assert len(result.perm_lmc_violation_fractions) == 0
        assert len(result.perm_tpa_violation_fractions) == result.n_permutations
        assert result.falsified is None
        text = "\n".join(result._summary_lines())
        assert "LMC comparison was not run" in text
        assert "not rejected" not in text
        assert "not_compared" in repr(result)
        assert "not_rejected" not in repr(result)
        html = result._repr_html_()
        assert "Not compared" in html
        assert "LMC comparison was not run" in html
        assert "Not rejected" not in html

    def test_falsify_full_rank_parents_do_not_warn(self, frame):
        model = pathmc.model("M ~ X\nY ~ M", data=frame[["X", "M", "Y"]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.falsify(n_permutations=6, random_seed=0)
        assert "effective_k" in result.local_violations.columns
        assert "df" in result.local_violations.columns
        assert "requested_k" in result.local_violations.columns
        assert "rank_collapsed" in result.local_violations.columns
        assert not result.local_violations["rank_collapsed"].any()
        assert "rank-collapsed" not in "\n".join(result._summary_lines())
        assert "rank-collapsed" not in result._repr_html_()
        assert "rank-collapsed" not in repr(result)

    def test_single_level_categorical_conditioner_is_not_the_written_test(self):
        """A named conditioner with no columns is not the written independence."""
        from pathmc.identify import test_implications

        rng = np.random.default_rng(1)
        n = 200
        x = rng.normal(size=n)
        y = 0.6 * x + rng.normal(scale=0.3, size=n)
        statement = ConditionalIndependence("x", "y", frozenset({"g"}))
        categorical = pd.DataFrame({"x": x, "y": y, "g": ["only"] * n})
        numeric = pd.DataFrame({"x": x, "y": y, "g": np.ones(n)})
        coded = pd.DataFrame({"x": x, "y": y, "g": np.ones(n, dtype=int)})
        two_level = pd.DataFrame({
            "x": x,
            "y": y,
            "g": np.array(["a", "b"] * (n // 2)),
        })
        cat_data = nw.from_native(categorical, eager_only=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            engine = _partial_correlation_result(cat_data, "x", "y", ["g"])
        assert engine.requested_k == 0
        assert engine.effective_k == 0
        assert engine.rank_collapsed is False
        assert engine.p is not None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            cat_result = test_implications([statement], cat_data)
        assert len(caught) == 1
        assert "g contributed no columns" in str(caught[0].message)
        row = cat_result.results.iloc[0]
        assert row["requested_k"] == 0
        assert row["effective_k"] == 0
        assert bool(row["rank_collapsed"]) is True
        assert not np.isnan(row["p_value"])
        assert cat_result.n_violations == 0
        assert cat_result.n_tests == 1
        assert "0 tests, 0 violations, 1 rank-collapsed" in repr(cat_result)
        html = cat_result._repr_html_()
        assert "rank-collapsed" in html
        assert "implied independences violated" not in html
        assert "All implied independences are consistent" not in html

        with warnings.catch_warnings(record=True) as numeric_caught:
            warnings.simplefilter("always", UserWarning)
            num_result = test_implications(
                [statement], nw.from_native(numeric, eager_only=True)
            )
        assert "contributed no columns" not in str(numeric_caught[0].message)
        assert bool(num_result.results.iloc[0]["rank_collapsed"]) is True
        assert row["p_value"] == pytest.approx(num_result.results.iloc[0]["p_value"])

        with warnings.catch_warnings(record=True) as coded_caught:
            warnings.simplefilter("always", UserWarning)
            coded_result = test_implications(
                [statement],
                nw.from_native(coded, eager_only=True),
                categorical_vars=["g"],
            )
        assert "g contributed no columns" in str(coded_caught[0].message)
        assert bool(coded_result.results.iloc[0]["rank_collapsed"]) is True
        assert coded_result.results.iloc[0]["requested_k"] == 0

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two_result = test_implications(
                [statement], nw.from_native(two_level, eager_only=True)
            )
        assert bool(two_result.results.iloc[0]["rank_collapsed"]) is False

        mixed_y = x + 0.05 * rng.normal(size=n)
        mixed = pd.DataFrame({
            "x": x,
            "y": mixed_y,
            "g": ["only"] * n,
            "m": rng.normal(size=n),
        })
        only_m = ConditionalIndependence("x", "y", frozenset({"m"}))
        mixed_statement = ConditionalIndependence("x", "y", frozenset({"g", "m"}))
        mixed_data = nw.from_native(mixed, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m_only = test_implications(
                [only_m], nw.from_native(mixed.drop(columns="g"), eager_only=True)
            )
        assert m_only.n_violations == 1
        with warnings.catch_warnings(record=True) as mixed_caught:
            warnings.simplefilter("always", UserWarning)
            mixed_result = test_implications([mixed_statement], mixed_data)
        mixed_row = mixed_result.results.iloc[0]
        assert mixed_row["requested_k"] == 1
        assert mixed_row["effective_k"] == 1
        assert bool(mixed_row["rank_collapsed"]) is True
        assert mixed_result.n_violations == 0
        assert mixed_row["p_value"] == pytest.approx(m_only.results.iloc[0]["p_value"])
        assert "g contributed no columns" in str(mixed_caught[0].message)

        graph = build_graph(parse_spec("Y ~ G\nZ ~ G"))
        cat_frame = pd.DataFrame({
            "G": ["only"] * n,
            "Y": y,
            "Z": y + rng.normal(scale=0.2, size=n),
        })
        num_frame = cat_frame.assign(G=np.ones(n))
        with warnings.catch_warnings(record=True) as falsify_caught:
            warnings.simplefilter("always", UserWarning)
            cat_falsify = falsify_graph(
                graph,
                nw.from_native(cat_frame, eager_only=True),
                n_permutations=2,
                random_seed=0,
                include_unconditional=False,
            )
        assert len(falsify_caught) == 1
        assert "G contributed no columns" in str(falsify_caught[0].message)
        assert cat_falsify.n_lmc_tests == 0
        assert cat_falsify.given_lmc_violations == 0
        assert cat_falsify.can_evaluate is False
        assert cat_falsify.falsified is None
        assert cat_falsify.local_violations["rank_collapsed"].all()
        assert cat_falsify.local_violations["requested_k"].eq(0).all()
        assert cat_falsify.local_violations["p_value"].notna().all()
        text = "\n".join(cat_falsify._summary_lines())
        assert "rank-collapsed" in text
        assert "we do not reject" not in text
        assert "not_rejected" not in repr(cat_falsify)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            num_falsify = falsify_graph(
                graph,
                nw.from_native(num_frame, eager_only=True),
                n_permutations=2,
                random_seed=0,
                include_unconditional=False,
            )
        assert num_falsify.n_lmc_tests == 0
        assert num_falsify.falsified is None
        assert cat_falsify.local_violations["p_value"].to_numpy() == pytest.approx(
            num_falsify.local_violations["p_value"].to_numpy()
        )
