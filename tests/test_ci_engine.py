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

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc
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
        assert r == pytest.approx(exp_r, rel=RTOL)
        assert p == pytest.approx(exp_p, rel=RTOL)
        assert n == exp_n

    def test_nan_rows_dropped(self, data_with_nans):
        r, p, n = _partial_correlation_test(data_with_nans, "X", "Y", ["M"])
        exp_r, exp_p, exp_n = NAN_CASE_EXPECTED
        assert r == pytest.approx(exp_r, rel=RTOL)
        assert p == pytest.approx(exp_p, rel=RTOL)
        assert n == exp_n


class TestFalsifyCharacterization:
    """Pin ``falsify._PartialCorrelationTester`` on well-conditioned inputs."""

    def test_pinned_values(self, data):
        tester = _PartialCorrelationTester(data, ["X", "M", "Y", "Z1", "Z2"])
        for (x, y, z), (_, exp_p, _) in WELL_CONDITIONED_CASES.items():
            assert tester.p_value(x, y, z) == pytest.approx(exp_p, rel=RTOL)

    def test_nan_rows_dropped(self, data_with_nans):
        tester = _PartialCorrelationTester(data_with_nans, ["X", "M", "Y"])
        _, exp_p, _ = NAN_CASE_EXPECTED
        assert tester.p_value("X", "Y", ("M",)) == pytest.approx(exp_p, rel=RTOL)


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
            assert p_identify == pytest.approx(p_falsify, rel=1e-12)

    def test_p_values_agree_with_nans(self, data_with_nans):
        _, p_identify, _ = _partial_correlation_test(data_with_nans, "X", "Y", ["M"])
        tester = _PartialCorrelationTester(data_with_nans, ["X", "M", "Y"])
        assert p_identify == pytest.approx(tester.p_value("X", "Y", ("M",)), rel=1e-12)


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
        assert row["partial_corr"] == pytest.approx(exp_r, rel=RTOL)
        assert row["p_value"] == pytest.approx(exp_p, rel=RTOL)
        assert row["n_obs"] == exp_n
        assert bool(row["significant"]) is True
