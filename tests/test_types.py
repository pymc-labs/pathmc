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
"""Tests for the shared type aliases and their runtime checks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pathmc._types import validate_bins


@pytest.mark.parametrize(
    "bins", [None, 1, 10, np.int64(10), "auto", "sturges", [0.0, 0.5, 1.0], (0.0, 1.0)]
)
def test_accepted(bins):
    validate_bins(bins)


def test_accepts_ndarray_edges():
    validate_bins(np.array([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("bins", [-1, 0])
def test_non_positive_bin_count_rejected(bins):
    with pytest.raises(
        ValueError, match=rf"^bins must be a positive integer, got {bins}\.$"
    ):
        validate_bins(bins)


@pytest.mark.parametrize("bins", [2.5, True, {1, 2}, object()])
def test_invalid_type_rejected(bins):
    # bool subclasses int, so it needs an explicit reject.
    with pytest.raises(
        ValueError,
        match=r"^bins must be a positive integer, a binning strategy name, "
        r"a sequence of bin edges, or None, got ",
    ):
        validate_bins(bins)


def test_error_names_the_offending_type():
    with pytest.raises(ValueError, match=r"of type float\.$"):
        validate_bins(2.5)


EDGES = [-2.0, -1.0, 0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    "bins",
    [
        pytest.param(pd.Series(EDGES), id="pandas-series"),
        pytest.param(pd.Index(EDGES), id="pandas-index"),
        pytest.param(xr.DataArray(EDGES), id="xarray-dataarray"),
        pytest.param(np.array(EDGES), id="numpy-array"),
    ],
)
def test_array_like_edges_accepted(bins):
    # Axes.hist takes all of these; rejecting them would break callers who
    # build edges from a dataframe column.
    validate_bins(bins)


def test_zero_dim_array_left_to_matplotlib():
    # Not a bin count pathmc can check, so it is delegated rather than
    # reported with a message that would contradict matplotlib's.
    validate_bins(np.array(-5))
