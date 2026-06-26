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
"""Tests locking in the xarray-backed storage of :class:`DoResult`
and :class:`EstimandResult` (issue #319).

These assert draws are stored in an :class:`xarray.Dataset` (via the public
:attr:`~DoResult.dataset` property) with named dims, while ``draws()`` and
related accessors continue to return numpy arrays. Fixtures are hand-built.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from pathmc.simulate import DoResult, EstimandResult
from _draw_fixtures import (
    build_dataset,
    do_result_from_flat,
    estimand_result_from_flat,
)

RNG = np.random.default_rng(0)
N_CHAINS = 2
N_DRAWS = 50
N_SAMPLES = N_CHAINS * N_DRAWS


def _draws(loc: float = 0.0, scale: float = 0.1) -> np.ndarray:
    return RNG.normal(loc=loc, scale=scale, size=N_SAMPLES)


# ---------------------------------------------------------------------------
# DoResult internal storage
# ---------------------------------------------------------------------------


class TestDoResultStorage:
    """DoResult stores draws in an xr.Dataset with named dims."""

    def test_dataset_is_xarray(self):
        result = do_result_from_flat(
            values={"Y": _draws()},
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert isinstance(result.dataset, xr.Dataset)

    def test_dataset_aliases_internal_store(self):
        result = do_result_from_flat(
            values={"Y": _draws()},
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert result.dataset is result._ds

    def test_chain_and_draw_dims_present(self):
        result = do_result_from_flat(
            values={"Y": _draws()},
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert "chain" in result.dataset.dims
        assert "draw" in result.dataset.dims
        assert result.dataset.sizes["chain"] == N_CHAINS
        assert result.dataset.sizes["draw"] == N_DRAWS

    def test_chain_coord_is_integer_range(self):
        result = do_result_from_flat(
            values={"Y": _draws()},
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        np.testing.assert_array_equal(
            result.dataset["chain"].values, np.arange(N_CHAINS)
        )
        np.testing.assert_array_equal(result.dataset["draw"].values, np.arange(N_DRAWS))

    def test_no_time_dim_for_cross_sectional(self):
        result = do_result_from_flat(
            values={"Y": _draws()}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        assert "time" not in result.dataset.dims

    def test_draws_returns_numpy_not_xarray(self):
        result = do_result_from_flat(
            values={"Y": _draws()}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        out = result.draws("Y")
        assert isinstance(out, np.ndarray)
        assert out.shape == (N_SAMPLES,)

    def test_draws_roundtrip_preserves_order(self):
        arr = np.arange(N_SAMPLES, dtype=float)
        result = do_result_from_flat(
            values={"Y": arr}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        np.testing.assert_array_equal(result.draws("Y"), arr)

    def test_unit_dim_for_predictive_length(self):
        n_units = 10
        predictive_len = N_CHAINS * N_DRAWS * n_units
        result = do_result_from_flat(
            values={"Y": RNG.normal(size=predictive_len)},
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
            n_units_per_var={"Y": n_units},
        )
        assert "unit" in result.dataset["Y"].dims
        assert result.dataset.sizes["unit"] == n_units


class TestDoResultPanelStorage:
    """Panel DoResult stores a time dim and time coord."""

    def test_time_dim_present(self):
        n_times = 5
        by_time = {var: RNG.normal(size=(n_times, N_SAMPLES)) for var in ("Y", "X")}
        result = do_result_from_flat(
            values={"Y": _draws(), "X": _draws()},
            values_by_time=by_time,
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert "time" in result.dataset.dims
        assert result.dataset.sizes["time"] == n_times

    def test_time_coord_matches_time_index(self):
        n_times = 5
        time_index = np.array([10, 20, 30, 40, 50])
        by_time = {"Y": RNG.normal(size=(n_times, N_SAMPLES))}
        result = do_result_from_flat(
            values={"Y": _draws()},
            values_by_time=by_time,
            time_index=time_index,
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        np.testing.assert_array_equal(result.dataset["time"].values, time_index)
        np.testing.assert_array_equal(result.time_index, time_index)

    def test_time_coord_defaults_to_range(self):
        n_times = 3
        by_time = {"Y": RNG.normal(size=(n_times, N_SAMPLES))}
        result = do_result_from_flat(
            values={"Y": _draws()},
            values_by_time=by_time,
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        np.testing.assert_array_equal(result.dataset["time"].values, np.arange(n_times))


# ---------------------------------------------------------------------------
# EstimandResult internal storage
# ---------------------------------------------------------------------------


class TestEstimandResultStorage:
    """EstimandResult stores draws in an xr.Dataset with named dims."""

    def test_dataset_is_xarray(self):
        er = estimand_result_from_flat(
            values={"Y": _draws(), "X": _draws()},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert isinstance(er.dataset, xr.Dataset)

    def test_chain_and_draw_dims_present(self):
        er = estimand_result_from_flat(
            values={"Y": _draws()},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert "chain" in er.dataset.dims
        assert "draw" in er.dataset.dims

    def test_draws_returns_numpy(self):
        er = estimand_result_from_flat(
            values={"Y": _draws()},
            outcome="Y",
            treatment="X",
            estimand="ATE",
            n_chains=N_CHAINS,
            n_draws=N_DRAWS,
        )
        assert isinstance(er.draws(), np.ndarray)
        assert er.draws().shape == (N_SAMPLES,)


# ---------------------------------------------------------------------------
# __sub__ and from_contrast on xarray
# ---------------------------------------------------------------------------


class TestSubOnXarray:
    """__sub__ subtracts shared variables via xarray on matching schemas."""

    def test_subtraction_produces_dataset(self):
        a = do_result_from_flat(
            values={"Y": _draws(1.0)}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        b = do_result_from_flat(
            values={"Y": _draws(0.0)}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        diff = a - b
        assert isinstance(diff.dataset, xr.Dataset)
        assert "chain" in diff.dataset.dims

    def test_subtraction_values_correct(self):
        a = do_result_from_flat(
            values={"Y": np.ones(N_SAMPLES)}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        b = do_result_from_flat(
            values={"Y": np.zeros(N_SAMPLES)}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        diff = a - b
        np.testing.assert_allclose(diff.draws("Y"), np.ones(N_SAMPLES))

    def test_subtraction_drops_non_shared_vars(self):
        a = do_result_from_flat(
            values={"Y": _draws(), "Z": _draws()}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        b = do_result_from_flat(
            values={"Y": _draws()}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        diff = a - b
        assert "Y" in diff.dataset.data_vars
        assert "Z" not in diff.dataset.data_vars


class TestFromContrastOnXarray:
    """from_contrast shares the DoResult's Dataset (no copy of numpy)."""

    def test_from_contrast_shares_dataset(self):
        contrast = do_result_from_flat(
            values={"Y": _draws(), "X": _draws()}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        er = EstimandResult.from_contrast(
            contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        assert er.dataset is contrast.dataset

    def test_from_contrast_draws_match(self):
        contrast = do_result_from_flat(
            values={"Y": _draws(0.5)}, n_chains=N_CHAINS, n_draws=N_DRAWS
        )
        er = EstimandResult.from_contrast(
            contrast, outcome="Y", treatment="X", estimand="ATE"
        )
        np.testing.assert_array_equal(er.draws(), contrast.draws("Y"))


class TestBuildDatasetRequiresChainDraw:
    """Flat dict fixture builder requires explicit chain/draw sizes."""

    def test_build_dataset_raises_without_chain_draw(self):
        with pytest.raises(ValueError, match="n_chains and n_draws are required"):
            build_dataset(values={"Y": _draws()}, values_by_time=None, time_index=None)

    def test_do_result_constructor_requires_ds(self):
        with pytest.raises(TypeError):
            DoResult(values={"Y": _draws()})  # type: ignore[call-arg]

    def test_estimand_constructor_requires_ds(self):
        with pytest.raises(TypeError):
            EstimandResult(  # type: ignore[call-arg]
                values={"Y": _draws()},
                outcome="Y",
                treatment="X",
                estimand="ATE",
            )
