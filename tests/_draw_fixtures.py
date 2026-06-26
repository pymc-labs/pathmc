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
"""Test-only helpers for building :class:`DoResult` / :class:`EstimandResult`
from flat ``dict[str, np.ndarray]`` fixtures.

Production code builds labelled :class:`xarray.Dataset` objects directly;
these helpers exist only for hand-built test fixtures.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from pathmc.simulate import DoResult, EstimandResult


def build_dataset(
    values: dict[str, np.ndarray],
    values_by_time: dict[str, np.ndarray] | None,
    time_index: np.ndarray | None,
    *,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_units_per_var: dict[str, int] | None = None,
) -> xr.Dataset:
    """Build a ``("chain", "draw"[, "unit"][, "time"])`` Dataset from flat arrays."""
    data_vars: dict[str, xr.DataArray] = {}

    def _chain_draw_coords(nc: int, nd: int) -> dict[str, np.ndarray]:
        return {"chain": np.arange(nc), "draw": np.arange(nd)}

    def _require_chain_draw() -> None:
        if n_chains is None or n_draws is None:
            raise ValueError(
                "n_chains and n_draws are required when building from flat "
                "values dicts. Pass ds= for a pre-built Dataset."
            )

    if values:
        _require_chain_draw()
        assert n_chains is not None and n_draws is not None
        time_var_names = set(values_by_time or {})

        for var, arr in values.items():
            if var in time_var_names:
                continue
            arr = np.asarray(arr)
            length = arr.shape[0]
            coords = _chain_draw_coords(n_chains, n_draws)
            var_n_units = (n_units_per_var or {}).get(var)
            if length == n_chains * n_draws:
                data_vars[var] = xr.DataArray(
                    arr.reshape(n_chains, n_draws),
                    dims=("chain", "draw"),
                    coords=coords,
                )
            elif var_n_units is not None and length == n_chains * n_draws * var_n_units:
                data_vars[var] = xr.DataArray(
                    arr.reshape(n_chains, n_draws, var_n_units),
                    dims=("chain", "draw", "unit"),
                    coords={**coords, "unit": np.arange(var_n_units)},
                )
            else:
                raise ValueError(
                    f"Cannot store variable '{var}': length {length} does not "
                    f"match chain*draw ({n_chains * n_draws})"
                    + (
                        f" or chain*draw*unit with n_units={var_n_units!r}."
                        if var_n_units is not None
                        else ". Supply n_units_per_var for predictive lengths."
                    )
                )

    if values_by_time:
        _require_chain_draw()
        assert n_chains is not None and n_draws is not None
        n_times = next(iter(values_by_time.values())).shape[0]
        time_coord = (
            np.asarray(time_index) if time_index is not None else np.arange(n_times)
        )
        for var, arr in values_by_time.items():
            arr = np.asarray(arr)
            expected = n_times * n_chains * n_draws
            if arr.size != expected:
                raise ValueError(
                    f"Cannot store per-time variable '{var}': size {arr.size} "
                    f"does not match n_times*chain*draw ({expected})."
                )
            coords = {
                "time": time_coord,
                **_chain_draw_coords(n_chains, n_draws),
            }
            data_vars[var] = xr.DataArray(
                arr.reshape(n_times, n_chains, n_draws),
                dims=("time", "chain", "draw"),
                coords=coords,
            )

    return xr.Dataset(data_vars)


def do_result_from_flat(
    values: dict[str, np.ndarray] | None = None,
    values_by_time: dict[str, np.ndarray] | None = None,
    time_index: np.ndarray | None = None,
    *,
    n_chains: int | None = None,
    n_draws: int | None = None,
    n_units_per_var: dict[str, int] | None = None,
) -> DoResult:
    """Construct a :class:`DoResult` from flat numpy dicts (tests only)."""
    return DoResult(
        ds=build_dataset(
            values or {},
            values_by_time,
            time_index,
            n_chains=n_chains,
            n_draws=n_draws,
            n_units_per_var=n_units_per_var,
        )
    )


def estimand_result_from_flat(
    values: dict[str, np.ndarray],
    outcome: str,
    treatment: str,
    estimand: str,
    values_by_time: dict[str, np.ndarray] | None = None,
    time_index: np.ndarray | None = None,
    *,
    n_chains: int | None = None,
    n_draws: int | None = None,
) -> EstimandResult:
    """Construct an :class:`EstimandResult` from flat numpy dicts (tests only)."""
    return EstimandResult(
        ds=build_dataset(
            values,
            values_by_time,
            time_index,
            n_chains=n_chains,
            n_draws=n_draws,
        ),
        outcome=outcome,
        treatment=treatment,
        estimand=estimand,
    )
