# Copyright 2025 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Population summaries for one history-dependent policy run.

``do(policy=...)`` returns :class:`PolicyRun`, not a
:class:`pathmc.simulate.DoResult`. The executor lives in
:mod:`pathmc.policy_exec` and does not own this type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from pathmc.idata import DEFAULT_HDI_PROB
from pathmc.idata import hdi as compute_hdi
from pathmc.simulate import _DrawStorageMixin, plot_panel_trajectory

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["PolicyRun", "PolicySimulationInfo"]


@dataclass(frozen=True)
class PolicySimulationInfo:
    """Identity of one policy run. Contrast requires every field except name."""

    policy_name: str
    target: str
    rule: str
    unit_labels: tuple[str, ...]
    time_values: tuple[Any, ...]
    n_simulations: int
    horizon: int
    seed: int
    baseline: tuple[tuple[str, tuple[float, ...]], ...]
    posterior_id: str
    exogenous_id: str
    coupling: str
    weights: tuple[float, ...]

    def matches(self, other: PolicySimulationInfo) -> list[str]:
        """Return identity fields that differ. Names are allowed to differ."""
        mismatched: list[str] = []
        for field in (
            "target",
            "unit_labels",
            "time_values",
            "n_simulations",
            "horizon",
            "seed",
            "baseline",
            "posterior_id",
            "exogenous_id",
            "coupling",
            "weights",
        ):
            if getattr(self, field) != getattr(other, field):
                mismatched.append(field)
        return mismatched


def _time_label_index(times: list[Any], label: Any, where: str) -> int:
    for index, value in enumerate(times):
        if value == label:
            return index
    raise ValueError(
        f"aggregate_time {where}={label!r} is not a fitted time label. "
        f"Available labels: {times}."
    )


def _reduce_policy_time(
    ds: xr.Dataset,
    var: str,
    reduction: str,
    start: Any,
    end: Any,
) -> xr.Dataset:
    """Reduce one variable's time dimension, dropping every other variable."""
    if var not in ds:
        raise KeyError(
            f"{var!r} is not in this policy result. Available: {list(ds.data_vars)}."
        )
    values = ds[var]
    if "time" not in values.dims:
        raise ValueError(
            f"{var!r} has no time dimension. aggregate_time() was already "
            "applied, or this result is not a trajectory."
        )
    times = list(values.coords["time"].values)
    start_i = 0 if start is None else _time_label_index(times, start, "start")
    end_i = len(times) - 1 if end is None else _time_label_index(times, end, "end")
    if end_i < start_i:
        raise ValueError(f"aggregate_time end={end!r} precedes start={start!r}.")
    window = values.isel(time=slice(start_i, end_i + 1))
    if reduction == "sum":
        reduced = window.sum(dim="time")
    elif reduction == "mean":
        reduced = window.mean(dim="time")
    elif reduction == "last":
        reduced = window.isel(time=-1)
    else:
        raise ValueError(
            "aggregate_time reduction must be 'sum', 'mean', or 'last'. "
            f"Got {reduction!r}."
        )
    return reduced.to_dataset()


def _require_matching_layout(left: xr.Dataset, right: xr.Dataset) -> None:
    """Reject implicit intersections or broadcasting in a paired contrast."""
    if (
        left.sizes != right.sizes
        or set(left.data_vars) != set(right.data_vars)
        or not left.coords.to_dataset().equals(right.coords.to_dataset())
        or any(left[var].dims != right[var].dims for var in left.data_vars)
    ):
        raise ValueError(
            "Cannot contrast policy results: result coordinates, dimensions, "
            "or variables differ. Rerun both on the same posterior and "
            "reporting grid, and apply the same aggregation before subtraction."
        )


@dataclass(frozen=True)
class PolicyRun(_DrawStorageMixin):
    """Population summaries for one history-dependent policy.

    ``do(policy=...)`` returns this type, not a :class:`pathmc.simulate.DoResult`.
    A ``set=`` query still returns a ``DoResult``. The two cannot be subtracted.
    """

    info: PolicySimulationInfo
    population: xr.Dataset
    simulation_draws: xr.Dataset
    _trajectories: xr.Dataset | None = None

    @property
    def _ds(self) -> xr.Dataset:
        return self.population

    @property
    def dataset(self) -> xr.Dataset:
        """Unit-mean posterior draws, dims ``("time", "chain", "draw")``."""
        return self.population

    def simulation_info(self) -> PolicySimulationInfo:
        """Identity of this run. Contrast requires every field except name."""
        return self.info

    @property
    def trajectories(self) -> xr.Dataset:
        """Full unit paths. Requires ``keep_trajectories=True``."""
        if self._trajectories is None:
            raise ValueError(
                "trajectories were not retained. Pass keep_trajectories=True "
                "to do(); the default omits them."
            )
        return self._trajectories

    def mean(self, var: str) -> float:
        """Return the posterior mean of *var* under this policy."""
        return float(np.mean(self._draw(var)))

    def hdi(self, var: str, prob: float = DEFAULT_HDI_PROB) -> np.ndarray:
        """Return the posterior highest-density interval ``[lower, upper]``."""
        return compute_hdi(self._draw(var), prob=prob)

    def plot(
        self,
        var: str,
        *,
        vs: str | None = None,
        observed: np.ndarray | None = None,
        ax: matplotlib.axes.Axes | None = None,
        prob: float = DEFAULT_HDI_PROB,
    ) -> matplotlib.figure.Figure:
        """Plot the posterior trajectory and HDI, with an optional observed series.

        Pass ``ax`` to reuse an axes, ``prob`` for the interval mass, and
        ``observed`` for a time-aligned overlay. Aggregate results have no
        time dimension; use :meth:`hdi` to summarize their uncertainty.
        """
        if var not in self.population.data_vars:
            available = sorted(str(name) for name in self.population.data_vars)
            raise KeyError(
                f"Unknown variable '{var}'. Available variables: {available}"
            )
        if self.time_index is None or "time" not in self.population[var].dims:
            raise ValueError(
                "PolicyRun.plot() needs a time dimension. aggregate_time() "
                "removed it; use hdi() on the reduced result."
            )
        overlay = None
        if observed is not None:
            overlay = np.asarray(observed, dtype=float)
        elif vs == "observed":
            raise ValueError(
                f"No observed series available for '{var}'. Pass observed= "
                f"with a length-{len(self.time_index)} array of unit-mean values."
            )
        elif vs is not None:
            raise ValueError(
                f"Unknown vs={vs!r}. Supported values: 'observed' or omit vs."
            )
        return plot_panel_trajectory(
            var,
            self._draw_by_time(var),
            self.time_index,
            overlay=overlay,
            ax=ax,
            prob=prob,
        )

    def by_time(self, var: str) -> np.ndarray:
        """Return per-time posterior draws, shape ``(n_times, n_samples)``."""
        if self.time_index is None or "time" not in self.population[var].dims:
            raise ValueError(
                f"{var!r} has no time dimension. aggregate_time() removed it; "
                "use mean() or hdi() on the reduced result."
            )
        return self._draw_by_time(var)

    def aggregate_time(
        self,
        var: str,
        *,
        reduction: str = "sum",
        start: Any = None,
        end: Any = None,
    ) -> PolicyRun:
        """Reduce the time dimension of one policy variable.

        ``reduction`` is ``"sum"``, ``"mean"``, or ``"last"``. ``start`` and
        ``end`` are inclusive fitted time labels. The result has no time
        dimension, so :meth:`mean` describes that scalar estimand.
        """
        return PolicyRun(
            info=self.info,
            population=_reduce_policy_time(self.population, var, reduction, start, end),
            simulation_draws=_reduce_policy_time(
                self.simulation_draws, var, reduction, start, end
            ),
            _trajectories=None,
        )

    def simulation_mcse(self, var: str, *, by: str | None = None) -> Any:
        """Per-posterior-draw Monte Carlo standard error of a policy estimand.

        Computed within each posterior draw as the simulation standard
        deviation divided by ``sqrt(n_simulations)``. The chain and draw
        dimensions are preserved. A time-indexed result keeps ``time`` as
        well; call :meth:`aggregate_time` first for a cumulative or terminal
        estimand. ``by='time'`` is accepted and does not average draws.
        """
        draws = self.simulation_draws[var]
        if by not in (None, "time"):
            raise ValueError(
                f"simulation_mcse by= must be 'time' or omitted. Got {by!r}."
            )
        if by == "time" and "time" not in draws.dims:
            raise ValueError(
                "simulation_mcse by='time' needs a time dimension. "
                "This result was already reduced with aggregate_time()."
            )
        std = draws.std(dim="simulation", ddof=1)
        return std / np.sqrt(draws.sizes["simulation"])

    def contrast(self, other: PolicyRun) -> PolicyRun:
        """Paired difference. Innovations already match when the seed matches."""
        if not isinstance(other, PolicyRun):
            return NotImplemented
        mismatched = self.info.matches(other.info)
        if mismatched:
            raise ValueError(
                "Cannot contrast policies "
                f"{self.info.policy_name!r} and {other.info.policy_name!r}: "
                f"{', '.join(mismatched)} differ. Rerun both with the same "
                "seed, initial baseline, horizon, simulation count, and "
                "fitted posterior. Refitting is unnecessary when those match."
            )
        _require_matching_layout(self.population, other.population)
        _require_matching_layout(self.simulation_draws, other.simulation_draws)
        trajectories = None
        if self._trajectories is not None and other._trajectories is not None:
            _require_matching_layout(self._trajectories, other._trajectories)
            trajectories = self._trajectories - other._trajectories
        name = f"{self.info.policy_name} - {other.info.policy_name}"
        return PolicyRun(
            info=PolicySimulationInfo(
                policy_name=name,
                target=self.info.target,
                rule=f"{self.info.rule} - {other.info.rule}",
                unit_labels=self.info.unit_labels,
                time_values=self.info.time_values,
                n_simulations=self.info.n_simulations,
                horizon=self.info.horizon,
                seed=self.info.seed,
                baseline=self.info.baseline,
                posterior_id=self.info.posterior_id,
                exogenous_id=self.info.exogenous_id,
                coupling=self.info.coupling,
                weights=self.info.weights,
            ),
            population=self.population - other.population,
            simulation_draws=self.simulation_draws - other.simulation_draws,
            _trajectories=trajectories,
        )

    def __sub__(self, other: PolicyRun) -> PolicyRun:
        """Paired difference of two policy results."""
        if not isinstance(other, PolicyRun):
            return NotImplemented
        return self.contrast(other)
