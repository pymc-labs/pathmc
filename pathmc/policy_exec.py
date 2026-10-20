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
"""Chunked evaluation of history-dependent panel treatment policies.

The fitted model is not mutated. Each call compiles a throwaway coherent
scan with the policy injected into ``step_fn``, then evaluates that scan
in draw/simulation chunks. Innovations are generated from a
``SeedSequence`` keyed by seed, variable, chain, draw, and simulation, so
chunk size does not change results and two policies with the same seed
share noise. ``do(policy=...)`` returns a :class:`PolicyRun`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pymc as pm
import xarray as xr

from pathmc.compile import compile_to_pymc, lag_base_kinds
from pathmc.idata import posterior as posterior_dataset
from pathmc.idata import DEFAULT_HDI_PROB
from pathmc.panel import normalize_initial_states
from pathmc.policy import (
    TimeSchedule,
    TreatmentPolicy,
    _POLICY_REQUIRES_SCAN,
    plan_policy,
    reject_unsupported_policy_model,
)
from pathmc.simulate import (
    DoResult,
    _execute_gaussian_panel,
    _stack_sample,
    _stack_sample_time,
)

_COUPLING = "common-random-numbers"


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
class PolicyRun:
    """Population summaries for one history-dependent policy.

    ``do(policy=...)`` returns this type, not a :class:`pathmc.simulate.DoResult`.
    A ``set=`` query still returns a ``DoResult``. The two cannot be subtracted.
    """

    info: PolicySimulationInfo
    population: xr.Dataset
    simulation_draws: xr.Dataset
    _trajectories: xr.Dataset | None = None

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
        return float(np.mean(_stack_sample(self.population[var])))

    def hdi(self, var: str, prob: float = DEFAULT_HDI_PROB) -> np.ndarray:
        """Return the posterior highest-density interval ``[lower, upper]``."""
        return DoResult(ds=self.population).hdi(var, prob=prob)

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
        return DoResult(ds=self.population).plot(
            var, vs=vs, observed=observed, ax=ax, prob=prob
        )

    def by_time(self, var: str) -> np.ndarray:
        """Return per-time posterior draws, shape ``(n_times, n_samples)``."""
        if "time" not in self.population[var].dims:
            raise ValueError(
                "Per-time data not available. "
                "Use do(simulate_over='time') to get per-time results."
            )
        return _stack_sample_time(self.population[var])

    @property
    def time_index(self) -> np.ndarray | None:
        """Time labels for the first axis of :meth:`by_time` results."""
        if "time" not in self.population.coords:
            return None
        return np.asarray(self.population["time"].values)

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


def policy_innovations(
    seed: int,
    states: Sequence[str],
    chain_labels: Sequence[Any],
    draw_labels: Sequence[Any],
    n_simulations: int,
    n_units: int,
    n_times: int,
    *,
    simulation_indices: Sequence[int] | None = None,
) -> dict[str, np.ndarray]:
    """Standard-normal noise keyed by variable, chain, draw, and simulation.

    Returns arrays shaped ``(chain, draw, simulation, unit, time)``. The
    same seed and labels reproduce the same arrays; chunk boundaries are
    not part of the key.
    """
    out: dict[str, np.ndarray] = {}
    simulations = (
        range(n_simulations) if simulation_indices is None else simulation_indices
    )
    for var in states:
        var_code = _label_code(var)
        arr = np.empty((
            len(chain_labels),
            len(draw_labels),
            n_simulations,
            n_units,
            n_times,
        ))
        for ci, chain in enumerate(chain_labels):
            for di, draw in enumerate(draw_labels):
                for si, sim in enumerate(simulations):
                    ss = np.random.SeedSequence([
                        _label_code(seed),
                        var_code,
                        _label_code(chain),
                        _label_code(draw),
                        int(sim),
                    ])
                    arr[ci, di, si] = np.random.default_rng(ss).standard_normal((
                        n_units,
                        n_times,
                    ))
        out[var] = arr
    return out


def run_policy_evaluation(
    model: Any,
    policy: TreatmentPolicy,
    *,
    set_values: dict[str, Any] | None,
    initial: Mapping[str, Any] | None,
    horizon: int | None,
    n_simulations: int | None,
    keep_trajectories: bool | None,
    random_seed: int | None,
    kind: str,
    simulate_over: str | None,
    batch_draws: int = 64,
    batch_sims: int = 32,
    innovations: Mapping[str, np.ndarray] | None = None,
) -> PolicyRun:
    """Compile a throwaway policy scan and return a :class:`PolicyRun`."""
    _reject_policy_arguments(
        policy,
        kind=kind,
        simulate_over=simulate_over,
        horizon=horizon,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    if not isinstance(policy, TreatmentPolicy):
        raise TypeError(
            "policy= must be a TreatmentPolicy, not "
            f"{type(policy).__name__}. Example: TreatmentPolicy("
            "name='responsive', variable='water', "
            "rule=where(current('stress') > 0.5, 1.0, 0.0))."
        )
    scan = getattr(model._gen_model, "_pathmc_panel_scan", None)
    if scan is None or model._panel_info is None:
        raise ValueError(_POLICY_REQUIRES_SCAN)
    n_times = int(scan.n_times)
    n_sim = 256 if n_simulations is None else _as_count(n_simulations, "n_simulations")
    if n_sim < 2:
        raise ValueError(
            "n_simulations must be an integer of at least 2 so a Monte "
            f"Carlo standard error is defined, got {n_sim}."
        )
    keep = False if keep_trajectories is None else bool(keep_trajectories)
    seed = _resolve_seed(random_seed)

    reject_unsupported_policy_model(
        model._spec,
        model._families,
        latent=set(model._latent),
        scan=True,
    )
    if horizon is None:
        raise ValueError(
            "horizon= is required and must equal the fitted time grid "
            f"({n_times} step(s)). Do not omit it; pass horizon={n_times}."
        )
    horizon_n = _as_count(horizon, "horizon")
    if horizon_n != n_times:
        raise ValueError(
            f"horizon={horizon_n} does not match the fitted time grid "
            f"({n_times} step(s)). Pass horizon={n_times}."
        )

    names = set(model._graph_info.endogenous) | set(model._graph_info.exogenous)
    plan = plan_policy(
        policy,
        model._graph_info,
        names=names,
        time_values=tuple(scan.time_values),
    )
    _reject_incomplete(model._data, names)
    schedules = _normalize_set(set_values, n_times, plan.target)
    arrays, baseline_record = _resolve_initial(
        model,
        plan_previous=plan.previous_deps,
        initial=initial,
    )
    scales = _unit_factors(
        model._scaling_factors,
        names,
        scan.unit_labels,
        model._panel_info.unit,
    )
    policy_model = compile_to_pymc(
        model._spec,
        model._data,
        model._design_matrices,
        families=model._families,
        panel_info=model._panel_info,
        pooling=model._pooling,
        latent=model._latent,
        graph_info=model._graph_info,
        priors=model._priors,
        generative=True,
        initial=arrays,
        policy_plan=plan,
        policy_scales=scales,
    )
    policy_scan = policy_model._pathmc_panel_scan
    states = tuple(policy_scan.gaussian_states)
    if not states:
        raise RuntimeError(
            "Policy compilation did not produce coherent Gaussian states. "
            "This is an internal error; policy= should have been rejected."
        )
    post = posterior_dataset(model._idata)
    noise = (
        {var: np.asarray(innovations[var], dtype=float) for var in states}
        if innovations is not None
        else None
    )
    if noise is not None:
        _check_innovations(noise, states, post, n_sim, int(scan.n_units), n_times)
    interventions = _model_interventions(
        schedules,
        scales,
        states,
        policy_scan.exogenous,
        policy_scan.unit_labels,
        policy_scan.time_values,
    )
    unit_means, retained_states = _execute_chunks(
        policy_model,
        post,
        states,
        noise,
        interventions,
        batch_draws=batch_draws,
        batch_sims=batch_sims,
        unit_labels=list(scan.unit_labels),
        time_values=list(scan.time_values),
        policy_name=policy.name,
        target=plan.target,
        n_simulations=n_sim,
        seed=seed,
        scales=scales,
        keep=keep,
    )
    population, sim_draws, trajectories = _pack_results(
        unit_means,
        retained_states,
        post,
        list(scan.time_values),
        list(scan.unit_labels),
    )
    info = PolicySimulationInfo(
        policy_name=policy.name,
        target=plan.target,
        rule=_rule_text(policy),
        unit_labels=tuple(str(label) for label in scan.unit_labels),
        time_values=tuple(scan.time_values),
        n_simulations=n_sim,
        horizon=horizon_n,
        seed=seed,
        baseline=baseline_record,
        posterior_id=_posterior_id(post, policy_model, states, model._spec, scales),
        exogenous_id=_exogenous_id(model, policy_scan, schedules),
        coupling=_COUPLING,
        weights=tuple(1.0 / int(scan.n_units) for _ in scan.unit_labels),
    )
    return PolicyRun(
        info=info,
        population=population,
        simulation_draws=sim_draws,
        _trajectories=trajectories,
    )


def _reject_policy_arguments(
    policy: Any,
    *,
    kind: str,
    simulate_over: str | None,
    horizon: int | None,
    n_simulations: int | None,
    random_seed: int | None,
) -> None:
    if kind != "mean":
        raise ValueError(
            "policy= integrates the stochastic history inside the scan. "
            "kind='predictive' is not a separate policy mode and does not "
            "match the static noise-free path. Omit kind, or pass kind='mean'."
        )
    if simulate_over != "time":
        raise ValueError(
            "policy= simulates a panel forward in time. Pass simulate_over='time'."
        )
    if horizon is not None:
        _as_count(horizon, "horizon")
    if n_simulations is not None:
        _as_count(n_simulations, "n_simulations")
    if random_seed is not None and (
        isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer))
    ):
        raise TypeError(
            "random_seed must be an integer, not "
            f"{type(random_seed).__name__}. Omit it to draw and record a seed."
        )


def _as_count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{where} must be an integer, not {type(value).__name__}. "
            "Pass a fitted time count or a simulation count of at least 2."
        )
    return int(value)


def _resolve_seed(random_seed: int | None) -> int:
    if random_seed is None:
        entropy = np.random.SeedSequence().entropy
        return int(np.asarray(entropy).ravel()[0]) % (2**31)
    return int(random_seed)


def _label_code(value: Any) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) % (2**32)
    digest = hashlib.sha256(str(value).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _rule_text(policy: TreatmentPolicy) -> str:
    rule = policy.rule
    if isinstance(rule, TimeSchedule):
        rendered = ", ".join(repr(value) for value in rule.values)
        return f"TimeSchedule({rendered})"
    if isinstance(rule, float):
        return repr(rule)
    return repr(rule)


def _reject_incomplete(data: Any, columns: set[str]) -> None:
    for name in sorted(columns):
        if name not in data.columns:
            continue
        values = np.asarray(data[name].to_numpy(), dtype=float)
        if np.isnan(values).any():
            raise ValueError(
                f"policy= requires complete data; column {name!r} contains "
                "missing values. Drop or fill those rows before fitting. "
                "Missing-data policies are not supported in v1."
            )


def _normalize_set(
    set_values: dict[str, Any] | None, n_times: int, target: str
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for var, val in (set_values or {}).items():
        if var == target:
            raise ValueError(
                f"set={{{var!r}: ...}} conflicts with policy target {target!r}. "
                "Drop that set= entry, or choose a different policy variable."
            )
        if isinstance(val, (list, tuple)):
            arr = np.asarray(val, dtype=float)
        elif isinstance(val, bool) or not isinstance(
            val, (int, float, np.integer, np.floating, np.ndarray)
        ):
            raise TypeError(
                f"set={{{var!r}}} must be a finite scalar or length-{n_times} "
                f"array, not {type(val).__name__}."
            )
        else:
            arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            series = np.full(n_times, float(arr))
        elif arr.shape == (n_times,):
            series = np.asarray(arr, dtype=float).copy()
        else:
            raise ValueError(
                f"set={{{var!r}}} must be a finite scalar or length-{n_times} "
                f"array, got shape {arr.shape}."
            )
        if not np.isfinite(series).all():
            raise ValueError(
                f"set={{{var!r}}} contains a non-finite value. Pass finite "
                "business-unit numbers."
            )
        out[str(var)] = series
    return out


def _resolve_initial(
    model: Any,
    *,
    plan_previous: tuple[str, ...],
    initial: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], tuple[tuple[str, tuple[float, ...]], ...]]:
    scan = model._gen_model._pathmc_panel_scan
    observed = set(scan.initial_nodes)
    known = set(model._graph_info.endogenous) | set(model._graph_info.exogenous)
    extra = sorted(set(plan_previous) - observed)
    baseline = model._initial_baseline
    if initial is None:
        if baseline is None or not baseline.explicit:
            raise ValueError(
                "policy= requires an explicit pre-sample baseline. Pass "
                f"initial= covering {sorted(observed | set(extra))} at model() "
                "or at do(). A legacy inferred first row is not an explicit "
                "baseline."
            )
        if extra:
            raise ValueError(
                "initial= must include policy previous references "
                f"{extra}. The fitted baseline only covers "
                f"{sorted(baseline.values)}. Pass the full mapping; "
                "partial maps are not mixed with the stored baseline."
            )
        arrays = {
            name: np.asarray(value.values, dtype=float)
            for name, value in baseline.values.items()
        }
    else:
        unknown = sorted(set(initial) - known)
        if unknown:
            raise ValueError(
                f"initial= names {unknown} are not fitted variables. "
                f"Available names: {sorted(known)}."
            )
        required = sorted(observed | set(extra))
        unused = sorted(set(initial) - set(required))
        if unused:
            raise ValueError(
                f"initial= names {unused} are unused by the lag or policy "
                f"previous references. Remove them; required keys: {required}."
            )
        missing = sorted((observed | set(extra)) - set(initial))
        if missing:
            raise ValueError(
                "initial= must cover the observed lag bases and any "
                f"policy previous references. Missing {missing}. "
                "Pass the full mapping; partial maps are not mixed with "
                "the stored baseline."
            )
        kinds = lag_base_kinds(model._spec, required)
        normalized = normalize_initial_states(
            initial,
            required=required,
            unit_labels=scan.unit_labels,
            data=model._data,
            panel_info=model._panel_info,
            scaling_factors=model._scaling_factors,
            kinds=kinds,
            latent=tuple(sorted(model._latent)),
        )
        arrays = {
            name: np.asarray(value.values, dtype=float)
            for name, value in normalized.items()
        }
    record = tuple(
        (name, tuple(float(v) for v in arrays[name])) for name in sorted(arrays)
    )
    compiled = {name: arrays[name] for name in sorted(observed | set(plan_previous))}
    return compiled, record


def _unit_factors(
    scaling: Any,
    names: set[str],
    unit_labels: Sequence[Any],
    unit_col: str,
) -> dict[str, np.ndarray]:
    if not scaling:
        return {}
    return {
        name: scaling.unit_divisors(name, unit_labels, unit_col)
        for name in sorted(names)
        if scaling.has_factor(name)
    }


def _model_interventions(
    schedules: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    states: tuple[str, ...],
    exogenous: tuple[str, ...],
    unit_labels: Sequence[Any],
    time_values: Sequence[Any],
) -> dict[str, xr.DataArray]:
    known = set(states) | set(exogenous)
    n_units = len(unit_labels)
    out: dict[str, xr.DataArray] = {}
    for var, series in schedules.items():
        if var not in known:
            raise ValueError(
                f"Unknown set= variable {var!r}. Use an endogenous or "
                f"exogenous column. Available: {sorted(known)}."
            )
        business = np.broadcast_to(series[:, None], (series.shape[0], n_units)).astype(
            float
        )
        factor = scales.get(var)
        model_units = business if factor is None else business / factor.reshape(1, -1)
        out[var] = xr.DataArray(
            np.ascontiguousarray(model_units).T,
            dims=("unit", "time"),
            coords={"unit": list(unit_labels), "time": list(time_values)},
        )
    return out


def _check_innovations(
    noise: Mapping[str, np.ndarray],
    states: tuple[str, ...],
    post: xr.Dataset,
    n_sim: int,
    n_units: int,
    n_times: int,
) -> None:
    expected = (
        post.sizes["chain"],
        post.sizes["draw"],
        n_sim,
        n_units,
        n_times,
    )
    missing = sorted(set(states) - set(noise))
    if missing:
        raise ValueError(f"innovations must cover {list(states)}. Missing {missing}.")
    for var, values in noise.items():
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"innovations[{var!r}] must be finite and shaped {expected}, "
                f"got {values.shape}."
            )


def _chunk_innovations(
    noise: Mapping[str, np.ndarray] | None,
    states: tuple[str, ...],
    post_slice: xr.Dataset,
    chain_i: int,
    draw_idx: np.ndarray,
    sim_idx: np.ndarray,
    unit_labels: Sequence[Any],
    time_values: Sequence[Any],
    seed: int,
) -> xr.Dataset:
    """One chunk of shared noise, labelled to the posterior slice."""
    data = {}
    generated = (
        policy_innovations(
            seed,
            states,
            list(post_slice.coords["chain"].values),
            list(post_slice.coords["draw"].values),
            len(sim_idx),
            len(unit_labels),
            len(time_values),
            simulation_indices=sim_idx.tolist(),
        )
        if noise is None
        else None
    )
    for var in states:
        if generated is not None:
            block = generated[var]
        else:
            assert noise is not None
            block = noise[var][chain_i][draw_idx[:, None], sim_idx[None, :]][None, ...]
        data[var] = (
            ("chain", "draw", "simulation", "unit", "time"),
            block,
        )
    return xr.Dataset(
        data,
        coords={
            "chain": np.asarray(post_slice.coords["chain"].values),
            "draw": np.asarray(post_slice.coords["draw"].values),
            "simulation": np.asarray(sim_idx),
            "unit": list(unit_labels),
            "time": list(time_values),
        },
    )


def _execute_chunks(
    policy_model: pm.Model,
    post: xr.Dataset,
    states: tuple[str, ...],
    noise: Mapping[str, np.ndarray] | None,
    interventions: dict[str, xr.DataArray],
    *,
    batch_draws: int,
    batch_sims: int,
    unit_labels: list[Any],
    time_values: list[Any],
    policy_name: str,
    target: str,
    n_simulations: int,
    seed: int,
    scales: dict[str, np.ndarray],
    keep: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    n_sim = n_simulations
    n_unit = len(unit_labels)
    n_time = len(time_values)
    output_names = tuple(dict.fromkeys((*states, target)))
    unit_means = {
        var: np.empty((n_chain, n_draw, n_sim, n_time)) for var in output_names
    }
    retained = (
        {
            var: np.empty((n_chain, n_draw, n_sim, n_unit, n_time))
            for var in output_names
        }
        if keep
        else None
    )
    scan_info = policy_model._pathmc_panel_scan
    action_name = scan_info.policy_action or "_policy_action"
    chain_labels = list(post.coords["chain"].values)
    draw_labels = list(post.coords["draw"].values)
    for chain_i in range(n_chain):
        for draw_start in range(0, n_draw, batch_draws):
            draw_idx = np.arange(draw_start, min(draw_start + batch_draws, n_draw))
            for sim_start in range(0, n_sim, batch_sims):
                sim_idx = np.arange(sim_start, min(sim_start + batch_sims, n_sim))
                post_slice = post.isel(chain=[chain_i], draw=draw_idx)
                result = _execute_gaussian_panel(
                    policy_model,
                    post_slice,
                    scan_info,
                    innovations=_chunk_innovations(
                        noise,
                        states,
                        post_slice,
                        chain_i,
                        draw_idx,
                        sim_idx,
                        scan_info.unit_labels,
                        scan_info.time_values,
                        seed,
                    ),
                    interventions=interventions or None,
                )
                chunk_action = np.asarray(result[action_name].values)[0]
                _reject_nonfinite_action(
                    chunk_action,
                    policy_name=policy_name,
                    target=target,
                    unit_labels=unit_labels,
                    time_values=time_values,
                    draw_labels=[draw_labels[i] for i in draw_idx],
                    sim_idx=sim_idx,
                    chain_label=chain_labels[chain_i],
                )
                for var in output_names:
                    values = np.asarray(
                        result[var if var in states else action_name].values
                    )[0]
                    business = _unscale(values, scales.get(var))
                    index = (chain_i, draw_idx[:, None], sim_idx[None, :])
                    unit_means[var][index] = business.mean(axis=2)
                    if retained is not None:
                        retained[var][index] = business
    return unit_means, retained


def _reject_nonfinite_action(
    action: np.ndarray,
    *,
    policy_name: str,
    target: str,
    unit_labels: list[Any],
    time_values: list[Any],
    draw_labels: list[Any],
    sim_idx: np.ndarray,
    chain_label: Any,
) -> None:
    bad = np.argwhere(~np.isfinite(action))
    if len(bad) == 0:
        return
    di, si, ui, ti = (int(v) for v in bad[0])
    raise ValueError(
        f"Policy {policy_name!r} assigned a non-finite action for "
        f"{target!r} at time {time_values[ti]!r}, unit {unit_labels[ui]!r} "
        f"(chain {chain_label!r}, draw {draw_labels[di]!r}, "
        f"simulation {int(sim_idx[si])}). Check the rule for division by "
        "zero or a non-finite current() value."
    )


def _unscale(values: np.ndarray, factor: np.ndarray | None) -> np.ndarray:
    if factor is None:
        return values
    shape = (1,) * (values.ndim - 2) + (factor.shape[0], 1)
    return values * factor.reshape(shape)


def _pack_results(
    unit_means: dict[str, np.ndarray],
    retained: dict[str, np.ndarray] | None,
    post: xr.Dataset,
    time_values: list[Any],
    unit_labels: list[Any],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    """Pack unit-averaged replicate summaries and optional business-unit paths.

    ``unit_means`` arrays are ``(chain, draw, simulation, time)``.
    """
    time = np.asarray(time_values)
    chain = post.coords["chain"].values
    draw = post.coords["draw"].values
    n_sim = next(iter(unit_means.values())).shape[2]
    sim = np.arange(n_sim)
    units = np.asarray(unit_labels)
    population = {}
    sim_draws = {}
    trajectories = {}
    for var, unit_mean in unit_means.items():
        population[var] = (
            ("time", "chain", "draw"),
            unit_mean.mean(axis=2).transpose(2, 0, 1),
        )
        sim_draws[var] = (
            ("time", "chain", "draw", "simulation"),
            unit_mean.transpose(3, 0, 1, 2),
        )
        if retained is not None:
            trajectories[var] = (
                ("chain", "draw", "simulation", "unit", "time"),
                retained[var],
            )
    coords = {"time": time, "chain": chain, "draw": draw}
    pop = xr.Dataset(population, coords=coords)
    sims = xr.Dataset(sim_draws, coords={**coords, "simulation": sim})
    traj = (
        xr.Dataset(trajectories, coords={**coords, "simulation": sim, "unit": units})
        if retained is not None
        else None
    )
    return pop, sims, traj


def _posterior_id(
    post: xr.Dataset,
    policy_model: pm.Model,
    states: tuple[str, ...],
    spec: Any,
    scales: Mapping[str, np.ndarray],
) -> str:
    noise = {f"carry_innovations_{var}" for var in states}
    names = [rv.name for rv in policy_model.free_RVs if rv.name not in noise]
    arrays = {name: np.asarray(post[name].values) for name in names}
    arrays["model:spec"] = np.asarray(repr(spec))
    for name, scale in scales.items():
        arrays[f"model:scale:{name}"] = np.asarray(scale)
    for name in names:
        arrays[f"dimensions:{name}"] = np.asarray(post[name].dims)
        for dim in post[name].dims:
            arrays[f"coordinate:{dim}"] = np.asarray([
                repr(label) for label in post.coords[dim].values
            ])
    return _hash_arrays(arrays)


def _exogenous_id(model: Any, scan: Any, schedules: dict[str, np.ndarray]) -> str:
    arrays: dict[str, np.ndarray] = {}
    data = model._data
    sort_idx = np.asarray(scan.sort_idx)
    n_units = int(scan.n_units)
    n_times = int(scan.n_times)
    for name in scan.exogenous:
        if name not in data.columns:
            continue
        flat = np.asarray(data[name].to_numpy(), dtype=float)[sort_idx]
        arrays[f"data:{name}"] = flat.reshape(n_units, n_times)
    for name, series in schedules.items():
        arrays[f"set:{name}"] = np.asarray(series, dtype=float)
    return _hash_arrays(arrays)


def _hash_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()
