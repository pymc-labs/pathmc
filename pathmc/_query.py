#   Copyright 2025 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Instance-owned execution plans for the initial linear Gaussian query slice."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any, TYPE_CHECKING, cast

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from pytensor.graph.replace import graph_replace, vectorize_graph
from pytensor.graph.basic import Constant, Variable
from pytensor.tensor.variable import TensorVariable

from pathmc.idata import posterior
from pathmc.simulate import (
    DoResult,
    _broadcast_intervention,
    _business_exog_value,
    _chain_draw_ones,
    _intervention_array,
)

try:
    from pymc.model.transform.optimization import freeze_model
except ImportError:  # Older PyMC releases use the existing prediction executor.
    freeze_model = None

if TYPE_CHECKING:
    from pathmc._model import PathModel


class QueryCache:
    """Bound retained plans and serialize access to compiled mutable state."""

    def __init__(self, limit: int = 16) -> None:
        self.limit = limit
        self.entries: OrderedDict[tuple, Any] = OrderedDict()
        self.lock = RLock()

    def clear(self) -> None:
        """Discard all plans when the owning model is recompiled or refitted."""
        with self.lock:
            self.entries.clear()

    def __getstate__(self) -> dict[str, int]:
        """Serialize configuration without process-local functions or locks."""
        return {"limit": self.limit}

    def __setstate__(self, state: dict[str, int]) -> None:
        """Start deserialized instances with an empty cache and a fresh lock."""
        QueryCache.__init__(self, state["limit"])

    def get(self, key: tuple, prepare: Callable[[], Any]) -> Any:
        """Prepare once, retaining only the most recently used entries."""
        with self.lock:
            if key not in self.entries:
                value = prepare()
                self.entries[key] = value
                if len(self.entries) > self.limit:
                    self.entries.popitem(last=False)
            self.entries.move_to_end(key)
            return self.entries[key]


def _linear_gaussian_model(model: PathModel) -> bool:
    """Identify the model slice independently of the selected compiler backend."""
    if (
        model._data is None
        or model._panel_info is not None
        or model._pooling is not None
        or model._latent
        or model._spec.residual_covs
        or model._scaling_factors is not None
    ):
        return False
    for reg in model._spec.regressions:
        if model._families.get(reg.lhs, "gaussian") != "gaussian":
            return False
        if any(
            term.transform
            or term.basis
            or term.categorical
            or term.lag_of
            or term.interaction_of
            for term in reg.terms
        ):
            return False
    names = model._graph_info.endogenous | model._graph_info.exogenous
    return all(
        name in model._data.columns
        and np.asarray(model._data[name].to_numpy()).dtype.kind in "biuf"
        and np.isfinite(model._data[name].to_numpy()).all()
        for name in names
    )


def eligible(model: PathModel) -> bool:
    """Limit optimization to the observed numeric linear Gaussian contract."""
    return _linear_gaussian_model(model) and str(pytensor.config.mode) in (
        "Mode",
        "FAST_RUN",
        "FAST_COMPILE",
        "CVM",
        "VM",
        "NUMBA",
    )


def _current_exogenous(model: PathModel) -> dict[str, np.ndarray]:
    """Read and validate the estimation graph's current numeric row inputs."""
    assert model._data is not None and model._pymc_model is not None
    rows = len(model._data)
    values = {}
    for name in sorted(model._graph_info.exogenous):
        value = model._pymc_model[name].get_value()
        if value.shape != (rows,):
            raise ValueError(
                f"Replacement data for '{name}' has shape {value.shape}, "
                f"but this model requires ({rows},). Restore the fitted row "
                "count or build a new model for the replacement data."
            )
        values[name] = value
    return values


@contextmanager
def current_data(model: PathModel) -> Iterator[None]:
    """Give delegated queries validated current inputs without retaining updates."""
    if not _linear_gaussian_model(model):
        yield
        return
    assert model._gen_model is not None
    values = _current_exogenous(model)
    previous = {name: model._gen_model[name].get_value() for name in values}
    try:
        for name, value in values.items():
            model._gen_model[name].set_value(value)
        yield
    finally:
        for name, value in previous.items():
            model._gen_model[name].set_value(value)


def _graph_signature(model: PathModel) -> tuple:
    """Identify graph generation, mutable input layout, and compiler settings."""
    assert model._gen_model is not None and model._pymc_model is not None
    return (
        id(model._gen_model),
        tuple((name, id(var)) for name, var in model._gen_model.named_vars.items()),
        tuple((name, id(var)) for name, var in model._pymc_model.named_vars.items()),
        tuple(
            (name, value.get_value().shape, str(value.dtype))
            for name, value in model._pymc_model.named_vars.items()
            if hasattr(value, "get_value")
        ),
        str(pytensor.config.mode),
        pytensor.config.floatX,
        pytensor.config.linker,
        pytensor.config.optimizer,
        pytensor.config.optimizer_excluding,
        pytensor.config.optimizer_including,
    )


def _mean_parameters(model: PathModel) -> tuple[str, ...]:
    """Order the coefficient inputs shared by mean preparation and evaluation."""
    assert model._gen_model is not None
    return tuple(
        f"beta_{reg.lhs}"
        for reg in model._spec.regressions
        if f"beta_{reg.lhs}" in model._gen_model.named_vars
    )


def _mean_signature(model: PathModel, post: xr.Dataset) -> tuple:
    """Key symbolic mean inputs without runtime sample axes or unused draws."""
    return (
        _graph_signature(model),
        tuple(
            (name, post[name].dims[2:], post[name].shape[2:], str(post[name].dtype))
            for name in _mean_parameters(model)
        ),
    )


def _prediction_signature(model: PathModel, post: xr.Dataset) -> tuple:
    """Retain prediction's trace schema and sampled-versus-computed roles."""
    return (
        _graph_signature(model),
        tuple(
            (name, value.dims, value.shape, str(value.dtype))
            for name, value in post.data_vars.items()
        ),
    )


@dataclass
class MeanPlan:
    """A symbolic posterior/scenario batch with no stored numeric query values."""

    function: Any
    parameters: tuple[str, ...]
    targets: tuple[str, ...]
    exogenous: tuple[str, ...]
    outputs: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    rows: int
    average: bool


def prepare_mean(model: PathModel, targets: tuple[str, ...], average: bool) -> MeanPlan:
    """Clone existing compiled mu graphs, then vectorize their runtime inputs."""
    assert model._gen_model is not None and model._data is not None
    source = model._gen_model
    rows = len(model._data)
    inputs = {
        name: pt.vector(name=f"query_{name}", dtype=source[name].dtype)
        for name in targets
    }
    replacements = {source[name]: value for name, value in inputs.items()}
    outputs = []
    names = []
    for name in model._graph_info.topological_order:
        if name not in model._graph_info.endogenous or name in targets:
            continue
        mu = graph_replace(source[f"mu_{name}"], replacements, strict=False)
        replacements[source[name]] = mu
        outputs.append(mu)
        names.append(name)
    parameters = _mean_parameters(model)
    exogenous = tuple(sorted(model._graph_info.exogenous - set(targets)))
    vectorized = {}
    for name in parameters:
        rv = source[name]
        vectorized[rv] = pt.tensor(
            dtype=rv.dtype, shape=(None, 1, *rv.type.shape), name=f"query_{name}"
        )
    for name in targets:
        vectorized[inputs[name]] = pt.tensor3(
            dtype=source[name].dtype, shape=(1, None, rows), name=f"scenarios_{name}"
        )
    for name in exogenous:
        vectorized[source[name]] = pt.tensor3(
            dtype=source[name].dtype, shape=(1, 1, rows), name=f"data_{name}"
        )
    batched = cast(list[TensorVariable], vectorize_graph(outputs, vectorized))
    if average:
        batched = [pt.mean(value, axis=-1) for value in batched]
    function = pytensor.function(
        list(vectorized.values()),
        batched,
        on_unused_input="ignore",
        allow_input_downcast=True,
    )
    return MeanPlan(
        function,
        parameters,
        targets,
        exogenous,
        tuple(names),
        tuple(value.dtype for value in batched),
        rows,
        average,
    )


def mean_scenarios(
    model: PathModel,
    scenarios: Sequence[Mapping[str, Any]],
    average: bool = True,
    *,
    max_elements: int = 1_000_000,
) -> list[DoResult] | None:
    """Evaluate bounded posterior/scenario batches, or delegate unsupported input.

    A scenario is an internal mapping; public intervention vectors remain row
    values. Reductions execute inside the compiled graph so averaged queries never
    allocate a full posterior-by-scenario-by-observation result in Python.
    """
    if not eligible(model) or not scenarios:
        return None
    assert model._data is not None and model._idata is not None
    assert model._gen_model is not None and model._pymc_model is not None
    targets = tuple(sorted(scenarios[0]))
    known = model._graph_info.endogenous | model._graph_info.exogenous
    if not set(targets) <= known or any(tuple(sorted(s)) != targets for s in scenarios):
        return None
    rows = len(model._data)
    for scenario in scenarios:
        for name in targets:
            values = np.asarray(scenario[name], dtype=float)
            if values.size != 1:
                _intervention_array(values, rows)
            if not np.isfinite(values).all():
                return None
    post = posterior(model._idata)
    for reg in model._spec.regressions:
        name = f"beta_{reg.lhs}"
        if name in model._gen_model.named_vars and (
            name not in post
            or post[name].dims != ("chain", "draw", f"{reg.lhs}_predictors")
            or post[name].sizes[f"{reg.lhs}_predictors"]
            != len(model._gen_model.coords[f"{reg.lhs}_predictors"])
        ):
            return None
    cache = model._query_cache
    with cache.lock:
        current = _current_exogenous(model)
        key = ("mean", _mean_signature(model, post), targets, average)
        plan: MeanPlan = cache.get(key, lambda: prepare_mean(model, targets, average))
        chains, draws = post.sizes["chain"], post.sizes["draw"]
        samples = chains * draws
        n_scenarios = len(scenarios)
        # Bound all intermediate batch axes; one row vector is the minimum work.
        scenario_batch = min(n_scenarios, max(1, max_elements // max(1, rows)))
        draw_batch = max(1, max_elements // max(1, rows * scenario_batch))
        tail = () if average else (rows,)
        result = {
            name: np.empty((samples, n_scenarios, *tail), dtype=dtype)
            for name, dtype in zip(plan.outputs, plan.output_dtypes, strict=True)
        }
        parameters = [
            post[name].values.reshape((samples, 1, *post[name].shape[2:]))
            for name in plan.parameters
        ]
        data = [current[name][None, None, :] for name in plan.exogenous]
        for arm in range(0, n_scenarios, scenario_batch):
            end = min(n_scenarios, arm + scenario_batch)
            interventions = [
                np.stack([
                    _intervention_array(s[name], rows) for s in scenarios[arm:end]
                ])[None].astype(model._gen_model[name].dtype, copy=False)
                for name in plan.targets
            ]
            for start in range(0, samples, draw_batch):
                stop = min(samples, start + draw_batch)
                inputs = [v[start:stop] for v in parameters]
                outputs = plan.function(*inputs, *interventions, *data)
                for name, value in zip(plan.outputs, outputs, strict=True):
                    result[name][start:stop, arm:end] = value
        ones = _chain_draw_ones(post)
        results = []
        for arm, scenario in enumerate(scenarios):
            variables = {}
            for name in model._graph_info.topological_order:
                if name in scenario:
                    variables[name] = _broadcast_intervention(
                        ones, scenario[name], rows
                    )
                elif name in model._graph_info.exogenous:
                    variables[name] = ones * _business_exog_value(
                        name, model._data, None, None
                    )
                else:
                    dims = ("chain", "draw") + (() if average else ("unit",))
                    coords = {"chain": post.chain, "draw": post.draw}
                    if not average:
                        coords["unit"] = np.arange(rows)
                    variables[name] = xr.DataArray(
                        result[name][:, arm].reshape((chains, draws, *tail)),
                        dims=dims,
                        coords=coords,
                    )
            results.append(DoResult(ds=xr.Dataset(variables)))
        return results


def prediction_model(model: PathModel, kwargs: Mapping[str, Any]) -> pm.Model:
    """Reuse PyMC's cache only where its data/RNG contract matches prediction."""
    assert model._pymc_model is not None and model._idata is not None
    if (
        freeze_model is None
        or not eligible(model)
        or set(kwargs)
        - {"random_seed", "progressbar", "extend_inferencedata", "var_names"}
        or any(
            isinstance(value, Variable) and not isinstance(value, Constant)
            for value in model._pymc_model.rvs_to_initial_values.values()
        )
    ):
        return model._pymc_model
    post = posterior(model._idata)
    selected = kwargs.get("var_names")
    outputs = None if selected is None else tuple(selected)
    observed_schema = (
        tuple(model._idata["observed_data"].data_vars)
        if "observed_data" in model._idata
        else ()
    )
    key = ("predict", _prediction_signature(model, post), outputs, observed_schema)
    frozen = model._query_cache.get(key, lambda: freeze_model(model._pymc_model))
    for name, var in model._pymc_model.named_vars.items():
        if hasattr(var, "get_value"):
            target = frozen[name]
            # Freezing a graph must never hide a mutable input behind a constant.
            if not hasattr(target, "set_value"):
                return model._pymc_model
            target.set_value(var.get_value())
    return frozen


def run_prediction(model: PathModel, kwargs: Mapping[str, Any]) -> Any:
    """Evaluate prediction with stable runtime roles for response means."""
    assert model._idata is not None and model._pymc_model is not None
    execution_model = prediction_model(model, kwargs)
    post = posterior(model._idata)
    means = [
        f"mu_{reg.lhs}" for reg in model._spec.regressions if f"mu_{reg.lhs}" in post
    ]
    with execution_model:
        if execution_model is model._pymc_model or not means:
            return pm.sample_posterior_predictive(model._idata, **kwargs)
        # Stored response means otherwise switch between trace inputs and graph
        # evaluation when data changes, forcing another PyMC compilation.
        trace = model._idata.copy(deep=False)
        trace["posterior"] = xr.DataTree(post.drop_vars(means))
        result = pm.sample_posterior_predictive(trace, **kwargs)
    if kwargs.get("extend_inferencedata", False):
        model._idata.update({
            group: result[group]
            for group in ("posterior_predictive", "observed_data", "constant_data")
            if group in result
        })
        return model._idata
    return result
