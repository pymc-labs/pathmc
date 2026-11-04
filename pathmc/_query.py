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
from collections.abc import Callable, Mapping, Sequence
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


def eligible(model: PathModel) -> bool:
    """Limit optimization to the observed numeric linear Gaussian contract."""
    if (
        model._data is None
        or model._panel_info is not None
        or model._pooling is not None
        or model._latent
        or model._spec.residual_covs
        or model._scaling_factors is not None
        or str(pytensor.config.mode)
        not in ("Mode", "FAST_RUN", "FAST_COMPILE", "CVM", "VM", "NUMBA")
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


def signature(model: PathModel, post: xr.Dataset) -> tuple:
    """Identify graph generation, posterior schema, and mutable input layout."""
    assert model._gen_model is not None and model._pymc_model is not None
    return (
        id(model._gen_model),
        tuple((name, id(var)) for name, var in model._gen_model.named_vars.items()),
        tuple((name, id(var)) for name, var in model._pymc_model.named_vars.items()),
        tuple(
            (name, value.dims, value.shape, str(value.dtype))
            for name, value in post.data_vars.items()
        ),
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
    parameters = tuple(
        f"beta_{reg.lhs}"
        for reg in model._spec.regressions
        if f"beta_{reg.lhs}" in source.named_vars
    )
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
        list(vectorized.values()), batched, on_unused_input="ignore"
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
    for name in model._graph_info.exogenous:
        if model._pymc_model[name].get_value().shape != (rows,):
            return None
    values = {
        name: np.stack([_intervention_array(s[name], rows) for s in scenarios]).astype(
            model._gen_model[name].dtype, copy=False
        )
        for name in targets
    }
    if any(not np.isfinite(value).all() for value in values.values()):
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
        key = ("mean", signature(model, post), targets, average)
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
        data = [
            np.asarray(model._pymc_model[name].get_value())[None, None, :]
            for name in plan.exogenous
        ]
        for start in range(0, samples, draw_batch):
            stop = min(samples, start + draw_batch)
            for arm in range(0, n_scenarios, scenario_batch):
                end = min(n_scenarios, arm + scenario_batch)
                inputs = [v[start:stop] for v in parameters]
                inputs += [values[name][None, arm:end] for name in plan.targets]
                outputs = plan.function(*inputs, *data)
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
    key = ("predict", signature(model, post), outputs, observed_schema)
    frozen = model._query_cache.get(key, lambda: freeze_model(model._pymc_model))
    for name, var in model._pymc_model.named_vars.items():
        if hasattr(var, "get_value"):
            target = frozen[name]
            # Freezing a graph must never hide a mutable input behind a constant.
            if not hasattr(target, "set_value"):
                return model._pymc_model
            target.set_value(var.get_value())
    return frozen
