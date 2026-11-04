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
"""Post-intervention DAG snapshots and their public result boundary."""

from dataclasses import FrozenInstanceError
import re

import graphviz
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc
from pathmc.compile import _requires_rectangular_panel
from pathmc.graph import build_graph
from pathmc.introspect import describe_do_graph, render_graph_description
from pathmc.parse import parse_spec
from pathmc.simulate import DoResult


SPEC = """\
tv ~ seasonality
digital ~ seasonality
search ~ a*tv + seasonality
sales ~ b*tv + c*search + d*digital
"""
NODES = {"seasonality", "tv", "digital", "search", "sales"}
EDGES = {
    ("seasonality", "tv"),
    ("seasonality", "digital"),
    ("seasonality", "search"),
    ("tv", "search"),
    ("tv", "sales"),
    ("search", "sales"),
    ("digital", "sales"),
}


def _topology(dot: graphviz.Digraph) -> tuple[set[str], set[tuple[str, str]]]:
    nodes = set(re.findall(r"^\s*(\w+)\s+\[", dot.source, re.MULTILINE))
    edges = set(re.findall(r"^\s*(\w+)\s+->\s+(\w+)", dot.source, re.MULTILINE))
    return nodes, edges


def _description(spec: str, set_values: dict, simulate_over: str | None = None):
    parsed = parse_spec(spec)
    return describe_do_graph(parsed, build_graph(parsed), {}, set_values, simulate_over)


@pytest.mark.parametrize(
    ("targets", "removed"),
    [
        ({"tv": 10.0}, {("seasonality", "tv")}),
        (
            {"tv": 10.0, "search": 2.0},
            {("seasonality", "tv"), ("tv", "search"), ("seasonality", "search")},
        ),
    ],
)
def test_hard_intervention_cuts_only_incoming_edges(targets, removed):
    description = _description(SPEC, targets)
    dot = render_graph_description(description)
    assert _topology(dot) == (NODES, EDGES - removed)
    assert {name for name, _ in description.interventions} == set(targets)
    assert "externally set" in dot.source
    assert "label=b" in dot.source
    assert "label=c" in dot.source


def test_scalar_and_complete_time_schedule_have_same_graph():
    scalar = _description(SPEC, {"tv": 10.0})
    scheduled = _description(SPEC, {"tv": np.array([1.0, 2.0, 3.0])}, "time")
    assert scalar.interventions == (("tv", "scalar"),)
    assert scheduled.interventions == (("tv", "schedule"),)
    assert _topology(render_graph_description(scalar)) == _topology(
        render_graph_description(scheduled)
    )
    assert "[1." not in render_graph_description(scheduled).source


def test_snapshot_is_immutable_and_each_render_is_fresh():
    description = _description(SPEC, {"tv": 10.0})
    with pytest.raises(FrozenInstanceError):
        description.edges = ()
    first = render_graph_description(description)
    first.node("unrelated")
    assert "unrelated" not in render_graph_description(description).source


def test_snapshot_does_not_retain_mutable_spec():
    parsed = parse_spec(SPEC)
    description = describe_do_graph(parsed, build_graph(parsed), {}, {"tv": 1.0}, None)
    expected = render_graph_description(description).source
    parsed.regressions.clear()
    assert render_graph_description(description).source == expected


@pytest.mark.parametrize(
    ("spec", "set_values", "simulate_over", "message"),
    [
        (SPEC, {"tv": np.array([1.0, np.nan])}, "time", "partially released"),
        (SPEC, {"tv": np.nan}, None, "nonfinite"),
        (SPEC, {"tv": np.array([1.0, 2.0])}, None, "shape"),
        ("x ~ lag(x)\ny ~ x", {"x": 1.0}, "time", "temporal"),
        (
            "y ~ logistic_saturation(adstock(x, decay=theta), lam=lam)",
            {"x": 1.0},
            "time",
            "temporal",
        ),
        (
            "y ~ delayed_adstock(x, decay=alpha, theta=theta)",
            {"x": 1.0},
            "time",
            "temporal",
        ),
        (
            "y ~ weibull_adstock(x, lam=lam, k=k)",
            {"x": 1.0},
            "time",
            "temporal",
        ),
        (
            "y ~ logistic_saturation(delayed_adstock(x, decay=alpha, theta=theta), lam=scale)",
            {"x": 1.0},
            "time",
            "temporal",
        ),
        (
            "y ~ logistic_saturation(weibull_adstock(x, lam=lam, k=k), lam=scale)",
            {"x": 1.0},
            "time",
            "temporal",
        ),
        ("x ~ z\ny ~ x\nx ~~ y", {"x": 1.0}, None, "residual covariance"),
    ],
)
def test_unsupported_graph_semantics_raise(spec, set_values, simulate_over, message):
    if message == "temporal":
        parsed = parse_spec(spec)
        assert _requires_rectangular_panel(parsed, build_graph(parsed))
    description = _description(spec, set_values, simulate_over)
    with pytest.raises(ValueError, match=message):
        render_graph_description(description)


def test_non_temporal_transform_still_has_intervention_graph():
    spec = parse_spec("y ~ logistic_saturation(x, lam=lam)")
    graph_info = build_graph(spec)
    assert not _requires_rectangular_panel(spec, graph_info)
    description = describe_do_graph(spec, graph_info, {}, {"x": 1.0}, None)
    assert _topology(render_graph_description(description)) == (
        {"x", "y"},
        {("x", "y")},
    )


def test_legacy_counterfactual_and_contrast_results_raise_clearly():
    ds = xr.Dataset({"sales": (("chain", "draw"), [[1.0, 2.0]])})
    legacy = DoResult(ds=ds)
    with pytest.raises(ValueError, match="no graph metadata"):
        legacy.graph()
    counterfactual = DoResult(ds=ds, scenario="counterfactual")
    with pytest.raises(ValueError, match="counterfactual"):
        counterfactual.graph()
    left = DoResult(ds=ds, graph_description=_description(SPEC, {"tv": 1.0}))
    right = DoResult(ds=ds, graph_description=_description(SPEC, {"tv": 2.0}))
    contrast = left - right
    np.testing.assert_array_equal(contrast.draws("sales"), [0.0, 0.0])
    with pytest.raises(ValueError, match="constituent scenario"):
        contrast.graph()


def test_model_do_attaches_detached_graph_without_changing_draws(monkeypatch):
    data = pd.DataFrame({name: np.arange(6, dtype=float) for name in NODES})
    model = pathmc.model(SPEC, data=data)
    ds = xr.Dataset({"sales": (("chain", "draw"), [[3.0, 4.0]])})
    monkeypatch.setattr(model, "_require_fitted", lambda _: object())
    monkeypatch.setattr(model, "_run_do", lambda *args, **kwargs: DoResult(ds=ds))

    original = model.graph().source
    result = model.do(set={"tv": 3.0})
    assert isinstance(result.graph(), graphviz.Digraph)
    assert _topology(result.graph()) == (NODES, EDGES - {("seasonality", "tv")})
    assert model.graph().source == original
    assert result.dataset.identical(ds)
    assert result.mean("sales") == 3.5
    result.graph().node("local_style_only")
    assert "local_style_only" not in result.graph().source


def test_panel_do_attaches_complete_schedule_graph(monkeypatch):
    import pathmc._model as model_module

    data = pd.DataFrame([
        {"unit": unit, "time": time, **{name: float(time) for name in NODES}}
        for unit in ("a", "b")
        for time in range(3)
    ])
    model = pathmc.model(SPEC, data=data, panel={"unit": "unit", "time": "time"})
    ds = xr.Dataset({"sales": (("chain", "draw", "time"), [[[1.0, 2.0, 3.0]]])})
    monkeypatch.setattr(model, "_require_fitted", lambda _: object())
    monkeypatch.setattr(
        model_module, "run_do_static_panel", lambda **kwargs: DoResult(ds=ds)
    )

    result = model.do(set={"tv": np.array([0.0, 1.0, 2.0])}, simulate_over="time")
    assert _topology(result.graph()) == (NODES, EDGES - {("seasonality", "tv")})
    assert result.dataset.identical(ds)


@pytest.mark.slow
def test_fitted_do_result_graph_requires_no_rerun(fitted_mediation, monkeypatch):
    result = fitted_mediation.do(set={"M": 1.0}, kind="mean")
    before = result.mean("Y")
    monkeypatch.setattr(
        fitted_mediation,
        "_run_do",
        lambda *args, **kwargs: pytest.fail("graph() reran posterior simulation"),
    )
    assert _topology(result.graph()) == (
        {"X", "M", "Y"},
        {("M", "Y"), ("X", "Y")},
    )
    assert result.mean("Y") == before
