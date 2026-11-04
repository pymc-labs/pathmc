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
"""Independent numerical and lifecycle contracts for prepared posterior queries."""

import gc
from unittest.mock import patch
import weakref

import numpy as np
import pandas as pd
import pymc as pm
import pymc.model.core as core
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

import pathmc
import pathmc._query as query


@pytest.fixture
def fitted():
    frame = pd.DataFrame({
        "X": [-1.0, 0.0, 1.0, 2.0],
        "M": [1.0, 3.0, 2.0, 5.0],
        "Y": [2.0, 5.0, 1.0, 4.0],
    })
    model = pathmc.model("M ~ X\nY ~ X + M", data=frame)
    bm = np.arange(12, dtype=float).reshape(2, 3, 2) / 10 + 1
    by = np.arange(18, dtype=float).reshape(2, 3, 3) / 10 - 1
    model._idata = xr.DataTree.from_dict({
        "posterior": xr.Dataset(
            {
                "beta_M": (("chain", "draw", "M_predictors"), bm),
                "beta_Y": (("chain", "draw", "Y_predictors"), by),
                "sigma_M": (("chain", "draw"), np.full((2, 3), 0.5)),
                "sigma_Y": (("chain", "draw"), np.full((2, 3), 0.2)),
            },
            coords={
                "chain": [7, 11],
                "draw": [23, 25, 29],
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "X", "M"],
            },
        )
    })
    return model


def oracle(model, x):
    post = model._idata.posterior.to_dataset()
    bm, by = post.beta_M.values, post.beta_Y.values
    m = bm[..., 0, None] + bm[..., 1, None] * x
    y = by[..., 0, None] + by[..., 1, None] * x + by[..., 2, None] * m
    return m, y


@pytest.mark.parametrize("average", [True, False])
def test_mean_values_and_ate_reuse_batched_graph(fitted, average):
    fitted.do(set={"X": 0}, average_units=average)
    with (
        patch.object(pytensor, "function", wraps=pytensor.function) as compile_pt,
        patch.object(core, "compile", wraps=core.compile) as compile_pm,
        patch.object(pm, "do", wraps=pm.do) as surgery,
        patch.object(query, "prepare_mean", wraps=query.prepare_mean) as prepare,
    ):
        for x in (1.0, np.array([-1.0, 0.0, 1.0, 2.0]), 0.0):
            result = fitted.do(set={"X": x}, average_units=average)
            expected_m, expected_y = oracle(fitted, x)
            if average:
                expected_m, expected_y = expected_m.mean(-1), expected_y.mean(-1)
            else:
                expected_m = np.broadcast_to(expected_m, (2, 3, 4))
                expected_y = np.broadcast_to(expected_y, (2, 3, 4))
            np.testing.assert_allclose(
                result.dataset.M, expected_m, rtol=1e-8, atol=1e-10
            )
            np.testing.assert_allclose(
                result.dataset.Y, expected_y, rtol=1e-8, atol=1e-10
            )
            np.testing.assert_array_equal(result.dataset.chain, [7, 11])
            np.testing.assert_array_equal(result.dataset.draw, [23, 25, 29])
        ate = fitted.ate("Y", "X", values=(0.0, 2.0), average_units=average)
        bm, by = (
            fitted._idata.posterior.beta_M.values,
            fitted._idata.posterior.beta_Y.values,
        )
        expected = 2 * (by[..., 1] + by[..., 2] * bm[..., 1])
        if not average:
            expected = np.broadcast_to(expected[..., None], (2, 3, 4))
        np.testing.assert_allclose(ate.dataset.Y, expected, rtol=1e-8, atol=1e-10)
        assert not compile_pt.called and not compile_pm.called and not surgery.called
        assert not prepare.called


def test_runtime_parameters_data_and_endogenous_targets(fitted):
    fitted.do(set={"M": 2}, average_units=False)
    fitted._idata.posterior["beta_Y"] = fitted._idata.posterior.beta_Y * 2
    with fitted.pymc_model:
        pm.set_data({"X": np.array([3.0, 2.0, 1.0, 0.0])})
    result = fitted.do(set={"M": np.arange(4.0)}, average_units=False)
    by = fitted._idata.posterior.beta_Y.values
    expected = (
        by[..., 0, None]
        + by[..., 1, None] * [3.0, 2.0, 1.0, 0.0]
        + by[..., 2, None] * np.arange(4.0)
    )
    np.testing.assert_allclose(result.dataset.Y, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(result.dataset.X, 0.5)


def test_scenarios_chunking_and_row_vector_semantics(fitted):
    original = fitted._idata.copy(deep=True)
    scenarios = [{"X": float(value)} for value in np.linspace(-1, 2, 25)]
    for average in (True, False):
        full = query.mean_scenarios(fitted, scenarios, average, max_elements=100_000)
        plan = next(reversed(fitted._query_cache.entries.values()))
        with (
            patch.object(query, "prepare_mean", wraps=query.prepare_mean) as prepare,
            patch.object(plan, "function", wraps=plan.function) as evaluate,
        ):
            chunked = query.mean_scenarios(fitted, scenarios, average, max_elements=8)
        assert not prepare.called
        assert evaluate.call_count > 1
        for call in evaluate.call_args_list:
            draw_count = call.args[0].shape[0]
            scenario_count = call.args[len(plan.parameters)].shape[1]
            assert draw_count * scenario_count * plan.rows <= 8
        assert len(full) == len(chunked) == 25
        for scenario, first, second in zip(scenarios, full, chunked, strict=True):
            xr.testing.assert_allclose(first.dataset, second.dataset)
            _, expected = oracle(fitted, scenario["X"])
            if average:
                expected = expected.mean(-1)
            else:
                expected = np.broadcast_to(expected, (2, 3, 4))
            np.testing.assert_allclose(
                second.dataset.Y, expected, rtol=1e-8, atol=1e-10
            )
    xr.testing.assert_equal(fitted._idata, original)


def test_predict_reuses_upstream_cache_and_reseeds(fitted):
    if query.freeze_model is None:
        pytest.skip("This installed PyMC uses the uncached fallback")
    kwargs = {"extend_inferencedata": False, "progressbar": False, "random_seed": 123}
    first = fitted.predict(**kwargs)
    original = fitted._idata.copy(deep=True)
    with (
        patch.object(core, "compile", wraps=core.compile) as compile_pm,
        patch.object(pm, "do", wraps=pm.do) as surgery,
        patch.object(query, "freeze_model", wraps=query.freeze_model) as freeze,
    ):
        again = fitted.predict(**kwargs)
        different = fitted.predict(**{**kwargs, "random_seed": 124})
        newdata = fitted._data.to_pandas().assign(X=lambda df: df.X + 1)
        changed = fitted.predict(data=newdata, **kwargs)
        restored = fitted.predict(**kwargs)
        assert not compile_pm.called and not surgery.called and not freeze.called
    xr.testing.assert_equal(first, again)
    xr.testing.assert_equal(first, restored)
    assert not first.posterior_predictive.to_dataset().equals(
        different.posterior_predictive.to_dataset()
    )
    before, after = (
        first.posterior_predictive.to_dataset(),
        changed.posterior_predictive.to_dataset(),
    )
    post = fitted._idata.posterior.to_dataset()
    np.testing.assert_allclose(
        after.M - before.M, np.broadcast_to(post.beta_M.values[..., 1, None], (2, 3, 4))
    )
    np.testing.assert_allclose(
        after.Y - before.Y,
        np.broadcast_to(
            (
                post.beta_Y.values[..., 1]
                + post.beta_Y.values[..., 2] * post.beta_M.values[..., 1]
            )[..., None],
            (2, 3, 4),
        ),
    )
    xr.testing.assert_equal(fitted._idata, original)
    returned = fitted.predict(progressbar=False, random_seed=123)
    assert returned is fitted._idata
    assert "posterior_predictive" in returned


def test_predict_missing_capability_and_options_delegate(fitted, monkeypatch):
    monkeypatch.setattr(query, "freeze_model", None)
    result = fitted.predict(
        extend_inferencedata=False, progressbar=False, random_seed=123
    )
    assert "posterior_predictive" in result
    assert not fitted._query_cache.entries


def test_invalidation_eviction_and_disposal(fitted):
    cache = fitted._query_cache
    cache.limit = 2
    fitted.do(set={"X": 1})
    old = weakref.ref(next(iter(cache.entries.values())))
    fitted.do(set={"M": 1})
    fitted.do(set={"Y": 1})
    gc.collect()
    assert len(cache.entries) == 2 and old() is None
    cache.clear()
    assert not cache.entries
    fitted.do(set={"X": 1})
    fitted.set_priors({})
    assert not cache.entries and fitted._idata is None


def test_schema_change_refit_and_exception_restore(fitted, monkeypatch):
    fitted.do(set={"X": 1})
    post = fitted._idata.posterior.to_dataset()
    fitted._idata = xr.DataTree.from_dict({"posterior": post.isel(draw=[0])})
    assert fitted.do(set={"X": 1}).dataset.sizes["draw"] == 1
    assert len(fitted._query_cache.entries) == 2
    frame = fitted._data.to_pandas()
    original = fitted.pymc_model["X"].get_value().copy()

    def fail(*args, **kwargs):
        raise ValueError("probe")

    monkeypatch.setattr(pm, "sample_posterior_predictive", fail)
    with pytest.raises(ValueError, match="probe"):
        fitted.predict(data=frame.assign(X=frame.X + 1))
    np.testing.assert_array_equal(fitted.pymc_model["X"].get_value(), original)
    saved = fitted._idata
    monkeypatch.setattr(pm, "sample", lambda **kwargs: saved)
    fitted.fit(compute_log_likelihood=False)
    assert not fitted._query_cache.entries


@pytest.mark.parametrize(
    "spec,kwargs",
    [
        ("Y ~ X", {"families": {"Y": "poisson"}}),
        ("Y ~ X + X:M", {}),
        ("Y ~ fourier(X, n=1, period=4)", {}),
        ("M ~ X\nY ~ X\nM ~~ Y", {}),
        ("M ~ X\nY ~ M", {"latent": ["M"]}),
    ],
)
def test_ineligible_routes_retain_executor(fitted, spec, kwargs):
    model = pathmc.model(spec, data=fitted._data.to_pandas(), **kwargs)
    assert not query.eligible(model)


def test_no_cross_instance_cache(fitted):
    fitted.do(set={"X": 1})
    other = pathmc.model("Y ~ X", data=fitted._data.to_pandas())
    assert other._query_cache is not fitted._query_cache
    assert not other._query_cache.entries


def test_fixed_coefficients_and_lru_hits(fitted):
    model = pathmc.model("Y ~ 0 + 2*X", data=fitted._data.to_pandas())
    model._idata = xr.DataTree.from_dict({
        "posterior": fitted._idata.posterior.to_dataset()[["sigma_Y"]]
    })
    np.testing.assert_allclose(model.do(set={"X": 1}).dataset.Y, 2)
    cache = fitted._query_cache
    cache.limit = 2
    fitted.do(set={"X": 1})
    first = weakref.ref(next(iter(cache.entries.values())))
    fitted.do(set={"M": 1})
    second = weakref.ref(next(reversed(cache.entries.values())))
    fitted.do(set={"X": 0})
    fitted.do(set={"Y": 1})
    gc.collect()
    assert first() is not None and second() is None


def test_prediction_runtime_posterior_values_and_option_fallback(fitted):
    options = dict(extend_inferencedata=False, progressbar=False, random_seed=123)
    before = fitted.predict(**options).posterior_predictive.to_dataset()
    fitted._idata.posterior["beta_Y"] = fitted._idata.posterior.beta_Y + [1.0, 0.0, 0.0]
    after = fitted.predict(**options).posterior_predictive.to_dataset()
    np.testing.assert_allclose(after.Y - before.Y, 1.0, rtol=1e-8, atol=1e-10)
    subset = fitted.predict(
        var_names=["Y"], **options
    ).posterior_predictive.to_dataset()
    assert list(subset.data_vars) == ["Y"]


def test_model_disposal_releases_plans(fitted):
    other = pathmc.model("M ~ X\nY ~ X + M", data=fitted._data.to_pandas())
    other._idata = fitted._idata
    other.do(set={"X": 1})
    model_ref = weakref.ref(other)
    plan_ref = weakref.ref(next(iter(other._query_cache.entries.values())))
    del other
    gc.collect()
    assert model_ref() is None and plan_ref() is None


def test_symbolic_initial_values_delegate_prediction(fitted):
    fitted.pymc_model.set_initval(fitted.pymc_model["beta_Y"], pt.ones(3))
    with patch.object(query, "freeze_model", wraps=query.freeze_model) as freeze:
        result = fitted.predict(
            extend_inferencedata=False, progressbar=False, random_seed=3
        )
    assert not freeze.called
    assert "Y" in result.posterior_predictive


def test_float32_mean_preserves_legacy_output_dtype():
    with pytensor.config.change_flags(floatX="float32"):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=pd.DataFrame({
                "X": [-1.0, 1.0],
                "M": [1.0, 2.0],
                "Y": [2.0, 3.0],
            }).astype("float32"),
        )
        post = xr.Dataset(
            coords={
                "chain": [0],
                "draw": [0, 1],
                "M_predictors": ["Intercept", "X"],
                "Y_predictors": ["Intercept", "M"],
            }
        )
        for name in ("M", "Y"):
            post[f"beta_{name}"] = (
                ("chain", "draw", f"{name}_predictors"),
                np.array([[[1.0, 2.0], [2.0, 3.0]]], dtype="float32"),
            )
            post[f"sigma_{name}"] = (
                ("chain", "draw"),
                np.ones((1, 2), dtype="float32"),
            )
        model._idata = xr.DataTree.from_dict({"posterior": post})
        for target, expected in (("X", [[7.0, 17.0]]), ("M", [[3.0, 5.0]])):
            prepared = model.do(set={target: 1.0}).dataset.Y
            with patch.object(query, "eligible", return_value=False):
                legacy = model.do(set={target: 1.0}).dataset.Y
            assert prepared.dtype == legacy.dtype
            np.testing.assert_allclose(prepared, expected, rtol=1e-6, atol=1e-7)
