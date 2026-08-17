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
"""Tests for additional MMM transforms (issue #214)."""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

from pathmc.transforms import (
    DelayedAdstock,
    MichaelisMenten,
    WeibullAdstock,
    _delayed_adstock,
    _michaelis_menten,
    _weibull_adstock,
    get_transform,
)


def _reference_delayed_adstock(
    x: np.ndarray, alpha: float, theta: float, l_max: int, normalize: bool = False
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    weights = alpha ** ((np.arange(l_max) - theta) ** 2)
    if normalize:
        weights = weights / weights.sum()
    y = np.zeros_like(x)
    for t in range(n):
        for i in range(l_max):
            if t - i >= 0:
                y[t] += weights[i] * x[t - i]
    return y


def _reference_weibull_pdf_adstock(
    x: np.ndarray, lam: float, k: float, l_max: int, normalize: bool = False
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.arange(l_max) + 1
    weights = (k / lam) * (t / lam) ** (k - 1) * np.exp(-((t / lam) ** k))
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    if normalize:
        weights = weights / weights.sum()
    n = x.shape[0]
    y = np.zeros_like(x)
    for time in range(n):
        for i in range(l_max):
            if time - i >= 0:
                y[time] += weights[i] * x[time - i]
    return y


class TestRegistry:
    def test_new_transforms_registered(self):
        assert get_transform("delayed_adstock").name == "delayed_adstock"
        assert get_transform("weibull_adstock").name == "weibull_adstock"
        assert get_transform("michaelis_menten").name == "michaelis_menten"


class TestDelayedAdstockKernel:
    @pytest.mark.parametrize("alpha", [0.3, 0.7])
    @pytest.mark.parametrize("theta", [0.0, 2.0])
    def test_matches_oracle(self, alpha, theta):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, size=20)
        l_max = 5
        expected = _reference_delayed_adstock(x, alpha, theta, l_max)
        actual = _delayed_adstock(
            pt.as_tensor_variable(x),
            alpha=alpha,
            theta=theta,
            l_max=l_max,
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


class TestWeibullAdstockKernel:
    def test_pdf_matches_oracle(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 5, size=15)
        lam, k, l_max = 3.0, 1.5, 6
        expected = _reference_weibull_pdf_adstock(x, lam, k, l_max)
        actual = _weibull_adstock(
            pt.as_tensor_variable(x),
            lam=lam,
            k=k,
            l_max=l_max,
            weibull_type="PDF",
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


class TestMichaelisMentenKernel:
    def test_matches_formula(self):
        x = np.array([0.0, 1.0, 2.0, 5.0])
        alpha, lam = 2.0, 1.0
        expected = alpha * x / (lam + x)
        actual = _michaelis_menten(
            pt.as_tensor_variable(x), alpha=alpha, lam=lam
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


class TestTransformApply:
    def test_delayed_apply_pymc(self):
        x = pt.as_tensor_variable(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        transform = DelayedAdstock(l_max=3)
        out = transform.apply_pymc(x, {"decay": 0.5, "theta": 0.0}).eval()
        assert out.shape == (5,)

    def test_delayed_theta_prior_is_left_skewed(self):
        import pymc as pm

        transform = DelayedAdstock()
        with pm.Model():
            prior = transform.emit_prior("peak", transform.param_specs["theta"])
        assert prior.owner.op.name == "beta"
        assert prior.name == "peak"

    def test_weibull_apply_pymc(self):
        x = pt.as_tensor_variable(np.linspace(0, 1, 8))
        transform = WeibullAdstock(l_max=4)
        out = transform.apply_pymc(x, {"lam": 2.0, "k": 1.0}).eval()
        assert out.shape == (8,)

    def test_michaelis_apply_pymc(self):
        x = pt.as_tensor_variable(np.linspace(0, 2, 6))
        transform = MichaelisMenten()
        out = transform.apply_pymc(x, {"alpha": 1.0, "lam": 0.5}).eval()
        assert out.shape == (6,)
