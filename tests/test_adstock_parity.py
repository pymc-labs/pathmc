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
"""Numerical parity tests for the geometric-adstock kernel (issue #327, item 7).

Pins pathmc's ``_geometric_adstock`` / ``Adstock`` transform against a
plain-numpy reference that mirrors
``pymc_marketing.mmm.transformers.geometric_adstock``:

    y[t] = sum_{i=0}^{l_max-1} alpha**i * x[t-i]   (zero-padded before t=0)
    y[t] /= sum_{i=0}^{l_max-1} alpha**i            (only when normalize=True)

pymc-marketing is not installed in this environment (it does not yet
support PyMC 6 / ArviZ 1 — see the module docstring in
``pathmc/transforms.py``), so parity is checked against this independently
implemented oracle rather than by importing the upstream package directly.
"""

from __future__ import annotations

import numpy as np
import pytensor.tensor as pt
import pytest

from pathmc.transforms import Adstock, _geometric_adstock


def _reference_geometric_adstock(
    x: np.ndarray, alpha: float, l_max: int, normalize: bool = False
) -> np.ndarray:
    """Plain-numpy oracle matching pymc_marketing's geometric_adstock."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    weights = alpha ** np.arange(l_max)
    y = np.zeros_like(x)
    for t in range(n):
        for i in range(l_max):
            if t - i >= 0:
                y[t] += weights[i] * x[t - i]
    if normalize:
        y = y / weights.sum()
    return y


class TestGeometricAdstockParity:
    """_geometric_adstock matches the pymc_marketing-equivalent oracle."""

    @pytest.mark.parametrize("alpha", [0.0, 0.3, 0.7, 0.95])
    @pytest.mark.parametrize("l_max", [1, 3, 5, 12])
    def test_unnormalized_matches_oracle(self, alpha, l_max):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, size=20)
        expected = _reference_geometric_adstock(x, alpha, l_max, normalize=False)
        actual = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=l_max
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("alpha", [0.3, 0.7, 0.95])
    @pytest.mark.parametrize("l_max", [1, 3, 5, 12])
    def test_normalized_matches_oracle(self, alpha, l_max):
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, size=20)
        expected = _reference_geometric_adstock(x, alpha, l_max, normalize=True)
        actual = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=l_max, normalize=True
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_normalize_divides_by_weight_sum(self):
        """normalize=True is exactly the unnormalized result / sum(weights)."""
        alpha, l_max = 0.6, 4
        x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        unnorm = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=l_max
        ).eval()
        norm = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=l_max, normalize=True
        ).eval()
        weight_sum = sum(alpha**i for i in range(l_max))
        np.testing.assert_allclose(norm, unnorm / weight_sum, rtol=1e-12)

    def test_pinned_values_l_max_3(self):
        """Regression-pin: exact hand-computed values for a small case."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha, l_max = 0.5, 3
        # y0 = 1
        # y1 = 2 + 0.5*1 = 2.5
        # y2 = 3 + 0.5*2 + 0.25*1 = 4.25
        # y3 = 4 + 0.5*3 + 0.25*2 = 6.0
        # y4 = 5 + 0.5*4 + 0.25*3 = 7.75
        expected = np.array([1.0, 2.5, 4.25, 6.0, 7.75])
        actual = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=l_max
        ).eval()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_l_max_truncates_versus_full_window(self):
        """A short l_max must diverge from the l_max = len(x) full window."""
        alpha = 0.9
        x = np.concatenate([np.array([10.0]), np.zeros(11)])
        short = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=3
        ).eval()
        full = _geometric_adstock(
            pt.as_tensor_variable(x), alpha=alpha, l_max=12
        ).eval()
        # At t=11, the l_max=3 kernel has already dropped the impulse's
        # contribution (only lags 0-2 are ever nonzero), but the full-window
        # kernel still carries decayed carryover from the impulse at t=0.
        assert short[11] == 0.0
        assert full[11] == pytest.approx(alpha**11 * 10.0)


class TestAdstockTransformConfigurable:
    """Adstock exposes l_max / normalize as constructor parameters (not
    hardcoded), matching pymc_marketing's per-call configurability."""

    def test_default_matches_previous_hardcoded_behavior(self):
        a = Adstock()
        assert a.l_max == 12
        assert a.normalize is False

    def test_l_max_is_configurable(self):
        a = Adstock(l_max=52)
        assert a.l_max == 52

    def test_normalize_is_configurable(self):
        a = Adstock(l_max=6, normalize=True)
        assert a.normalize is True

    def test_apply_pymc_uses_instance_l_max_and_normalize(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(0, 5, size=15)
        alpha = 0.4
        l_max = 4

        default_transform = Adstock(l_max=l_max, normalize=False)
        normalized_transform = Adstock(l_max=l_max, normalize=True)

        xt = pt.as_tensor_variable(x)
        unnorm = default_transform.apply_pymc(xt, {"decay": alpha}).eval()
        norm = normalized_transform.apply_pymc(xt, {"decay": alpha}).eval()

        weight_sum = sum(alpha**i for i in range(l_max))
        np.testing.assert_allclose(norm, unnorm / weight_sum, rtol=1e-10)
