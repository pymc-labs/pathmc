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
"""Unit tests for the pathmc.idata ArviZ accessor helpers."""

import arviz as az
import numpy as np
import xarray as xr

from pathmc.idata import DEFAULT_HDI_PROB, beta_draws, hdi, posterior


def _idata_with_beta(beta: np.ndarray) -> xr.DataTree:
    """Build a posterior ``DataTree`` with a single ``beta_Y`` coefficient variable."""
    return az.from_dict(
        {"posterior": {"beta_Y": beta}},
        coords={"Y_predictors": ["Intercept", "X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )


class TestPosterior:
    def test_returns_posterior_group(self) -> None:
        idata = _idata_with_beta(np.array([[[0.0, 2.0]]]))
        post = posterior(idata)
        assert "beta_Y" in post
        assert post.sizes["chain"] == 1


class TestBetaDraws:
    def test_selects_named_predictor(self) -> None:
        idata = _idata_with_beta(np.array([[[0.0, 2.0]]]))
        assert np.allclose(beta_draws(idata, "beta_Y", "Y_predictors", "X"), 2.0)
        assert np.allclose(
            beta_draws(idata, "beta_Y", "Y_predictors", "Intercept"), 0.0
        )

    def test_flattens_chain_and_draw_dims(self) -> None:
        # 2 chains x 3 draws x 2 predictors.
        beta = np.arange(12, dtype=float).reshape(2, 3, 2)
        idata = _idata_with_beta(beta)
        draws = beta_draws(idata, "beta_Y", "Y_predictors", "X")
        assert draws.shape == (6,)
        # "X" is the second predictor -> odd-indexed entries 1, 3, 5, ...
        assert np.allclose(np.sort(draws), beta[:, :, 1].flatten())


class TestHDI:
    def test_default_prob_is_module_constant(self) -> None:
        assert DEFAULT_HDI_PROB == 0.94

    def test_default_interval_brackets_the_mean(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(size=20_000)
        lo, hi = hdi(draws)
        assert lo < 0 < hi

    def test_higher_prob_widens_interval(self) -> None:
        rng = np.random.default_rng(0)
        draws = rng.normal(size=20_000)
        narrow_lo, narrow_hi = hdi(draws, prob=0.5)
        wide_lo, wide_hi = hdi(draws, prob=0.99)
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)
