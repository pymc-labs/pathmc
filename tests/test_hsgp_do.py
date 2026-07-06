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
"""Slow HSGP tests: intervention recomputation and smooth recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc


def _fit(spec_string, df, seed=1):
    model = pathmc.model(spec_string, data=df)
    idata = model.fit(random_seed=seed, progressbar=False)
    return model, idata


@pytest.mark.slow
def test_do_recomputes_basis():
    """do() on the HSGP input changes the predicted mean (basis recomputes)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 6.0, 60)
    y = np.sin(x) + rng.normal(0, 0.2, size=x.size)
    df = pd.DataFrame({"x": x, "y": y})
    model, _ = _fit("y ~ hsgp(x, m=15, c=1.5)", df)

    # Interventions stay within the fitted support (see boundary caveat: with
    # c=, L is frozen from fit-time data, so a wider grid would extrapolate).
    low = float(model.do(set={"x": 1.5}).mean("y"))
    high = float(model.do(set={"x": 4.5}).mean("y"))
    assert abs(low - high) > 0.1


@pytest.mark.slow
def test_recovers_known_smooth():
    """Posterior-mean smooth correlates with the true function (>0.9)."""
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 6.0, 80)
    true = np.sin(x)
    y = true + rng.normal(0, 0.2, size=x.size)
    df = pd.DataFrame({"x": x, "y": y})
    model, idata = _fit("y ~ hsgp(x, m=20, c=1.5)", df)

    fmean = idata.posterior["f_y_x"].mean(("chain", "draw")).values
    corr = np.corrcoef(fmean - fmean.mean(), true - true.mean())[0, 1]
    assert corr > 0.9


@pytest.mark.slow
def test_recovers_smooth_with_nonzero_mean_input():
    """Centering guard: recovery must hold for x far from zero.

    A zero-centered fixture would hide a bug where the basis is built on
    uncentered input.  Here x in [10, 20] exercises internal centering.
    """
    rng = np.random.default_rng(2)
    x = np.linspace(10.0, 20.0, 80)
    true = np.sin(x - 10.0)
    y = true + rng.normal(0, 0.2, size=x.size)
    df = pd.DataFrame({"x": x, "y": y})
    model, idata = _fit("y ~ hsgp(x, m=20, c=1.5)", df)

    fmean = idata.posterior["f_y_x"].mean(("chain", "draw")).values
    corr = np.corrcoef(fmean - fmean.mean(), true - true.mean())[0, 1]
    assert corr > 0.9
