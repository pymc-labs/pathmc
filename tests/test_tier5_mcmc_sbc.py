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
"""Tier 5 nightly MCMC recovery and light SBC (issue #326).

``make test`` never calls ``pm.sample``. These tests do, and they check that
the posterior actually concentrates near the simulation truth. They are marked
``slow`` and ``nightly`` and kept off ``make test`` / ``make test-fast``
because the ``slow`` marker already runs on every pull request. Run them with
``make test-nightly`` or the nightly workflow.

Sampler kwargs come from :data:`conftest.SAMPLE_KWARGS`. The suite's autouse
sampler clamp (50 draws, 50 tune, 1 chain) still applies. Tolerances are the
wide ones used by the rest of the suite.

Simulation-based calibration is 50 fits of one Bernoulli model, enough to
catch a gross posterior-shape bug, not a calibration study. The panel lag
model is recovery-only.

Recovery cells live in ``tests/_consistency.py``. This module filters that
table to records whose ``recover_methods`` include ``mcmc``. The light SBC
reuses the Bernoulli cell's specification and families and draws its own
data, because the rank statistic depends on the draw seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

import pathmc
from _consistency import CELLS, assert_recovered, cells, recover
from conftest import SAMPLE_KWARGS

pytestmark = [pytest.mark.nightly, pytest.mark.slow]

_MCMC = [cell for cell in CELLS if "mcmc" in cell.recover_methods]
_SBC_N = 50


@pytest.mark.parametrize("cell", _MCMC, ids=lambda cell: cell.id)
def test_mcmc_recovers_known_coefficients(cell):
    """MCMC recovers known coefficients from the shared cell record.

    Mediation (``M ~ a*X``, ``Y ~ b*M + c*X``) asserts only the non-intercept
    slopes: intercepts are zero in the parameter dictionary but are not part
    of the truth. Partial-pooling ``sales ~ lag(spend)`` omits the intercept
    absorbed into the unit intercepts. Bernoulli reuses the MAP cell's
    specification with its own truth and seed. Tolerance is 0.15 absolute.
    """
    assert_recovered(cell, recover(cell, "mcmc", sample_kwargs=SAMPLE_KWARGS))


def test_light_sbc_bernoulli_slope_ranks_are_not_degenerate():
    """Prior -> simulate -> fit ranks for one Bernoulli slope are not piled up.

    Fifty simulations of the shared Bernoulli cell. A posterior stuck at
    zero, the #316 failure mode, pushes each rank to 0 or 1 with the sign
    of the prior draw, so the mean rank stays near one half. The
    extreme-rank share does not. This is not a formal calibration study.
    The specification and families come from the cell. The rank statistic
    still draws its own exogenous column, because the draw seed has to
    drive both the prior truth and the design.
    """
    cell = cells("xsec-bernoulli")[0]
    outcome = cell.spec.split("~", 1)[0].strip()
    predictor = "X1"
    seed = 41504
    rng = np.random.default_rng(seed)
    beta = f"beta_{outcome}"
    dims = f"{outcome}_predictors"
    priors = {beta: Prior("Normal", mu=0, sigma=1, dims=dims)}
    probe = cell.model(cell.data(n=30, seed=0), priors=priors)
    names = list(probe.pymc_model.coords[dims])
    slope_index = names.index(predictor)
    ranks = []
    for _ in range(_SBC_N):
        truth = rng.normal(scale=1.0, size=len(names))
        draw_seed = int(rng.integers(1, 2**31 - 1))
        simulated = pathmc.simulate(
            cell.spec,
            data=pd.DataFrame({"X1": rng.normal(size=300)}),
            params={beta: truth},
            families=cell.families,
            random_seed=draw_seed,
        )
        model = cell.model(simulated, priors=priors)
        idata = model.fit(**SAMPLE_KWARGS, progressbar=False, random_seed=draw_seed)
        draws = idata.posterior[beta].sel({dims: predictor}).values.reshape(-1)
        ranks.append(float(np.mean(draws < truth[slope_index])))
    ranks = np.asarray(ranks)
    extreme = float(np.mean((ranks < 0.1) | (ranks > 0.9)))
    assert extreme < 0.5, (
        f"{cell.spec} (seed={seed}): share of SBC ranks outside [0.1, 0.9] is "
        f"{extreme:.3f} over {_SBC_N} simulations (about 0.2 if uniform, near "
        "1 if the posterior is stuck at zero). Look at the Bernoulli logp graph."
    )
