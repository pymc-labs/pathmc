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

Tiers 0–4 never call ``pm.sample``. These tests do, and they check that the
posterior actually concentrates near the simulation truth. They are marked
``nightly`` and kept off ``make test`` / ``make test-fast`` because the
``slow`` marker already runs on every pull request. Run them with
``make test-nightly`` or the nightly workflow.

The suite's autouse sampler clamp (50 draws, 50 tune, 1 chain) still
applies. Tolerances are the wide ones used by the rest of the suite.
Simulation-based calibration is 50 fits of one Bernoulli model, enough to
catch a gross posterior-shape bug, not a calibration study. The panel lag
model is recovery-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

import pathmc

pytestmark = pytest.mark.nightly

_TOL = 0.15
_SBC_N = 50
_FIT = {"draws": 50, "tune": 50, "chains": 1, "cores": 1, "progressbar": False}


def _coef(idata, outcome: str, name: str) -> float:
    posterior = idata.posterior[f"beta_{outcome}"]
    return float(posterior.sel({f"{outcome}_predictors": name}).mean())


def _assert_coef(got: float, expected: float, *, spec: str, seed: int, name: str):
    assert abs(got - expected) <= _TOL, (
        f"{spec} (seed={seed}): {name} expected {expected}, got {got}, tol={_TOL}"
    )


def test_mcmc_recovers_mediation():
    """Cross-sectional mediation slopes concentrate near the known truth."""
    spec = "M ~ a*X\nY ~ b*M + c*X"
    seed = 41501
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"X": rng.normal(size=500)})
    truth = {"M": {"X": 0.8}, "Y": {"M": 0.7, "X": 0.3}}
    simulated = pathmc.simulate(
        spec,
        data=data,
        params={
            "beta_M": [0.0, truth["M"]["X"]],
            "sigma_M": 0.4,
            "beta_Y": [0.0, truth["Y"]["M"], truth["Y"]["X"]],
            "sigma_Y": 0.4,
        },
        random_seed=seed,
    )
    model = pathmc.model(spec, data=simulated)
    idata = model.fit(**_FIT, random_seed=seed)
    for outcome, coefs in truth.items():
        for name, expected in coefs.items():
            _assert_coef(
                _coef(idata, outcome, name),
                expected,
                spec=spec,
                seed=seed,
                name=f"beta_{outcome}[{name}]",
            )


def test_mcmc_recovers_panel_lag_spend():
    """``sales ~ lag(spend)`` with partial pooling recovers the lag slope.

    This is the issue #316 model class. The formula intercept is absorbed
    by the unit intercepts, so the recovered coefficient is ``lag(spend)``.
    """
    spec = "sales ~ lag(spend)"
    seed = 41502
    panel = {"unit": "geo", "time": "week"}
    rng = np.random.default_rng(1)
    rows = []
    for unit in range(8):
        spend = rng.normal(size=40)
        for week, value in enumerate(spend):
            rows.append({"geo": f"g{unit}", "week": week, "spend": float(value)})
    truth = 1.0
    simulated = pathmc.simulate(
        spec,
        data=pd.DataFrame(rows),
        params={
            "beta_sales": [truth],
            "sigma_sales": 0.4,
            "alpha_sales": np.linspace(-0.4, 0.4, 8),
            "mu_alpha_sales": 0.0,
            "sigma_alpha_sales": 0.3,
        },
        panel=panel,
        pooling="partial",
        random_seed=seed,
    )
    model = pathmc.model(spec, data=simulated, panel=panel, pooling="partial")
    idata = model.fit(**_FIT, random_seed=seed)
    _assert_coef(
        _coef(idata, "sales", "lag(spend)"),
        truth,
        spec=spec,
        seed=seed,
        name="beta_sales[lag(spend)]",
    )


def test_mcmc_recovers_bernoulli():
    """A Bernoulli slope concentrates near the known truth."""
    spec = "Y ~ X1"
    seed = 41503
    truth = 1.5
    rng = np.random.default_rng(2)
    simulated = pathmc.simulate(
        spec,
        data=pd.DataFrame({"X1": rng.normal(size=800)}),
        params={"beta_Y": [-0.2, truth]},
        families={"Y": "bernoulli"},
        random_seed=seed,
    )
    model = pathmc.model(spec, data=simulated, families={"Y": "bernoulli"})
    idata = model.fit(**_FIT, random_seed=seed)
    _assert_coef(
        _coef(idata, "Y", "X1"),
        truth,
        spec=spec,
        seed=seed,
        name="beta_Y[X1]",
    )


def test_light_sbc_bernoulli_slope_ranks_are_not_degenerate():
    """Prior → simulate → fit ranks for one Bernoulli slope are not piled up.

    Fifty simulations. A posterior stuck at zero, the #316 failure mode,
    pushes each rank to 0 or 1 with the sign of the prior draw, so the mean
    rank stays near one half. The extreme-rank share does not. This is not
    a formal calibration study.
    """
    spec = "Y ~ X1"
    seed = 41504
    rng = np.random.default_rng(seed)
    priors = {"beta_Y": Prior("Normal", mu=0, sigma=1, dims="Y_predictors")}
    ranks = []
    for _ in range(_SBC_N):
        truth = rng.normal(scale=1.0, size=2)
        draw_seed = int(rng.integers(1, 2**31 - 1))
        simulated = pathmc.simulate(
            spec,
            data=pd.DataFrame({"X1": rng.normal(size=300)}),
            params={"beta_Y": truth},
            families={"Y": "bernoulli"},
            random_seed=draw_seed,
        )
        model = pathmc.model(
            spec,
            data=simulated,
            families={"Y": "bernoulli"},
            priors=priors,
        )
        idata = model.fit(**_FIT, random_seed=draw_seed)
        draws = idata.posterior["beta_Y"].sel(Y_predictors="X1").values.reshape(-1)
        ranks.append(float(np.mean(draws < truth[1])))
    ranks = np.asarray(ranks)
    extreme = float(np.mean((ranks < 0.1) | (ranks > 0.9)))
    assert extreme < 0.5, (
        f"{spec} (seed={seed}): share of SBC ranks outside [0.1, 0.9] is "
        f"{extreme:.3f} over {_SBC_N} simulations (about 0.2 if uniform, near "
        "1 if the posterior is stuck at zero). Look at the Bernoulli logp graph."
    )
