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
"""Tier 4 MAP parameter recovery (issue #326).

Simulate from known coefficients and recover the mode with ``pm.find_MAP``.
This exercises logp and its gradient end to end without NUTS, at far lower
cost than the Tier 5 nightly sample. A miss means the objective being
optimized is not the intended likelihood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc

_TOL = 0.15


def _map_beta(spec, data, params, *, outcome, sim_seed, families=None, panel=None):
    simulated = pathmc.simulate(
        spec,
        data=data,
        params=params,
        families=families,
        panel=panel,
        random_seed=sim_seed,
    )
    model = pathmc.model(spec, data=simulated, families=families, panel=panel)
    with model.pymc_model:
        estimate = pm.find_MAP(progressbar=False)
    names = list(model.pymc_model.coords[f"{outcome}_predictors"])
    got = np.asarray(estimate[f"beta_{outcome}"], dtype=float).reshape(-1)
    return got, names


def _assert_beta(got, names, truth, tol=_TOL):
    for name, expected in truth.items():
        value = float(got[names.index(name)])
        assert abs(value - expected) <= tol, (
            f"{name}: expected {expected}, got {value}, tol={tol}"
        )


def test_map_recovers_cross_sectional_gaussian():
    """Cross-sectional Gaussian slopes recover under find_MAP."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"X1": rng.normal(size=400)})
    truth = {"Intercept": 0.2, "X1": 1.5}
    got, names = _map_beta(
        "Y ~ X1",
        data,
        {"beta_Y": [truth["Intercept"], truth["X1"]], "sigma_Y": 0.3},
        outcome="Y",
        sim_seed=1,
    )
    _assert_beta(got, names, truth)


def test_map_recovers_panel_lag_spend():
    """The #316 cell ``sales ~ lag(spend)`` recovers the lag coefficient."""
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(8):
        spend = rng.normal(size=40)
        for time, value in enumerate(spend):
            rows.append({"geo": unit, "week": time, "spend": value})
    truth = {"Intercept": 0.5, "lag(spend)": 1.0}
    got, names = _map_beta(
        "sales ~ lag(spend)",
        pd.DataFrame(rows),
        {
            "beta_sales": [truth["Intercept"], truth["lag(spend)"]],
            "sigma_sales": 0.4,
        },
        outcome="sales",
        sim_seed=3,
        panel={"unit": "geo", "time": "week"},
    )
    _assert_beta(got, names, truth)


@pytest.mark.parametrize(
    ("family", "truth"),
    [
        ("bernoulli", {"Intercept": -0.2, "X1": 1.2}),
        ("poisson", {"Intercept": 0.2, "X1": 0.6}),
    ],
    ids=["xsec-bernoulli", "xsec-poisson"],
)
def test_map_recovers_discrete_family(family, truth):
    """Bernoulli and Poisson slopes recover where the mode is identified."""
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"X1": rng.normal(size=400)})
    got, names = _map_beta(
        "Y ~ X1",
        data,
        {"beta_Y": [truth["Intercept"], truth["X1"]]},
        outcome="Y",
        sim_seed=2,
        families={"Y": family},
    )
    _assert_beta(got, names, truth)
