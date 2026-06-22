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
"""Graph-consistency tests across the compile matrix (issues #316, #326).

These tests deliberately touch the **logp / gradient graph** — the part of a
compiled model ``pm.sample`` optimizes — which the suite's predictive-sampling
speed trick never exercises. They run **without sampling**, so they are fast
and deterministic.

Implements, from #326:
  * Tier 0 — ``assert_mu_context_invariant`` (generative mu == logp-graph mu)
  * Tier 1 — ``gaussian_likelihood_oracle_gap`` (hand logp oracle)
  * Tier 2 — ``assert_logp_and_grad_finite``

The ``lag(x)`` panel cells are the **red test for #316**: they are marked
``xfail(strict=True)`` so the suite stays green today and will flip to a hard
failure (alerting us to un-mark them) the moment #316 is fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc

from _consistency import (
    assert_model_consistent,
    gaussian_likelihood_oracle_gap,
)

ISSUE_316 = "https://github.com/pymc-labs/pathmc/issues/316"


# ---------------------------------------------------------------------------
# Model builders — one per compile-matrix cell. Each returns a *built* (not
# fitted) model; the consistency checks need no posterior.
# ---------------------------------------------------------------------------


def _xsec_data(seed=1, n=150):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n)
    counts = rng.poisson(np.exp(0.2 * x1), size=n)
    binary = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.5 * x1)))).astype(int)
    return pd.DataFrame({"X1": x1, "X2": x2, "Y": y, "Ycount": counts, "Ybin": binary})


def _panel_data(seed=1, ngeo=5, ntime=12):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(ngeo):
        spend = rng.normal(10, 1, ntime)
        x2 = rng.normal(0, 1, ntime)
        sales = np.ones(ntime) * 20
        sales[1:] = 10 + spend[:-1] + rng.normal(0, 1, ntime - 1)
        frames.append(
            pd.DataFrame({
                "spend": spend,
                "x2": x2,
                "sales": sales,
                "week": np.arange(ntime),
                "geo": g,
            })
        )
    return pd.concat(frames, ignore_index=True)


_PANEL = {"unit": "geo", "time": "week"}


def _m_xsec_gaussian():
    return pathmc.model("Y ~ X1 + X2", data=_xsec_data())


def _m_xsec_interaction():
    return pathmc.model("Y ~ X1 + X1:X2", data=_xsec_data())


def _m_xsec_bernoulli():
    return pathmc.model("Ybin ~ X1", data=_xsec_data(), families={"Ybin": "bernoulli"})


def _m_xsec_poisson():
    return pathmc.model(
        "Ycount ~ X1", data=_xsec_data(), families={"Ycount": "poisson"}
    )


def _m_panel_plain_complete():
    return pathmc.model("sales ~ x2", data=_panel_data(), panel=_PANEL, pooling=None)


def _m_panel_plain_partial():
    return pathmc.model(
        "sales ~ x2", data=_panel_data(), panel=_PANEL, pooling="partial"
    )


def _m_panel_lag_endogenous():
    # lag of the *outcome* — a distinct scan path from exogenous lag, and it
    # passes today (the #316 corruption is specific to exogenous lag carries).
    return pathmc.model(
        "sales ~ lag(sales) + x2", data=_panel_data(), panel=_PANEL, pooling=None
    )


def _m_panel_lag_complete():
    return pathmc.model(
        "sales ~ lag(spend)", data=_panel_data(), panel=_PANEL, pooling=None
    )


def _m_panel_lag_partial():
    return pathmc.model(
        "sales ~ lag(spend)", data=_panel_data(), panel=_PANEL, pooling="partial"
    )


# (id, builder, expected_to_pass_today). Cells that fail are the #316 red tests.
_CELLS = [
    ("xsec-gaussian", _m_xsec_gaussian, True),
    ("xsec-interaction", _m_xsec_interaction, True),
    ("xsec-bernoulli", _m_xsec_bernoulli, True),
    ("xsec-poisson", _m_xsec_poisson, True),
    ("panel-plain-complete", _m_panel_plain_complete, True),
    ("panel-plain-partial", _m_panel_plain_partial, True),
    ("panel-lag(y)-complete", _m_panel_lag_endogenous, True),
    ("panel-lag(x)-complete", _m_panel_lag_complete, False),
    ("panel-lag(x)-partial", _m_panel_lag_partial, False),
]


def _param(cell):
    cell_id, builder, passes = cell
    marks = (
        ()
        if passes
        else (pytest.mark.xfail(strict=True, reason=f"issue #316 — {ISSUE_316}"),)
    )
    return pytest.param(builder, id=cell_id, marks=marks)


_ALL = [_param(c) for c in _CELLS]
_GAUSSIAN = [_param(c) for c in _CELLS if c[0].startswith(("xsec-gaussian", "panel"))]


# ---------------------------------------------------------------------------
# Tier 0 + Tier 2 — full consistency battery across every cell.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _ALL)
def test_model_consistent(builder):
    """Generative mu must equal the logp-graph mu, and logp/grad stay finite."""
    assert_model_consistent(builder(), seed=0)


# ---------------------------------------------------------------------------
# Tier 1 — hand-computed Gaussian logp oracle (Gaussian cells only).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", _GAUSSIAN)
def test_gaussian_likelihood_matches_hand_oracle(builder):
    """The likelihood term the model emits must match a hand-built Gaussian logp."""
    gap = gaussian_likelihood_oracle_gap(builder(), seed=0)
    assert gap < 1e-6, (
        f"compiled likelihood logp differs from the hand oracle by {gap:.3g} — "
        "the likelihood is scoring against a different mu than the generative one"
    )
