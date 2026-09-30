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

All ``lag(x)`` panel cells were the **red tests for #316** (marked
``xfail(strict=True)`` in the prior PR).  The fix in the follow-up commit
eliminates the exogenous-lag scan carry that triggered PyTensor's scan-merge
optimizer bug, so all cells now pass unconditionally.
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


def _mediation_data(seed=42, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    y = 0.8 * m + 0.3 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "M": m, "Y": y})


_PANEL = {"unit": "geo", "time": "week"}


def _m_xsec_gaussian():
    return pathmc.model("Y ~ X1 + X2", data=_xsec_data())


def _m_xsec_mediation():
    # Two-outcome SCM (mu_M and mu_Y) — exercises the harness's multi-`mu`
    # path, which the single-outcome cells do not.
    return pathmc.model("M ~ a*X\nY ~ b*M + c*X", data=_mediation_data())


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


def _m_panel_lag_no_intercept():
    return pathmc.model(
        "sales ~ 0 + lag(spend)", data=_panel_data(), panel=_PANEL, pooling="partial"
    )


def _panel_count_data(seed=1, ngeo=5, ntime=12):
    """Panel frame with a small non-negative count outcome for negbinomial."""
    df = _panel_data(seed=seed, ngeo=ngeo, ntime=ntime).copy()
    rng = np.random.default_rng(seed + 7)
    rate = np.exp(0.3 * df["x2"].to_numpy())
    df["count"] = rng.poisson(rate)
    return df


def _m_panel_adstock():
    # Geometric adstock forces the scan-panel compiler (same family as #316).
    return pathmc.model(
        "sales ~ adstock(spend, decay=d)",
        data=_panel_data(),
        panel=_PANEL,
        pooling=None,
    )


def _m_panel_lag_adstock():
    # Co-occurring terms: lag(adstock(...)) is a parse error.
    return pathmc.model(
        "sales ~ lag(spend) + adstock(x2, decay=d)",
        data=_panel_data(),
        panel=_PANEL,
        pooling=None,
    )


def _m_xsec_negbinomial():
    return pathmc.model(
        "Ycount ~ X1",
        data=_xsec_data(),
        families={"Ycount": "negbinomial"},
    )


def _m_panel_negbinomial():
    return pathmc.model(
        "count ~ x2",
        data=_panel_count_data(),
        panel=_PANEL,
        families={"count": "negbinomial"},
    )


def _m_panel_lag_slopes():
    # Slopes bind to a raw column. lag(spend) is not a data column.
    return pathmc.model(
        "sales ~ lag(spend) + x2",
        data=_panel_data(),
        panel=_PANEL,
        pooling={"intercept": True, "slopes": ["x2"]},
    )


def _m_panel_logistic_saturation():
    return pathmc.model(
        "sales ~ logistic_saturation(spend, lam=lam)",
        data=_panel_data(),
        panel=_PANEL,
        pooling=None,
    )


# (id, builder). All cells must pass since issue #316 is fixed.
_CELLS = [
    ("xsec-gaussian", _m_xsec_gaussian),
    ("xsec-mediation-2outcome", _m_xsec_mediation),
    ("xsec-interaction", _m_xsec_interaction),
    ("xsec-bernoulli", _m_xsec_bernoulli),
    ("xsec-poisson", _m_xsec_poisson),
    ("xsec-negbinomial", _m_xsec_negbinomial),
    ("panel-plain-complete", _m_panel_plain_complete),
    ("panel-plain-partial", _m_panel_plain_partial),
    ("panel-lag(y)-complete", _m_panel_lag_endogenous),
    ("panel-lag(x)-complete", _m_panel_lag_complete),
    ("panel-lag(x)-partial", _m_panel_lag_partial),
    ("panel-lag(x)-no-intercept", _m_panel_lag_no_intercept),
    ("panel-adstock-complete", _m_panel_adstock),
    ("panel-lag(x)-adstock", _m_panel_lag_adstock),
    ("panel-negbinomial", _m_panel_negbinomial),
    ("panel-lag(x)-slopes", _m_panel_lag_slopes),
    ("panel-logistic-saturation", _m_panel_logistic_saturation),
]


def _param(cell):
    cell_id, builder = cell
    return pytest.param(builder, id=cell_id)


_ALL = [_param(c) for c in _CELLS]
# Negbinomial has no sigma_* likelihood, so the Gaussian hand oracle does not apply.
_GAUSSIAN = [
    _param(c)
    for c in _CELLS
    if c[0].startswith(("xsec-gaussian", "panel")) and "negbinomial" not in c[0]
]


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
