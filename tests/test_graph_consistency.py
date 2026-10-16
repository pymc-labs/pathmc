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
* Tier 1 — ``gaussian_likelihood_oracle_gap`` (hand built oracle)
* Tier 2 — ``assert_logp_and_grad_finite``

All ``lag(x)`` panel cells were the **red** tests for #316* (marked
``xfail(strict=True)`` in the prior PR). The fix in the follow-up commit
eliminates the exogenous-lag scan carry that triggered optimizer bug, so all
cells now pass unconditionally.

The cell table lives in ``tests/_consistency.py``. This module filters it by
id and by the explicit ``gaussian_oracle`` flag. It does not restate specs.
"""

from __future__ import annotations

import pytest

from _consistency import CELLS, assert_model_consistent, gaussian_likelihood_oracle_gap

_GAUSSIAN = [cell for cell in CELLS if cell.gaussian_oracle]


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_model_consistent(cell):
    """Generative mu must equal the logp-graph mu, and logp/grad stay finite."""
    assert_model_consistent(cell.model(cell.data()), seed=0)


@pytest.mark.parametrize("cell", _GAUSSIAN, ids=lambda cell: cell.id)
def test_gaussian_likelihood_matches_hand_oracle(cell):
    """The likelihood term the model emits must match a hand-built Gaussian logp."""
    gap = gaussian_likelihood_oracle_gap(cell.model(cell.data()), seed=0)
    assert gap < 1e-6, (
        f"compiled likelihood logp differs from the hand oracle by {gap:.3g} — "
        "the likelihood is scoring against a different mu than the generative one"
    )
