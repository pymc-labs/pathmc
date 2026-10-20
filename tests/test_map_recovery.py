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

Cells live in ``tests/_consistency.py``. This module filters that table to
records whose ``recover_methods`` include ``map`` and calls :func:`recover`.
Partial pooling is MCMC-only: those cells do not share a stable mode.
"""

from __future__ import annotations

import pytest

from _consistency import CELLS, assert_recovered, recover

_MAP = [cell for cell in CELLS if "map" in cell.recover_methods]


@pytest.mark.parametrize("cell", _MAP, ids=lambda cell: cell.id)
def test_map_recovers_known_coefficients(cell):
    """MAP recovers known coefficients from the shared cell record.

    Cross-sectional Gaussian slopes, the #316 ``sales ~ lag(spend)`` lag
    coefficient, and Bernoulli/Poisson slopes are the cells whose
    ``recover_methods`` include ``map``. The lag intercept is identified
    only after centering spend. Partial pooling is excluded. Tolerance is
    0.15 absolute.
    """
    assert_recovered(cell, recover(cell, "map"))
