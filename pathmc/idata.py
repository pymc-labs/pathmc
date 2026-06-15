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
"""Internal accessors for ArviZ ``InferenceData``.

ArviZ exposes inference groups (``posterior``, ...) as dynamically-added
attributes that static type checkers cannot see, and ``az.hdi`` is called with
the same default credible mass throughout the package. Centralizing both here
keeps these fragile, dynamically-typed access points in one tested place.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np

DEFAULT_HDI_PROB = 0.94


def posterior(idata: az.InferenceData) -> Any:
    """Return the ``posterior`` group of *idata*."""
    return idata.posterior


def beta_draws(
    idata: az.InferenceData,
    beta_name: str,
    coord_name: str,
    predictor: str,
) -> np.ndarray:
    """Return flattened posterior draws for one labeled coefficient.

    Selects ``beta_name[predictor]`` along ``coord_name`` and flattens the
    chain/draw dimensions into a 1-D array.
    """
    return posterior(idata)[beta_name].sel({coord_name: predictor}).to_numpy().flatten()


def hdi(draws: Any, prob: float = DEFAULT_HDI_PROB) -> Any:
    """Compute the highest-density interval at the package-default mass."""
    return az.hdi(draws, prob=prob)
