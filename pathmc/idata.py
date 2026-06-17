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
"""Internal accessors for an ArviZ posterior ``DataTree``.

ArviZ 1.0 replaced the ``InferenceData`` class with :class:`xarray.DataTree`,
where inference groups are child nodes. Accessing a group as a node attribute
(``idata.posterior``) is dynamically typed and yields a ``DataTree`` rather than
a ``Dataset``; this module routes group access through ``idata["posterior"].dataset``
so callers get a statically typed :class:`xarray.Dataset`.

It also pins the package-wide credible mass for ``az.hdi``. ArviZ 1.0 changed
its global defaults (``ci_prob`` 0.94 -> 0.89, ``ci_kind`` "hdi" -> "eti"), so
calling ``az.hdi`` with an explicit ``prob`` keeps pathmc on a stable 0.94 HDI
regardless of the installed ArviZ's ``rcParams``.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import xarray as xr

DEFAULT_HDI_PROB = 0.94


def posterior(idata: xr.DataTree) -> xr.Dataset:
    """Return the ``posterior`` group of *idata* as a :class:`xarray.Dataset`."""
    return idata["posterior"].dataset


def beta_draws(
    idata: xr.DataTree,
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
