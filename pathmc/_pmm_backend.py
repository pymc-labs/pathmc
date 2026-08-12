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
"""Optional delegation to ``pymc_marketing.mmm.transformers``.

``pymc-marketing`` 1.0.0 pins ``pymc<6.1``, which transitively constrains
``pytensor<3.1``. pathmc requires ``pytensor>=3.1.1`` for exog-lag scan carry
(#333), so the two stacks cannot be installed together until upstream relaxes
its pins. When a compatible ``pymc-marketing`` release is installed, the
built-in transforms delegate through the xtensor bridging helpers below;
otherwise ``pathmc.transforms`` falls back to vendored pytensor kernels with
matching numerics.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pytensor.tensor as pt
from pytensor.xtensor.type import as_xtensor

_PMM_IMPORT_ERROR: str | None = None


@lru_cache(maxsize=1)
def pmm_available() -> bool:
    """Return whether a compatible ``pymc-marketing`` install is importable."""
    global _PMM_IMPORT_ERROR
    try:
        from pymc_marketing.mmm import transformers as _pmm_transformers  # noqa: F401
    except ImportError as exc:
        _PMM_IMPORT_ERROR = str(exc)
        return False
    _PMM_IMPORT_ERROR = None
    return True


def pmm_import_error() -> str | None:
    """Return the import error from the last failed availability check, if any."""
    pmm_available()
    return _PMM_IMPORT_ERROR


def _xtensor_dims(x: Any) -> tuple[str, ...]:
    return tuple(f"d{i}" for i in range(x.ndim))


def adstock_pmm(
    x: Any,
    *,
    alpha: Any,
    l_max: int,
    normalize: bool,
    dims: tuple[str, ...],
) -> Any:
    """Geometric adstock via pymc-marketing, unwrapped to a ``TensorVariable``."""
    from pymc_marketing.mmm.transformers import geometric_adstock

    xx = as_xtensor(x, dims=dims)
    return geometric_adstock(
        xx,
        alpha=alpha,
        l_max=l_max,
        normalize=normalize,
        dim=dims[0],
    ).values


def delayed_adstock_pmm(
    x: Any,
    *,
    alpha: Any,
    theta: Any,
    l_max: int,
    normalize: bool,
    dims: tuple[str, ...],
) -> Any:
    """Delayed adstock via pymc-marketing, unwrapped to a ``TensorVariable``."""
    from pymc_marketing.mmm.transformers import delayed_adstock

    xx = as_xtensor(x, dims=dims)
    return delayed_adstock(
        xx,
        alpha=alpha,
        theta=theta,
        l_max=l_max,
        normalize=normalize,
        dim=dims[0],
    ).values


def weibull_adstock_pmm(
    x: Any,
    *,
    lam: Any,
    k: Any,
    l_max: int,
    normalize: bool,
    dims: tuple[str, ...],
    weibull_type: str = "PDF",
) -> Any:
    """Weibull adstock via pymc-marketing, unwrapped to a ``TensorVariable``."""
    from pymc_marketing.mmm.transformers import WeibullType, weibull_adstock

    xx = as_xtensor(x, dims=dims)
    return weibull_adstock(
        xx,
        lam=lam,
        k=k,
        l_max=l_max,
        normalize=normalize,
        dim=dims[0],
        type=WeibullType(weibull_type),
    ).values


def logistic_saturation_pmm(x: Any, *, lam: Any) -> Any:
    """Logistic saturation via pymc-marketing, unwrapped to a ``TensorVariable``."""
    from pymc_marketing.mmm.transformers import logistic_saturation

    dims = _xtensor_dims(x)
    return logistic_saturation(as_xtensor(x, dims=dims), lam=lam).values


def michaelis_menten_pmm(x: Any, *, alpha: Any, lam: Any) -> Any:
    """Michaelis-Menten saturation via pymc-marketing."""
    from pymc_marketing.mmm.transformers import michaelis_menten

    dims = _xtensor_dims(x)
    return michaelis_menten(as_xtensor(x, dims=dims), alpha=alpha, lam=lam).values


def _batched_convolution(
    x: Any,
    w: Any,
    *,
    l_max: int,
) -> Any:
    """1D trailing convolution along the leading axis (``ConvMode.After``).

    Vendored fallback matching ``pymc_marketing.mmm.transformers.batched_convolution``
    for the default adstock padding mode.
    """
    result = w[0] * x
    for i in range(1, l_max):
        shifted = pt.zeros_like(x)
        shifted = pt.set_subtensor(shifted[i:], x[:-i])
        result = result + w[i] * shifted
    return result
