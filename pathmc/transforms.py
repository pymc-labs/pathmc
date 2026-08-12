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
"""Transform registry and built-in transforms (adstock, saturation).

Each transform produces PyMC tensor operations for model compilation and
provides a ``step()`` method for use inside ``pytensor.scan`` bodies.

Built-in geometric adstock and logistic saturation delegate to
``pymc_marketing.mmm.transformers`` when a compatible ``pymc-marketing``
release is installed (see :mod:`pathmc._pmm_backend`); otherwise they use
vendored pytensor kernels with matching numerics. Additional MMM variants
(delayed / Weibull adstock, Michaelis-Menten saturation) follow the same
pattern. Install ``pymc-marketing`` manually once a release compatible with
pathmc's ``pytensor>=3.1.1`` floor is published upstream to activate delegation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pathmc._pmm_backend import (
    _batched_convolution,
    adstock_pmm,
    delayed_adstock_pmm,
    logistic_saturation_pmm,
    michaelis_menten_pmm,
    pmm_available,
    weibull_adstock_pmm,
)

__all__ = ["ParamSpec", "Transform", "register_transform"]


def _geometric_adstock(
    x: Any, *, alpha: Any, l_max: int, normalize: bool = False
) -> Any:
    """Geometric adstock along the leading (time) axis.

    ``y[t] = sum_{i=0}^{l_max-1} alpha**i * x[t-i]``, with zero padding before
    ``t = 0``. When ``normalize`` is ``True`` the result is divided by
    ``sum_{i=0}^{l_max-1} alpha**i`` so the lag weights sum to 1.
    """
    if l_max < 1:
        raise ValueError(
            f"l_max must be >= 1, got {l_max}. Geometric adstock needs at "
            f"least one lag weight; set l_max to the maximum carryover "
            f"duration in time steps."
        )
    if pmm_available():
        dims = tuple(f"d{i}" for i in range(x.ndim))
        return adstock_pmm(x, alpha=alpha, l_max=l_max, normalize=normalize, dims=dims)
    w = pt.power(alpha, pt.arange(l_max))
    result = _batched_convolution(x, w, l_max=l_max)
    if normalize:
        result = result / pt.sum(w)
    return result


def _delayed_adstock(
    x: Any,
    *,
    alpha: Any,
    theta: Any,
    l_max: int,
    normalize: bool = False,
) -> Any:
    """Delayed adstock along the leading (time) axis."""
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}.")
    if pmm_available():
        dims = tuple(f"d{i}" for i in range(x.ndim))
        return delayed_adstock_pmm(
            x,
            alpha=alpha,
            theta=theta,
            l_max=l_max,
            normalize=normalize,
            dims=dims,
        )
    lags = pt.arange(l_max, dtype=x.dtype)
    w = pt.power(alpha, (lags - theta) ** 2)
    if normalize:
        w = w / pt.sum(w)
    return _batched_convolution(x, w, l_max=l_max)


def _weibull_adstock(
    x: Any,
    *,
    lam: Any,
    k: Any,
    l_max: int,
    normalize: bool = False,
    weibull_type: str = "PDF",
) -> Any:
    """Weibull adstock along the leading (time) axis (PDF mode when vendored)."""
    if l_max < 1:
        raise ValueError(f"l_max must be >= 1, got {l_max}.")
    if pmm_available():
        dims = tuple(f"d{i}" for i in range(x.ndim))
        return weibull_adstock_pmm(
            x,
            lam=lam,
            k=k,
            l_max=l_max,
            normalize=normalize,
            dims=dims,
            weibull_type=weibull_type,
        )
    if weibull_type != "PDF":
        raise NotImplementedError(
            "Weibull CDF adstock requires pymc-marketing; install pathmc[marketing] "
            "once a pymc-marketing release compatible with this PyMC version is "
            "available."
        )
    t = pt.arange(l_max, dtype=x.dtype) + 1
    w = (k / lam) * pt.power(t / lam, k - 1) * pt.exp(-pt.power(t / lam, k))
    w = (w - pt.min(w)) / (pt.max(w) - pt.min(w))
    if normalize:
        w = w / pt.sum(w)
    return _batched_convolution(x, w, l_max=l_max)


def _logistic_saturation(x: Any, *, lam: Any) -> Any:
    """Pointwise logistic saturation: ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))``."""
    if pmm_available():
        return logistic_saturation_pmm(x, lam=lam)
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


def _michaelis_menten(x: Any, *, alpha: Any, lam: Any) -> Any:
    """Pointwise Michaelis-Menten saturation: ``alpha * x / (lam + x)``."""
    if pmm_available():
        return michaelis_menten_pmm(x, alpha=alpha, lam=lam)
    return alpha * x / (lam + x)


def _apply_conv_panel(
    kernel_fn: Any,
    x: Any,
    panel_info: Any,
    data: Any,
    **kernel_kwargs: Any,
) -> Any:
    """Apply a convolution kernel per panel unit via matrix reshaping."""
    unit_col = panel_info.unit
    time_col = panel_info.time
    units = panel_info.unit_labels
    n_units = len(units)
    n_time = len(data) // n_units

    sorted_idx = (
        data
        .with_row_index("__nw_row_pos__")
        .sort([unit_col, time_col])["__nw_row_pos__"]
        .to_numpy()
    )
    reverse_idx = np.argsort(sorted_idx)

    x_sorted = x[sorted_idx]
    x_matrix = x_sorted.reshape((n_units, n_time)).T  # (time, units)
    convolved = kernel_fn(x_matrix, **kernel_kwargs)
    result_flat = convolved.T.flatten()
    return result_flat[reverse_idx]


@dataclass
class ParamSpec:
    """Specification for a single transform parameter.

    Parameters
    ----------
    constraint : str
        ``"unit_interval"`` for (0, 1), ``"positive"`` for (0, inf).
    default_prior : str
        Human-readable prior description for introspection.
    """

    constraint: str
    default_prior: str


class Transform:
    """Base class for named transforms with estimable parameters.

    Subclasses must implement :meth:`apply_pymc` and define
    :attr:`name` and :attr:`param_specs`.  Stateful transforms
    (e.g. adstock) should also override :meth:`step` and
    :attr:`has_state`.
    """

    name: str
    param_specs: dict[str, ParamSpec]

    def emit_prior(self, param_name: str, spec: ParamSpec) -> Any:
        """Create a PyMC random variable for a transform parameter."""
        if spec.constraint == "unit_interval":
            return pm.Beta(param_name, alpha=2, beta=2)
        if spec.constraint == "positive":
            return pm.HalfNormal(param_name, sigma=1)
        return pm.Normal(param_name, mu=0, sigma=10)

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        """Apply the transform in the PyMC computation graph."""
        raise NotImplementedError

    @property
    def has_state(self) -> bool:
        """Whether this transform carries state across time steps."""
        return False

    def step(self, x_t: Any, state: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        """Apply one time step inside a ``pytensor.scan`` body."""
        return self.apply_pymc(x_t, params), state


class _ConvAdstockBase(Transform):
    """Shared panel / scan behaviour for convolution-based adstock transforms."""

    l_max: int
    normalize: bool

    DEFAULT_L_MAX = 12
    DEFAULT_NORMALIZE = False

    def __init__(
        self, l_max: int = DEFAULT_L_MAX, normalize: bool = DEFAULT_NORMALIZE
    ) -> None:
        if isinstance(l_max, bool) or not isinstance(l_max, (int, np.integer)):
            raise ValueError(f"l_max must be an int >= 1, got {l_max!r}.")
        if l_max < 1:
            raise ValueError(f"l_max must be >= 1, got {l_max!r}.")
        if not isinstance(normalize, bool):
            raise ValueError(f"normalize must be a bool, got {normalize!r}.")
        self.l_max = int(l_max)
        self.normalize = normalize

    def _convolve(
        self,
        x: Any,
        panel_info: Any | None,
        data: Any | None,
        **kernel_kwargs: Any,
    ) -> Any:
        kernel_kwargs = {
            **kernel_kwargs,
            "l_max": self.l_max,
            "normalize": self.normalize,
        }
        if panel_info is not None and data is not None:
            return _apply_conv_panel(self._kernel, x, panel_info, data, **kernel_kwargs)
        return self._kernel(x, **kernel_kwargs)

    def _kernel(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def has_state(self) -> bool:
        return True

    def step(self, x_t: Any, state: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} is vectorized-only and cannot be used on "
            f"scan-compiled panel models (models with lag() or geometric "
            f"adstock() combined with panel data). Use cross-sectional data, "
            f"remove temporal dependencies, or switch to geometric adstock()."
        )


class Adstock(_ConvAdstockBase):
    """Geometric adstock: ``y_t = sum_{i=0}^{l_max-1} decay**i * x_{t-i}``.

    Applied along the time axis within each panel unit.
    For cross-sectional data, applied along the row axis.

    ``l_max`` and ``normalize`` mirror
    ``pymc_marketing.mmm.transformers.geometric_adstock`` — pass them to
    the constructor and re-register (see :func:`register_transform`) to
    match a reference model's configuration, e.g.
    ``register_transform(Adstock(l_max=8, normalize=True))``.

    Panel models with temporal dependencies are compiled with
    ``pytensor.scan`` and go through :meth:`step`, which only implements the
    *unbounded, unnormalized* recursion ``y_t = x_t + decay * y_{t-1}``.
    Non-default ``l_max`` / ``normalize`` raise ``NotImplementedError`` on
    scan-compiled models rather than silently disagreeing with the vectorized
    kernel.
    """

    name = "adstock"
    param_specs = {
        "decay": ParamSpec(constraint="unit_interval", default_prior="Beta(2, 2)"),
    }

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        return self._convolve(x, panel_info, data, alpha=params["decay"])

    def _kernel(self, x: Any, **kwargs: Any) -> Any:
        return _geometric_adstock(x, **kwargs)

    def step(self, x_t: Any, state: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        if self.l_max != self.DEFAULT_L_MAX or self.normalize != self.DEFAULT_NORMALIZE:
            raise NotImplementedError(
                f"Adstock(l_max={self.l_max}, normalize={self.normalize}) is "
                f"not supported on scan-compiled panel models (models with "
                f"temporal dependencies: adstock()/lag() combined with panel "
                f"data). Adstock.step() only implements the unbounded, "
                f"unnormalized recursion y_t = x_t + decay * y_{{t-1}}. "
                f"Use the default Adstock() for these models."
            )
        decay = params["decay"]
        adstock_t = x_t + decay * state
        return adstock_t, adstock_t


class DelayedAdstock(_ConvAdstockBase):
    """Delayed adstock with peak at lag ``theta``.

    Convolution-based; vectorized only (no ``step()`` for scan paths).
    """

    name = "delayed_adstock"
    param_specs = {
        "decay": ParamSpec(constraint="unit_interval", default_prior="Beta(2, 2)"),
        "theta": ParamSpec(constraint="unit_interval", default_prior="Beta(2, 5)"),
    }

    def emit_prior(self, param_name: str, spec: ParamSpec) -> Any:
        if spec.default_prior == "Beta(2, 5)":
            return pm.Beta(param_name, alpha=2, beta=5)
        return super().emit_prior(param_name, spec)

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        theta = params["theta"] * (self.l_max - 1)
        return self._convolve(x, panel_info, data, alpha=params["decay"], theta=theta)

    def _kernel(self, x: Any, **kwargs: Any) -> Any:
        return _delayed_adstock(x, **kwargs)


class WeibullAdstock(_ConvAdstockBase):
    """Weibull adstock with shape ``k`` and scale ``lam``.

    Convolution-based; vectorized only (no ``step()`` for scan paths).
    """

    name = "weibull_adstock"
    param_specs = {
        "lam": ParamSpec(constraint="positive", default_prior="HalfNormal(1)"),
        "k": ParamSpec(constraint="positive", default_prior="HalfNormal(1)"),
    }

    def __init__(
        self,
        l_max: int = _ConvAdstockBase.DEFAULT_L_MAX,
        normalize: bool = _ConvAdstockBase.DEFAULT_NORMALIZE,
        *,
        weibull_type: str = "PDF",
    ) -> None:
        super().__init__(l_max=l_max, normalize=normalize)
        if weibull_type not in {"PDF", "CDF"}:
            raise ValueError(
                f"weibull_type must be 'PDF' or 'CDF', got {weibull_type!r}."
            )
        self.weibull_type = weibull_type

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        return self._convolve(
            x,
            panel_info,
            data,
            lam=params["lam"],
            k=params["k"],
            weibull_type=self.weibull_type,
        )

    def _kernel(self, x: Any, **kwargs: Any) -> Any:
        return _weibull_adstock(x, **kwargs)


class LogisticSaturation(Transform):
    """Logistic saturation: ``y = (1 - exp(-lam*x)) / (1 + exp(-lam*x))``.

    Pointwise — no temporal dependence.
    """

    name = "logistic_saturation"
    param_specs = {
        "lam": ParamSpec(constraint="positive", default_prior="HalfNormal(1)"),
    }

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        return _logistic_saturation(x, lam=params["lam"])


class MichaelisMenten(Transform):
    """Michaelis-Menten saturation: ``y = alpha * x / (lam + x)``.

    Pointwise — no temporal dependence.
    """

    name = "michaelis_menten"
    param_specs = {
        "alpha": ParamSpec(constraint="positive", default_prior="HalfNormal(1)"),
        "lam": ParamSpec(constraint="positive", default_prior="HalfNormal(1)"),
    }

    def apply_pymc(
        self,
        x: Any,
        params: dict[str, Any],
        *,
        panel_info: Any | None = None,
        data: Any | None = None,
    ) -> Any:
        return _michaelis_menten(x, alpha=params["alpha"], lam=params["lam"])


REGISTRY: dict[str, Transform] = {
    "adstock": Adstock(),
    "delayed_adstock": DelayedAdstock(),
    "weibull_adstock": WeibullAdstock(),
    "logistic_saturation": LogisticSaturation(),
    "michaelis_menten": MichaelisMenten(),
}


def get_transform(name: str) -> Transform:
    """Look up a registered transform by DSL name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available transforms: {', '.join(sorted(REGISTRY))}. "
            f"Register custom transforms with register_transform()."
        )
    return REGISTRY[name]


def register_transform(transform: Transform) -> None:
    """Register a custom transform for use in the pathmc DSL."""
    REGISTRY[transform.name] = transform
