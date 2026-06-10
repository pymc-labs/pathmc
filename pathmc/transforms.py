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
"""Transform registry and built-in transforms (adstock, logistic_saturation).

Each transform produces PyMC tensor operations for model compilation and
provides a ``step()`` method for use inside ``pytensor.scan`` bodies.

Both built-in transforms delegate to ``pymc_marketing.mmm.transformers``. That
module exposes its transforms through the xtensor API (``pymc-marketing >=
0.19``), so the small ``_pmm_*`` helpers below bridge plain ``TensorVariable``s
to xtensor and back. The unwrapped results also lower correctly inside
``pytensor.scan`` bodies, so the same helpers serve both the graph and the
scan-step paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
from pymc_marketing.mmm.transformers import (
    geometric_adstock as _pmm_geometric_adstock,
)
from pymc_marketing.mmm.transformers import (
    logistic_saturation as _pmm_logistic_saturation,
)
from pytensor.xtensor.type import as_xtensor


def _adstock_pmm(x: Any, *, alpha: Any, l_max: int, dims: tuple[str, ...]) -> Any:
    """Geometric adstock convolved along the leading (time) axis.

    ``dims`` labels each axis of ``x``; ``dims[0]`` is the time axis the
    convolution runs over. The result is unwrapped back to a ``TensorVariable``.
    """
    xx = as_xtensor(x, dims=dims)
    adstocked = _pmm_geometric_adstock(xx, alpha=alpha, l_max=l_max, dim=dims[0])
    return adstocked.values


def _logistic_saturation_pmm(x: Any, *, lam: Any) -> Any:
    """Pointwise logistic saturation: ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))``.

    The transform is elementwise, so the xtensor dim labels are immaterial; the
    result is unwrapped back to a plain ``TensorVariable``.
    """
    dims = tuple(f"d{i}" for i in range(x.ndim))
    return _pmm_logistic_saturation(as_xtensor(x, dims=dims), lam=lam).values


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
        """Create a PyMC random variable for a transform parameter.

        Parameters
        ----------
        param_name : str
            The user-chosen name for this parameter instance.
        spec : ParamSpec
            Constraint and prior specification.

        Returns
        -------
        Any
            A PyMC random variable.
        """
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
        """Apply the transform in the PyMC computation graph.

        Parameters
        ----------
        x : tensor
            Input tensor (data column or output of inner transform).
        params : dict
            PyMC random variables for each parameter.
        panel_info : PanelInfo | None
            Panel metadata for time-aware transforms.
        data : DataFrame | None
            Observed data for panel indexing.
        """
        raise NotImplementedError

    @property
    def has_state(self) -> bool:
        """Whether this transform carries state across time steps."""
        return False

    def step(self, x_t: Any, state: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        """Apply one time step inside a ``pytensor.scan`` body.

        Parameters
        ----------
        x_t : tensor
            Input value(s) at time *t*.
        state : tensor
            Carry state from the previous time step.
        params : dict
            PyMC random variables for each parameter.

        Returns
        -------
        tuple[tensor, tensor]
            ``(output_t, new_state)``.  Pointwise transforms return
            ``(f(x_t), state)`` unchanged.
        """
        return self.apply_pymc(x_t, params), state


class Adstock(Transform):
    """Geometric adstock: ``y_t = x_t + decay * y_{t-1}``.

    Applied along the time axis within each panel unit.
    For cross-sectional data, applied along the row axis.

    The PyMC graph uses the convolution-based implementation from
    ``pymc_marketing.mmm.transformers.geometric_adstock``, which is
    significantly faster than ``pytensor.scan`` and produces cleaner
    gradients for NUTS sampling.
    """

    name = "adstock"
    l_max: int = 12
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
        decay = params["decay"]

        if panel_info is not None and data is not None:
            return self._apply_pymc_panel(x, decay, panel_info, data)
        return _adstock_pmm(x, alpha=decay, l_max=self.l_max, dims=("time",))

    def _apply_pymc_panel(self, x: Any, decay: Any, panel_info: Any, data: Any) -> Any:
        """Apply adstock per unit via matrix reshaping, not per-unit scans."""
        unit_col = panel_info.unit
        time_col = panel_info.time
        units = panel_info.unit_labels
        n_units = len(units)
        n_time = len(data) // n_units

        sorted_idx = np.array(data.sort_values([unit_col, time_col]).index)
        reverse_idx = np.argsort(sorted_idx)

        x_sorted = x[sorted_idx]
        x_matrix = x_sorted.reshape((n_units, n_time)).T  # (time, units)

        adstocked = _adstock_pmm(
            x_matrix, alpha=decay, l_max=self.l_max, dims=("time", "unit")
        )

        result_flat = adstocked.T.flatten()  # back to unit-major order
        return result_flat[reverse_idx]

    @property
    def has_state(self) -> bool:
        return True

    def step(self, x_t: Any, state: Any, params: dict[str, Any]) -> tuple[Any, Any]:
        """Single time-step geometric adstock: ``y_t = x_t + decay * y_{t-1}``."""
        decay = params["decay"]
        adstock_t = x_t + decay * state
        return adstock_t, adstock_t


class LogisticSaturation(Transform):
    """Logistic saturation: ``y = 1 - exp(-lam * x)``.

    Pointwise — no temporal dependence.

    The PyMC graph uses ``pymc_marketing.mmm.transformers.logistic_saturation``
    for consistency with the adstock implementation.
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
        lam = params["lam"]
        return _logistic_saturation_pmm(x, lam=lam)


REGISTRY: dict[str, Transform] = {
    "adstock": Adstock(),
    "logistic_saturation": LogisticSaturation(),
}


def get_transform(name: str) -> Transform:
    """Look up a transform by name. Raises ValueError if not found."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available transforms: {', '.join(sorted(REGISTRY))}. "
            f"Register custom transforms with register_transform()."
        )
    return REGISTRY[name]


def register_transform(transform: Transform) -> None:
    """Register a custom transform for use in the DSL."""
    REGISTRY[transform.name] = transform
