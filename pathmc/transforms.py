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

The built-in transforms are implemented directly in pytensor. They previously
delegated to ``pymc_marketing.mmm.transformers``, but pymc-marketing does not
yet support PyMC 6 / ArviZ 1, so the two small kernels are vendored here (see
the geometric-adstock truncation note on :func:`_geometric_adstock`). Once
pymc-marketing ships PyMC 6 support we may delegate again — tracked in a
follow-up issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ["ParamSpec", "Transform", "register_transform"]


def _geometric_adstock(
    x: Any, *, alpha: Any, l_max: int, normalize: bool = False
) -> Any:
    """Geometric adstock along the leading (time) axis.

    ``y[t] = sum_{i=0}^{l_max-1} alpha**i * x[t-i]``, with zero padding before
    ``t = 0``. When ``normalize`` is ``True`` the result is divided by
    ``sum_{i=0}^{l_max-1} alpha**i`` so the lag weights sum to 1. The
    carryover is truncated at ``l_max`` lags, and both the truncation and the
    ``normalize`` behaviour match
    ``pymc_marketing.mmm.transformers.geometric_adstock``. This kernel is
    significantly faster than ``pytensor.scan`` and has cleaner gradients for
    NUTS sampling. Batch axes after the first (e.g. panel units) are
    broadcast over.
    """
    if l_max < 1:
        raise ValueError(
            f"l_max must be >= 1, got {l_max}. Geometric adstock needs at "
            f"least one lag weight; set l_max to the maximum carryover "
            f"duration in time steps."
        )
    w = pt.power(alpha, pt.arange(l_max))
    result = w[0] * x
    for i in range(1, l_max):
        shifted = pt.zeros_like(x)
        shifted = pt.set_subtensor(shifted[i:], x[:-i])
        result = result + w[i] * shifted
    if normalize:
        result = result / pt.sum(w)
    return result


def _logistic_saturation(x: Any, *, lam: Any) -> Any:
    """Pointwise logistic saturation: ``(1 - exp(-lam*x)) / (1 + exp(-lam*x))``."""
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


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
    """Geometric adstock: ``y_t = sum_{i=0}^{l_max-1} decay**i * x_{t-i}``.

    Applied along the time axis within each panel unit.
    For cross-sectional data, applied along the row axis.

    The PyMC graph uses the vectorized :func:`_geometric_adstock`
    kernel, which is significantly faster than ``pytensor.scan`` and
    produces cleaner gradients for NUTS sampling. ``l_max`` and
    ``normalize`` mirror the corresponding arguments of
    ``pymc_marketing.mmm.transformers.geometric_adstock`` — pass them to
    the constructor and re-register (see :func:`register_transform`) to
    match a reference model's configuration, e.g.
    ``register_transform(Adstock(l_max=8, normalize=True))``.

    Note that :meth:`step`, used inside ``pytensor.scan`` for panel
    interventions with temporal dependencies, implements the *unbounded*
    recursion ``y_t = x_t + decay * y_{t-1}`` — the ``l_max -> inf`` limit
    of the truncated kernel above. For ``decay`` close to 1 with a small
    ``l_max`` the two paths can disagree; increase ``l_max`` to tighten
    the approximation.
    """

    name = "adstock"
    param_specs = {
        "decay": ParamSpec(constraint="unit_interval", default_prior="Beta(2, 2)"),
    }

    def __init__(self, l_max: int = 12, normalize: bool = False) -> None:
        """Create a geometric-adstock transform.

        Parameters
        ----------
        l_max : int
            Number of lags (including lag 0) included in the truncated
            carryover sum. Matches ``pymc_marketing``'s ``l_max``.
        normalize : bool
            If ``True``, divide by the sum of the lag weights so they sum
            to 1. Matches ``pymc_marketing``'s ``normalize``.
        """
        self.l_max = l_max
        self.normalize = normalize

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
        return _geometric_adstock(
            x, alpha=decay, l_max=self.l_max, normalize=self.normalize
        )

    def _apply_pymc_panel(self, x: Any, decay: Any, panel_info: Any, data: Any) -> Any:
        """Apply adstock per unit via matrix reshaping, not per-unit scans."""
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

        adstocked = _geometric_adstock(
            x_matrix, alpha=decay, l_max=self.l_max, normalize=self.normalize
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
        lam = params["lam"]
        return _logistic_saturation(x, lam=lam)


REGISTRY: dict[str, Transform] = {
    "adstock": Adstock(),
    "logistic_saturation": LogisticSaturation(),
}


def get_transform(name: str) -> Transform:
    """Look up a registered transform by DSL name.

    Parameters
    ----------
    name : str
        Transform name used in the model DSL, such as ``"adstock"``.

    Returns
    -------
    Transform
        Registered transform instance.

    Raises
    ------
    ValueError
        If no transform is registered under *name*.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available transforms: {', '.join(sorted(REGISTRY))}. "
            f"Register custom transforms with register_transform()."
        )
    return REGISTRY[name]


def register_transform(transform: Transform) -> None:
    """Register a custom transform for use in the pathmc DSL.

    Parameters
    ----------
    transform : Transform
        Transform instance with a unique ``name`` and ``param_specs``.

    Examples
    --------
    Register a custom transform before building a model:

    >>> class Square(Transform):
    ...     name = "square"
    ...     param_specs = {}
    ...
    ...     def apply_pymc(self, x, params, *, panel_info=None, data=None):
    ...         return x**2
    >>> register_transform(Square())
    """
    REGISTRY[transform.name] = transform
