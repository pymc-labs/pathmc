"""Transform registry and built-in transforms (adstock, logistic_saturation).

Each transform is a callable that can produce both PyMC tensor operations
(for model compilation) and numpy array operations (for do() simulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan as pt_scan


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

    Subclasses must implement :meth:`apply_pymc` and :meth:`apply_numpy`
    and define :attr:`name` and :attr:`param_specs`.
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

    def apply_numpy(
        self,
        x: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply the transform using numpy arrays (for do() simulation).

        Parameters
        ----------
        x : np.ndarray
            Input array.
        params : dict
            Posterior draws for each parameter.
        """
        raise NotImplementedError


class Adstock(Transform):
    """Geometric adstock: ``y_t = x_t + decay * y_{t-1}``.

    Applied along the time axis within each panel unit.
    For cross-sectional data, applied along the row axis.
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
        decay = params["decay"]

        if panel_info is not None and data is not None:
            return self._apply_pymc_panel(x, decay, panel_info, data)
        return self._apply_pymc_scan(x, decay)

    def _apply_pymc_scan(self, x: Any, decay: Any) -> Any:
        """Apply geometric adstock using pytensor scan."""

        def step(x_t: Any, y_prev: Any, d: Any) -> Any:
            return x_t + d * y_prev

        result, _ = pt_scan(
            fn=step,
            sequences=[x],
            outputs_info=[pt.zeros(())],
            non_sequences=[decay],
        )
        return result

    def _apply_pymc_panel(self, x: Any, decay: Any, panel_info: Any, data: Any) -> Any:
        """Apply adstock within each unit, concatenate back."""
        unit_col = panel_info.unit
        time_col = panel_info.time

        sorted_idx = np.array(data.sort_values([unit_col, time_col]).index)
        reverse_idx = np.argsort(sorted_idx)

        x_sorted = x[sorted_idx]

        units = panel_info.unit_labels
        unit_sizes = data.groupby(unit_col).size()

        chunks = []
        offset = 0
        for unit in units:
            n = int(unit_sizes[unit])
            chunk = x_sorted[offset : offset + n]

            def step(x_t: Any, y_prev: Any, d: Any) -> Any:
                return x_t + d * y_prev

            result, _ = pt_scan(
                fn=step,
                sequences=[chunk],
                outputs_info=[pt.zeros(())],
                non_sequences=[decay],
            )
            chunks.append(result)
            offset += n

        concatenated = pt.concatenate(chunks)
        return concatenated[reverse_idx]

    def apply_numpy(
        self,
        x: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> np.ndarray:
        decay = params["decay"]
        if np.ndim(decay) == 0:
            decay = float(decay)
        n = len(x)
        result = np.zeros(n)
        for t in range(n):
            result[t] = x[t] + (decay * result[t - 1] if t > 0 else 0.0)
        return result

    def apply_numpy_scalar(
        self,
        x_val: float,
        prev_adstocked: float,
        decay: float | np.ndarray,
    ) -> float | np.ndarray:
        """Single-step adstock for time-forward do() simulation."""
        return x_val + decay * prev_adstocked


class LogisticSaturation(Transform):
    """Logistic saturation: ``y = 1 - exp(-lam * x)``.

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
        return 1.0 - pt.exp(-lam * x)

    def apply_numpy(
        self,
        x: np.ndarray,
        params: dict[str, np.ndarray],
    ) -> np.ndarray:
        lam = params["lam"]
        return 1.0 - np.exp(-lam * x)


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
