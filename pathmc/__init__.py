"""pathmc — Bayesian path analysis via PyMC."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

_pymc_ver_str = _pkg_version("pymc")
_pymc_ver_tuple = tuple(int(x) for x in _pymc_ver_str.split(".")[:3])
if _pymc_ver_tuple < (5, 22, 0):
    raise ImportError(
        f"pathmc requires PyMC >= 5.22.0 (found {_pymc_ver_str}). "
        f"The generative model architecture depends on pm.do() fixes "
        f"from PyMC 5.22. Please upgrade: pip install 'pymc>=5.22.0'"
    )

from pathmc.model import model, simulate  # noqa: E402
from pathmc.panel import add_lags  # noqa: E402
from pymc_extras.prior import Prior  # noqa: E402

__all__ = ["Prior", "add_lags", "model", "simulate"]
