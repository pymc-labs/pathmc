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
"""pathmc — Bayesian path analysis via PyMC."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from packaging.version import Version as _Version

__version__ = _pkg_version("pathmc")

_pymc_ver_str = _pkg_version("pymc")
_pymc_ver = _Version(_pymc_ver_str)
if not (_Version("6.0") <= _pymc_ver < _Version("7")):
    raise ImportError(
        f"pathmc requires PyMC >= 6.0, < 7 (found {_pymc_ver_str}). "
        f"The generative model architecture depends on the PyMC 6 "
        f"do-operator API. Please install a compatible version: "
        f"pip install 'pymc>=6.0,<7'"
    )

from pathmc.effects import EffectResult  # noqa: E402
from pathmc.falsify import FalsificationResult  # noqa: E402
from pathmc.identify import ConditionalIndependence, ImplicationTestResult  # noqa: E402
from pathmc.introspect import EquationList, ModelEquations, PriorTable  # noqa: E402
from pathmc.model import PathModel, model, simulate  # noqa: E402
from pathmc.panel import add_lags as add_lags  # noqa: E402
from pathmc.sensitivity import SensitivityResult  # noqa: E402
from pathmc.simulate import DoResult  # noqa: E402
from pathmc.transforms import ParamSpec, Transform, register_transform  # noqa: E402

# Deliberate re-export so users can build custom priors without a separate
# pymc_extras import (see the `priors=` argument to model()).
from pymc_extras.prior import Prior  # noqa: E402

__all__ = [
    "ConditionalIndependence",
    "DoResult",
    "EffectResult",
    "EquationList",
    "FalsificationResult",
    "ImplicationTestResult",
    "ModelEquations",
    "ParamSpec",
    "PathModel",
    "Prior",
    "PriorTable",
    "SensitivityResult",
    "Transform",
    "__version__",
    "model",
    "register_transform",
    "simulate",
]
