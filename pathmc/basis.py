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
"""Registry for terms that expand one input into an owned coefficient basis.

A basis term is different from a transform: a transform produces one column
which is multiplied by the equation's scalar ``beta``; a basis produces an
M-column matrix and owns an M-vector of coefficients. The compiler therefore
only knows the common contribution contract, never a basis's identity.

There are two build contracts. A *data basis* implements :meth:`build_data`
and receives observed numeric input, returning its columns and any fit-time
state. That state (levels, knots, centering constants, and so on) must be
frozen at fit and replayed unchanged for prediction and intervention. A
*graph basis* implements :meth:`build_graph`, receiving a PyTensor expression
and returning symbolic columns. It is required whenever input is latent or
inside ``scan``, and when columns depend on estimated parameters. The choice
is a property of the function/input pairing, not the function alone: Fourier
on an observed calendar is a data basis, while Fourier on a latent mediator is
a graph basis. HSGP is always graph-built because its columns depend on kernel
parameters, although its support boundary is fit-time state.

The independent coefficient-prior contract is :meth:`contribution`. The
default is iid Normal weights; bases such as HSGP can declare a structured
prior without teaching the compiler about that structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pathmc.parse import BasisCall, HSGPCall

if TYPE_CHECKING:
    from pathmc.priors import PriorConfig

__all__ = ["Basis", "BasisCall", "register_basis"]


Call = BasisCall | HSGPCall


@dataclass(frozen=True)
class BasisCapabilities:
    """Declarative compatibility metadata for a basis implementation."""

    supports_panel: bool = False
    supports_multidim: bool = False
    supports_endogenous: bool = True
    terminal: bool = True


class Basis:
    """Base class for registered basis terms.

    Subclasses implement either or both column-building contracts and may
    override :meth:`contribution` for a structured coefficient prior.
    """

    name: str
    capabilities = BasisCapabilities()

    def n_basis(self, call: Call) -> int:
        """Return the number of columns produced by *call*."""
        raise NotImplementedError

    def build_data(self, x: np.ndarray, call: Call) -> tuple[np.ndarray, Any]:
        """Build numeric columns and frozen fit-time state from observed input."""
        raise NotImplementedError

    def build_graph(
        self, x: Any, call: Call, *, lhs: str, priors: "PriorConfig"
    ) -> tuple[Any, Any]:
        """Build symbolic columns and basis-specific state from a graph input."""
        raise NotImplementedError

    def default_priors(self, lhs: str, call: Call, priors: "PriorConfig") -> None:
        """Register default priors for this basis's coefficient structure."""
        from pymc_extras.prior import Prior

        key = self.beta_name(lhs, call)
        if key not in priors:
            priors[key] = Prior(
                "Normal", mu=0, sigma=1, dims=(self.weights_dim(lhs, call),)
            )

    def prior_descriptions(self, lhs: str, call: Call) -> dict[str, str]:
        """Return introspection labels for this basis's default priors."""
        return {self.beta_name(lhs, call): "Normal(0, 1)"}

    def contribution(
        self,
        columns: Any,
        state: Any,
        *,
        lhs: str,
        call: Call,
        priors: "PriorConfig",
    ) -> Any:
        """Combine columns with the default iid-Normal owned coefficient vector."""
        from pathmc.priors import _ensure_dims

        beta_name = self.beta_name(lhs, call)
        beta = _ensure_dims(
            priors[beta_name], self.weights_dim(lhs, call)
        ).create_variable(beta_name)
        return pm.Deterministic(self.contribution_name(lhs, call), columns @ beta)

    def assemble_graph(
        self, x: Any, *, lhs: str, call: Call, priors: "PriorConfig"
    ) -> Any:
        """Build a graph basis and return its complete contribution vector."""
        columns, state = self.build_graph(x, call, lhs=lhs, priors=priors)
        return self.contribution(columns, state, lhs=lhs, call=call, priors=priors)

    def weights_dim(self, lhs: str, call: Call) -> str:
        """Return the stable coordinate name for this basis's coefficients."""
        return f"{lhs}_{call.variable}_{self.name}"

    def beta_name(self, lhs: str, call: Call) -> str:
        """Return the stable random-variable name for basis coefficients."""
        return f"beta_{self.name}_{lhs}_{call.variable}"

    def contribution_name(self, lhs: str, call: Call) -> str:
        """Return the deterministic name for this basis contribution."""
        return f"f_{lhs}_{call.variable}"

    def render(self, call: Call) -> str:
        """Return the plain-text equation rendering for *call*."""
        return f"f_{self.name}({call.variable})"

    def render_latex(self, variable: str) -> str:
        """Return a LaTex fragment for an already-escaped input variable."""
        return rf"f_{{\mathrm{{{self.name}}}}}({variable})"


class FourierBasis(Basis):
    """Harmonic sine/cosine expansion with iid Normal coefficient weights."""

    name = "fourier"

    def n_basis(self, call: Call) -> int:
        """Return two columns for each requested harmonic."""
        assert isinstance(call, BasisCall)
        return 2 * int(call.params["n"])

    def build_data(self, x: np.ndarray, call: Call) -> tuple[np.ndarray, None]:
        """Materialize Fourier columns for observed data without fitted state."""
        assert isinstance(call, BasisCall)
        values = np.asarray(x, dtype=float).reshape(-1, 1)
        harmonics = np.arange(1, int(call.params["n"]) + 1, dtype=float)
        angles = 2 * np.pi * values * harmonics / float(call.params["period"])
        return np.concatenate((np.sin(angles), np.cos(angles)), axis=1), None

    def build_graph(
        self, x: Any, call: Call, *, lhs: str, priors: "PriorConfig"
    ) -> tuple[Any, None]:
        """Build Fourier columns symbolically so they update under ``do()``."""
        assert isinstance(call, BasisCall)
        values = pt.as_tensor_variable(x).reshape((-1, 1))
        harmonics = pt.arange(1, int(call.params["n"]) + 1, dtype=values.dtype)
        angles = 2 * np.pi * values * harmonics / float(call.params["period"])
        return pt.concatenate((pt.sin(angles), pt.cos(angles)), axis=1), None


REGISTRY: dict[str, Basis] = {"fourier": FourierBasis()}


def get_basis(name: str) -> Basis:
    """Look up a registered basis by DSL name."""
    if name == "hsgp" and name not in REGISTRY:
        from pathmc.hsgp import HSGPBasis

        REGISTRY[name] = HSGPBasis()
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown basis '{name}'. Available bases: {', '.join(sorted(REGISTRY))}. "
            "Register custom bases with register_basis()."
        )
    return REGISTRY[name]


def register_basis(basis: Basis) -> None:
    """Register a custom basis for use in the pathmc formula DSL."""
    REGISTRY[basis.name] = basis
