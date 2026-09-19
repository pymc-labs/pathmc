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
"""Hilbert Space Gaussian Process (HSGP) basis construction.

Isolates the kernel/basis math from the PyMC compiler so it can be unit
tested and later extended (Matern variants, multi-D, grouped GPs) without
touching ``compile.py``.  The compiler calls :func:`assemble_hsgp_term`; the
kernel factory and basis builder are exposed for direct testing.

The design follows PyMC's linearized HSGP recipe
(:meth:`pymc.gp.HSGP.prior_linearized`): ``f = phi @ (beta * sqrt_psd)``
for the non-centered parametrization.  ``prior_linearized`` centers the input
internally (from ``L``/``c``) and derives the boundary at build time, so the
basis recomputes under ``pm.do()`` only within the frozen boundary
``[mid - L, mid + L]``; beyond it the sinusoidal eigenfunctions alias.
:func:`hsgp_intervention_bounds` recomputes that boundary so ``do()`` can
reject out-of-support interventions.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable

from pathmc.basis import Basis, BasisCapabilities, Call
from pathmc.parse import HSGPCall, Spec
from pathmc.priors import PriorConfig

__all__: list[str] = []

TensorLike: TypeAlias = TensorVariable | np.ndarray

COV_FUNCS: dict[str, type] = {
    "expquad": pm.gp.cov.ExpQuad,
    "matern52": pm.gp.cov.Matern52,
    "matern32": pm.gp.cov.Matern32,
}


class HSGPBasis(Basis):
    """Registry adapter for HSGP's parameter-dependent graph basis."""

    name = "hsgp"
    capabilities = BasisCapabilities(supports_endogenous=False)

    def n_basis(self, call: Call) -> int:
        """Return the requested number of Laplacian eigenfunctions."""
        assert isinstance(call, HSGPCall)
        return call.m

    def build_data(
        self, x: np.ndarray, call: Call, *, state: Any | None = None
    ) -> tuple[np.ndarray, Any]:
        """Reject numeric materialization because HSGP depends on parameters."""
        raise NotImplementedError(
            "hsgp() is a graph basis because its columns depend on the "
            "estimated ell and eta parameters."
        )

    def build_graph(
        self,
        x: TensorLike,
        call: Call,
        *,
        lhs: str,
        priors: PriorConfig,
        state: Any | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """Build HSGP columns and their spectral scaling state."""
        assert isinstance(call, HSGPCall)
        ell = priors[f"ell_{lhs}_{call.variable}"].create_variable(
            f"ell_{lhs}_{call.variable}"
        )
        eta = priors[f"eta_{lhs}_{call.variable}"].create_variable(
            f"eta_{lhs}_{call.variable}"
        )
        columns, sqrt_psd, n_basis = hsgp_basis(
            call, x, cov_func=make_cov_func(call.cov, eta=eta, ell=ell)
        )
        if n_basis != call.m:
            raise ValueError(
                f"HSGP basis count {n_basis} != requested m={call.m} for "
                f"'{lhs}_{call.variable}'."
            )
        return columns, sqrt_psd

    def default_priors(self, lhs: str, call: Call, priors: PriorConfig) -> None:
        """Register HSGP hyperpriors and non-centered weight priors."""
        from pymc_extras.prior import Prior

        assert isinstance(call, HSGPCall)
        var = call.variable
        priors.setdefault(f"ell_{lhs}_{var}", Prior("InverseGamma", alpha=3, beta=1))
        priors.setdefault(f"eta_{lhs}_{var}", Prior("HalfNormal", sigma=1))
        if not call.centered:
            priors.setdefault(
                self.beta_name(lhs, call),
                Prior("Normal", mu=0, sigma=1, dims=(self.weights_dim(lhs, call),)),
            )

    def prior_descriptions(self, lhs: str, call: Call) -> dict[str, str]:
        """Return the HSGP hyperprior labels used by introspection."""
        assert isinstance(call, HSGPCall)
        descriptions = {
            f"ell_{lhs}_{call.variable}": "InverseGamma(3, 1)",
            f"eta_{lhs}_{call.variable}": "HalfNormal(1)",
        }
        if not call.centered:
            descriptions[self.beta_name(lhs, call)] = "Normal(0, 1)"
        return descriptions

    def contribution(
        self,
        columns: TensorLike,
        sqrt_psd: TensorLike,
        *,
        lhs: str,
        call: Call,
        priors: PriorConfig,
    ) -> TensorVariable:
        """Apply HSGP's centered or non-centered spectral prior structure."""
        from pathmc.priors import _ensure_dims

        assert isinstance(call, HSGPCall)
        beta_name = self.beta_name(lhs, call)
        if call.centered:
            beta = pm.Normal(
                beta_name,
                mu=0.0,
                sigma=sqrt_psd,
                dims=self.weights_dim(lhs, call),
            )
            contribution = columns @ beta
        else:
            beta = _ensure_dims(
                priors[beta_name], self.weights_dim(lhs, call)
            ).create_variable(beta_name)
            contribution = columns @ (beta * sqrt_psd)
        return pm.Deterministic(self.contribution_name(lhs, call), contribution)


def make_cov_func(
    cov: str, *, eta: TensorLike, ell: TensorLike
) -> pm.gp.cov.Covariance:
    """Build the HSGP covariance function ``eta**2 * Kernel(input_dim=1, ls=ell)``.

    Parameters
    ----------
    cov : str
        Kernel name, one of ``"expquad"``, ``"matern52"``, ``"matern32"``
        (case-insensitive).
    eta : tensor
        Amplitude random variable; the kernel is scaled by ``eta**2``.
    ell : tensor
        Lengthscale random variable passed as the kernel ``ls``.

    Returns
    -------
    pymc.gp.cov.Covariance
        The parameterized covariance function.

    Raises
    ------
    ValueError
        If *cov* is not a known kernel name.
    """
    key = cov.lower()
    if key not in COV_FUNCS:
        raise ValueError(
            f"Unknown hsgp cov '{cov}'. Valid kernels: {sorted(COV_FUNCS)}."
        )
    kernel = COV_FUNCS[key]
    return eta**2 * kernel(input_dim=1, ls=ell)


def hsgp_basis(
    call: HSGPCall, x: TensorLike, *, cov_func: pm.gp.cov.Covariance
) -> tuple[TensorLike, TensorLike, int]:
    """Return ``(phi, sqrt_psd, n_basis)`` for a 1-D input via ``prior_linearized``.

    Constructs ``pm.gp.HSGP(m=[call.m], L=[call.L] if call.L is not None
    else None, c=call.c, cov_func=cov_func)`` -- both ``m`` and ``L`` must be
    per-dimension sequences, so the scalar ``call.L`` is wrapped into
    ``[call.L]``.

    ``cov_func`` is required and must be the ``eta``/``ell``-parameterized
    kernel from :func:`make_cov_func`; ``pm.gp.HSGP`` derives ``sqrt_psd`` from
    it, so the basis is only consistent when the same kernel object that
    carries the estimated hyperparameters is threaded in.  ``hsgp_basis`` does
    not build the kernel itself.

    Parameters
    ----------
    call : HSGPCall
        Parsed HSGP term carrying ``m``, ``c``/``L``.
    x : tensor
        Input of shape ``(n, 1)``.
    cov_func : pymc.gp.cov.Covariance
        Covariance function from :func:`make_cov_func`.

    Returns
    -------
    tuple
        ``(phi, sqrt_psd, n_basis)`` -- ``phi`` of shape ``(n, m)``,
        ``sqrt_psd`` of shape ``(m,)``, and ``n_basis == m``.
    """
    boundary = [call.L] if call.L is not None else None
    gp = pm.gp.HSGP(m=[call.m], L=boundary, c=call.c, cov_func=cov_func)
    phi, sqrt_psd = gp.prior_linearized(x)
    return phi, sqrt_psd, gp.n_basis_vectors


def hsgp_intervention_bounds(spec: Spec, data: Any) -> dict[str, tuple[float, float]]:
    """Return the valid intervention interval per HSGP input variable.

    Mirrors how ``pm.gp.HSGP.prior_linearized`` freezes the basis support at
    build time (``pymc.gp.hsgp_approx``, pinned to ``pymc >= 6, < 7``): the
    midpoint is ``(max(x) + min(x)) / 2`` and the boundary is ``L`` when given
    explicitly, else ``c * (max(x) - min(x)) / 2``.  Outside
    ``[mid - L, mid + L]`` the sinusoidal eigenfunctions alias, so any value
    there produces a finite but meaningless smooth.

    Parameters
    ----------
    spec : Spec
        Parsed model spec whose regression terms may carry ``HSGPCall``s.
    data : DataFrame
        The exact frame the model was compiled against (columns expose
        ``.min()`` / ``.max()``; pandas and narwhals frames both qualify).

    Returns
    -------
    dict
        ``{variable: (lower, upper)}``.  A variable smoothed by several
        terms gets the intersection of their intervals, since a value must
        lie inside every basis support to be valid for all of them.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for reg in spec.regressions:
        for term in reg.terms:
            call = term.hsgp
            if call is None or call.variable not in data.columns:
                continue
            col_min = data[call.variable].min()
            col_max = data[call.variable].max()
            if col_min is None or col_max is None:
                continue
            mid = (float(col_max) + float(col_min)) / 2.0
            half = (float(col_max) - float(col_min)) / 2.0
            if call.L is not None:
                length = float(call.L)
            else:
                # The parser enforces exactly one of c/L.
                assert call.c is not None
                length = call.c * half
            lo, hi = mid - length, mid + length
            if call.variable in bounds:
                prev_lo, prev_hi = bounds[call.variable]
                lo, hi = max(lo, prev_lo), min(hi, prev_hi)
            bounds[call.variable] = (lo, hi)
    return bounds


def assemble_hsgp_term(
    call: HSGPCall,
    x: TensorLike,
    *,
    lhs: str,
    priors: PriorConfig,
) -> TensorVariable:
    """Emit the HSGP hyperparameters and coefficients, returning ``f_{lhs}_{var}``.

    Must be called inside an active ``pm.Model`` context.  In order:

    1. create the ``ell_{lhs}_{var}`` / ``eta_{lhs}_{var}`` RVs from *priors*;
    2. build ``cov_func = make_cov_func(call.cov, eta=eta, ell=ell)``;
    3. call ``hsgp_basis(call, x, cov_func=cov_func)``;
    4. assert ``n_basis == call.m``;
    5. create ``beta_hsgp_{lhs}_{var}`` per the parametrization branch;
    6. return ``pm.Deterministic(f"f_{lhs}_{var}", ...)``.

    Parameters
    ----------
    call : HSGPCall
        Parsed HSGP term.
    x : tensor
        Input of shape ``(n, 1)``.
    lhs : str
        Left-hand-side variable name of the equation, used to name RVs.
    priors : PriorConfig
        Merged prior config providing ``ell_{lhs}_{var}``, ``eta_{lhs}_{var}``,
        and (non-centered) ``beta_hsgp_{lhs}_{var}``.

    Returns
    -------
    TensorVariable
        The ``f_{lhs}_{var}`` deterministic smooth of shape ``(n,)``.
    """
    return HSGPBasis().assemble_graph(x, lhs=lhs, call=call, priors=priors)
