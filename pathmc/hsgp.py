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
basis recomputes under ``pm.do()`` as long as the intervention stays within
the fitted support.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pymc as pm
from pytensor.tensor.variable import TensorVariable

from pathmc.parse import HSGPCall
from pathmc.priors import PriorConfig

__all__: list[str] = []

TensorLike: TypeAlias = TensorVariable | np.ndarray

COV_FUNCS: dict[str, type] = {
    "expquad": pm.gp.cov.ExpQuad,
    "matern52": pm.gp.cov.Matern52,
    "matern32": pm.gp.cov.Matern32,
}


def make_cov_func(cov: str, *, eta: TensorLike, ell: TensorLike) -> object:
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
    call: HSGPCall, x: TensorLike, *, cov_func: object
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
    from pathmc.priors import _ensure_dims

    var = call.variable
    ell = priors[f"ell_{lhs}_{var}"].create_variable(f"ell_{lhs}_{var}")
    eta = priors[f"eta_{lhs}_{var}"].create_variable(f"eta_{lhs}_{var}")

    cov_func = make_cov_func(call.cov, eta=eta, ell=ell)
    phi, sqrt_psd, n_basis = hsgp_basis(call, x, cov_func=cov_func)
    assert n_basis == call.m, (
        f"HSGP basis count {n_basis} != requested m={call.m} for '{lhs}_{var}'."
    )

    weights_dim = f"{lhs}_{var}_hsgp"
    beta_name = f"beta_hsgp_{lhs}_{var}"

    if call.centered:
        # Coefficient scale is data-derived (sqrt_psd), so beta is built
        # directly rather than from the Prior config; a user override on
        # beta_hsgp has no effect in centered mode (tune ell/eta instead).
        beta = pm.Normal(beta_name, mu=0.0, sigma=sqrt_psd, dims=weights_dim)
        f = phi @ beta
    else:
        beta_prior = _ensure_dims(priors[beta_name], weights_dim)
        beta = beta_prior.create_variable(beta_name)
        f = phi @ (beta * sqrt_psd)

    return pm.Deterministic(f"f_{lhs}_{var}", f)
