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
"""Prior configuration for pathmc models.

Provides default prior specifications and utilities for customizing
priors using the ``Prior`` class from ``pymc_extras``.
"""

from __future__ import annotations

from pymc_extras.prior import Prior

from pathmc.parse import HSGPCall, Spec, TransformCall

PriorConfig = dict[str, Prior]
"""Mapping from parameter name to ``Prior`` specification."""

__all__: list[str] = []


def default_priors(
    spec: Spec,
    families: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
) -> PriorConfig:
    """Build default priors for all customizable model parameters.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    families : dict[str, str] | None
        Per-variable distribution families.
    pooling : str | dict | None
        Pooling specification for panel models.
    latent : set[str] | None
        Latent variables (no sigma/likelihood priors emitted).

    Returns
    -------
    PriorConfig
        Mapping from parameter name to default ``Prior``.
    """
    from pathmc.compile import get_free_predictor_columns

    if families is None:
        families = {}
    if latent is None:
        latent = set()

    has_intercepts = pooling == "partial" or (
        isinstance(pooling, dict) and pooling.get("intercept", False)
    )
    slope_vars: list[str] = []
    if isinstance(pooling, dict):
        slope_vars = list(pooling.get("slopes", []))

    priors: PriorConfig = {}
    seen_transform_params: set[str] = set()

    for reg in spec.regressions:
        if get_free_predictor_columns(reg):
            priors[f"beta_{reg.lhs}"] = Prior("Normal", mu=0, sigma=10)

        family = families.get(reg.lhs, "gaussian")
        if reg.lhs in latent:
            if family == "latent_normal":
                priors[f"sigma_{reg.lhs}"] = Prior("HalfNormal", sigma=1)
        else:
            if family not in ("bernoulli", "poisson", "negbinomial"):
                priors[f"sigma_{reg.lhs}"] = Prior("HalfNormal", sigma=1)
            if family == "negbinomial":
                priors[f"alpha_disp_{reg.lhs}"] = Prior("HalfNormal", sigma=1)
            if family == "studentt":
                priors[f"nu_{reg.lhs}"] = Prior("Gamma", alpha=2, beta=0.1)

        if has_intercepts:
            priors[f"mu_alpha_{reg.lhs}"] = Prior("Normal", mu=0, sigma=10)
            priors[f"sigma_alpha_{reg.lhs}"] = Prior("HalfNormal", sigma=1)

        for svar in slope_vars:
            term_variables = {t.variable for t in reg.terms}
            if svar in term_variables:
                priors[f"mu_slope_{reg.lhs}_{svar}"] = Prior("Normal", mu=0, sigma=10)
                priors[f"sigma_slope_{reg.lhs}_{svar}"] = Prior("HalfNormal", sigma=1)

        for term in reg.terms:
            if term.transform is not None:
                _collect_transform_defaults(
                    term.transform, priors, seen_transform_params
                )
            if term.hsgp is not None:
                _collect_hsgp_defaults(reg.lhs, term.hsgp, priors)

    return priors


def merge_priors(
    defaults: PriorConfig,
    overrides: dict[str, Prior] | None,
) -> PriorConfig:
    """Merge user-specified prior overrides into default prior config.

    Parameters
    ----------
    defaults : PriorConfig
        Default priors built by :func:`default_priors`.
    overrides : dict[str, Prior] | None
        User-supplied overrides. Keys must exist in *defaults*.

    Returns
    -------
    PriorConfig
        Merged prior config.

    Raises
    ------
    ValueError
        If an override key does not match any default prior.
    """
    if overrides is None:
        return dict(defaults)

    unknown = set(overrides) - set(defaults)
    if unknown:
        valid = sorted(defaults.keys())
        raise ValueError(
            f"Unknown prior key(s): {sorted(unknown)}. "
            f"Valid keys for this model: {valid}"
        )

    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _ensure_dims(prior: Prior, dims: tuple[str, ...] | str | None) -> Prior:
    """Return a copy of *prior* with structural dims set.

    If the prior already has matching dims, returns it unchanged.
    If the prior has no dims (``None`` or ``()``), a copy with the
    expected dims is returned. If the prior has conflicting dims,
    raises ``ValueError``.
    """
    if dims is None:
        return prior

    expected = (dims,) if isinstance(dims, str) else tuple(dims)

    if prior.dims is None or prior.dims == ():
        p = prior.deepcopy()
        p.dims = expected
        return p

    if prior.dims == expected:
        return prior

    raise ValueError(
        f"Prior has dims={prior.dims} but the model requires dims={expected}. "
        f"Either omit dims (they will be set automatically) or match the "
        f"expected dimensions."
    )


def _collect_transform_defaults(
    tc: TransformCall,
    priors: PriorConfig,
    seen: set[str],
) -> None:
    """Recursively add default transform parameter priors."""
    from pathmc.transforms import get_transform

    if isinstance(tc.input_expr, TransformCall):
        _collect_transform_defaults(tc.input_expr, priors, seen)

    transform = get_transform(tc.name)
    for param_key, param_name in tc.params.items():
        if param_name not in seen:
            seen.add(param_name)
            pspec = transform.param_specs[param_key]
            priors[param_name] = _default_prior_for_constraint(pspec.constraint)


def _collect_hsgp_defaults(
    lhs: str,
    call: HSGPCall,
    priors: PriorConfig,
) -> None:
    """Register default HSGP hyperpriors (ell, eta, beta_hsgp) for one term.

    Keys are the flat, override-able RV names ``ell_{lhs}_{var}`` (lengthscale),
    ``eta_{lhs}_{var}`` (amplitude), and ``beta_hsgp_{lhs}_{var}`` (standardized
    basis weights, non-centered).  Deduplicated by RV name.
    """
    var = call.variable
    ell_key = f"ell_{lhs}_{var}"
    eta_key = f"eta_{lhs}_{var}"
    beta_key = f"beta_hsgp_{lhs}_{var}"
    weights_dim = f"{lhs}_{var}_hsgp"

    if ell_key not in priors:
        # Weakly-informative InverseGamma avoids the ell -> 0 degeneracy via
        # its thin left tail; LogNormal is a documented alternative.
        priors[ell_key] = Prior("InverseGamma", alpha=3, beta=1)
    if eta_key not in priors:
        priors[eta_key] = Prior("HalfNormal", sigma=1)
    if beta_key not in priors:
        priors[beta_key] = Prior("Normal", mu=0, sigma=1, dims=(weights_dim,))


def _default_prior_for_constraint(constraint: str) -> Prior:
    """Return the default Prior for a given parameter constraint."""
    if constraint == "unit_interval":
        return Prior("Beta", alpha=2, beta=2)
    if constraint == "positive":
        return Prior("HalfNormal", sigma=1)
    return Prior("Normal", mu=0, sigma=10)
