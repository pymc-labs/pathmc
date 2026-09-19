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

from typing import TYPE_CHECKING, Any

from pymc_extras.prior import Prior

from pathmc.parse import HSGPCall, Spec, TransformCall

if TYPE_CHECKING:
    from pathmc.panel import PanelInfo

PriorConfig = dict[str, Prior]
"""Mapping from parameter name to ``Prior`` specification."""

__all__: list[str] = []


def default_priors(
    spec: Spec,
    families: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: set[str] | None = None,
    panel_info: PanelInfo | None = None,
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
    from pathmc.compile import (
        _is_scan_panel,
        _parse_by_var_pooling,
        get_free_predictor_columns,
    )

    if families is None:
        families = {}
    if latent is None:
        latent = set()

    has_intercepts = pooling == "partial" or (
        isinstance(pooling, dict) and pooling.get("intercept", False)
    )
    slope_vars: list[str] = []
    by_var_entries: dict[str, dict[str, Any]] = {}
    if isinstance(pooling, dict):
        slope_vars = list(pooling.get("slopes", []))
        # Dim names are validated against the panel at compile time; here
        # the grammar and variable names are still checked eagerly so
        # model() construction fails fast on malformed pooling configs.
        by_var_entries = _parse_by_var_pooling(pooling, spec, require_panel=False)

    def _is_coef_entry(entry: dict[str, Any]) -> bool:
        return entry["kind"] in ("coefficient", "none_coefficient")

    coef_names = {n for n, e in by_var_entries.items() if _is_coef_entry(e)}

    # Estimated initial conditions (``init_{var}``) exist only for latent
    # variables that feed a ``lag()`` term in scan-compiled panel models:
    # those are the latents whose recursion actually reads its t=0 state.
    lag_base_vars = (
        {
            term.lag_of
            for reg in spec.regressions
            for term in reg.terms
            if term.lag_of is not None
        }
        if _is_scan_panel(spec, panel_info)
        else set()
    )

    priors: PriorConfig = {}
    seen_transform_params: set[str] = set()

    for reg in spec.regressions:
        free_cols = [
            c
            for c in get_free_predictor_columns(
                reg, pooling=pooling, panel_info=panel_info
            )
            if c not in coef_names
        ]
        if free_cols:
            priors[f"beta_{reg.lhs}"] = Prior("Normal", mu=0, sigma=10)

        family = families.get(reg.lhs, "gaussian")
        if reg.lhs in latent:
            if family == "latent_normal":
                priors[f"sigma_{reg.lhs}"] = Prior("HalfNormal", sigma=1)
            if reg.lhs in lag_base_vars:
                priors[f"init_{reg.lhs}"] = Prior("Normal", mu=0, sigma=1)
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

    # --- by_var structured pooling ---
    for name, entry in by_var_entries.items():
        if entry["kind"] == "coefficient":
            key = entry["key"]
            priors[f"mu_{name}_{key}"] = Prior(
                "Normal", mu=0, sigma=10, dims=entry["dims"]
            )
            priors[f"sigma_{name}_{key}"] = Prior("HalfNormal", sigma=1)
        elif entry["kind"] == "none_coefficient":
            priors[f"beta_{name}"] = Prior("Normal", mu=0, sigma=10, dims=("unit",))
        elif name in priors:
            # "none" on a transform parameter: same default family, but
            # per-cell (one parameter per panel unit) instead of a shared scalar.
            per_cell = priors[name].deepcopy()
            per_cell.dims = ("unit",)
            priors[name] = per_cell

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

    ``beta_hsgp`` is registered only in the non-centered parametrization.  In
    centered mode ``assemble_hsgp_term`` builds ``beta`` directly with the
    data-derived ``sqrt_psd`` scale and never reads a ``beta_hsgp`` prior, so
    registering the key would advertise a tunable knob that has no effect and
    silently swallow user overrides.  Leaving it unregistered means an override
    on ``beta_hsgp`` in centered mode raises the usual "Unknown prior key"
    instead of being a silent no-op; tune ``ell``/``eta`` in centered mode.
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
    if not call.centered and beta_key not in priors:
        priors[beta_key] = Prior("Normal", mu=0, sigma=1, dims=(weights_dim,))


def _default_prior_for_constraint(constraint: str) -> Prior:
    """Return the default Prior for a given parameter constraint."""
    if constraint == "unit_interval":
        return Prior("Beta", alpha=2, beta=2)
    if constraint == "positive":
        return Prior("HalfNormal", sigma=1)
    return Prior("Normal", mu=0, sigma=10)
