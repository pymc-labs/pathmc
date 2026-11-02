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
"""Residual covariance structures for pathmc.

Provides a protocol for pluggable residual structures and the
default LKJ Cholesky implementation. The coefficient betas and mu
construction stay in the main compiler; the protocol only owns the
covariance parameterization and likelihood emission.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pymc as pm

__all__: list[str] = []


@runtime_checkable
class ResidualStructure(Protocol):
    """Protocol for residual covariance structures.

    Encapsulates how correlated residuals are parameterized and
    emitted as a joint likelihood. Implementations receive pre-built
    linear predictor tensors and emit the joint distribution.
    """

    def emit(
        self,
        block_vars: list[str],
        mu_dict: dict[str, Any],
        data_dict: dict[str, np.ndarray],
        priors: dict[str, Any] | None,
        *,
        observed: bool = True,
    ) -> Any:
        """Create covariance priors and emit the joint distribution.

        Parameters
        ----------
        block_vars : list[str]
            Sorted variable names in the residual block.
        mu_dict : dict[str, Any]
            Pre-built linear predictor tensors keyed by variable name.
        data_dict : dict[str, np.ndarray]
            Observed data arrays keyed by variable name.
        priors : dict[str, Any] | None
            Prior configuration (may contain structure-specific keys).
        observed : bool
            When True (estimation), emit an observed likelihood. When False
            (``simulate()``), emit a free joint RV whose draws are the
            realized correlated residuals.
        """
        ...

    def prior_keys(self, block_vars: list[str]) -> list[str]:
        """Return prior parameter names this structure introduces.

        Used by introspection to display customizable keys.

        Parameters
        ----------
        block_vars : list[str]
            Sorted variable names in the residual block.

        Returns
        -------
        list[str]
            Prior parameter names (e.g. ``["chol_M1_M2"]``).
        """
        ...


def _sigma_priors(block_vars: list[str], priors: dict[str, Any] | None) -> list[Any]:
    """Return the residual-scale prior for each block member, in order."""
    from pymc_extras.prior import Prior

    result = []
    for var in block_vars:
        key = f"sigma_{var}"
        if priors and key in priors:
            result.append(priors[key])
        else:
            result.append(Prior("HalfNormal", sigma=1))
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _is_default_residual_scale(prior: Any) -> bool:
    params = getattr(prior, "parameters", None) or {}
    if getattr(prior, "distribution", None) != "HalfNormal":
        return False
    if getattr(prior, "transform", None) or not getattr(prior, "centered", True):
        return False
    if getattr(prior, "dims", None):
        return False
    if set(params) - {"sigma"}:
        return False
    sigma = params.get("sigma", 1)
    return _is_number(sigma) and float(sigma) == 1.0


def _same_family(specs: list[Any]) -> bool:
    first = specs[0]
    keys = tuple(first.parameters)
    return all(
        spec.distribution == first.distribution
        and tuple(spec.parameters) == keys
        and not spec.transform
        and spec.centered
        and not spec.dims
        for spec in specs
    )


def _vectorizable(specs: list[Any]) -> bool:
    if not _same_family(specs):
        return False
    for name in specs[0].parameters:
        values = [spec.parameters[name] for spec in specs]
        if all(_is_number(value) for value in values):
            continue
        if all(hasattr(value, "create_variable") for value in values):
            continue
        return False
    return True


def _format_param(value: Any) -> str:
    if hasattr(value, "distribution"):
        parts = ", ".join(
            f"{name}={_format_param(param)}" for name, param in value.parameters.items()
        )
        return f"{value.distribution}({parts})"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_param(item) for item in value) + "]"
    return str(value)


def _format_sd_dist(specs: list[Any]) -> str:
    if _vectorizable(specs):
        parts = []
        for name in specs[0].parameters:
            values = [spec.parameters[name] for spec in specs]
            if (
                all(_is_number(value) for value in values)
                and len({float(value) for value in values}) == 1
            ):
                parts.append(f"{name}={_format_param(values[0])}")
            else:
                parts.append(f"{name}={_format_param(values)}")
        return f"{specs[0].distribution}({', '.join(parts)})"
    members = [_format_param(spec) for spec in specs]
    return "[" + ", ".join(members) + "]"


def residual_sd_description(
    block_vars: list[str], priors: dict[str, Any] | None
) -> str:
    """Describe the residual-scale prior that compilation will use."""
    specs = _sigma_priors(block_vars, priors)
    if all(_is_default_residual_scale(spec) for spec in specs):
        return "LKJCholeskyCov(eta=2, sd_dist=HalfNormal(1))"
    return f"LKJCholeskyCov(eta=2, sd_dist={_format_sd_dist(specs)})"


def _component_dist(spec: Any, var: str) -> Any:
    params = {}
    for name, value in spec.parameters.items():
        if hasattr(value, "create_variable"):
            params[name] = value.create_variable(f"{var}_{name}")
        else:
            params[name] = value
    return spec.pymc_distribution.dist(**params)


def _mixed_sd_dist(block_vars: list[str], specs: list[Any]) -> Any:
    import pytensor.tensor as pt

    components = [
        _component_dist(spec, var) for var, spec in zip(block_vars, specs, strict=True)
    ]

    def dist_fn(size: Any) -> Any:
        del size
        return pt.stack(components)

    def logp(value: Any) -> Any:
        return pt.stack([pm.logp(comp, value[i]) for i, comp in enumerate(components)])

    return pm.CustomDist.dist(
        dist=dist_fn,
        logp=logp,
        ndim_supp=1,
        signature="()->(n)",
    )


def residual_sd_dist(block_vars: list[str], priors: dict[str, Any] | None) -> Any:
    """Build the LKJ ``sd_dist`` from each member's ``sigma_{var}`` prior."""
    import pytensor.tensor as pt

    specs = _sigma_priors(block_vars, priors)
    if all(_is_default_residual_scale(spec) for spec in specs):
        return pm.HalfNormal.dist(1.0)
    if not _vectorizable(specs):
        return _mixed_sd_dist(block_vars, specs)

    params: dict[str, Any] = {}
    vector = False
    for name in specs[0].parameters:
        values = [spec.parameters[name] for spec in specs]
        if all(_is_number(value) for value in values):
            if len({float(value) for value in values}) == 1:
                params[name] = float(values[0])
            else:
                params[name] = np.asarray(values, dtype=float)
                vector = True
        else:
            params[name] = pt.stack([
                value.create_variable(f"{var}_{name}")
                for var, value in zip(block_vars, values, strict=True)
            ])
            vector = True
    dist = specs[0].pymc_distribution
    if vector:
        return dist.dist(**params, shape=len(block_vars))
    return dist.dist(**params)


class LKJResidual:
    """Full LKJ Cholesky covariance for residual blocks.

    Parameterizes the residual covariance as an LKJ Cholesky factor.
    Marginal standard deviations follow each member's ``sigma_{var}``
    prior (default ``HalfNormal(1)``). Emits ``{block}_obs`` (observed
    MvNormal) when ``observed=True``, or a free ``{block}_joint`` RV
    when ``observed=False``.
    """

    def emit(
        self,
        block_vars: list[str],
        mu_dict: dict[str, Any],
        data_dict: dict[str, np.ndarray],
        priors: dict[str, Any] | None,
        *,
        observed: bool = True,
    ) -> Any:
        """Emit LKJ Cholesky covariance + MvNormal (observed or free)."""
        k = len(block_vars)
        block_name = "_".join(block_vars)

        mu_stacked = pm.math.stack([mu_dict[v] for v in block_vars], axis=1)

        sd_dist = residual_sd_dist(block_vars, priors)
        chol, _, _ = pm.LKJCholeskyCov(
            f"chol_{block_name}",
            n=k,
            eta=2.0,
            sd_dist=sd_dist,
            compute_corr=True,
        )
        if observed:
            y_stacked = np.column_stack([data_dict[v] for v in block_vars])
            return pm.MvNormal(
                f"{block_name}_obs", mu=mu_stacked, chol=chol, observed=y_stacked
            )
        return pm.MvNormal(f"{block_name}_joint", mu=mu_stacked, chol=chol)

    def prior_keys(self, block_vars: list[str]) -> list[str]:
        """Return LKJ-specific prior keys for introspection."""
        block_name = "_".join(block_vars)
        return [f"chol_{block_name}"]
