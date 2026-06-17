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
    ) -> None:
        """Create covariance priors and emit the joint likelihood.

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


class LKJResidual:
    """Full LKJ Cholesky covariance for residual blocks.

    Parameterizes the residual covariance as an LKJ Cholesky factor
    with half-normal marginal standard deviations. Emits the block
    as a single observed MvNormal.
    """

    def emit(
        self,
        block_vars: list[str],
        mu_dict: dict[str, Any],
        data_dict: dict[str, np.ndarray],
        priors: dict[str, Any] | None,
    ) -> None:
        """Emit LKJ Cholesky covariance + MvNormal likelihood."""
        k = len(block_vars)
        block_name = "_".join(block_vars)

        mu_stacked = pm.math.stack([mu_dict[v] for v in block_vars], axis=1)
        y_stacked = np.column_stack([data_dict[v] for v in block_vars])

        chol, _, _ = pm.LKJCholeskyCov(
            f"chol_{block_name}",
            n=k,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0),
            compute_corr=True,
        )
        pm.MvNormal(f"{block_name}_obs", mu=mu_stacked, chol=chol, observed=y_stacked)

    def prior_keys(self, block_vars: list[str]) -> list[str]:
        """Return LKJ-specific prior keys for introspection."""
        block_name = "_".join(block_vars)
        return [f"chol_{block_name}"]
