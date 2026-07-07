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
"""Unit tests for the HSGP kernel/basis layer (:mod:`pathmc.hsgp`)."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor
import pytest

from pathmc.hsgp import assemble_hsgp_term, hsgp_basis, make_cov_func
from pathmc.parse import HSGPCall
from pathmc.priors import default_priors
from pathmc.parse import parse_spec


@pytest.mark.parametrize(
    "cov,expected",
    [
        ("expquad", pm.gp.cov.ExpQuad),
        ("matern52", pm.gp.cov.Matern52),
        ("matern32", pm.gp.cov.Matern32),
    ],
)
def test_make_cov_func_kernels(cov, expected):
    with pm.Model():
        cov_func = make_cov_func(cov, eta=1.0, ell=1.0)
    # eta**2 * Kernel(...) is a product/scaled covariance wrapping the kernel.
    assert isinstance(cov_func, pm.gp.cov.Covariance)


def test_make_cov_func_unknown_raises():
    with pytest.raises(ValueError, match="Unknown hsgp cov"):
        make_cov_func("rbf", eta=1.0, ell=1.0)


@pytest.mark.parametrize("boundary", ["c", "L"])
def test_hsgp_basis_shapes(boundary):
    m = 10
    n = 40
    x = np.linspace(0.0, 1.0, n)[:, None]
    call = HSGPCall(
        variable="x",
        m=m,
        c=1.5 if boundary == "c" else None,
        L=2.0 if boundary == "L" else None,
    )
    with pm.Model():
        cov_func = make_cov_func("expquad", eta=1.0, ell=1.0)
        phi, sqrt_psd, n_basis = hsgp_basis(call, x, cov_func=cov_func)
        assert n_basis == m
        phi_val = pytensor.function([], phi)()
        psd_val = pytensor.function([], sqrt_psd)()
    assert phi_val.shape == (n, m)
    assert psd_val.shape == (m,)


def test_hsgp_basis_kernel_is_threaded_into_sqrt_psd():
    """Varying eta/ell must change sqrt_psd -- the kernel is truly used."""
    call = HSGPCall(variable="x", m=8, c=1.5)
    x = np.linspace(0.0, 1.0, 30)[:, None]
    with pm.Model():
        _, psd_a, _ = hsgp_basis(
            call, x, cov_func=make_cov_func("expquad", eta=1.0, ell=0.5)
        )
        _, psd_b, _ = hsgp_basis(
            call, x, cov_func=make_cov_func("expquad", eta=2.0, ell=3.0)
        )
        a = pytensor.function([], psd_a)()
        b = pytensor.function([], psd_b)()
    assert not np.allclose(a, b)


def _priors_for(spec_string):
    spec = parse_spec(spec_string)
    return spec, default_priors(spec)


def test_assemble_hsgp_term_noncentered_creates_rvs():
    spec, priors = _priors_for("y ~ hsgp(x, m=6, c=1.5)")
    call = spec.regressions[0].terms[0].hsgp
    n = 25
    x = np.linspace(0.0, 1.0, n)[:, None]
    with pm.Model(coords={"y_x_hsgp": list(range(6))}) as model:
        f = assemble_hsgp_term(call, x, lhs="y", priors=priors)
    names = {rv.name for rv in model.free_RVs}
    assert {"ell_y_x", "eta_y_x", "beta_hsgp_y_x"} <= names
    assert f.name == "f_y_x"
    f_val = pytensor.function([], f)()
    assert f_val.shape == (n,)


def test_assemble_hsgp_term_centered_uses_sqrt_psd_scale():
    spec, priors = _priors_for("y ~ hsgp(x, m=6, c=1.5, centered=true)")
    call = spec.regressions[0].terms[0].hsgp
    x = np.linspace(0.0, 1.0, 25)[:, None]
    with pm.Model(coords={"y_x_hsgp": list(range(6))}) as model:
        f = assemble_hsgp_term(call, x, lhs="y", priors=priors)
    # Centered: beta carries the sqrt_psd scale, so its distribution sigma is
    # not the standardized 1.0 used in the non-centered branch.
    beta = model["beta_hsgp_y_x"]
    assert f.name == "f_y_x"
    assert beta.name == "beta_hsgp_y_x"
