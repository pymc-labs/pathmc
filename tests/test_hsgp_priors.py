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
"""Default/override prior behavior for HSGP hyperpriors (fast)."""

from __future__ import annotations

import pytest

from pathmc import Prior
from pathmc.parse import parse_spec
from pathmc.priors import default_priors, merge_priors


def test_default_priors_include_hsgp_hyperpriors():
    spec = parse_spec("Y ~ hsgp(x, m=12, c=1.5)")
    priors = default_priors(spec)

    assert priors["ell_Y_x"].distribution == "InverseGamma"
    assert priors["eta_Y_x"].distribution == "HalfNormal"
    beta = priors["beta_hsgp_Y_x"]
    assert beta.distribution == "Normal"
    assert beta.dims == ("Y_x_hsgp",)


def test_hsgp_prior_overrides_round_trip():
    spec = parse_spec("Y ~ hsgp(x, m=12, c=1.5)")
    defaults = default_priors(spec)
    overrides = {
        "ell_Y_x": Prior("InverseGamma", alpha=5, beta=2),
        "eta_Y_x": Prior("HalfNormal", sigma=2),
    }
    merged = merge_priors(defaults, overrides)
    assert merged["ell_Y_x"].parameters["alpha"] == 5
    assert merged["eta_Y_x"].parameters["sigma"] == 2


def test_unknown_hsgp_prior_key_raises():
    spec = parse_spec("Y ~ hsgp(x, m=12, c=1.5)")
    defaults = default_priors(spec)
    with pytest.raises(ValueError, match="Unknown prior key"):
        merge_priors(defaults, {"ell_Y_z": Prior("HalfNormal", sigma=1)})
