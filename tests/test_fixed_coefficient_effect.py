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
"""Path products must include fixed regression coefficients."""

import arviz as az
import numpy as np
import pandas as pd
import pytest

import pathmc


def _pin(model, mapping: dict[str, dict[str, float]]) -> None:
    posterior: dict[str, np.ndarray] = {}
    coords: dict[str, list[str]] = {}
    dims: dict[str, list[str]] = {}
    for lhs, coeffs in mapping.items():
        coord = f"{lhs}_predictors"
        order = [str(v) for v in model.pymc_model.coords[coord]]
        posterior[f"beta_{lhs}"] = np.tile([coeffs[n] for n in order], (1, 4, 1))
        coords[coord] = order
        dims[f"beta_{lhs}"] = [coord]
    model._idata = az.from_dict({"posterior": posterior}, coords=coords, dims=dims)


def test_effect_includes_fixed_measurement_loading() -> None:
    """A fixed loading is the regression term; the product includes it."""
    df = pd.DataFrame({
        "X": np.linspace(0, 2, 10),
        "Y": np.linspace(0, 3, 10),
        "M_obs": np.linspace(0, 1, 10),
    })
    model = pathmc.model(
        "M ~ a*X\nY ~ b*M + c*X\nM_obs ~ 0 + 1*M",
        data=df,
        latent=["M"],
    )
    _pin(
        model,
        {
            "M": {"Intercept": 0.0, "X": 0.4},
            "Y": {"Intercept": 0.0, "M": 2.5, "X": 0.1},
        },
    )

    assert model.effect("X -> M -> Y").mean == pytest.approx(1.0)
    loading = model.effect("M -> M_obs")
    assert loading.mean == pytest.approx(1.0)
    assert np.allclose(loading.draws, 1.0)


def test_effect_multiplies_nonzero_fixed_coefficient() -> None:
    """The failure class is any fixed coefficient, not only a unit loading."""
    df = pd.DataFrame({
        "X": np.linspace(0, 1, 8),
        "M": np.linspace(0, 1, 8),
        "Y": np.linspace(0, 2, 8),
    })
    model = pathmc.model("M ~ 2*X\nY ~ b*M + -1.5*X", data=df)
    _pin(model, {"Y": {"Intercept": 0.0, "M": 0.5}})

    assert model.effect("X -> M").mean == pytest.approx(2.0)
    assert model.effect("X -> Y").mean == pytest.approx(-1.5)
    indirect = model.effect("X -> M -> Y")
    assert indirect.mean == pytest.approx(1.0)
    assert np.allclose(indirect.draws, 1.0)


def test_missing_edge_still_names_the_absent_edge() -> None:
    df = pd.DataFrame({"X": [0.0, 1.0], "Y": [0.0, 1.0]})
    model = pathmc.model("Y ~ 2*X", data=df)
    model._idata = az.from_dict({"posterior": {"sigma_Y": np.ones((1, 2))}})

    with pytest.raises(ValueError, match="No direct edge from 'Z' to 'Y'"):
        model.effect("Z -> Y")
