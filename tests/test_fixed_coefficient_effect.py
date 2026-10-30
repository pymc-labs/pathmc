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

import warnings

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


def _panel_frame(*columns: str) -> pd.DataFrame:
    rows = []
    for unit in ("a", "b"):
        for time, value in enumerate((0.0, 1.0)):
            row = {"unit": unit, "time": time}
            for column in columns:
                row[column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _pin_panel(model, posterior, coords, dims) -> None:
    model._idata = az.from_dict({"posterior": posterior}, coords=coords, dims=dims)


def test_effect_adds_hierarchical_random_slope_mean() -> None:
    """Population path coefficient is beta + mu_slope, not the flat beta."""
    model = pathmc.model(
        "Y ~ 0 + b*X",
        data=_panel_frame("X", "Y"),
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": True, "slopes": ["X"]},
    )
    unit = list(model.pymc_model.coords["unit"])
    _pin_panel(
        model,
        {
            "beta_Y": np.array([[[4.0], [5.0]]]),
            "slope_Y_X": np.array([[[1.0, 5.0], [0.0, 2.0]]]),
            "mu_slope_Y_X": np.array([[[3.0], [1.0]]]),
        },
        {"unit": unit, "Y_predictors": ["X"]},
        {"beta_Y": ["Y_predictors"], "slope_Y_X": ["unit"]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = model.effect("X -> Y")

    assert result.mean == pytest.approx(6.5)
    assert np.asarray(result.draws).shape == (2,)
    assert np.allclose(result.draws, [7.0, 6.0])


def test_effect_issue_probe_mean_is_population_coefficient() -> None:
    model = pathmc.model(
        "Y ~ 0 + b*X",
        data=_panel_frame("X", "Y"),
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": True, "slopes": ["X"]},
    )
    unit = list(model.pymc_model.coords["unit"])
    _pin_panel(
        model,
        {
            "beta_Y": np.full((1, 1, 1), 4.0),
            "slope_Y_X": np.full((1, 1, 2), 3.0),
            "mu_slope_Y_X": np.full((1, 1), 3.0),
        },
        {"unit": unit, "Y_predictors": ["X"]},
        {"beta_Y": ["Y_predictors"], "slope_Y_X": ["unit"]},
    )

    result = model.effect("X -> Y")
    assert result.mean == pytest.approx(7.0)
    assert np.asarray(result.draws).ndim == 1
    assert np.allclose(result.draws, 7.0)


def test_effect_averages_unit_slopes_when_hierarchical_mean_is_absent() -> None:
    model = pathmc.model(
        "Y ~ 0 + b*X",
        data=_panel_frame("X", "Y"),
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": True, "slopes": ["X"]},
    )
    unit = list(model.pymc_model.coords["unit"])
    _pin_panel(
        model,
        {
            "beta_Y": np.array([[[4.0]]]),
            "slope_Y_X": np.array([[[1.0, 5.0]]]),
        },
        {"unit": unit, "Y_predictors": ["X"]},
        {"beta_Y": ["Y_predictors"], "slope_Y_X": ["unit"]},
    )

    with pytest.warns(UserWarning, match="not a unit-specific effect"):
        result = model.effect("X -> Y")
    assert result.mean == pytest.approx(7.0)
    assert np.asarray(result.draws).shape == (1,)


def test_path_product_adds_population_slope_on_each_edge() -> None:
    """Both edges contribute beta + mu_slope before the product."""
    from pathmc.effects import compute_path_effect
    from pathmc.parse import parse_spec

    spec = parse_spec("M ~ 0 + a*X\nY ~ 0 + b*M")
    idata = az.from_dict(
        {
            "posterior": {
                "beta_M": np.array([[[2.0]]]),
                "beta_Y": np.array([[[3.0]]]),
                "mu_slope_M_X": np.array([[[1.0]]]),
                "mu_slope_Y_M": np.array([[[0.5]]]),
                "slope_M_X": np.array([[[9.0, -4.0]]]),
                "slope_Y_M": np.array([[[8.0, -1.0]]]),
            }
        },
        coords={"M_predictors": ["X"], "Y_predictors": ["M"], "unit": ["a", "b"]},
        dims={
            "beta_M": ["M_predictors"],
            "beta_Y": ["Y_predictors"],
            "slope_M_X": ["unit"],
            "slope_Y_M": ["unit"],
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        direct = compute_path_effect("X -> M", spec, idata)
        path = compute_path_effect("X -> M -> Y", spec, idata)
    assert direct.mean == pytest.approx(3.0)
    assert path.mean == pytest.approx(10.5)
    assert np.asarray(path.draws).shape == (1,)


def test_fixed_coefficient_includes_population_random_slope() -> None:
    model = pathmc.model(
        "Y ~ 0 + 4*X",
        data=_panel_frame("X", "Y"),
        panel={"unit": "unit", "time": "time"},
        pooling={"intercept": False, "slopes": ["X"]},
    )
    unit = list(model.pymc_model.coords["unit"])
    _pin_panel(
        model,
        {"mu_slope_Y_X": np.array([[[3.0]]]), "slope_Y_X": np.full((1, 1, 2), 9.0)},
        {"unit": unit},
        {"slope_Y_X": ["unit"]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = model.effect("X -> Y")
    assert result.mean == pytest.approx(7.0)
    assert np.allclose(result.draws, 7.0)


def test_unlabeled_edge_adds_only_its_own_random_slope() -> None:
    model = pathmc.model(
        "Y ~ 0 + X + Z",
        data=_panel_frame("X", "Z", "Y"),
        panel={"unit": "unit", "time": "time"},
        pooling={"slopes": ["X"]},
    )
    unit = list(model.pymc_model.coords["unit"])
    preds = [str(name) for name in model.pymc_model.coords["Y_predictors"]]
    beta = np.zeros((1, 2, len(preds)))
    beta[..., preds.index("X")] = 4.0
    beta[..., preds.index("Z")] = 1.5
    _pin_panel(
        model,
        {
            "beta_Y": beta,
            "mu_slope_Y_X": np.array([[[3.0], [1.0]]]),
            "slope_Y_X": np.array([[[3.0, 3.0], [1.0, 1.0]]]),
        },
        {"unit": unit, "Y_predictors": preds},
        {"beta_Y": ["Y_predictors"], "slope_Y_X": ["unit"]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        x_effect = model.effect("X -> Y")
        z_effect = model.effect("Z -> Y")
    assert np.allclose(x_effect.draws, [7.0, 5.0])
    assert np.allclose(z_effect.draws, [1.5, 1.5])
