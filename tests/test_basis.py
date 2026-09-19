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
"""Fast contract tests for generic basis terms and the built-in Fourier basis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc
from pathmc.basis import Basis, get_basis, register_basis, replay_data_bases
from pathmc.parse import BasisCall, parse_spec


def test_fourier_parses_as_a_generic_basis_call():
    """Fourier uses the reusable basis AST rather than a dedicated term field."""
    term = parse_spec("y ~ fourier(week, n=3, period=52)").regressions[0].terms[0]

    assert term.basis == BasisCall(
        name="fourier", variable="week", params={"n": 3, "period": 52.0}
    )
    assert term.hsgp is None


def test_fourier_data_columns_match_the_harmonic_oracle():
    """Numeric data columns have the expected sin/cos order and period."""
    call = BasisCall("fourier", "week", {"n": 2, "period": 4.0})
    columns, state = get_basis("fourier").build_data(np.array([0.0, 1.0]), call)

    np.testing.assert_allclose(
        columns,
        np.array([[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, -1.0]]),
        atol=1e-12,
    )
    assert state is None


def test_fourier_owns_weights_without_misaligning_plain_or_fixed_terms():
    """A basis does not consume a scalar beta slot between ordinary terms."""
    data = pd.DataFrame({
        "x": np.arange(8.0),
        "z": np.arange(8.0) + 1,
        "y": np.arange(8.0),
    })
    model = pathmc.model("y ~ x + fourier(x, n=2, period=4) + 2*z", data=data)
    gm = model._gen_model

    assert list(gm.coords["y_predictors"]) == ["Intercept", "x"]
    assert list(gm.coords["y_x_fourier"]) == [0, 1, 2, 3]
    assert gm["beta_y"].eval().shape == (2,)
    assert gm["beta_fourier_y_x"].eval().shape == (4,)


def test_fourier_graph_basis_accepts_a_latent_mediator_input():
    """Fourier can compose with an endogenous graph value without ``eval()``."""
    data = pd.DataFrame({"x": np.arange(8.0), "m": np.arange(8.0), "y": np.arange(8.0)})
    model = pathmc.model("m ~ x\ny ~ fourier(m, n=2, period=4)", data=data)

    assert "f_y_m" in model._gen_model.named_vars


def test_fourier_contribution_recomputes_when_its_input_data_changes():
    """The graph contract keeps Fourier responsive to an intervention input."""
    data = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0)})
    model = pathmc.model("y ~ fourier(x, n=2, period=4)", data=data)
    gm = model._gen_model
    with gm:
        before = pm.draw(gm["f_y_x"], draws=1, random_seed=1)
        pm.set_data({"x": np.full(len(data), 1.0)})
        pm.set_data(
            replay_data_bases(gm._pathmc_data_bases, {"x": np.full(len(data), 1.0)})
        )
        after = pm.draw(gm["f_y_x"], draws=1, random_seed=1)

    assert not np.allclose(before, after)


class _CenteredDataBasis(Basis):
    """Small stateful basis used to exercise the data/graph hand-off."""

    name = "test_centered_data_basis"
    supports_data_contract = True

    def n_basis(self, call: Any) -> int:
        """Return one centered column."""
        return 1

    def build_data(
        self, x: np.ndarray, call: Any, *, state: Any | None = None
    ) -> tuple[np.ndarray, float]:
        """Fit or replay a mean-centering state."""
        center = float(np.mean(x)) if state is None else float(state)
        return (np.asarray(x, dtype=float) - center)[:, None], center

    def build_graph(
        self,
        x: Any,
        call: Any,
        *,
        lhs: str,
        priors: Any,
        state: Any | None = None,
    ) -> tuple[Any, None]:
        """Prove observed data bases compile without a graph builder."""
        raise AssertionError("Observed data basis unexpectedly used build_graph().")


def test_data_basis_state_is_frozen_and_replayed_under_new_input_data():
    """A data basis must not refit centering state after a new intervention input."""
    register_basis(_CenteredDataBasis())
    data = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]})
    model = pathmc.model("y ~ test_centered_data_basis(x)", data=data)
    gm = model._gen_model

    assert gm._pathmc_basis_states[("y", "test_centered_data_basis", "x")] == 1.0
    with gm:
        pm.set_data({"x": np.array([10.0, 11.0, 12.0])})
        pm.set_data(
            replay_data_bases(
                gm._pathmc_data_bases, {"x": np.array([10.0, 11.0, 12.0])}
            )
        )
        contribution_without_weight = pm.draw(
            gm["f_y_x"] / gm["beta_test_centered_data_basis_y_x"][0],
            draws=1,
            random_seed=1,
        )

    np.testing.assert_allclose(contribution_without_weight, [9.0, 10.0, 11.0])


@pytest.mark.slow
def test_data_basis_state_survives_predict_and_do():
    """The frozen state remains attached to the fitted graph APIs."""
    register_basis(_CenteredDataBasis())
    data = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0)})
    model = pathmc.model("y ~ test_centered_data_basis(x)", data=data)
    model.fit(draws=50, tune=50, chains=1, cores=1, progressbar=False, random_seed=1)

    with model._pymc_model:
        pm.set_data({"x": np.arange(20.0, 32.0)})
    model.predict(progressbar=False)
    with model._pymc_model:
        contribution_without_weight = pm.draw(
            model._pymc_model["f_y_x"]
            / model._pymc_model["beta_test_centered_data_basis_y_x"][0],
            draws=1,
            random_seed=1,
        )
    intervened = model.do(set={"x": 20.0}, kind="mean")

    np.testing.assert_allclose(contribution_without_weight, np.arange(14.5, 26.5))
    assert float(intervened.mean("y")) > float(data["y"].mean())
