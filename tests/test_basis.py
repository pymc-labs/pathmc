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
from pathmc.exceptions import ParseError
from pathmc.parse import BasisCall, parse_spec


def test_basis_extension_is_available_from_package_root():
    """External basis authors can use the public package entry point."""
    assert pathmc.Basis is Basis
    assert pathmc.register_basis is register_basis
    assert {"Basis", "register_basis"} <= set(pathmc.__all__)


@pytest.mark.parametrize(
    "expression",
    [
        "adstock(fourier(x, n=2, period=4), decay=theta)",
        "adstock(logistic_saturation(fourier(x, n=2, period=4), lam=theta), decay=theta)",
    ],
)
def test_fourier_cannot_be_nested_inside_a_transform(expression: str):
    """A basis owns coefficients and cannot be a transform's scalar input."""
    with pytest.raises(
        ParseError, match="Apply fourier\\(\\.\\.\\.\\) directly"
    ) as exc:
        parse_spec(f"y ~ {expression}")
    assert "fourier(x, n=2, period=4)" in str(exc.value)


class _DataOnlyBasis(Basis):
    """A stateful public extension with no graph builder or dispatch flag."""

    name = "test_data_only_basis"

    def n_basis(self, call: Any) -> int:
        """Return the single centered column."""
        return 1

    def build_data(
        self, x: np.ndarray, call: Any, *, state: Any | None = None
    ) -> tuple[np.ndarray, float]:
        """Center on the fitted mean, then reuse it for new values."""
        center = float(np.mean(x)) if state is None else float(state)
        return (np.asarray(x, dtype=float) - center)[:, None], center


def test_registered_data_only_basis_rejects_transform_nesting():
    """The parser applies the same composition rule to custom bases."""
    pathmc.register_basis(_DataOnlyBasis())
    expression = "adstock(test_data_only_basis(x), decay=theta)"
    with pytest.raises(ParseError, match="Apply test_data_only_basis") as exc:
        parse_spec(f"y ~ {expression}")
    assert expression in str(exc.value)


def test_data_only_basis_rejects_endogenous_input_with_a_directed_error():
    """Data-only extensions explain why latent inputs require graph code."""
    pathmc.register_basis(_DataOnlyBasis())
    data = pd.DataFrame({"x": np.arange(8.0), "m": np.arange(8.0), "y": np.arange(8.0)})
    with pytest.raises(NotImplementedError, match="implement build_graph") as exc:
        pathmc.model("m ~ x\ny ~ test_data_only_basis(m)", data=data)
    message = str(exc.value)
    assert "input 'm'" in message
    assert "'y' equation" in message
    assert "use an exogenous input" in message


@pytest.mark.slow
def test_data_only_basis_compiles_and_replays_state_for_predict_and_do():
    """The advertised data contract works without graph code or a flag."""
    pathmc.register_basis(_DataOnlyBasis())
    data = pd.DataFrame({"x": np.arange(12.0), "y": np.arange(12.0)})
    model = pathmc.model("y ~ test_data_only_basis(x)", data=data)
    gm = model._gen_model
    binding = gm._pathmc_data_bases["basis_y_test_data_only_basis_x"]
    assert binding.state == 5.5
    np.testing.assert_allclose(
        replay_data_bases(gm._pathmc_data_bases, {"x": np.arange(20.0, 32.0)})[
            "basis_y_test_data_only_basis_x"
        ][:, 0],
        np.arange(14.5, 26.5),
    )

    model.fit(draws=50, tune=50, chains=1, cores=1, progressbar=False, random_seed=1)
    with model._pymc_model:
        pm.set_data({"x": np.arange(20.0, 32.0)})
    model.predict(progressbar=False)
    np.testing.assert_allclose(
        model._pymc_model["basis_y_test_data_only_basis_x"].get_value()[:, 0],
        np.arange(14.5, 26.5),
    )
    baseline = model.do(set={"x": 0.0}, kind="mean")
    intervened = model.do(set={"x": 20.0}, kind="mean")
    assert float(intervened.mean("y")) - float(baseline.mean("y")) > 10.0


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


def test_fourier_period_uses_declared_units_with_channel_scaling():
    """Fourier inputs stay raw, including when also used as a regressor."""
    week = np.array([0.0, 13.0, 26.0, 39.0, 52.0])
    data = pd.DataFrame({"week": week, "y": np.arange(1.0, 6.0)})

    with pytest.warns(UserWarning, match="remain in their declared units"):
        model = pathmc.model(
            "y ~ week + fourier(week, n=1, period=52)",
            data=data,
            scaling=pathmc.Scaling(channel={"method": "max"}),
        )

    raw_angles = 2 * np.pi * week / 52
    expected = np.column_stack((np.sin(raw_angles), np.cos(raw_angles)))
    np.testing.assert_array_equal(model._data["week"].to_numpy(), week)
    np.testing.assert_allclose(
        model._gen_model["basis_y_fourier_week"].get_value(), expected, atol=1e-12
    )


def test_fourier_period_ignores_prefitted_input_scaling():
    """Reused factors cannot change a Fourier input's declared period."""
    week = np.array([0.0, 13.0, 26.0, 39.0, 52.0])
    data = pd.DataFrame({"week": week, "y": np.arange(1.0, 6.0)})
    factors = pathmc.ScalingFactors(factors={"week": ((), {(): 52.0})})

    with pytest.warns(UserWarning, match="remain in their declared units"):
        model = pathmc.model(
            "y ~ fourier(week, n=1, period=52)", data=data, scaling=factors
        )

    raw_angles = 2 * np.pi * week / 52
    expected = np.column_stack((np.sin(raw_angles), np.cos(raw_angles)))
    np.testing.assert_array_equal(model._data["week"].to_numpy(), week)
    np.testing.assert_allclose(
        model._gen_model["basis_y_fourier_week"].get_value(), expected, atol=1e-12
    )


def test_fourier_input_skips_and_warns_about_target_scaling():
    """Endogenous Fourier inputs remain raw and announce skipped scaling."""
    x = np.arange(8.0)
    data = pd.DataFrame({"x": x, "m": 2 * x + 1, "y": x})

    with pytest.warns(UserWarning, match="'m'"):
        model = pathmc.model(
            "m ~ x\ny ~ fourier(m, n=1, period=52)",
            data=data,
            scaling=pathmc.Scaling(target={"method": "max"}, channel={"method": "max"}),
        )

    np.testing.assert_array_equal(model._data["m"].to_numpy(), data["m"])
    np.testing.assert_allclose(model._data["x"].to_numpy(), data["x"] / data["x"].max())


def test_fourier_input_warns_when_fixed_scaling_is_skipped():
    """Explicit fixed scaling on a Fourier input is never silently ignored."""
    data = pd.DataFrame({"week": np.arange(5.0), "y": np.arange(5.0)})

    with pytest.warns(UserWarning, match="'week'"):
        model = pathmc.model(
            "y ~ fourier(week, n=1, period=52)",
            data=data,
            scaling=pathmc.Scaling(channel={"method": "fixed", "value": 52.0}),
        )

    np.testing.assert_array_equal(model._data["week"].to_numpy(), data["week"])


def test_simulate_preserves_fourier_period_under_channel_scaling():
    """Simulation uses the same declared-unit Fourier inputs as model()."""
    week = np.array([0.0, 13.0, 26.0, 39.0, 52.0])
    data = pd.DataFrame({"week": week, "y": np.zeros_like(week)})
    params = {
        "beta_y": np.array([0.0]),
        "beta_fourier_y_week": np.array([1.0, 0.0]),
        "sigma_y": 1e-6,
    }

    with pytest.warns(UserWarning, match="'week'"):
        simulated = pathmc.simulate(
            "y ~ fourier(week, n=1, period=52)",
            data=data,
            params=params,
            scaling=pathmc.Scaling(channel={"method": "max"}),
            random_seed=4,
        )

    expected = np.sin(2 * np.pi * week / 52)
    np.testing.assert_allclose(simulated["y"].to_numpy(), expected, atol=2e-5)


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


def test_multiple_fourier_calls_on_one_input_get_independent_data_bindings():
    """Distinct Fourier calls must not overwrite each other's frozen data state."""
    data = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0)})
    model = pathmc.model(
        "y ~ fourier(x, n=2, period=4) + fourier(x, n=3, period=7)", data=data
    )
    gm = model._gen_model

    assert {"beta_fourier_y_x_1", "beta_fourier_y_x_2"} <= set(gm.named_vars)
    assert {"basis_y_fourier_x_1", "basis_y_fourier_x_2"} == set(gm._pathmc_data_bases)
    updates = replay_data_bases(gm._pathmc_data_bases, {"x": np.arange(8.0) + 10})
    assert updates["basis_y_fourier_x_1"].shape == (8, 4)
    assert updates["basis_y_fourier_x_2"].shape == (8, 6)


def test_mixed_basis_calls_on_one_input_get_independent_contributions():
    """HSGP and Fourier on one input must not claim the same deterministic name."""
    data = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0)})
    model = pathmc.model(
        "y ~ hsgp(x, m=5, c=1.5) + fourier(x, n=2, period=4)", data=data
    )
    gm = model._gen_model

    assert {"f_y_x_1", "f_y_x_2"} <= set(gm.named_vars)
    assert {"beta_hsgp_y_x_1", "beta_fourier_y_x_2"} <= set(gm.named_vars)


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

    assert gm._pathmc_basis_states[("y", "test_centered_data_basis", "x", None)] == 1.0
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
