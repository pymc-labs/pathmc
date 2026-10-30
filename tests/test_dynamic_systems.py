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
"""Lagged products and ``d(x)`` rate equations (spike for #639).

Oracles live outside pathmc: a NumPy recurrence for the discrete map and
``scipy.integrate.solve_ivp`` for the rate equations.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

import pathmc
from pathmc.exceptions import ParseError
from pathmc.graph import build_graph
from pathmc.parse import parse_spec

PANEL = {"unit": "unit", "time": "time"}

LV_RATES = """
d(prey, substeps={k}) ~ 0 + alpha*prey + beta*prey:predator
d(predator, substeps={k}) ~ 0 + gamma*predator + delta*prey:predator
hare ~ 0 + 1*prey
lynx ~ 0 + 1*predator
"""
ALPHA, BETA, GAMMA, DELTA = 0.55, 0.028, 0.80, 0.024
X0, Y0 = 33.9, 5.9


def _frame(n_times: int, units: tuple[str, ...] = ("a",)) -> pd.DataFrame:
    return pd.DataFrame({
        "unit": np.repeat(units, n_times),
        "time": np.tile(np.arange(n_times), len(units)),
    })


def _lv_ode(x0: float, y0: float, n_times: int) -> np.ndarray:
    """LV states at t = 0..n_times-1 starting from the pre-sample state at t=-1."""
    sol = solve_ivp(
        lambda _t, z: [
            ALPHA * z[0] - BETA * z[0] * z[1],
            DELTA * z[0] * z[1] - GAMMA * z[1],
        ],
        (-1, n_times - 1),
        [x0, y0],
        t_eval=np.arange(n_times),
        rtol=1e-11,
        atol=1e-11,
    )
    return sol.y.T


def _lv_params(x0: list[float], y0: list[float]) -> dict:
    return {
        "beta_prey": [ALPHA, -BETA],
        "beta_predator": [-GAMMA, DELTA],
        "init_prey": x0,
        "init_predator": y0,
        "sigma_hare": 1e-12,
        "sigma_lynx": 1e-12,
    }


class TestParse:
    def test_lagged_product_is_one_interaction_term(self) -> None:
        reg = parse_spec("y ~ 0 + k*lag( x ):lag(z)").regressions[0]
        (term,) = reg.terms
        assert term.interaction_of == ("lag(x)", "lag(z)")
        assert term.variable == "lag(x):lag(z)"
        assert term.label == "k"

    def test_lag_part_can_mix_with_current_value(self) -> None:
        (term,) = parse_spec("y ~ lag(x):z").regressions[0].terms
        assert term.interaction_of == ("lag(x)", "z")

    def test_rate_equation_lowers_state_references_to_lag(self) -> None:
        spec = parse_spec(LV_RATES.format(k=3))
        prey = spec.regressions[0]
        assert prey.lhs == "prey"
        assert prey.rate
        assert prey.substeps == 3
        assert [t.variable for t in prey.terms] == ["lag(prey)", "lag(prey):lag(predator)"]
        assert prey.terms[0].lag_of == "prey"
        hare = spec.regressions[2]
        assert not hare.rate
        assert hare.terms[0].variable == "prey"

    def test_rate_equation_keeps_exogenous_input_contemporaneous(self) -> None:
        (reg,) = parse_spec("d(x) ~ 0 + k*x + u + x:u").regressions
        assert [t.variable for t in reg.terms] == ["lag(x)", "u", "lag(x):u"]
        assert reg.substeps == 10

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("d(x) ~ 0 + lag(x)", "lag\\(\\) is not allowed"),
            ("d(x, substeps=2) ~ 0 + x\nd(y, substeps=3) ~ 0 + y", "same substeps"),
            ("d(x, steps=2) ~ 0 + x", "Malformed rate equation"),
            ("d(x, substeps=0) ~ 0 + x", "at least 1"),
            ("d(x) ~ 0 + adstock(u, decay=a)", "not supported in the rate equation"),
        ],
    )
    def test_rejects_unsupported_rate_syntax(self, spec: str, match: str) -> None:
        with pytest.raises(ParseError, match=match):
            parse_spec(spec)


class TestGraph:
    def test_lagged_product_adds_no_node_and_no_cycle(self) -> None:
        spec = parse_spec(
            "prey ~ 0 + lag(prey) + lag(prey):lag(predator)\n"
            "predator ~ 0 + lag(predator) + lag(prey):lag(predator)"
        )
        info = build_graph(spec)
        assert set(info._dag.nodes) == {"prey", "predator", "lag(prey)", "lag(predator)"}
        assert set(info.temporal_edges) == {
            ("prey", "lag(prey)"),
            ("predator", "lag(predator)"),
        }

    def test_rate_states_do_not_form_a_contemporaneous_cycle(self) -> None:
        info = build_graph(parse_spec(LV_RATES.format(k=2)), latent={"prey", "predator"})
        order = info.topological_order
        assert order.index("prey") < order.index("hare")
        assert order.index("predator") < order.index("lynx")

    def test_rate_equation_rejects_same_step_outcome(self) -> None:
        df = _frame(4).assign(y=1.0, m=0.0, u=0.0)
        with pytest.raises(NotImplementedError, match="exogenous inputs only"):
            pathmc.model("d(x) ~ 0 + x + m\nm ~ u\ny ~ 0 + 1*x", data=df, panel=PANEL, latent=["x"])


def test_state_read_only_inside_a_product_gets_an_init_prior() -> None:
    template = pathmc.simulate_params_template(
        "prey ~ 0 + r*lag(prey) + a*lag(prey):lag(predator)\n"
        "predator ~ 0 + c*lag(prey):lag(predator)",
        _frame(5),
        panel=PANEL,
        latent=["prey", "predator"],
    )
    assert template["init_predator"]["shape"] == (1,)


class TestSimulateOracle:
    def test_lagged_product_matches_numpy_recurrence(self) -> None:
        spec = (
            "prey ~ 0 + r*lag(prey) + attack*lag(prey):lag(predator)\n"
            "predator ~ 0 + s*lag(predator) + conv*lag(prey):lag(predator)"
        )
        n_times = 40
        sim = pathmc.simulate(
            spec,
            _frame(n_times),
            params={
                "beta_prey": [1.1, -0.02],
                "beta_predator": [0.9, 0.004],
                "sigma_prey": 1e-12,
                "sigma_predator": 1e-12,
                "init_prey": [20.0],
                "init_predator": [8.0],
            },
            panel=PANEL,
            families={"prey": "latent_normal", "predator": "latent_normal"},
            latent=["prey", "predator"],
            random_seed=1,
        )
        x, y, expected = 20.0, 8.0, []
        for _ in range(n_times):
            x, y = 1.1 * x - 0.02 * x * y, 0.9 * y + 0.004 * x * y
            expected.append((x, y))
        np.testing.assert_allclose(sim[["prey", "predator"]].to_numpy(), expected, rtol=1e-9)
        np.testing.assert_allclose(sim.loc[0, ["prey", "predator"]], [18.8, 7.84])

    @pytest.mark.parametrize(("substeps", "rtol"), [(1, 3e-2), (2, 2e-3), (4, 1e-4)])
    def test_rate_equations_match_scipy(self, substeps: int, rtol: float) -> None:
        n_times = 21
        sim = pathmc.simulate(
            LV_RATES.format(k=substeps),
            _frame(n_times),
            params=_lv_params([X0], [Y0]),
            panel=PANEL,
            latent=["prey", "predator"],
            random_seed=1,
        )
        np.testing.assert_allclose(
            sim[["hare", "lynx"]].to_numpy(), _lv_ode(X0, Y0, n_times), rtol=rtol
        )

    def test_units_integrate_independently(self) -> None:
        n_times = 15
        sim = pathmc.simulate(
            LV_RATES.format(k=4),
            _frame(n_times, ("a", "b")),
            params=_lv_params([X0, 20.0], [Y0, 9.0]),
            panel=PANEL,
            latent=["prey", "predator"],
            random_seed=1,
        )
        for unit, (x0, y0) in {"a": (X0, Y0), "b": (20.0, 9.0)}.items():
            got = sim.loc[sim["unit"] == unit, ["hare", "lynx"]].to_numpy()
            np.testing.assert_allclose(got, _lv_ode(x0, y0, n_times), rtol=1e-4)

    def test_exogenous_input_is_held_over_the_step(self) -> None:
        n_times = 12
        u = np.where(np.arange(n_times) % 4 < 2, 1.0, 0.0)
        df = _frame(n_times).assign(u=u)
        sim = pathmc.simulate(
            "d(x, substeps=8) ~ 0 + k*x + u\ny ~ 0 + 1*x",
            df,
            params={"beta_x": [-0.5, 2.0], "init_x": [1.0], "sigma_y": 1e-12},
            panel=PANEL,
            latent=["x"],
            random_seed=1,
        )
        state, expected = 1.0, []
        for t in range(n_times):
            # Input u[t] applies on (t-1, t]; the exact step is closed form.
            target = 2.0 * u[t] / 0.5
            state = target + (state - target) * np.exp(-0.5)
            expected.append(state)
        np.testing.assert_allclose(sim["y"].to_numpy(), expected, rtol=1e-6)


@pytest.fixture
def lv_model() -> pathmc.PathModel:
    n_times = 21
    df = _frame(n_times).assign(hare=1.0, lynx=1.0)
    model = pathmc.model(
        LV_RATES.format(k=4), data=df, panel=PANEL, latent=["prey", "predator"]
    )
    one = np.ones((1, 1))
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_prey": np.array([[[ALPHA, -BETA]]]),
                "beta_predator": np.array([[[-GAMMA, DELTA]]]),
                "init_prey": np.array([[[X0]]]),
                "init_predator": np.array([[[Y0]]]),
                "sigma_hare": one,
                "sigma_lynx": one,
            }
        },
        coords={
            "prey_predictors": ["lag(prey)", "lag(prey):lag(predator)"],
            "predator_predictors": ["lag(predator)", "lag(prey):lag(predator)"],
            "unit": ["a"],
        },
        dims={
            "beta_prey": ["prey_predictors"],
            "beta_predator": ["predator_predictors"],
            "init_prey": ["unit"],
            "init_predator": ["unit"],
        },
    )
    return model


class TestDo:
    def test_free_run_matches_scipy(self, lv_model: pathmc.PathModel) -> None:
        result = lv_model.do(simulate_over="time", kind="mean")
        got = np.column_stack([
            np.asarray(result.by_time("prey")).mean(axis=1),
            np.asarray(result.by_time("predator")).mean(axis=1),
        ])
        np.testing.assert_allclose(got, _lv_ode(X0, Y0, 21), rtol=1e-4)

    def test_released_clamp_reports_structural_path(
        self, lv_model: pathmc.PathModel
    ) -> None:
        window = slice(5, 10)
        cull = np.full(21, np.nan)
        cull[window] = 2.0
        base = lv_model.do(simulate_over="time", kind="mean")
        pulse = lv_model.do(set={"predator": cull}, simulate_over="time", kind="mean")

        prey_base = np.asarray(base.by_time("prey")).mean(axis=1)
        prey_pulse = np.asarray(pulse.by_time("prey")).mean(axis=1)
        predator = np.asarray(pulse.by_time("predator")).mean(axis=1)

        np.testing.assert_allclose(prey_pulse[:5], prey_base[:5], rtol=1e-12)
        assert np.all(np.abs(prey_pulse[5:] - prey_base[5:]) > 1e-3)
        assert not np.isnan(predator).any()
        np.testing.assert_allclose(predator[window], 2.0)

        # Oracle: integrate to t=4, hold the predator at 2 over (4, 9], release.
        states = _lv_ode(X0, Y0, 5)
        x, y = states[-1]
        held = solve_ivp(
            lambda _t, z: [ALPHA * z[0] - BETA * z[0] * 2.0],
            (4, 9),
            [x],
            t_eval=np.arange(5, 10),
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(prey_pulse[window], held.y[0], rtol=1e-4)
        after = solve_ivp(
            lambda _t, z: [
                ALPHA * z[0] - BETA * z[0] * z[1],
                DELTA * z[0] * z[1] - GAMMA * z[1],
            ],
            (9, 20),
            [held.y[0][-1], 2.0],
            t_eval=np.arange(10, 21),
            rtol=1e-11,
            atol=1e-11,
        )
        np.testing.assert_allclose(prey_pulse[10:], after.y[0], rtol=1e-3)
        np.testing.assert_allclose(predator[10:], after.y[1], rtol=1e-3)
