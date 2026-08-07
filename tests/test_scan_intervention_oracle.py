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
"""Analytic oracle for #327 item 2: do() through the temporal scan.

The scan-compiled panel model exists "so pm.do() handles interventions
natively" (compile.py docstring). But the internal recursion (lag() /
contemporaneous references inside ``step_fn``) is resolved from *Python-level*
``new_endo`` / ``prev_endo`` dicts built fresh at each scan step -- never from
the outer free RV / Deterministic node named ``var`` that ``pm.do()`` graph
surgery operates on. Before the fix in this PR, intervening on an *endogenous*
variable (one used as ``lag(var)`` or as a contemporaneous predictor by
another equation) was applied only cosmetically to the reported value at the
intervened timestep; every other equation and every future timestep's lag()
term silently kept using the model's *natural*, un-intervened dynamics.

These tests pin a linear-Gaussian temporal SCM with known coefficients (no
MCMC -- the fitted idata's posterior is overwritten with fixed values, and
``sigma_*`` is pinned to 0 so the recursion is exactly deterministic) and
compare ``do()`` output against the closed-form interventional mean: a
product of path coefficients through time.

Model::

    y_t = beta_x * x_t                  (no lag; contemporaneous driver)
    z_t = beta_lag * y_{t-1}            (endogenous lag: reads y's carry)

``y_0`` (the pre-sample carry, i.e. the value ``lag(y)`` resolves to at the
first observed timestep) is baked in at compile time from the *training*
data's first row and is not touched by ``do()`` -- there is no "period 0" to
intervene on. Every later timestep's ``lag(y)`` must see the *intervened*
trajectory, not the natural one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import pathmc

pytestmark = [
    pytest.mark.parametrize("kind", ["mean", "predictive"]),
    # These intervention values are deliberately outside the observed data
    # range -- that is the point of an interventional oracle probing values
    # the natural data-generating process never produced.
    pytest.mark.filterwarnings("ignore:Intervention value.*outside the observed"),
]


BETA_X = 2.0
BETA_LAG = 3.0
N_TIMES = 6


def _panel_df() -> pd.DataFrame:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = BETA_X * x
    return pd.DataFrame({
        "u": ["u1"] * N_TIMES,
        "t": np.arange(1, N_TIMES + 1),
        "x": x,
        "y": y,
        "z": np.zeros(N_TIMES),
    })


@pytest.fixture(scope="module")
def lag_base_model(mock_pymc_sample_module):
    """``y ~ x`` and ``z ~ lag(y)``, coefficients pinned, sigma pinned to 0.

    ``z`` never appears as its own regressor -- the only route from an
    intervention on ``y`` to ``z`` is the scan's internal lag() carry, which
    is exactly the edge #327 item 2 suspected was severed only cosmetically.
    """
    df = _panel_df()
    model = pathmc.model(
        "y ~ x\nz ~ lag(y)",
        data=df,
        panel={"unit": "u", "time": "t"},
    )
    model.fit(draws=1, tune=1, chains=1, random_seed=0)

    posterior = model._idata["posterior"].dataset.copy(deep=True)
    posterior["beta_y"].loc[{"y_predictors": "Intercept"}] = 0.0
    posterior["beta_y"].loc[{"y_predictors": "x"}] = BETA_X
    posterior["beta_z"].loc[{"z_predictors": "Intercept"}] = 0.0
    posterior["beta_z"].loc[{"z_predictors": "lag(y)"}] = BETA_LAG
    posterior["sigma_y"] = posterior["sigma_y"] * 0.0
    posterior["sigma_z"] = posterior["sigma_z"] * 0.0
    model._idata["posterior"] = xr.DataTree(posterior)
    return model


def _analytic_z(y_traj: np.ndarray, y0: float) -> np.ndarray:
    """Closed-form z_t = BETA_LAG * y_{t-1} under a hard intervention on y."""
    prev = y0
    out = np.empty(N_TIMES)
    for i, y_t in enumerate(y_traj):
        out[i] = BETA_LAG * prev
        prev = y_t
    return out


class TestInterveneOnEndogenousLagBase:
    """Oracle: intervening on ``y`` (read via ``lag(y)`` by ``z``) severs y's
    structural equation and must propagate through the scan's own carry."""

    def test_intervention_propagates_through_lag(self, lag_base_model, kind):
        y_traj = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        y0 = _panel_df()["y"].to_numpy()[0]  # pre-sample carry, fixed at compile time

        result = lag_base_model.do(set={"y": y_traj}, simulate_over="time", kind=kind)
        z_got = np.asarray(result._ds["z"].values).reshape(N_TIMES)
        z_expected = _analytic_z(y_traj, y0)

        np.testing.assert_allclose(z_got, z_expected, atol=1e-8)

    def test_intervened_var_reports_the_set_value(self, lag_base_model, kind):
        """The intervened var's own reported value is exactly what was set."""
        y_traj = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result = lag_base_model.do(set={"y": y_traj}, simulate_over="time", kind=kind)
        y_got = np.asarray(result._ds["y"].values).reshape(N_TIMES)
        np.testing.assert_allclose(y_got, y_traj, atol=1e-8)

    def test_mid_recursion_intervention_diverges_exactly_one_step_later(
        self, lag_base_model, kind
    ):
        """A trajectory that matches the natural mean for t=1..3 and diverges
        for t=4..6 must leave z untouched through t=4 and diverge starting
        exactly at t=5 (one step behind), not earlier or later."""
        natural_y = BETA_X * _panel_df()["x"].to_numpy()
        y_traj = natural_y.copy()
        y_traj[3:] = np.array([999.0, 999.0, 999.0])  # diverge from t=4 onward
        y0 = _panel_df()["y"].to_numpy()[0]

        result = lag_base_model.do(set={"y": y_traj}, simulate_over="time", kind=kind)
        z_got = np.asarray(result._ds["z"].values).reshape(N_TIMES)
        z_expected = _analytic_z(y_traj, y0)

        np.testing.assert_allclose(z_got, z_expected, atol=1e-8)
        # z_expected itself should match the *natural* recursion through t=4
        # (index 3) and only then diverge -- a sanity check on the oracle.
        natural_z = _analytic_z(natural_y, y0)
        np.testing.assert_allclose(z_expected[:4], natural_z[:4], atol=1e-8)
        assert not np.allclose(z_expected[4:], natural_z[4:])

    def test_no_intervention_matches_natural_recursion(self, lag_base_model, kind):
        """Baseline: with no do() at all, z must equal the natural mean
        recursion through the *un-intervened* structural equations."""
        natural_y = BETA_X * _panel_df()["x"].to_numpy()
        y0 = _panel_df()["y"].to_numpy()[0]
        natural_z = _analytic_z(natural_y, y0)

        result = lag_base_model.do(set=None, simulate_over="time", kind=kind)
        z_got = np.asarray(result._ds["z"].values).reshape(N_TIMES)
        np.testing.assert_allclose(z_got, natural_z, atol=1e-8)


class TestNoStateLeakageAcrossRepeatedCalls:
    """The do()-intervention channel is set on the (reused) generative model
    and must be restored afterwards -- a stale value from a previous call
    must not leak into a later, differently-intervened or un-intervened
    call (#327 item 2: "the carry leaks the pre-intervention value")."""

    def test_second_call_is_independent_of_first(self, lag_base_model, kind):
        y0 = _panel_df()["y"].to_numpy()[0]

        y_traj_a = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result_a = lag_base_model.do(
            set={"y": y_traj_a}, simulate_over="time", kind=kind
        )
        z_a = np.asarray(result_a._ds["z"].values).reshape(N_TIMES)
        np.testing.assert_allclose(z_a, _analytic_z(y_traj_a, y0), atol=1e-8)

        # A completely different intervention, then back to no intervention.
        y_traj_b = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
        result_b = lag_base_model.do(
            set={"y": y_traj_b}, simulate_over="time", kind=kind
        )
        z_b = np.asarray(result_b._ds["z"].values).reshape(N_TIMES)
        np.testing.assert_allclose(z_b, _analytic_z(y_traj_b, y0), atol=1e-8)

        natural_y = BETA_X * _panel_df()["x"].to_numpy()
        result_none = lag_base_model.do(set=None, simulate_over="time", kind=kind)
        z_none = np.asarray(result_none._ds["z"].values).reshape(N_TIMES)
        np.testing.assert_allclose(z_none, _analytic_z(natural_y, y0), atol=1e-8)

    def test_generative_model_intervention_data_restored_to_nan(
        self, lag_base_model, kind
    ):
        """After do() returns, the scan's override channel must be back to
        its all-NaN (no-op) sentinel so fit()/predict() on the same model
        object are unaffected."""
        lag_base_model.do(
            set={"y": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])},
            simulate_over="time",
            kind=kind,
        )
        node = lag_base_model._gen_model["_do_intervene_y"]
        assert np.all(np.isnan(node.get_value()))


class TestPreInterventionCarryDoesNotLeakForward:
    """The pre-sample carry (t=0, used by the first period's lag()) is fixed
    from the *training* data at compile time -- it is not part of the
    intervention array and correctly cannot be overridden by do(). But it
    also must not leak into t=2 onward once the intervention is active."""

    def test_first_period_uses_training_data_carry_not_intervention(
        self, lag_base_model, kind
    ):
        y0 = _panel_df()["y"].to_numpy()[0]
        y_traj = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        result = lag_base_model.do(set={"y": y_traj}, simulate_over="time", kind=kind)
        z_got = np.asarray(result._ds["z"].values).reshape(N_TIMES)
        # z_1 = BETA_LAG * y0 (the pre-sample carry), *not* BETA_LAG * y_traj[0]
        assert abs(z_got[0] - BETA_LAG * y0) < 1e-8
        assert abs(z_got[0] - BETA_LAG * y_traj[0]) > 1e-6

    def test_later_periods_use_intervened_values_not_the_stale_carry(
        self, lag_base_model, kind
    ):
        y0 = _panel_df()["y"].to_numpy()[0]
        y_traj = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        result = lag_base_model.do(set={"y": y_traj}, simulate_over="time", kind=kind)
        z_got = np.asarray(result._ds["z"].values).reshape(N_TIMES)
        # From t=2 onward, z must track the intervened trajectory, not keep
        # replaying the pre-intervention carry y0.
        for i in range(1, N_TIMES):
            assert abs(z_got[i] - BETA_LAG * y0) > 1e-6
            assert abs(z_got[i] - BETA_LAG * y_traj[i - 1]) < 1e-8
