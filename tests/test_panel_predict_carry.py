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
"""Observed-carry flag handling in ``predict()`` for panel AR models.

``fit()`` needs the scan to carry observed previous values (the conditional
AR likelihood). ``predict()`` must set the flag it wants explicitly and put
it back afterwards, so the choice never leaks between calls.
"""

import sys

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc
import pathmc.simulate
from pathmc._model import _OBSERVED_CARRY_FLAG

# ``pathmc.simulate`` the attribute is the public ``simulate()`` function, so
# reach the module object explicitly for monkeypatching.
simulate_module = sys.modules["pathmc.simulate"]


@pytest.fixture(scope="module")
def ar_panel_data():
    """3 regions x 25 weeks from ``y ~ 2 + 0.5 * lag(y) + 0.4 * x``."""
    rng = np.random.default_rng(0)
    rows = []
    for region in ["A", "B", "C"]:
        y_prev = 4.0
        for week in range(1, 26):
            x = rng.normal()
            y = 2.0 + 0.5 * y_prev + 0.4 * x + rng.normal(scale=0.3)
            rows.append({"region": region, "week": week, "x": x, "y": y})
            y_prev = y
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_ar_panel(ar_panel_data):
    model = pathmc.model(
        "y ~ x + lag(y)",
        data=ar_panel_data,
        panel={"unit": "region", "time": "week"},
    )
    model.fit(random_seed=42, compute_log_likelihood=False)
    return model


def _flag(model):
    return int(model.pymc_model[_OBSERVED_CARRY_FLAG].get_value())


@pytest.mark.slow
class TestFlagLifecycle:
    """The flag is set explicitly per call and restored afterwards."""

    def test_fit_leaves_flag_set_for_observed_carry(self, fitted_ar_panel):
        assert _flag(fitted_ar_panel) == 1

    @pytest.mark.parametrize("one_step_ahead", [True, False])
    def test_flag_restored_after_predict(self, fitted_ar_panel, one_step_ahead):
        fitted_ar_panel.predict(
            one_step_ahead=one_step_ahead,
            extend_inferencedata=False,
            progressbar=False,
        )
        assert _flag(fitted_ar_panel) == 1

    @pytest.mark.parametrize(
        ("one_step_ahead", "expected"),
        [(True, 1), (False, 0)],
    )
    def test_flag_value_seen_by_sampler(
        self, fitted_ar_panel, monkeypatch, one_step_ahead, expected
    ):
        seen = []
        original = pm.sample_posterior_predictive

        def _spy(*args, **kwargs):
            seen.append(_flag(fitted_ar_panel))
            return original(*args, **kwargs)

        monkeypatch.setattr(pm, "sample_posterior_predictive", _spy)
        fitted_ar_panel.predict(
            one_step_ahead=one_step_ahead,
            extend_inferencedata=False,
            progressbar=False,
        )
        assert seen == [expected]

    def test_do_still_runs_generatively(self, fitted_ar_panel):
        """``do()`` uses ``_gen_model``, whose flag must stay at 0."""
        gen_model = fitted_ar_panel._gen_model
        assert int(gen_model[_OBSERVED_CARRY_FLAG].get_value()) == 0

    @pytest.mark.parametrize("kind", ["mean", "predictive"])
    def test_do_forces_generative_carry_even_if_leaked(
        self, fitted_ar_panel, monkeypatch, kind
    ):
        """``do()`` must not inherit a leaked observed-carry flag (#327).

        Regression test: ``do()``/``run_do_panel_unified()`` used to call
        ``pm.sample_posterior_predictive``/``compute_deterministics``
        without explicitly managing the flag on the generative model, so
        correctness relied on the implicit invariant that the generative
        model's flag is always 0. If anything ever leaves it at 1 (e.g. a
        future code path, or a model where the observation and generative
        models are literally the same object), an interventional ``do()``
        query would silently condition on observed data instead of
        forward-simulating under the intervention. ``do()`` must force the
        flag off regardless of the generative model's ambient state.

        Parameterized over both ``kind`` branches, which are separate code
        paths in ``run_do_panel_unified``. The cross-sectional
        ``run_do_pymc`` call sites cannot be reached with a flagged model:
        the flag only exists on scan-compiled panel models, and those raise
        a shape error on the cross-sectional path (pre-existing on ``main``,
        see #394), so the guard there is defensive.
        """
        gen_model = fitted_ar_panel._gen_model
        flag = gen_model[_OBSERVED_CARRY_FLAG]

        kwargs = {"set": {"x": 0.0}, "kind": kind, "simulate_over": "time"}

        seen = []
        original_do = simulate_module.pm.do

        def _spy(*args, **kwargs_):
            seen.append(int(flag.get_value()))
            return original_do(*args, **kwargs_)

        monkeypatch.setattr(simulate_module.pm, "do", _spy)

        baseline = fitted_ar_panel.do(**kwargs)

        flag.set_value(np.array(1, dtype="int8"))
        leaked = fitted_ar_panel.do(**kwargs)
        # Assert restoration *before* any cleanup, so a context manager that
        # fails to put the previous value back cannot pass this test.
        assert int(flag.get_value()) == 1
        flag.set_value(np.array(0, dtype="int8"))

        # Graph surgery always saw a generative (0) carry flag, including
        # the call made while the ambient flag was leaked at 1.
        assert seen == [0, 0]

        if kind == "mean":
            # Deterministic branch: the leaked flag must not perturb values.
            # (The predictive branch draws fresh residual noise per call and
            # takes no seed, so it is covered by the spy assertion above.)
            np.testing.assert_allclose(
                leaked.dataset["y"].values,
                baseline.dataset["y"].values,
                rtol=1e-10,
            )

    def test_flag_restored_when_do_raises(self, fitted_ar_panel, monkeypatch):
        """The flag is restored on the exception path, not just on success."""
        gen_model = fitted_ar_panel._gen_model
        flag = gen_model[_OBSERVED_CARRY_FLAG]

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(simulate_module.pm, "do", _boom)

        flag.set_value(np.array(1, dtype="int8"))
        try:
            with pytest.raises(RuntimeError, match="boom"):
                fitted_ar_panel.do(set={"x": 0.0}, kind="mean", simulate_over="time")
            assert int(flag.get_value()) == 1
        finally:
            flag.set_value(np.array(0, dtype="int8"))

    def test_fit_do_predict_reuse_cycle(self, fitted_ar_panel):
        """A fit -> do -> predict cycle must not cross-contaminate flags.

        Regression test for #327 item 12: no test previously covered a
        full reuse cycle. ``do()`` operates on ``_gen_model`` and
        ``predict()`` operates on ``pymc_model``; calling one must not
        change the other's behavior on a subsequent call.
        """
        gen_model = fitted_ar_panel._gen_model
        obs_model = fitted_ar_panel.pymc_model

        # Sanity: fit() leaves the observation model conditioning on
        # observed carries, and the generative model purely generative.
        assert int(obs_model[_OBSERVED_CARRY_FLAG].get_value()) == 1
        assert int(gen_model[_OBSERVED_CARRY_FLAG].get_value()) == 0

        fitted_ar_panel.do(set={"x": 0.0}, kind="mean", simulate_over="time")

        # do() must not have disturbed either flag.
        assert int(obs_model[_OBSERVED_CARRY_FLAG].get_value()) == 1
        assert int(gen_model[_OBSERVED_CARRY_FLAG].get_value()) == 0

        fitted_ar_panel.predict(
            one_step_ahead=True, extend_inferencedata=False, progressbar=False
        )

        # predict() must restore the observation model's flag, and must
        # never have touched the generative model's flag.
        assert int(obs_model[_OBSERVED_CARRY_FLAG].get_value()) == 1
        assert int(gen_model[_OBSERVED_CARRY_FLAG].get_value()) == 0

        # do() again, after predict() -- must still be purely generative.
        after_predict = fitted_ar_panel.do(
            set={"x": 0.0}, kind="mean", simulate_over="time"
        )
        assert int(gen_model[_OBSERVED_CARRY_FLAG].get_value()) == 0
        assert np.isfinite(after_predict.dataset["y"].values).all()


@pytest.mark.slow
class TestGenerativePredict:
    """Free-running predictions are looser than one-step-ahead ones."""

    def test_generative_spread_exceeds_one_step_ahead(self, fitted_ar_panel):
        one_step = fitted_ar_panel.predict(
            extend_inferencedata=False, random_seed=1, progressbar=False
        )
        generative = fitted_ar_panel.predict(
            one_step_ahead=False,
            extend_inferencedata=False,
            random_seed=1,
            progressbar=False,
        )
        one_step_sd = float(
            one_step["posterior_predictive"].dataset["y"].std("draw").mean()
        )
        generative_sd = float(
            generative["posterior_predictive"].dataset["y"].std("draw").mean()
        )
        assert generative_sd > one_step_sd


class TestNonPanelModels:
    """Models without a lagged endogenous term ignore the argument."""

    def test_cross_sectional_predict_accepts_flag(self, mediation_data):
        model = pathmc.model("M ~ X\nY ~ M + X", data=mediation_data)
        model.fit(random_seed=42, compute_log_likelihood=False)
        assert _OBSERVED_CARRY_FLAG not in model.pymc_model.named_vars
        pp = model.predict(
            one_step_ahead=False, extend_inferencedata=False, progressbar=False
        )
        assert "Y" in pp["posterior_predictive"].dataset
