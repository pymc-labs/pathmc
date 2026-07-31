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

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc
from pathmc._model import _OBSERVED_CARRY_FLAG


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
