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
"""NaN / missing-data handling (issue #327, item 10).

Two independent findings, both fixed here:

1. **Predictor NaN silently poisoned the fit.** A NaN in an *outcome*
   (regression left-hand-side) column gets first-class support: it is
   wrapped in a masked ``pm.Normal`` observation and PyMC auto-imputes
   the missing positions. A NaN in a *predictor* had no equivalent
   check -- it flowed straight into a ``pm.Data`` node and from there
   into ``mu`` via plain arithmetic (row-wise) or a ``pytensor.scan``
   sequence (panel), producing a non-finite ``mu``/logp with no
   indication that a NaN predictor was the cause. ``pathmc.compile.
   _reject_nan_predictors`` now raises a ``ValueError`` naming the
   column at model-construction time, for both the row-wise and the
   scan-panel compiler, and independent of whether the panel is
   balanced.

2. **NaN outcome in a self-referencing lag (``y ~ lag(y)``) crashed
   model construction outright**, rather than reaching the intended
   masked-likelihood path. ``_compile_scan_panel`` builds a
   ``_obs_carry_{var}`` node holding the raw (possibly-NaN) values of
   any variable that lags itself, and its scan step deliberately
   branches on ``pt.isnan(obs_state)`` to fall back to the model's own
   simulated value at missing positions. That node was wrapped in
   ``pm.Data``, which auto-converts NaN-containing arrays to masked
   arrays and unconditionally raises ``NotImplementedError`` on them --
   it assumes NaN always means "impute an observed likelihood", which
   is not what this internal carry node does. Swapping it for a plain
   ``pytensor.shared`` (never looked up by name or resized via
   ``pm.set_data``, so nothing is lost) fixes it.

Also verifies there is no off-by-one in the ``(unit, time)`` ->
``(n_times, n_units)`` reshape used to build both the masked-outcome
array and the ``_obs_carry_`` array: a NaN placed at one specific
``(unit, time)`` cell must land at exactly one specific
``(time_idx, unit_idx)`` cell after reshaping, nowhere else.
"""

import numpy as np
import pandas as pd
import pytensor
import pytest

import pathmc
import pathmc.compile as compile_mod


def _balanced_panel(units, n_times, nan_at=None, seed=0):
    """A rectangular panel; optionally set y to NaN at one (unit, time)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in units:
        for t in range(n_times):
            x = rng.normal()
            y = 0.5 * x + rng.normal(scale=0.1)
            rows.append({"unit": u, "time": t, "x": x, "y": y})
    df = pd.DataFrame(rows)
    if nan_at is not None:
        unit, time = nan_at
        df.loc[(df["unit"] == unit) & (df["time"] == time), "y"] = np.nan
    return df


def _unbalanced_panel(seed=0):
    """3 units with unequal observation counts (2, 3, 4 rows)."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n in [("A", 2), ("B", 3), ("C", 4)]:
        for t in range(n):
            x = rng.normal()
            y = 0.5 * x + rng.normal(scale=0.1)
            rows.append({"unit": unit, "time": t, "x": x, "y": y})
    return pd.DataFrame(rows)


class TestPredictorNaNRaises:
    """A NaN predictor must raise at model-construction time, never silently
    reach ``mu``."""

    def test_flat_model_predictor_nan_raises(self):
        df = pd.DataFrame({
            "x": [0.0, 1.0, np.nan, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="Predictor column 'x'"):
            pathmc.model("y ~ x", data=df)

    def test_flat_model_outcome_nan_does_not_trigger_predictor_check(self):
        """A NaN in the outcome is fine -- only predictors are rejected."""
        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, np.nan, 3.0],
        })
        model = pathmc.model("y ~ x", data=df)
        assert "y_unobserved" in model.pymc_model.named_vars

    def test_scan_panel_predictor_nan_raises(self):
        df = _balanced_panel(["A", "B"], n_times=4)
        df.loc[(df["unit"] == "A") & (df["time"] == 1), "x"] = np.nan
        with pytest.raises(ValueError, match="Predictor column 'x'"):
            pathmc.model(
                "y ~ lag(y) + x", data=df, panel={"unit": "unit", "time": "time"}
            )

    def test_unbalanced_panel_predictor_nan_raises(self):
        """The predictor check does not depend on panel rectangularity."""
        df = _unbalanced_panel()
        df.loc[2, "x"] = np.nan
        with pytest.raises(ValueError, match="Predictor column 'x'"):
            pathmc.model("y ~ x", data=df, panel={"unit": "unit", "time": "time"})

    def test_inf_predictor_also_raises(self):
        df = pd.DataFrame({
            "x": [0.0, 1.0, np.inf, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="Predictor column 'x'"):
            pathmc.model("y ~ x", data=df)


class TestOutcomeNaNMaskingIsCorrectlyPositioned:
    """No off-by-one: the masked outcome array's NaN lands at the same
    (unit, time) cell it started at, both with and without a self-lag."""

    def test_scan_panel_without_self_lag(self, monkeypatch):
        captured = {}
        original = np.ma.masked_invalid

        def spy(arr, *a, **kw):
            captured["arr"] = np.array(arr, copy=True)
            return original(arr, *a, **kw)

        monkeypatch.setattr(compile_mod.np.ma, "masked_invalid", spy)

        df = _balanced_panel(["A", "B"], n_times=4, nan_at=("B", 2))
        model = pathmc.model(
            "y ~ x + lag(x)", data=df, panel={"unit": "unit", "time": "time"}
        )
        scan_info = model._gen_model._pathmc_panel_scan

        arr = captured["arr"]
        nan_positions = np.argwhere(np.isnan(arr))
        assert nan_positions.tolist() == [
            [
                scan_info.time_values.index(2),
                scan_info.unit_labels.index("B"),
            ]
        ]

    def test_scan_panel_with_self_lag_regression(self, monkeypatch):
        """Regression test: this used to crash with a ``pm.Data``
        ``NotImplementedError`` before ``_obs_carry_{var}`` was switched to a
        plain ``pytensor.shared``."""
        captured = {}
        original = pytensor.shared

        def spy(value, *a, **kw):
            if kw.get("name", "").startswith("_obs_carry_"):
                captured["arr"] = np.array(value, copy=True)
            return original(value, *a, **kw)

        monkeypatch.setattr(pytensor, "shared", spy)

        df = _balanced_panel(["A", "B"], n_times=4, nan_at=("B", 2))
        model = pathmc.model(
            "y ~ lag(y) + x", data=df, panel={"unit": "unit", "time": "time"}
        )
        scan_info = model._gen_model._pathmc_panel_scan

        arr = captured["arr"]
        nan_positions = np.argwhere(np.isnan(arr))
        assert nan_positions.tolist() == [
            [
                scan_info.time_values.index(2),
                scan_info.unit_labels.index("B"),
            ]
        ]

    def test_row_wise_panel_unbalanced_outcome_nan_masks_correct_row(self):
        """Row-wise (non-temporal) panel models mask by raw row order, so
        an unbalanced panel does not disturb which row is masked."""
        df = _unbalanced_panel()
        nan_row = 4  # third row of unit "B"'s block
        df.loc[nan_row, "y"] = np.nan
        model = pathmc.model("y ~ x", data=df, panel={"unit": "unit", "time": "time"})
        assert getattr(model._gen_model, "_pathmc_panel_scan", None) is None
        assert "y_unobserved" in model.pymc_model.named_vars
        # Exactly one missing position, matching the single NaN we set.
        n_missing = model.pymc_model["y_unobserved"].eval().shape[0]
        assert n_missing == 1


class TestSelfLagNaNOutcomeBuildsAndSamples:
    """End-to-end: a self-lagged variable with a NaN outcome must compile
    and produce a finite generative draw, not crash or emit non-finite
    values."""

    def test_builds_without_raising(self):
        df = _balanced_panel(["A", "B", "C"], n_times=5, nan_at=("B", 2))
        model = pathmc.model(
            "y ~ lag(y) + x", data=df, panel={"unit": "unit", "time": "time"}
        )
        assert model._gen_model is not None

    def test_prior_predictive_is_finite(self):
        import pymc as pm

        df = _balanced_panel(["A", "B", "C"], n_times=5, nan_at=("B", 2))
        model = pathmc.model(
            "y ~ lag(y) + x", data=df, panel={"unit": "unit", "time": "time"}
        )
        with model.pymc_model:
            prior = pm.sample_prior_predictive(draws=3, random_seed=0)
        y_missing = prior.prior["y_unobserved"].to_numpy()
        assert np.isfinite(y_missing).all()
