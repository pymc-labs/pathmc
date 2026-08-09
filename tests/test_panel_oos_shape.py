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
"""Out-of-sample prediction on a differently-shaped panel (issue #327, item 9).

The scan-based panel compiler (``_compile_scan_panel`` in
``pathmc/compile.py``) bakes ``n_units``/``n_times`` into the shapes of its
free RVs (e.g. ``carry_innovations_y``) and scan-carry initial values at
model-*compile* time. Swapping in a differently-shaped panel afterwards via
``pm.set_data()`` does not resize those baked shapes.

Two failure modes were possible before the fix:

* A mismatched *unit* count raises a cryptic PyTensor error deep inside
  ``sample_posterior_predictive`` (confirmed by manual reproduction).
* A mismatched *time* count (same units) is worse: ``pytensor.scan`` derives
  its step count from the *shortest* sequence among carry/innovation RVs and
  exogenous ``pm.Data``, so lengthening the panel *silently truncates* the
  extra rows instead of raising -- ``predict()`` returns a result with fewer
  timesteps than the caller's new data, with no error.

``pathmc.compile.validate_panel_scan_shape`` closes both: ``predict()`` now
checks every 2-D ``pm.Data`` node against the compile-time ``(n_times,
n_units)`` shape and raises ``ValueError`` before sampling if anything has
been resized.

These tests build the posterior manually (``az.from_dict``) instead of
running MCMC, so they are fast and deterministic.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc


def _make_panel(n_units: int, n_times: int, seed: int = 0) -> pd.DataFrame:
    """A balanced panel with a trivial ``y ~ lag(y) + x`` structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        for t in range(n_times):
            x = rng.normal()
            y = 0.5 * x + rng.normal(scale=0.1)
            rows.append({"unit": f"u{u}", "time": t, "x": x, "y": y})
    return pd.DataFrame(rows)


def _fake_fit(model: pathmc.PathModel, n_times: int, n_units: int) -> None:
    """Attach a hand-built posterior, skipping MCMC for test speed.

    Matches the free RVs of ``y ~ lag(y) + x`` under the scan compiler:
    ``beta_y`` (3 predictors: Intercept, lag(y), x), ``sigma_y`` (scalar),
    and ``carry_innovations_y`` (shape ``(n_times, n_units)``).
    """
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_y": np.tile([0.0, 0.5, 0.5], (1, 2, 1)),
                "sigma_y": np.full((1, 2), 0.1),
                "carry_innovations_y": np.zeros((1, 2, n_times, n_units)),
            }
        },
        coords={"y_predictors": ["Intercept", "lag(y)", "x"]},
        dims={
            "beta_y": ["y_predictors"],
            "carry_innovations_y": [
                "carry_innovations_y_dim_0",
                "carry_innovations_y_dim_1",
            ],
        },
    )


@pytest.fixture
def scan_panel_model():
    """A fitted (fake-posterior) scan panel model: 3 units x 5 times."""
    df = _make_panel(n_units=3, n_times=5)
    model = pathmc.model(
        "y ~ lag(y) + x", data=df, panel={"unit": "unit", "time": "time"}
    )
    _fake_fit(model, n_times=5, n_units=3)
    return model


class TestUnchangedShapePredictsFine:
    """Baseline: predicting without touching the data must still work."""

    def test_predict_without_set_data_succeeds(self, scan_panel_model):
        pp = scan_panel_model.predict(extend_inferencedata=False, progressbar=False)
        y = pp.posterior_predictive["y"].to_numpy()
        assert y.shape[-2:] == (5, 3)
        assert np.isfinite(y).all()

    def test_predict_after_set_data_same_shape_succeeds(self, scan_panel_model):
        """Re-setting data at the identical shape is a no-op, not an error."""
        same_x = _make_panel(3, 5, seed=1)
        sort_same = same_x.sort_values(["unit", "time"])
        x_mat = sort_same["x"].to_numpy().reshape(3, 5).T
        with scan_panel_model.pymc_model:
            pm.set_data({"x": x_mat})
        pp = scan_panel_model.predict(extend_inferencedata=False, progressbar=False)
        assert np.isfinite(pp.posterior_predictive["y"].to_numpy()).all()


class TestReshapedPanelRaises:
    """A differently-shaped panel must raise, never silently mis-reshape."""

    @pytest.mark.parametrize(
        ("new_n_units", "new_n_times"),
        [
            (5, 5),  # more units, same times
            (2, 5),  # fewer units, same times
            (3, 8),  # same units, more times (the silent-truncation case)
            (3, 2),  # same units, fewer times
            (5, 3),  # different split, same total row count (3*5 == 5*3)
        ],
        ids=[
            "more-units",
            "fewer-units",
            "more-times",
            "fewer-times",
            "same-size-diff-split",
        ],
    )
    def test_set_data_then_predict_raises_value_error(
        self, scan_panel_model, new_n_units, new_n_times
    ):
        new_df = _make_panel(new_n_units, new_n_times, seed=2)
        sort_new = new_df.sort_values(["unit", "time"])
        x_new = (
            sort_new["x"].to_numpy().reshape(new_n_units, new_n_times).T
        )  # (new_n_times, new_n_units)

        with scan_panel_model.pymc_model:
            pm.set_data({"x": x_new})

        with pytest.raises(ValueError, match="panel"):
            scan_panel_model.predict(extend_inferencedata=False, progressbar=False)

    def test_error_message_names_the_offending_variable_and_shapes(
        self, scan_panel_model
    ):
        new_df = _make_panel(3, 8, seed=3)
        sort_new = new_df.sort_values(["unit", "time"])
        x_new = sort_new["x"].to_numpy().reshape(3, 8).T
        with scan_panel_model.pymc_model:
            pm.set_data({"x": x_new})

        with pytest.raises(ValueError) as excinfo:
            scan_panel_model.predict(extend_inferencedata=False, progressbar=False)

        message = str(excinfo.value)
        assert "'x'" in message
        assert "(8, 3)" in message
        assert "(5, 3)" in message


class TestNonScanModelsAreUnaffected:
    """Non-temporal models have no ``PanelScanInfo`` and skip the new check."""

    def test_cross_sectional_model_has_no_scan_info(self):
        df = pd.DataFrame({
            "x": np.arange(10, dtype=float),
            "y": np.arange(10, dtype=float) * 0.5,
        })
        model = pathmc.model("y ~ x", data=df)
        assert getattr(model._gen_model, "_pathmc_panel_scan", None) is None
