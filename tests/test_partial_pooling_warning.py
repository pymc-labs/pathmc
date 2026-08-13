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
"""Partial pooling auto-drops redundant formula intercepts on panel models."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def panel_data():
    """Panel data: 5 units, 10 time periods, y ~ x."""
    rng = np.random.default_rng(42)
    frames = []
    for g in range(5):
        x = rng.normal(0, 1, 10)
        y = 5 + 2 * x + rng.normal(0, 1, 10)
        frames.append(pd.DataFrame({"x": x, "y": y, "week": np.arange(10), "geo": g}))
    return pd.concat(frames, ignore_index=True)


class TestPartialPoolingInterceptAutoDrop:
    """Partial pooling on panel models drops the fixed formula intercept."""

    def test_no_warning_with_implicit_intercept(self, panel_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )

    def test_intercept_dropped_from_predictors(self, panel_data):
        model = pathmc.model(
            "y ~ x",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling="partial",
        )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]

    def test_explicit_no_intercept_unchanged(self, panel_data):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = pathmc.model(
                "y ~ 0 + x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]

    def test_no_drop_without_pooling(self, panel_data):
        model = pathmc.model(
            "y ~ x",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
        )
        assert "Intercept" in model.pymc_model.coords["y_predictors"]

    def test_dict_pooling_intercept_true_drops(self, panel_data):
        model = pathmc.model(
            "y ~ x",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling={"intercept": True},
        )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]

    def test_dict_pooling_intercept_false_keeps(self, panel_data):
        model = pathmc.model(
            "y ~ x",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling={"intercept": False},
        )
        assert "Intercept" in model.pymc_model.coords["y_predictors"]

    def test_all_equations_drop_intercept(self, panel_data):
        panel_data["z"] = np.random.default_rng(43).normal(0, 1, len(panel_data))
        model = pathmc.model(
            "y ~ x; z ~ y",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling="partial",
        )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]
        assert "Intercept" not in model.pymc_model.coords["z_predictors"]


class TestPartialPoolingInterceptAutoDropWithLag:
    """Scan-compiled panel models also drop redundant intercepts."""

    def test_lag_model_drops_intercept(self, panel_data):
        model = pathmc.model(
            "y ~ x + lag(y)",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling="partial",
        )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]

    def test_lag_model_explicit_no_intercept(self, panel_data):
        model = pathmc.model(
            "y ~ 0 + x + lag(y)",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling="partial",
        )
        assert "Intercept" not in model.pymc_model.coords["y_predictors"]
