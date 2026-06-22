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
"""Regression tests for nutpie compilation of scan-panel models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc

nutpie = pytest.importorskip("nutpie")


def _panel_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(5):
        spend = rng.normal(10, 1, 10)
        sales = np.ones(10) * 20
        sales[1:] = 10 + spend[:-1] + rng.normal(0, 1, 9)
        frames.append(
            pd.DataFrame({
                "spend": spend,
                "sales": sales,
                "week": np.arange(10),
                "geo": g,
            })
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("pooling", [None, "partial"])
def test_no_intercept_exogenous_lag_compiles_with_nutpie(pooling):
    """Issue #336: length-one scan beta vectors must survive nutpie cloning."""
    model = pathmc.model(
        "sales ~ 0 + lag(spend)",
        data=_panel_data(),
        panel={"unit": "geo", "time": "week"},
        pooling=pooling,
    )

    assert list(model.pymc_model.coords["sales_predictors"]) == ["lag(spend)"]
    nutpie.compile_pymc_model(model.pymc_model)


@pytest.mark.parametrize(
    "spec",
    [
        "sales ~ 0 + lag(sales)",
        "sales ~ 0 + adstock(spend, decay=d)",
    ],
)
def test_no_intercept_single_predictor_temporal_models_compile_with_nutpie(spec):
    """The #336 workaround should apply to all scan temporal predictors."""
    model = pathmc.model(
        spec,
        data=_panel_data(),
        panel={"unit": "geo", "time": "week"},
    )

    nutpie.compile_pymc_model(model.pymc_model)
