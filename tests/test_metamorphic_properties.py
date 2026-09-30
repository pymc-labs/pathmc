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
"""Tier 3 metamorphic logp-graph properties (issues #326, #412).

These checks must hold at any parameter point. They do not sample and they
do not need a known ground truth. Each property names the bug class it
catches in the helper docstring in ``tests/_consistency.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pathmc
from _consistency import (
    assert_duplicated_observations_double_likelihood,
    assert_lag_matches_manual_shift,
    assert_predictor_scale_covariance,
    assert_row_order_invariant,
)
from test_graph_consistency import _PANEL, _panel_data, _xsec_data

_N_GEO = 5
_N_TIME = 12


def _shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _manual_lag_column(
    df: pd.DataFrame, col: str, unit: str, time: str, out_col: str
) -> pd.DataFrame:
    """Per-unit scan lag: position 0 repeats, later positions lag by one."""
    parts = []
    ordered = df.sort_values([unit, time])
    for _, group in ordered.groupby(unit, sort=True):
        group = group.copy()
        values = group[col].to_numpy()
        group[out_col] = np.concatenate([values[:1], values[:-1]])
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _duplicate_units(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    extra = df.copy()
    extra[unit] = extra[unit] + df[unit].nunique()
    return pd.concat([df, extra], ignore_index=True)


# (spec, data, panel, pooling). Panel lag cells plus one cross-sectional Gaussian.
_ROW_ORDER_CASES = [
    ("Y ~ X1 + X2", _xsec_data(), None, None),
    ("sales ~ lag(spend)", _panel_data(), _PANEL, None),
    ("sales ~ lag(spend)", _panel_data(), _PANEL, "partial"),
    ("sales ~ 0 + lag(spend)", _panel_data(), _PANEL, "partial"),
    ("sales ~ lag(sales) + x2", _panel_data(), _PANEL, None),
]


@pytest.mark.parametrize(
    ("spec", "data", "panel", "pooling"),
    _ROW_ORDER_CASES,
    ids=[
        "xsec-gaussian",
        "panel-lag(x)-complete",
        "panel-lag(x)-partial",
        "panel-lag(x)-no-intercept",
        "panel-lag(y)-complete",
    ],
)
def test_row_permutation_invariance(spec, data, panel, pooling):
    """Shuffling row order leaves joint logp unchanged."""
    original = pathmc.model(spec, data=data, panel=panel, pooling=pooling)
    shuffled = pathmc.model(
        spec, data=_shuffle(data, seed=0), panel=panel, pooling=pooling
    )
    assert_row_order_invariant(original, shuffled, seed=0)


@pytest.mark.parametrize(
    ("spec", "pooling"),
    [
        ("sales ~ lag(spend)", None),
        ("sales ~ lag(spend)", "partial"),
        ("sales ~ 0 + lag(spend)", "partial"),
    ],
    ids=[
        "panel-lag(x)-complete",
        "panel-lag(x)-partial",
        "panel-lag(x)-no-intercept",
    ],
)
def test_lag_definition_equivalence(spec, pooling):
    """``y ~ lag(x)`` forward mu matches a hand-shifted column."""
    raw = _panel_data()
    shifted = _manual_lag_column(raw, "spend", "geo", "week", "spend_shifted")
    lag = pathmc.model(spec, data=raw, panel=_PANEL, pooling=pooling)
    twin = pathmc.model(
        spec.replace("lag(spend)", "spend_shifted"),
        data=shifted,
        panel=_PANEL,
        pooling=pooling,
    )
    assert_lag_matches_manual_shift(
        lag,
        twin,
        outcome="sales",
        n_times=_N_TIME,
        n_units=_N_GEO,
        seed=0,
    )


@pytest.mark.parametrize(
    ("spec", "data", "panel", "duplicate"),
    [
        ("Y ~ X1 + X2", _xsec_data(), None, "rows"),
        ("sales ~ lag(spend)", _panel_data(), _PANEL, "units"),
        ("sales ~ lag(sales) + x2", _panel_data(), _PANEL, "units"),
    ],
    ids=[
        "xsec-gaussian",
        "panel-lag(x)-complete",
        "panel-lag(y)-complete",
    ],
)
def test_duplicate_data_scales_likelihood(spec, data, panel, duplicate):
    """Duplicating rows or units doubles the observation likelihood."""
    if duplicate == "rows":
        doubled = pd.concat([data, data], ignore_index=True)
    else:
        doubled = _duplicate_units(data, "geo")
    original = pathmc.model(spec, data=data, panel=panel, pooling=None)
    copied = pathmc.model(spec, data=doubled, panel=panel, pooling=None)
    assert_duplicated_observations_double_likelihood(original, copied, seed=0)


def test_scale_shift_panel_lag():
    """Scaling spend by c and dividing the lag slope by c leaves mu unchanged."""
    raw = _panel_data()
    scale = 2.5
    scaled = raw.copy()
    scaled["spend"] = scaled["spend"] * scale
    spec = "sales ~ 0 + lag(spend)"
    original = pathmc.model(spec, data=raw, panel=_PANEL, pooling=None)
    scaled_model = pathmc.model(spec, data=scaled, panel=_PANEL, pooling=None)
    assert_predictor_scale_covariance(
        original,
        scaled_model,
        outcome="sales",
        scale=scale,
        beta_index=0,
        n_times=_N_TIME,
        n_units=_N_GEO,
        seed=0,
    )


def test_scale_shift_cross_sectional_gaussian():
    """Scaling X1 by c and dividing its slope by c leaves mu unchanged."""
    raw = _xsec_data()
    scale = 2.5
    scaled = raw.copy()
    scaled["X1"] = scaled["X1"] * scale
    spec = "Y ~ X1"
    original = pathmc.model(spec, data=raw)
    scaled_model = pathmc.model(spec, data=scaled)
    assert_predictor_scale_covariance(
        original,
        scaled_model,
        outcome="Y",
        scale=scale,
        beta_index=1,
        seed=0,
    )
