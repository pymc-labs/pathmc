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
"""Tier 3 metamorphic log-graph properties (issues #326, #412).

These checks must hold at any parameter we do not sample and they do not need
a known ground truth. Each property names the log class it catches in the
helper docstring in ``tests/_consistency.py``.

Cells come from the shared compile matrix. This module filters that table by
id. It does not restate specifications or data formulas.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from _consistency import (
    _align_mu,
    _shared_points,
    assert_duplicated_observations_double_likelihood,
    assert_lag_matches_manual_shift,
    assert_predictor_scale_covariance,
    assert_row_order_invariant,
    cells,
)

_LAG_TERM = re.compile(r"lag\(([^)]+)\)")

# Explicit ids. The record stays the requested fields; selection is not a new
# column on ConsistencyCell.
_ROW_ORDER = (
    "xsec-gaussian",
    "panel-lag(x)-complete",
    "panel-lag(x)-partial",
    "panel-lag(x)-no-intercept",
    "panel-lag(y)-complete",
)
_LAG_SHIFT = (
    "panel-lag(x)-complete",
    "panel-lag(x)-partial",
    "panel-lag(x)-no-intercept",
)
# (cell id, point keys whose unit axis grows when units are duplicated).
_DUPLICATE = (
    ("xsec-gaussian", ()),
    ("panel-lag(x)-complete", ()),
    ("panel-lag(y)-complete", ("carry_innovations_sales",)),
)
# (cell id, data column to scale, predictor name in the beta coordinate).
_SCALE = (
    ("xsec-gaussian", "X1", "X1"),
    ("panel-lag(x)-no-intercept", "spend", "lag(spend)"),
)


def _outcome(spec: str) -> str:
    return spec.split("~", 1)[0].strip()


def _lag_column(spec: str) -> str:
    found = _LAG_TERM.findall(spec)
    if len(found) != 1:
        raise ValueError(
            f"{spec!r} does not have exactly one lag() term. Filter the cell "
            "table instead of hardcoding the column."
        )
    return found[0]


def _beta_index(model, outcome: str, predictor: str) -> int:
    names = list(model.pymc_model.coords[f"{outcome}_predictors"])
    return names.index(predictor)


@pytest.mark.parametrize("cell", cells(*_ROW_ORDER), ids=lambda cell: cell.id)
def test_row_order_is_invariant(cell):
    """Permuting rows must not change joint logp."""
    raw = cell.data()
    shuffled = raw.sample(frac=1, random_state=1).reset_index(drop=True)
    assert_row_order_invariant(
        cell.model(raw),
        cell.model(shuffled),
        seed=0,
    )


@pytest.mark.parametrize("cell", cells(*_LAG_SHIFT), ids=lambda cell: cell.id)
def test_lag_definition_matches_manual_shift(cell):
    """``lag(x)`` matches a column that repeats period 0, then shifts by one."""
    column = _lag_column(cell.spec)
    shifted = f"{column}_shifted"
    raw = cell.data()
    unit = cell.panel["unit"]
    manual = raw.copy()
    manual[shifted] = manual.groupby(unit, sort=False)[column].transform(
        lambda series: series.shift(1).fillna(series.iloc[0])
    )
    twin_spec = cell.spec.replace(f"lag({column})", shifted)
    assert_lag_matches_manual_shift(
        cell.model(raw),
        cell.model(manual, spec=twin_spec),
        outcome=_outcome(cell.spec),
        seed=0,
    )


@pytest.mark.parametrize(
    ("cell_id", "allow"),
    _DUPLICATE,
    ids=[item[0] for item in _DUPLICATE],
)
def test_duplicate_data_doubles_likelihood(cell_id, allow):
    """Duplicating observations doubles likelihood, not the joint logp."""
    cell = cells(cell_id)[0]
    raw = cell.data()
    if cell.panel is None:
        doubled = pd.concat([raw, raw], ignore_index=True)
    else:
        unit = cell.panel["unit"]
        extra = raw.copy()
        extra[unit] = extra[unit] + raw[unit].nunique()
        doubled = pd.concat([raw, extra], ignore_index=True)
    assert_duplicated_observations_double_likelihood(
        cell.model(raw),
        cell.model(doubled),
        seed=0,
        allow=allow,
    )


@pytest.mark.parametrize(
    ("cell_id", "column", "predictor"),
    _SCALE,
    ids=[item[0] for item in _SCALE],
)
def test_scale_shift_covariance(cell_id, column, predictor):
    """Scaling a predictor by ``c`` and dividing its slope by ``c`` preserves mu."""
    cell = cells(cell_id)[0]
    scale = 2.5
    raw = cell.data()
    scaled = raw.copy()
    scaled[column] = scaled[column] * scale
    original = cell.model(raw)
    outcome = _outcome(cell.spec)
    assert_predictor_scale_covariance(
        original,
        cell.model(scaled),
        outcome=outcome,
        scale=scale,
        beta_index=_beta_index(original, outcome, predictor),
        seed=0,
    )


def test_square_panel_lag_mu_aligns_to_sorted_unit_time():
    """A square panel must still sort ``(unit, time)``, not skip the transpose.

    ``n_times == n_units`` makes a shape sniff ambiguous. Scan layout comes
    from the compiler; the hand-shifted twin is not scan-compiled, so a
    skipped transpose fails this comparison.
    """
    cell = cells("panel-lag(x)-complete")[0]
    raw = cell.data(ngeo=4, ntime=4)
    column = _lag_column(cell.spec)
    shifted = f"{column}_shifted"
    unit = cell.panel["unit"]
    manual = raw.copy()
    manual[shifted] = manual.groupby(unit, sort=False)[column].transform(
        lambda series: series.shift(1).fillna(series.iloc[0])
    )
    assert_lag_matches_manual_shift(
        cell.model(raw),
        cell.model(manual, spec=cell.spec.replace(f"lag({column})", shifted)),
        outcome=_outcome(cell.spec),
        seed=0,
    )


def test_shared_points_raises_on_shape_mismatch():
    """Partial pooling grows the unit intercept when units are added.

    The silent skip used to drop ``alpha_sales``. Callers that expect the
    mismatch pass ``allow``; this test does not.
    """
    cell = cells("panel-lag(x)-partial")[0]
    raw = cell.data()
    extra = raw.copy()
    unit = cell.panel["unit"]
    extra[unit] = extra[unit].to_numpy() + int(raw[unit].nunique())
    doubled = pd.concat([raw, extra], ignore_index=True)
    with pytest.raises(ValueError, match="alpha_sales"):
        _shared_points(cell.model(raw), cell.model(doubled), seed=0)


def test_align_mu_rejects_square_nonscan_panel():
    """A square array on a non-scan panel is not a row vector. Do not reshape."""
    cell = cells("panel-plain-complete")[0]
    model = cell.model(cell.data(ngeo=4, ntime=4))
    with pytest.raises(ValueError, match="square panel"):
        _align_mu(model, np.zeros((4, 4)))
