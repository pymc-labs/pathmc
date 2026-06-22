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
"""Behaviour of the exogenous-lag scan builder when the base column is missing.

The exog-lag fix (the #316 follow-up) builds the lagged sequence directly from
the ``pm.Data`` nodes::

    lagged_exog_sequences[base] = pt.concatenate(
        [init_row, exog_data_nodes[base][:-1]], axis=0
    )

But ``exog_lag_bases`` is filtered only on ``base not in endo_set`` while
``exog_data_nodes`` *additionally* requires ``base in data_sorted.columns``.
So a ``lag(x)`` term whose contemporaneous column ``x`` is absent from the data
lands in ``exog_lag_bases`` **without** a corresponding entry in
``exog_data_nodes`` — and a direct index there raises ``KeyError``.

``_compile_scan_panel`` already treats that as a reachable state (the
``init_exog_lag`` ``else`` branch falls back to zeros), and the carry path that
this code replaced tolerated it via ``exog_t.get(k, pt.zeros(n_units))`` —
resolving the lag to the init row at ``t=0`` and zeros for ``t>=1``.  These
tests pin that behaviour: the model must compile, take the missing-base
``else`` branch, stay graph-consistent, and contribute exactly zero from the
absent lag regressor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytensor

import pathmc
from _consistency import assert_model_consistent

_PANEL = {"unit": "geo", "time": "week"}


def _panel_data(columns, *, ngeo=4, ntime=8, seed=1):
    """Panel frame with a ``sales`` outcome and the requested extra columns.

    ``columns`` are filled with noise; ``spend`` is deliberately *never* added
    so that ``lag(spend)`` references a base absent from the data.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(ngeo):
        cols = {"week": np.arange(ntime), "geo": g}
        for name in columns:
            cols[name] = rng.normal(0, 1, ntime)
        cols["sales"] = 10.0 + rng.normal(0, 1, ntime)
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def _mu_fn(pm_model):
    """Compile ``mu_sales`` in value space over all value vars."""
    (mu_node,) = pm_model.replace_rvs_by_values([pm_model["mu_sales"]])
    value_vars = pm_model.value_vars
    fn = pytensor.function(value_vars, mu_node, on_unused_input="ignore")
    return fn, value_vars


def test_missing_lag_base_compiles_without_keyerror():
    """``lag(spend)`` with no ``spend`` column must build, not raise KeyError.

    Pre-fix this raised ``KeyError: 'spend'`` from the direct
    ``exog_data_nodes[base]`` index; the missing-base ``else`` branch is what
    lets it compile.
    """
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=_panel_data(["x2"]),
        panel=_PANEL,
        pooling=None,
    )
    pm_model = model.pymc_model

    # The absent base must NOT have been materialised as a pm.Data node —
    # i.e. we genuinely took the missing-base branch rather than the happy path.
    assert "spend" not in pm_model.named_vars
    # The lag effect is still a free coefficient in the model.
    assert "beta_sales" in pm_model.named_vars


def test_missing_lag_base_graph_is_consistent():
    """The compiled graph must pass the no-sampling #316 battery."""
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=_panel_data(["x2"]),
        panel=_PANEL,
        pooling=None,
    )
    # mu context-invariance + finite logp/grad (no #316-style corruption).
    assert_model_consistent(model)


def test_missing_lag_base_contributes_zero():
    """An absent lag base behaves as an all-zero regressor.

    The missing-base sequence is all zeros, so ``lag(spend)`` contributes
    nothing to ``mu_sales``: perturbing *only* the lag coefficient (holding the
    intercept and every other value fixed) must leave ``mu_sales`` bit-for-bit
    unchanged.  This is the precise statement of "the absent base contributes
    zero" — independent of the intercept, which the formula adds by default.
    """
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=_panel_data(["x2"]),
        panel=_PANEL,
        pooling=None,
    )
    pm_model = model.pymc_model
    fn, value_vars = _mu_fn(pm_model)

    # Locate the lag coefficient within the (Intercept, lag(spend)) beta vector.
    predictors = list(pm_model.coords["sales_predictors"])
    lag_idx = predictors.index("lag(spend)")

    point = pm_model.initial_point(random_seed=0)
    args = {v.name: np.asarray(point[v.name], dtype=float).copy() for v in value_vars}

    mu_base = np.asarray(fn(*[args[v.name] for v in value_vars]))

    perturbed = {k: v.copy() for k, v in args.items()}
    perturbed["beta_sales"][lag_idx] += 7.0  # move ONLY the lag coefficient
    mu_perturbed = np.asarray(fn(*[perturbed[v.name] for v in value_vars]))

    np.testing.assert_allclose(mu_base, mu_perturbed, atol=1e-12)


def test_missing_base_with_lag1_column_seeds_init_row():
    """A precomputed ``spend_lag1`` column (no ``spend``) still compiles.

    Here ``init_exog_lag`` is seeded from ``spend_lag1`` (a non-zero init row),
    while the contemporaneous ``spend`` column is still absent — so the
    missing-base ``else`` branch builds ``[init_row, zeros]``.  The base must
    not appear as a data node, and the graph must stay consistent.
    """
    model = pathmc.model(
        "sales ~ lag(spend)",
        data=_panel_data(["spend_lag1"]),
        panel=_PANEL,
        pooling=None,
    )
    pm_model = model.pymc_model

    assert "spend" not in pm_model.named_vars
    assert_model_consistent(model)
