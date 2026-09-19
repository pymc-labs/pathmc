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
"""Fit-time state and treatment coding for categorical predictors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd

from pathmc.parse import CategoricalCall, Spec

__all__: list[str] = []


def _is_categorical_series(series: pd.Series) -> bool:
    """Return whether a pandas series should be auto-treated as categorical."""
    dtype = series.dtype
    return isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype)


def _fit_levels(series: pd.Series) -> tuple[Any, ...]:
    """Return deterministic levels, preserving declared categorical order."""
    if series.isna().any():
        raise ValueError(
            f"Categorical predictor '{series.name}' contains missing values. "
            "Drop or impute missing categories before fitting."
        )
    if isinstance(series.dtype, pd.CategoricalDtype):
        levels = tuple(series.cat.categories.tolist())
    else:
        values = series.unique().tolist()
        levels = tuple(sorted(values, key=lambda value: str(value)))
    if len(levels) < 2:
        raise ValueError(
            f"Categorical predictor '{series.name}' has {len(levels)} level(s). "
            "Provide at least two observed levels or remove the predictor."
        )
    return levels


def _level_column(variable: str, level: Any, *, reference_coded: bool) -> str:
    marker = "T." if reference_coded else ""
    return f"{variable}[{marker}{level}]"


def fit_categorical_terms(spec: Spec, data: nw.DataFrame) -> set[str]:
    """Resolve explicit and inferred categorical terms against fitting data.

    The function updates the parsed term objects with immutable level, reference,
    and output-column state. It returns the source column names that were treated
    as categorical.
    """
    pandas_data = data.to_pandas()
    categorical_vars: set[str] = set()
    endogenous = {reg.lhs for reg in spec.regressions}

    for reg in spec.regressions:
        for term in reg.terms:
            if term.variable not in pandas_data.columns:
                continue
            explicit = term.categorical is not None
            if not explicit and not _is_categorical_series(pandas_data[term.variable]):
                continue
            if term.transform is not None or term.interaction_of is not None:
                raise NotImplementedError(
                    f"Categorical predictor '{term.variable}' is used in a transform "
                    "or interaction. Categorical interactions are not supported yet; "
                    "use the categorical as a standalone term."
                )
            if term.variable in endogenous:
                raise NotImplementedError(
                    f"Categorical endogenous variable '{term.variable}' is not "
                    "supported. Categorical outcomes require a categorical "
                    "likelihood; use a categorical predictor only."
                )

            call = term.categorical or CategoricalCall(variable=term.variable)
            levels = _fit_levels(pandas_data[term.variable])
            reference = levels[0] if call.reference is None else call.reference
            if reference not in levels:
                raise ValueError(
                    f"Reference level {reference!r} for categorical predictor "
                    f"'{term.variable}' was not observed. Available levels: "
                    f"{list(levels)!r}."
                )
            coefficient_levels = (
                tuple(level for level in levels if level != reference)
                if reg.has_intercept
                else levels
            )
            columns = tuple(
                _level_column(
                    term.variable,
                    level,
                    reference_coded=reg.has_intercept,
                )
                for level in coefficient_levels
            )
            term.categorical = replace(
                call,
                reference=reference,
                levels=levels,
                columns=columns,
            )
            categorical_vars.add(term.variable)

    return categorical_vars


def coefficient_levels(call: CategoricalCall) -> tuple[Any, ...]:
    """Return the levels represented by a categorical coefficient vector."""
    if len(call.columns) == len(call.levels):
        return call.levels
    return tuple(level for level in call.levels if level != call.reference)


def encode_categorical(
    call: CategoricalCall,
    values: Any,
    *,
    variable: str | None = None,
) -> np.ndarray:
    """Encode values using fitted treatment-coding state.

    Raises a clear error for unseen values instead of allowing pandas or patsy to
    infer a fresh level set or silently turn an unseen value into missing data.
    """
    name = variable or call.variable
    arr = np.asarray(values, dtype=object).reshape(-1)
    missing = pd.isna(arr)
    if missing.any():
        raise ValueError(
            f"Categorical predictor '{name}' contains missing values. Drop or "
            "impute missing categories before prediction or intervention."
        )
    unseen = sorted({value for value in arr if value not in call.levels}, key=str)
    if unseen:
        raise ValueError(
            f"Categorical predictor '{name}' contains unseen level(s) "
            f"{unseen!r}. Fitted levels are {list(call.levels)!r}."
        )
    levels = coefficient_levels(call)
    return np.column_stack([arr == level for level in levels]).astype(float)


def categorical_terms(spec: Spec) -> list[tuple[str, CategoricalCall]]:
    """Return ``(lhs, call)`` pairs for fitted categorical terms."""
    return [
        (reg.lhs, term.categorical)
        for reg in spec.regressions
        for term in reg.terms
        if term.categorical is not None and term.categorical.levels
    ]
