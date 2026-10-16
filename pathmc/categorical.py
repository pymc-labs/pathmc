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

import warnings
from dataclasses import replace
from typing import Any

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd

from pathmc.exceptions import ParseError
from pathmc.parse import CategoricalCall, Regression, Spec, Term

__all__: list[str] = []


def _is_categorical_series(series: pd.Series) -> bool:
    """Return whether a pandas series should be auto-treated as categorical."""
    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype | pd.StringDtype):
        return True
    if not pd.api.types.is_object_dtype(dtype):
        return False
    inferred = pd.api.types.infer_dtype(series, skipna=True)
    return inferred in {"bytes", "string", "unicode"}


def _reject_mixed_object_series(series: pd.Series) -> None:
    """Reject object columns that mix labels with numbers.

    Such a column is neither a clean set of labels (so it is not inferred as
    categorical) nor numeric (so it cannot be a continuous predictor). Left
    alone, patsy would expand it into its own dummy columns and the design
    lookup would fail with a bare ``KeyError``.
    """
    if not pd.api.types.is_object_dtype(series.dtype):
        return
    inferred = pd.api.types.infer_dtype(series, skipna=True)
    if inferred in {"mixed", "mixed-integer"}:
        raise ValueError(
            f"Predictor '{series.name}' is an object column mixing numbers and "
            f"non-numeric values (pandas infers {inferred!r}), so it is neither "
            "a numeric predictor nor a set of labels. Wrap it as "
            f"C({series.name}) to treat every distinct value as a level, or "
            "recode the column to one consistent type."
        )


def _validate_level_labels(name: str, levels: tuple[Any, ...]) -> None:
    """Reject distinct levels whose public coefficient labels would collide."""
    rendered: dict[str, Any] = {}
    for level in levels:
        label = str(level)
        if label in rendered:
            previous = rendered[label]
            raise ValueError(
                f"Categorical predictor '{name}' has distinct levels "
                f"{previous!r} ({type(previous).__name__}) and {level!r} "
                f"({type(level).__name__}) that both render as {label!r}. "
                "Recode the levels to unique string labels before fitting."
            )
        rendered[label] = level


def _fit_levels(series: pd.Series) -> tuple[Any, ...]:
    """Return deterministic levels, preserving declared categorical order."""
    if series.isna().any():
        raise ValueError(
            f"Categorical predictor '{series.name}' contains missing values. "
            "Drop or impute missing categories before fitting."
        )
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.unique().tolist())
        levels = tuple(
            level for level in series.cat.categories.tolist() if level in observed
        )
    else:
        values = series.unique().tolist()
        levels = tuple(sorted(values, key=lambda value: str(value)))
    _validate_level_labels(str(series.name), levels)
    if len(levels) < 2:
        raise ValueError(
            f"Categorical predictor '{series.name}' has {len(levels)} level(s). "
            "Provide at least two observed levels or remove the predictor."
        )
    return levels


def _level_column(variable: str, level: Any, *, reference_coded: bool) -> str:
    marker = "T." if reference_coded else ""
    return f"{variable}[{marker}{level}]"


def baseline_variable(reg: Regression) -> str | None:
    """Return the categorical variable that supplies the equation baseline.

    With an intercept, only a hierarchical term can supply it, and only when
    no earlier categorical already does: the intercept itself is the baseline
    until then, so a treatment-coded term leaves it in place for a later
    hierarchical term. Without an intercept the first categorical supplies it.
    Returns ``None`` when no categorical term is visible yet. Inferred string
    columns stay invisible until fitting attaches a :class:`CategoricalCall`,
    which is the documented data-free exception.
    """
    baseline: str | None = None
    for term in reg.terms:
        call = term.categorical
        if call is None:
            continue
        if baseline is None and (not reg.has_intercept or call.prior == "hierarchical"):
            baseline = call.variable
    return baseline


def keeps_every_level(reg: Regression, call: CategoricalCall) -> bool:
    """Whether this term keeps one coefficient per level.

    That is true when the term supplies the baseline, or when its prior is
    hierarchical (the coefficients are then zero-mean deviations, so no level
    escapes pooling by being the reference).
    """
    return call.variable == baseline_variable(reg) or call.prior == "hierarchical"


def fit_categorical_terms(spec: Spec, data: nw.DataFrame) -> set[str]:
    """Resolve explicit and inferred categorical terms against fitting data.

    The function updates the parsed term objects with immutable level, reference,
    and output-column state. It returns the source column names that were treated
    as categorical.
    """
    pandas_data = data.to_pandas()
    categorical_vars: set[str] = set()
    endogenous = {reg.lhs for reg in spec.regressions}

    for lhs in sorted(endogenous):
        if lhs in pandas_data.columns and _is_categorical_series(pandas_data[lhs]):
            raise NotImplementedError(
                f"Outcome '{lhs}' is a string or categorical column. Categorical "
                "outcomes are not supported: every '~' left-hand side needs a "
                "numeric column. Recode the outcome (for example as a 0/1 "
                "indicator with families={'" + lhs + "': 'bernoulli'}) or use "
                f"'{lhs}' only as a predictor."
            )

    for reg in spec.regressions:
        seen_categorical: set[str] = set()
        # Attach a call to every categorical term before asking which one
        # supplies the baseline. Inferred string columns have no call until
        # this pass, so baseline_variable cannot see them any earlier.
        fitted: list[tuple[Term, tuple[Any, ...]]] = []
        for term in reg.terms:
            if term.interaction_of is not None:
                for variable in term.interaction_of:
                    if variable not in pandas_data.columns:
                        continue
                    series = pandas_data[variable]
                    if not _is_categorical_series(series):
                        _reject_mixed_object_series(series)
                categorical_components = [
                    variable
                    for variable in term.interaction_of
                    if variable in pandas_data.columns
                    and _is_categorical_series(pandas_data[variable])
                ]
                if categorical_components:
                    raise NotImplementedError(
                        f"Interaction '{term.variable}' includes categorical "
                        f"predictor(s) {categorical_components!r}. Categorical "
                        "interactions are not supported yet; use each categorical "
                        "predictor as a standalone term."
                    )
                continue

            explicit = term.categorical is not None
            if term.variable not in pandas_data.columns:
                if explicit:
                    raise ValueError(
                        f"Categorical predictor '{term.variable}' was not found in "
                        f"the data columns. Add a '{term.variable}' column to data "
                        f"or remove C({term.variable}) from the specification."
                    )
                continue
            if not explicit and not _is_categorical_series(pandas_data[term.variable]):
                _reject_mixed_object_series(pandas_data[term.variable])
                continue
            if term.label is not None or term.fixed_value is not None:
                raise ParseError(
                    f"Categorical predictor '{term.variable}' cannot take a "
                    "coefficient prefix because it expands to one coefficient "
                    "per non-reference level. Remove the 'k*' or 'label*' prefix."
                )
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

            if term.variable in seen_categorical:
                raise ValueError(
                    f"Categorical predictor '{term.variable}' appears more than "
                    f"once in equation '{reg.lhs}'. Each categorical predictor "
                    "already expands to one coefficient per level, so list it once."
                )
            seen_categorical.add(term.variable)

            call = term.categorical or CategoricalCall(variable=term.variable)
            levels = _fit_levels(pandas_data[term.variable])
            reference = levels[0] if call.reference is None else call.reference
            if reference not in levels:
                raise ValueError(
                    f"Reference level {reference!r} for categorical predictor "
                    f"'{term.variable}' was not observed. Available levels: "
                    f"{list(levels)!r}."
                )
            # Leave the user-supplied reference on the call until columns are
            # built, so the ineffective-reference warning can still see it.
            term.categorical = replace(call, levels=levels, columns=())
            fitted.append((term, levels))
            categorical_vars.add(term.variable)

        for term, levels in fitted:
            fitted_call = term.categorical
            if fitted_call is None:
                continue
            all_levels = keeps_every_level(reg, fitted_call)
            if all_levels and fitted_call.reference is not None:
                warnings.warn(
                    f"reference={fitted_call.reference!r} for categorical predictor "
                    f"'{term.variable}' in equation '{reg.lhs}' has no effect: "
                    "the term gets one coefficient per level, so there is no "
                    "reference level. Remove the reference= argument.",
                    UserWarning,
                    stacklevel=3,
                )
            reference = (
                levels[0] if fitted_call.reference is None else fitted_call.reference
            )
            represented = (
                levels
                if all_levels
                else tuple(level for level in levels if level != reference)
            )
            columns = tuple(
                _level_column(
                    term.variable,
                    level,
                    reference_coded=not all_levels,
                )
                for level in represented
            )
            term.categorical = replace(
                fitted_call,
                reference=reference,
                levels=levels,
                columns=columns,
            )

    for reg in spec.regressions:
        for term in reg.terms:
            if term.categorical is not None:
                continue
            if term.interaction_of is not None:
                source_vars = set(term.interaction_of)
            elif term.lag_of is not None:
                source_vars = {term.lag_of}
            else:
                source_vars = {term.variable}
            mixed = sorted(source_vars & categorical_vars)
            if mixed:
                raise NotImplementedError(
                    f"Predictor(s) {mixed!r} are used both categorically and "
                    f"continuously in the model specification (including term "
                    f"'{term.variable}' in equation '{reg.lhs}'). Mixed use is "
                    "not supported because prediction and intervention require "
                    "one canonical representation. Create a separate numeric "
                    "column for the continuous effect or use the predictor only "
                    "inside C(...)."
                )

    return categorical_vars


def is_reference_coded(call: CategoricalCall) -> bool:
    """Read whether the fitted columns drop the reference level.

    This is a readout of the columns :func:`keeps_every_level` built, not a
    second coding rule. It is meaningless before fitting, when both columns
    and levels are empty.
    """
    return bool(call.levels) and len(call.columns) < len(call.levels)


def has_population_mean(reg: Regression, call: CategoricalCall) -> bool:
    """Return whether a hierarchical term gets its own free population mean.

    Only the categorical that supplies the baseline can carry ``mu_beta``.
    A later hierarchical term keeps every level, but its coefficients are
    zero-mean deviations, so a free mean would be unidentified.
    """
    return call.prior == "hierarchical" and call.variable == baseline_variable(reg)


def absorbs_intercept(reg: Regression) -> bool:
    """Return whether a categorical term replaces the equation's intercept.

    True when the formula has an intercept but a hierarchical categorical
    supplies the baseline, so the ``Intercept`` column must be left out.
    """
    return reg.has_intercept and baseline_variable(reg) is not None


def coefficient_levels(call: CategoricalCall) -> tuple[Any, ...]:
    """Return the levels represented by a categorical coefficient vector."""
    if not is_reference_coded(call):
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
