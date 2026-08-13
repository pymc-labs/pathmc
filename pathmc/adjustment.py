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
"""Backdoor adjustment model facade around a reduced single-equation PathModel."""

from __future__ import annotations

from typing import Any, Literal

import graphviz
import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from narwhals.stable.v1.typing import IntoFrame

from pathmc.compile import _term_base_vars
from pathmc.graph import GraphInfo
from pathmc.identify import adjustment_sets, is_valid_adjustment_set
from pathmc.interpret import InterpretResult
from pathmc.introspect import build_dag_viz
from pathmc.parse import Spec, parse_spec
from pathmc.priors import default_priors, merge_priors
from pathmc.simulate import EstimandResult

__all__ = ["AdjustmentModel"]

_OUTCOME_PRIOR_SUFFIXES = ("sigma", "nu", "alpha_disp")


def _parse_treatment_outcome_query(query: str) -> tuple[str, str]:
    """Split a ``treatment -> outcome`` query string into node names."""
    for arrow in ("->", "→"):
        if arrow in query:
            parts = [p.strip() for p in query.split(arrow)]
            if len(parts) != 2:
                if len(parts) > 2:
                    nodes = " -> ".join(parts)
                    raise ValueError(
                        f"Path-specific queries with mediators require "
                        f"effect(). Got '{query.strip()}' ({len(parts)} nodes: "
                        f"{nodes}). Pass treatment= and outcome= for a "
                        f"total-effect query, or use PathModel.effect() for "
                        f"path-specific effects."
                    )
                raise ValueError(
                    f"Invalid query '{query.strip()}'. Expected exactly two "
                    f"nodes separated by '->' or '→', e.g. 'X -> Y'."
                )
            if not parts[0] or not parts[1]:
                raise ValueError(
                    f"Invalid query '{query.strip()}'. Both treatment and "
                    f"outcome must be non-empty node names."
                )
            return parts[0], parts[1]
    raise ValueError(
        f"Invalid query '{query.strip()}'. Expected 'treatment -> outcome' "
        f"(or 'treatment → outcome')."
    )


def _resolve_treatment_outcome(
    query: str | None,
    treatment: str | None,
    outcome: str | None,
) -> tuple[str, str]:
    """Resolve treatment and outcome from query string and/or keyword args."""
    from_query: tuple[str, str] | None = None
    if query is not None:
        from_query = _parse_treatment_outcome_query(query)

    if treatment is None and outcome is None:
        if from_query is None:
            raise ValueError(
                "Pass a query string (e.g. 'X -> Y') or both treatment= and outcome=."
            )
        return from_query

    if treatment is None or outcome is None:
        raise ValueError(
            "Pass both treatment= and outcome= when not using a query string."
        )

    if from_query is not None and from_query != (treatment, outcome):
        raise ValueError(
            f"Query '{query}' specifies {from_query[0]} -> {from_query[1]}, "
            f"but treatment='{treatment}' and outcome='{outcome}' were also "
            f"passed. Remove the redundant arguments or make them agree."
        )

    return treatment, outcome


def _select_adjustment_set(
    graph_info: GraphInfo,
    treatment: str,
    outcome: str,
    adjustment_set: set[str] | None,
) -> frozenset[str]:
    """Resolve and validate the backdoor adjustment set."""
    sets = adjustment_sets(graph_info, treatment, outcome)
    if not sets:
        raise ValueError(
            f"No valid backdoor adjustment set for the effect of "
            f"'{treatment}' on '{outcome}'. The effect is not identifiable "
            f"via the backdoor criterion. Call adjustment_sets() or "
            f"is_identifiable() on the structural model, or revise the "
            f"DAG. Do not fit an associational model when identification "
            f"fails."
        )

    if adjustment_set is None:
        if len(sets) > 1:
            formatted = ", ".join(
                f"{{{', '.join(sorted(s))}}}" if s else "{}" for s in sets
            )
            raise ValueError(
                f"Several minimal backdoor adjustment sets exist for "
                f"'{treatment}' -> '{outcome}': {formatted}. Pass "
                f"adjustment_set= explicitly — pathmc does not choose among "
                f"valid sets automatically."
            )
        return frozenset(sets[0])

    z = set(adjustment_set)
    is_valid_adjustment_set(graph_info, treatment, outcome, z)
    return frozenset(z)


def _default_formula(
    outcome: str, treatment: str, adjustment_set: frozenset[str]
) -> str:
    """Build the default reduced regression formula."""
    predictors = [treatment] + sorted(adjustment_set)
    return f"{outcome} ~ " + " + ".join(predictors)


def _validate_reduced_spec(
    spec: Spec,
    outcome: str,
    treatment: str,
    adjustment_set: frozenset[str],
    graph_info: GraphInfo,
    data: nw.DataFrame | None,
) -> None:
    """Validate a user-supplied reduced formula before fitting."""
    if spec.residual_covs:
        raise ValueError(
            "Reduced adjustment formulas cannot include residual covariances "
            "(~~). Pass a single regression equation only."
        )
    if spec.defined_params:
        raise ValueError(
            "Reduced adjustment formulas cannot include defined parameters "
            "(:=). Pass a single regression equation only."
        )
    if len(spec.regressions) != 1:
        raise ValueError(
            f"Reduced adjustment formula must be a single regression. "
            f"Got {len(spec.regressions)} equation(s). Pass one equation "
            f"like '{outcome} ~ {treatment} + Z'."
        )

    reg = spec.regressions[0]
    if reg.lhs != outcome:
        raise ValueError(
            f"Reduced formula LHS must be the outcome '{outcome}'. Got '{reg.lhs}'."
        )

    for term in reg.terms:
        if term.lag_of is not None:
            raise ValueError(
                "lag() terms are not supported in adjustment model formulas. "
                "Use the structural PathModel for panel/lagged models."
            )
        if term.hsgp is not None:
            raise ValueError(
                "hsgp() terms are not supported in adjustment model formulas. "
                "Use the structural PathModel for HSGP models."
            )

    predictors: set[str] = set()
    for term in reg.terms:
        predictors.update(_term_base_vars(term))

    missing_adj = adjustment_set - predictors
    if missing_adj:
        raise ValueError(
            f"Formula must include every adjustment-set variable. Missing: "
            f"{sorted(missing_adj)}. Add them to formula= or remove them "
            f"from adjustment_set=."
        )

    if treatment not in predictors:
        raise ValueError(
            f"Formula must include the treatment '{treatment}'. Add "
            f"'{treatment}' to formula=."
        )

    dag = graph_info.contemporaneous_dag
    dag_nodes = set(dag.nodes)
    extra_dag_vars = predictors - adjustment_set - {treatment}
    for var in sorted(extra_dag_vars):
        if var in dag_nodes:
            is_valid_adjustment_set(
                graph_info,
                treatment,
                outcome,
                set(adjustment_set | {var}),
            )
        elif data is not None and var not in data.columns:
            raise ValueError(
                f"Formula variable '{var}' is not in the DAG and not found "
                f"in data columns. Available columns: "
                f"{sorted(data.columns)}."
            )


def _inherit_outcome_priors(
    parent_priors: dict[str, Any],
    construction_priors: dict[str, Any] | None,
    outcome: str,
) -> dict[str, Any]:
    """Copy outcome dispersion priors from the parent; reject beta overrides."""
    beta_key = f"beta_{outcome}"
    if construction_priors and beta_key in construction_priors:
        raise ValueError(
            f"The structural model has a custom prior on '{beta_key}'. "
            f"Adjustment models do not inherit coefficient priors. Pass "
            f"priors= on adjustment_model() for the reduced equation."
        )

    inherited: dict[str, Any] = {}
    for suffix in _OUTCOME_PRIOR_SUFFIXES:
        key = f"{suffix}_{outcome}"
        if key in parent_priors:
            inherited[key] = parent_priors[key]
    return inherited


class AdjustmentModel:
    """Facade for a backdoor-adjusted single-equation outcome model.

    Wraps a reduced ``PathModel`` built from ``Y ~ treatment + Z`` while
    retaining the original causal DAG for identification context.

    Parameters
    ----------
    treatment : str
        Treatment variable in the designated query.
    outcome : str
        Outcome variable in the designated query.
    adjustment_set : frozenset[str]
        Validated backdoor adjustment set used to build the formula.
    formula : str
        Reduced regression formula string.
    graph_info : GraphInfo
        Original causal DAG (not the reduced estimation graph).
    outcome_model : PathModel
        Inner data-bound model for the reduced equation.
    parent_spec : Spec
        Original structural specification for DAG visualization.
    parent_families : dict[str, str]
        Outcome family inherited from the structural parent.
    """

    def __init__(
        self,
        *,
        treatment: str,
        outcome: str,
        adjustment_set: frozenset[str],
        formula: str,
        graph_info: GraphInfo,
        outcome_model: Any,
        parent_spec: Spec,
        parent_families: dict[str, str],
    ) -> None:
        self._treatment = treatment
        self._outcome = outcome
        self._adjustment_set = adjustment_set
        self._formula = formula
        self._graph_info = graph_info
        self._outcome_model = outcome_model
        self._parent_spec = parent_spec
        self._parent_families = parent_families

    @classmethod
    def from_path_model(
        cls,
        parent: Any,
        *,
        query: str | None = None,
        treatment: str | None = None,
        outcome: str | None = None,
        adjustment_set: set[str] | None = None,
        formula: str | None = None,
        data: IntoFrame | None = None,
        families: dict[str, str] | None = None,
        priors: dict[str, Any] | None = None,
    ) -> AdjustmentModel:
        """Build an adjustment model from a structural :class:`PathModel`."""
        from pathmc._model import model as build_model

        if parent._panel_info is not None:
            raise NotImplementedError(
                "Panel adjustment models are not supported. Use the "
                "structural PathModel for panel data."
            )

        treatment_name, outcome_name = _resolve_treatment_outcome(
            query, treatment, outcome
        )
        graph_info = parent._graph_info

        selected_set = _select_adjustment_set(
            graph_info,
            treatment_name,
            outcome_name,
            adjustment_set,
        )

        if data is not None:
            nw_data = nw.from_native(data, eager_only=True)
        elif parent._data is not None:
            nw_data = parent._data
        else:
            raise ValueError(
                "data= is required when the structural model has no data. "
                "Pass data= to adjustment_model(), or create a data-bound "
                "structural model first."
            )

        if formula is None:
            formula_str = _default_formula(outcome_name, treatment_name, selected_set)
            reduced_spec = parse_spec(formula_str)
        else:
            formula_str = formula
            reduced_spec = parse_spec(formula)
            _validate_reduced_spec(
                reduced_spec,
                outcome_name,
                treatment_name,
                selected_set,
                graph_info,
                nw_data,
            )

        outcome_families: dict[str, str] = {}
        parent_family = parent._families.get(outcome_name)
        if parent_family is not None:
            outcome_families[outcome_name] = parent_family
        if families is not None:
            outcome_families.update(families)

        construction_priors = (
            parent._construction.get("priors") if parent._construction else None
        )
        inherited_priors = _inherit_outcome_priors(
            parent._priors,
            construction_priors,
            outcome_name,
        )
        reduced_defaults = default_priors(
            reduced_spec,
            families=outcome_families or None,
        )
        merged_priors = merge_priors(reduced_defaults, inherited_priors)
        if priors is not None:
            merged_priors = merge_priors(merged_priors, priors)

        inner = build_model(
            formula_str,
            data=nw_data.to_native(),
            families=outcome_families or None,
            priors=merged_priors,
        )

        return cls(
            treatment=treatment_name,
            outcome=outcome_name,
            adjustment_set=selected_set,
            formula=formula_str,
            graph_info=graph_info,
            outcome_model=inner,
            parent_spec=parent._spec,
            parent_families=dict(parent._families),
        )

    @property
    def treatment(self) -> str:
        """Treatment variable in the designated query."""
        return self._treatment

    @property
    def outcome(self) -> str:
        """Outcome variable in the designated query."""
        return self._outcome

    @property
    def adjustment_set(self) -> frozenset[str]:
        """Validated backdoor adjustment set."""
        return self._adjustment_set

    @property
    def formula(self) -> str:
        """Reduced regression formula string."""
        return self._formula

    @property
    def outcome_model(self) -> Any:
        """Inner :class:`PathModel` for the reduced equation."""
        return self._outcome_model

    @property
    def pymc_model(self) -> pm.Model:
        """Compiled PyMC model of the inner outcome regression."""
        return self._outcome_model.pymc_model

    @property
    def graph_info(self) -> GraphInfo:
        """Original causal DAG (not the reduced estimation graph)."""
        return self._graph_info

    @property
    def idata(self) -> xr.DataTree | None:
        """Posterior samples from the inner model, if fitted."""
        return self._outcome_model._idata

    def graph(self) -> graphviz.Digraph:
        """DAG of the original causal system (not the reduced equation)."""
        return build_dag_viz(
            self._parent_spec,
            self._graph_info,
            families=self._parent_families,
        )

    def fit(self, **kwargs: Any) -> xr.DataTree:
        """Fit the inner outcome model via MCMC."""
        return self._outcome_model.fit(**kwargs)

    def _resolve_query(
        self,
        outcome: str | None,
        treatment: str | None,
    ) -> tuple[str, str]:
        resolved_outcome = outcome if outcome is not None else self._outcome
        resolved_treatment = treatment if treatment is not None else self._treatment
        if resolved_outcome != self._outcome or resolved_treatment != self._treatment:
            raise ValueError(
                f"This AdjustmentModel answers {self._treatment} -> "
                f"{self._outcome}. Got outcome='{resolved_outcome}', "
                f"treatment='{resolved_treatment}'. Pass the designated query "
                f"variables or omit them."
            )
        return resolved_outcome, resolved_treatment

    def _resolve_outcome(self, outcome: str | None) -> str:
        """Resolve outcome, defaulting to the designated query outcome."""
        resolved_outcome = outcome if outcome is not None else self._outcome
        if resolved_outcome != self._outcome:
            raise ValueError(
                f"This AdjustmentModel answers {self._treatment} -> "
                f"{self._outcome}. Got outcome='{resolved_outcome}'. "
                f"Pass the designated outcome or omit outcome=."
            )
        return resolved_outcome

    def _formula_predictors(self) -> set[str]:
        """Predictor names in the reduced outcome regression."""
        reg = self._outcome_model._spec.regressions[0]
        predictors: set[str] = set()
        for term in reg.terms:
            predictors.update(_term_base_vars(term))
        return predictors

    def _validate_formula_variable(self, name: str, *, role: str) -> None:
        """Raise when a query variable is absent from the reduced formula."""
        predictors = self._formula_predictors()
        if name not in predictors:
            raise ValueError(
                f"{role} '{name}' is not in the reduced outcome formula "
                f"'{self._formula}'. Available predictors: "
                f"{sorted(predictors)}."
            )

    def _stamp(
        self,
        result: EstimandResult | InterpretResult,
        varied_variable: str | None = None,
    ) -> EstimandResult | InterpretResult:
        """Stamp regression-adjustment metadata on an interpret or estimand result."""
        interventional = result.interventional
        causal = interventional and varied_variable == self._treatment
        ds = result.dataset.copy()
        ds.attrs["adjustment_set"] = tuple(sorted(self._adjustment_set))

        if isinstance(result, EstimandResult):
            return EstimandResult(
                ds=ds,
                outcome=result.outcome,
                treatment=result.treatment,
                estimand=result._estimand,
                estimator="regression_adjustment",
                causal=causal,
                interventional=interventional,
                identifiable=True,
            )

        return InterpretResult(
            ds=ds,
            outcome=result.outcome,
            quantity=result.quantity,
            variable=result.variable,
            estimator="regression_adjustment",
            causal=causal,
            interventional=interventional,
            identifiable=True,
        )

    def predictions(
        self,
        outcome: str | None = None,
        *,
        set: dict[str, float | np.ndarray] | None = None,
        newdata: IntoFrame | None = None,
    ) -> InterpretResult:
        """Response-mean predictions on the reduced outcome model.

        Defaults ``outcome`` to the designated query outcome. Without ``set``,
        predictions are associational. ``causal`` is ``True`` only when
        ``set`` intervenes on the designated treatment alone.
        """
        outcome_name = self._resolve_outcome(outcome)
        set_dict = None if set is None else dict(set)
        if set_dict is not None:
            for key in set_dict:
                self._validate_formula_variable(key, role="set key")
        result = self._outcome_model.predictions(
            outcome_name, set=set_dict, newdata=newdata
        )
        varied = next(iter(set_dict)) if set_dict and len(set_dict) == 1 else None
        stamped = self._stamp(result, varied_variable=varied)
        assert isinstance(stamped, InterpretResult)
        return stamped

    def comparisons(
        self,
        outcome: str | None = None,
        variable: str | None = None,
        *,
        contrast: tuple[float, float] = (0.0, 1.0),
        comparison: Literal["diff", "ratio", "lift"] = "diff",
        conditional: dict[str, float] | None = None,
        average_by: Literal["all"] | None = "all",
    ) -> EstimandResult | InterpretResult:
        """Interventional contrasts on the reduced outcome model.

        Defaults ``outcome`` and ``variable`` to the designated query.
        ``causal`` is ``True`` only when ``variable`` is the designated
        treatment.
        """
        outcome_name = self._resolve_outcome(outcome)
        variable_name = variable if variable is not None else self._treatment
        self._validate_formula_variable(variable_name, role="variable")
        result = self._outcome_model.comparisons(
            outcome_name,
            variable_name,
            contrast=contrast,
            comparison=comparison,
            conditional=conditional,
            average_by=average_by,
        )
        return self._stamp(result, varied_variable=variable_name)

    def slopes(
        self,
        outcome: str | None = None,
        wrt: str | None = None,
        *,
        slope: Literal["dydx", "eyex", "eydx", "dyex"] = "dydx",
        eps: float = 1e-4,
        conditional: dict[str, float] | None = None,
        average_by: Literal["all"] | None = "all",
    ) -> EstimandResult | InterpretResult:
        """Finite-difference slopes on the reduced outcome model.

        Defaults ``outcome`` and ``wrt`` to the designated query. ``causal`` is
        ``True`` only when ``wrt`` is the designated treatment.
        """
        outcome_name = self._resolve_outcome(outcome)
        wrt_name = wrt if wrt is not None else self._treatment
        self._validate_formula_variable(wrt_name, role="wrt")
        result = self._outcome_model.slopes(
            outcome_name,
            wrt_name,
            slope=slope,
            eps=eps,
            conditional=conditional,
            average_by=average_by,
        )
        return self._stamp(result, varied_variable=wrt_name)

    def datagrid(self, **cols: list[float] | list[int]) -> pd.DataFrame:
        """Build a covariate grid from the inner outcome model's data."""
        return self._outcome_model.datagrid(**cols)

    def ate(
        self,
        outcome: str | None = None,
        treatment: str | None = None,
        values: tuple[float, float] = (0.0, 1.0),
        **do_kwargs: Any,
    ) -> EstimandResult:
        """Average treatment effect via the inner model's ``do()`` path."""
        outcome_name, treatment_name = self._resolve_query(outcome, treatment)
        result = self._outcome_model.ate(
            outcome_name, treatment_name, values=values, **do_kwargs
        )
        stamped = self._stamp(result, varied_variable=treatment_name)
        assert isinstance(stamped, EstimandResult)
        return stamped

    def cate(
        self,
        outcome: str | None = None,
        treatment: str | None = None,
        values: tuple[float, float] = (0.0, 1.0),
        condition: dict[str, float] | None = None,
        **do_kwargs: Any,
    ) -> EstimandResult:
        """Conditional average treatment effect."""
        outcome_name, treatment_name = self._resolve_query(outcome, treatment)
        result = self._outcome_model.cate(
            outcome_name,
            treatment_name,
            values=values,
            condition=condition,
            **do_kwargs,
        )
        stamped = self._stamp(result, varied_variable=treatment_name)
        assert isinstance(stamped, EstimandResult)
        return stamped

    def att(
        self,
        outcome: str | None = None,
        treatment: str | None = None,
        values: tuple[float, float] = (0.0, 1.0),
        treated_value: float = 1.0,
        kind: str = "mean",
    ) -> EstimandResult:
        """Average treatment effect on the treated."""
        outcome_name, treatment_name = self._resolve_query(outcome, treatment)
        result = self._outcome_model.att(
            outcome_name,
            treatment_name,
            values=values,
            treated_value=treated_value,
            kind=kind,
        )
        stamped = self._stamp(result, varied_variable=treatment_name)
        assert isinstance(stamped, EstimandResult)
        return stamped

    def atu(
        self,
        outcome: str | None = None,
        treatment: str | None = None,
        values: tuple[float, float] = (0.0, 1.0),
        untreated_value: float = 0.0,
        kind: str = "mean",
    ) -> EstimandResult:
        """Average treatment effect on the untreated."""
        outcome_name, treatment_name = self._resolve_query(outcome, treatment)
        result = self._outcome_model.atu(
            outcome_name,
            treatment_name,
            values=values,
            untreated_value=untreated_value,
            kind=kind,
        )
        stamped = self._stamp(result, varied_variable=treatment_name)
        assert isinstance(stamped, EstimandResult)
        return stamped
