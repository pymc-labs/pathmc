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
"""History-dependent treatment policies for panel ``do()``.

A policy is a typed specification, not a Python callback. Rules may read
``current(name)`` and ``previous(name)`` for the same unit, combine them with
arithmetic and comparisons, and assign a finite numeric action through
``where``. There is no future-time accessor and no cross-unit lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal as Kind, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from pathmc.graph import GraphInfo
    from pathmc.parse import Spec

__all__ = ["TimeSchedule", "TreatmentPolicy", "current", "previous", "where"]

_TRUTH_MESSAGE = (
    "Policy expressions do not support Python truth tests (if, and, or, not). "
    "Use &, |, ~, comparisons, and where()."
)

_POLICY_REQUIRES_SCAN = (
    "policy= requires a scan-compiled panel model with lag() terms. "
    "Use do(set=) for a cross-sectional intervention."
)


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise TypeError(
            f"{where} must be a finite number, not {type(value).__name__}. "
            "Pass a float, or build the value with current(), previous(), "
            "and where()."
        )
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{where} must be finite, got {value}. Pass a finite number.")
    return number


class _Expr:
    """Base for policy expressions. Rejects Python truth tests."""

    def __bool__(self) -> bool:
        raise TypeError(_TRUTH_MESSAGE)

    def dependencies(self) -> set[tuple[str, str]]:
        raise NotImplementedError

    def lower(self, ctx: _LowerCtx) -> Any:
        raise NotImplementedError

    def __add__(self, other: Any) -> _Expr:
        return _binop("+", self, other)

    def __radd__(self, other: Any) -> _Expr:
        return _binop("+", other, self)

    def __sub__(self, other: Any) -> _Expr:
        return _binop("-", self, other)

    def __rsub__(self, other: Any) -> _Expr:
        return _binop("-", other, self)

    def __mul__(self, other: Any) -> _Expr:
        return _binop("*", self, other)

    def __rmul__(self, other: Any) -> _Expr:
        return _binop("*", other, self)

    def __truediv__(self, other: Any) -> _Expr:
        return _binop("/", self, other)

    def __rtruediv__(self, other: Any) -> _Expr:
        return _binop("/", other, self)

    def __lt__(self, other: Any) -> _Expr:
        return _cmp("<", self, other)

    def __le__(self, other: Any) -> _Expr:
        return _cmp("<=", self, other)

    def __gt__(self, other: Any) -> _Expr:
        return _cmp(">", self, other)

    def __ge__(self, other: Any) -> _Expr:
        return _cmp(">=", self, other)

    def __eq__(self, other: object) -> _Expr:  # type: ignore[override]
        return _cmp("==", self, other)

    def __ne__(self, other: object) -> _Expr:  # type: ignore[override]
        return _cmp("!=", self, other)

    def __and__(self, other: Any) -> _Expr:
        return _boolop("&", self, other)

    def __rand__(self, other: Any) -> _Expr:
        return _boolop("&", other, self)

    def __or__(self, other: Any) -> _Expr:
        return _boolop("|", self, other)

    def __ror__(self, other: Any) -> _Expr:
        return _boolop("|", other, self)

    def __invert__(self) -> _Expr:
        return Not(self)


def _wrap(value: Any, where: str) -> _Expr:
    if isinstance(value, _Expr):
        return value
    if isinstance(value, TimeSchedule):
        raise TypeError(
            f"{where} cannot contain a TimeSchedule. Use a number, "
            "current(), previous(), or where(), or pass TimeSchedule as "
            "the policy rule itself."
        )
    return Literal(_as_number(value, where))


def _binop(op: str, left: Any, right: Any) -> _Expr:
    return BinOp(op, _wrap(left, f"left of {op}"), _wrap(right, f"right of {op}"))


def _cmp(op: str, left: Any, right: Any) -> _Expr:
    return Comparison(op, _wrap(left, f"left of {op}"), _wrap(right, f"right of {op}"))


def _boolop(op: str, left: Any, right: Any) -> _Expr:
    return BoolOp(op, _wrap(left, f"left of {op}"), _wrap(right, f"right of {op}"))


@dataclass(frozen=True, eq=False)
class Literal(_Expr):
    value: float

    def dependencies(self) -> set[tuple[str, str]]:
        return set()

    def lower(self, ctx: _LowerCtx) -> Any:
        del ctx
        return np.asarray(self.value, dtype=float)


@dataclass(frozen=True, eq=False)
class Ref(_Expr):
    name: str
    timing: str

    def dependencies(self) -> set[tuple[str, str]]:
        return {(self.timing, self.name)}

    def lower(self, ctx: _LowerCtx) -> Any:
        return ctx.reference(self.timing, self.name)


@dataclass(frozen=True, eq=False)
class BinOp(_Expr):
    op: str
    left: _Expr
    right: _Expr

    def dependencies(self) -> set[tuple[str, str]]:
        return self.left.dependencies() | self.right.dependencies()

    def lower(self, ctx: _LowerCtx) -> Any:
        import pytensor.tensor as pt

        a = self.left.lower(ctx)
        b = self.right.lower(ctx)
        ops = {"+": pt.add, "-": pt.sub, "*": pt.mul, "/": pt.true_div}
        return ops[self.op](a, b)


@dataclass(frozen=True, eq=False)
class Comparison(_Expr):
    op: str
    left: _Expr
    right: _Expr

    def dependencies(self) -> set[tuple[str, str]]:
        return self.left.dependencies() | self.right.dependencies()

    def lower(self, ctx: _LowerCtx) -> Any:
        import pytensor.tensor as pt

        a = self.left.lower(ctx)
        b = self.right.lower(ctx)
        ops = {
            "<": pt.lt,
            "<=": pt.le,
            ">": pt.gt,
            ">=": pt.ge,
            "==": pt.eq,
            "!=": pt.neq,
        }
        return ops[self.op](a, b)


@dataclass(frozen=True, eq=False)
class BoolOp(_Expr):
    op: str
    left: _Expr
    right: _Expr

    def dependencies(self) -> set[tuple[str, str]]:
        return self.left.dependencies() | self.right.dependencies()

    def lower(self, ctx: _LowerCtx) -> Any:
        import pytensor.tensor as pt

        a = self.left.lower(ctx)
        b = self.right.lower(ctx)
        if self.op == "&":
            return pt.and_(a, b)
        return pt.or_(a, b)


@dataclass(frozen=True, eq=False)
class Not(_Expr):
    expr: _Expr

    def dependencies(self) -> set[tuple[str, str]]:
        return self.expr.dependencies()

    def lower(self, ctx: _LowerCtx) -> Any:
        import pytensor.tensor as pt

        return pt.invert(self.expr.lower(ctx))


@dataclass(frozen=True, eq=False)
class Where(_Expr):
    condition: _Expr
    yes: _Expr
    no: _Expr

    def dependencies(self) -> set[tuple[str, str]]:
        return (
            self.condition.dependencies()
            | self.yes.dependencies()
            | self.no.dependencies()
        )

    def lower(self, ctx: _LowerCtx) -> Any:
        import pytensor.tensor as pt

        return pt.switch(
            self.condition.lower(ctx),
            self.yes.lower(ctx),
            self.no.lower(ctx),
        )


@dataclass(frozen=True)
class TimeSchedule:
    """One finite action per fitted time, copied and frozen at construction.

    Length is checked against the fitted time grid when the policy is planned,
    not here: construction does not know the panel.
    """

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.values, (str, bytes)) or not isinstance(self.values, tuple):
            object.__setattr__(self, "values", tuple(self.values))
        cleaned: list[float] = []
        for index, value in enumerate(self.values):
            cleaned.append(_as_number(value, f"TimeSchedule value at index {index}"))
        if not cleaned:
            raise ValueError(
                "TimeSchedule requires one finite value per fitted time. "
                "Pass a non-empty sequence."
            )
        object.__setattr__(self, "values", tuple(cleaned))


@dataclass(frozen=True)
class TreatmentPolicy:
    """Immutable rule that assigns one numeric variable during panel simulation."""

    name: str
    variable: str
    rule: _Expr | TimeSchedule | float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                "TreatmentPolicy name must be a non-empty string. "
                "Pass name='responsive' or another label."
            )
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError(
                "TreatmentPolicy variable must be a non-empty string naming "
                "the intervened column. Pass variable='water'."
            )
        if isinstance(self.rule, bool) or (
            not isinstance(self.rule, (_Expr, TimeSchedule))
            and not isinstance(self.rule, (int, float, np.floating))
        ):
            raise TypeError(
                "TreatmentPolicy rule must be a finite number, a TimeSchedule, "
                "or an expression built from current(), previous(), and where(). "
                "Callbacks are not supported."
            )
        if not isinstance(self.rule, (_Expr, TimeSchedule)):
            object.__setattr__(self, "rule", _as_number(self.rule, "rule"))


def current(name: str) -> Ref:
    """Read this unit's simulated value at the decision time.

    ``current`` declares that the measurement is available before the action.
    An outcome caused by the current action cannot be a current input.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "current() needs the variable name as a string. Example: current('stress')."
        )
    return Ref(name, "current")


def previous(name: str) -> Ref:
    """Read this unit's simulated value from the preceding period.

    At the first fitted time this is the explicit pre-sample state. A previous
    treatment reference sees the last assigned action, not the observed value.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "previous() needs the variable name as a string. "
            "Example: previous('water')."
        )
    return Ref(name, "previous")


def where(condition: Any, yes: Any, no: Any) -> Where:
    """Assign ``yes`` where ``condition`` is true and ``no`` otherwise.

    Equality follows the comparison operator exactly: ``>`` is false at the
    threshold, ``>=`` is true.
    """
    return Where(
        _wrap(condition, "where() condition"),
        _wrap(yes, "where() yes"),
        _wrap(no, "where() no"),
    )


@dataclass(frozen=True)
class PolicyPlan:
    """Within-period order and carry requirements for one policy query."""

    policy: TreatmentPolicy
    target: str
    current_deps: tuple[str, ...]
    previous_deps: tuple[str, ...]
    before: tuple[str, ...]
    after: tuple[str, ...]
    kind: Kind["schedule", "expression"]
    extra_carries: tuple[str, ...]
    schedule: tuple[float, ...] | None
    time_values: tuple[Any, ...]


@dataclass(frozen=True)
class PolicyScanSlots:
    """Extra scan slots implied by a policy plan. Empty when *plan* is None.

    Names are candidates. The compiler applies data-column and endogenous
    filters; this object does not see the fitted frame.
    """

    extra_exog: tuple[str, ...] = ()
    extra_carries: tuple[str, ...] = ()
    schedule_name: str | None = None
    drop_intervene: frozenset[str] = frozenset()
    drop_carry_mu: frozenset[str] = frozenset()
    action_name: str | None = None
    zero_init: frozenset[str] = frozenset()


def policy_scan_slots(plan: PolicyPlan | None) -> PolicyScanSlots:
    """Map a policy plan to extra scan slots. ``None`` returns an empty layout.

    The schedule is its own sequence name, never an exogenous column. Carry
    and intervention drops are names, not offsets.
    """
    if plan is None:
        return PolicyScanSlots()
    extra_exog = tuple(
        sorted(set(plan.current_deps) | set(plan.previous_deps) | {plan.target})
    )
    return PolicyScanSlots(
        extra_exog=extra_exog,
        extra_carries=plan.extra_carries,
        schedule_name="_policy_schedule" if plan.kind == "schedule" else None,
        drop_intervene=frozenset({plan.target}),
        drop_carry_mu=frozenset({plan.target}),
        action_name="_policy_action",
        zero_init=frozenset(plan.extra_carries),
    )


def _rule_deps(rule: _Expr | TimeSchedule | float) -> set[tuple[str, str]]:
    if isinstance(rule, _Expr):
        return rule.dependencies()
    return set()


def plan_policy(
    policy: TreatmentPolicy,
    graph: GraphInfo,
    *,
    names: set[str],
    time_values: Sequence[Any],
) -> PolicyPlan:
    """Build the query order. Previous edges affect carry, not cycles.

    Incoming structural edges of the target are removed. Current dependencies
    are added. A cycle, including a self-read, is rejected.
    """
    import networkx as nx

    if policy.variable not in names:
        raise ValueError(
            f"Policy {policy.name!r} targets unknown variable "
            f"{policy.variable!r}. Use a fitted endogenous or exogenous "
            f"column. Available names: {sorted(names)}."
        )
    times = tuple(time_values)
    schedule: tuple[float, ...] | None = None
    if isinstance(policy.rule, TimeSchedule):
        if len(policy.rule.values) != len(times):
            raise ValueError(
                f"TimeSchedule for policy {policy.name!r} has "
                f"{len(policy.rule.values)} value(s), but the fitted grid has "
                f"{len(times)} time coordinate(s). Pass one finite value per "
                "fitted time."
            )
        schedule = policy.rule.values
    elif isinstance(policy.rule, float):
        schedule = tuple(policy.rule for _ in times)

    unknown = sorted({
        name for _timing, name in _rule_deps(policy.rule) if name not in names
    })
    if unknown:
        raise ValueError(
            f"Policy {policy.name!r} references unknown name(s) {unknown}. "
            "Use current() or previous() on a fitted variable. "
            f"Available names: {sorted(names)}."
        )
    current_deps = tuple(
        sorted(name for timing, name in _rule_deps(policy.rule) if timing == "current")
    )
    previous_deps = tuple(
        sorted(name for timing, name in _rule_deps(policy.rule) if timing == "previous")
    )
    if policy.variable in current_deps:
        raise ValueError(
            f"Policy {policy.name!r} reads current({policy.variable!r}) to set "
            f"{policy.variable!r}. That is a self-read. Use previous("
            f"{policy.variable!r}) or drop the current reference."
        )

    intervened = graph.contemporaneous_dag.copy()
    for pred in list(intervened.predecessors(policy.variable)):
        intervened.remove_edge(pred, policy.variable)
    for dep in current_deps:
        intervened.add_edge(dep, policy.variable)
    if not nx.is_directed_acyclic_graph(intervened):
        raise ValueError(
            f"Policy {policy.name!r} makes the within-period graph cyclic: "
            f"current inputs {list(current_deps)} cannot be known before "
            f"{policy.variable!r} is assigned. An outcome caused by the current "
            "action cannot choose that action. Use previous() for those "
            "measurements, or remove the edge."
        )
    descendants = {
        node
        for node in nx.descendants(intervened, policy.variable)
        if node in graph.endogenous and node != policy.variable
    }
    endogenous_order = [
        node for node in graph.topological_order if node in graph.endogenous
    ]
    before = [
        node
        for node in endogenous_order
        if node not in descendants and node != policy.variable
    ]
    after = [node for node in endogenous_order if node in descendants]

    kind: Kind["schedule", "expression"] = (
        "schedule" if schedule is not None else "expression"
    )
    endogenous = set(graph.endogenous)
    extra_carries = tuple(
        sorted(
            name for name in (*previous_deps, policy.variable) if name not in endogenous
        )
    )
    return PolicyPlan(
        policy=policy,
        target=policy.variable,
        current_deps=current_deps,
        previous_deps=previous_deps,
        before=tuple(before),
        after=tuple(after),
        kind=kind,
        extra_carries=extra_carries,
        schedule=schedule,
        time_values=times,
    )


def reject_unsupported_policy_model(
    spec: Spec,
    families: Mapping[str, str],
    *,
    latent: set[str],
    scan: bool,
) -> None:
    """Reject v1-unsupported structure before a policy query compiles."""
    if not scan:
        raise ValueError(_POLICY_REQUIRES_SCAN)
    problems: list[str] = []
    if latent:
        problems.append(f"latent variables {sorted(latent)}")
    bad_families = sorted(
        f"{var}={family}" for var, family in families.items() if family != "gaussian"
    )
    if bad_families:
        problems.append(f"non-Gaussian families ({', '.join(bad_families)})")
    if spec.residual_covs:
        problems.append("residual covariance (~~)")
    for reg in spec.regressions:
        for term in reg.terms:
            if term.categorical is not None:
                problems.append(f"categorical term {term.variable!r}")
            if term.basis is not None:
                problems.append(f"basis/HSGP term {term.variable!r}")
            if term.transform is not None:
                problems.append(f"transform/adstock term {term.variable!r}")
    if problems:
        raise ValueError(
            "policy= does not support "
            + "; ".join(problems)
            + " in v1. Remove those features or evaluate a supported Gaussian "
            "lag model. pathmc will not fall back to a less-correct executor."
        )


class _LowerCtx:
    def __init__(
        self,
        *,
        current_tensors: Mapping[str, Any],
        previous_tensors: Mapping[str, Any],
        factor_for: Any,
    ) -> None:
        self._current = current_tensors
        self._previous = previous_tensors
        self._factor_for = factor_for

    def reference(self, timing: str, name: str) -> Any:
        import pytensor.tensor as pt

        source = self._current if timing == "current" else self._previous
        if name not in source:
            raise ValueError(
                f"Policy evaluation cannot read {timing}({name!r}); "
                "that value is not in the step context. This is an internal error."
            )
        raw = source[name]
        factor = self._factor_for(name)
        if factor is None:
            return raw
        return pt.as_tensor_variable(raw) * pt.as_tensor_variable(factor)


def policy_unit_factor(
    scaling: Any,
    name: str,
    unit_labels: Sequence[Any],
    unit_col: str,
) -> np.ndarray | None:
    """Per-unit divisor from the scaler, or None when the column is unscaled.

    ``None`` means the scaler has no factor for *name*. A column the scaler
    says is scaled either returns that factor or raises; a missing lookup is
    not treated as identity.
    """
    if scaling is None or not scaling.has_factor(name):
        return None
    try:
        return scaling.unit_divisors(name, unit_labels, unit_col)
    except ValueError as exc:
        raise ValueError(
            f"Cannot align the policy scale for {name!r} to the panel units. {exc} "
            "Refit with scaling dims that include the panel unit, or drop scaling "
            "for this query."
        ) from exc


def lower_action(
    plan: PolicyPlan,
    *,
    schedule_t: Any,
    current_tensors: Mapping[str, Any],
    previous_tensors: Mapping[str, Any],
    scaling: Any,
    unit_labels: Sequence[Any],
    unit_col: str,
    n_units: int,
) -> Any:
    """Return the assigned action in model units, shape ``(n_units,)``.

    Context tensors are model units. The rule sees business units
    (``model * scale``). The returned action is divided by the target scale
    once. A schedule is the scan sequence for this step, not a time index.
    """
    import pytensor.tensor as pt

    def factor_for(name: str) -> np.ndarray | None:
        return policy_unit_factor(scaling, name, unit_labels, unit_col)

    ctx = _LowerCtx(
        current_tensors=current_tensors,
        previous_tensors=previous_tensors,
        factor_for=factor_for,
    )
    if plan.kind == "schedule":
        if schedule_t is None:
            raise RuntimeError(
                "A schedule policy has no scan sequence. This is an internal error."
            )
        business = schedule_t
    else:
        rule = plan.policy.rule
        assert isinstance(rule, _Expr)
        business = rule.lower(ctx)
    business = pt.broadcast_to(pt.as_tensor_variable(business), (n_units,)).astype(
        "float64"
    )
    factor = factor_for(plan.target)
    if factor is None:
        return business
    return business / pt.as_tensor_variable(np.asarray(factor, dtype=float))
