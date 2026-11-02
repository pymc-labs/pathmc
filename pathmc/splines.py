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
"""B-spline, natural cubic spline, and P-spline basis terms.

The compiler only sees :meth:`Basis.contribution`. Knot placement, centering,
and the random-walk penalty live here so ``compile.py`` has no spline branch.
Knots and column means are frozen in :class:`SplineState` and replayed by
``predict()`` and ``do()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.interpolate import BSpline

from pathmc.basis import Basis, BasisCapabilities, Call, register_basis
from pathmc.parse import BasisCall
from pathmc.spline_policy import (
    DEFAULT_DEGREE,
    DEFAULT_ORDER,
    SPLINE_NAMES,
    implied_df,
    spline_call_problem,
)

NULL_SPACE_SCALE = 10.0
UNPENALIZED_SIGMA = 10.0
DEFAULT_TAU_SIGMA = 2.0


@dataclass(frozen=True)
class SplineState:
    """Frozen knot placement and centering for one spline call.

    ``projector`` is the natural-spline null-space basis. It is stored, not
    recomputed, because a QR/SVD sign flip would silently change replayed
    columns. B-splines leave it ``None``.
    """

    interior: tuple[float, ...]
    boundary: tuple[float, float]
    degree: int
    column_means: tuple[float, ...]
    projector: tuple[tuple[float, ...], ...] | None = None


def _basis_call(call: Call) -> BasisCall:
    if isinstance(call, BasisCall):
        return call
    raise TypeError(f"Expected a BasisCall, got {type(call).__name__}.")


def _number(call: BasisCall, key: str, default: float | None = None) -> float:
    """Return a numeric spline parameter, rejecting a knot tuple."""
    value = call.params[key] if key in call.params else default
    if isinstance(value, tuple) or value is None:
        raise TypeError(
            f"{call.name}() parameter '{key}' must be a number, got {value!r}."
        )
    return float(value)


def _knots(call: BasisCall) -> tuple[float, ...]:
    value = call.params["knots"]
    if not isinstance(value, tuple):
        raise TypeError(f"{call.name}() knots must be a tuple, got {value!r}.")
    return tuple(float(knot) for knot in value)


def spline_degree(call: Call) -> int:
    """Return the B-spline degree, defaulting to cubic."""
    return int(_number(_basis_call(call), "degree", DEFAULT_DEGREE))


def spline_order(call: Call) -> int:
    """Return the P-spline penalty order, defaulting to second differences."""
    return int(_number(_basis_call(call), "order", DEFAULT_ORDER))


def n_interior_knots(name: str, call: Call) -> int:
    """Return how many interior knots a ``df`` call places."""
    df = int(_number(_basis_call(call), "df"))
    # implied_df(name, n, degree) == df, so n is df minus the zero-knot count.
    return df - implied_df(name, 0, spline_degree(call))


def random_walk_map(n: int, order: int) -> np.ndarray:
    """Map random-walk innovations onto B-spline coefficients.

    Column ``j`` is the coefficient vector of a unit innovation at index ``j``.
    Order 1 is a cumulative sum. Order 2 leaves the level and the slope
    unpenalized: their second differences are zero, and each later column is
    one penalized second difference.
    """
    if order not in (1, 2):
        raise ValueError(
            f"P-spline order must be 1 or 2, got {order}. "
            "Use order=1 for a first-order penalty or omit order= for the default."
        )
    if n < 1:
        raise ValueError(f"A random-walk map needs at least one coefficient, got {n}.")
    mapping = np.zeros((n, n), dtype=float)
    if order == 1:
        for column in range(n):
            mapping[column:, column] = 1.0
        return mapping
    mapping[:, 0] = 1.0
    if n > 1:
        mapping[:, 1] = np.arange(n, dtype=float)
    for column in range(2, n):
        mapping[column:, column] = np.arange(1, n - column + 1, dtype=float)
    return mapping


def innovation_prior_sigma(
    n: int, order: int, *, null_scale: float = NULL_SPACE_SCALE
) -> np.ndarray:
    """Prior standard deviations of the random-walk innovations.

    The first ``order`` innovations span the penalty's null space and stay
    wide. The rest are unit-scale innovations that ``tau`` multiplies.
    """
    sigma = np.ones(n, dtype=float)
    sigma[:order] = null_scale
    return sigma


def penalty_multiplier(n: int, order: int, tau: float) -> np.ndarray:
    """Scale applied on top of the innovation prior.

    Null-space innovations keep the prior scale. Penalized innovations are
    multiplied by ``tau``.
    """
    multiplier = np.ones(n, dtype=float)
    if n > order:
        multiplier[order:] = tau
    return multiplier


def effective_degrees_of_freedom(
    columns: np.ndarray,
    tau: float,
    *,
    order: int = DEFAULT_ORDER,
    noise_scale: float = 1.0,
    null_scale: float = NULL_SPACE_SCALE,
) -> float:
    """Trace of the P-spline smoother at a fixed penalty scale.

    Uses the same random-walk map and scales as :meth:`SplineBasis.contribution`.
    ``columns`` are the centered basis columns stored on the model. ``tau`` is
    the random-walk scale. ``noise_scale`` is the observation standard deviation.
    """
    design = np.asarray(columns, dtype=float)
    n_obs, n_coef = design.shape
    if n_obs < 1 or n_coef < 1:
        raise ValueError(
            "effective_degrees_of_freedom() needs at least one row and one column."
        )
    scales = innovation_prior_sigma(n_coef, order, null_scale=null_scale)
    scales = scales * penalty_multiplier(n_coef, order, tau)
    mapping = random_walk_map(n_coef, order)
    covariance = (mapping * scales) @ (mapping * scales).T
    gram = design @ covariance @ design.T
    smoother = np.linalg.solve(
        gram + (noise_scale**2) * np.eye(n_obs),
        gram,
    )
    return float(np.trace(smoother))


def register_splines() -> None:
    """Register ``bs``, ``ns``, and ``ps`` for the formula DSL."""
    for name in ("bs", "ns", "ps"):
        register_basis(SplineBasis(name))


class SplineBasis(Basis):
    """A centered spline basis with an owned coefficient vector.

    ``bs`` is a B-spline that continues as a polynomial outside the boundary
    knots. ``ns`` is a natural cubic spline that continues linearly. ``ps``
    uses the ``bs`` columns and a random-walk prior on adjacent coefficients.
    All three center columns at the frozen training means so the intercept
    remains the mean response.
    """

    def __init__(self, name: str) -> None:
        if name not in SPLINE_NAMES:
            raise ValueError(
                f"Unknown spline '{name}'. Expected one of {sorted(SPLINE_NAMES)}."
            )
        self.name = name
        self.capabilities = BasisCapabilities(
            supports_endogenous=False,
            supports_panel=False,
            requires_raw_input_units=True,
            requires_intercept=True,
        )

    def n_basis(self, call: Call) -> int:
        """Return the number of free coefficients stored in ``beta_*``."""
        call = self._require_call(call)
        if "df" in call.params:
            return int(_number(call, "df"))
        return implied_df(self.name, len(_knots(call)), spline_degree(call))

    def build_data(
        self, x: np.ndarray, call: Call, *, state: Any | None = None
    ) -> tuple[np.ndarray, SplineState]:
        """Build centered spline columns, replaying frozen knots when given."""
        call = self._require_call(call)
        values = _as_numeric_column(self.name, x)
        if state is None:
            frozen = self._fit_state(values, call)
        else:
            if not isinstance(state, SplineState):
                raise TypeError(
                    f"{self.name}() replay state must be a SplineState, "
                    f"got {type(state).__name__}."
                )
            frozen = state
        if len(frozen.column_means) != self.n_basis(call):
            raise ValueError(
                f"{self.name}() replay state has {len(frozen.column_means)} columns "
                f"but this call requests {self.n_basis(call)}. Replay the state "
                "with the call that produced it."
            )
        return _centered_columns(self.name, values, frozen), frozen

    def default_priors(self, lhs: str, call: Call, priors: Any) -> None:
        """Register the coefficient prior, and ``tau`` for a P-spline."""
        from pymc_extras.prior import Prior

        beta_name = self.beta_name(lhs, call)
        dim = self.weights_dim(lhs, call)
        if beta_name not in priors:
            if self.name == "ps":
                order = spline_order(call)
                sigma = innovation_prior_sigma(self.n_basis(call), order)
                priors[beta_name] = Prior("Normal", mu=0, sigma=sigma, dims=(dim,))
            else:
                priors[beta_name] = Prior(
                    "Normal", mu=0, sigma=UNPENALIZED_SIGMA, dims=(dim,)
                )
        if self.name == "ps":
            tau_name = self.tau_name(lhs, call)
            if tau_name not in priors:
                priors[tau_name] = Prior("HalfNormal", sigma=DEFAULT_TAU_SIGMA)

    def prior_descriptions(self, lhs: str, call: Call) -> dict[str, str]:
        """Return readable labels for the spline priors."""
        beta_name = self.beta_name(lhs, call)
        if self.name != "ps":
            return {beta_name: f"Normal(0, {UNPENALIZED_SIGMA:g})"}
        order = spline_order(call)
        return {
            beta_name: (
                f"Normal(0, {NULL_SPACE_SCALE:g}) on the {order} unpenalized "
                "random-walk directions; Normal(0, 1) innovations elsewhere"
            ),
            self.tau_name(lhs, call): f"HalfNormal({DEFAULT_TAU_SIGMA:g})",
        }

    def contribution(
        self,
        columns: Any,
        state: Any,
        *,
        lhs: str,
        call: Call,
        priors: Any,
    ) -> Any:
        """Combine columns with the owned coefficient prior.

        A P-spline maps innovations through the random-walk matrix and
        multiplies only the penalized innovations by ``tau``. The null-space
        scale stays in the innovation prior so overriding that prior replaces
        it instead of fighting a hidden multiplier.
        """
        if self.name != "ps":
            return super().contribution(
                columns, state, lhs=lhs, call=call, priors=priors
            )
        from pathmc.priors import _ensure_dims

        beta_name = self.beta_name(lhs, call)
        beta = _ensure_dims(
            priors[beta_name], self.weights_dim(lhs, call)
        ).create_variable(beta_name)
        tau_name = self.tau_name(lhs, call)
        tau = priors[tau_name].create_variable(tau_name)
        n_coef = self.n_basis(call)
        order = spline_order(call)
        mapping = pt.as_tensor_variable(random_walk_map(n_coef, order))
        multiplier = pt.ones((n_coef,))
        if n_coef > order:
            multiplier = pt.set_subtensor(multiplier[order:], tau)
        coefficients = mapping @ (multiplier * beta)
        smooth = columns @ coefficients
        return pm.Deterministic(self.contribution_name(lhs, call), smooth)

    def render(self, call: Call) -> str:
        """Return the DSL call, omitting parameters that have their default."""
        if not isinstance(call, BasisCall):
            return super().render(call)
        parts = [call.variable]
        if "df" in call.params:
            parts.append(f"df={int(_number(call, 'df'))}")
        if "knots" in call.params:
            parts.append(f"knots={_format_knots(_knots(call))}")
        degree = _number(call, "degree", DEFAULT_DEGREE)
        if "degree" in call.params and int(degree) != DEFAULT_DEGREE:
            parts.append(f"degree={int(degree)}")
        order = _number(call, "order", DEFAULT_ORDER)
        if "order" in call.params and int(order) != DEFAULT_ORDER:
            parts.append(f"order={int(order)}")
        return f"{self.name}({', '.join(parts)})"

    def render_latex(self, variable: str) -> str:
        """Return a LaTeX fragment. Parameters stay in the plain-text render."""
        return rf"\operatorname{{{self.name}}}({variable})"

    def tau_name(self, lhs: str, call: Call) -> str:
        """Return the random-walk scale name for a P-spline."""
        return f"tau_ps_{lhs}_{call.variable}{self._call_suffix(call)}"

    def _require_call(self, call: Call) -> BasisCall:
        if not isinstance(call, BasisCall):
            raise TypeError(f"{self.name}() expected a BasisCall, got {type(call)!r}.")
        problem = spline_call_problem(call.name, call.variable, call.params)
        if problem is not None:
            raise ValueError(problem)
        if call.name != self.name:
            raise ValueError(
                f"{self.name}() received a {call.name}() call. "
                "Use the basis registered under that name."
            )
        return call

    def _fit_state(self, values: np.ndarray, call: BasisCall) -> SplineState:
        _require_varying_numeric(self.name, call.variable, values)
        degree = spline_degree(call)
        if "knots" in call.params:
            interior = _knots(call)
        else:
            interior = _quantile_knots(
                self.name, values, n_interior_knots(self.name, call)
            )
        boundary = (float(np.min(values)), float(np.max(values)))
        _require_interior_knots(self.name, interior, boundary)
        knots = _knot_vector(interior, boundary, degree)
        projector = None
        if self.name == "ns":
            projector_array = _natural_projector(knots, degree)
            projector = tuple(tuple(float(v) for v in row) for row in projector_array)
        raw = _raw_columns(
            self.name,
            values,
            SplineState(interior, boundary, degree, (), projector),
        )
        if raw.shape[1] != self.n_basis(call):
            raise ValueError(
                f"{self.name}() built {raw.shape[1]} columns but df requests "
                f"{self.n_basis(call)}. This is a bug in the spline basis."
            )
        rank = int(np.linalg.matrix_rank(raw, tol=1e-8))
        if rank < raw.shape[1]:
            raise ValueError(
                f"{self.name}() basis on '{call.variable}' has rank {rank} "
                f"but {raw.shape[1]} columns. Lower df, or pass distinct knots= "
                "spread through the observed range."
            )
        means = tuple(float(v) for v in raw.mean(axis=0))
        return SplineState(interior, boundary, degree, means, projector)


def _as_numeric_column(name: str, values: np.ndarray) -> np.ndarray:
    try:
        column = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name}() input must be numeric. Got {type(values).__name__}."
        ) from exc
    if column.ndim == 2 and column.shape[1] == 1:
        column = column[:, 0]
    if column.ndim != 1:
        raise ValueError(
            f"{name}() takes one column, got shape {tuple(column.shape)}. "
            "Pass a single numeric predictor."
        )
    if column.size == 0:
        raise ValueError(f"{name}() received no observations.")
    if not np.isfinite(column).all():
        raise ValueError(
            f"{name}() input contains NaN or infinity. Drop those rows or "
            "impute the predictor before fitting."
        )
    return column


def _require_varying_numeric(name: str, variable: str, values: np.ndarray) -> None:
    if float(np.min(values)) == float(np.max(values)):
        raise ValueError(
            f"{name}() input '{variable}' is constant, so it has no range for "
            "knots. Drop the term or use a column that varies."
        )


def _require_interior_knots(
    name: str, interior: tuple[float, ...], boundary: tuple[float, float]
) -> None:
    if not interior:
        return
    knots = np.asarray(interior, dtype=float)
    if np.any(np.diff(knots) <= 0):
        raise ValueError(
            f"{name}() knots must be strictly increasing, got {interior}. "
            "Sort them and drop duplicates."
        )
    lo, hi = boundary
    if np.any(knots <= lo) or np.any(knots >= hi):
        raise ValueError(
            f"{name}() knots {interior} must lie strictly inside the observed "
            f"range [{lo}, {hi}]. Omit knots= to place them at quantiles, or "
            "pass values between the training minimum and maximum."
        )


def _quantile_knots(
    name: str, values: np.ndarray, n_interior: int
) -> tuple[float, ...]:
    if n_interior == 0:
        return ()
    probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.quantile(values, probs)
    if np.unique(np.round(knots, decimals=10)).size < n_interior:
        raise ValueError(
            f"{name}() cannot place {n_interior} distinct quantile knots in "
            "this sample. Lower df, or pass knots= yourself."
        )
    return tuple(float(knot) for knot in knots)


def _knot_vector(
    interior: tuple[float, ...], boundary: tuple[float, float], degree: int
) -> np.ndarray:
    lo, hi = boundary
    return np.concatenate([
        np.repeat(lo, degree + 1),
        np.asarray(interior, dtype=float),
        np.repeat(hi, degree + 1),
    ])


def _bspline_design(
    x: np.ndarray, knots: np.ndarray, degree: int, *, extrapolate: bool
) -> np.ndarray:
    n_basis = len(knots) - degree - 1
    if x.size == 0:
        return np.empty((0, n_basis), dtype=float)
    eye = np.eye(n_basis)
    return np.column_stack([
        BSpline(knots, eye[i], degree, extrapolate=extrapolate)(x)
        for i in range(n_basis)
    ])


def _natural_projector(knots: np.ndarray, degree: int) -> np.ndarray:
    """Project the B-spline basis onto the natural-cubic null space.

    The constraints are a zero second derivative at each boundary plus removal
    of the constant, which B-splines represent by the all-ones coefficient.
    The result has ``n_interior + 1`` columns for a cubic.
    """
    n_basis = len(knots) - degree - 1
    eye = np.eye(n_basis)
    boundary = (float(knots[degree]), float(knots[-degree - 1]))
    rows = [
        np.array([
            float(BSpline(knots, eye[i], degree).derivative(2)(point))
            for i in range(n_basis)
        ])
        for point in boundary
    ]
    rows.append(np.ones(n_basis))
    constraints = np.vstack(rows)
    _, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular > 1e-8 * singular[0]))
    if rank < constraints.shape[0]:
        raise ValueError(
            "ns() could not enforce linear extrapolation at both boundaries. "
            "Raise df or pass knots that are not piled at one end of the range."
        )
    projector = vt[rank:].T.copy()
    for column in range(projector.shape[1]):
        nonzero = np.flatnonzero(np.abs(projector[:, column]) > 1e-8)
        if nonzero.size and projector[nonzero[0], column] < 0:
            projector[:, column] *= -1.0
    return projector


def _raw_columns(name: str, x: np.ndarray, state: SplineState) -> np.ndarray:
    knots = _knot_vector(state.interior, state.boundary, state.degree)
    if name == "ns":
        return _natural_columns(x, knots, state)
    design = _bspline_design(x, knots, state.degree, extrapolate=True)
    return design[:, :-1]


def _natural_columns(
    x: np.ndarray, knots: np.ndarray, state: SplineState
) -> np.ndarray:
    if state.projector is None:
        raise ValueError("ns() replay state is missing its projector.")
    projector = np.asarray(state.projector, dtype=float)
    lo, hi = state.boundary
    columns = np.empty((x.size, projector.shape[1]), dtype=float)
    inside = (x >= lo) & (x <= hi)
    if np.any(inside):
        columns[inside] = (
            _bspline_design(x[inside], knots, state.degree, extrapolate=False)
            @ projector
        )
    outside = ~inside
    if np.any(outside):
        boundary = np.array([lo, hi], dtype=float)
        values = (
            _bspline_design(boundary, knots, state.degree, extrapolate=False)
            @ projector
        )
        slopes = _boundary_slopes(knots, state.degree, projector, boundary)
        points = x[outside]
        on_right = points > hi
        base = np.where(on_right, hi, lo)
        level = np.where(on_right[:, None], values[1], values[0])
        slope = np.where(on_right[:, None], slopes[1], slopes[0])
        columns[outside] = level + (points - base)[:, None] * slope
    return columns


def _boundary_slopes(
    knots: np.ndarray,
    degree: int,
    projector: np.ndarray,
    boundary: np.ndarray,
) -> np.ndarray:
    n_basis = projector.shape[0]
    eye = np.eye(n_basis)
    slopes = np.empty((boundary.size, projector.shape[1]), dtype=float)
    for index, point in enumerate(boundary):
        raw = np.array([
            float(BSpline(knots, eye[i], degree).derivative(1)(float(point)))
            for i in range(n_basis)
        ])
        slopes[index] = raw @ projector
    return slopes


def _centered_columns(name: str, x: np.ndarray, state: SplineState) -> np.ndarray:
    raw = _raw_columns(name, x, state)
    return raw - np.asarray(state.column_means, dtype=float)


def _format_knots(knots: Any) -> str:
    values = tuple(float(knot) for knot in knots)
    if not values:
        return "()"
    body = ", ".join(_format_number(value) for value in values)
    if len(values) == 1:
        return f"({body},)"
    return f"({body})"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))
