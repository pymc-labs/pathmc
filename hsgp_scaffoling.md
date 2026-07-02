# HSGP term — implementation scaffolding (Phase 1: 1-D, cross-sectional)

Design/scaffolding document for [issue #360](https://github.com/pymc-labs/pathmc/issues/360): add a Hilbert Space Gaussian Process (HSGP) term to the pathmc DSL, giving a scalable nonparametric smooth analogous to Bambi's `hsgp()` formula term. This document specifies the components, fully typed function signatures, and test structure so the implementation is modular, testable, and maintainable. It does not contain the implementation itself.

Status: scaffolding only. Every `pathmc/*.py`, test, and `.qmd` file named below is a target for a follow-up implementation step.

## 0. Verified facts

These were checked live against the installed environment and the upstream sources, so signatures below are concrete rather than aspirational.

Installed stack: `pymc 6.0.1`, `pytensor 3.0.4`, `arviz 1.1.0`, `numpy 2.4.6`.

PyMC HSGP API (verified):

- `pm.gp.HSGP(m=[M], L=None, c=None, drop_first=False, parametrization="noncentered", *, mean_func=Zero, cov_func)`. Both `m` and `L` are per-dimension sequences: `m=[M]` and, when an explicit boundary is used, `L=[L]`. Passing a scalar `L` raises a `ValueError`, so the DSL's scalar `L` must be wrapped into `[L]` before construction (see `hsgp_basis`, Section 6).
- `phi, sqrt_psd = gp.prior_linearized(X)` where `X` is a `(n, 1)` tensor (a `pm.Data` node works); returns `phi` of shape `(n, M)` and `sqrt_psd` of shape `(M,)`. In this PyMC version `prior_linearized` centers `X` internally from `L`/`c`, so raw inputs are fine.
- `gp.n_basis_vectors == prod(m)` (for a single 1-D input with `drop_first=False`, `M == m`).
- Kernels: `pm.gp.cov.ExpQuad(input_dim, ls=...)`, `pm.gp.cov.Matern52(input_dim, ls=...)`, `pm.gp.cov.Matern32(input_dim, ls=...)`.

Canonical PyMC linearized recipe (from the `prior_linearized` docstring), which this design follows:

```python
cov_func = eta**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=ell)
gp = pm.gp.HSGP(m=[M], c=c, cov_func=cov_func)
phi, sqrt_psd = gp.prior_linearized(X=x_data)      # x_data: pm.Data, shape (n, 1)
beta = pm.Normal("beta", size=gp.n_basis_vectors)  # non-centered
f = pm.Deterministic("f", phi @ (beta * sqrt_psd))
```

Bambi parity (from `bambi/terms/hsgp.py`): the `hsgp(...)` term exposes `m, L, c, by_levels, cov, share_cov, scale, iso, drop_first, centered, mean`; priors are a dict keyed by covariance-parameter names (`sigma`, `ell`) whose values are `Prior` objects or numeric constants; the weights coordinate is `{name}_weights_dim = arange(prod(m))`; `centered` is a boolean flag. Bambi names the amplitude `sigma`; issue #360 names it `eta`. This design follows the issue (`eta`, `ell`) and documents the mapping for Bambi users.

pathmc internals (verified in `pathmc/compile.py`), which determine the exact integration points and edge cases handled in Section 6:

- `get_free_predictor_columns` (`compile.py:124`) currently appends `t.variable` for every non-fixed term. Left unchanged it would create a spurious `beta_{lhs}` column named after the HSGP input.
- `beta` is created only when `free_cols` is non-empty (`compile.py:463-466`), so `Y ~ 0 + hsgp(x)` yields `beta=None`.
- Top-level `coords` are assembled at `compile.py:390-394` before `pm.Model(coords=...)`. Because `m` is a compile-time literal, the HSGP weights coordinate can be registered there.
- `build_mu` (`compile.py:506`), the cross-sectional resolver (`compile.py:555`), the residual-block compiler (`compile.py:745`, `:803-811`), and the scan-panel compiler (`compile.py:1251`, `:1644-1654`) all funnel through `build_mu` plus a resolver. These are the only choke points for dispatch and guardrails.
- `_make_cross_sectional_resolver` (`compile.py:555`) currently takes `(data, data_vars, endogenous_rvs, transform_map, transform_param_rvs, panel_info)` and has no `lhs` or `priors` in scope. HSGP dispatch needs both, so the factory must gain `lhs: str` and `priors: PriorConfig` parameters; the per-equation call site (`compile.py:468`) has `var` and `priors` in scope and passes `lhs=var, priors=priors`.
- `_compile_residual_block` (`compile.py:788-802`) creates `beta_{var}` whenever `has_free = any(s.coeff_type == "free" for s in ms.slots)` is true, using `dims=f"{var}_predictors"`. HSGP slots are `coeff_type="free"` yet excluded from `{var}_predictors`, so an HSGP term on a block LHS would allocate `beta` against a missing coordinate. This is why HSGP inside a residual-covariance block is rejected in Phase 1 (Section 6).
- Exogenous inputs become 1-D `pm.Data(var, shape (n,))` at `compile.py:416-419`, so HSGP must reshape to `(n, 1)` inside the graph.

## 1. Scope, goals, and Bambi parity

Phase 1 goal: `Y ~ hsgp(x, m=20, c=1.5)` parses, compiles, samples, and recovers a known smooth on cross-sectional data, matching the behavior of Bambi's `hsgp_1d` example.

Acceptance criteria (copied verbatim from issue #360, Phase 1):

- `y ~ hsgp(x, m=..., c=...)` parses, compiles, and samples on a cross-sectional model.
- Recovers a known smooth function on synthetic data (test), comparable to Bambi's `hsgp_1d`.
- `pm.do()` intervention on `x` recomputes the basis (test).
- Clear errors for panel/scan use, nesting, and interaction use.
- Docs example mirroring Bambi's `hsgp_1d`.

Parity table. "Supported" means Phase 1 implements it; "Deferred" means Phase 1 must raise a clear, actionable error and a follow-up issue owns it.

| Bambi `hsgp()` feature | pathmc Phase 1 | Notes |
| --- | --- | --- |
| single 1-D input `hsgp(x)` | Supported | `HSGPCall.variable` |
| `m` (basis count) | Supported | required literal int |
| `c` (boundary factor) | Supported | exactly one of `c`/`L` |
| `L` (explicit boundary) | Supported | exactly one of `c`/`L` |
| `cov` (kernel) | Supported | `expquad`, `matern52`, `matern32` |
| `centered` | Supported | boolean literal, default `False` |
| multi-D `hsgp(x1, x2)` | Deferred | reject >1 positional input |
| `by=` / grouped GPs | Deferred | reject `by` kwarg |
| `share_cov`, `iso`, `scale`, `drop_first`, `mean` | Deferred | reject unknown kwargs |
| panel / scan models | Deferred | model-level guard raises when panel mode is active |
| `simulate()` generative draws | Deferred | `simulate()` raises |

Rationale for a dedicated path (not the `Transform` registry). Two hard constraints make reusing `TransformCall` the wrong choice:

1. The transform parser `_parse_transform_expr` treats every keyword-argument value as an RV name string (`decay=theta`). HSGP needs literal configuration (`m=20`, `c=1.5`, `cov="expquad"`, `centered=True`). Overloading `TransformCall` would break that contract.
2. `build_mu` assumes one slot equals one coefficient equals one vector (`mu += coef * tensor`). HSGP is `mu += phi @ (sqrt_psd * beta)` with `M` coefficients, so it needs its own coefficient vector outside the shared `beta_{lhs}` and its own `build_mu` branch.

Therefore HSGP gets a dedicated `HSGPCall` AST node and a new `kind="hsgp"` predictor slot, as recommended by the issue.

## 2. Module layout (modularity and testability)

The kernel/basis math is isolated in a new module so it can be unit-tested and later extended (Matern variants, multi-D, grouped) without touching the compiler. This preserves the AGENTS.md architecture principle that the graph/kernel layer is independent of the PyMC compiler.

| Module | Responsibility for HSGP | New / edited |
| --- | --- | --- |
| `pathmc/parse.py` | `HSGPCall` dataclass, `Term.hsgp` field, `_parse_hsgp_expr` (pure, no PyMC) | edited |
| `pathmc/hsgp.py` | kernel factory, basis construction, term assembly (parametrization) | new |
| `pathmc/compile.py` | slot kind, `build_mu` branch, resolver dispatch, coord registration, guardrails | edited |
| `pathmc/priors.py` | `_collect_hsgp_defaults` for `ell`/`eta`/`beta_hsgp` | edited |
| `pathmc/introspect.py` | equation/LaTeX/DAG rendering, prior listing | edited |
| `pathmc/simulate.py`, `pathmc/_model.py` | `simulate()` guardrail | edited |

`pathmc/hsgp.py` public surface (all fully typed, NumPy-docstringed):

```python
from typing import TypeAlias

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from pathmc.parse import HSGPCall
from pathmc.priors import PriorConfig

TensorLike: TypeAlias = TensorVariable | np.ndarray

COV_FUNCS: dict[str, type[pm.gp.cov.Stationary]] = {
    "expquad": pm.gp.cov.ExpQuad,
    "matern52": pm.gp.cov.Matern52,
    "matern32": pm.gp.cov.Matern32,
}


def make_cov_func(cov: str, *, eta: TensorLike, ell: TensorLike) -> pm.gp.cov.Covariance:
    """Build the HSGP covariance function ``eta**2 * Kernel(input_dim=1, ls=ell)``."""


def hsgp_basis(call: HSGPCall, x: TensorLike) -> tuple[TensorLike, TensorLike, int]:
    """Return ``(phi, sqrt_psd, n_basis)`` for a 1-D input via ``prior_linearized``.

    Constructs ``pm.gp.HSGP(m=[call.m], L=[call.L] if call.L is not None
    else None, c=call.c, cov_func=...)`` -- both ``m`` and ``L`` must be
    per-dimension sequences, so the scalar ``call.L`` is wrapped into ``[call.L]``.
    """


def assemble_hsgp_term(
    call: HSGPCall,
    x: TensorLike,
    *,
    lhs: str,
    priors: PriorConfig,
) -> TensorVariable:
    """Emit the HSGP hyperparameters and coefficients, returning ``f_{lhs}_{var}``."""
```

Naming convention for all HSGP random variables and coordinates uses `{param}_{lhs}_{var}`, matching pathmc's predictable, stable-across-runs naming principle. Because the names key only on `(lhs, var)`, HSGP terms on different variables of the same left-hand side never collide (`f_Y_x` vs `f_Y_z`), but two HSGP terms sharing the same `(lhs, var)` pair -- for example `Y ~ hsgp(x, m=20, c=1.5) + hsgp(x, m=30, c=2.0)` -- would reuse `ell_Y_x`/`eta_Y_x`/`beta_hsgp_Y_x`/`f_Y_x` and cause duplicate PyMC variable registration. Phase 1 rejects such a repeated `(lhs, var)` pair at validation time (Section 3 guardrails), so the names below are unique by construction:

- lengthscale RV: `ell_{lhs}_{var}`
- amplitude RV: `eta_{lhs}_{var}`
- basis-weight RV: `beta_hsgp_{lhs}_{var}` (dim `{lhs}_{var}_hsgp`)
- deterministic smooth: `f_{lhs}_{var}`
- weights coordinate: `{lhs}_{var}_hsgp` = `range(m)`

## 3. DSL syntax and AST

New AST node in `pathmc/parse.py`:

```python
@dataclass
class HSGPCall:
    """Parsed ``hsgp(...)`` term (Phase 1: a single 1-D input).

    Parameters
    ----------
    variable : str
        Name of the input column the smooth is a function of.
    m : int
        Number of Laplacian eigenfunction basis vectors. Compile-time literal.
    c : float | None
        Boundary-condition expansion factor. Exactly one of ``c`` or ``L``.
    L : float | None
        Explicit boundary as a user-facing scalar. Exactly one of ``c`` or
        ``L``. ``hsgp_basis`` wraps it into the one-element sequence
        ``[L]`` that ``pm.gp.HSGP`` requires.
    cov : str
        Covariance kernel: ``"expquad"``, ``"matern52"``, or ``"matern32"``.
    centered : bool
        If ``True`` use the centered parametrization; otherwise non-centered.
    """

    variable: str
    m: int
    c: float | None = None
    L: float | None = None
    cov: str = "expquad"
    centered: bool = False
```

`Term` gains one field:

```python
@dataclass
class Term:
    variable: str
    label: str | None = None
    transform: TransformCall | None = None
    lag_of: str | None = None
    interaction_of: tuple[str, ...] | None = None
    fixed_value: float | None = None
    hsgp: HSGPCall | None = None  # new
```

Parser wiring. In `_parse_term` (`parse.py:234`), before the generic transform branch (`if "(" in raw:`), detect a top-level call whose function name is `hsgp` and route to a dedicated parser:

```python
def _parse_hsgp_expr(raw: str) -> HSGPCall:
    """Parse ``hsgp(x, m=..., c=..., cov=..., centered=...)`` into an HSGPCall.

    Unlike transform parsing, keyword values are parsed as literals
    (int / float / bool / bare string), not as random-variable names.

    Raises
    ------
    ParseError
        If ``m`` is missing, both/neither of ``c``/``L`` are given, an
        unknown keyword is used, more than one positional input is given,
        or a value cannot be coerced to its expected literal type.
    """
```

Literal coercion: `m -> int`, `c -> float`, `L -> float`, `cov -> str` (strip quotes, lowercased), `centered -> bool` (`"true"`/`"false"`, case-insensitive). Reuse `_split_top_level_args` for argument splitting so behavior matches the transform parser.

The resulting term is `Term(variable=call.variable, hsgp=call)`. Coefficient prefixes are not allowed on an HSGP term: the smooth carries its own basis weights, so a numeric prefix (`2*hsgp(x, ...)`, which would parse to `fixed_value=2.0`) or a label prefix (`b*hsgp(x, ...)`) is ambiguous and would be silently ignored by the `build_mu` HSGP branch (`mu += resolver(slot)`), diverging from every other predictor term. Both are rejected at parse time (see guardrails).

Guardrails, each raising `ParseError` with a message that names the problem and the fix:

- Coefficient prefix on an HSGP term (`2*hsgp(x, ...)` or `b*hsgp(x, ...)`): "hsgp(...) cannot take a coefficient prefix; the smooth carries its own basis weights. Remove the 'k*' or 'label*' prefix."
- Missing `m`: "hsgp(...) requires m=<int> (number of basis vectors). Example: hsgp(x, m=20, c=1.5)."
- Both or neither of `c`/`L`: "hsgp(...) needs exactly one of c=<float> or L=<float>."
- Unknown keyword (e.g. `by`, `iso`, `share_cov`): "hsgp(...) does not support '<kw>' in Phase 1. Supported: m, c, L, cov, centered."
- More than one positional input (`hsgp(x1, x2)`): "Multi-dimensional hsgp(x1, x2) is not supported yet (see #360 follow-up). Use a single input."
- Nesting inside a transform (`adstock(hsgp(...))`): `_parse_transform_expr` must reject an `hsgp(` input with "hsgp(...) cannot be nested inside a transform. Apply hsgp() directly to a variable."
- Inside an interaction (`hsgp(x):z`): `_parse_interaction_term` already rejects non-identifier parts; add a targeted message pointing at `hsgp`.
- Duplicate `(lhs, variable)` HSGP terms in the same equation (`Y ~ hsgp(x, m=20, c=1.5) + hsgp(x, m=30, c=2.0)`): reject at regression-validation time, since both would reuse the `{lhs}_{var}` RV/coord names and duplicate PyMC variable registration. Message: "Two hsgp() terms on the same variable 'x' in equation 'Y' would collide. Use a single hsgp() per variable per equation (multiple smooths of one variable are a #360 follow-up)." HSGP terms on different variables of the same LHS are allowed.

## 4. Covariance kernel and its parameters

The covariance function is `cov_func = eta**2 * Kernel(input_dim=1, ls=ell)`, where `Kernel` is selected by the `cov` literal through `make_cov_func` (case-insensitive lookup in `COV_FUNCS`). Unknown kernel names raise `ValueError` naming the valid options.

The kernel exposes two estimable parameters. They use predictable, flat, override-able RV names consistent with pathmc's existing `beta_Y` / `sigma_Y` convention (rather than Bambi's nested per-term prior dict), so the priors API stays uniform across the package:

| Kernel parameter | RV name | Default prior | Meaning |
| --- | --- | --- | --- |
| lengthscale | `ell_{lhs}_{var}` | `Prior("InverseGamma", alpha=3, beta=1)` | smoothness / wiggliness of the smooth; weakly-informative prior avoids `ell -> 0` degeneracy |
| amplitude | `eta_{lhs}_{var}` | `Prior("HalfNormal", sigma=1)` | vertical scale of the smooth |
| basis weights | `beta_hsgp_{lhs}_{var}` | `Prior("Normal", mu=0, sigma=1, dims=("{lhs}_{var}_hsgp",))` | standardized coefficients (non-centered); see Section 5 |

A LogNormal lengthscale is a documented alternative to InverseGamma; the default is InverseGamma for its thin left tail.

How a user sets covariance parameters:

- Kernel family via the DSL: `hsgp(x, m=20, c=1.5, cov='matern52')`.
- Kernel hyperpriors via the priors API, keyed by RV name:

```python
import pathmc
from pathmc import Prior

model = pathmc.model(
    "y ~ hsgp(x, m=20, c=1.5)",
    data=df,
    priors={
        "ell_y_x": Prior("InverseGamma", alpha=3, beta=1),
        "eta_y_x": Prior("HalfNormal", sigma=2),
    },
)
```

`assemble_hsgp_term` reads the merged prior config for `ell_{lhs}_{var}` and `eta_{lhs}_{var}` (falling back to defaults from `default_priors`), creates the RVs via `Prior.create_variable`, and passes them to `make_cov_func`. Extra kernel-specific settings (for example Matern order) are expressed through kernel choice, not free parameters, in Phase 1.

Bambi mapping note for users migrating from Bambi: Bambi's automatic GP priors use `sigma` (amplitude) and `ell` (lengthscale); here amplitude is `eta` and the prior keys are the flat `eta_{lhs}_{var}` / `ell_{lhs}_{var}` names.

## 5. Centered vs non-centered parametrization

Because the design uses `prior_linearized`, the parametrization is expressed in how `assemble_hsgp_term` builds the smooth from `phi` and `sqrt_psd`, selected by the DSL `centered` flag (mirroring Bambi's `centered`):

- Non-centered (default, `centered=False`):

```python
beta = pm.Normal(f"beta_hsgp_{lhs}_{var}", mu=0.0, sigma=1.0, dims=weights_dim)
f = pm.Deterministic(f"f_{lhs}_{var}", phi @ (beta * sqrt_psd))
```

- Centered (`centered=True`):

```python
beta = pm.Normal(f"beta_hsgp_{lhs}_{var}", mu=0.0, sigma=sqrt_psd, dims=weights_dim)
f = pm.Deterministic(f"f_{lhs}_{var}", phi @ beta)
```

DSL: `hsgp(x, m=20, c=1.5, centered=True)`.

Guidance to document for users: non-centered is the robust default and is preferred when the GP signal is weak relative to observation noise; the centered form can sample more efficiently when the GP signal dominates the noise.

Prior-interaction caveat (stated explicitly to avoid a silent-override loophole): in non-centered mode the `beta_hsgp_{lhs}_{var}` prior is the standardized `Normal(0, 1)` and a user may rescale it through the priors API. In centered mode the coefficient scale is data-derived (`sqrt_psd`), so `beta` is constructed directly rather than from the `Prior` config, and a user-supplied `sigma` on `beta_hsgp_{lhs}_{var}` has no effect. In centered mode users should tune `ell` and `eta` instead. The implementation should either drop the `beta_hsgp_*` key from the overrides surface in centered mode or emit a clear warning if it is set.

## 6. Compile integration and verified edge cases

Changes in `pathmc/compile.py`, each anchored to a verified choke point:

- `PredictorSlot` (`:70`): add `"hsgp"` to the `kind` `Literal` and a field `hsgp: HSGPCall | None = None`.
- `build_mu_specs` (`:157`): when `term.hsgp is not None`, emit `PredictorSlot(kind="hsgp", name=term.variable, coeff_type="free", hsgp=term.hsgp)`. Kind dispatch priority: `hsgp` before `transform`/`interaction`/`lag`.
- `get_free_predictor_columns` (`:124`): skip terms where `term.hsgp is not None` so no spurious `beta_{lhs}` column is created. The parallel all-columns helper used by the design matrix (`:119-121`) keeps the raw input (see `design()` note below).
- `build_mu` (`:506`): handle `slot.kind == "hsgp"` before the free/fixed coefficient logic and `continue` without advancing `free_idx`:

```python
if slot.kind == "hsgp":
    mu = mu + resolver(slot)
    continue
```

  This is why an HSGP-only regression (`Y ~ 0 + hsgp(x)`, `beta=None`) is safe: the HSGP slot never indexes `beta`.
- Coordinate registration: in the coords loop (`:390-394`), for each HSGP term add `coords[f"{lhs}_{var}_hsgp"] = list(range(call.m))`. `m` is known at parse time, so no `add_coord` inside the model is required.
- `_make_cross_sectional_resolver` (`:555`): the factory must gain two parameters, `lhs: str` and `priors: PriorConfig`, because the current signature `(data, data_vars, endogenous_rvs, transform_map, transform_param_rvs, panel_info)` exposes neither and HSGP dispatch needs both. The per-equation call site (`:468`) passes `lhs=var, priors=priors` (both are in scope there). Dispatch `slot.kind == "hsgp"` to `assemble_hsgp_term`, passing the reshaped input tensor. The input is resolved from `data_vars[slot.name]` (the `pm.Data` node) and reshaped to `(n, 1)` inside the graph via `data_vars[slot.name][:, None]`, which keeps `do(set={var: ...})` working through broadcasting.

```python
if slot.kind == "hsgp":
    assert slot.hsgp is not None
    x_node = data_vars.get(slot.name)
    x = (x_node if x_node is not None else _resolve_var(slot.name))[:, None]
    return assemble_hsgp_term(slot.hsgp, x, lhs=lhs, priors=priors)
```

- Model-level panel guard: the scan resolver only runs when `_has_temporal_deps` is true, so a `panel={...}` model with no `lag()`/adstock terms compiles on the cross-sectional path and would silently bypass a scan-only guard. Add an explicit pre-compile check in `compile_to_pymc` (and mirror it in `model()` validation in `pathmc/_model.py` for an early, friendly error): if `panel_info is not None` and the spec contains any HSGP term, raise `NotImplementedError` ("HSGP terms are not supported in panel models yet (see #360 follow-up). Fit the HSGP smooth in a cross-sectional model, or remove the hsgp() term."). This is the guard the parity table and compile test rely on.
- `_make_scan_resolver` (`:600`): dispatch `slot.kind == "hsgp"` to `raise NotImplementedError("HSGP terms are not supported in panel/scan models yet (see #360 follow-up). Use a cross-sectional model or remove the hsgp() term.")`. This is defense-in-depth behind the model-level guard above.
- `_has_temporal_deps` (`~:1102`): an HSGP term alone must not report a temporal dependency, so a cross-sectional model with only `hsgp()` does not get routed to the scan compiler.
- Residual-covariance block (`:745`, `:788-811`): HSGP does not flow through this path safely. `_compile_residual_block` creates `beta_{var}` whenever `has_free = any(s.coeff_type == "free" for s in ms.slots)` (`:790`), using `dims=f"{var}_predictors"`; because HSGP slots are `coeff_type="free"` but excluded from `{var}_predictors`, the block would allocate a `beta` against a missing coordinate, and the block resolver at `:803` is constructed without `lhs`/`priors`. Phase 1 therefore rejects HSGP inside a residual-covariance block: before dispatching to `_compile_residual_block`, if any block-member regression contains an HSGP term, raise a clear `NotImplementedError` naming the limitation and pointing to a follow-up (for example "HSGP terms are not supported on a variable that participates in a ~~ residual-covariance block yet (see #360 follow-up)."). A dedicated compile test asserts the raise.
- `simulate()` guardrail (`pathmc/simulate.py` and the `simulate()` entry in `pathmc/_model.py`): raise a clear `NotImplementedError` when the spec contains any HSGP term, mirroring the existing rejection of `~~` residual covariances. Message names the limitation and suggests `model().fit().do()` instead.
- `design(var)` / `build_design_matrix`: leaves the raw input column in the introspection design matrix for Phase 1 (the basis expansion is an internal graph detail). No patsy change is needed. Documented as a known limitation.

`assemble_hsgp_term` responsibilities (in `pathmc/hsgp.py`): resolve `ell`/`eta` priors from the merged `priors` config (default or user override), build `cov_func` via `make_cov_func`, call `hsgp_basis` to get `(phi, sqrt_psd, n_basis)` (which builds `pm.gp.HSGP` with the sequence-wrapped `m=[call.m]` and `L=[call.L]`), assert `n_basis == call.m`, create `beta_hsgp_{lhs}_{var}` per the parametrization branch, and return `pm.Deterministic(f"f_{lhs}_{var}", ...)`. All PyMC object creation happens inside the active model context supplied by the caller (`pm.Model.get_context()`), consistent with how transforms emit RVs.

## 7. Priors defaults/overrides and introspection

Priors (`pathmc/priors.py`):

- In `default_priors`, after the transform loop (`:99-103`), for each term with `term.hsgp is not None` call a new helper:

```python
def _collect_hsgp_defaults(lhs: str, call: HSGPCall, priors: PriorConfig) -> None:
    """Register default HSGP hyperpriors (ell, eta, beta_hsgp) for one term."""
```

  It adds `ell_{lhs}_{var}` (InverseGamma), `eta_{lhs}_{var}` (HalfNormal), and `beta_hsgp_{lhs}_{var}` (Normal(0, 1) with `dims=("{lhs}_{var}_hsgp",)`). Deduplicate by RV name, mirroring `seen_transform_params`.
- Overrides flow through the existing `merge_priors`; unknown keys still raise the existing helpful error. Keys are the flat RV names above.

Introspection (`pathmc/introspect.py`):

- `_format_term` (`:346`): when `t.hsgp is not None`, render `f_hsgp({var})` (no coefficient prefix, since prefixes are rejected at parse time).
- `_format_term_latex` (`:447`): render `f_{\mathrm{hsgp}}(\mathrm{var})`.
- `build_dag_viz` (`:196`): draw an edge `var -> lhs` labeled `hsgp(var)` with a distinct style (dotted) so the smooth reads as nonparametric; no extra nodes for `ell`/`eta`.
- `model.priors()` (`build_priors` / `_collect_transform_priors` at `:628`): include the HSGP hyperpriors. Add a sibling collector `_collect_hsgp_priors` so the priors listing and `default_priors` stay consistent (both keyed by the same RV names).

## 8. Code-quality conventions and testing structure

### 8.1 Conventions the implementation must follow

- Full type hints on every new function, method, and dataclass (public and private). Introduce the `TensorLike` alias in `pathmc/hsgp.py` for tensor/ndarray unions. `mypy` must pass under `make lint`.
- NumPy-style docstrings (matching existing modules such as `build_mu_specs`) on all public functions, methods, and classes, with `Parameters`, `Returns`, and `Raises` sections and a short usage example for the user-facing DSL and priors.
- Error messages name the problem and suggest a fix (AGENTS.md). No narrating comments; comments explain why, not what. Line length 88; `ruff` and `ruff-format` clean.
- Keep pure functions pure so they are testable without sampling: `_parse_hsgp_expr` (no PyMC), `make_cov_func` and `hsgp_basis` (build tensors given inputs), `_collect_hsgp_defaults` (no PyMC). `assemble_hsgp_term` is the only piece requiring a model context.

### 8.2 Test files and what each asserts

Mirror the transform test triad (parse / compile / do) and add a focused unit-test module. Follow existing conventions: `@pytest.mark.slow` for anything that fits, the `conftest.py` autouse fixture that forces `pm.sample()` to `draws=50, tune=50, chains=1`, deterministic synthetic fixtures with a known data-generating process, and the AGENTS.md rule to add tests without editing existing ones.

`tests/test_hsgp_parse.py` (fast, no PyMC):

- `hsgp(x, m=20, c=1.5)` -> `Term(variable="x", hsgp=HSGPCall(variable="x", m=20, c=1.5, cov="expquad", centered=False))`.
- Literal coercion: `m` is `int`, `c`/`L` are `float`, `cov` is a lowercased `str`, `centered` is `bool`.
- `L=` variant; `cov='matern52'` variant; `centered=true` variant.
- Guardrail error paths: missing `m`; both `c` and `L`; neither `c` nor `L`; unknown kwarg (`by=`); multiple positional inputs; `hsgp` nested inside `adstock`; `hsgp` inside an interaction; a coefficient prefix (`2*hsgp(x, ...)` and `b*hsgp(x, ...)`); duplicate `(lhs, variable)` HSGP terms in one equation (`Y ~ hsgp(x, m=20, c=1.5) + hsgp(x, m=30, c=2.0)`).

`tests/test_hsgp.py` (fast, unit, minimal `pm.Model` context):

- `make_cov_func` returns an `ExpQuad`/`Matern52`/`Matern32` instance for each `cov`; unknown `cov` raises `ValueError`.
- `hsgp_basis` returns `phi` shape `(n, m)`, `sqrt_psd` shape `(m,)`, and `n_basis == m` for a 1-D input, both for the `c=` and the explicit `L=` forms (the scalar `L` is wrapped into `[L]`, so an `L=`-configured `HSGPCall` builds without a `ValueError`).
- `assemble_hsgp_term` in non-centered mode creates `ell_Y_x`, `eta_Y_x`, `beta_hsgp_Y_x` and an `f_Y_x` deterministic of shape `(n,)`.
- `assemble_hsgp_term` in centered mode uses `sqrt_psd` as `beta` sigma and `f = phi @ beta` (assert the graph differs from non-centered).

`tests/test_hsgp_compile.py` (fast):

- `pathmc.model("Y ~ hsgp(x, m=15, c=1.5)", data=df)` compiles to a `pm.Model`.
- `free_RVs` contain `ell_Y_x`, `eta_Y_x`, `beta_hsgp_Y_x`; `beta_hsgp_Y_x` has size `m`; the `f_Y_x` deterministic has shape `(n_obs,)`.
- HSGP input is absent from the `Y_predictors` coordinate (no spurious `beta_Y` column).
- `Y ~ 0 + hsgp(x)` compiles with `beta=None` (no `beta_Y` RV).
- Each `cov` value and both `centered` values compile; an `L=`-configured term compiles (no scalar-`L` `ValueError`).
- `model.priors()` lists the HSGP hyperpriors.
- `panel={"unit": ..., "time": ...}` with an HSGP term raises `NotImplementedError` (model-level guard, even without lags); an HSGP term on a variable in a `~~` residual-covariance block raises `NotImplementedError`; `pathmc.simulate("Y ~ hsgp(x, ...)", ...)` raises `NotImplementedError`.

`tests/test_hsgp_do.py` (slow, `@pytest.mark.slow`):

- After `fit`, `do(set={"x": grid})` changes `f_Y_x` / the predicted mean relative to baseline (the acceptance-criteria intervention test that the basis recomputes from `pm.Data`).
- Recovery: simulate `y = sin(2*x) + noise` on a grid, fit `y ~ hsgp(x, m=..., c=...)`, and assert the posterior-mean `f_Y_x` correlates with the true smooth at `> 0.9` (Bambi `hsgp_1d` parity).

`tests/test_priors.py` (additions, fast):

- `default_priors` for a spec with `hsgp(x)` includes `ell_Y_x` (InverseGamma), `eta_Y_x` (HalfNormal), and `beta_hsgp_Y_x` (Normal) with the expected parameters and dims.
- Overrides `priors={"ell_Y_x": Prior(...), "eta_Y_x": Prior(...)}` round-trip through `merge_priors`; an unknown key still raises.

Gate commands (AGENTS.md): `uv run pytest tests/test_hsgp*.py -x -v`, then `make test-fast`, then `make lint`.

### 8.3 Docs

Add `docs/examples/<group>/hsgp_1d.qmd` mirroring Bambi's `hsgp_1d`: simulate a 1-D smooth with numpy (transform-style DGP, since `simulate()` does not cover HSGP), build `pathmc.model("y ~ hsgp(x, m=..., c=...)", ...)`, show `.equations()`, `.fit()`, plot the posterior `f_Y_x` against the truth, and demonstrate a `do()` grid over `x`. Freeze the executable page (`uv run great-docs freeze docs/examples/<group>/hsgp_1d.qmd`) and commit `_freeze/`.

## 9. Implementation order (suggested)

1. `pathmc/parse.py`: `HSGPCall`, `Term.hsgp`, `_parse_hsgp_expr`, guardrails. Land `tests/test_hsgp_parse.py`.
2. `pathmc/hsgp.py`: `make_cov_func`, `hsgp_basis`, `assemble_hsgp_term`. Land `tests/test_hsgp.py`.
3. `pathmc/priors.py`: `_collect_hsgp_defaults`. Extend `tests/test_priors.py`.
4. `pathmc/compile.py`: slot kind, `build_mu` branch, coord registration, resolver dispatch, guardrails. Land `tests/test_hsgp_compile.py`.
5. `pathmc/introspect.py`: equation/LaTeX/DAG/prior rendering.
6. Slow recovery and do() tests: `tests/test_hsgp_do.py`.
7. Docs example and freeze.
