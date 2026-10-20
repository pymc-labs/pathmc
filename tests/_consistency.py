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
"""Adversarial graph-consistency checks for compiled pathmc models.

Motivation (issues #316 and #326)
---------------------------------
The suite's "replace posterior with predictive sampling" speed trick only
exercises the *forward / generative* graph. Issue #316 lived entirely in the
**logp / gradient graph** that ``pm.sample`` rebuilds (via
``join_nonshared_inputs`` -> ``clone_replace``): the generative ``mu`` was
correct, but the ``mu`` the likelihood/gradient actually used was corrupted,
so NUTS optimized the wrong objective. No predictive-sampling test can see a
logp-graph-only defect.

These helpers are the cheap, deterministic, *no-sampling* layer that does
touch that graph. The compile matrix is :class:`ConsistencyCell`. Tier tests
filter that table and call :meth:`ConsistencyCell.model`; they do not restate
specs. Recovery is the same record plus ``params`` and ``truth``, through
:func:`recover`.

Tier 3 metamorphic helpers (issue #412) compare two builds of the same
structure. They do not need a known parameter value.

This module is intentionally not named ``test_*`` so pytest does not collect
it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor

import pathmc

RECOVER_TOL = 0.15


def _mu_deterministics(pm_model):
    """Every ``mu_<y>`` deterministic in the model (one per likelihood mean)."""
    return [d for d in pm_model.deterministics if d.name.startswith("mu_")]


def _jittered_point(pm_model, seed):
    """Point dict in transformed space, jittered off the prior mode.

    The only jitter in this module. The initial point alone is a poor probe:
    many bugs only show up once a coefficient is non-zero (a flat ``mu`` is
    invariant to the #316 corruption).
    """
    point = pm_model.initial_point(random_seed=seed)
    rng = np.random.default_rng(seed)
    jittered = {}
    for name, val in point.items():
        base = np.asarray(val, dtype=float)
        jittered[name] = base + 0.5 * rng.normal(size=base.shape)
    return jittered


def _compiled_values(fn, inputs, expected, names):
    result = fn(*inputs)
    if not isinstance(result, (list, tuple)):
        result = [result]
    else:
        result = list(result)
    if len(result) < expected:
        raise RuntimeError(
            f"eval_mu compiled {len(names)} node(s) {list(names)} but got "
            f"{len(result)} value(s). Refusing to pair them by position."
        )
    return [np.asarray(value) for value in result[:expected]]


def eval_mu(model, names, point):
    """Evaluate named deterministics at *point*. The only compile-and-eval.

    *names* are PyMC variable names (``mu_Y``, ``sigma_Y``, ...). Compiles
    those nodes alone and jointly with ``logp`` — the graph ``pm.sample``
    optimizes — and returns ``(alone, joint)``. Each is a list of arrays in
    *names* order. Callers that only need the forward graph use the first
    list. Both graphs are produced here so the #316 comparison cannot drift
    into a second helper or a flag.
    """
    pm_model = model.pymc_model
    value_vars = pm_model.value_vars
    nodes = [pm_model[name] for name in names]
    valued = list(pm_model.replace_rvs_by_values(nodes))
    inputs = [point[v.name] for v in value_vars]
    alone_fn = pytensor.function(value_vars, valued, on_unused_input="ignore")
    joint_fn = pytensor.function(
        value_vars,
        [*valued, pm_model.logp()],
        on_unused_input="ignore",
    )
    alone = _compiled_values(alone_fn, inputs, len(names), names)
    joint = _compiled_values(joint_fn, inputs, len(names) + 1, names)
    return alone, joint[: len(names)]


def mu_context_discrepancy(model, *, seed=0):
    """Per-node max-abs gap between the generative ``mu`` and the logp-graph ``mu``.

    For every ``mu_<y>`` deterministic, evaluate it (a) on its own and (b)
    jointly with the model's ``logp`` scalar, at the *same* perturbed point.
    With a correct compiler the two are bit-for-bit equal; a nonzero gap means
    a graph rewrite changed ``mu`` once the logp/gradient was in scope — exactly
    the issue #316 signature.

    Returns
    -------
    dict[str, float]
        ``{deterministic_name: max_abs_difference}``. Empty if the model has no
        ``mu_<y>`` deterministics.
    """
    pm_model = model.pymc_model
    mu_dets = _mu_deterministics(pm_model)
    if not mu_dets:
        return {}

    point = _jittered_point(pm_model, seed)
    names = [det.name for det in mu_dets]
    alone, joint = eval_mu(model, names, point)
    return {
        name: float(np.max(np.abs(a - j)))
        for name, a, j in zip(names, alone, joint, strict=True)
    }


def assert_mu_context_invariant(model, *, seed=0, atol=1e-8):
    """Assert every generative ``mu`` matches the ``mu`` the logp graph uses.

    This is the Tier-0 check from #326 and the red test for #316.
    """
    discrepancy = mu_context_discrepancy(model, seed=seed)
    offenders = {name: gap for name, gap in discrepancy.items() if gap > atol}
    assert not offenders, (
        "Generative mu disagrees with the mu used by the logp/gradient graph "
        f"(max abs diff per node: {offenders}). This is the issue #316 class: "
        "the mean pm.sample optimizes is not the model's generative mean."
    )


def assert_logp_and_grad_finite(model, *, seed=0):
    """Assert ``logp`` and ``dlogp`` are finite at a perturbed point.

    A cheap Tier-2 guard: a scan/clone defect frequently surfaces as a
    non-finite gradient even when ``logp`` itself looks fine, and NUTS only
    ever consumes the gradient.
    """
    point = _jittered_point(model.pymc_model, seed)
    logp = float(model.pymc_model.compile_logp()(point))
    assert np.isfinite(logp), f"logp is non-finite at a perturbed point: {logp}"

    grad = np.asarray(model.pymc_model.compile_dlogp()(point))
    assert np.all(np.isfinite(grad)), (
        "dlogp has non-finite entries at a perturbed point — the gradient NUTS "
        f"consumes is broken: {grad}"
    )


def assert_model_consistent(model, *, seed=0, atol=1e-8):
    """Run the full no-sampling consistency battery on a built model.

    Bundles the Tier-0 (mu context-invariance) and Tier-2 (finite logp/grad)
    checks from #326 into one call so it can be dropped into any test or
    parametrized across the compile matrix.
    """
    assert_mu_context_invariant(model, seed=seed, atol=atol)
    assert_logp_and_grad_finite(model, seed=seed)


def gaussian_loglike(y_obs, mu, sigma):
    """Closed-form Gaussian log-likelihood — a hand oracle for Tier-1 tests."""
    y_obs = np.asarray(y_obs, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return float(
        np.sum(
            -0.5 * np.log(2.0 * np.pi * sigma**2) - 0.5 * ((y_obs - mu) / sigma) ** 2
        )
    )


def gaussian_likelihood_oracle_gap(model, *, seed=0):
    """Gap between the model's compiled likelihood logp and a hand oracle.

    The Tier-1 check from #326. For the first Gaussian observed node, build the
    log-likelihood *by hand* from the model's own forward ``mu`` (compiled
    alone, which is correct) plus the observed data and ``sigma`` at a fixed
    point, then compare to the likelihood term ``compile_logp`` actually emits.
    A large gap means the likelihood scores against a different ``mu`` than the
    generative one — independent confirmation of the issue #316 class, and the
    cleanest "the objective NUTS climbs is wrong" oracle.

    Returns
    -------
    float
        ``abs(model_likelihood_logp - hand_oracle_logp)``.
    """
    pm_model = model.pymc_model
    obs_rv = pm_model.observed_RVs[0]
    var = obs_rv.name
    point = _jittered_point(pm_model, seed)
    (mu_val, sigma_val), _joint = eval_mu(model, [f"mu_{var}", f"sigma_{var}"], point)
    mu = np.asarray(mu_val)
    sigma = float(np.asarray(sigma_val))
    y_obs = np.asarray(pm_model.rvs_to_values[obs_rv].eval())

    oracle = gaussian_loglike(y_obs, mu, sigma)
    model_loglike = float(pm_model.compile_logp(vars=[obs_rv])(point))
    return abs(model_loglike - oracle)


def _shared_points(model_a, model_b, seed, *, allow=()):
    """Jitter *model_a* and copy every key onto *model_b*.

    Raises if a name is missing or a shape differs, unless that name is in
    *allow*. A silent skip hides a panel whose unit dimension grew. Pass
    *allow* only for keys the caller expects to differ, such as a
    unit-shaped intercept after duplicating units.
    """
    allowed = set(allow)
    point_a = _jittered_point(model_a.pymc_model, seed)
    point_b = _jittered_point(model_b.pymc_model, seed)
    names_a = set(point_a)
    names_b = set(point_b)
    missing = (names_a - names_b) | (names_b - names_a)
    unexpected_names = sorted(missing - allowed)
    if unexpected_names:
        raise ValueError(
            "shared point name mismatch: "
            f"{unexpected_names}. Pass allow=[...] only for keys that are "
            "expected to differ (for example a unit intercept after adding "
            "units). Do not skip the key silently."
        )
    shape_mismatch = [
        name
        for name in sorted(names_a & names_b)
        if name not in allowed and np.shape(point_a[name]) != np.shape(point_b[name])
    ]
    if shape_mismatch:
        detail = ", ".join(
            f"{name} {np.shape(point_a[name])} vs {np.shape(point_b[name])}"
            for name in shape_mismatch
        )
        raise ValueError(
            f"shared point shape mismatch: {detail}. Pass allow=[...] only "
            "for keys that are expected to differ. A square panel does not "
            "make unequal shapes interchangeable."
        )
    for name, val in point_a.items():
        if name in allowed:
            continue
        point_b[name] = np.array(val, copy=True)
    return point_a, point_b


def _scan_info(model):
    gen = getattr(model, "_gen_model", None)
    if gen is None:
        return None
    return getattr(gen, "_pathmc_panel_scan", None)


def _sorted_unit_time_order(unit, time) -> np.ndarray:
    """Row indices that sort by ``(unit, time)``, type-agnostic."""
    keys = list(zip(np.asarray(unit).tolist(), np.asarray(time).tolist(), strict=True))
    return np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=int)


def _align_mu(model, mu):
    """Return *mu* in sorted ``(unit, time)`` row order.

    Scan means are time-major ``(n_times, n_units)``. A square panel has the
    same shape either way, so the layout comes from the compiler's scan info,
    not from sniffing the array. Non-scan panel means follow the frame's row
    order and are permuted to that same sort. A shape that is not the
    compile-time scan shape raises; it is not flattened.
    """
    mu = np.asarray(mu, dtype=float)
    scan = _scan_info(model)
    if scan is not None:
        expected = (scan.n_times, scan.n_units)
        if mu.shape != expected:
            raise ValueError(
                f"scan mu shape {tuple(mu.shape)} does not match compile-time "
                f"(n_times, n_units)={expected}. Refusing to guess layout from "
                "the array shape: a square panel makes that sniff ambiguous. "
                "Pass the scan-compiled mean."
            )
        return np.swapaxes(mu, 0, 1).reshape(-1)

    flat = np.reshape(mu, -1) if mu.ndim == 1 else None
    panel = model._panel_info
    if panel is None:
        if flat is None:
            raise ValueError(
                f"cross-sectional mu shape {tuple(mu.shape)} is not a row "
                "vector. Refusing to reshape."
            )
        return flat

    n_rows = len(model._data)
    if flat is None or flat.shape != (n_rows,):
        raise ValueError(
            f"panel mu shape {tuple(mu.shape)} is not a length-{n_rows} row "
            "vector. Refusing to reshape; a square panel makes a "
            "(n_times, n_units) sniff ambiguous. Scan means are aligned from "
            "the compiler's scan info, not from the array shape."
        )
    unit = model._data[panel.unit].to_numpy()
    time = model._data[panel.time].to_numpy()
    return flat[_sorted_unit_time_order(unit, time)]


def assert_row_order_invariant(model_a, model_b, *, seed=0, atol=1e-6):
    """Joint logp is unchanged when observation rows are reordered.

    Catches ordering and reshape bugs. A compiler that feeds rows to scan
    in input order, instead of sorting by ``(unit, time)`` first, scores a
    different likelihood after a shuffle. Look at the sort index in
    ``_compile_scan_panel`` and the observe-path row order.
    """
    point_a, point_b = _shared_points(model_a, model_b, seed)
    logp_a = float(model_a.pymc_model.compile_logp()(point_a))
    logp_b = float(model_b.pymc_model.compile_logp()(point_b))
    assert np.isfinite(logp_a) and np.isfinite(logp_b), (
        "row-permutation invariance: joint logp is non-finite "
        f"({logp_a}, {logp_b}). Look at the scan reshape before debugging "
        "the permutation itself."
    )
    gap = abs(logp_a - logp_b)
    assert gap <= atol, (
        "row-permutation invariance failed: shuffling observation row order "
        f"changed joint logp by {gap:.3g} ({logp_a:.6g} vs {logp_b:.6g}). "
        "Look at panel sort/reshape in _compile_scan_panel."
    )


def assert_lag_matches_manual_shift(
    model_lag,
    model_shifted,
    *,
    outcome,
    seed=0,
    atol=1e-6,
):
    """Forward ``mu`` of ``y ~ lag(x)`` matches a hand-shifted column.

    The hand shift must repeat the first period and then lag by one. That
    is the scan init, not a zero pad. A mismatch means the carry used by
    the generative graph is not that lag. Look at exogenous-lag init in
    ``_compile_scan_panel``.

    Both means are compared in sorted ``(unit, time)`` order. Scan layout
    comes from the compiler's scan info, not from an array-shape sniff, so
    a square panel is not ambiguous.
    """
    point_lag, point_shifted = _shared_points(model_lag, model_shifted, seed)
    (mu_lag,), _joint_lag = eval_mu(model_lag, [f"mu_{outcome}"], point_lag)
    (mu_shifted,), _joint_shifted = eval_mu(
        model_shifted, [f"mu_{outcome}"], point_shifted
    )
    aligned_lag = _align_mu(model_lag, mu_lag)
    aligned_shifted = _align_mu(model_shifted, mu_shifted)
    gap = float(np.max(np.abs(aligned_lag - aligned_shifted)))
    assert gap <= atol, (
        "lag-definition equivalence failed: forward mu of y ~ lag(x) differs "
        f"from y ~ x_shifted by {gap:.3g}. The manual column must repeat the "
        "first period (scan init), not zero-pad. Look at exogenous-lag init "
        "in _compile_scan_panel."
    )


def assert_duplicated_observations_double_likelihood(
    model, model_dup, *, seed=0, rtol=1e-6, atol=1e-6, allow=()
):
    """Duplicating every unit (or every cross-sectional row) doubles likelihood.

    The check uses the observation likelihood only. Joint logp is not
    doubled, because priors are counted once. A ratio far from 2 means
    rows were dropped or scored twice. Look at the likelihood reshape in
    the scan compiler.

    Do not call this on partial pooling. Unit-shaped coefficients change
    the parameter space when units are added, so there is no shared point.
    ``_shared_points`` raises on that mismatch unless the caller passes an
    allow-list. Endogenous lag grows ``carry_innovations_*`` with the unit
    count; pass that name in *allow* rather than skipping it silently.
    """
    point, point_dup = _shared_points(model, model_dup, seed, allow=allow)
    observed = list(model.pymc_model.observed_RVs)
    observed_dup = list(model_dup.pymc_model.observed_RVs)
    like = float(model.pymc_model.compile_logp(vars=observed)(point))
    like_dup = float(model_dup.pymc_model.compile_logp(vars=observed_dup)(point_dup))
    assert np.isfinite(like) and np.isfinite(like_dup), (
        "duplicate-data scaling: observation likelihood is non-finite "
        f"({like}, {like_dup}). Look at the likelihood reshape in the scan "
        "compiler."
    )
    expected = 2.0 * like
    gap = abs(like_dup - expected)
    tol = atol + rtol * abs(expected)
    assert gap <= tol, (
        "duplicate-data scaling failed: duplicating observations should "
        f"double the likelihood ({like:.6g} -> {expected:.6g}) but got "
        f"{like_dup:.6g}. Look at the likelihood reshape in the scan "
        "compiler. This compares likelihood only, not joint logp."
    )


def assert_predictor_scale_covariance(
    model,
    model_scaled,
    *,
    outcome,
    scale,
    beta_index=0,
    seed=0,
    atol=1e-6,
):
    """``mu(c * x, beta / c)`` matches ``mu(x, beta)`` in both graphs.

    *model_scaled* is the same specification on data whose target predictor
    was multiplied by *scale*. The matching slope entry is divided by
    *scale*. Both the forward ``mu`` and the ``mu`` compiled jointly with
    ``logp`` are checked, in sorted ``(unit, time)`` order. A gap in only
    the joint graph is the issue #316 class: the likelihood mean does not
    see the scaled design. Look at predictor wiring in the scan step.
    """
    point, point_scaled = _shared_points(model, model_scaled, seed)
    beta_name = f"beta_{outcome}"
    scaled_beta = np.array(point[beta_name], dtype=float, copy=True)
    scaled_beta[beta_index] = scaled_beta[beta_index] / scale
    point_scaled[beta_name] = scaled_beta

    alone, joint = eval_mu(model, [f"mu_{outcome}"], point)
    alone_scaled, joint_scaled = eval_mu(model_scaled, [f"mu_{outcome}"], point_scaled)
    graphs = (
        ("forward", alone[0], alone_scaled[0]),
        ("logp-graph", joint[0], joint_scaled[0]),
    )
    for label, mu, mu_scaled in graphs:
        aligned = _align_mu(model, mu)
        aligned_scaled = _align_mu(model_scaled, mu_scaled)
        gap = float(np.max(np.abs(aligned - aligned_scaled)))
        assert gap <= atol, (
            "scale/shift covariance failed: "
            f"{label} mu(c * x, beta / c) differs from mu(x, beta) by "
            f"{gap:.3g} (scale={scale}, beta index {beta_index}). "
            "Look at predictor wiring in the scan step and in the logp "
            "graph. A joint-graph-only gap is the issue #316 class."
        )


# ---------------------------------------------------------------------------
# Compile matrix. One record per cell. Tiers filter this table.
# ---------------------------------------------------------------------------


def _xsec_data(seed=1, n=150):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 + 0.3 * x2 + rng.normal(scale=0.5, size=n)
    counts = rng.poisson(np.exp(0.2 * x1), size=n)
    binary = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.5 * x1)))).astype(int)
    return pd.DataFrame({"X1": x1, "X2": x2, "Y": y, "Ycount": counts, "Ybin": binary})


def _panel_data(seed=1, ngeo=5, ntime=12, spend_loc=10.0):
    rng = np.random.default_rng(seed)
    frames = []
    for g in range(ngeo):
        spend = rng.normal(spend_loc, 1, ntime)
        x2 = rng.normal(0, 1, ntime)
        sales = np.ones(ntime) * 20
        sales[1:] = 10 + spend[:-1] + rng.normal(0, 1, ntime - 1)
        frames.append(
            pd.DataFrame({
                "spend": spend,
                "x2": x2,
                "sales": sales,
                "week": np.arange(ntime),
                "geo": g,
            })
        )
    return pd.concat(frames, ignore_index=True)


def _mediation_data(seed=42, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    y = 0.8 * m + 0.3 * x + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "M": m, "Y": y})


def _panel_count_data(seed=1, ngeo=5, ntime=12):
    """Panel frame with a small non-negative count outcome for negbinomial."""
    df = _panel_data(seed=seed, ngeo=ngeo, ntime=ntime).copy()
    rng = np.random.default_rng(seed + 7)
    rate = np.exp(0.3 * df["x2"].to_numpy())
    df["count"] = rng.poisson(rate)
    return df


_PANEL = {"unit": "geo", "time": "week"}


@dataclass(frozen=True)
class ConsistencyCell:
    """One compile-matrix cell.

    ``gaussian_oracle`` is explicit. Do not infer it from the id string or by
    probing the built model for ``sigma_*``. Recovery, when present, is this
    same record plus ``params`` and ``truth``. Partial pooling is MCMC-only:
    those cells do not share a stable mode under ``find_MAP``.
    """

    id: str
    spec: str
    data_factory: Callable[..., pd.DataFrame]
    panel: dict[str, str] | None = None
    pooling: str | dict | None = None
    families: dict[str, str] | None = None
    gaussian_oracle: bool = False
    params: Mapping[str, Any] | None = None
    truth: Mapping[str, Mapping[str, float]] | None = None
    recover_data: Mapping[str, Any] | None = None
    recover_seed: int = 0
    recover_methods: frozenset[str] = frozenset()

    def data(self, **kwargs) -> pd.DataFrame:
        """Build this cell's frame. Keyword arguments override factory defaults."""
        return self.data_factory(**kwargs)

    def model(self, data, *, spec: str | None = None, priors=None):
        """Build this cell on *data*.

        *spec* overrides the formula only for a metamorphic twin (a
        hand-shifted lag column). *priors* overrides defaults for the light
        SBC draw. Panel, pooling, and families always come from the record.
        """
        return pathmc.model(
            self.spec if spec is None else spec,
            data=data,
            panel=self.panel,
            pooling=self.pooling,
            families=self.families,
            priors=priors,
        )


CELLS: tuple[ConsistencyCell, ...] = (
    ConsistencyCell(
        id="xsec-gaussian",
        spec="Y ~ X1 + X2",
        data_factory=_xsec_data,
        gaussian_oracle=True,
        params={"beta_Y": [0.2, 1.5, 0.3], "sigma_Y": 0.3},
        truth={"Y": {"Intercept": 0.2, "X1": 1.5, "X2": 0.3}},
        recover_data={"n": 400},
        recover_seed=1,
        recover_methods=frozenset({"map"}),
    ),
    # Two-outcome SCM (mu_M and mu_Y). The hand oracle scores one observed
    # RV, so this cell is not in the Tier-1 set.
    ConsistencyCell(
        id="xsec-mediation-2outcome",
        spec="M ~ a*X\nY ~ b*M + c*X",
        data_factory=_mediation_data,
        params={
            "beta_M": [0.0, 0.8],
            "sigma_M": 0.4,
            "beta_Y": [0.0, 0.7, 0.3],
            "sigma_Y": 0.4,
        },
        truth={"M": {"X": 0.8}, "Y": {"M": 0.7, "X": 0.3}},
        recover_data={"n": 500, "seed": 0},
        recover_seed=41501,
        recover_methods=frozenset({"mcmc"}),
    ),
    # Gaussian, but not in the Tier-1 set. Membership is the flag, not a
    # prefix of the id.
    ConsistencyCell(
        id="xsec-interaction",
        spec="Y ~ X1 + X1:X2",
        data_factory=_xsec_data,
    ),
    ConsistencyCell(
        id="xsec-bernoulli",
        spec="Ybin ~ X1",
        data_factory=_xsec_data,
        families={"Ybin": "bernoulli"},
        params={"beta_Ybin": [-0.2, 1.2]},
        truth={"Ybin": {"Intercept": -0.2, "X1": 1.2}},
        # n=400 leaves the intercept inside 0.15 with almost no margin for
        # the shared MAP and MCMC truth. n=4000 keeps both methods off that edge.
        recover_data={"n": 4000},
        recover_seed=2,
        recover_methods=frozenset({"map", "mcmc"}),
    ),
    ConsistencyCell(
        id="xsec-poisson",
        spec="Ycount ~ X1",
        data_factory=_xsec_data,
        families={"Ycount": "poisson"},
        params={"beta_Ycount": [0.2, 0.6]},
        truth={"Ycount": {"Intercept": 0.2, "X1": 0.6}},
        recover_data={"n": 400},
        recover_seed=2,
        recover_methods=frozenset({"map"}),
    ),
    ConsistencyCell(
        id="xsec-negbinomial",
        spec="Ycount ~ X1",
        data_factory=_xsec_data,
        families={"Ycount": "negbinomial"},
    ),
    ConsistencyCell(
        id="panel-plain-complete",
        spec="sales ~ x2",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
    ),
    ConsistencyCell(
        id="panel-plain-partial",
        spec="sales ~ x2",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling="partial",
        gaussian_oracle=True,
    ),
    # Lag of the outcome is a distinct scan path from exogenous lag. The
    # #316 corruption was specific to exogenous lag carries.
    ConsistencyCell(
        id="panel-lag(y)-complete",
        spec="sales ~ lag(sales) + x2",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
    ),
    ConsistencyCell(
        id="panel-lag(x)-complete",
        spec="sales ~ lag(spend)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
        params={"beta_sales": [0.5, 1.0], "sigma_sales": 0.4},
        truth={"sales": {"Intercept": 0.5, "lag(spend)": 1.0}},
        # Center spend so the intercept is not collinear with lag(spend).
        recover_data={"ngeo": 8, "ntime": 40, "spend_loc": 0.0},
        recover_seed=3,
        recover_methods=frozenset({"map"}),
    ),
    ConsistencyCell(
        id="panel-lag(x)-partial",
        spec="sales ~ lag(spend)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling="partial",
        gaussian_oracle=True,
        params={
            "beta_sales": [1.0],
            "sigma_sales": 0.4,
            "alpha_sales": np.linspace(-0.4, 0.4, 8),
            "mu_alpha_sales": 0.0,
            "sigma_alpha_sales": 0.3,
        },
        # The formula intercept is absorbed by the unit intercepts.
        truth={"sales": {"lag(spend)": 1.0}},
        recover_data={"ngeo": 8, "ntime": 40, "spend_loc": 0.0},
        recover_seed=41502,
        recover_methods=frozenset({"mcmc"}),
    ),
    ConsistencyCell(
        id="panel-lag(x)-no-intercept",
        spec="sales ~ 0 + lag(spend)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling="partial",
        gaussian_oracle=True,
    ),
    # Geometric adstock forces the scan-panel compiler (same family as #316).
    ConsistencyCell(
        id="panel-adstock-complete",
        spec="sales ~ adstock(spend, decay=d)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
    ),
    # Co-occurring terms: lag(adstock(...)) is a parse error.
    ConsistencyCell(
        id="panel-lag(x)-adstock",
        spec="sales ~ lag(spend) + adstock(x2, decay=d)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
    ),
    ConsistencyCell(
        id="panel-negbinomial",
        spec="count ~ x2",
        data_factory=_panel_count_data,
        panel=_PANEL,
        families={"count": "negbinomial"},
    ),
    # Slopes bind to a raw column. lag(spend) is not a data column, so the
    # random slope is x2, which does enter mu.
    ConsistencyCell(
        id="panel-lag(x)-slopes",
        spec="sales ~ lag(spend) + x2",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling={"intercept": True, "slopes": ["x2"]},
        gaussian_oracle=True,
    ),
    ConsistencyCell(
        id="panel-logistic-saturation",
        spec="sales ~ logistic_saturation(spend, lam=lam)",
        data_factory=_panel_data,
        panel=_PANEL,
        pooling=None,
        gaussian_oracle=True,
    ),
)


def cells(*ids: str) -> list[ConsistencyCell]:
    """Return compile-matrix cells in the requested id order."""
    by_id = {cell.id: cell for cell in CELLS}
    missing = [cell_id for cell_id in ids if cell_id not in by_id]
    if missing:
        raise KeyError(
            f"unknown consistency cell(s) {missing}. Filter CELLS by id; "
            "do not restate the spec."
        )
    return [by_id[cell_id] for cell_id in ids]


def _coefs_from_point(model, estimate, truth):
    got = {}
    for outcome, coefs in truth.items():
        names = list(model.pymc_model.coords[f"{outcome}_predictors"])
        values = np.asarray(estimate[f"beta_{outcome}"], dtype=float).reshape(-1)
        got[outcome] = {name: float(values[names.index(name)]) for name in coefs}
    return got


def _coefs_from_idata(idata, truth):
    got = {}
    for outcome, coefs in truth.items():
        posterior = idata.posterior[f"beta_{outcome}"]
        coord = f"{outcome}_predictors"
        got[outcome] = {
            name: float(posterior.sel({coord: name}).mean()) for name in coefs
        }
    return got


def recover(cell, method, *, sample_kwargs=None):
    """Simulate from ``cell.params`` and recover coefficients.

    *method* is ``"map"`` (``pm.find_MAP``) or ``"mcmc"`` (``model.fit``).
    MCMC must receive :data:`conftest.SAMPLE_KWARGS`; do not pass a local
    draws/tune copy. Returns ``{outcome: {predictor: estimate}}`` for the
    keys in ``cell.truth``.
    """
    if cell.params is None or cell.truth is None:
        raise ValueError(
            f"{cell.id} has no recovery params/truth. Set them on the cell "
            "instead of restating the spec in a tier test."
        )
    if method not in {"map", "mcmc"}:
        raise ValueError(f"recovery method must be 'map' or 'mcmc', got {method!r}.")
    if method == "mcmc" and not sample_kwargs:
        raise ValueError(
            "mcmc recovery requires sample_kwargs. Pass conftest.SAMPLE_KWARGS "
            "instead of a local draws/tune copy."
        )
    data = cell.data(**dict(cell.recover_data or {}))
    simulated = pathmc.simulate(
        cell.spec,
        data=data,
        params=dict(cell.params),
        families=cell.families,
        panel=cell.panel,
        pooling=cell.pooling,
        random_seed=cell.recover_seed,
    )
    model = cell.model(data=simulated)
    if method == "map":
        with model.pymc_model:
            estimate = pm.find_MAP(progressbar=False)
        return _coefs_from_point(model, estimate, cell.truth)
    idata = model.fit(**sample_kwargs, progressbar=False, random_seed=cell.recover_seed)
    return _coefs_from_idata(idata, cell.truth)


def assert_recovered(cell, got, *, tol=RECOVER_TOL):
    """Assert recovered coefficients match ``cell.truth`` within *tol*."""
    for outcome, coefs in cell.truth.items():
        for name, expected in coefs.items():
            value = got[outcome][name]
            assert abs(value - expected) <= tol, (
                f"{cell.spec} (seed={cell.recover_seed}): {outcome}[{name}] "
                f"expected {expected}, got {value}, tol={tol}"
            )
