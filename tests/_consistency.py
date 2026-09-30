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
touch that graph. They are model-agnostic so they can be parametrized across
the whole compile matrix (see ``test_graph_consistency.py``).

Tier 3 metamorphic helpers (issue #412) compare two builds of the same
structure. They do not need a known parameter value.

This module is intentionally not named ``test_*`` so pytest does not collect
it directly.
"""

from __future__ import annotations

import numpy as np
import pytensor


def _mu_deterministics(pm_model):
    """Every ``mu_<y>`` deterministic in the model (one per likelihood mean)."""
    return [d for d in pm_model.deterministics if d.name.startswith("mu_")]


def _perturbed_point(pm_model, seed):
    """A reproducible off-origin point in value (unconstrained) space.

    The initial point alone is a poor probe — many bugs only show up once a
    coefficient is non-zero (a flat ``mu`` is invariant to the #316 corruption).
    We jitter every value variable so the check actually stresses the graph.
    """
    point = pm_model.initial_point(random_seed=seed)
    rng = np.random.default_rng(seed)
    args = []
    for v in pm_model.value_vars:
        base = np.asarray(point[v.name], dtype=float)
        args.append(base + 0.5 * rng.normal(size=base.shape))
    return args


def mu_context_discrepancy(model, *, seed=0):
    """Per-node max-abs gap between the generative ``mu`` and the logp-graph ``mu``.

    For every ``mu_<y>`` deterministic, compile it (a) on its own and (b)
    jointly with the model's ``logp`` scalar, then evaluate both at the *same*
    perturbed point. With a correct compiler the two are bit-for-bit equal; a
    nonzero gap means a graph rewrite changed ``mu`` once the logp/gradient was
    in scope — exactly the issue #316 signature.

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

    value_vars = pm_model.value_vars
    # Express mu in value (sampler) space, alongside the logp it is compiled
    # with during sampling.
    mu_valued = pm_model.replace_rvs_by_values(mu_dets)
    logp = pm_model.logp()

    f_alone = pytensor.function(value_vars, mu_valued, on_unused_input="ignore")
    f_joint = pytensor.function(
        value_vars, [*mu_valued, logp], on_unused_input="ignore"
    )

    args = _perturbed_point(pm_model, seed)

    def _to_list(out):
        # pytensor.function returns a list/tuple for a multi-output graph and a
        # bare array for a single-variable output. Normalize to a list so the
        # zip below pairs every mu with its node regardless of how many there
        # are (a multi-outcome model has one mu_<y> per outcome).
        if isinstance(out, (list, tuple)):
            return list(out)
        return [out]

    alone = _to_list(f_alone(*args))
    joint = _to_list(f_joint(*args))[:-1]  # drop the trailing logp output

    return {
        det.name: float(np.max(np.abs(np.asarray(a) - np.asarray(j))))
        for det, a, j in zip(mu_dets, alone, joint)
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
    pm_model = model.pymc_model
    point = pm_model.initial_point(random_seed=seed)
    rng = np.random.default_rng(seed)
    point = {
        name: np.asarray(val, dtype=float)
        + 0.5 * rng.normal(size=np.asarray(val).shape)
        for name, val in point.items()
    }

    logp = float(pm_model.compile_logp()(point))
    assert np.isfinite(logp), f"logp is non-finite at a perturbed point: {logp}"

    grad = np.asarray(pm_model.compile_dlogp()(point))
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

    point = pm_model.initial_point(random_seed=seed)
    rng = np.random.default_rng(seed)
    point = {
        name: np.asarray(val, dtype=float)
        + 0.5 * rng.normal(size=np.asarray(val).shape)
        for name, val in point.items()
    }

    # Evaluate the *forward* mu and sigma in value space over all value vars,
    # so the function accepts the same point dict the logp does.
    value_vars = pm_model.value_vars
    mu_node, sigma_node = pm_model.replace_rvs_by_values([
        pm_model[f"mu_{var}"],
        pm_model[f"sigma_{var}"],
    ])
    forward = pytensor.function(
        value_vars, [mu_node, sigma_node], on_unused_input="ignore"
    )
    mu_val, sigma_val = forward(*[point[v.name] for v in value_vars])
    mu = np.asarray(mu_val)
    sigma = float(np.asarray(sigma_val))
    y_obs = np.asarray(pm_model.rvs_to_values[obs_rv].eval())

    oracle = gaussian_loglike(y_obs, mu, sigma)
    model_loglike = float(pm_model.compile_logp(vars=[obs_rv])(point))
    return abs(model_loglike - oracle)


def _jittered_point(pm_model, seed):
    """Point dict in transformed space, jittered off the prior mode."""
    point = pm_model.initial_point(random_seed=seed)
    rng = np.random.default_rng(seed)
    jittered = {}
    for name, val in point.items():
        base = np.asarray(val, dtype=float)
        jittered[name] = base + 0.5 * rng.normal(size=base.shape)
    return jittered


def _shared_points(model_a, model_b, seed):
    """Jitter *model_a*, then copy every same-shaped key onto *model_b*."""
    point_a = _jittered_point(model_a.pymc_model, seed)
    point_b = {
        name: np.asarray(val, dtype=float)
        for name, val in model_b.pymc_model.initial_point(random_seed=seed).items()
    }
    for name, val in point_a.items():
        if name in point_b and np.shape(point_b[name]) == np.shape(val):
            point_b[name] = np.array(val, copy=True)
    return point_a, point_b


def _eval_mu(model, outcome, point, *, with_logp):
    """Evaluate ``mu_<outcome>`` at *point*.

    When *with_logp* is true the mean is compiled in the same graph as
    ``logp``, which is the graph ``pm.sample`` actually optimizes.
    """
    pm_model = model.pymc_model
    value_vars = pm_model.value_vars
    (mu_node,) = pm_model.replace_rvs_by_values([pm_model[f"mu_{outcome}"]])
    outputs = [mu_node, pm_model.logp()] if with_logp else mu_node
    fn = pytensor.function(value_vars, outputs, on_unused_input="ignore")
    args = [point[v.name] for v in value_vars]
    result = fn(*args)
    mu = result[0] if with_logp else result
    return np.asarray(mu, dtype=float)


def _unit_major(mu, n_times, n_units):
    """Flatten scan ``(n_times, n_units)`` mu into unit-then-time row order."""
    mu = np.asarray(mu, dtype=float)
    if n_times is not None and mu.shape == (n_times, n_units):
        return np.swapaxes(mu, 0, 1).reshape(-1)
    return mu.reshape(-1)


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
    n_times,
    n_units,
    seed=0,
    atol=1e-6,
):
    """Forward ``mu`` of ``y ~ lag(x)`` matches a hand-shifted column.

    The hand shift must repeat the first period and then lag by one. That
    is the scan init, not a zero pad. A mismatch means the carry used by
    the generative graph is not that lag. Look at exogenous-lag init in
    ``_compile_scan_panel``.

    ``mu`` from a scan model is ``(n_times, n_units)``. It is compared in
    unit-then-time order, which is the row order of a frame sorted by
    ``(unit, time)``.
    """
    point_lag, point_shifted = _shared_points(model_lag, model_shifted, seed)
    mu_lag = _unit_major(
        _eval_mu(model_lag, outcome, point_lag, with_logp=False),
        n_times,
        n_units,
    )
    mu_shifted = _unit_major(
        _eval_mu(model_shifted, outcome, point_shifted, with_logp=False),
        n_times,
        n_units,
    )
    gap = float(np.max(np.abs(mu_lag - mu_shifted)))
    assert gap <= atol, (
        "lag-definition equivalence failed: forward mu of y ~ lag(x) differs "
        f"from y ~ x_shifted by {gap:.3g}. The manual column must repeat the "
        "first period (scan init), not zero-pad. Look at exogenous-lag init "
        "in _compile_scan_panel."
    )


def assert_duplicated_observations_double_likelihood(
    model, model_dup, *, seed=0, rtol=1e-6, atol=1e-6
):
    """Duplicating every unit (or every cross-sectional row) doubles likelihood.

    The check uses the observation likelihood only. Joint logp is not
    doubled, because priors are counted once. A ratio far from 2 means
    rows were dropped or scored twice. Look at the likelihood reshape in
    the scan compiler.

    Do not call this on partial pooling. Unit-shaped coefficients change
    the parameter space when units are added, so there is no shared point.
    """
    point, point_dup = _shared_points(model, model_dup, seed)
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
    n_times=None,
    n_units=None,
    seed=0,
    atol=1e-6,
):
    """``mu(c * x, beta / c)`` matches ``mu(x, beta)`` in both graphs.

    *model_scaled* is the same specification on data whose target predictor
    was multiplied by *scale*. The matching slope entry is divided by
    *scale*. Both the forward ``mu`` and the ``mu`` compiled jointly with
    ``logp`` are checked. A gap in only the joint graph is the issue #316
    class: the likelihood mean does not see the scaled design. Look at
    predictor wiring in the scan step.
    """
    point, point_scaled = _shared_points(model, model_scaled, seed)
    beta_name = f"beta_{outcome}"
    scaled_beta = np.array(point[beta_name], dtype=float, copy=True)
    scaled_beta[beta_index] = scaled_beta[beta_index] / scale
    point_scaled[beta_name] = scaled_beta

    for with_logp, label in ((False, "forward"), (True, "logp-graph")):
        mu = _unit_major(
            _eval_mu(model, outcome, point, with_logp=with_logp),
            n_times,
            n_units,
        )
        mu_scaled = _unit_major(
            _eval_mu(model_scaled, outcome, point_scaled, with_logp=with_logp),
            n_times,
            n_units,
        )
        gap = float(np.max(np.abs(mu - mu_scaled)))
        assert gap <= atol, (
            "scale/shift covariance failed: "
            f"{label} mu(c * x, beta / c) differs from mu(x, beta) by "
            f"{gap:.3g} (scale={scale}, beta index {beta_index}). "
            "Look at predictor wiring in the scan step and in the logp "
            "graph. A joint-graph-only gap is the issue #316 class."
        )
