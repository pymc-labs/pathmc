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
