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
"""Item 8 (issue #327) / #191: non-Gaussian mediators in the scan compiler.

Two independent concerns:

1. Non-Gaussian endogenous variables used as predictors in scan-compiled panel
   models must carry *sampled* Bernoulli/count values through the temporal
   recursion, not response-scale means. Issue #191 added stochastic discrete
   carry (uniform innovations for Bernoulli, ``pt.random.poisson`` /
   ``pt.random.negative_binomial`` for count families). These tests enumerate
   the topologies (plain term, self-lag, cross-equation lag, single transform,
   nested transform, interaction, random slope, and transitive chains) and
   prove each one compiles with the expected innovation nodes, while
   confirming terminal outcomes and cross-sectional models stay unchanged.

2. The ``clip(mu, -20, 20)`` guard on Poisson/NegativeBinomial rates
   inside the scan (``pathmc.compile._SCAN_MU_CLIP_BOUND``) is a
   numerical-stability guard, not a modeling choice, but it silently
   truncates genuinely high rates. ``_warn_high_rate_clip_risk`` surfaces
   that via a ``UserWarning`` at compile time instead of staying silent;
   these tests check the warning fires/doesn't fire at the right boundary.

All of this is resolved at model-construction time (no MCMC), so these
tests are intentionally cheap.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc

NON_GAUSSIAN_FAMILIES = ["bernoulli", "poisson", "negbinomial"]


def _panel_data(n_units: int = 2, n_weeks: int = 8, seed: int = 42) -> pd.DataFrame:
    """Generic panel data with columns usable across all topology tests.

    Values don't need to be causally consistent with the specs under
    test -- these tests only exercise compile-time validation, never
    fitting/sampling.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        for week in range(1, n_weeks + 1):
            rows.append({
                "region": f"R{unit}",
                "week": week,
                "X": rng.normal(),
                "Z": rng.normal(),
                "M": float(rng.integers(0, 2)),
                "N": float(rng.integers(0, 5)),
                "Y": rng.normal(),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel_data() -> pd.DataFrame:
    return _panel_data()


def _assert_discrete_sample_nodes(model, family: str, var: str = "M") -> None:
    """Assert the scan model has the expected stochastic discrete carry nodes."""
    nv = model.pymc_model.named_vars
    assert f"mu_{var}" in nv or var in nv
    if family == "bernoulli":
        assert f"discrete_uniforms_{var}" in nv
    assert "_use_observed_carry" in nv


class TestDiscreteSampleCompilation:
    """Each of these specs has *some* temporal dep (so the scan compiler
    engages) and uses a non-Gaussian endogenous node as a predictor
    somewhere -- every one of them must compile with stochastic carry."""

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_plain_term(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ M + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_self_lag(self, panel_data, family):
        model = pathmc.model(
            "M ~ X + lag(M)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_cross_equation_lag(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ lag(M)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_single_transform(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ adstock(M, decay=theta) + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_nested_transform(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ adstock(logistic_saturation(M, lam=lam), decay=theta) + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_interaction_term(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ M:Z + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        _assert_discrete_sample_nodes(model, family)

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_random_slope_term(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ M + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
            pooling={"intercept": True, "slopes": ["M"]},
        )
        _assert_discrete_sample_nodes(model, family)

    def test_transitive_chain_compiles(self, panel_data):
        """M (bernoulli) -> N (poisson) -> Y: both mediators get sampled carry."""
        model = pathmc.model(
            "M ~ X + lag(M)\nN ~ M\nY ~ N + lag(Y)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": "bernoulli", "N": "poisson"},
        )
        _assert_discrete_sample_nodes(model, "bernoulli", var="M")
        assert "discrete_uniforms_M" in model.pymc_model.named_vars
        assert "mu_N" in model.pymc_model.named_vars


class TestTerminalDiscreteOutcomesUnchanged:
    """Cases that must compile cleanly: no probability/rate leaks into a
    downstream equation, or the scan compiler never engages at all."""

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_terminal_scan_outcome_allowed(self, panel_data, family):
        # M has a temporal dep (lag(X)) but is never itself used as a
        # predictor -- not even via self-lag -- so it's a safe terminal
        # scan outcome.
        model = pathmc.model(
            "M ~ lag(X)",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        assert "_use_observed_carry" in model.pymc_model.named_vars
        assert "discrete_uniforms_M" not in model.pymc_model.named_vars

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_no_temporal_deps_skips_scan_entirely(self, panel_data, family):
        """No lag()/adstock() anywhere -> the cross-sectional compiler is
        used, not the scan, so the mean-leak concern doesn't apply and the
        validator must not fire even though M feeds Y directly."""
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=panel_data,
            panel={"unit": "region", "time": "week"},
            families={"M": family},
        )
        assert "_use_observed_carry" not in model.pymc_model.named_vars

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_non_panel_model_skips_scan_entirely(self, panel_data, family):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=panel_data,
            families={"M": family},
        )
        assert model.pymc_model is not None


class TestDownstreamMuUsesSampledValues:
    """Regression: downstream ``mu`` must depend on sampled discrete carry,
    not on response-scale means that are invariant to the uniform draw."""

    def test_bernoulli_mediator_mu_y_differs_with_uniform_draw(self):
        import pytensor

        rows = []
        for unit in range(2):
            for week in range(1, 6):
                rows.append({
                    "region": f"R{unit}",
                    "week": week,
                    "X": 0.0,
                    "M": 0.0,
                    "Y": 0.0,
                })
        df = pd.DataFrame(rows)

        model = pathmc.model(
            "M ~ 0 + X\nY ~ 0 + M + lag(Y)",
            data=df,
            panel={"unit": "region", "time": "week"},
            families={"M": "bernoulli"},
        )
        pmodel = model.pymc_model
        n_times = len(df) // 2
        n_units = 2

        with pmodel:
            pmodel["_use_observed_carry"].set_value(np.array(0, dtype="int8"))
            uniforms = pmodel["discrete_uniforms_M"]
            mu_y = pmodel["mu_Y"]
            fn = pytensor.function([uniforms], mu_y)
            mu_at_t0_all0 = fn(np.zeros((n_times, n_units)))[0]
            mu_at_t0_all1 = fn(np.ones((n_times, n_units)))[0]

        # If ``mu_Y`` consumed sigmoid(mu_M) instead of a Bernoulli draw,
        # the two uniform inputs would yield identical ``mu_Y`` at t=0.
        assert not np.allclose(mu_at_t0_all0, mu_at_t0_all1)


class TestLatentDiscreteFamilyRejected:
    """A latent node with a discrete family is rejected outright: latent
    nodes only support the deterministic mu pass-through or
    ``family='latent_normal'`` (see ``_validate_latent_families``).
    Setting e.g. ``family='bernoulli'`` on a latent node would otherwise
    silently compile as a plain deterministic mu pass-through -- no
    sigmoid/exp link at all -- which is not what the family name implies.
    """

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_latent_discrete_family_raises_non_panel(self, panel_data, family):
        with pytest.raises(ValueError, match="latent nodes only support"):
            pathmc.model(
                "M ~ X\nY ~ M",
                data=panel_data,
                latent=["M"],
                families={"M": family},
            )

    @pytest.mark.parametrize("family", NON_GAUSSIAN_FAMILIES)
    def test_latent_discrete_family_raises_panel_scan(self, panel_data, family):
        with pytest.raises(ValueError, match="latent nodes only support"):
            pathmc.model(
                "M ~ X\nY ~ M + lag(Y)",
                data=panel_data,
                panel={"unit": "region", "time": "week"},
                latent=["M"],
                families={"M": family},
            )

    def test_latent_normal_still_allowed(self, panel_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=panel_data,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        assert model.pymc_model is not None

    def test_latent_gaussian_default_still_allowed(self, panel_data):
        model = pathmc.model(
            "M ~ X\nY ~ M",
            data=panel_data,
            latent=["M"],
        )
        assert model.pymc_model is not None


class TestHighRateClipWarning:
    """``_warn_high_rate_clip_risk`` should warn only when observed counts
    approach the internal ``mu`` clip bound used inside the scan."""

    def _panel_with_counts(self, max_count: float) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        for unit in range(2):
            for week in range(1, 6):
                rows.append({
                    "region": f"R{unit}",
                    "week": week,
                    "X": rng.normal(),
                    "spend": rng.uniform(1, 5),
                    "count": max_count if week == 1 and unit == 0 else 1.0,
                })
        return pd.DataFrame(rows)

    def test_high_rate_warns(self):
        # exp(19) ~= 1.78e8, well within the margin of the 20.0 clip bound.
        data = self._panel_with_counts(max_count=float(np.exp(19)))
        with pytest.warns(UserWarning, match="clip bound"):
            pathmc.model(
                "count ~ lag(spend)",
                data=data,
                panel={"unit": "region", "time": "week"},
                families={"count": "poisson"},
            )

    def test_low_rate_no_warning(self, recwarn):
        data = self._panel_with_counts(max_count=5.0)
        pathmc.model(
            "count ~ lag(spend)",
            data=data,
            panel={"unit": "region", "time": "week"},
            families={"count": "poisson"},
        )
        assert not any("clip bound" in str(w.message) for w in recwarn.list)

    def test_gaussian_family_never_warns(self, recwarn):
        """The clip only applies to poisson/negbinomial rates; a Gaussian
        outcome with the same large values must not trigger the warning."""
        data = self._panel_with_counts(max_count=float(np.exp(19)))
        data = data.rename(columns={"count": "sales"})
        pathmc.model(
            "sales ~ lag(spend)",
            data=data,
            panel={"unit": "region", "time": "week"},
        )
        assert not any("clip bound" in str(w.message) for w in recwarn.list)


class TestGaussianChildFollowsDiscreteMediator:
    """Issue #581: a scan-compiled Gaussian child follows the discrete draw.

    ``Y ~ 0 + M`` with ``sigma_Y = 0`` and a unit coefficient is ``Y = M``
    on every row. A dummy lag forces the scan compiler. The mediator must
    vary across time; one broadcast draw is not the structural equation.
    """

    def test_poisson_identity_is_not_a_broadcast_draw(self):
        df = pd.DataFrame({"unit": ["A"] * 6, "time": range(6)})
        saw_variation = False
        for seed in range(30):
            sim = pathmc.simulate(
                "M ~ 0 + 1.0\nY ~ 0 + M\nW ~ 0 + lag(W)",
                data=df,
                params={
                    "beta_M": [1.0],
                    "beta_Y": [1.0],
                    "beta_W": [0.0],
                    "sigma_Y": 0.0,
                    "sigma_W": 0.0,
                },
                panel={"unit": "unit", "time": "time"},
                families={"M": "poisson"},
                random_seed=seed,
            )
            ordered = sim.sort_values("time")
            m = ordered.M.to_numpy()
            y = ordered.Y.to_numpy()
            assert np.array_equal(y, m)
            saw_variation = saw_variation or len(set(m.tolist())) > 1
        assert saw_variation

    def test_bernoulli_identity_uses_the_structural_draw(self):
        df = pd.DataFrame({"unit": ["A"] * 6, "time": range(6)})
        uniforms = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], dtype=float).reshape(6, 1)
        sim = pathmc.simulate(
            "M ~ 0 + 1.0\nY ~ 0 + M\nW ~ 0 + lag(W)",
            data=df,
            params={
                "beta_M": [0.0],
                "beta_Y": [1.0],
                "beta_W": [0.0],
                "sigma_Y": 0.0,
                "sigma_W": 0.0,
                "discrete_uniforms_M": uniforms,
            },
            panel={"unit": "unit", "time": "time"},
            families={"M": "bernoulli"},
            random_seed=0,
        )
        ordered = sim.sort_values("time")
        m = ordered.M.to_numpy()
        y = ordered.Y.to_numpy()
        assert np.array_equal(m, np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
        assert np.array_equal(y, m)

    def test_scaled_coefficient_and_second_unit(self):
        df = pd.DataFrame({
            "unit": ["A", "A", "B", "B", "A", "B"],
            "time": [0, 1, 0, 1, 2, 2],
        })
        sim = pathmc.simulate(
            "M ~ 0 + 1.0\nY ~ 0 + M\nW ~ 0 + lag(W)",
            data=df,
            params={
                "beta_M": [1.0],
                "beta_Y": [2.0],
                "beta_W": [0.0],
                "sigma_Y": 0.0,
                "sigma_W": 0.0,
            },
            panel={"unit": "unit", "time": "time"},
            families={"M": "poisson"},
            random_seed=1,
        )
        ordered = sim.sort_values(["unit", "time"])
        assert np.array_equal(ordered.Y.to_numpy(), 2.0 * ordered.M.to_numpy())
        assert len(set(ordered.M.to_numpy().tolist())) > 1


class TestObservedNonGaussianParent:
    """Issue #596: a scan child conditions on an observed non-Gaussian parent
    even when nothing lags that parent.

    The estimation model (``_use_observed_carry`` left at its compiled 1)
    must put the observed column into the child's mean. An auxiliary
    Bernoulli uniform, a Poisson draw, and the parent's structural mean
    must not.
    """

    def _panel(self, y: np.ndarray, z: float = 0.0) -> pd.DataFrame:
        n = len(y)
        n_times = n // 2
        return pd.DataFrame({
            "unit": np.repeat(["U0", "U1"], n_times),
            "time": np.tile(np.arange(n_times), 2),
            "X": np.ones(n),
            "Y": y,
            "Z": np.full(n, z),
        })

    def _mu_z(self, model, **values) -> np.ndarray:
        import pymc as pm

        cast = {}
        for name, val in values.items():
            rv = model.pymc_model[name]
            cast[name] = np.broadcast_to(
                np.asarray(val, dtype=float), tuple(rv.shape.eval())
            ).copy()
        graph = pm.do(model.pymc_model, cast)
        return np.asarray(pm.draw(graph["mu_Z"], draws=1, random_seed=0))

    def test_bernoulli_parent_ignores_auxiliary_uniform(self):
        y = np.zeros(6, dtype=int)
        model = pathmc.model(
            "Y ~ X\nZ ~ Y + lag(Z)",
            data=self._panel(y),
            families={"Y": "bernoulli"},
            panel={"unit": "unit", "time": "time"},
        )
        assert "Y" in str(model.equations())
        assert int(model.pymc_model["_use_observed_carry"].get_value()) == 1
        common = {
            "beta_Y": [0.0, 0.0],
            "beta_Z": [0.0, 10.0, 0.0],
            "sigma_Z": 1.0,
        }
        for uniform in (0.05, 0.95):
            mu = self._mu_z(model, discrete_uniforms_Y=uniform, **common)
            np.testing.assert_allclose(mu, 0.0)

    def test_mixed_bernoulli_parent_matches_observed_column(self):
        # Each unit is 0, 1, 0. A uniform of 0 would draw 1 wherever p > 0,
        # so a mean that still follows the auxiliary draw cannot match this
        # column.
        y = np.array([0, 1, 0, 0, 1, 0], dtype=int)
        model = pathmc.model(
            "Y ~ X\nZ ~ Y + lag(Z)",
            data=self._panel(y),
            families={"Y": "bernoulli"},
            panel={"unit": "unit", "time": "time"},
        )
        mu = self._mu_z(
            model,
            discrete_uniforms_Y=0.0,
            beta_Y=[0.0, 0.0],
            beta_Z=[0.0, 4.0, 0.0],
            sigma_Z=1.0,
        )
        expected = y.reshape(2, 3).T * 4.0
        np.testing.assert_allclose(mu.reshape(3, 2), expected)

    def test_poisson_parent_matches_observed_counts(self):
        import pymc as pm
        import pytensor

        y = np.array([0, 2, 1, 0, 2, 1], dtype=int)
        model = pathmc.model(
            "Y ~ X\nZ ~ Y + lag(Z)",
            data=self._panel(y),
            families={"Y": "poisson"},
            panel={"unit": "unit", "time": "time"},
        )
        graph = pm.do(
            model.pymc_model,
            {
                "beta_Y": np.zeros(2),
                "beta_Z": np.array([0.0, 3.0, 0.0]),
                "sigma_Z": np.array(1.0),
            },
        )
        # The unused Poisson draw stays in the switch, and Numba cannot
        # compile that scan RNG. Python mode still evaluates the mean.
        mu = pytensor.function([], graph["mu_Z"], mode="FAST_COMPILE")()
        expected = y.reshape(2, 3).T * 3.0
        np.testing.assert_allclose(mu.reshape(3, 2), expected)

    def test_studentt_parent_is_not_the_structural_mean(self):
        y = np.full(6, 5.0)
        model = pathmc.model(
            "Y ~ X\nZ ~ Y + lag(Z)",
            data=self._panel(y),
            families={"Y": "studentt"},
            panel={"unit": "unit", "time": "time"},
        )
        mu = self._mu_z(
            model,
            beta_Y=[0.0, 0.0],
            beta_Z=[0.0, 2.0, 0.0],
            sigma_Y=1.0,
            sigma_Z=1.0,
            nu_Y=5.0,
        )
        np.testing.assert_allclose(mu, 10.0)
