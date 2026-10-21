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
"""M7 gate tests: residual covariance (~~) compilation.

TestResidualCovCompilation verifies that ~~ produces MvNormal blocks.
TestResidualCovGuards verifies that ~~ is rejected for non-Gaussian outcomes.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

import pathmc

from conftest import PARALLEL_MEDIATORS_SPEC


@pytest.fixture(scope="module")
def fitted_parallel_mediators(mock_pymc_sample_module):
    """Parallel mediators model with ~~ fitted once for this module."""
    rng = np.random.default_rng(42)
    n = 200
    T = rng.normal(size=n)
    eps = rng.multivariate_normal([0, 0], [[0.16, 0.08], [0.08, 0.16]], size=n)
    M1 = 0.6 * T + eps[:, 0]
    M2 = 0.4 * T + eps[:, 1]
    Y = 0.5 * M1 + 0.3 * M2 + 0.2 * T + rng.normal(scale=0.5, size=n)
    data = pd.DataFrame({"T": T, "M1": M1, "M2": M2, "Y": Y})
    model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=data)
    model.fit(draws=100, tune=100, chains=1, random_seed=42)
    return model


class TestResidualCovCompilation:
    def test_residual_cov_compiles(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_correlation_parameter_in_model(self, parallel_mediators_data):
        """The compiled model should contain an LKJ or correlation-related parameter."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        rv_names = {rv.name.lower() for rv in model.pymc_model.free_RVs}
        has_corr_param = any(
            keyword in name
            for name in rv_names
            for keyword in ("chol", "corr", "lkj", "rho")
        )
        assert has_corr_param, (
            f"No correlation parameter found in model. Free RV names: {rv_names}"
        )

    def test_residual_cov_model_has_observed(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        assert len(model.pymc_model.observed_RVs) > 0


class TestResidualCovGuards:
    def test_residual_cov_requires_gaussian(self):
        """~~ between a Bernoulli and a Gaussian outcome should raise."""
        spec = "Y1 ~ X\nY2 ~ X\nY1 ~~ Y2"
        rng = np.random.default_rng(99)
        data = pd.DataFrame({
            "X": rng.normal(size=100),
            "Y1": rng.binomial(1, 0.5, size=100).astype(float),
            "Y2": rng.normal(size=100),
        })
        with pytest.raises(
            Exception, match="(?i)(gaussian|continuous|family|bernoulli|covariance)"
        ):
            pathmc.model(spec, data=data, families={"Y1": "bernoulli"})


class TestBlockVarDeterministics:
    """Block variables should have mu_{var} deterministics and wire through endogenous_rvs."""

    def test_mu_deterministics_exist(self, parallel_mediators_data):
        """Each block variable should have a mu_{var} Deterministic in the model."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        det_names = {d.name for d in model.pymc_model.deterministics}
        assert "mu_M1" in det_names, f"mu_M1 missing from deterministics: {det_names}"
        assert "mu_M2" in det_names, f"mu_M2 missing from deterministics: {det_names}"

    def test_block_vars_not_free_rvs(self, parallel_mediators_data):
        """Block variables should NOT appear as individual free RVs."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        free_rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "M1" not in free_rv_names
        assert "M2" not in free_rv_names
        assert not any(name.endswith("_joint") for name in free_rv_names)

    def test_lkj_prior_in_priors_output(self, parallel_mediators_data):
        """priors() should show the chol_{block} LKJ entry."""
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        prior_str = str(model.priors())
        assert "chol_M1_M2" in prior_str
        assert "LKJCholeskyCov" in prior_str


class TestBlockMemberDualWiring:
    """Downstream equations split estimation vs generative block wiring (#217)."""

    def test_likelihood_and_mu_det_use_different_block_inputs(
        self, parallel_mediators_data
    ):
        """Y's likelihood mu uses observed M1/M2; mu_Y uses structural mu_M1/mu_M2."""
        from pytensor.graph.traversal import ancestors

        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        gen = model._gen_model

        def named_ancestors(var: object) -> set[str]:
            return {
                name
                for v in ancestors([var])
                if (name := getattr(v, "name", None)) is not None
            }

        y_likelihood_mu = gen["Y"].owner.inputs[2]
        mu_y_det = gen["mu_Y"]

        assert y_likelihood_mu is not mu_y_det
        assert "mu_M1" in named_ancestors(mu_y_det)
        assert "mu_M2" in named_ancestors(mu_y_det)
        assert "mu_M1" not in named_ancestors(y_likelihood_mu)
        assert "mu_M2" not in named_ancestors(y_likelihood_mu)

    def test_interaction_with_block_member_uses_observed_in_likelihood(self, rng):
        """Interaction terms on block members also resolve via observed data."""
        from pytensor.graph.traversal import ancestors

        n = 80
        t = rng.normal(size=n)
        eps = rng.multivariate_normal([0, 0], [[0.16, 0.08], [0.08, 0.16]], size=n)
        m1 = 0.6 * t + eps[:, 0]
        m2 = 0.4 * t + eps[:, 1]
        y = 0.3 * m1 * m2 + rng.normal(scale=0.5, size=n)
        data = pd.DataFrame({"T": t, "M1": m1, "M2": m2, "Y": y})
        spec = "M1 ~ T\nM2 ~ T\nY ~ M1:M2\nM1 ~~ M2"

        model = pathmc.model(spec, data=data)
        gen = model._gen_model

        def named_ancestors(var: object) -> set[str]:
            return {
                name
                for v in ancestors([var])
                if (name := getattr(v, "name", None)) is not None
            }

        y_likelihood_mu = gen["Y"].owner.inputs[2]
        mu_y_det = gen["mu_Y"]
        assert y_likelihood_mu is not mu_y_det
        assert "mu_M1" in named_ancestors(mu_y_det)
        assert "mu_M1" not in named_ancestors(y_likelihood_mu)


@pytest.mark.slow
class TestResidualCovSampling:
    def test_residual_cov_model_samples(self, fitted_parallel_mediators):
        """The model with ~~ should sample without errors."""
        summary = fitted_parallel_mediators.summary()
        assert summary is not None


@pytest.mark.slow
class TestBlockVarDoOperator:
    """do() should propagate through block variables correctly."""

    def test_do_through_block_vars_mean(self, fitted_parallel_mediators):
        """do() on an exogenous variable should propagate through block vars."""
        r0 = fitted_parallel_mediators.do(set={"T": 0.0}, kind="mean")
        r1 = fitted_parallel_mediators.do(set={"T": 1.0}, kind="mean")
        ate = r1.mean("Y") - r0.mean("Y")
        assert ate > 0, f"ATE of T on Y should be positive, got {ate}"

    def test_do_on_block_var_mean(self, fitted_parallel_mediators):
        """do() directly on a block variable runs and returns finite contrasts."""
        r0 = fitted_parallel_mediators.do(set={"M1": 0.0}, kind="mean")
        r1 = fitted_parallel_mediators.do(set={"M1": 1.0}, kind="mean")
        ate = r1.mean("Y") - r0.mean("Y")
        assert np.isfinite(ate), f"ATE of M1 on Y should be finite, got {ate}"

    def test_do_on_block_var_predictive(self, fitted_parallel_mediators):
        """do() directly on a block variable should work with kind='predictive'."""
        r0 = fitted_parallel_mediators.do(set={"M1": 0.0}, kind="predictive")
        r1 = fitted_parallel_mediators.do(set={"M1": 1.0}, kind="predictive")
        ate = r1.mean("Y") - r0.mean("Y")
        assert np.isfinite(ate), f"ATE of M1 on Y should be finite, got {ate}"


class TestResidualPredictiveDo:
    """Predictive do() must carry residual-block noise into descendants.

    The estimation graph observes block members, so a wrong implementation
    samples the descendant from observed parent columns. A hard intervention
    then leaves the descendant's predictive distribution unchanged, and an
    unintervened predictive draw omits upstream residual variance.
    """

    SPEC = "Y1 ~ X\nY2 ~ X\nZ ~ Y1\nY1 ~~ Y2"

    def _model(self) -> pathmc.PathModel:
        rng = np.random.default_rng(0)
        n = 80
        x = rng.normal(size=n)
        eps = rng.multivariate_normal([0, 0], [[1.0, 0.3], [0.3, 1.0]], size=n)
        y1 = 0.5 * x + eps[:, 0]
        y2 = -0.2 * x + eps[:, 1]
        z = 2.0 * y1 + rng.normal(scale=0.3, size=n)
        data = pd.DataFrame({"X": x, "Y1": y1, "Y2": y2, "Z": z})
        return pathmc.model(self.SPEC, data=data)

    def _fitted_point_model(self) -> pathmc.PathModel:
        import arviz as az

        model = self._model()
        free = [rv.name for rv in model.pymc_model.free_RVs]
        with model.pymc_model:
            prior = pm.sample_prior_predictive(draws=1, random_seed=1)
        point = prior["prior"].to_dataset().isel(chain=0, draw=0)
        n_draws = 40
        posterior = {}
        for name in free:
            arr = np.asarray(point[name].values)
            posterior[name] = np.broadcast_to(arr, (1, n_draws, *arr.shape)).copy()
        # Slope and intercept both 2: the Y1 contrast isolates the slope.
        posterior["beta_Z"][...] = 2.0
        posterior["sigma_Z"][...] = 0.3
        # Packed lower Cholesky of I_2: each block member has residual sd 1.
        posterior["chol_Y1_Y2"][...] = np.array([1.0, 0.0, 1.0])
        model._idata = az.from_dict({"posterior": posterior})
        return model

    def _point_model(
        self,
        spec: str,
        data: pd.DataFrame,
        overrides: dict[str, float | np.ndarray],
        latent: list[str] | None = None,
        families: dict[str, str] | None = None,
    ) -> pathmc.PathModel:
        import arviz as az

        model = pathmc.model(spec, data=data, latent=latent, families=families)
        free = [rv.name for rv in model.pymc_model.free_RVs]
        with model.pymc_model:
            prior = pm.sample_prior_predictive(draws=1, random_seed=1)
        point = prior["prior"].to_dataset().isel(chain=0, draw=0)
        n_draws = 40
        posterior = {}
        for name in free:
            arr = np.asarray(point[name].values)
            posterior[name] = np.broadcast_to(arr, (1, n_draws, *arr.shape)).copy()
        for name, value in overrides.items():
            posterior[name][...] = value
        model._idata = az.from_dict({"posterior": posterior})
        return model

    def test_sibling_wires_descendant_through_block_member(self):
        from pytensor.graph.traversal import ancestors

        model = self._model()
        sibling = model._predictive_gen_model()
        names = {getattr(v, "name", None) for v in ancestors([sibling["Z"]])}
        assert "Y1" in names
        assert "mu_Y1" not in names
        assert any(name and str(name).endswith("_joint") for name in names)
        free = {rv.name for rv in model.pymc_model.free_RVs}
        observed = {rv.name for rv in model.pymc_model.observed_RVs}
        assert "Y1" not in free
        assert "Z" in observed
        assert model.pymc_model.observed_RVs

    def test_mean_intervention_stays_structural(self):
        model = self._fitted_point_model()
        shift = model.do(set={"Y1": 1.0}, kind="mean").mean("Z") - model.do(
            set={"Y1": 0.0}, kind="mean"
        ).mean("Z")
        assert abs(shift - 2.0) < 1e-6, shift
        draws = model.do(set={"Y1": 1.0}, kind="mean").dataset["Z"].to_numpy()
        assert np.allclose(draws, draws.flat[0])

    def test_predictive_intervention_shifts_descendant_and_drops_block_noise(self):
        model = self._fitted_point_model()
        baseline = model.do(kind="predictive")
        at0 = model.do(set={"Y1": 0.0}, kind="predictive")
        at1 = model.do(set={"Y1": 1.0}, kind="predictive")
        shift = at1.mean("Z") - at0.mean("Z")
        assert abs(shift - 2.0) < 0.25, shift
        base_var = float(np.var(baseline.dataset["Z"].to_numpy()))
        fixed_var = float(np.var(at1.dataset["Z"].to_numpy()))
        assert base_var > 10 * fixed_var, (base_var, fixed_var)

    def test_predictive_intervention_shifts_within_block_descendant(self):
        rng = np.random.default_rng(0)
        n = 40
        data = pd.DataFrame({
            "X": rng.normal(size=n),
            "Y1": rng.normal(size=n),
            "Y2": rng.normal(size=n),
        })
        model = self._point_model(
            "Y1 ~ X\nY2 ~ Y1\nY1 ~~ Y2",
            data,
            {"beta_Y2": 2.0, "chol_Y1_Y2": np.array([1.0, 0.0, 1.0])},
        )
        shift = model.do(set={"Y1": 1.0}, kind="predictive").mean("Y2") - model.do(
            set={"Y1": 0.0}, kind="predictive"
        ).mean("Y2")
        assert abs(shift - 2.0) < 0.35, shift

    def test_within_block_descendant_reads_parent_residual(self):
        from pytensor.graph.traversal import ancestors

        n = 12
        data = pd.DataFrame({c: np.zeros(n) for c in ("X", "Y1", "Y2", "Y3")})
        model = pathmc.model(
            "Y1 ~ X\nY2 ~ Y1\nY3 ~ Y2\nY1 ~~ Y2\nY2 ~~ Y3",
            data=data,
        )
        sibling = model._predictive_gen_model()
        y2_names = {getattr(v, "name", None) for v in ancestors([sibling["Y2"]])}
        y3_names = {getattr(v, "name", None) for v in ancestors([sibling["Y3"]])}
        # A slice of the joint alone does not name the parent member.
        assert "Y1" in y2_names
        assert "Y2" in y3_names
        assert any(name and str(name).endswith("_joint") for name in y2_names)
        assert "Y1" not in model.pymc_model.named_vars
        est_mu = {
            getattr(v, "name", None) for v in ancestors([model.pymc_model["mu_Y2"]])
        }
        assert "mu_Y1" in est_mu
        assert "Y1" not in est_mu

    def test_predictive_within_block_keeps_upstream_residual(self):
        n = 80
        data = pd.DataFrame({c: np.zeros(n) for c in ("X", "Y1", "Y2", "Y3")})
        model = self._point_model(
            "Y1 ~ X\nY2 ~ Y1\nY3 ~ Y2\nY1 ~~ Y2\nY2 ~~ Y3",
            data,
            {
                "beta_Y1": 0.0,
                "beta_Y2": 2.0,
                "beta_Y3": 2.0,
                # Packed lower Cholesky of I_3: each member has residual sd 1.
                "chol_Y1_Y2_Y3": np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            },
        )
        draws = model.do(kind="predictive")
        var_y2 = float(np.var(draws.dataset["Y2"].to_numpy()))
        var_y3 = float(np.var(draws.dataset["Y3"].to_numpy()))
        # Y2 = 2*Y1 + e2 => 5. Wiring through mu only leaves Var(e2) ≈ 1.
        assert var_y2 > 3.0, var_y2
        # Y3 = 2*Y2 + e3 => 21. The same miss leaves ≈ 1.
        assert var_y3 > 10.0, var_y3

        intervened = model.do(set={"Y1": 0.0}, kind="predictive")
        fixed_y2 = float(np.var(intervened.dataset["Y2"].to_numpy()))
        fixed_y3 = float(np.var(intervened.dataset["Y3"].to_numpy()))
        # Fixing Y1 removes its residual. Y2 keeps e2; Y3 still sees e2.
        assert fixed_y2 < 2.0, fixed_y2
        assert fixed_y3 > 3.0, fixed_y3

    def test_predictive_latent_reads_sampled_parent_joint(self):
        n = 40
        data = pd.DataFrame({
            "X": np.zeros(n),
            "Y1": np.zeros(n),
            "Y2": np.zeros(n),
            "Z": np.zeros(n),
        })
        model = self._point_model(
            "Y1 ~ X\nY2 ~ X\nL ~ Y1\nZ ~ L\nY1 ~~ Y2",
            data,
            {
                "beta_Y1": 3.0,
                "beta_L": 2.0,
                "beta_Z": 1.0,
                "sigma_Z": 0.3,
                "chol_Y1_Y2": np.array([1.0, 0.0, 1.0]),
            },
            latent=["L"],
        )
        # X is 0 and both Y1 coefficients are 3, so mu_Y1 is 3.
        # Both L coefficients are 2, so the structural latent is 2 + 2*3.
        structural = model.do(kind="mean").mean("L")
        predictive = model.do(kind="predictive")
        assert abs(structural - 8.0) < 1e-6, structural
        assert abs(predictive.mean("L") - structural) < 0.75, predictive.mean("L")
        assert float(np.var(predictive.dataset["L"].to_numpy())) > 1.0

    def test_reentry_parent_is_structural_not_data_column(self):
        """A block member must not read an outside parent from the data column.

        Compiling the block when the first member is referenced resolves the
        later member's parent before that parent exists, so the resolver falls
        back to the raw column. Predictive variance then collapses to the
        member's own residual.
        """
        from pytensor.graph.traversal import ancestors

        n = 80
        data = pd.DataFrame({c: np.zeros(n) for c in ("X", "Y1", "M", "Y2")})
        spec = "Y1 ~ X\nM ~ Y1\nY2 ~ M\nY1 ~~ Y2"
        model = self._point_model(
            spec,
            data,
            {
                "beta_Y1": 0.0,
                "beta_M": 1.0,
                "beta_Y2": 1.0,
                "sigma_M": 1.0,
                "chol_Y1_Y2": np.array([1.0, 0.0, 1.0]),
            },
        )
        est_mu = {
            getattr(v, "name", None) for v in ancestors([model.pymc_model["mu_Y2"]])
        }
        assert "mu_M" in est_mu
        sibling = model._predictive_gen_model()
        y2_names = {getattr(v, "name", None) for v in ancestors([sibling["Y2"]])}
        assert "M" in y2_names
        assert "Y1" in y2_names
        draws = model.do(kind="predictive")
        var_y2 = float(np.var(draws.dataset["Y2"].to_numpy()))
        # Y2 = M + e2 and M = Y1 + e_M, so variance is about 3, not Var(e2).
        assert var_y2 > 2.0, var_y2

    def test_mean_do_on_outside_parent_reaches_later_member(self):
        """do(M) must shift a later block member that reads M structurally.

        The mean walk used to replace only the free RV. Block members read
        ``mu_M``, so the intervention never arrived and ate() stayed 0.
        """
        n = 40
        data = pd.DataFrame({c: np.zeros(n) for c in ("X", "Y1", "M", "Y2", "Z")})
        model = self._point_model(
            "Y1 ~ X\nM ~ 0 + 2*Y1\nY2 ~ 0 + 4*M\nZ ~ 0 + 1*Y2\nY1 ~~ Y2",
            data,
            {"beta_Y1": 0.0},
        )
        shift = model.do(set={"M": 1.0}, kind="mean").mean("Y2") - model.do(
            set={"M": 0.0}, kind="mean"
        ).mean("Y2")
        assert abs(shift - 4.0) < 1e-6, shift
        assert abs(model.ate("Z", "M", kind="mean").mean() - 4.0) < 1e-6
        product = self._point_model(
            "Y1 ~ X\nM ~ Y1\nY2 ~ 0 + 4*M:X\nY1 ~~ Y2",
            pd.DataFrame({
                "X": np.ones(n),
                "Y1": np.zeros(n),
                "M": np.zeros(n),
                "Y2": np.zeros(n),
            }),
            {"beta_Y1": 0.0, "beta_M": 0.0},
        )
        product_shift = product.do(set={"M": 1.0}, kind="mean").mean("Y2") - product.do(
            set={"M": 0.0}, kind="mean"
        ).mean("Y2")
        assert abs(product_shift - 4.0) < 1e-6, product_shift

    def test_non_gaussian_parent_is_response_scale(self):
        """A Bernoulli parent of a block member is on the probability scale.

        Leaving ``mu_M`` on the logit scale makes an unintervened descendant
        track the linear predictor, and do(M) never reaches that descendant.
        """
        n = 60
        data = pd.DataFrame({
            "X": np.zeros(n),
            "M": np.zeros(n),
            "Y1": np.zeros(n),
            "Y2": np.zeros(n),
        })
        model = self._point_model(
            "M ~ X\nY1 ~ X\nY2 ~ 0 + 4*M\nY1 ~~ Y2",
            data,
            {"beta_M": np.array([1.0, 0.0]), "beta_Y1": np.array([0.0, 0.0])},
            families={"M": "bernoulli"},
        )
        expected = 4.0 / (1.0 + np.exp(-1.0))
        observed = model.do(kind="mean").mean("Y2")
        assert abs(observed - expected) < 1e-6, (observed, expected)
        shift = model.do(set={"M": 1.0}, kind="mean").mean("Y2") - model.do(
            set={"M": 0.0}, kind="mean"
        ).mean("Y2")
        assert abs(shift - 4.0) < 1e-6, shift


def _panel_residual_cov_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    regions = ["North", "South", "East"]
    rows = []
    for region in regions:
        for week in range(1, 26):
            rows.append({
                "region": region,
                "week": week,
                "X": rng.normal(),
                "Y1": rng.normal(),
                "Y2": rng.normal(),
            })
    return pd.DataFrame(rows)


PANEL = {"unit": "region", "time": "week"}


class TestScanPanelResidualCovRejection:
    """Scan-compiled panel models must reject ~~ before fitting independent residuals."""

    def test_model_lag_scan_panel_raises(self):
        df = _panel_residual_cov_data()
        with pytest.raises(NotImplementedError, match="(?i)residual covariances"):
            pathmc.model(
                "Y1 ~ X + lag(Y1)\nY2 ~ X\nY1 ~~ Y2",
                data=df,
                panel=PANEL,
            )

    def test_model_adstock_scan_panel_raises(self):
        df = _panel_residual_cov_data()
        with pytest.raises(NotImplementedError, match="(?i)residual covariances"):
            pathmc.model(
                "Y1 ~ adstock(X, decay=theta)\nY2 ~ X\nY1 ~~ Y2",
                data=df,
                panel=PANEL,
            )

    def test_simulate_params_template_scan_panel_raises(self):
        df = _panel_residual_cov_data()[["region", "week", "X"]]
        with pytest.raises(NotImplementedError, match="(?i)residual covariances"):
            pathmc.simulate_params_template(
                "Y1 ~ X + lag(Y1)\nY2 ~ X\nY1 ~~ Y2",
                data=df,
                panel=PANEL,
            )

    def test_model_panel_without_scan_still_has_chol(self):
        df = _panel_residual_cov_data()
        model = pathmc.model("Y1 ~ X\nY2 ~ X\nY1 ~~ Y2", data=df, panel=PANEL)
        free_rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "chol_Y1_Y2" in free_rv_names
