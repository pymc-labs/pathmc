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
"""AR(1) latent dynamics with sparse measurements in scan-compiled panel models.

Covers GitHub issue #88 Phase 1:

- estimated latent initial conditions (``init_{var}`` free parameters used
  as the scan carry initial value instead of zeros),
- ``PathModel.latent_trajectory(var)`` posterior extraction with time/unit
  coordinates,
- an end-to-end simulate-and-recover smoke test.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.priors import Prior


def _ar1_survey_panel(
    phi: float = 0.8,
    init_mean: float = 0.5,
    process_sd: float = 0.15,
    meas_sd: float = 0.05,
    n_units: int = 3,
    n_times: int = 30,
    obs_times: tuple[int, ...] = (2, 9, 16, 22),
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate an AR(1) latent state sparsely anchored by survey rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        a = init_mean + rng.normal(scale=0.2)
        for t in range(n_times):
            if t > 0:
                a = phi * a + rng.normal(scale=process_sd)
            survey = np.nan
            if t in obs_times:
                survey = a + rng.normal(scale=meas_sd)
            rows.append({"unit": f"u{u}", "time": t, "survey": survey})
    return pd.DataFrame(rows)


SPEC = "survey ~ 0 + 1*awareness\nawareness ~ lag(awareness)"
PANEL = {"unit": "unit", "time": "time"}


def _build_model(df: pd.DataFrame, **kwargs) -> pathmc.PathModel:
    base = dict(
        data=df,
        panel=PANEL,
        latent=["awareness"],
        families={"awareness": "latent_normal"},
    )
    base.update(kwargs)
    return pathmc.model(SPEC, **base)


# ---------------------------------------------------------------------------
# Compilation — estimated initial conditions
# ---------------------------------------------------------------------------


def test_init_prior_absent_without_panel():
    from pathmc.parse import parse_spec
    from pathmc.priors import default_priors

    priors = default_priors(
        parse_spec("a ~ lag(a) + x\ny ~ a"),
        latent={"a"},
        families={"a": "latent_normal"},
    )
    assert "init_a" not in priors


class TestLatentInitCompilation:
    """Latent variables get an ``init_{var}`` free parameter as scan seed."""

    def test_init_rv_uses_unit_coord(self):
        m = _build_model(_ar1_survey_panel())
        assert m.pymc_model.named_vars_to_dims.get("init_awareness") == ("unit",)

    def test_stochastic_latent_has_init_rv(self):
        m = _build_model(_ar1_survey_panel())
        init_rv = m.pymc_model["init_awareness"]
        assert tuple(init_rv.shape.eval()) == (3,)  # one per unit

    def test_deterministic_latent_has_init_rv(self):
        df = _ar1_survey_panel()
        m = pathmc.model(
            SPEC,
            data=df,
            panel=PANEL,
            latent=["awareness"],
        )
        assert "init_awareness" in {rv.name for rv in m.pymc_model.free_RVs}

    def test_custom_prior_override(self):
        m = _build_model(
            _ar1_survey_panel(),
            priors={"init_awareness": Prior("Normal", mu=0.5, sigma=0.2)},
        )
        # Compiles and the RV exists; the custom location is reflected in
        # the compiled graph's point function.
        assert "init_awareness" in {rv.name for rv in m.pymc_model.free_RVs}
        point = m.pymc_model.initial_point()
        assert "init_awareness" in point

    def test_non_latent_lag_model_has_no_init(self):
        rng = np.random.default_rng(0)
        rows = [
            {"unit": f"u{u}", "time": t, "Y": rng.normal()}
            for u in range(2)
            for t in range(10)
        ]
        m = pathmc.model(
            "Y ~ lag(Y)",
            data=pd.DataFrame(rows),
            panel=PANEL,
        )
        free_names = {rv.name for rv in m.pymc_model.free_RVs}
        assert not any(name.startswith("init_") for name in free_names)

    def test_unlagged_latent_has_no_init(self):
        """A latent that never feeds a lag() term gets no init parameter."""
        rng = np.random.default_rng(2)
        rows = [
            {"unit": f"u{u}", "time": t, "tv": rng.normal(), "survey": rng.normal()}
            for u in range(2)
            for t in range(12)
        ]
        df = pd.DataFrame(rows)
        # adstock() forces the scan engine without giving the latent its own
        # lag term, so the recursion never reads a t=0 carry state.
        m = pathmc.model(
            "survey ~ 0 + 1*awareness\nawareness ~ adstock(tv, decay=0.5)",
            data=df,
            panel=PANEL,
            latent=["awareness"],
            families={"awareness": "latent_normal"},
        )
        free_names = {rv.name for rv in m.pymc_model.free_RVs}
        assert not any(name.startswith("init_") for name in free_names)


# ---------------------------------------------------------------------------
# simulate() / simulate_params_template() integration
# ---------------------------------------------------------------------------


class TestInitInSimulate:
    """``init_{var}`` is a required simulation parameter like any other."""

    def test_template_includes_init(self):
        template = pathmc.simulate_params_template(
            SPEC,
            data=_ar1_survey_panel(),
            panel=PANEL,
            latent=["awareness"],
            families={"awareness": "latent_normal"},
        )
        assert "init_awareness" in template
        assert template["init_awareness"]["shape"] == (3,)

    def test_simulate_accepts_init_param(self):
        out = pathmc.simulate(
            SPEC,
            data=_ar1_survey_panel(),
            params={
                "beta_awareness": [0.0, 0.8],
                "sigma_awareness": 0.15,
                "init_awareness": [0.5, 0.4, 0.6],
                "sigma_survey": 0.05,
                "survey_unobserved": 0.0,
            },
            panel=PANEL,
            latent=["awareness"],
            families={"awareness": "latent_normal"},
            random_seed=3,
        )
        assert list(out.columns) == ["unit", "time", "survey", "awareness"]
        awareness = out["awareness"].to_numpy().reshape(3, 30)
        # Non-zero start: the estimated initial condition seeds t=0.
        assert not np.allclose(awareness[:, 0], 0.0)

    def test_simulate_missing_init_raises(self):
        with pytest.raises(ValueError, match="init_awareness"):
            pathmc.simulate(
                SPEC,
                data=_ar1_survey_panel(),
                params={
                    "beta_awareness": [0.8],
                    "sigma_awareness": 0.15,
                    "sigma_survey": 0.05,
                    "survey_unobserved": 0.0,
                },
                panel=PANEL,
                latent=["awareness"],
                families={"awareness": "latent_normal"},
                random_seed=3,
            )


# ---------------------------------------------------------------------------
# latent_trajectory accessor
# ---------------------------------------------------------------------------


class TestLatentTrajectoryAccessor:
    """latent_trajectory returns posterior states with panel coordinates."""

    def _fitted(self, **kwargs):
        m = _build_model(_ar1_survey_panel(), **kwargs)
        m.fit(random_seed=42, progressbar=False)
        return m

    def test_requires_fit(self):
        m = _build_model(_ar1_survey_panel())
        with pytest.raises(RuntimeError, match="fit"):
            m.latent_trajectory("awareness")

    def test_requires_latent_variable(self):
        m = _build_model(_ar1_survey_panel())
        m.fit(random_seed=42, progressbar=False)
        with pytest.raises(ValueError, match="not a latent"):
            m.latent_trajectory("survey")

    def test_requires_scan_compiled_model(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"X": rng.normal(size=60), "M_obs": rng.normal(size=60)})
        df.loc[::2, "M_obs"] = np.nan
        m = pathmc.model(
            "M ~ X\nM_obs ~ 0 + 1*M",
            data=df,
            latent=["M"],
            families={"M": "latent_normal"},
        )
        m.fit(random_seed=42, progressbar=False)
        with pytest.raises(ValueError, match="scan-compiled"):
            m.latent_trajectory("M")

    def test_stochastic_latent_dims_and_coords(self):
        m = self._fitted()
        traj = m.latent_trajectory("awareness")
        assert traj.dims == ("chain", "draw", "time", "unit")
        assert traj.sizes["chain"] == 1
        assert traj.sizes["time"] == 30
        assert traj.sizes["unit"] == 3
        np.testing.assert_array_equal(traj.coords["time"].values, np.arange(30))
        np.testing.assert_array_equal(traj.coords["unit"].values, ["u0", "u1", "u2"])

    def test_stochastic_latent_named_after_var(self):
        m = self._fitted()
        assert m.latent_trajectory("awareness").name == "awareness"

    def test_deterministic_latent_uses_mu_var(self):
        df = _ar1_survey_panel()
        m = pathmc.model(SPEC, data=df, panel=PANEL, latent=["awareness"])
        m.fit(random_seed=42, progressbar=False)
        traj = m.latent_trajectory("awareness")
        assert traj.name == "mu_awareness"
        assert traj.dims == ("chain", "draw", "time", "unit")
        assert traj.sizes["time"] == 30


# ---------------------------------------------------------------------------
# Recovery — AR(1) latent dynamics from sparse surveys (slow: MCMC)
# ---------------------------------------------------------------------------


class TestAR1Recovery:
    """End-to-end: sparse surveys recover the latent AR(1) trajectory."""

    @pytest.fixture(scope="class")
    def recovered(self):
        df = _ar1_survey_panel(phi=0.8, init_mean=0.5, seed=7)
        m = _build_model(df)
        m.fit(random_seed=11, progressbar=False)
        return df, m

    @pytest.mark.slow
    def test_trajectory_tracks_true_state(self, recovered):
        df, m = recovered
        true = (
            df
            .assign(row=np.arange(len(df)))
            .pivot(index="time", columns="unit", values="survey")
            .to_numpy()
        )  # NaN where unobserved
        traj = m.latent_trajectory("awareness")
        est = traj.mean(dim=("chain", "draw")).values  # (n_times, n_units)
        observed = ~np.isnan(true)
        r = np.corrcoef(est[observed], true[observed])[0, 1]
        assert r > 0.7, f"latent trajectory poorly recovered (r={r:.3f})"

    @pytest.mark.slow
    def test_init_rv_in_posterior(self, recovered):
        _, m = recovered
        post = m._idata.posterior
        assert "init_awareness" in post
        # One initial condition per unit.
        assert post["init_awareness"].shape[-1] == 3
