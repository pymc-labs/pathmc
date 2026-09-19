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
"""Gate tests for AdjustmentModel (#437)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
import pytest
from pymc_extras.prior import Prior

import pathmc
from pathmc.graph import build_graph
from pathmc.identify import is_valid_adjustment_set
from pathmc.idata import posterior as _posterior
from pathmc.parse import parse_spec


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _fork_df(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(scale=0.5, size=n)
    y = 0.5 * x + 0.3 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Z": z, "Y": y})


def _n_posterior_samples(model) -> int:
    ds = _posterior(model._idata)
    return ds.sizes["chain"] * ds.sizes["draw"]


class TestForkDAGConstruction:
    """X ~ Z; Y ~ X + Z → auto set {Z}, formula Y ~ X + Z."""

    def test_auto_adjustment_set_and_formula(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert adjusted.treatment == "X"
        assert adjusted.outcome == "Y"
        assert adjusted.adjustment_set == frozenset({"Z"})
        assert adjusted.formula == "Y ~ X + Z"


class TestEmptyAdjustmentSet:
    """Chain / total-effect DAGs with empty valid set."""

    def test_chain_formula_is_outcome_only_treatment(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=100),
            "M": rng.normal(size=100),
            "Y": rng.normal(size=100),
        })
        model = pathmc.model("M ~ X\nY ~ M", data=df)
        adjusted = model.adjustment_model(treatment="X", outcome="Y")
        assert adjusted.adjustment_set == frozenset()
        assert adjusted.formula == "Y ~ X"

    def test_mediation_total_effect_empty_set(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=100),
            "M": rng.normal(size=100),
            "Y": rng.normal(size=100),
        })
        model = pathmc.model("M ~ X\nY ~ M + X", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert adjusted.adjustment_set == frozenset()
        assert adjusted.formula == "Y ~ X"


class TestMultipleMinimalSets:
    MULTI_SPEC = "M1 ~ Z1 + X\nM2 ~ Z2\nY ~ M1 + M2\nX ~ Z2"

    def test_raises_and_lists_sets(self, rng):
        df = pd.DataFrame({
            v: rng.normal(size=80) for v in ["X", "Y", "Z1", "Z2", "M1", "M2"]
        })
        model = pathmc.model(self.MULTI_SPEC, data=df)
        with pytest.raises(ValueError, match="Several minimal"):
            model.adjustment_model("X -> Y")

    def test_succeeds_with_explicit_set(self, rng):
        df = pd.DataFrame({
            v: rng.normal(size=80) for v in ["X", "Y", "Z1", "Z2", "M1", "M2"]
        })
        model = pathmc.model(self.MULTI_SPEC, data=df)
        adjusted = model.adjustment_model("X -> Y", adjustment_set={"Z2"})
        assert adjusted.adjustment_set == frozenset({"Z2"})
        assert "Z2" in adjusted.formula


class TestUnidentifiedQuery:
    def test_raises_without_compiling_inner_model(self, rng):
        spec = "T ~ Z\nY ~ T\nT ~~ Y"
        df = pd.DataFrame({
            "T": rng.normal(size=80),
            "Y": rng.normal(size=80),
            "Z": rng.normal(size=80),
        })
        model = pathmc.model(spec, data=df)
        with pytest.raises(ValueError, match="not identifiable"):
            model.adjustment_model("T -> Y")


class TestMultiHopQuery:
    def test_raises_mentioning_effect(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=50),
            "M": rng.normal(size=50),
            "Y": rng.normal(size=50),
        })
        model = pathmc.model("M ~ X\nY ~ M", data=df)
        with pytest.raises(ValueError, match="effect\\(\\)"):
            model.adjustment_model("X -> M -> Y")


class TestQueryForms:
    def test_query_and_kwargs_agree(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y", treatment="X", outcome="Y")
        assert adjusted.treatment == "X"

    def test_mismatched_query_and_kwargs_error(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        with pytest.raises(ValueError, match="specifies"):
            model.adjustment_model("X -> Y", treatment="Z", outcome="Y")


class TestDataBinding:
    def test_data_free_parent_requires_data(self):
        dag = pathmc.model("X ~ Z\nY ~ X + Z")
        with pytest.raises(ValueError, match="data="):
            dag.adjustment_model("X -> Y")

    def test_data_free_with_data_arg(self, rng):
        df = _fork_df(rng)
        dag = pathmc.model("X ~ Z\nY ~ X + Z")
        adjusted = dag.adjustment_model("X -> Y", data=df)
        assert adjusted.formula == "Y ~ X + Z"

    def test_data_bound_parent_works_without_data(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert adjusted.outcome_model._data is not None


class TestInvalidAdjustmentAndFormulaVars:
    def test_descendant_in_adjustment_set(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=80),
            "M": rng.normal(size=80),
            "Y": rng.normal(size=80),
        })
        model = pathmc.model("M ~ X\nY ~ M", data=df)
        with pytest.raises(ValueError, match="descendant"):
            model.adjustment_model("X -> Y", adjustment_set={"M"})

    def test_collider_in_adjustment_set(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=80),
            "Y": rng.normal(size=80),
            "C": rng.normal(size=80),
        })
        model = pathmc.model("C ~ X + Y", data=df)
        with pytest.raises(ValueError, match="collider|descendant"):
            model.adjustment_model("X -> Y", adjustment_set={"C"})

    def test_descendant_in_formula_extra_term(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=80),
            "Z": rng.normal(size=80),
            "M": rng.normal(size=80),
            "Y": rng.normal(size=80),
        })
        model = pathmc.model("Z ~ X\nM ~ X + Z\nY ~ M + Z", data=df)
        with pytest.raises(ValueError, match="descendant"):
            model.adjustment_model(
                "X -> Y",
                formula="Y ~ X + Z + M",
            )


class TestTreatmentAlwaysInFormula:
    def test_no_direct_edge_still_includes_treatment(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=100),
            "Z": rng.normal(size=100),
            "M": rng.normal(size=100),
            "Y": rng.normal(size=100),
        })
        model = pathmc.model("Z ~ X\nM ~ X + Z\nY ~ M + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert "X" in adjusted.formula
        assert adjusted.formula.startswith("Y ~ X")


class TestPriorInheritance:
    def test_sigma_copied_not_beta(self, rng):
        df = _fork_df(rng)
        model = pathmc.model(
            "X ~ Z\nY ~ X + Z",
            data=df,
            priors={"sigma_Y": Prior("HalfNormal", sigma=2.0)},
        )
        adjusted = model.adjustment_model("X -> Y")
        assert "sigma_Y" in adjusted.outcome_model._priors
        inherited = adjusted.outcome_model._priors["sigma_Y"]
        assert inherited.to_dict()["dist"] == "HalfNormal"
        assert inherited.to_dict()["kwargs"]["sigma"] == 2.0

    def test_beta_override_on_parent_raises(self, rng):
        df = _fork_df(rng)
        model = pathmc.model(
            "X ~ Z\nY ~ X + Z",
            data=df,
            priors={"beta_Y": Prior("Normal", mu=0, sigma=1)},
        )
        with pytest.raises(ValueError, match="beta_Y"):
            model.adjustment_model("X -> Y")

    def test_user_priors_on_adjustment_model(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model(
            "X -> Y",
            priors={"sigma_Y": Prior("HalfNormal", sigma=3.0)},
        )
        assert (
            adjusted.outcome_model._priors["sigma_Y"].to_dict()["kwargs"]["sigma"]
            == 3.0
        )


class TestInnerModelTypes:
    def test_inner_is_pathmodel_with_pymc_model(self, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert isinstance(adjusted.outcome_model, pathmc.PathModel)
        assert isinstance(adjusted.pymc_model, pm.Model)

    def test_no_bambi_import(self):
        import sys

        mods_before = set(sys.modules)
        import pathmc.adjustment  # noqa: F401

        new_mods = set(sys.modules) - mods_before
        assert "bambi" not in new_mods


class TestGraphVsReducedSpec:
    def test_graph_info_has_mediator_not_in_formula(self, rng):
        df = pd.DataFrame({
            "X": rng.normal(size=100),
            "M": rng.normal(size=100),
            "Y": rng.normal(size=100),
        })
        model = pathmc.model("M ~ X\nY ~ M", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert "M" in adjusted.graph_info.contemporaneous_dag.nodes
        inner_lhs = {reg.lhs for reg in adjusted.outcome_model._spec.regressions}
        assert inner_lhs == {"Y"}
        inner_predictors = {
            t.variable
            for reg in adjusted.outcome_model._spec.regressions
            for t in reg.terms
        }
        assert "M" not in inner_predictors


class TestFitAndEstimands:
    def test_ate_draws_shape(self, mock_pymc_sample, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        adjusted.fit(draws=100, chains=2, random_seed=42)
        ate = adjusted.ate(values=(0.0, 1.0))
        assert "chain" in ate.dataset["Y"].dims
        assert "draw" in ate.dataset["Y"].dims
        assert len(ate.draws()) == 2 * 100

    def test_ate_without_outcome_treatment(self, mock_pymc_sample, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        adjusted.fit(draws=50, chains=2, random_seed=42)
        ate_default = adjusted.ate()
        ate_values = adjusted.ate(values=(0, 1))
        assert ate_default.outcome == "Y"
        assert ate_values.treatment == "X"

    def test_wrong_treatment_raises(self, mock_pymc_sample, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        adjusted.fit(draws=50, chains=1, random_seed=42)
        with pytest.raises(ValueError, match="This AdjustmentModel"):
            adjusted.ate("Y", "Z")

    def test_estimator_metadata(self, mock_pymc_sample, rng):
        df = _fork_df(rng)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        adjusted.fit(draws=50, chains=2, random_seed=42)
        ate = adjusted.ate()
        assert ate.estimator == "regression_adjustment"
        assert ate.causal is True
        assert ate.interventional is True
        assert ate.identifiable is True
        assert ate.dataset.attrs["estimator"] == "regression_adjustment"


class TestBernoulliATE:
    def test_ate_not_equal_treatment_coefficient(self, mock_pymc_sample, rng):
        n = 300
        x = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(0.2 + 1.5 * x)))
        y = rng.binomial(1, p, size=n).astype(float)
        df = pd.DataFrame({"X": x, "Y": y})
        model = pathmc.model("Y ~ X", data=df, families={"Y": "bernoulli"})
        adjusted = model.adjustment_model("X -> Y")
        adjusted.fit(draws=100, chains=2, random_seed=42)
        ate = adjusted.ate(values=(0.0, 1.0))
        beta_post = _posterior(adjusted.outcome_model._idata)["beta_Y"]
        beta_x = float(beta_post.sel(Y_predictors="X").mean())
        ate_mean = ate.mean()
        assert ate.estimator == "regression_adjustment"
        # Coefficient is on logit scale; ATE is on probability scale.
        assert not np.isclose(ate_mean, beta_x, rtol=0.05, atol=0.05)


class TestPolarsInput:
    def test_polars_data_constructs(self, rng):
        pdf = _fork_df(rng)
        df = pl.from_pandas(pdf)
        model = pathmc.model("X ~ Z\nY ~ X + Z", data=df)
        adjusted = model.adjustment_model("X -> Y")
        assert adjusted.formula == "Y ~ X + Z"


class TestAttAtuSmoke:
    def test_att_smoke(self, mock_pymc_sample, rng):
        n = 200
        x = rng.normal(size=n)
        t = (rng.uniform(size=n) < 0.5).astype(float)
        y = 0.5 * t + 0.3 * x + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"T": t, "X": x, "Y": y})
        model = pathmc.model("Y ~ T + X", data=df)
        adjusted = model.adjustment_model("T -> Y")
        adjusted.fit(draws=50, chains=2, random_seed=42)
        att = adjusted.att(values=(0, 1))
        assert att.estimator == "regression_adjustment"
        assert att._estimand == "ATT"

    def test_atu_smoke(self, mock_pymc_sample, rng):
        n = 200
        x = rng.normal(size=n)
        t = (rng.uniform(size=n) < 0.5).astype(float)
        y = 0.5 * t + 0.3 * x + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"T": t, "X": x, "Y": y})
        model = pathmc.model("Y ~ T + X", data=df)
        adjusted = model.adjustment_model("T -> Y")
        adjusted.fit(draws=50, chains=2, random_seed=42)
        atu = adjusted.atu(values=(0, 1))
        assert atu.estimator == "regression_adjustment"


class TestNonMinimalSuperset:
    def test_valid_superset_accepted(self, rng):
        spec = "Z ~ W\nX ~ Z\nY ~ X + Z"
        df = pd.DataFrame({
            "W": rng.normal(size=100),
            "X": rng.normal(size=100),
            "Z": rng.normal(size=100),
            "Y": rng.normal(size=100),
        })
        model = pathmc.model(spec, data=df)
        adjusted = model.adjustment_model("X -> Y", adjustment_set={"Z", "W"})
        assert adjusted.adjustment_set == frozenset({"Z", "W"})
        assert "W" in adjusted.formula
        assert "Z" in adjusted.formula


class TestIsValidAdjustmentSet:
    def test_valid_minimal_and_superset(self):
        fork = build_graph(parse_spec("X ~ Z\nY ~ X + Z"))
        assert is_valid_adjustment_set(fork, "X", "Y", {"Z"})
        chain = build_graph(parse_spec("M ~ X\nY ~ M"))
        assert is_valid_adjustment_set(chain, "X", "Y", set())

    def test_descendant_raises(self):
        g = build_graph(parse_spec("M ~ X\nY ~ M"))
        with pytest.raises(ValueError, match="descendant"):
            is_valid_adjustment_set(g, "X", "Y", {"M"})


class TestResidualBlockValidation:
    """``is_valid_adjustment_set`` must honor ``~~`` blocks like ``adjustment_sets``."""

    IV_SPEC = "T ~ Z\nY ~ T\nT ~~ Y"

    def test_explicit_set_rejected_when_unidentifiable(self, rng):
        df = pd.DataFrame({
            "T": rng.normal(size=80),
            "Y": rng.normal(size=80),
            "Z": rng.normal(size=80),
        })
        model = pathmc.model(self.IV_SPEC, data=df)
        with pytest.raises(ValueError, match="not identifiable"):
            model.adjustment_model("T -> Y")

    def test_is_valid_adjustment_set_rejects_with_residual_block(self):
        g = build_graph(parse_spec(self.IV_SPEC))
        with pytest.raises(ValueError, match="does not block all backdoor"):
            is_valid_adjustment_set(g, "T", "Y", {"Z"})
