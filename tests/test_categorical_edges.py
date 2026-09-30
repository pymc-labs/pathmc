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
"""Edge cases of categorical predictors reported in #532.

Covers rank-deficient multi-categorical designs, hierarchical terms with an
intercept, dummy-encoded categoricals in the CI engine behind
``test_implications()`` and ``falsify()``, categorical treatments in
``refute_placebo()``, categorical evidence in ``counterfactual()``, and
mixed-type object columns.
"""

from __future__ import annotations

import warnings

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import pathmc
from pathmc._ci import indicator_block, partial_correlation_ci
from pathmc.falsify import _PartialCorrelationTester

LEVEL_EFFECT = {"north": 0.0, "south": 1.0, "west": -1.0}


@pytest.fixture
def two_categorical_data() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 120
    region = rng.choice(list(LEVEL_EFFECT), size=n)
    channel = rng.choice(["a", "b"], size=n)
    x = rng.normal(size=n)
    y = (
        0.5 * x
        + np.array([LEVEL_EFFECT[r] for r in region])
        + np.where(channel == "b", 0.7, 0.0)
        + rng.normal(scale=0.3, size=n)
    )
    return pd.DataFrame({"y": y, "region": region, "ch": channel, "x": x})


@pytest.fixture
def mediation_data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 300
    region = rng.choice(list(LEVEL_EFFECT), size=n)
    effect = np.array([LEVEL_EFFECT[r] for r in region])
    x = rng.normal(size=n)
    m = 2.0 * x + effect + rng.normal(scale=0.3, size=n)
    y = 1.5 * m + 0.5 * effect + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "region": region, "m": m, "y": y})


def _design_rank(model: pathmc.PathModel, var: str) -> tuple[int, int]:
    design = np.asarray(model.design(var).to_numpy(), dtype=float)
    return int(np.linalg.matrix_rank(design)), design.shape[1]


# ---------------------------------------------------------------------------
# Item 1: only the first categorical in a no-intercept equation is cell means
# ---------------------------------------------------------------------------


class TestNoInterceptMultipleCategoricals:
    def test_second_categorical_is_treatment_coded(self, two_categorical_data):
        model = pathmc.model("y ~ 0 + region + ch", data=two_categorical_data)
        assert list(model.design("y").columns) == [
            "region[north]",
            "region[south]",
            "region[west]",
            "ch[T.b]",
        ]
        assert _design_rank(model, "y") == (4, 4)

    def test_order_decides_the_cell_means_term(self, two_categorical_data):
        model = pathmc.model("y ~ 0 + ch + region + x", data=two_categorical_data)
        assert list(model.design("y").columns) == [
            "ch[a]",
            "ch[b]",
            "region[T.south]",
            "region[T.west]",
            "x",
        ]
        assert _design_rank(model, "y") == (5, 5)

    def test_intercept_equation_is_full_rank(self, two_categorical_data):
        model = pathmc.model("y ~ region + ch", data=two_categorical_data)
        assert _design_rank(model, "y") == (4, 4)

    def test_equations_omit_reference_for_cell_means(self, two_categorical_data):
        model = pathmc.model("y ~ 0 + region + ch", data=two_categorical_data)
        text = str(model.equations())
        assert "C(region, levels=['north', 'south', 'west'])" in text
        assert "C(ch, reference='a', levels=['a', 'b'])" in text

    def test_explicit_reference_on_cell_means_term_warns(self, two_categorical_data):
        with pytest.warns(UserWarning, match="reference='south'.*no effect.*Remove"):
            pathmc.model(
                "y ~ 0 + C(region, reference='south')", data=two_categorical_data
            )

    def test_data_free_render_has_no_empty_state(self):
        assert "C(region)" in str(pathmc.model("y ~ C(region)").equations())
        text = str(pathmc.model("y ~ C(region, reference='south')").equations())
        assert "C(region, reference='south')" in text
        assert "levels=[]" not in text


# ---------------------------------------------------------------------------
# Item 7: hierarchical terms keep every level and never depend on the reference
# ---------------------------------------------------------------------------


class TestHierarchicalCategorical:
    def test_leading_hierarchical_term_absorbs_intercept(self, two_categorical_data):
        model = pathmc.model(
            "y ~ x + C(region, prior='hierarchical')", data=two_categorical_data
        )
        assert list(model.design("y").columns) == [
            "x",
            "region[north]",
            "region[south]",
            "region[west]",
        ]
        assert list(model.pymc_model.coords["y_predictors"]) == ["x"]
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert {"mu_beta_y_region", "sigma_beta_y_region", "beta_y_region"} <= rv_names
        text = str(model.equations())
        assert "mu_y = x + C(region, prior='hierarchical'" in text
        assert "reference=" not in text

    @pytest.mark.parametrize(
        "spec",
        [
            "y ~ x + C(region, prior='hierarchical') + ch",
            "y ~ x + ch + C(region, prior='hierarchical')",
        ],
    )
    def test_treatment_coded_term_does_not_block_absorption(
        self, two_categorical_data, spec
    ):
        model = pathmc.model(spec, data=two_categorical_data)
        design = model.design("y")
        assert "Intercept" not in design.columns
        assert set(design.columns) == {
            "x",
            "ch[T.b]",
            "region[north]",
            "region[south]",
            "region[west]",
        }
        assert np.linalg.matrix_rank(design.to_numpy(dtype=float)) == 5
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "mu_beta_y_region" in rv_names

    def test_later_hierarchical_term_is_zero_mean_deviation(self, two_categorical_data):
        model = pathmc.model(
            "y ~ 0 + x + C(region) + C(ch, prior='hierarchical')",
            data=two_categorical_data,
        )
        assert list(model.design("y").columns) == [
            "x",
            "region[north]",
            "region[south]",
            "region[west]",
            "ch[a]",
            "ch[b]",
        ]
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert "sigma_beta_y_ch" in rv_names
        assert "mu_beta_y_ch" not in rv_names
        assert "beta_y_ch: Normal(0, sigma_beta_y_ch)" in str(model.priors())

    def test_data_free_introspection_matches_fitted_model(self, two_categorical_data):
        spec = "y ~ x + C(region, prior='hierarchical')"
        free = pathmc.model(spec)
        fitted = pathmc.model(spec, data=two_categorical_data)
        free_priors = str(free.priors())
        assert "mu_beta_y_region: Normal(mu=0, sigma=10)" in free_priors
        assert "beta_y_region: Normal(mu_beta_y_region, sigma_beta_y_region)" in (
            free_priors
        )
        assert "\nmu_y = x + C(region" in str(free.equations())
        assert "\nmu_y = x + C(region" in str(fitted.equations())
        free.set_priors({"mu_beta_y_region": pathmc.Prior("Normal", mu=1, sigma=2)})
        assert "mu_beta_y_region: Normal(mu=1, sigma=2)" in str(free.priors())

    def test_data_free_second_hierarchical_term_has_no_population_mean(self):
        model = pathmc.model(
            "y ~ C(region, prior='hierarchical') + C(ch, prior='hierarchical')"
        )
        text = str(model.priors())
        assert "mu_beta_y_region" in text
        assert "mu_beta_y_ch" not in text
        assert "beta_y_ch: Normal(0, sigma_beta_y_ch)" in text

    def test_reference_does_not_change_the_hierarchical_model(
        self, two_categorical_data
    ):
        logps = []
        point = None
        for reference in ("north", "west"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = pathmc.model(
                    f"y ~ x + C(region, reference='{reference}', prior='hierarchical')",
                    data=two_categorical_data,
                )
            pm_model = model.pymc_model
            if point is None:
                point = pm_model.initial_point(random_seed=0)
            logps.append(float(pm_model.compile_logp()(point)))
        assert logps[0] == pytest.approx(logps[1], rel=0, abs=1e-12)

    def test_simulate_round_trips_through_hierarchical_design(self):
        exog = pd.DataFrame({
            "region": ["north", "south", "west"],
            "x": [0.0, 1.0, 2.0],
        })
        simulated = pathmc.simulate(
            "y ~ x + C(region, prior='hierarchical')",
            data=exog,
            params={
                "beta_y": [0.5],
                "beta_y_region": [1.0, 2.0, 3.0],
                "mu_beta_y_region": 2.0,
                "sigma_beta_y_region": 1.0,
                "sigma_y": 1e-6,
            },
            random_seed=0,
        )
        np.testing.assert_allclose(simulated["y"], [1.0, 2.5, 4.0], atol=1e-4)


# ---------------------------------------------------------------------------
# Item 6: object columns mixing numbers and labels are rejected up front
# ---------------------------------------------------------------------------


class TestMixedObjectColumn:
    @pytest.fixture
    def mixed(self) -> pd.DataFrame:
        n = 12
        return pd.DataFrame({
            "y": np.arange(n, dtype=float),
            "mix": pd.Series([1 if i % 2 else "a" for i in range(n)], dtype=object),
        })

    def test_plain_term_raises_actionable_error(self, mixed):
        with pytest.raises(
            ValueError,
            match="Predictor 'mix'.*mixing numbers and non-numeric.*C\\(mix\\)",
        ):
            pathmc.model("y ~ mix", data=mixed)

    def test_explicit_c_treats_every_value_as_a_level(self, mixed):
        model = pathmc.model("y ~ C(mix)", data=mixed)
        assert list(model.design("y").columns) == ["Intercept", "mix[T.a]"]


# ---------------------------------------------------------------------------
# Items 2 and 3: categoricals enter the CI engine as indicator blocks
# ---------------------------------------------------------------------------


class TestBlockCIEngine:
    def test_indicator_block_drops_first_level_and_propagates_missing(self):
        block = indicator_block(np.array(["b", "a", None, "c"], dtype=object))
        expected = np.array([[1.0, 0.0], [0.0, 0.0], [np.nan, np.nan], [0.0, 1.0]])
        np.testing.assert_array_equal(block, expected)

    def test_single_column_blocks_reproduce_the_t_test(self):
        rng = np.random.default_rng(1)
        n = 150
        z = rng.normal(size=(n, 2))
        x = rng.normal(size=n)
        y = 0.3 * x + z[:, 0] + rng.normal(size=n)
        scalar = partial_correlation_ci(x, y, z)
        block = partial_correlation_ci(x[:, None], y[:, None], z)
        assert block.p == pytest.approx(scalar.p, rel=1e-12)
        assert block.df == scalar.df

    def test_categorical_block_matches_partial_f_test(self):
        rng = np.random.default_rng(2)
        n = 200
        z = rng.normal(size=(n, 1))
        region = rng.choice(["a", "b", "c", "d"], size=n)
        effect = {"a": 0.0, "b": 0.5, "c": -0.3, "d": 0.1}
        y = np.array([effect[r] for r in region]) + z[:, 0] + rng.normal(size=n)
        block = indicator_block(region)

        reduced = np.column_stack([np.ones(n), z])
        full = np.column_stack([reduced, block])

        def rss(design: np.ndarray) -> float:
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            resid = y - design @ beta
            return float(resid @ resid)

        df1 = block.shape[1]
        df2 = n - full.shape[1]
        f_stat = ((rss(reduced) - rss(full)) / df1) / (rss(full) / df2)
        expected_p = stats.f.sf(f_stat, df1, df2)

        result = partial_correlation_ci(block, y, z)
        assert result.skip_reason is None
        assert result.p == pytest.approx(expected_p, rel=1e-9)
        assert result.df == df2

    def test_two_categorical_blocks_detect_dependence(self):
        rng = np.random.default_rng(4)
        n = 300
        region = rng.choice(["a", "b", "c"], size=n)
        independent = rng.choice(["p", "q", "r", "s"], size=n)
        dependent = np.where(region == "a", "p", independent)
        p_ind = partial_correlation_ci(
            indicator_block(region), indicator_block(independent)
        ).p
        p_dep = partial_correlation_ci(
            indicator_block(region), indicator_block(dependent)
        ).p
        assert p_dep < 1e-6
        assert p_dep < p_ind

    def test_two_categorical_blocks_are_calibrated_under_the_null(self):
        p_values = []
        for seed in range(200):
            rng = np.random.default_rng(seed)
            a = rng.choice(list("abc"), size=120)
            b = rng.choice(list("pqrs"), size=120)
            p_values.append(
                partial_correlation_ci(indicator_block(a), indicator_block(b)).p
            )
        assert stats.kstest(np.asarray(p_values), "uniform").pvalue > 0.01

    def test_single_level_categorical_is_a_named_skip(self):
        rng = np.random.default_rng(5)
        result = partial_correlation_ci(
            indicator_block(np.array(["only"] * 30)), rng.normal(size=30)
        )
        assert result.skip_reason == "zero_variance"

    def test_block_fully_explained_by_conditioners_is_a_named_skip(self):
        rng = np.random.default_rng(6)
        labels = rng.choice(["a", "b", "c"], size=120)
        onehot = pd.get_dummies(labels).to_numpy(dtype=float)
        result = partial_correlation_ci(
            indicator_block(labels), rng.normal(size=120), onehot
        )
        assert result.skip_reason == "zero_residual_variance"

    def test_tester_uses_string_columns_instead_of_skipping(self):
        rng = np.random.default_rng(7)
        n = 200
        group = np.array(["a", "b"] * (n // 2))
        x = rng.normal(size=n)
        df = pd.DataFrame({
            "X": x,
            "Y": 0.5 * x + rng.normal(scale=0.5, size=n),
            "G": group,
        })
        tester = _PartialCorrelationTester(nw.from_native(df), ["X", "Y", "G"])
        numeric = _PartialCorrelationTester(
            nw.from_native(df.assign(G=(group == "b").astype(float))), ["X", "Y", "G"]
        )
        for triple in [("X", "G", ()), ("X", "Y", ("G",)), ("G", "Y", ("X",))]:
            p_string = tester.p_value(*triple)
            assert p_string is not None
            assert p_string == pytest.approx(numeric.p_value(*triple), rel=1e-12)


class TestImplicationsAndFalsifyWithCategoricals:
    def test_test_implications_runs_and_matches_numeric_twin(self):
        rng = np.random.default_rng(8)
        n = 250
        group = rng.choice(["ctl", "trt"], size=n)
        x = rng.normal(size=n)
        m = 1.5 * x + np.where(group == "trt", 1.0, 0.0) + rng.normal(size=n)
        y = 0.8 * m + rng.normal(size=n)
        frame = pd.DataFrame({"x": x, "g": group, "m": m, "y": y})
        spec = "m ~ x + g\ny ~ m"
        string_result = pathmc.model(spec, data=frame).test_implications()
        numeric_result = pathmc.model(
            spec, data=frame.assign(g=(group == "trt").astype(float))
        ).test_implications()
        assert not string_result.results["p_value"].isna().any()
        np.testing.assert_allclose(
            string_result.results["p_value"], numeric_result.results["p_value"]
        )

    def test_test_implications_reports_multi_level_categorical(self, mediation_data):
        model = pathmc.model("m ~ x + region\ny ~ m + region", data=mediation_data)
        result = model.test_implications()
        statements = {
            (row["x"], row["y"], row["conditioning_set"])
            for _, row in result.results.iterrows()
        }
        assert ("region", "x", "") in statements
        assert ("x", "y", "m, region") in statements
        assert not result.results["p_value"].isna().any()
        assert (result.results["partial_corr"] >= 0).all()
        assert result.n_violations == 0

    def test_integer_coded_c_term_is_dummy_encoded(self, mediation_data):
        coded = mediation_data.assign(
            region=mediation_data["region"].map({"north": 1, "south": 2, "west": 3})
        )
        spec = "m ~ x + C(region)\ny ~ m + C(region)"
        coded_result = pathmc.model(spec, data=coded).test_implications()
        string_result = pathmc.model(spec, data=mediation_data).test_implications()
        np.testing.assert_allclose(
            coded_result.results["p_value"], string_result.results["p_value"]
        )

    def test_falsify_evaluates_categorical_model(self, mediation_data):
        model = pathmc.model("m ~ x + region\ny ~ m + region", data=mediation_data)
        result = model.falsify(random_seed=0)
        assert result.can_evaluate
        assert result.n_lmc_tests > 0
        assert result.falsified is not None

    def test_falsify_matches_numeric_twin_for_binary_categorical(self):
        rng = np.random.default_rng(9)
        n = 250
        group = rng.choice(["ctl", "trt"], size=n)
        x = rng.normal(size=n)
        m = 1.5 * x + np.where(group == "trt", 1.0, 0.0) + rng.normal(size=n)
        y = 0.8 * m + rng.normal(size=n)
        frame = pd.DataFrame({"x": x, "g": group, "m": m, "y": y})
        spec = "m ~ x + g\ny ~ m"
        string_result = pathmc.model(spec, data=frame).falsify(random_seed=1)
        numeric_result = pathmc.model(
            spec, data=frame.assign(g=(group == "trt").astype(float))
        ).falsify(random_seed=1)
        assert string_result.p_value_lmc == numeric_result.p_value_lmc
        assert string_result.given_lmc_violations == numeric_result.given_lmc_violations


# ---------------------------------------------------------------------------
# Item 4: refute_placebo() with a categorical treatment
# ---------------------------------------------------------------------------


class TestRefutePlaceboCategorical:
    @pytest.fixture
    def fitted(self, mediation_data, mock_pymc_sample) -> pathmc.PathModel:
        model = pathmc.model("m ~ x + region\ny ~ m + region", data=mediation_data)
        model.fit(draws=20, tune=20, chains=1, random_seed=0)
        return model

    def test_unseen_label_is_rejected(self, fitted):
        with pytest.raises(ValueError, match="fitted level labels.*'central'"):
            fitted.refute_placebo("y", "region", values=("north", "central"))

    def test_equal_labels_are_rejected(self, fitted):
        with pytest.raises(ValueError, match="distinct \\(lo, hi\\) labels"):
            fitted.refute_placebo("y", "region", values=("north", "north"))

    def test_labels_on_numeric_treatment_are_rejected(self, fitted):
        with pytest.raises(ValueError, match="numeric treatment 'x'.*Labels"):
            fitted.refute_placebo("y", "x", values=("north", "south"))

    def test_categorical_treatment_runs(self, fitted):
        result = fitted.refute_placebo(
            "y", "region", values=("north", "south"), n_permutations=2, random_seed=0
        )
        assert result.n_permutations == 2
        assert np.isfinite(result.fold_means).all()

    def test_permutation_preserves_declared_level_order(self, mock_pymc_sample):
        rng = np.random.default_rng(12)
        n = 60
        group = pd.Categorical(
            rng.choice(["west", "north"], size=n), categories=["west", "north"]
        )
        frame = pd.DataFrame({"g": group, "y": rng.normal(size=n)})
        model = pathmc.model("y ~ g", data=frame)
        assert list(model.design("y").columns) == ["Intercept", "g[T.north]"]
        model.fit(draws=10, tune=10, chains=1, random_seed=0)
        clone = model._refit_permuted("g", seed=1, sample_kwargs={})
        assert list(clone.design("y").columns) == ["Intercept", "g[T.north]"]
        original = np.asarray(model._data["g"].to_numpy(), dtype=object)
        permuted = np.asarray(clone._data["g"].to_numpy(), dtype=object)
        assert sorted(original.tolist()) == sorted(permuted.tolist())
        assert not np.array_equal(original, permuted)


# ---------------------------------------------------------------------------
# Item 5: counterfactual() with categorical evidence and interventions
# ---------------------------------------------------------------------------


class TestCounterfactualCategorical:
    @pytest.fixture
    def pinned(self, mediation_data, mock_pymc_sample) -> pathmc.PathModel:
        model = pathmc.model(
            "m ~ x + region\ny ~ m + C(region, reference='west')",
            data=mediation_data,
        )
        model.fit(draws=10, tune=10, chains=1, random_seed=0)
        posterior = model._idata.posterior.copy(deep=True)
        posterior["beta_m"].loc[{"m_predictors": "Intercept"}] = 0.25
        posterior["beta_m"].loc[{"m_predictors": "x"}] = 2.0
        posterior["beta_m_region"].loc[{"m_region_levels": "south"}] = 1.0
        posterior["beta_m_region"].loc[{"m_region_levels": "west"}] = -1.0
        posterior["beta_y"].loc[{"y_predictors": "Intercept"}] = -0.5
        posterior["beta_y"].loc[{"y_predictors": "m"}] = 1.5
        posterior["beta_y_region"].loc[{"y_region_levels": "north"}] = 0.4
        posterior["beta_y_region"].loc[{"y_region_levels": "south"}] = 0.9
        model._idata["posterior"] = posterior
        return model

    def test_matches_hand_computed_three_step_counterfactual(self, pinned):
        evidence = {"x": 0.4, "region": "south", "m": 2.0, "y": 3.5}
        result = pinned.counterfactual(evidence=evidence, do={"region": "north"})

        # Abduction with the pinned coefficients, then action + prediction.
        u_m = 2.0 - (0.25 + 2.0 * 0.4 + 1.0)
        u_y = 3.5 - (-0.5 + 1.5 * 2.0 + 0.9)
        m_cf = 0.25 + 2.0 * 0.4 + 0.0 + u_m
        y_cf = -0.5 + 1.5 * m_cf + 0.4 + u_y
        assert result.mean("m") == pytest.approx(m_cf)
        assert result.mean("y") == pytest.approx(y_cf)
        assert "region" not in result.dataset.data_vars

    def test_numeric_intervention_with_categorical_evidence(self, pinned):
        evidence = {"x": 0.4, "region": "west", "m": 2.0, "y": 3.5}
        result = pinned.counterfactual(evidence=evidence, do={"x": 1.4})
        u_m = 2.0 - (0.25 + 2.0 * 0.4 - 1.0)
        u_y = 3.5 - (-0.5 + 1.5 * 2.0 + 0.0)
        m_cf = 0.25 + 2.0 * 1.4 - 1.0 + u_m
        assert result.mean("m") == pytest.approx(m_cf)
        assert result.mean("y") == pytest.approx(-0.5 + 1.5 * m_cf + u_y)

    def test_categorical_evidence_is_required_even_when_partial(self, pinned):
        with pytest.raises(ValueError, match="level label for categorical.*'region'"):
            pinned.counterfactual(
                evidence={"x": 0.4, "m": 2.0, "y": 3.5},
                do={"x": 1.0},
                allow_partial_evidence=True,
            )

    def test_unseen_label_is_rejected(self, pinned):
        evidence = {"x": 0.4, "region": "south", "m": 2.0, "y": 3.5}
        with pytest.raises(ValueError, match="do value for categorical 'region'"):
            pinned.counterfactual(evidence=evidence, do={"region": "central"})
        with pytest.raises(ValueError, match="evidence value for categorical"):
            pinned.counterfactual(evidence={**evidence, "region": 3.0}, do={"x": 1.0})

    def test_no_extrapolation_warning_for_labels(self, pinned):
        evidence = {"x": 0.4, "region": "south", "m": 2.0, "y": 3.5}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pinned.counterfactual(evidence=evidence, do={"region": "west"})
