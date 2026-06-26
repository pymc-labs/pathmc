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
"""Tests for ResultReprMixin and repr consistency across all result objects.

No MCMC sampling — all result objects are constructed directly from synthetic
numpy draws or minimal fixture data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathmc.effects import EffectResult
from pathmc.falsify import FalsificationResult
from pathmc.identify import ImplicationTestResult
from pathmc.reprs import ReprSpec, ResultReprMixin, _render_html
from pathmc.sensitivity import SensitivityResult
from pathmc.simulate import DoResult, EstimandResult


# ---------------------------------------------------------------------------
# Fixtures — lightweight, no MCMC
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


@pytest.fixture
def draws_positive() -> np.ndarray:
    return RNG.normal(loc=0.37, scale=0.05, size=1000)


@pytest.fixture
def effect_result(draws_positive) -> EffectResult:
    return EffectResult(name="X→Y", draws=draws_positive)


@pytest.fixture
def do_result(draws_positive) -> DoResult:
    rng = np.random.default_rng(1)
    return DoResult(
        values={
            "X": rng.normal(loc=1.0, scale=0.1, size=1000),
            "Y": draws_positive,
            "Z": rng.normal(loc=0.5, scale=0.2, size=1000),
        },
        n_chains=1,
        n_draws=1000,
    )


@pytest.fixture
def estimand_result(draws_positive) -> EstimandResult:
    rng = np.random.default_rng(2)
    return EstimandResult(
        values={"Y": draws_positive, "X": rng.normal(size=1000)},
        outcome="Y",
        treatment="X",
        estimand="ATE",
        n_chains=1,
        n_draws=1000,
    )


@pytest.fixture
def sensitivity_result(draws_positive) -> SensitivityResult:
    from pathmc.sensitivity import compute_sensitivity

    return compute_sensitivity(
        observed_ate_draws=draws_positive,
        treatment="X",
        outcome="Y",
        gamma_range=(-1.0, 1.0),
        delta_range=(-1.0, 1.0),
        n_grid=10,
    )


@pytest.fixture
def falsification_result() -> FalsificationResult:
    rng = np.random.default_rng(3)
    return FalsificationResult(
        given_lmc_violations=1,
        n_lmc_tests=4,
        given_lmc_violation_fraction=0.25,
        perm_lmc_violation_fractions=rng.uniform(0, 1, size=100),
        perm_tpa_violation_fractions=rng.uniform(0, 1, size=100),
        p_value_lmc=0.04,
        p_value_tpa=0.03,
        n_permutations=100,
        n_in_mec=3,
        significance_level=0.05,
        significance_ci=0.05,
        local_violations=pd.DataFrame(
            columns=[
                "node",
                "non_descendant",
                "conditioning_set",
                "p_value",
                "violation",
            ]
        ),
    )


@pytest.fixture
def implication_result() -> ImplicationTestResult:
    df = pd.DataFrame({
        "x": ["A", "B"],
        "y": ["C", "D"],
        "conditioning_set": ["", "E"],
        "partial_corr": [0.01, 0.02],
        "p_value": [0.8, 0.7],
        "significant": [False, False],
    })
    return ImplicationTestResult(results=df, alpha=0.05)


# ---------------------------------------------------------------------------
# ReprSpec and _render_html unit tests
# ---------------------------------------------------------------------------


class TestReprSpec:
    def test_defaults(self):
        spec = ReprSpec(title="Test")
        assert spec.title == "Test"
        assert spec.rows == []
        assert spec.columns is None
        assert spec.footer is None

    def test_with_all_fields(self):
        spec = ReprSpec(
            title="My Result",
            rows=[["Mean", "0.37"], ["SD", "0.05"]],
            columns=None,
            footer="Methods: .hdi()",
        )
        assert len(spec.rows) == 2
        assert spec.footer == "Methods: .hdi()"


class TestRenderHtml:
    def test_returns_string(self):
        spec = ReprSpec(title="Test", rows=[["A", "1"], ["B", "2"]])
        html = _render_html(spec)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_contains_title(self):
        spec = ReprSpec(title="ATE of X on Y", rows=[["Mean", "0.37"]])
        html = _render_html(spec)
        assert "ATE of X on Y" in html

    def test_label_value_layout_has_no_thead(self):
        spec = ReprSpec(title="Test", rows=[["Mean", "0.37"]], columns=None)
        html = _render_html(spec)
        assert "<thead>" not in html

    def test_tabular_layout_has_thead(self):
        spec = ReprSpec(
            title="Test",
            rows=[["X", "0.37", "[0.32, 0.43]"]],
            columns=["variable", "mean", "94% HDI"],
        )
        html = _render_html(spec)
        assert "<thead>" in html
        assert "variable" in html

    def test_footer_rendered(self):
        spec = ReprSpec(title="Test", rows=[], footer="Methods: .hdi()")
        html = _render_html(spec)
        assert "Methods: .hdi()" in html

    def test_no_footer_renders_clean(self):
        spec = ReprSpec(title="Test", rows=[["Mean", "0.1"]], footer=None)
        html = _render_html(spec)
        assert 'class="pathmc-result-footer"' not in html

    def test_contains_table_tag(self):
        spec = ReprSpec(title="T", rows=[["A", "1"]])
        html = _render_html(spec)
        assert "<table" in html


# ---------------------------------------------------------------------------
# ResultReprMixin unit tests
# ---------------------------------------------------------------------------


class TestResultReprMixin:
    def test_default_repr_compact_returns_classname(self):
        class Bare(ResultReprMixin):
            pass

        obj = Bare()
        assert repr(obj) == "<Bare>"

    def test_repr_never_raises_on_broken_compact(self):
        class Broken(ResultReprMixin):
            def _repr_compact(self) -> str:
                raise RuntimeError("oops")

        obj = Broken()
        r = repr(obj)
        assert r == "<Broken>"

    def test_repr_html_never_raises_on_broken_spec(self):
        class Broken(ResultReprMixin):
            def _repr_compact(self) -> str:
                return "Broken()"

            def _repr_spec(self):
                raise RuntimeError("oops")

        obj = Broken()
        html = obj._repr_html_()
        assert isinstance(html, str)

    def test_custom_repr_compact_used_for_repr(self):
        class MyResult(ResultReprMixin):
            def _repr_compact(self) -> str:
                return "MyResult(answer=42)"

        obj = MyResult()
        assert repr(obj) == "MyResult(answer=42)"


# ---------------------------------------------------------------------------
# EffectResult repr tests
# ---------------------------------------------------------------------------


class TestEffectResultRepr:
    def test_repr_is_single_line(self, effect_result):
        r = repr(effect_result)
        assert "\n" not in r

    def test_repr_contains_name(self, effect_result):
        assert "X→Y" in repr(effect_result)

    def test_repr_contains_mean(self, effect_result):
        assert "mean=" in repr(effect_result)

    def test_repr_contains_hdi(self, effect_result):
        assert "HDI=" in repr(effect_result)

    def test_repr_html_contains_table(self, effect_result):
        html = effect_result._repr_html_()
        assert "<table" in html

    def test_repr_html_contains_p_gt_zero(self, effect_result):
        html = effect_result._repr_html_()
        assert "P(> 0)" in html

    def test_repr_html_contains_title(self, effect_result):
        html = effect_result._repr_html_()
        assert "X→Y" in html

    def test_prob_gt_zero_property(self, draws_positive):
        result = EffectResult(name="X→Y", draws=draws_positive)
        assert 0.0 <= result.prob_gt_zero <= 1.0
        assert result.prob_gt_zero > 0.99  # all positive draws

    def test_repr_never_raises_on_nan_draws(self):
        bad = EffectResult(name="X→Y", draws=np.array([np.nan] * 100))
        r = repr(bad)
        assert isinstance(r, str)


# ---------------------------------------------------------------------------
# DoResult repr tests
# ---------------------------------------------------------------------------


class TestDoResultRepr:
    def test_repr_is_single_line(self, do_result):
        r = repr(do_result)
        assert "\n" not in r

    def test_repr_contains_draws(self, do_result):
        assert "1000 draws" in repr(do_result)

    def test_repr_contains_n_vars(self, do_result):
        assert "3 variables" in repr(do_result)

    def test_repr_empty(self):
        r = repr(DoResult(values={}))
        assert "empty" in r.lower()

    def test_repr_html_contains_table(self, do_result):
        html = do_result._repr_html_()
        assert "<table" in html

    def test_repr_html_contains_columns(self, do_result):
        html = do_result._repr_html_()
        assert "variable" in html
        assert "mean" in html

    def test_repr_html_contains_variable_names(self, do_result):
        html = do_result._repr_html_()
        assert "Y" in html

    def test_repr_html_empty_do_result(self):
        html = DoResult(values={})._repr_html_()
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# EstimandResult repr tests
# ---------------------------------------------------------------------------


class TestEstimandResultRepr:
    def test_repr_is_single_line(self, estimand_result):
        r = repr(estimand_result)
        assert "\n" not in r

    def test_repr_contains_estimand(self, estimand_result):
        assert "ATE" in repr(estimand_result)

    def test_repr_contains_treatment_and_outcome(self, estimand_result):
        assert "X" in repr(estimand_result)
        assert "Y" in repr(estimand_result)

    def test_repr_contains_mean(self, estimand_result):
        assert "mean=" in repr(estimand_result)

    def test_repr_html_contains_p_gt_zero(self, estimand_result):
        html = estimand_result._repr_html_()
        assert "P(> 0)" in html

    def test_repr_html_contains_draws(self, estimand_result):
        html = estimand_result._repr_html_()
        assert "1000" in html

    def test_repr_html_contains_title(self, estimand_result):
        html = estimand_result._repr_html_()
        assert "ATE" in html


# ---------------------------------------------------------------------------
# SensitivityResult repr tests
# ---------------------------------------------------------------------------


class TestSensitivityResultRepr:
    def test_repr_is_single_line(self, sensitivity_result):
        r = repr(sensitivity_result)
        assert "\n" not in r

    def test_repr_contains_treatment_and_outcome(self, sensitivity_result):
        r = repr(sensitivity_result)
        assert "X" in r
        assert "Y" in r

    def test_repr_contains_ate(self, sensitivity_result):
        assert "ATE=" in repr(sensitivity_result)

    def test_repr_contains_tipping_point(self, sensitivity_result):
        assert "tipping_point=" in repr(sensitivity_result)

    def test_repr_html_contains_table(self, sensitivity_result):
        html = sensitivity_result._repr_html_()
        assert "<table" in html

    def test_repr_html_contains_tipping_point_row(self, sensitivity_result):
        html = sensitivity_result._repr_html_()
        assert "Tipping point" in html

    def test_repr_html_contains_hdi(self, sensitivity_result):
        html = sensitivity_result._repr_html_()
        assert "HDI" in html


# ---------------------------------------------------------------------------
# FalsificationResult repr tests
# ---------------------------------------------------------------------------


class TestFalsificationResultRepr:
    def test_repr_is_single_line(self, falsification_result):
        r = repr(falsification_result)
        assert "\n" not in r

    def test_repr_contains_verdict(self, falsification_result):
        r = repr(falsification_result)
        assert any(v in r for v in ("not_rejected", "falsified", "not_falsifiable"))

    def test_repr_contains_p_lmc(self, falsification_result):
        assert "p_lmc=" in repr(falsification_result)

    def test_repr_html_preserved(self, falsification_result):
        """Existing _repr_html_ must still return rich HTML (not the mixin's stat table)."""
        html = falsification_result._repr_html_()
        assert "<h4>DAG Falsification" in html

    def test_repr_not_evaluable(self):
        rng = np.random.default_rng(99)
        result = FalsificationResult(
            given_lmc_violations=0,
            n_lmc_tests=0,
            given_lmc_violation_fraction=0.0,
            perm_lmc_violation_fractions=rng.uniform(size=10),
            perm_tpa_violation_fractions=rng.uniform(size=10),
            p_value_lmc=1.0,
            p_value_tpa=1.0,
            n_permutations=0,
            n_in_mec=0,
            significance_level=0.05,
            significance_ci=0.05,
            local_violations=pd.DataFrame(
                columns=[
                    "node",
                    "non_descendant",
                    "conditioning_set",
                    "p_value",
                    "violation",
                ]
            ),
        )
        r = repr(result)
        assert "not_evaluable" in r
        assert "\n" not in r


# ---------------------------------------------------------------------------
# ImplicationTestResult repr tests
# ---------------------------------------------------------------------------


class TestImplicationTestResultRepr:
    def test_repr_is_single_line(self, implication_result):
        r = repr(implication_result)
        assert "\n" not in r

    def test_repr_contains_n_tests(self, implication_result):
        assert "2 tests" in repr(implication_result)

    def test_repr_contains_violations(self, implication_result):
        assert "0 violations" in repr(implication_result)

    def test_repr_contains_alpha(self, implication_result):
        assert "α=0.05" in repr(implication_result)

    def test_repr_html_preserved(self, implication_result):
        """Existing _repr_html_ must still return rich HTML (not the mixin's stat table)."""
        html = implication_result._repr_html_()
        assert "<h4>DAG Implication Tests" in html

    def test_repr_with_violations(self):
        df = pd.DataFrame({
            "x": ["A"],
            "y": ["B"],
            "conditioning_set": [""],
            "partial_corr": [0.55],
            "p_value": [0.001],
            "significant": [True],
        })
        result = ImplicationTestResult(results=df, alpha=0.05)
        r = repr(result)
        assert "1 violations" in r
        assert "\n" not in r
