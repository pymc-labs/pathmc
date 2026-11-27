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
"""Tests for ATT/ATU: canonical subgroup-aware treatment effects.

TestAttAtuAPI contains fast tests verifying the interface surface.
TestAttAtuSemantics contains slow tests verifying estimand correctness.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

import pathmc
from pathmc.idata import posterior as _posterior


def _n_posterior_samples(model) -> int:
    """Number of posterior draws (chains * draws) for a fitted model."""
    ds = _posterior(model._idata)
    return ds.sizes["chain"] * ds.sizes["draw"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFOUNDED_SPEC = "Y ~ T + X"
INTERACTION_SPEC = "Y ~ T + X + T:X"


def _make_binary_treatment_data(rng, n=500):
    """Confounded binary treatment with known DGP.

    X -> T (confounding), X -> Y, T -> Y.
    True causal effect of T on Y = 0.5 (constant, no interaction).
    """
    X = rng.normal(size=n)
    T = (rng.uniform(size=n) < expit(0.8 * X)).astype(float)
    Y = 0.5 * T + 0.3 * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "T": T, "Y": Y})


def _make_interaction_data(rng, n=500):
    """Binary treatment with effect modification (interaction).

    T -> Y effect = 0.5 + 0.4*X, so ATT != ATU when E[X|T=1] != E[X|T=0].
    """
    X = rng.normal(size=n)
    T = (rng.uniform(size=n) < expit(0.8 * X)).astype(float)
    Y = 0.5 * T + 0.3 * X + 0.4 * T * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "T": T, "Y": Y})


def _fixed_categorical_model(draws=4):
    """Bernoulli model with fixed draws and subgroup-specific logit effects."""
    region = np.repeat(["north", "south", "west"], [13, 7, 5])
    x = np.concatenate([
        np.full(13, np.log(3.0)),
        np.zeros(7),
        np.full(5, np.log(2.0)),
    ])
    data = pd.DataFrame({"Y": np.zeros(len(region)), "x": x, "region": region})
    model = pathmc.model(
        "Y ~ x + region",
        data=data,
        families={"Y": "bernoulli"},
    )
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.tile([0.0, 1.0], (1, draws, 1)),
                "beta_Y_region": np.tile([np.log(3.0), 0.0], (1, draws, 1)),
            },
            "observed_data": {"Y": data["Y"].to_numpy()},
        },
        coords={
            "obs": np.arange(len(data)),
            "Y_predictors": ["Intercept", "x"],
            "Y_region_levels": ["south", "west"],
        },
        dims={
            "Y": ["obs"],
            "beta_Y": ["Y_predictors"],
            "beta_Y_region": ["Y_region_levels"],
        },
    )
    return model, data


def _fixed_integer_categorical_model(low, high):
    region = np.repeat([low, high], [11, 5])
    x = np.concatenate([np.full(11, np.log(3.0)), np.zeros(5)])
    data = pd.DataFrame({"Y": np.zeros(len(region)), "x": x, "region": region})
    model = pathmc.model(
        "Y ~ x + C(region)",
        data=data,
        families={"Y": "bernoulli"},
    )
    call = model._categorical_terms[0][1]
    reference = call.reference
    coefficient_levels = [level for level in call.levels if level != reference]
    coefficient_coords = list(model.pymc_model.coords["Y_region_levels"])
    effects = {low: 0.0, high: np.log(3.0)}
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.tile([effects[reference], 1.0], (1, 4, 1)),
                "beta_Y_region": np.tile(
                    [
                        effects[level] - effects[reference]
                        for level in coefficient_levels
                    ],
                    (1, 4, 1),
                ),
            }
        },
        coords={
            "Y_predictors": ["Intercept", "x"],
            "Y_region_levels": coefficient_coords,
        },
        dims={
            "beta_Y": ["Y_predictors"],
            "beta_Y_region": ["Y_region_levels"],
        },
    )
    return model


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_data(rng):
    return _make_binary_treatment_data(rng)


@pytest.fixture
def interaction_data(rng):
    return _make_interaction_data(rng)


@pytest.fixture(scope="module")
def fitted_binary():
    """Binary treatment model fitted with minimal draws."""
    rng = np.random.default_rng(42)
    df = _make_binary_treatment_data(rng)
    model = pathmc.model(CONFOUNDED_SPEC, data=df)
    model.fit(draws=300, tune=300, chains=2, random_seed=42)
    return model


@pytest.fixture(scope="module")
def fitted_interaction():
    """Interaction model fitted with minimal draws."""
    rng = np.random.default_rng(42)
    df = _make_interaction_data(rng)
    model = pathmc.model(INTERACTION_SPEC, data=df)
    model.fit(draws=300, tune=300, chains=2, random_seed=42)
    return model


# ---------------------------------------------------------------------------
# Fast tests: API surface
# ---------------------------------------------------------------------------


class TestAttAtuAPI:
    def test_att_method_exists(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        assert hasattr(model, "att")
        assert callable(model.att)

    def test_atu_method_exists(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        assert hasattr(model, "atu")
        assert callable(model.atu)

    def test_att_before_sampling_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.att("Y", "T")

    def test_atu_before_sampling_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        with pytest.raises(RuntimeError, match="No posterior samples"):
            model.atu("Y", "T")

    def test_att_no_matching_rows_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        model._idata = True  # bypass sampling check
        with pytest.raises(
            ValueError, match="No observations.*Check the treated_value parameter"
        ):
            model.att("Y", "T", treated_value=999.0)

    def test_atu_no_matching_rows_raises(self, binary_data):
        model = pathmc.model(CONFOUNDED_SPEC, data=binary_data)
        model._idata = True  # bypass sampling check
        with pytest.raises(
            ValueError, match="No observations.*Check the untreated_value parameter"
        ):
            model.atu("Y", "T", untreated_value=999.0)


class TestCategoricalAttAtu:
    def test_observed_subgroups_not_whole_frame(self):
        model, _ = _fixed_categorical_model()

        att = model.att("Y", "region", values=("north", "south"))
        atu = model.atu("Y", "region", values=("north", "south"))

        assert att.mean() == pytest.approx(1 / 4)
        assert atu.mean() == pytest.approx(3 / 20)
        assert att.outcome == atu.outcome == "Y"
        assert att.treatment == atu.treatment == "region"
        assert list(att.summary().index) == ["ATT"]
        assert list(atu.summary().index) == ["ATU"]

    def test_swapped_endpoints_reverse_sign_in_fixed_subgroup(self):
        model, _ = _fixed_categorical_model()

        forward = model.att(
            "Y",
            "region",
            values=("north", "south"),
            treated_value="south",
        )
        reverse = model.att(
            "Y",
            "region",
            values=("south", "north"),
            treated_value="south",
        )

        np.testing.assert_allclose(reverse.draws(), -forward.draws())

    def test_explicit_third_level_subgroup_uses_only_those_rows(self):
        model, _ = _fixed_categorical_model()

        result = model.att(
            "Y",
            "region",
            values=("north", "south"),
            treated_value="west",
        )

        assert result.mean() == pytest.approx(4 / 21)

    def test_default_subgroups_follow_contrast_endpoints(self):
        model, _ = _fixed_categorical_model()

        att = model.att("Y", "region", values=("south", "west"))
        atu = model.atu("Y", "region", values=("south", "north"))

        expected_att = expit(np.log(2.0)) - expit(np.log(6.0))
        expected_atu = expit(0.0) - expit(np.log(3.0))
        assert att.mean() == pytest.approx(expected_att)
        assert atu.mean() == pytest.approx(expected_atu)

    def test_large_adjacent_integer_labels_are_not_merged(self):
        low = 2**53
        high = low + 1
        model = _fixed_integer_categorical_model(low, high)

        result = model.att("Y", "region", values=(low, high))

        assert result.mean() == pytest.approx(1 / 4)

    @pytest.mark.parametrize(
        "values",
        [
            np.array(["north", "south"]),
            pd.Index(["north", "south"]),
        ],
    )
    def test_endpoint_pair_accepts_non_string_iterables(self, values):
        model, _ = _fixed_categorical_model()

        result = model.att("Y", "region", values=values)

        assert result.mean() == pytest.approx(1 / 4)

    def test_endpoint_pair_accepts_generator(self):
        model, _ = _fixed_categorical_model()

        generated = model.att(
            "Y", "region", values=(value for value in ("north", "south"))
        )
        paired = model.att("Y", "region", values=("north", "south"))

        np.testing.assert_array_equal(generated.draws(), paired.draws())

    @pytest.mark.parametrize("values", ["ab", b"ab"])
    def test_endpoint_pair_rejects_string_like_values(self, values):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError, match="non-string iterable of exactly two labels"
        ):
            model.att("Y", "region", values=values)

    def test_endpoint_pair_rejects_zero_dimensional_array(self):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError, match="non-string iterable of exactly two labels"
        ):
            model.att("Y", "region", values=np.array("north"))

    def test_endpoint_pair_rejects_three_item_array(self):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError, match="non-string iterable of exactly two labels"
        ):
            model.att(
                "Y",
                "region",
                values=np.array(["north", "south", "west"]),
            )

    @pytest.mark.parametrize("method", ["att", "atu"])
    def test_missing_categorical_pair_names_fix(self, method):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError,
            match=r"region.*explicit endpoint pair.*north.*south.*values=",
        ):
            getattr(model, method)("Y", "region")

    @pytest.mark.parametrize(
        "values",
        [("north",), ("north", "south", "west")],
    )
    def test_categorical_pair_requires_exactly_two_labels(self, values):
        model, _ = _fixed_categorical_model()

        with pytest.raises(ValueError) as exc_info:
            model.att("Y", "region", values=values)

        message = str(exc_info.value)
        assert "region" in message
        assert repr(values) in message
        assert repr(["north", "south", "west"]) in message
        assert "Pass values=(lo_label, hi_label)" in message

    @pytest.mark.parametrize("endpoint", ["east", 2])
    def test_unknown_endpoint_names_label_levels_and_fix(self, endpoint):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError,
            match=rf"region.*endpoint.*{endpoint!s}.*north.*south.*Pass",
        ):
            model.att(
                "Y",
                "region",
                values=("north", endpoint),
                treated_value="south",
            )

    @pytest.mark.parametrize("subgroup", ["east", 2])
    def test_unknown_subgroup_names_label_levels_and_fix(self, subgroup):
        model, _ = _fixed_categorical_model()

        with pytest.raises(
            ValueError,
            match=rf"region.*treated_value.*{subgroup!s}.*north.*south.*Pass",
        ):
            model.att(
                "Y",
                "region",
                values=("north", "south"),
                treated_value=subgroup,
            )

    def test_numeric_looking_text_does_not_match_integer_level(self):
        model = _fixed_integer_categorical_model(2, 10)

        with pytest.raises(
            ValueError,
            match=r"region.*treated_value.*'2'.*10.*2.*Pass",
        ):
            model.att("Y", "region", values=(2, 10), treated_value="2")

    def test_bool_does_not_match_integer_level(self):
        model = _fixed_integer_categorical_model(0, 1)

        with pytest.raises(
            ValueError,
            match=r"region.*treated_value.*True.*0.*1.*Pass",
        ):
            model.att("Y", "region", values=(0, 1), treated_value=True)

    def test_boolean_categorical_labels_remain_valid(self):
        model = _fixed_integer_categorical_model(False, True)

        result = model.att("Y", "region", values=(False, True))

        assert result.mean() == pytest.approx(1 / 4)

    def test_ordered_categorical_att_atu_use_declared_levels_and_subgroups(self):
        _, data = _fixed_categorical_model()
        ordered = data.copy()
        ordered["region"] = pd.Categorical(
            ordered["region"],
            categories=["west", "north", "south"],
            ordered=True,
        )
        model = pathmc.model(
            "Y ~ x + region",
            data=ordered,
            families={"Y": "bernoulli"},
        )
        call = model._categorical_terms[0][1]
        assert list(call.levels) == ["west", "north", "south"]
        effects = {"west": np.log(2.0), "north": 0.0, "south": np.log(3.0)}
        coefficient_levels = [level for level in call.levels if level != call.reference]
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_Y": np.tile(
                        [effects[call.reference], 1.0],
                        (1, 4, 1),
                    ),
                    "beta_Y_region": np.tile(
                        [
                            effects[level] - effects[call.reference]
                            for level in coefficient_levels
                        ],
                        (1, 4, 1),
                    ),
                }
            },
            coords={
                "Y_predictors": ["Intercept", "x"],
                "Y_region_levels": list(model.pymc_model.coords["Y_region_levels"]),
            },
            dims={
                "beta_Y": ["Y_predictors"],
                "beta_Y_region": ["Y_region_levels"],
            },
        )

        att = model.att("Y", "region", values=("north", "south"))
        atu = model.atu("Y", "region", values=("north", "south"))

        assert att.mean() == pytest.approx(1 / 4)
        assert atu.mean() == pytest.approx(3 / 20)

    @pytest.mark.parametrize(
        ("method", "expected"),
        [("att", 1 / 4), ("atu", 3 / 20)],
    )
    def test_predictive_mode_with_fixed_posterior(self, method, expected):
        model, _ = _fixed_categorical_model(draws=800)

        result = getattr(model, method)(
            "Y",
            "region",
            values=("north", "south"),
            kind="predictive",
        )

        assert result.mean() == pytest.approx(expected, abs=0.04)

    def test_intervention_reaches_each_equation_using_category(self):
        region = np.repeat(["north", "south"], 4)
        data = pd.DataFrame({
            "region": region,
            "M": np.zeros(len(region)),
            "Y": np.zeros(len(region)),
        })
        model = pathmc.model("M ~ region\nY ~ M + region", data=data)
        model._idata = az.from_dict(
            {
                "posterior": {
                    "beta_M": np.zeros((1, 2, 1)),
                    "beta_M_region": np.ones((1, 2, 1)),
                    "sigma_M": np.ones((1, 2)),
                    "beta_Y": np.tile([0.0, 1.0], (1, 2, 1)),
                    "beta_Y_region": np.full((1, 2, 1), 2.0),
                    "sigma_Y": np.ones((1, 2)),
                }
            },
            coords={
                "M_predictors": ["Intercept"],
                "M_region_levels": ["south"],
                "Y_predictors": ["Intercept", "M"],
                "Y_region_levels": ["south"],
            },
            dims={
                "beta_M": ["M_predictors"],
                "beta_M_region": ["M_region_levels"],
                "beta_Y": ["Y_predictors"],
                "beta_Y_region": ["Y_region_levels"],
            },
        )

        result = model.att("Y", "region", values=("north", "south"))

        assert result.mean() == pytest.approx(3.0)


def test_numeric_defaults_and_matching_regression():
    data = pd.DataFrame({
        "T": [0.0, 1.0, 0.0, 1.0],
        "X": [0.0, 1.0, 2.0, 3.0],
        "Y": np.zeros(4),
    })
    model = pathmc.model("Y ~ T + X + T:X", data=data)
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.tile([0.0, 2.0, 0.0, 1.0], (1, 3, 1)),
                "sigma_Y": np.ones((1, 3)),
            }
        },
        coords={"Y_predictors": ["Intercept", "T", "X", "T:X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )

    implicit_att = model.att("Y", "T", values=(0.0, 2.0))
    explicit_att = model.att("Y", "T", values=(0.0, 2.0), treated_value=1.0)
    implicit_atu = model.atu("Y", "T", values=(0.0, 2.0))
    explicit_atu = model.atu("Y", "T", values=(0.0, 2.0), untreated_value=0.0)

    np.testing.assert_array_equal(implicit_att.draws(), explicit_att.draws())
    np.testing.assert_array_equal(implicit_atu.draws(), explicit_atu.draws())
    assert implicit_att.mean() == pytest.approx(8.0)
    assert implicit_atu.mean() == pytest.approx(6.0)


def test_numeric_treatment_accepts_numpy_endpoint_pair():
    data = pd.DataFrame({
        "T": [0.0, 1.0, 0.0, 1.0],
        "Y": np.zeros(4),
    })
    model = pathmc.model("Y ~ T", data=data)
    model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.tile([0.0, 2.0], (1, 3, 1)),
                "sigma_Y": np.ones((1, 3)),
            }
        },
        coords={"Y_predictors": ["Intercept", "T"]},
        dims={"beta_Y": ["Y_predictors"]},
    )

    array_result = model.att("Y", "T", values=np.array([0.0, 2.0]))
    tuple_result = model.att("Y", "T", values=(0.0, 2.0))

    np.testing.assert_array_equal(array_result.draws(), tuple_result.draws())


@pytest.mark.parametrize("method", ["att", "atu"])
def test_adjustment_wrapper_categorical_defaults_match_direct_model(method):
    model, _ = _fixed_categorical_model()
    adjusted = model.adjustment_model("region -> Y", formula="Y ~ x + region")
    adjusted.outcome_model._idata = model._idata

    direct = getattr(adjusted.outcome_model, method)(
        "Y", "region", values=("north", "south")
    )
    wrapped = getattr(adjusted, method)(values=("north", "south"))

    np.testing.assert_array_equal(wrapped.draws(), direct.draws())


@pytest.mark.parametrize("method", ["att", "atu"])
@pytest.mark.parametrize(
    "values",
    [
        ["north", "south"],
        np.array(["north", "south"]),
    ],
)
def test_adjustment_wrapper_accepts_iterable_endpoint_pair(method, values):
    model, _ = _fixed_categorical_model()
    adjusted = model.adjustment_model("region -> Y", formula="Y ~ x + region")
    adjusted.outcome_model._idata = model._idata

    direct = getattr(adjusted.outcome_model, method)(
        "Y", "region", values=("north", "south")
    )
    wrapped = getattr(adjusted, method)(values=values)

    np.testing.assert_array_equal(wrapped.draws(), direct.draws())


@pytest.mark.parametrize("method", ["att", "atu"])
def test_adjustment_wrapper_integer_categorical_defaults_match_direct_model(method):
    model = _fixed_integer_categorical_model(1, 2)
    adjusted = model.adjustment_model("region -> Y", formula="Y ~ x + C(region)")
    adjusted.outcome_model._idata = model._idata

    direct = getattr(adjusted.outcome_model, method)("Y", "region", values=(1, 2))
    wrapped = getattr(adjusted, method)(values=(1, 2))

    np.testing.assert_array_equal(wrapped.draws(), direct.draws())


@pytest.mark.parametrize("method", ["att", "atu"])
def test_adjustment_wrapper_numeric_defaults_match_direct_model(method):
    data = pd.DataFrame({
        "T": [0.0, 1.0, 0.0, 1.0],
        "X": [0.0, 1.0, 2.0, 3.0],
        "Y": np.zeros(4),
    })
    model = pathmc.model("Y ~ T + X + T:X", data=data)
    adjusted = model.adjustment_model("T -> Y", formula="Y ~ T + X + T:X")
    adjusted.outcome_model._idata = az.from_dict(
        {
            "posterior": {
                "beta_Y": np.tile([0.0, 2.0, 0.0, 1.0], (1, 3, 1)),
                "sigma_Y": np.ones((1, 3)),
            }
        },
        coords={"Y_predictors": ["Intercept", "T", "X", "T:X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )

    direct = getattr(adjusted.outcome_model, method)("Y", "T")
    wrapped = getattr(adjusted, method)()

    np.testing.assert_array_equal(wrapped.draws(), direct.draws())


# ---------------------------------------------------------------------------
# Slow tests: estimand semantics
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAttAtuSemantics:
    """Verify ATT/ATU return finite, reasonable values."""

    def test_att_returns_do_result(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        assert hasattr(att, "mean")
        assert hasattr(att, "hdi")

    def test_atu_returns_do_result(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        assert hasattr(atu, "mean")
        assert hasattr(atu, "hdi")

    def test_att_finite(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        assert np.isfinite(att.mean("Y"))

    def test_atu_finite(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        assert np.isfinite(atu.mean("Y"))

    def test_att_draws_count_equals_posterior_samples(self, fitted_binary):
        # The subgroup average must collapse over treated units within each
        # draw, leaving exactly one value per posterior sample.
        n_samples = _n_posterior_samples(fitted_binary)
        att = fitted_binary.att("Y", "T")
        assert att.draws().shape == (n_samples,)

    def test_atu_draws_count_equals_posterior_samples(self, fitted_binary):
        n_samples = _n_posterior_samples(fitted_binary)
        atu = fitted_binary.atu("Y", "T")
        assert atu.draws().shape == (n_samples,)

    def test_att_hdi(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        hdi = att.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]

    def test_atu_hdi(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T")
        hdi = atu.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]


@pytest.mark.slow
class TestAttAtuLinearConsistency:
    """In a linear model without interactions, ATT ≈ ATU ≈ ATE."""

    def test_att_close_to_ate(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T")
        att = fitted_binary.att("Y", "T")
        assert abs(att.mean("Y") - ate.mean("Y")) < 0.1

    def test_atu_close_to_ate(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T")
        atu = fitted_binary.atu("Y", "T")
        assert abs(atu.mean("Y") - ate.mean("Y")) < 0.1

    def test_att_close_to_atu(self, fitted_binary):
        att = fitted_binary.att("Y", "T")
        atu = fitted_binary.atu("Y", "T")
        assert abs(att.mean("Y") - atu.mean("Y")) < 0.1


@pytest.mark.slow
class TestAttAtuInteraction:
    """With effect modification, ATT and ATU should diverge."""

    def test_att_differs_from_atu(self, fitted_interaction):
        att = fitted_interaction.att("Y", "T")
        atu = fitted_interaction.atu("Y", "T")
        # ATT > ATU because treated units have higher X on average
        # and the interaction coefficient is positive
        assert att.mean("Y") > atu.mean("Y")

    def test_att_positive(self, fitted_interaction):
        att = fitted_interaction.att("Y", "T")
        assert att.mean("Y") > 0

    def test_atu_positive(self, fitted_interaction):
        atu = fitted_interaction.atu("Y", "T")
        assert atu.mean("Y") > 0


@pytest.mark.slow
class TestNonDefaultCoding:
    """Treatment coded as {-1, +1} instead of {0, 1}."""

    @pytest.fixture(scope="class")
    def fitted_alt_coding(self):
        rng = np.random.default_rng(42)
        n = 500
        X = rng.normal(size=n)
        T = np.where(rng.uniform(size=n) < expit(0.8 * X), 1.0, -1.0)
        Y = 0.5 * T + 0.3 * X + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"X": X, "T": T, "Y": Y})
        model = pathmc.model("Y ~ T + X", data=df)
        model.fit(draws=300, tune=300, chains=2, random_seed=42)
        return model

    def test_att_with_custom_treated_value(self, fitted_alt_coding):
        att = fitted_alt_coding.att("Y", "T", treated_value=1.0)
        assert np.isfinite(att.mean("Y"))

    def test_atu_with_custom_untreated_value(self, fitted_alt_coding):
        atu = fitted_alt_coding.atu("Y", "T", untreated_value=-1.0)
        assert np.isfinite(atu.mean("Y"))

    def test_custom_values_tuple(self, fitted_alt_coding):
        att = fitted_alt_coding.att("Y", "T", values=(-1.0, 1.0), treated_value=1.0)
        assert att.mean("Y") > 0


@pytest.mark.slow
class TestKindVariants:
    """Both kind='mean' and kind='predictive' work."""

    def test_att_kind_mean(self, fitted_binary):
        att = fitted_binary.att("Y", "T", kind="mean")
        assert np.isfinite(att.mean("Y"))

    def test_att_kind_predictive(self, fitted_binary):
        att = fitted_binary.att("Y", "T", kind="predictive")
        assert np.isfinite(att.mean("Y"))

    def test_atu_kind_mean(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T", kind="mean")
        assert np.isfinite(atu.mean("Y"))

    def test_atu_kind_predictive(self, fitted_binary):
        atu = fitted_binary.atu("Y", "T", kind="predictive")
        assert np.isfinite(atu.mean("Y"))

    def test_mean_and_predictive_similar(self, fitted_binary):
        att_mean = fitted_binary.att("Y", "T", kind="mean")
        att_pred = fitted_binary.att("Y", "T", kind="predictive")
        assert abs(att_mean.mean("Y") - att_pred.mean("Y")) < 0.3


@pytest.mark.slow
class TestExistingBehaviorUnchanged:
    """Existing ate()/cate() behavior is not affected."""

    def test_ate_unchanged(self, fitted_binary):
        ate = fitted_binary.ate("Y", "T", values=(0.0, 1.0))
        r0 = fitted_binary.do(set={"T": 0.0})
        r1 = fitted_binary.do(set={"T": 1.0})
        manual = r1 - r0
        assert abs(ate.mean("Y") - manual.mean("Y")) < 1e-10

    def test_cate_unchanged(self, fitted_binary):
        cate = fitted_binary.cate("Y", "T", values=(0.0, 1.0), condition={"X": 1.0})
        r0 = fitted_binary.do(set={"T": 0.0, "X": 1.0})
        r1 = fitted_binary.do(set={"T": 1.0, "X": 1.0})
        manual = r1 - r0
        assert abs(cate.mean("Y") - manual.mean("Y")) < 1e-10
