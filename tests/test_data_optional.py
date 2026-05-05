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
"""Tests for data-free DAG exploration (issue #138).

Verifies that pathmc.model(spec) without data supports introspection
and identification, while data-requiring methods raise RuntimeError.
"""

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior

import pathmc

MEDIATION_SPEC = """\
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""

FORK_SPEC = """\
X ~ Z
Y ~ X + Z
"""

COLLIDER_SPEC = "C ~ X + Y"


class TestDataFreeCreation:
    """pathmc.model(spec) without data succeeds."""

    def test_model_without_data(self):
        m = pathmc.model(MEDIATION_SPEC)
        assert m is not None

    def test_model_without_data_simple(self):
        m = pathmc.model("Y ~ X")
        assert m is not None

    def test_model_without_data_fork(self):
        m = pathmc.model(FORK_SPEC)
        assert m is not None


class TestIntrospectionWithoutData:
    """graph(), equations(), priors() work on data-free model."""

    def test_graph(self):
        m = pathmc.model(MEDIATION_SPEC)
        g = m.graph()
        assert g is not None

    def test_equations(self):
        m = pathmc.model(MEDIATION_SPEC)
        eqs = m.equations()
        text = str(eqs)
        assert "M" in text
        assert "Y" in text

    def test_equations_structural(self):
        m = pathmc.model(MEDIATION_SPEC)
        eqs = m.equations(show="structural")
        assert eqs is not None

    def test_equations_priors(self):
        m = pathmc.model(MEDIATION_SPEC)
        p = m.equations(show="priors")
        assert p is not None

    def test_priors(self):
        m = pathmc.model(MEDIATION_SPEC)
        p = m.priors()
        assert p is not None


class TestIdentificationWithoutData:
    """All identification methods work on data-free model."""

    def test_adjustment_sets(self):
        m = pathmc.model(FORK_SPEC)
        sets = m.adjustment_sets("X", "Y")
        assert {"Z"} in sets

    def test_is_identifiable(self):
        m = pathmc.model(FORK_SPEC)
        assert m.is_identifiable("X", "Y")

    def test_frontdoor_identifiable(self):
        m = pathmc.model(MEDIATION_SPEC)
        ok, msg = m.frontdoor_identifiable("X", "M", "Y")
        assert isinstance(ok, bool)
        assert isinstance(msg, str)

    def test_collider_warnings(self):
        m = pathmc.model(COLLIDER_SPEC)
        warnings = m.collider_warnings({"C"}, "X", "Y")
        assert isinstance(warnings, list)

    def test_implied_independences(self):
        m = pathmc.model(MEDIATION_SPEC)
        indeps = m.implied_independences()
        assert isinstance(indeps, list)


class TestDataRequiredMethods:
    """Data-requiring methods raise RuntimeError with helpful message."""

    @pytest.fixture()
    def data_free_model(self):
        return pathmc.model(MEDIATION_SPEC)

    @pytest.mark.parametrize(
        "method_name,call",
        [
            ("fit", lambda m: m.fit()),
            ("do", lambda m: m.do(set={"X": 1.0})),
            ("design", lambda m: m.design("Y")),
            ("test_implications", lambda m: m.test_implications()),
            ("predict", lambda m: m.predict()),
            ("sample_prior_predictive", lambda m: m.sample_prior_predictive()),
            ("summary", lambda m: m.summary()),
            ("effects_summary", lambda m: m.effects_summary()),
            ("standardized", lambda m: m.standardized()),
            ("effect", lambda m: m.effect("X -> M -> Y")),
            ("ate", lambda m: m.ate("Y", "X")),
            ("cate", lambda m: m.cate("Y", "X")),
            ("att", lambda m: m.att("Y", "X")),
            ("atu", lambda m: m.atu("Y", "X")),
            ("prob", lambda m: m.prob("Y > 0")),
            ("sensitivity", lambda m: m.sensitivity("Y", "X")),
            ("pymc_model", lambda m: m.pymc_model),
        ],
    )
    def test_raises_runtime_error(self, data_free_model, method_name, call):
        with pytest.raises(RuntimeError, match="requires data"):
            call(data_free_model)


class TestSetPriorsWithoutData:
    """set_priors() on data-free model updates priors without error."""

    def test_set_priors_updates(self):
        m = pathmc.model(MEDIATION_SPEC)
        m.set_priors({"beta_M": Prior("Normal", mu=0, sigma=10)})
        p = m.priors()
        text = str(p)
        assert "10" in text

    def test_set_priors_no_error(self):
        m = pathmc.model(MEDIATION_SPEC)
        m.set_priors({"sigma_Y": Prior("HalfNormal", sigma=5)})


class TestDataBoundRegression:
    """Creating a data-bound model with the same spec still works."""

    def test_data_bound_model_works(self):
        rng = np.random.default_rng(42)
        n = 50
        X = rng.normal(size=n)
        M = 0.5 * X + rng.normal(size=n) * 0.1
        Y = 0.3 * M + 0.2 * X + rng.normal(size=n) * 0.1
        df = pd.DataFrame({"X": X, "M": M, "Y": Y})

        m = pathmc.model(MEDIATION_SPEC, data=df)
        assert m.pymc_model is not None
        assert m.design("M") is not None
        assert m.graph() is not None
        assert m.equations() is not None

    def test_panel_requires_data(self):
        with pytest.raises(ValueError, match="panel= requires data"):
            pathmc.model(
                "Y ~ X",
                panel={"unit": "id", "time": "t"},
            )
