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
"""M9 gate tests: end-to-end integration smoke tests.

These tests exercise the full pipeline (spec -> fit -> sample -> query)
with actual MCMC sampling. They use minimal draws for speed but verify
that all components work together.
"""

import numpy as np
import pandas as pd
import pytest

import pathmc

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC


def _make_mediation_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.mark.slow
class TestMediationEndToEnd:
    @pytest.fixture(scope="class")
    def mediation_model(self):
        mediation_data = _make_mediation_data()
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=200, tune=200, chains=2, random_seed=42)
        return model

    def test_summary_effects_and_ate(self, mediation_model):
        """Data generated with positive total X->Y effect; ATE should be positive."""
        summary = mediation_model.summary()
        assert summary is not None
        assert len(summary) > 0
        effects = mediation_model.effects_summary()
        assert "indirect" in str(effects)
        r0 = mediation_model.do(set={"X": 0.0}, kind="mean")
        r1 = mediation_model.do(set={"X": 1.0}, kind="mean")
        ate = (r1 - r0).mean("Y")
        assert ate > 0, f"ATE should be positive but got {ate}"
        assert 0.2 < ate < 1.5, f"ATE={ate:.3f} outside plausible range for true=0.7"


@pytest.mark.slow
class TestCorrelatedResidualsEndToEnd:
    def test_parallel_mediators_pipeline(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        model.fit(draws=100, tune=100, chains=1, random_seed=42)

        summary = model.summary()
        assert summary is not None

        effects = model.effects_summary()
        for name in ["indirect1", "indirect2", "total"]:
            assert name in str(effects), f"'{name}' missing from effects summary"


@pytest.mark.slow
class TestIntrospectionAfterSampling:
    """Verify introspection methods still work after sampling."""

    @pytest.fixture(scope="class")
    def sampled_model(self):
        mediation_data = _make_mediation_data()
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=50, tune=50, chains=1, random_seed=42)
        return model

    def test_graph_after_sample(self, sampled_model):
        g = sampled_model.graph()
        assert g is not None

    def test_equations_after_sample(self, sampled_model):
        eqs = sampled_model.equations()
        assert "M" in str(eqs)
        assert "Y" in str(eqs)
