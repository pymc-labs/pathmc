"""M9 gate tests: end-to-end integration smoke tests.

These tests exercise the full pipeline (spec -> fit -> sample -> query)
with actual MCMC sampling. They use minimal draws for speed but verify
that all components work together.
"""

import pytest

import pathmc

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC


@pytest.mark.slow
class TestMediationEndToEnd:
    def test_fit_sample_summarize(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=100, tune=100, chains=1, random_seed=42)
        summary = model.summary()
        assert summary is not None
        assert len(summary) > 0

    def test_indirect_effect_in_summary(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=100, tune=100, chains=1, random_seed=42)
        effects = model.effects_summary()
        assert "indirect" in str(effects)

    def test_do_ate_positive_for_positive_dgp(self, mediation_data):
        """Data generated with positive total X->Y effect; ATE should be positive."""
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=200, tune=200, chains=2, random_seed=42)

        r0 = model.do(set={"X": 0.0}, kind="mean")
        r1 = model.do(set={"X": 1.0}, kind="mean")
        ate = (r1 - r0).mean("Y")

        # True total effect is c + a*b = 0.3 + 0.5*0.8 = 0.7
        assert ate > 0, f"ATE should be positive but got {ate}"

    def test_do_ate_magnitude_reasonable(self, mediation_data):
        """ATE should be in a reasonable range around the true value of 0.7."""
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=200, tune=200, chains=2, random_seed=42)

        r0 = model.do(set={"X": 0.0}, kind="mean")
        r1 = model.do(set={"X": 1.0}, kind="mean")
        ate = (r1 - r0).mean("Y")

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

    def test_graph_after_sample(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=50, tune=50, chains=1, random_seed=42)
        g = model.graph()
        assert g is not None

    def test_equations_after_sample(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        model.fit(draws=50, tune=50, chains=1, random_seed=42)
        eqs = model.equations()
        assert "M" in str(eqs)
        assert "Y" in str(eqs)
