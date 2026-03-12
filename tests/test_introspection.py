"""M5 gate tests: model introspection.

These methods should work BEFORE sampling — they describe model
structure, not posterior results. All tests are fast.
"""

import pymc as pm

import pathmc

from conftest import MEDIATION_SPEC, PARALLEL_MEDIATORS_SPEC


class TestGraph:
    def test_graph_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        g = model.graph()
        assert g is not None

    def test_graph_for_larger_model(self, parallel_mediators_data):
        model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
        g = model.graph()
        assert g is not None


class TestEquations:
    def test_equations_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        assert eqs is not None

    def test_equations_mentions_endogenous(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        text = str(eqs)
        assert "M" in text
        assert "Y" in text

    def test_equations_mentions_predictors(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        eqs = model.equations()
        text = str(eqs)
        assert "X" in text


class TestDesignIntrospection:
    def test_design_returns_columns(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        design = model.design("M")
        assert hasattr(design, "columns")

    def test_design_for_each_endogenous(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        for var in ["M", "Y"]:
            design = model.design(var)
            assert design is not None
            assert len(design.columns) > 0


class TestPriors:
    def test_priors_returns_object(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        priors = model.priors()
        assert priors is not None

    def test_priors_mentions_equations(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        priors = model.priors()
        text = str(priors)
        assert "M" in text or "Y" in text


class TestPyMCModelAccess:
    def test_pymc_model_accessible(self, mediation_data):
        model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
        assert isinstance(model.pymc_model, pm.Model)
