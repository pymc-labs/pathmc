"""M3–M4 gate tests: design matrices and PyMC model compilation.

TestDesignMatrix gates M3 (PathModel scaffold + design matrices).
TestGaussianCompilation and TestSimpleCompilation gate M4 (compiler).
"""

import pymc as pm

import pathmc

from conftest import (
    SIMPLE_REGRESSION,
    MEDIATION_SPEC,
    NO_INTERCEPT_SPEC,
)


class TestDesignMatrix:
    def test_design_columns_present(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        design_m = model.design("M")
        assert "X" in design_m.columns

    def test_intercept_included_by_default(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        design_m = model.design("M")
        col_names = set(str(c) for c in design_m.columns)
        assert "Intercept" in col_names or "1" in col_names

    def test_intercept_excluded_when_suppressed(self, simple_data):
        model = pathmc.fit(NO_INTERCEPT_SPEC, data=simple_data)
        design_y = model.design("Y")
        col_names = set(str(c) for c in design_y.columns)
        assert "Intercept" not in col_names

    def test_all_predictors_in_design(self, simple_data):
        model = pathmc.fit(SIMPLE_REGRESSION, data=simple_data)
        design_y = model.design("Y")
        assert "X1" in design_y.columns
        assert "X2" in design_y.columns


class TestGaussianCompilation:
    def test_compiles_to_pymc_model(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_free_rvs_reference_equations(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        rv_names = {rv.name for rv in model.pymc_model.free_RVs}
        assert any("M" in name for name in rv_names)
        assert any("Y" in name for name in rv_names)

    def test_observed_rvs_present(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        assert len(model.pymc_model.observed_RVs) > 0

    def test_multiple_equations_produce_multiple_observed(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        assert len(model.pymc_model.observed_RVs) >= 2


class TestSimpleCompilation:
    def test_simple_regression_compiles(self, simple_data):
        model = pathmc.fit(SIMPLE_REGRESSION, data=simple_data)
        assert isinstance(model.pymc_model, pm.Model)

    def test_no_intercept_compiles(self, simple_data):
        model = pathmc.fit(NO_INTERCEPT_SPEC, data=simple_data)
        assert isinstance(model.pymc_model, pm.Model)
