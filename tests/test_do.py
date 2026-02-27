"""M6 gate tests: do() cross-sectional operator.

TestDoAPI contains fast tests that verify the do() interface exists.
TestDoSemantics contains slow tests that verify propagation with actual draws.
"""

import numpy as np
import pytest

import pathmc

from conftest import MEDIATION_SPEC


class TestDoAPI:
    """Fast tests: verify do() API surface before sampling."""

    def test_do_method_exists(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        assert hasattr(model, "do")
        assert callable(model.do)

    def test_do_before_sampling_raises(self, mediation_data):
        model = pathmc.fit(MEDIATION_SPEC, data=mediation_data)
        with pytest.raises(Exception):
            model.do(kind="mean")


@pytest.mark.slow
class TestDoSemantics:
    """Slow tests: verify do() propagation and arithmetic."""

    def test_do_baseline_returns_result(self, fitted_mediation):
        result = fitted_mediation.do(kind="mean")
        assert result is not None
        assert np.isfinite(result.mean("Y"))

    def test_do_set_returns_result(self, fitted_mediation):
        result = fitted_mediation.do(set={"X": 1.0}, kind="mean")
        assert np.isfinite(result.mean("Y"))
        assert np.isfinite(result.mean("M"))

    def test_do_contrast_arithmetic(self, fitted_mediation):
        baseline = fitted_mediation.do(kind="mean")
        scenario = fitted_mediation.do(set={"X": 1.0}, kind="mean")
        contrast = scenario - baseline
        assert np.isfinite(contrast.mean("Y"))

    def test_do_hdi_has_two_bounds(self, fitted_mediation):
        result = fitted_mediation.do(set={"X": 1.0}, kind="mean")
        hdi = result.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]

    def test_different_interventions_produce_different_means(self, fitted_mediation):
        r0 = fitted_mediation.do(set={"X": 0.0}, kind="mean")
        r2 = fitted_mediation.do(set={"X": 2.0}, kind="mean")
        assert r0.mean("Y") != r2.mean("Y")

    def test_do_contrast_hdi(self, fitted_mediation):
        baseline = fitted_mediation.do(set={"X": 0.0}, kind="mean")
        scenario = fitted_mediation.do(set={"X": 1.0}, kind="mean")
        contrast = scenario - baseline
        hdi = contrast.hdi("Y")
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]
