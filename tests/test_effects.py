"""M8 gate tests: effects and defined parameter evaluation.

All tests are slow (require a fitted model with posterior draws).
"""

import pandas as pd
import pytest


@pytest.mark.slow
class TestEffectsSummary:
    def test_effects_summary_returns_dataframe(self, fitted_mediation):
        summary = fitted_mediation.effects_summary()
        assert isinstance(summary, pd.DataFrame)

    def test_effects_summary_includes_labels(self, fitted_mediation):
        summary = fitted_mediation.effects_summary()
        text = str(summary)
        for label in ["a", "b", "c"]:
            assert label in text, f"Label '{label}' missing from effects summary"

    def test_effects_summary_includes_defined_params(self, fitted_mediation):
        summary = fitted_mediation.effects_summary()
        text = str(summary)
        assert "indirect" in text

    def test_effects_summary_values_finite(self, fitted_mediation):
        summary = fitted_mediation.effects_summary()
        numeric_cols = summary.select_dtypes(include="number")
        assert numeric_cols.notna().all().all(), "Effects summary contains NaN values"


@pytest.mark.slow
class TestEffectPath:
    def test_effect_method_exists(self, fitted_mediation):
        assert hasattr(fitted_mediation, "effect")
        assert callable(fitted_mediation.effect)

    def test_effect_returns_result(self, fitted_mediation):
        result = fitted_mediation.effect("X -> Y")
        assert result is not None

    def test_indirect_effect_path(self, fitted_mediation):
        result = fitted_mediation.effect("X -> M -> Y")
        assert result is not None
