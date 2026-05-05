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
