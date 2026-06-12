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
    def test_effects_summary_structure_and_values(self, fitted_mediation):
        summary = fitted_mediation.effects_summary()
        assert isinstance(summary, pd.DataFrame)
        text = str(summary)
        for label in ["a", "b", "c"]:
            assert label in text, f"Label '{label}' missing from effects summary"
        assert "indirect" in text
        numeric_cols = summary.select_dtypes(include="number")
        assert numeric_cols.notna().all().all(), "Effects summary contains NaN values"


@pytest.mark.slow
class TestEffectPath:
    def test_effect_paths_return_results(self, fitted_mediation):
        result = fitted_mediation.effect("X -> Y")
        assert result is not None
        indirect = fitted_mediation.effect("X -> M -> Y")
        assert indirect is not None
