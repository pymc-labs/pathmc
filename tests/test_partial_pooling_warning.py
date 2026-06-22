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
"""Test that partial pooling with intercepts raises a clear warning."""

import warnings

import numpy as np
import pandas as pd
import pytest

import pathmc


@pytest.fixture
def panel_data():
    """Panel data: 5 units, 10 time periods, y ~ x."""
    rng = np.random.default_rng(42)
    frames = []
    for g in range(5):
        x = rng.normal(0, 1, 10)
        y = 5 + 2 * x + rng.normal(0, 1, 10)
        frames.append(pd.DataFrame({"x": x, "y": y, "week": np.arange(10), "geo": g}))
    return pd.concat(frames, ignore_index=True)


class TestPartialPoolingInterceptWarning:
    """Verify warning is raised for partial pooling with formula intercepts."""

    def test_warning_raised_with_intercept(self, panel_data):
        """Warning raised when pooling='partial' and formula has intercept."""
        with pytest.warns(
            UserWarning,
            match="PARTIAL POOLING WITH REDUNDANT INTERCEPT",
        ):
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )

    def test_warning_message_content(self, panel_data):
        """Warning message contains actionable solution."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            assert len(w) == 1
            msg = str(w[0].message)
            assert "y ~ 0 + x" in msg
            assert "mu_alpha" in msg
            assert "beta[Intercept]" in msg
            assert "NON-IDENTIFIABLE" in msg

    def test_no_warning_without_intercept(self, panel_data):
        """No warning when formula explicitly removes intercept."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pathmc.model(
                "y ~ 0 + x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )

    def test_no_warning_without_pooling(self, panel_data):
        """No warning when pooling is not enabled."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
            )

    def test_warning_with_dict_pooling_intercept_true(self, panel_data):
        """Warning raised when pooling dict explicitly enables intercept."""
        with pytest.warns(
            UserWarning,
            match="PARTIAL POOLING WITH REDUNDANT INTERCEPT",
        ):
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling={"intercept": True},
            )

    def test_no_warning_with_dict_pooling_intercept_false(self, panel_data):
        """No warning when pooling dict disables intercept."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pathmc.model(
                "y ~ x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling={"intercept": False},
            )

    def test_warning_mentions_all_equations_with_intercepts(self, panel_data):
        """Warning lists all equations that have intercepts."""
        panel_data["z"] = np.random.default_rng(43).normal(0, 1, len(panel_data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pathmc.model(
                "y ~ x; z ~ y",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            assert len(w) == 1
            msg = str(w[0].message)
            assert "'y'" in msg
            assert "'z'" in msg


class TestPartialPoolingInterceptWarningWithLag:
    """Verify warning is raised in scan-compiled panel models too."""

    def test_warning_raised_with_lag_and_intercept(self, panel_data):
        """Warning raised in scan path when formula has intercept and lag."""
        with pytest.warns(
            UserWarning,
            match="PARTIAL POOLING WITH REDUNDANT INTERCEPT",
        ):
            pathmc.model(
                "y ~ x + lag(y)",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )

    def test_no_warning_with_lag_no_intercept(self, panel_data):
        """No warning in scan path when intercept is removed."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pathmc.model(
                "y ~ 0 + x + lag(y)",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )

    def test_no_double_fire_on_lag_models(self, panel_data):
        """Warning fires exactly once on lag models (not twice)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pathmc.model(
                "y ~ x + lag(y)",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            # Should be exactly 1 warning, not 2
            assert len(w) == 1
            assert "PARTIAL POOLING WITH REDUNDANT INTERCEPT" in str(w[0].message)


class TestPartialPoolingWarningTransformHandling:
    """Verify warning correctly renders transforms in suggested formulas."""

    def test_warning_preserves_log_transform(self, panel_data):
        """Transform terms like log(x) are preserved in suggested formula."""
        panel_data["log_x"] = np.log(panel_data["x"] + 10)  # add constant for positive
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Can't actually test with real transforms since pathmc doesn't have log()
            # but we can test the reconstruction logic with labeled coefficients
            pathmc.model(
                "y ~ 2*x",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            msg = str(w[0].message)
            assert "y ~ 0 + 2*x" in msg or "y ~ 0 + 2.0*x" in msg

    def test_warning_preserves_interaction(self, panel_data):
        """Interaction terms are preserved in suggested formula."""
        panel_data["z"] = np.random.default_rng(44).normal(0, 1, len(panel_data))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pathmc.model(
                "y ~ x:z",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            msg = str(w[0].message)
            assert "y ~ 0 + x:z" in msg

    def test_warning_preserves_lag(self, panel_data):
        """Lag terms are preserved in suggested formula."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pathmc.model(
                "y ~ x + lag(y)",
                data=panel_data,
                panel={"unit": "geo", "time": "week"},
                pooling="partial",
            )
            msg = str(w[0].message)
            assert "y ~ 0 + x + lag(y)" in msg

    def test_no_intercept_lag_model_works(self, panel_data):
        """Confirm that '~ 0 + lag(...)' models compile without error."""
        # Verifies the fix from #337 works and our advice isn't broken
        model = pathmc.model(
            "y ~ 0 + x + lag(y)",
            data=panel_data,
            panel={"unit": "geo", "time": "week"},
            pooling="partial",
        )
        assert model.pymc_model is not None
