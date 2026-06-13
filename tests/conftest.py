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
import numpy as np
import pandas as pd
import pymc as pm
import pytest

try:  # pragma: no cover - depends on the installed PyMC test helpers
    from pymc.testing import mock_sample_setup_and_teardown

    _HAVE_PYMC_TESTING = True
except Exception:  # pragma: no cover
    mock_sample_setup_and_teardown = None
    _HAVE_PYMC_TESTING = False


# Single, fast sampler configuration applied to every ``pm.sample()`` call in
# the suite. The tests assert on coarse, wide-tolerance behavior rather than
# precise posteriors, so short single-chain runs are enough — and a single
# in-process chain avoids the per-chain process spawn/serialization overhead
# that dominates wall-time for these tiny models.
SAMPLE_KWARGS = {"draws": 50, "tune": 50, "chains": 1, "cores": 1}


@pytest.fixture(autouse=True)
def _fast_sampling(monkeypatch):
    """Force ``SAMPLE_KWARGS`` onto every ``pm.sample()`` call in the suite."""
    original_sample = pm.sample

    def _sample(*args, **kwargs):
        kwargs.update(SAMPLE_KWARGS)
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(pm, "sample", _sample)


if _HAVE_PYMC_TESTING:
    mock_pymc_sample = pytest.fixture(mock_sample_setup_and_teardown)
    mock_pymc_sample_class = pytest.fixture(
        mock_sample_setup_and_teardown,
        scope="class",
    )
    mock_pymc_sample_module = pytest.fixture(
        mock_sample_setup_and_teardown,
        scope="module",
    )
else:
    _PYMC_MOCK_SKIP_REASON = (
        "pymc.testing.mock_sample_setup_and_teardown is unavailable"
    )

    @pytest.fixture
    def mock_pymc_sample():
        pytest.skip(_PYMC_MOCK_SKIP_REASON)

    @pytest.fixture(scope="class")
    def mock_pymc_sample_class():
        pytest.skip(_PYMC_MOCK_SKIP_REASON)

    @pytest.fixture(scope="module")
    def mock_pymc_sample_module():
        pytest.skip(_PYMC_MOCK_SKIP_REASON)


# ---------------------------------------------------------------------------
# Spec strings — shared across test files
# ---------------------------------------------------------------------------

SIMPLE_REGRESSION = "Y ~ X1 + X2"

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

PARALLEL_MEDIATORS_SPEC = """\
M1 ~ a1*T
M2 ~ a2*T
Y  ~ b1*M1 + b2*M2 + c*T
M1 ~~ M2
indirect1 := a1*b1
indirect2 := a2*b2
total     := c + a1*b1 + a2*b2
"""

NO_INTERCEPT_SPEC = "Y ~ 0 + X1 + X2"

SEMICOLON_SPEC = "M ~ a*X; Y ~ b*M + c*X; indirect := a*b"

CYCLIC_SPEC = """\
X ~ Y
Y ~ X
"""

DUPLICATE_LHS_SPEC = """\
Y ~ X1
Y ~ X2
"""

# ---------------------------------------------------------------------------
# Data fixtures — deterministic via fixed seed
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def mediation_data(rng):
    """X -> M -> Y with direct effect X -> Y.

    True coefficients: a=0.5, b=0.8, c=0.3, indirect=0.4, total=0.7
    """
    n = 200
    X = rng.normal(size=n)
    M = 0.5 * X + rng.normal(scale=0.5, size=n)
    Y = 0.8 * M + 0.3 * X + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


@pytest.fixture
def fork_data(rng):
    """Z -> X and Z -> Y (common cause)."""
    n = 200
    Z = rng.normal(size=n)
    X = 0.7 * Z + rng.normal(scale=0.5, size=n)
    Y = 0.4 * X + 0.6 * Z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


@pytest.fixture
def simple_data(rng):
    """Y = 0.5*X1 + 0.3*X2 + noise."""
    n = 200
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    Y = 0.5 * X1 + 0.3 * X2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X1": X1, "X2": X2, "Y": Y})


@pytest.fixture
def parallel_mediators_data(rng):
    """T -> M1, T -> M2, M1 -> Y, M2 -> Y, T -> Y with correlated M1/M2 residuals."""
    n = 200
    T = rng.normal(size=n)
    eps = rng.multivariate_normal([0, 0], [[0.16, 0.08], [0.08, 0.16]], size=n)
    M1 = 0.6 * T + eps[:, 0]
    M2 = 0.4 * T + eps[:, 1]
    Y = 0.5 * M1 + 0.3 * M2 + 0.2 * T + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"T": T, "M1": M1, "M2": M2, "Y": Y})


# ---------------------------------------------------------------------------
# Fitted model fixtures (slow — only instantiated when requested)
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_mediation(mediation_data):
    """Mediation model fitted with minimal draws for testing."""
    import pathmc

    model = pathmc.model(MEDIATION_SPEC, data=mediation_data)
    model.fit(draws=100, tune=100, chains=1, random_seed=42)
    return model


@pytest.fixture
def fitted_parallel_mediators(parallel_mediators_data):
    """Parallel mediators model with ~~ fitted with minimal draws."""
    import pathmc

    model = pathmc.model(PARALLEL_MEDIATORS_SPEC, data=parallel_mediators_data)
    model.fit(draws=100, tune=100, chains=1, random_seed=42)
    return model
