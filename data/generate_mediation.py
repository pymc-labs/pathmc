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
"""Generate the committed quickstart example dataset.

``data/mediation.csv`` backs the README quickstart. It is produced with
:func:`pathmc.simulate` so the data is guaranteed to be consistent with the
generative model compiled from the same spec — the DGP *is* the package's
own generative graph, with the ``params`` dict recording the true
coefficients. The CSV is committed so it can also be fetched from GitHub by
installed users (see ``pathmc.paths``).

Regenerate from the repository root with
``uv run python data/generate_mediation.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pathmc

readme_spec = """
M ~ a*X
Y ~ b*M + c*X
"""

# True parameter values, keyed by PyMC variable name. ``beta_{var}`` is the
# coefficient vector in term order (intercept first), ``sigma_{var}`` the
# residual scale.
truth = {
    "beta_M": [0.0, 0.5],  # M = 0.5*X + noise
    "sigma_M": 0.5,
    "beta_Y": [1.0, 0.8, 0.3],  # Y = 1.0 + 0.8*M + 0.3*X + noise
    "sigma_Y": 0.5,
}


seed = sum(map(ord, "README Quickstart"))
rng = np.random.default_rng(seed)

n = 500
exog = pd.DataFrame({"X": rng.normal(size=n)})

df = pathmc.simulate(readme_spec, data=exog, params=truth, random_seed=rng)
df.to_csv(Path(__file__).parent / "mediation.csv", index=False)
