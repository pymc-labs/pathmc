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
"""Reproduce nutpie 0.16.10 rejection of a finite zero-gradient initial coordinate.

No pathmc or missing values are involved. With --offset 0.01, only the unused
prior coordinate moves; the statistical model and active coordinate stay fixed.
"""

import argparse
from importlib.metadata import version

import numpy as np
import nutpie
import pymc as pm


def main() -> None:
    """Compare an exact support point with a nearby valid initialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()
    with pm.Model() as model:
        x = pm.Normal("x", shape=2)
        pm.Normal("y", mu=x[0], sigma=1, observed=1.0)
    compiled = nutpie.compile_pymc_model(
        model,
        backend="numba",
        jitter_rvs=set(),
        initial_points={"x": np.array([0.0, args.offset])},
    )
    point = compiled.initial_point_func(85)
    logp, gradient = compiled.logp_func(point)
    assert np.isfinite(logp) and np.isfinite(gradient).all()
    print(
        f"PyMC {pm.__version__}; nutpie {version('nutpie')}; point {point}; logp {logp}; gradient {gradient}",
        flush=True,
    )
    nutpie.sample(
        compiled, draws=5, tune=20, chains=1, cores=1, progress_bar=False, seed=85
    )
    print("Sampling executed (not a convergence check)")


if __name__ == "__main__":
    main()
