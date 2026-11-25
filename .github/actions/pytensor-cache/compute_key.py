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
job level) so that the portable-target pins are part of the version hash:
the pins change compile flags and cache keys, so leaving them out would
let a cache publish entries that later runs cannot use.
"""

import importlib.metadata as md
import os
import sys

pkgs = ["pytensor", "numba", "llvmlite", "numpy", "scipy"]


def ver(pkg: str) -> str:
    """Return the installed version of ``pkg``, or ``missing``."""
    try:
        return md.version(pkg)
    except Exception:
        return "missing"


pflags = os.environ["PYTENSOR_FLAGS"]
ncpu = os.environ["NUMBA_CPU_NAME"]

with open(".pytensor-cache-key", "w") as f:
    f.write(f"python={sys.version_info[0]}.{sys.version_info[1]}\n")
    for pkg in pkgs:
        f.write(f"{pkg}={ver(pkg)}\n")
    f.write(f"pytensor_flags={pflags}\n")
    f.write(f"numba_cpu={ncpu}\n")
