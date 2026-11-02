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
"""Warning helpers shared by modules that must not import each other.

``compile`` imports ``transforms`` at module level. A walker defined in
either of those modules forces the other into a cycle or a second copy.
This module stays a leaf so both, and ``_ci_report``, can share one
definition.
"""

from __future__ import annotations


def _user_stacklevel() -> int:
    """Stacklevel to attribute a ``warnings.warn`` to the first non-pathmc frame.

    A fixed ``stacklevel`` constant cannot be correct for every call site
    because internal call-chain depth varies (e.g. warnings from
    ``_compile_scan_panel`` vs ``compile_to_pymc``). Walking frames is
    robust to refactors.
    """
    import sys

    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None:
        frame = frame.f_back
        level += 1
        module = frame.f_globals.get("__name__", "")
        if module != "pathmc" and not module.startswith("pathmc."):
            return level
    return level
