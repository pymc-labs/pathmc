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
"""Public modeling entry points for API reference pages.

Implementation lives in ``pathmc._model``. This module exists so Great Docs /
Quarto emit ``modeling.PathModel.html`` rather than ``_model.PathModel.html``:
Quarto skips files whose names start with an underscore.
"""

from __future__ import annotations

from pathmc._model import PathModel, model, simulate

__all__ = ["PathModel", "model", "simulate"]
