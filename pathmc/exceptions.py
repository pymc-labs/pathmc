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
"""Custom exceptions for pathmc."""

from __future__ import annotations


class ParseError(ValueError):
    """Raised when the DSL spec string cannot be parsed."""


class DuplicateEquationError(ParseError):
    """Raised when a variable appears as LHS in more than one regression."""


class CycleError(ValueError):
    """Raised when the structural model contains a directed cycle."""
