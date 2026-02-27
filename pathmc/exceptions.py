"""Custom exceptions for pathmc."""

from __future__ import annotations


class ParseError(ValueError):
    """Raised when the DSL spec string cannot be parsed."""


class DuplicateEquationError(ParseError):
    """Raised when a variable appears as LHS in more than one regression."""


class CycleError(ValueError):
    """Raised when the structural model contains a directed cycle."""
