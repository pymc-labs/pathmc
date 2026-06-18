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
"""Tests for the supported public API surface."""

from __future__ import annotations

from importlib import import_module

import pathmc


def test_top_level_public_api_is_explicit() -> None:
    assert pathmc.__all__ == [
        "DoResult",
        "EffectResult",
        "FalsificationResult",
        "ImplicationTestResult",
        "ParamSpec",
        "PathModel",
        "PlaceboRefutationResult",
        "Prior",
        "SensitivityResult",
        "Transform",
        "__version__",
        "model",
        "register_transform",
        "simulate",
    ]


def test_public_submodule_exports_are_intentional() -> None:
    expected = {
        "pathmc.compile": [],
        "pathmc.effects": ["EffectResult"],
        "pathmc.exceptions": ["CycleError", "DuplicateEquationError", "ParseError"],
        "pathmc.falsify": ["FalsificationResult", "falsify_graph"],
        "pathmc.graph": [],
        "pathmc.identify": ["ImplicationTestResult"],
        "pathmc.idata": [],
        "pathmc.introspect": [],
        "pathmc.model": ["DoResult", "PathModel", "model", "simulate"],
        "pathmc.panel": ["PanelInfo"],
        "pathmc.parse": [],
        "pathmc.priors": [],
        "pathmc.refute": ["PlaceboRefutationResult", "refute_placebo"],
        "pathmc.residuals": [],
        "pathmc.sensitivity": ["SensitivityResult"],
        "pathmc.simulate": ["DoResult"],
        "pathmc.transforms": [
            "ParamSpec",
            "Transform",
            "register_transform",
        ],
    }

    for module_name, public_names in expected.items():
        module = import_module(module_name)
        assert module.__all__ == public_names


def test_top_level_public_symbols_import() -> None:
    for name in pathmc.__all__:
        assert getattr(pathmc, name) is not None
