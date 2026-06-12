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
"""Unit tests for shared validation helpers and the not-fitted guards."""

import narwhals.stable.v1 as nw
import networkx as nx
import pandas as pd
import pytest

import pathmc
from pathmc.identify import _require_nodes
from pathmc.panel import _require_column


class TestRequireNodes:
    def test_passes_when_all_present(self) -> None:
        dag = nx.DiGraph([("X", "Y")])
        _require_nodes(dag, treatment="X", outcome="Y")

    def test_raises_with_capitalized_role(self) -> None:
        dag = nx.DiGraph([("X", "Y")])
        with pytest.raises(ValueError, match="Treatment 'Z' not in DAG"):
            _require_nodes(dag, treatment="Z")

    def test_reports_available_nodes(self) -> None:
        dag = nx.DiGraph([("X", "Y")])
        with pytest.raises(ValueError, match=r"Available nodes: \['X', 'Y'\]"):
            _require_nodes(dag, mediator="Q")


class TestRequireColumn:
    def test_passes_when_present(self) -> None:
        df = nw.from_native(pd.DataFrame({"a": [1, 2]}))
        _require_column(df, "a", "Variable")

    def test_raises_keyerror_with_label(self) -> None:
        df = nw.from_native(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(KeyError, match="Variable 'b' not found in data"):
            _require_column(df, "b", "Variable")


class TestNotFittedGuard:
    """The _require_fitted helper backs every post-estimation method."""

    def _unfit_model(self):
        df = pd.DataFrame({
            "X": [0.0, 1.0, 0.0, 1.0],
            "Y": [0.1, 0.9, 0.2, 1.1],
        })
        return pathmc.model("Y ~ X", data=df)

    @pytest.mark.parametrize("method", ["summary", "effects_summary", "standardized"])
    def test_methods_raise_before_fit(self, method: str) -> None:
        model = self._unfit_model()
        with pytest.raises(RuntimeError, match=rf"Call .fit\(\) before .{method}\(\)"):
            getattr(model, method)()

    def test_do_raises_before_fit(self) -> None:
        model = self._unfit_model()
        with pytest.raises(RuntimeError, match=r"Call .fit\(\) before .do\(\)"):
            model.do(set={"X": 1.0})

    def test_data_free_model_reports_missing_data(self) -> None:
        model = pathmc.model("Y ~ X")
        with pytest.raises(RuntimeError, match="summary.. requires data"):
            model.summary()
