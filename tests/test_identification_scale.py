# Copyright 2025 - 2026 The PyMC Labs Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scale checks for backdoor identification.

Listing every minimal set is still exponential in the number of
ancestors. Checking one set, and deciding whether any set exists, must
not be: the path walk this replaced took seconds on a 24-node DAG and
minutes once ``adjustment_model`` listed sets before accepting an
explicit one.
"""

from __future__ import annotations

import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.graph import build_graph
from pathmc.identify import (
    adjustment_sets,
    is_identifiable,
    is_valid_adjustment_set,
)
from pathmc.parse import parse_spec


def _spec_from_dag(dag: nx.DiGraph) -> str:
    lines = []
    for node in dag.nodes:
        parents = sorted(dag.predecessors(node))
        if parents:
            lines.append(f"{node} ~ {' + '.join(parents)}")
    return "\n".join(lines)


def _random_dag(n: int, p: float, seed: int) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    nodes = [f"V{i}" for i in range(n - 2)] + ["T", "Y"]
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if rng.random() < p:
                dag.add_edge(source, target)
    if dag.degree("T") == 0:
        dag.add_edge("V0", "T")
    if dag.degree("Y") == 0:
        dag.add_edge("T", "Y")
    return dag


def _oracle_sets(dag: nx.DiGraph) -> list[set[str]]:
    """Minimal backdoor sets via d-separation, not pathmc."""
    descendants = nx.descendants(dag, "T")
    candidates = set(dag.nodes) - {"T", "Y"} - descendants
    mutilated = dag.copy()
    mutilated.remove_edges_from(list(dag.out_edges("T")))
    found: list[set[str]] = []
    ordered = sorted(candidates)
    for size in range(len(ordered) + 1):
        for subset in combinations(ordered, size):
            z = set(subset)
            if any(existing < z for existing in found):
                continue
            if nx.is_d_separator(mutilated, {"T"}, {"Y"}, z):
                found.append(z)
    found.sort(key=lambda s: (len(s), sorted(s)))
    return found


class TestAdjustmentSetScale:
    def test_non_ancestor_pads_a_valid_set_but_is_not_minimal(self):
        """Ancestor restriction applies only to the listed minimal sets."""
        g = build_graph(parse_spec("T ~ Z\nY ~ T + Z\nW ~ Z"))
        assert adjustment_sets(g, "T", "Y") == [{"Z"}]
        assert is_valid_adjustment_set(g, "T", "Y", {"Z", "W"})

    def test_listed_sets_match_dseparation_oracle(self):
        dag = _random_dag(12, 0.35, seed=486)
        g = build_graph(parse_spec(_spec_from_dag(dag)))
        assert adjustment_sets(g, "T", "Y") == _oracle_sets(dag)
        assert is_identifiable(g, "T", "Y") == bool(_oracle_sets(dag))

    def test_one_set_on_a_24_node_dag_is_not_a_path_walk(self):
        dag = _random_dag(24, 0.4, seed=24)
        g = build_graph(parse_spec(_spec_from_dag(dag)))
        start = time.perf_counter()
        identified = is_identifiable(g, "T", "Y")
        elapsed = time.perf_counter() - start
        mutilated = dag.copy()
        mutilated.remove_edges_from(list(dag.out_edges("T")))
        descendants = nx.descendants(dag, "T")
        allowed = set(dag.nodes) - {"T", "Y"} - descendants
        separator = nx.find_minimal_d_separator(
            mutilated, {"T"}, {"Y"}, restricted=allowed
        )
        assert identified == (separator is not None)
        # The path walk took about 2.5s on this shape. D-separation does not.
        assert elapsed < 1.0

    def test_explicit_set_skips_enumerating_a_20_node_dag(self, rng):
        """``adjustment_model(adjustment_set=)`` must not list every set first."""
        nodes = [f"Z{i}" for i in range(18)] + ["T", "Y"]
        lines = [f"T ~ {' + '.join(f'Z{i}' for i in range(18))}"]
        lines.append(f"Y ~ T + {' + '.join(f'Z{i}' for i in range(18))}")
        spec = "\n".join(lines)
        df = pd.DataFrame({name: rng.normal(size=30) for name in nodes})
        model = pathmc.model(spec, data=df)
        start = time.perf_counter()
        adjustment = {f"Z{i}" for i in range(18)}
        adjusted = model.adjustment_model("T -> Y", adjustment_set=adjustment)
        elapsed = time.perf_counter() - start
        assert adjusted.adjustment_set == frozenset(adjustment)
        assert "Z0" in adjusted.formula and "Z17" in adjusted.formula
        # Listing 2^18 subsets is minutes. Validating the given set is not.
        assert elapsed < 5.0

    def test_identifiability_matches_separator_existence(self):
        dag = _random_dag(16, 0.3, seed=16)
        g = build_graph(parse_spec(_spec_from_dag(dag)))
        mutilated = dag.copy()
        mutilated.remove_edges_from(list(dag.out_edges("T")))
        descendants = nx.descendants(dag, "T")
        allowed = set(dag.nodes) - {"T", "Y"} - descendants
        separator = nx.find_minimal_d_separator(
            mutilated, {"T"}, {"Y"}, restricted=allowed
        )
        assert is_identifiable(g, "T", "Y") == (separator is not None)


def test_unidentifiable_explicit_set_keeps_the_backdoor_error(rng):
    df = pd.DataFrame({name: rng.normal(size=40) for name in ("T", "Y", "Z")})
    model = pathmc.model("T ~ Z\nY ~ T\nT ~~ Y", data=df)
    with pytest.raises(ValueError, match="not identifiable"):
        model.adjustment_model("T -> Y", adjustment_set={"Z"})
