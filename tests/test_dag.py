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
"""Tests for :mod:`pathmc.dag` (``BuildModelFromDAG`` and ``dag_to_spec``)."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pymc_extras.prior import Prior

import pathmc
from pathmc import TBFPC, BuildModelFromDAG, dag_to_spec
from pathmc.simulate import EstimandResult


def _make_tbfpc(**kwargs) -> TBFPC:
    """Construct a TBFPC while suppressing the experimental UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TBFPC(**kwargs)


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=6, freq="D")


@pytest.fixture
def xy_df(rng, dates):
    """Single-dim ('date') frame with X and Y columns."""
    return pd.DataFrame({
        "date": dates,
        "X": rng.normal(size=len(dates)),
        "Y": rng.normal(size=len(dates)),
    })


@pytest.fixture
def multidim_df(rng, dates):
    """Two-dim ('date', 'country') long-form frame with X and Y columns."""
    countries = ["a", "b"]
    rows = []
    for d in dates:
        for c in countries:
            rows.append({"date": d, "country": c, "X": rng.normal(), "Y": rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture
def multidim_coords(dates):
    return {"date": dates, "country": ["a", "b"]}


@pytest.fixture
def chain_df(rng):
    """Wide frame for the X -> M -> Y chain (native / round-trip)."""
    n = 20
    return pd.DataFrame({
        "X": rng.normal(size=n),
        "M": rng.normal(size=n),
        "Y": rng.normal(size=n),
    })


# ===========================================================================
# dag_to_spec
# ===========================================================================


class TestDagToSpec:
    def test_dot_chain_and_direct_edge(self):
        spec = dag_to_spec("digraph { X -> M; M -> Y; X -> Y; }")
        assert spec == "M ~ X\nY ~ M + X"

    def test_edge_list(self):
        assert dag_to_spec("A->B, B->C") == "B ~ A\nC ~ B"

    def test_networkx_matches_dot(self):
        g = nx.DiGraph()
        g.add_edges_from([("X", "M"), ("M", "Y"), ("X", "Y")])
        assert dag_to_spec(g) == dag_to_spec("digraph { X -> M; M -> Y; X -> Y; }")

    def test_parents_are_sorted(self):
        # Add edges out of alphabetical order to confirm the RHS is sorted.
        g = nx.DiGraph()
        g.add_edges_from([("Z", "Y"), ("A", "Y")])
        assert dag_to_spec(g) == "Y ~ A + Z"

    def test_no_edges_dot_raises(self):
        with pytest.raises(ValueError, match="no edges"):
            dag_to_spec("digraph { A; }")

    def test_isolated_digraph_raises(self):
        g = nx.DiGraph()
        g.add_nodes_from(["A", "B"])
        with pytest.raises(ValueError, match="no edges"):
            dag_to_spec(g)

    def test_invalid_identifier_leading_digit(self):
        g = nx.DiGraph()
        g.add_edge("1bad", "B")
        with pytest.raises(ValueError, match="identifier"):
            dag_to_spec(g)

    def test_invalid_identifier_hyphen(self):
        g = nx.DiGraph()
        g.add_edge("a-b", "B")
        with pytest.raises(ValueError, match="identifier"):
            dag_to_spec(g)

    def test_target_not_a_node(self):
        with pytest.raises(ValueError, match="is not a node"):
            dag_to_spec("A->B", target="Z")

    def test_target_present_is_accepted(self):
        # A valid target does not change the emitted spec.
        assert dag_to_spec("A->B", target="B") == "B ~ A"

    def test_cyclic_raises_not_a_dag(self):
        with pytest.raises(ValueError, match="not a DAG"):
            dag_to_spec("A->B, B->A")

    def test_bad_dag_type(self):
        with pytest.raises(TypeError, match="DOT string"):
            dag_to_spec(123)

    def test_roundtrip_into_pathmc_model(self, chain_df):
        spec = dag_to_spec("digraph { X -> M; M -> Y; X -> Y; }")
        model = pathmc.model(spec, data=chain_df)
        assert isinstance(model, pathmc.PathModel)


# ===========================================================================
# BuildModelFromDAG._parse_dag (staticmethod)
# ===========================================================================


class TestParseDag:
    def test_dot_and_edge_list_same_edges(self):
        e_dot = set(BuildModelFromDAG._parse_dag("digraph { A -> B; B -> C; }").edges)
        e_list = set(BuildModelFromDAG._parse_dag("A->B, B->C").edges)
        assert e_dot == e_list == {("A", "B"), ("B", "C")}

    def test_cycle_raises(self):
        with pytest.raises(ValueError, match="not a DAG"):
            BuildModelFromDAG._parse_dag("A->B, B->A")

    def test_malformed_dot_missing_braces(self):
        with pytest.raises(ValueError, match="Malformed DOT digraph: missing braces"):
            BuildModelFromDAG._parse_dag("digraph A -> B")

    def test_invalid_simple_token(self):
        with pytest.raises(ValueError, match="Invalid edge token"):
            BuildModelFromDAG._parse_dag("A--B")

    def test_comments_and_standalone_nodes(self):
        g = BuildModelFromDAG._parse_dag(
            "digraph { A -> B; // line comment\n# hash comment\n C; }"
        )
        assert set(g.edges) == {("A", "B")}
        # The standalone declaration adds an isolated node.
        assert "C" in g.nodes


# ===========================================================================
# BuildModelFromDAG construction / validation (digraph style)
# ===========================================================================


class TestDigraphConstruction:
    def test_df_must_be_dataframe(self, dates):
        with pytest.raises(TypeError, match="pandas.DataFrame"):
            BuildModelFromDAG(
                dag="X->Y",
                df=[1, 2],
                target="Y",
                dims=("date",),
                coords={"date": dates},
            )

    def test_target_must_be_nonempty_string(self, xy_df, dates):
        with pytest.raises(ValueError, match="non-empty string"):
            BuildModelFromDAG(
                dag="X->Y", df=xy_df, target="", dims=("date",), coords={"date": dates}
            )

    def test_unknown_style(self, xy_df, dates):
        with pytest.raises(ValueError, match="Unknown style"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                style="bogus",
            )

    def test_target_not_in_dag(self, xy_df, dates):
        with pytest.raises(ValueError, match="Target 'Z' not in DAG nodes"):
            BuildModelFromDAG(
                dag="X->Y", df=xy_df, target="Z", dims=("date",), coords={"date": dates}
            )

    def test_coord_key_not_in_columns(self, xy_df, dates):
        with pytest.raises(KeyError, match="Coordinate key 'ghost' not found"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates, "ghost": [1]},
            )

    def test_dim_missing_from_coords(self, xy_df, dates):
        with pytest.raises(ValueError, match="Missing coordinate values for dim"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date", "country"),
                coords={"date": dates},
            )

    def test_prior_dim_not_in_coords(self, xy_df, dates):
        mc = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=("region",)),
            "slope": Prior("Normal", mu=0, sigma=2, dims=("region",)),
            "likelihood": Prior(
                "Normal", sigma=Prior("HalfNormal", sigma=2), dims=("date",)
            ),
        }
        with pytest.raises(ValueError, match="Dim '.*' declared in Prior"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                model_config=mc,
            )

    def test_likelihood_dims_must_match_class_dims(self, xy_df, dates):
        mc = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=()),
            "slope": Prior("Normal", mu=0, sigma=2, dims=()),
            "likelihood": Prior(
                "Normal", sigma=Prior("HalfNormal", sigma=2), dims=("country",)
            ),
        }
        df = xy_df.assign(country="a")
        with pytest.raises(
            ValueError, match=r"Likelihood Prior dims .* must match class dims"
        ):
            BuildModelFromDAG(
                dag="X->Y",
                df=df,
                target="Y",
                dims=("date",),
                coords={"date": dates, "country": ["a"]},
                model_config=mc,
            )

    def test_model_config_likelihood_non_prior(self, xy_df, dates):
        mc = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=()),
            "slope": Prior("Normal", mu=0, sigma=2, dims=()),
            "likelihood": None,
        }
        with pytest.raises(
            TypeError, match=r"model_config\['likelihood'\] must be a Prior"
        ):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                model_config=mc,
            )


# ===========================================================================
# default_model_config
# ===========================================================================


class TestDefaultModelConfig:
    def _builder(self, df, dims, coords):
        return BuildModelFromDAG(
            dag="X->Y", df=df, target="Y", dims=dims, coords=coords
        )

    def test_keys_are_all_priors(self, xy_df, dates):
        b = self._builder(xy_df, ("date",), {"date": dates})
        mc = b.default_model_config
        assert set(mc) == {"intercept", "slope", "likelihood"}
        assert all(isinstance(p, Prior) for p in mc.values())

    def test_single_dim_slope_is_scalar(self, xy_df, dates):
        b = self._builder(xy_df, ("date",), {"date": dates})
        mc = b.default_model_config
        assert mc["slope"].dims == ()
        assert mc["intercept"].dims == mc["slope"].dims
        assert mc["likelihood"].dims == ("date",)

    def test_multidim_slope_drops_date(self, multidim_df, multidim_coords):
        b = self._builder(multidim_df, ("date", "country"), multidim_coords)
        mc = b.default_model_config
        assert mc["slope"].dims == ("country",)
        assert mc["intercept"].dims == mc["slope"].dims
        assert mc["likelihood"].dims == ("date", "country")


# ===========================================================================
# slope-dims mismatch warning
# ===========================================================================


class TestSlopeDimsWarning:
    def test_mismatch_warns(self, multidim_df, multidim_coords):
        mc = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=()),
            "slope": Prior("Normal", mu=0, sigma=2, dims=()),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2),
                dims=("date", "country"),
            ),
        }
        with pytest.warns(UserWarning, match="Slope prior dims"):
            BuildModelFromDAG(
                dag="X->Y",
                df=multidim_df,
                target="Y",
                dims=("date", "country"),
                coords=multidim_coords,
                model_config=mc,
            )

    def test_matching_does_not_warn(self, multidim_df, multidim_coords):
        mc = {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=("country",)),
            "slope": Prior("Normal", mu=0, sigma=2, dims=("country",)),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2),
                dims=("date", "country"),
            ),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            BuildModelFromDAG(
                dag="X->Y",
                df=multidim_df,
                target="Y",
                dims=("date", "country"),
                coords=multidim_coords,
                model_config=mc,
            )
        assert not [w for w in caught if "Slope prior dims" in str(w.message)]


# ===========================================================================
# build() and graph accessors (digraph style)
# ===========================================================================


class TestDigraphBuild:
    def _builder(self, df, dates):
        return BuildModelFromDAG(
            dag="X->Y", df=df, target="Y", dims=("date",), coords={"date": dates}
        )

    def test_build_returns_pymc_model(self, xy_df, dates):
        model = self._builder(xy_df, dates).build()
        assert isinstance(model, pm.Model)

    def test_build_free_rv_names(self, xy_df, dates):
        model = self._builder(xy_df, dates).build()
        names = {v.name for v in model.free_RVs}
        assert {"x:y", "x_intercept", "y_intercept", "X_sigma", "Y_sigma"} <= names

    def test_build_missing_node_column(self, dates):
        df = pd.DataFrame({"date": dates, "X": np.zeros(len(dates))})  # no Y
        b = self._builder(df, dates)
        with pytest.raises(KeyError, match=r"Column '.*' not found in df"):
            b.build()

    def test_model_graph_before_build(self, xy_df, dates):
        b = self._builder(xy_df, dates)
        with pytest.raises(RuntimeError, match=r"Call build\(\) first"):
            b.model_graph()

    def test_model_graph_after_build(self, xy_df, dates):
        import graphviz

        b = self._builder(xy_df, dates)
        b.build()
        g = b.model_graph()
        assert isinstance(g, graphviz.Digraph | graphviz.Source)

    def test_dag_graph_edges(self, xy_df, dates):
        b = self._builder(xy_df, dates)
        g = b.dag_graph()
        assert isinstance(g, nx.DiGraph)
        assert set(g.edges) == {("X", "Y")}


# ===========================================================================
# New style-gating behaviours (digraph)
# ===========================================================================


class TestStyleGating:
    def test_digraph_without_dims(self, xy_df):
        with pytest.raises(ValueError, match="requires both 'dims' and 'coords'"):
            BuildModelFromDAG(dag="X->Y", df=xy_df, target="Y", coords={"date": []})

    def test_digraph_without_coords(self, xy_df):
        with pytest.raises(ValueError, match="requires both 'dims' and 'coords'"):
            BuildModelFromDAG(dag="X->Y", df=xy_df, target="Y", dims=("date",))

    def test_digraph_coords_none_explicit_is_valueerror(self, xy_df):
        # Passing coords=None explicitly now hits the dims/coords-required
        # ValueError on the public path (not a pydantic error).
        with pytest.raises(ValueError, match="requires both 'dims' and 'coords'"):
            BuildModelFromDAG(
                dag="X->Y", df=xy_df, target="Y", dims=("date",), coords=None
            )

    def test_digraph_with_families(self, xy_df, dates):
        with pytest.raises(ValueError, match="only apply to style='native'"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                families={"Y": "gaussian"},
            )

    def test_digraph_with_priors(self, xy_df, dates):
        with pytest.raises(ValueError, match="only apply to style='native'"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                priors={"Y": "something"},
            )

    def test_digraph_with_latent(self, xy_df, dates):
        with pytest.raises(ValueError, match="only apply to style='native'"):
            BuildModelFromDAG(
                dag="X->Y",
                df=xy_df,
                target="Y",
                dims=("date",),
                coords={"date": dates},
                latent=["L"],
            )


# ===========================================================================
# Native style
# ===========================================================================


class TestNativeStyle:
    def test_to_pathmodel_returns_pathmodel(self, chain_df):
        b = BuildModelFromDAG(dag="X->Y", df=chain_df, target="Y", style="native")
        model = b.to_pathmodel()
        assert isinstance(model, pathmc.PathModel)

    def test_build_returns_pymc_model(self, chain_df):
        b = BuildModelFromDAG(dag="X->Y", df=chain_df, target="Y", style="native")
        assert isinstance(b.build(), pm.Model)

    def test_to_pathmodel_is_cached(self, chain_df):
        b = BuildModelFromDAG(dag="X->Y", df=chain_df, target="Y", style="native")
        assert b.to_pathmodel() is b.to_pathmodel()

    def test_model_config_rejected(self, chain_df):
        with pytest.raises(
            ValueError, match="model_config only applies to style='digraph'"
        ):
            BuildModelFromDAG(
                dag="X->Y",
                df=chain_df,
                target="Y",
                style="native",
                model_config={"likelihood": Prior("Normal")},
            )

    def test_dims_coords_ignored_warns(self, chain_df, dates):
        with pytest.warns(UserWarning, match="ignored by style='native'"):
            BuildModelFromDAG(
                dag="X->Y",
                df=chain_df,
                target="Y",
                style="native",
                dims=("date",),
                coords={"date": dates},
            )

    def test_to_pathmodel_on_digraph_raises(self, xy_df, dates):
        b = BuildModelFromDAG(
            dag="X->Y", df=xy_df, target="Y", dims=("date",), coords={"date": dates}
        )
        with pytest.raises(RuntimeError, match="only available for style='native'"):
            b.to_pathmodel()

    def test_native_bernoulli_builds(self, rng):
        n = 20
        df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.integers(0, 2, size=n)})
        b = BuildModelFromDAG(
            dag="X->Y",
            df=df,
            target="Y",
            style="native",
            families={"Y": "bernoulli"},
        )
        assert isinstance(b.build(), pm.Model)

    def test_native_model_graph_returns_object(self, chain_df):
        b = BuildModelFromDAG(dag="X->Y", df=chain_df, target="Y", style="native")
        assert b.model_graph() is not None


# ===========================================================================
# Discovery -> dag handoff (the documented seam) and parser robustness
# ===========================================================================


class TestDiscoveryHandoff:
    """TBFPC emits quoted DOT; it must flow into dag_to_spec / native build."""

    def test_quoted_dot_dag_to_spec(self):
        assert dag_to_spec('digraph { "X" -> "Y"; }') == "Y ~ X"

    def test_enumerated_dag_builds_via_pathmc_model(self):
        rng = np.random.default_rng(7)
        n = 400
        x = rng.normal(size=n)
        m = 0.8 * x + rng.normal(size=n)
        y = 0.7 * m + rng.normal(size=n)
        df = pd.DataFrame({"X": x, "M": m, "Y": y})
        df = (df - df.mean()) / df.std()
        model = _make_tbfpc(target="Y", target_edge_rule="fullS")
        model.fit(df, drivers=["X", "M"])
        # A genuine DAG from discovery carries quoted node names.
        dag0 = model.get_all_cdags_from_cpdag()[0]
        assert '"' in dag0
        spec = dag_to_spec(dag0)
        assert spec
        built = pathmc.model(spec, data=df)
        assert isinstance(built, pathmc.PathModel)

    def test_quoted_dot_native_builds(self, chain_df):
        b = BuildModelFromDAG(
            dag='digraph { "X" -> "Y"; }', df=chain_df, target="Y", style="native"
        )
        assert isinstance(b.to_pathmodel(), pathmc.PathModel)


class TestParseDagRobustness:
    def test_phantom_arrow_in_label_ignored(self):
        g = BuildModelFromDAG._parse_dag('digraph { A -> B [label="C -> D"]; }')
        assert set(g.edges()) == {("A", "B")}
        assert set(g.nodes()) == {"A", "B"}
        assert dag_to_spec('digraph { A -> B [label="C -> D"]; }') == "B ~ A"

    def test_undirected_edge_rejected(self):
        with pytest.raises(ValueError, match="fully directed DAG"):
            dag_to_spec("digraph { A -> B [dir=none]; }")


class TestMultiDimBuildAlignment:
    """build() must align pivoted data to the (possibly unsorted) coord order."""

    def test_unsorted_coords_aligned_to_labels(self, dates):
        rows = []
        for di, d in enumerate(dates):
            for ci, c in enumerate(["a", "b"]):
                val = float(di * 10 + ci)
                rows.append({"date": d, "country": c, "X": val, "Y": val})
        # Scramble row order; alignment must come from coords, not row order.
        df = pd.DataFrame(rows).sample(frac=1, random_state=42)
        coords = {"date": dates, "country": ["b", "a"]}  # deliberately unsorted
        b = BuildModelFromDAG(
            dag="X->Y", df=df, target="Y", dims=("date", "country"), coords=coords
        )
        model = b.build()
        x = model["_X"].get_value()
        assert x.shape == (len(dates), 2)
        # Column 0 is country 'b' (ci=1), column 1 is country 'a' (ci=0).
        np.testing.assert_array_equal(
            x[:, 0], [float(di * 10 + 1) for di in range(len(dates))]
        )
        np.testing.assert_array_equal(
            x[:, 1], [float(di * 10 + 0) for di in range(len(dates))]
        )


@pytest.mark.slow
class TestNativeFitEndToEnd:
    def test_fit_and_ate(self):
        rng = np.random.default_rng(0)
        n = 60
        x = rng.normal(size=n)
        m = 0.8 * x + rng.normal(size=n)
        y = 0.6 * m + rng.normal(size=n)
        df = pd.DataFrame({"X": x, "M": m, "Y": y})
        df = (df - df.mean()) / df.std()
        b = BuildModelFromDAG(dag="X->M, M->Y, X->Y", df=df, target="Y", style="native")
        pmod = b.to_pathmodel()
        pmod.fit(draws=50, tune=50, chains=1, progressbar=False, random_seed=0)
        names = [str(i) for i in pmod.summary().index]
        assert any(n.startswith("beta_Y") for n in names)
        assert any(n.startswith("beta_M") for n in names)
        est = pmod.ate("X", "Y")
        assert isinstance(est, EstimandResult)
        assert np.isfinite(est.mean())
