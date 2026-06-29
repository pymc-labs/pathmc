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
"""Turn a causal DAG into a pathmc model.

A directed acyclic graph (in DOT, an ``"A->B"`` edge list, or a
``networkx.DiGraph``) describes *which* variables cause which. Two routes
turn that structure into a fitted model:

- **digraph style** (:class:`BuildModelFromDAG` with ``style="digraph"``) —
  the graph is read literally as a *fully linear* Gaussian model: every edge
  ``A -> B`` is one slope into ``B``'s mean, every node gets an intercept and
  a likelihood, and observations are aligned to ``dims``/``coords`` via
  xarray. This is the direct, no-frills reading of the graph.

- **native style** (``style="native"``, or :func:`dag_to_spec` +
  :func:`pathmc.model`) — the graph is translated into pathmc's lavaan-style
  DSL and handed to :func:`pathmc.model`, unlocking everything pathmc offers
  on top of the bare structure: non-Gaussian families, transforms, latent
  mediators, custom priors, panel/pooling, and the full intervention API.

:func:`dag_to_spec` is the translator that both the native style and the
downstream model-averaging layer build on.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pymc_extras.prior import Prior

if TYPE_CHECKING:
    from pathmc._model import PathModel

__all__ = ["BuildModelFromDAG", "dag_to_spec"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _parse_dag(dag_str: str) -> nx.DiGraph:
    """Parse a DOT ``digraph`` or an ``"A->B"`` edge list into a DAG.

    Parameters
    ----------
    dag_str : str
        Either DOT (``digraph { A -> B; B -> C; }``) or a simple
        comma/newline-separated edge list (``"A->B, B->C"``).

    Returns
    -------
    networkx.DiGraph
        The parsed graph.

    Raises
    ------
    ValueError
        If the DOT braces are malformed, an edge token is invalid, or the
        result is not acyclic.
    """
    s = dag_str.strip()
    g = nx.DiGraph()

    if s.lower().startswith("digraph"):
        brace_start = s.find("{")
        brace_end = s.rfind("}")
        if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
            raise ValueError(
                "Malformed DOT digraph: missing braces. Expected "
                "'digraph { A -> B; ... }'."
            )
        # Reuse the shared DOT body parser so quoted ids (as emitted by
        # TBFPC.to_digraph / get_all_cdags_from_cpdag), chained edges, and
        # attribute brackets are all handled consistently.
        from pathmc.cpdag import _parse_dot_body

        nodes, directed, undirected = _parse_dot_body(
            s[brace_start + 1 : brace_end], is_undirected_graph=False
        )
        if undirected:
            pretty = ", ".join(
                f"{u}--{v}" for u, v in sorted(tuple(sorted(e)) for e in undirected)
            )
            raise ValueError(
                "BuildModelFromDAG needs a fully directed DAG, but found "
                f"unoriented edge(s): {pretty}. Orient them first — e.g. pick a "
                "DAG from TBFPC.get_all_cdags_from_cpdag()."
            )
        g.add_nodes_from(nodes)
        g.add_edges_from(directed)

    else:
        edges: list[tuple[str, str]] = []
        for token in re.split(r"[,\n]+", s):
            token = token.strip().rstrip(";")
            if not token:
                continue
            medge = re.match(r"^([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)$", token)
            if not medge:
                raise ValueError(
                    f"Invalid edge token: '{token}'. Use 'A->B' separated by "
                    "commas or newlines, or DOT 'digraph { ... }' syntax."
                )
            a, b = medge.group(1), medge.group(2)
            edges.append((a, b))
        g.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(g):
        raise ValueError(
            "Provided graph is not a DAG. Remove or reverse an edge to break the cycle."
        )
    return g


def _as_digraph(dag: str | nx.DiGraph) -> nx.DiGraph:
    """Coerce a DOT/edge-list string or a ``networkx.DiGraph`` into a DAG."""
    if isinstance(dag, nx.DiGraph):
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError(
                "Provided graph is not a DAG. Remove or reverse an edge to "
                "break the cycle."
            )
        return dag
    if isinstance(dag, str):
        return _parse_dag(dag)
    raise TypeError(
        "dag must be a DOT string, an 'A->B' edge list, or a networkx.DiGraph. "
        f"Got {type(dag).__name__}."
    )


def dag_to_spec(dag: str | nx.DiGraph, *, target: str | None = None) -> str:
    """Translate a causal DAG into a pathmc DSL specification string.

    Every node with at least one parent becomes a regression equation
    ``node ~ parent1 + parent2 + ...``; root (exogenous) nodes appear only on
    the right-hand side. The result is a plain pathmc spec that can be passed
    straight to :func:`pathmc.model`, optionally after editing in transforms,
    labels, or families.

    Parameters
    ----------
    dag : str | networkx.DiGraph
        The graph, as DOT (``"digraph { A -> B; }"``), an ``"A->B"`` edge
        list, or a ``networkx.DiGraph``.
    target : str | None
        If given, the node is validated to exist in the DAG. It does not
        change the emitted spec (pathmc derives estimands at query time), but
        guards against typos when wiring discovery output into a model.

    Returns
    -------
    str
        A pathmc DSL string, one equation per line, ordered topologically.

    Raises
    ------
    ValueError
        If the graph is not a DAG, has no edges, contains a node name that is
        not a valid pathmc identifier, or *target* is not a node.

    Examples
    --------
    ::

        from pathmc import dag_to_spec, model

        spec = dag_to_spec("digraph { X -> M; M -> Y; X -> Y; }")
        print(spec)
        # M ~ X
        # Y ~ M + X
        m = model(spec, data=df, families={"Y": "bernoulli"})
    """
    graph = _as_digraph(dag)

    if target is not None and target not in graph.nodes:
        raise ValueError(
            f"target '{target}' is not a node in the DAG. Nodes: {sorted(graph.nodes)}."
        )

    bad = [n for n in graph.nodes if not _IDENTIFIER_RE.match(str(n))]
    if bad:
        raise ValueError(
            "DAG node names must be valid pathmc identifiers (letters, digits, "
            f"underscore; not starting with a digit). Offending names: "
            f"{sorted(bad)}."
        )

    if graph.number_of_edges() == 0:
        raise ValueError(
            "DAG has no edges, so there is nothing to model. Provide a graph "
            "with at least one 'A -> B' edge."
        )

    equations: list[str] = []
    for node in nx.topological_sort(graph):
        parents = list(graph.predecessors(node))
        if not parents:
            continue
        parents_sorted = sorted(parents)
        equations.append(f"{node} ~ {' + '.join(parents_sorted)}")

    return "\n".join(equations)


class BuildModelFromDAG:
    """Build a probabilistic model directly from a causal DAG and a dataset.

    The DAG is read as a structural model: each node is a column in *df* and
    each edge ``A -> B`` contributes a slope from ``A`` into the mean of
    ``B``. Two build styles are available:

    - ``style="digraph"`` (default) — a **fully linear** Gaussian model built
      directly in PyMC. Every node gets an intercept, a slope per parent, and
      a likelihood; observed data is aligned to ``dims``/``coords`` via
      xarray. ``dims`` and ``coords`` are **required**.
    - ``style="native"`` — the DAG is translated to pathmc's DSL (see
      :func:`dag_to_spec`) and compiled with :func:`pathmc.model`, so the
      richer pathmc machinery (families, transforms, latent variables, custom
      priors) becomes available. ``dims``/``coords``/``model_config`` do not
      apply; pass ``families``/``priors``/``latent`` instead.

    Parameters
    ----------
    dag : str | networkx.DiGraph
        DAG in DOT format (``"digraph { A -> B; }"``), an ``"A->B"`` edge
        list, or a ``networkx.DiGraph``.
    df : pandas.DataFrame
        DataFrame containing a column for every DAG node (and, for the
        digraph style, every column named in *dims*).
    target : str
        Name of the target node; validated to exist in the DAG.
    dims : tuple[str, ...] | None
        (digraph style) Dims for the observed/likelihood variables, e.g.
        ``("date",)`` or ``("date", "country")``. Required when
        ``style="digraph"``.
    coords : dict | None
        (digraph style) Coordinate values for *dims* (and any prior dims).
        All keys must be columns of *df*. Required when ``style="digraph"``.
    model_config : dict | None
        (digraph style) Optional ``Prior`` objects for ``"intercept"``,
        ``"slope"`` and ``"likelihood"``; missing keys fall back to
        :pyattr:`default_model_config`.
    time_dim : str
        (digraph style) Name of the time dimension within *dims*. It is the
        one dim that slope/intercept priors are *not* broadcast over (slopes
        vary across the remaining, cross-sectional dims but are shared across
        time). Defaults to ``"date"``; set it to match your own time column
        (``"week"``, ``"t"``, ...) so slopes are not accidentally given a
        per-timestep dimension.
    style : {"digraph", "native"}
        Which builder to use (see above). Defaults to ``"digraph"``.
    families : dict[str, str] | None
        (native style) Per-variable distribution families forwarded to
        :func:`pathmc.model`.
    priors : dict[str, Any] | None
        (native style) Custom priors forwarded to :func:`pathmc.model`.
    latent : list[str] | None
        (native style) Latent variables forwarded to :func:`pathmc.model`.

    Examples
    --------
    Digraph style (fully linear)::

        import numpy as np, pandas as pd
        from pathmc import BuildModelFromDAG

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "X": np.random.normal(size=5),
            "Y": np.random.normal(size=5),
        })
        builder = BuildModelFromDAG(
            dag="X->Y", df=df, target="Y", dims=("date",), coords={"date": dates}
        )
        pymc_model = builder.build()

    Native style (full pathmc flexibility)::

        builder = BuildModelFromDAG(
            dag="X->Y", df=df, target="Y", style="native", families={"Y": "gaussian"}
        )
        path_model = builder.to_pathmodel()
        path_model.fit(draws=500, tune=500)
    """

    def __init__(
        self,
        *,
        dag: str | nx.DiGraph,
        df: pd.DataFrame,
        target: str,
        dims: tuple[str, ...] | None = None,
        coords: dict | None = None,
        model_config: dict | None = None,
        time_dim: str = "date",
        style: Literal["digraph", "native"] = "digraph",
        families: dict[str, str] | None = None,
        priors: dict[str, Any] | None = None,
        latent: list[str] | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}.")
        if not isinstance(target, str) or not target:
            raise ValueError(f"target must be a non-empty string, got {target!r}.")
        if not isinstance(time_dim, str) or not time_dim:
            raise ValueError(f"time_dim must be a non-empty string, got {time_dim!r}.")
        if style not in ("digraph", "native"):
            raise ValueError(
                f"Unknown style {style!r}. Choose 'digraph' (fully linear) or "
                "'native' (pathmc DSL)."
            )

        self.dag = dag
        self.df = df
        self.target = target
        self.dims = dims
        self.coords = coords
        self.time_dim = time_dim
        self.style = style
        self.families = families
        self.priors = priors
        self.latent = latent

        self.graph = _as_digraph(dag)
        self.nodes = list(nx.topological_sort(self.graph))
        if self.target not in self.nodes:
            raise ValueError(f"Target '{self.target}' not in DAG nodes: {self.nodes}")

        if style == "native":
            self._validate_native_args(model_config)
            self._path_model: PathModel | None = None
            return

        if families is not None or priors is not None or latent is not None:
            raise ValueError(
                "'families', 'priors' and 'latent' only apply to style='native' "
                "(did you mean style='native'?). "
                "For style='digraph', configure priors via 'model_config'."
            )

        if dims is None or coords is None:
            raise ValueError(
                "style='digraph' requires both 'dims' and 'coords'. Provide "
                "them, or use style='native' for a long-form pathmc model."
            )

        provided = model_config
        self.model_config = self.default_model_config
        if provided is not None:
            self.model_config.update(provided)

        self._validate_model_config_priors()
        self._validate_coords_required_are_consistent()
        self._warning_if_slope_dims_dont_match_likelihood_dims()
        self._validate_intercept_dims_match_slope_dims()

    def _validate_native_args(self, model_config: dict | None) -> None:
        """Reject digraph-only arguments supplied to the native style."""
        if model_config is not None:
            raise ValueError(
                "model_config only applies to style='digraph'. For "
                "style='native', pass 'priors' (a pathmc prior config) instead."
            )
        if self.dims is not None or self.coords is not None:
            warnings.warn(
                "'dims'/'coords' are ignored by style='native'; pathmc consumes "
                "the long-form DataFrame directly.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def default_model_config(self) -> dict[str, Prior]:
        """Default ``Prior`` objects for intercepts, slopes and likelihood.

        Returns
        -------
        dict
            Keys ``"intercept"``, ``"slope"`` and ``"likelihood"`` mapping to
            ``Prior`` instances whose dims derive from :pyattr:`dims`.
        """
        slope_dims = tuple(dim for dim in (self.dims or ()) if dim != self.time_dim)
        return {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=slope_dims),
            "slope": Prior("Normal", mu=0, sigma=2, dims=slope_dims),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2),
                dims=self.dims,
            ),
        }

    def _warning_if_slope_dims_dont_match_likelihood_dims(self) -> None:
        """Warn if slope dims differ from likelihood dims minus the time dim."""
        slope_prior = self.model_config["slope"]
        likelihood_prior = self.model_config["likelihood"]

        like_dims = getattr(likelihood_prior, "dims", None)
        if isinstance(like_dims, str):
            like_dims = (like_dims,)
        elif isinstance(like_dims, list):
            like_dims = tuple(like_dims)

        if like_dims is None:
            expected_slope_dims: tuple[str, ...] = ()
        else:
            expected_slope_dims = tuple(
                dim for dim in like_dims if dim != self.time_dim
            )

        slope_dims = getattr(slope_prior, "dims", None)
        if slope_dims is None or not isinstance(slope_dims, tuple):
            slope_dims = ()
        elif isinstance(slope_dims, str):
            slope_dims = (slope_dims,)
        elif isinstance(slope_dims, list):
            slope_dims = tuple(slope_dims)

        if slope_dims != expected_slope_dims:
            warnings.warn(
                "Slope prior dims "
                f"{slope_dims if slope_dims else '()'} do not match expected dims "
                f"{expected_slope_dims} (likelihood dims without the time dim "
                f"{self.time_dim!r}).",
                stacklevel=2,
            )

    def _validate_intercept_dims_match_slope_dims(self) -> None:
        """Ensure intercept dims match slope dims exactly."""

        def _to_tuple(maybe_dims):
            if maybe_dims is None:
                return ()
            if isinstance(maybe_dims, str):
                return (maybe_dims,)
            if isinstance(maybe_dims, list | tuple):
                return tuple(maybe_dims)
            return ()

        slope_dims = _to_tuple(getattr(self.model_config["slope"], "dims", None))
        intercept_dims = _to_tuple(
            getattr(self.model_config["intercept"], "dims", None)
        )

        if slope_dims != intercept_dims:
            raise ValueError(
                "model_config['intercept'].dims must match "
                "model_config['slope'].dims. "
                f"Got intercept dims {intercept_dims or '()'} and slope dims "
                f"{slope_dims or '()'}."
            )

    def _validate_model_config_priors(self) -> None:
        """Ensure required model_config entries are ``Prior`` instances."""
        required_keys = ("intercept", "slope", "likelihood")
        for key in required_keys:
            if key not in self.model_config:
                raise ValueError(f"model_config must include '{key}' as a Prior.")
        for key in required_keys:
            if not isinstance(self.model_config[key], Prior):
                raise TypeError(
                    f"model_config['{key}'] must be a Prior, got "
                    f"{type(self.model_config[key]).__name__}."
                )

    def _validate_coords_required_are_consistent(self) -> None:
        """Validate mutual consistency among dims, coords, priors, and columns."""
        if self.coords is None:
            raise ValueError("'coords' is required and cannot be None.")

        for key in self.coords.keys():
            if key not in self.df.columns:
                raise KeyError(
                    f"Coordinate key '{key}' not found in DataFrame columns. "
                    f"Present columns: {list(self.df.columns)}"
                )

        assert self.dims is not None
        for d in self.dims:
            if d not in self.coords:
                raise ValueError(f"Missing coordinate values for dim '{d}' in coords.")

        def _to_tuple(maybe_dims):
            if isinstance(maybe_dims, str):
                return (maybe_dims,)
            if isinstance(maybe_dims, list | tuple):
                return tuple(maybe_dims)
            return ()

        for prior_name, prior in self.model_config.items():
            if not isinstance(prior, Prior):
                continue
            for d in _to_tuple(getattr(prior, "dims", None)):
                if d not in self.coords:
                    raise ValueError(
                        f"Dim '{d}' declared in Prior '{prior_name}' must be "
                        "present in coords."
                    )

        likelihood_prior = self.model_config["likelihood"]
        likelihood_dims = _to_tuple(getattr(likelihood_prior, "dims", None))
        if likelihood_dims and tuple(self.dims) != likelihood_dims:
            raise ValueError(
                "Likelihood Prior dims "
                f"{likelihood_dims} must match class dims {tuple(self.dims)}. "
                "When supplying a custom model_config, ensure likelihood.dims "
                "equals the 'dims' argument."
            )

    def _parents(self, node: str) -> list[str]:
        """Return the parent node names of *node* in the DAG."""
        return list(self.graph.predecessors(node))

    def build(self) -> pm.Model:
        """Construct and return the PyMC model implied by the DAG and data.

        For ``style="digraph"`` this builds the fully linear Gaussian model
        directly. For ``style="native"`` it compiles via :func:`pathmc.model`
        and returns the underlying PyMC model (use :meth:`to_pathmodel` for
        the richer pathmc object).

        Returns
        -------
        pymc.Model
            The compiled PyMC model. For ``style="digraph"`` this is a fully
            linear Gaussian model with a slope per edge and a likelihood for
            every node. For ``style="native"`` it is the model produced by
            :func:`pathmc.model`, whose families and latent nodes determine the
            parametrization (latent nodes have no likelihood).
        """
        if self.style == "native":
            return self._build_native().pymc_model

        dims = self.dims
        coords = self.coords
        assert dims is not None and coords is not None

        with pm.Model(coords=coords) as model:
            # Index once outside the node loop; to_xarray() then yields a
            # Dataset whose every variable shares this index.
            indexed = self.df.set_index(list(dims))
            dataset = indexed.to_xarray()
            target_coords = {d: list(coords[d]) for d in dims}

            data_containers: dict[str, pm.Data] = {}
            for node in self.nodes:
                if node not in self.df.columns:
                    raise KeyError(f"Column '{node}' not found in df.")
                xarr = dataset[node]
                nan_before = int(xarr.isnull().sum())
                # to_xarray() sorts each index; realign to the user-declared
                # coord order so the data columns line up with the model's
                # coord labels (and the dim-indexed slope/intercept priors).
                xarr = xarr.reindex(target_coords)
                nan_after = int(xarr.isnull().sum())
                if nan_after > nan_before:
                    raise ValueError(
                        f"Reindexing '{node}' to the declared coords introduced "
                        f"{nan_after - nan_before} missing value(s): some coord "
                        f"label(s) in {list(dims)} are absent from the data. PyMC "
                        "cannot build a likelihood over NaN, so this would fail "
                        "downstream — check that every value in coords appears in "
                        "df (and vice versa)."
                    )
                values = xarr.values

                data_containers[node] = pm.Data(f"_{node}", values, dims=dims)

            slope_rvs: dict[tuple[str, str], pt.TensorVariable] = {}

            for node in self.nodes:
                parents = self._parents(node)
                mu_expr: Any = 0
                for parent in parents:
                    slope_name = f"{parent.lower()}:{node.lower()}"
                    slope_rv = self.model_config["slope"].create_variable(slope_name)
                    slope_rvs[(parent, node)] = slope_rv
                    mu_expr += slope_rv * data_containers[parent]
                intercept_rv = self.model_config["intercept"].create_variable(
                    f"{node.lower()}_intercept"
                )

                self.model_config["likelihood"].create_likelihood_variable(
                    name=node,
                    mu=mu_expr + intercept_rv,
                    observed=data_containers[node],
                )

            self.model = model
        return self.model

    def _build_native(self) -> PathModel:
        """Compile (once) and cache the native pathmc model."""
        if self._path_model is None:
            from pathmc._model import model as _pathmc_model

            spec = dag_to_spec(self.graph, target=self.target)
            self._path_model = _pathmc_model(
                spec,
                data=self.df,
                families=self.families,
                priors=self.priors,
                latent=self.latent,
            )
        return self._path_model

    def to_pathmodel(self) -> PathModel:
        """Return the native :class:`pathmc.PathModel` for this DAG.

        Only available for ``style="native"``. The returned model exposes the
        full pathmc API (``fit``, ``do``, ``ate``, ``falsify``, ...).

        Returns
        -------
        PathModel
            The compiled pathmc model.

        Raises
        ------
        RuntimeError
            If called on a ``style="digraph"`` builder.
        """
        if self.style != "native":
            raise RuntimeError(
                "to_pathmodel() is only available for style='native'. For a "
                "digraph model, use build() to get the PyMC model, or call "
                "dag_to_spec(dag) and pass the result to pathmc.model()."
            )
        return self._build_native()

    def model_graph(self):
        """Return a Graphviz visualization of the built model.

        Returns
        -------
        graphviz.Source | graphviz.Digraph
            Graphviz object representing the model graph.

        Raises
        ------
        RuntimeError
            If called before :meth:`build` (digraph style).
        """
        if self.style == "native":
            return self._build_native().to_graphviz()
        if not hasattr(self, "model"):
            raise RuntimeError("Call build() first.")
        return pm.model_to_graphviz(self.model)

    def dag_graph(self) -> nx.DiGraph:
        """Return a copy of the parsed DAG as a ``networkx.DiGraph``.

        Returns
        -------
        networkx.DiGraph
            A directed acyclic graph with the same nodes and edges as input.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes)
        g.add_edges_from(self.graph.edges)
        return g

    @staticmethod
    def _parse_dag(dag_str: str) -> nx.DiGraph:
        """Parse a DOT digraph or ``"A->B"`` edge list into a DAG."""
        return _parse_dag(dag_str)
