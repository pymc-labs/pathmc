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
"""PathModel: the primary user-facing object returned by model()."""

from __future__ import annotations

import sys
import warnings
from typing import Any, Literal

import arviz as az
import graphviz
import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from narwhals.stable.v1.typing import IntoFrame, IntoFrameT

from pathmc.compile import build_design_matrix, compile_to_pymc, get_predictor_columns
from pathmc.effects import (
    EffectResult,
    _has_labeled_terms,
    build_effects_summary,
    build_standardized_effects,
    compute_path_effect,
)
from pathmc.falsify import FalsificationResult, falsify_graph as _falsify_graph
from pathmc.graph import GraphInfo, build_graph
from pathmc.identify import (
    ConditionalIndependence,
    ImplicationTestResult,
    adjustment_sets as _adjustment_sets,
    collider_warnings as _collider_warnings,
    frontdoor_identifiable as _frontdoor_identifiable,
    implied_independences as _implied_independences,
    is_identifiable as _is_identifiable,
    test_implications as _test_implications,
)
from pathmc.introspect import (
    EquationList,
    ModelEquations,
    PriorTable,
    build_dag_viz,
    build_equations,
    build_priors,
)
from pathmc.panel import PanelInfo, build_panel_info
from pathmc.parse import Spec, parse_spec
from pathmc.sensitivity import SensitivityResult, compute_sensitivity
from pathmc.simulate import (
    DoResult,
    run_do_panel_unified,
    run_do_pymc,
)

__all__ = ["DoResult", "PathModel", "model", "simulate"]


class PathModel:
    """A compiled Bayesian path model.

    Created by :func:`pathmc.model`. Holds the parsed specification, graph
    structure, design matrices, and the compiled PyMC model.

    When *data* is ``None``, the model operates in **data-free mode**:
    introspection (``graph()``, ``equations()``, ``priors()``) and
    identification helpers work, but methods requiring data or a compiled
    PyMC model raise ``RuntimeError``.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    graph_info : GraphInfo
        DAG with topological order and node classification.
    data : nw.DataFrame | None
        Observed data used to build design matrices. ``None`` for
        data-free DAG exploration.
    families : dict[str, str] | None
        Per-variable distribution families.
    panel_info : PanelInfo | None
        Panel metadata (unit/time structure).
    pooling : str | dict | None
        Pooling specification for hierarchical panel models.
    latent : set[str] | None
        Endogenous variables with no observed data column.
    priors : dict | None
        Custom prior configuration mapping parameter names to ``Prior``
        objects from ``pymc_extras``. Overrides are merged with defaults.
    """

    def __init__(
        self,
        spec: Spec,
        graph_info: GraphInfo,
        data: nw.DataFrame | None,
        families: dict[str, str] | None = None,
        panel_info: PanelInfo | None = None,
        pooling: str | dict | None = None,
        latent: set[str] | None = None,
        priors: dict[str, Any] | None = None,
    ) -> None:
        from pathmc.priors import default_priors, merge_priors

        self._spec = spec
        self._graph_info = graph_info
        self._data = data
        self._panel_info = panel_info
        self._pooling = pooling
        self._latent: set[str] = latent if latent is not None else set()
        self._families: dict[str, str] = families if families is not None else {}

        defaults = default_priors(
            spec,
            families=self._families,
            pooling=pooling,
            latent=self._latent,
        )
        self._priors = merge_priors(defaults, priors)

        if data is None:
            self._design_matrices: dict[str, nw.DataFrame] = {}
            self._gen_model: pm.Model | None = None
            self._pymc_model: pm.Model | None = None
            self._idata: xr.DataTree | None = None
            return

        self._design_matrices = {}
        for reg in spec.regressions:
            missing: list[str] = []
            for t in reg.terms:
                if t.interaction_of is not None:
                    for v in t.interaction_of:
                        if v not in data.columns and v not in missing:
                            missing.append(v)
                elif t.variable not in data.columns:
                    missing.append(t.variable)
            if missing:
                cols = get_predictor_columns(reg)
                self._design_matrices[reg.lhs] = nw.from_dict(
                    {c: np.array([], dtype=float) for c in cols},
                    backend=data.implementation,
                )
            else:
                self._design_matrices[reg.lhs] = build_design_matrix(reg, data)

        self._compile()

    def _require_data(self, method_name: str) -> None:
        """Raise RuntimeError if the model has no data."""
        if self._data is None:
            raise RuntimeError(
                f"{method_name}() requires data. Create a data-bound model: "
                f"m = pathmc.model(spec, data=df)"
            )

    def _require_fitted(self, method_name: str) -> xr.DataTree:
        """Return the posterior DataTree, or raise if not fitted.

        Also verifies the model is data-bound (a fitted model always has
        data), so callers can drop the separate ``_require_data`` check.
        """
        self._require_data(method_name)
        if self._idata is None:
            raise RuntimeError(
                f"No posterior samples available. Call .fit() before .{method_name}()."
            )
        return self._idata

    def _compile(self) -> None:
        """Compile the generative PyMC model and attach observations."""
        assert self._data is not None
        self._gen_model = compile_to_pymc(
            self._spec,
            self._data,
            self._design_matrices,
            families=self._families,
            panel_info=self._panel_info,
            pooling=self._pooling,
            latent=self._latent,
            graph_info=self._graph_info,
            priors=self._priors,
        )

        block_vars = (
            set().union(*self._graph_info.residual_blocks)
            if self._graph_info.residual_blocks
            else set()
        )

        scan_info = getattr(self._gen_model, "_pathmc_panel_scan", None)

        observations: dict[str, Any] = {}
        for reg in self._spec.regressions:
            var = reg.lhs
            if var in block_vars:
                continue
            if var not in self._latent and var in self._data.columns:
                if np.isnan(np.asarray(self._data[var].to_numpy(), dtype=float)).any():
                    continue
                family = self._families.get(var, "gaussian")
                vals = self._data[var].to_numpy()
                if family in ("bernoulli", "poisson", "negbinomial"):
                    vals = vals.astype(int)

                if scan_info is not None:
                    vals = (
                        vals[scan_info.sort_idx]
                        .reshape(scan_info.n_units, scan_info.n_times)
                        .T
                    )
                observations[var] = vals

        if observations:
            self._pymc_model = pm.observe(self._gen_model, observations)
            if "_use_observed_carry" in self._pymc_model.named_vars:
                with self._pymc_model:
                    pm.set_data({"_use_observed_carry": np.array(1, dtype="int8")})
        else:
            self._pymc_model = self._gen_model
        self._idata = None

    @property
    def pymc_model(self) -> pm.Model:
        """The compiled PyMC model.

        Raises
        ------
        RuntimeError
            If the model was created without data.
        """
        self._require_data("pymc_model")
        assert self._pymc_model is not None  # guaranteed after _require_data
        return self._pymc_model

    def to_graphviz(self) -> graphviz.Digraph:
        """Render the underlying PyMC model as a graphviz plate diagram.

        Delegates to ``pm.Model.to_graphviz()`` on the compiled model,
        showing random variables, deterministics, observed data, and
        plate structure.

        Returns
        -------
        graphviz.Digraph
            Graphviz diagram of the PyMC model graph.

        Raises
        ------
        RuntimeError
            If the model was created without data.
        """
        self._require_data("to_graphviz")
        assert self._pymc_model is not None
        return self._pymc_model.to_graphviz()

    def design(self, var: str) -> IntoFrame:
        """Return the design matrix for an endogenous variable's equation.

        Parameters
        ----------
        var : str
            Name of the endogenous (LHS) variable.

        Returns
        -------
        IntoFrame
            Design matrix with named columns, backed by the same DataFrame
            library as the data passed to :func:`model`.

        Raises
        ------
        RuntimeError
            If the model was created without data.
        KeyError
            If *var* is not an endogenous variable in the model.
        """
        self._require_data("design")
        if var not in self._design_matrices:
            available = ", ".join(sorted(self._design_matrices))
            raise KeyError(
                f"No equation for '{var}'. Available endogenous variables: {available}"
            )
        return self._design_matrices[var].to_native()

    def graph(self) -> graphviz.Digraph:
        """Return a graphviz DAG of the structural model.

        Works before sampling. Exogenous nodes are drawn as boxes,
        endogenous nodes as ellipses. Latent nodes get dashed borders
        (bold for stochastic latent). Labeled coefficients appear on edges.
        """
        return build_dag_viz(self._spec, self._graph_info, families=self._families)

    def equations(
        self,
        show: Literal["all", "structural", "priors"] = "all",
    ) -> ModelEquations | EquationList | PriorTable:
        """Return model equations and/or prior specifications.

        Works before sampling. By default renders both structural
        equations and priors as combined LaTeX in Jupyter / Quarto.

        Parameters
        ----------
        show : {"all", "structural", "priors"}, default "all"
            What to display:

            - ``"all"`` — structural equations **and** priors (default).
            - ``"structural"`` — only the structural (regression) equations.
            - ``"priors"`` — only the prior distributions.

        Returns
        -------
        ModelEquations | EquationList | PriorTable
            ``ModelEquations`` when *show="all"*, ``EquationList`` when
            *show="structural"*, ``PriorTable`` when *show="priors"*.
        """
        if show == "structural":
            return self._build_equations()
        if show == "priors":
            return self._build_priors()
        if show == "all":
            return ModelEquations(self._build_equations(), self._build_priors())
        raise ValueError(
            f"Unknown show={show!r}. Choose from 'all', 'structural', or 'priors'."
        )

    def _build_equations(self) -> EquationList:
        """Build structural equations from the spec."""
        return build_equations(self._spec, latent=self._latent, families=self._families)

    def _build_priors(self) -> PriorTable:
        """Build prior table from the spec and config."""
        return build_priors(
            self._spec,
            families=self._families,
            pooling=self._pooling,
            latent=self._latent,
            prior_config=self._priors,
        )

    def priors(self) -> PriorTable:
        """Return a summary of prior distributions for all parameters.

        Works before sampling. The table reflects any custom priors
        set via ``set_priors()`` or the ``priors`` argument to
        :func:`pathmc.model`.

        .. tip::

            Use ``equations()`` for a unified view of both structural
            equations and priors.
        """
        return self._build_priors()

    def set_priors(self, overrides: dict[str, Any]) -> None:
        """Update prior distributions and recompile the model.

        Only the specified priors are changed; all others keep their
        current values. Any existing posterior samples are invalidated.

        On a data-free model, priors are updated without recompilation.

        Parameters
        ----------
        overrides : dict[str, Prior]
            Mapping from parameter name to ``Prior`` object. Use
            ``.equations()`` to see available parameter names.

        Raises
        ------
        ValueError
            If a key does not match any model parameter.
        """
        from pathmc.priors import merge_priors

        had_samples = self._idata is not None
        self._priors = merge_priors(self._priors, overrides)
        if self._data is not None:
            self._compile()
        if had_samples:
            warnings.warn(
                "Priors changed — previous posterior samples have been "
                "discarded. Call .fit() again.",
                stacklevel=2,
            )

    def sample_prior_predictive(self, **kwargs: Any) -> xr.DataTree:
        """Draw samples from the prior predictive distribution.

        Useful for checking whether default or custom priors generate
        plausible data ranges before running MCMC.

        Parameters
        ----------
        **kwargs
            Forwarded to ``pm.sample_prior_predictive()``.

        Returns
        -------
        xarray.DataTree
            Prior predictive samples.
        """
        self._require_data("sample_prior_predictive")
        assert self._gen_model is not None
        with self._gen_model:
            return pm.sample_prior_predictive(**kwargs)

    def summary(self) -> pd.DataFrame:
        """Return a posterior summary table.

        Returns
        -------
        pd.DataFrame
            ArviZ summary of all model parameters. Columns are
            full-precision floats; apply ``.round()`` downstream as
            needed.

        Raises
        ------
        RuntimeError
            If the model was created without data, or called before
            ``.fit()``.
        """
        idata = self._require_fitted("summary")
        return az.summary(idata, round_to="none")

    def effects_summary(self) -> pd.DataFrame:
        """Return a posterior summary of labeled coefficients and defined parameters.

        Returns
        -------
        pd.DataFrame
            Summary with mean, sd, and HDI for each labeled coefficient
            and ``:=`` defined parameter.

        Raises
        ------
        RuntimeError
            If the model was created without data, or called before
            ``.fit()``.
        """
        idata = self._require_fitted("effects_summary")
        if not _has_labeled_terms(self._spec) and not self._spec.defined_params:
            warnings.warn(
                "No labeled coefficients or defined parameters (:=) in the spec. "
                "effects_summary() only reports labeled terms. "
                "Use labels like 'Y ~ a*X' or add ':= ' definitions to see results here. "
                "For all coefficients, use .summary() instead.",
                UserWarning,
                stacklevel=2,
            )
        return build_effects_summary(self._spec, idata)

    def standardized(self) -> pd.DataFrame:
        """Return stdyx-standardized coefficients for labeled effects.

        Each coefficient is standardized as ``coef * sd(X) / sd(Y)``,
        giving the expected change in Y (in SD units) per SD change in X.

        Returns
        -------
        pd.DataFrame
            Summary with mean, sd, and HDI for each standardized coefficient.

        Raises
        ------
        RuntimeError
            If the model was created without data, or called before
            ``.fit()``.
        """
        idata = self._require_fitted("standardized")
        if not _has_labeled_terms(self._spec):
            warnings.warn(
                "No labeled coefficients in the spec. "
                "standardized() only reports labeled terms. "
                "Use labels like 'Y ~ a*X + b*Z' to get standardized effects. "
                "For raw coefficients, use .summary() instead.",
                UserWarning,
                stacklevel=2,
            )
        assert self._data is not None
        return build_standardized_effects(
            self._spec, idata, self._data, latent=self._latent
        )

    def effect(self, path: str) -> EffectResult:
        """Compute the effect along a causal path in the DAG.

        Parameters
        ----------
        path : str
            A path string like ``"X -> M -> Y"`` specifying the causal
            pathway. Each edge must correspond to a regression term.

        Returns
        -------
        EffectResult
            Posterior draws for the path-specific effect.

        Raises
        ------
        RuntimeError
            If the model was created without data, or called before
            ``.fit()``.
        ValueError
            If a node is not endogenous or an edge does not exist.
        """
        idata = self._require_fitted("effect")
        return compute_path_effect(path, self._spec, idata)

    def fit(self, **kwargs: Any) -> xr.DataTree:
        """Run MCMC sampling and store the resulting posterior DataTree.

        All keyword arguments are forwarded to ``pm.sample()``.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to ``pm.sample()``. Common
            options include ``draws``, ``tune``, ``chains``,
            ``random_seed``, ``target_accept``, and ``nuts_sampler``.

        Returns
        -------
        xarray.DataTree
            Posterior samples.
        """
        self._require_data("fit")
        assert self._pymc_model is not None
        if sys.platform == "darwin" and "mp_ctx" not in kwargs:
            kwargs.setdefault("mp_ctx", "forkserver")
        with self._pymc_model:
            self._idata = pm.sample(**kwargs)
            # Silent unless the caller explicitly asked for progress bars;
            # matches the pre-PyMC 6 behavior where the log-likelihood was
            # computed inside pm.sample() without its own bar.
            pm.compute_log_likelihood(
                self._idata, progressbar=kwargs.get("progressbar", False)
            )
        return self._idata

    def predict(self, **kwargs: Any) -> xr.DataTree:
        """Run posterior predictive sampling.

        Wraps ``pm.sample_posterior_predictive()`` and extends the
        stored DataTree with a ``posterior_predictive`` group.

        Parameters
        ----------
        **kwargs
            Passed directly to ``pm.sample_posterior_predictive()``.
            Pass ``extend_inferencedata=False`` to leave the stored
            DataTree untouched and get the standalone posterior
            predictive samples back instead.

        Returns
        -------
        xarray.DataTree
            The stored DataTree with a ``posterior_predictive``
            group added, or the standalone posterior predictive samples
            when ``extend_inferencedata=False`` is passed.

        Raises
        ------
        RuntimeError
            If the model was created without data, or called before
            ``.fit()``.
        """
        idata = self._require_fitted("predict")
        assert self._pymc_model is not None
        kwargs.setdefault("extend_inferencedata", True)
        with self._pymc_model:
            pp = pm.sample_posterior_predictive(idata, **kwargs)
        if not kwargs["extend_inferencedata"]:
            return pp
        return idata

    def adjustment_sets(
        self,
        treatment: str,
        outcome: str,
    ) -> list[set[str]]:
        """Find valid backdoor adjustment sets for the causal effect
        of *treatment* on *outcome*.

        .. note::

            This function reasons about the DAG structure declared in the
            model specification. It cannot detect omitted variables,
            missing edges, or other forms of misspecification. Use
            ``test_implications()`` to check whether the DAG's structural
            assumptions are consistent with observed data.

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        list[set[str]]
            All valid minimal adjustment sets, sorted by size.
        """
        return _adjustment_sets(self._graph_info, treatment, outcome)

    def is_identifiable(self, treatment: str, outcome: str) -> bool:
        """Check if the causal effect of *treatment* on *outcome* is
        identifiable via the backdoor criterion.

        .. note::

            This function reasons about the DAG structure declared in the
            model specification. It cannot detect omitted variables,
            missing edges, or other forms of misspecification. Use
            ``test_implications()`` to check whether the DAG's structural
            assumptions are consistent with observed data.

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        bool
            True if at least one valid adjustment set exists.
        """
        return _is_identifiable(self._graph_info, treatment, outcome)

    def frontdoor_identifiable(
        self,
        treatment: str,
        mediator: str,
        outcome: str,
    ) -> tuple[bool, str]:
        """Check whether the front-door criterion identifies the causal
        effect of *treatment* on *outcome* through *mediator*.

        .. note::

            This function reasons about the DAG structure declared in the
            model specification. It cannot detect omitted variables,
            missing edges, or other forms of misspecification. Use
            ``test_implications()`` to check whether the DAG's structural
            assumptions are consistent with observed data. If the spec
            includes adjustment variables that add edges absent from the
            true causal DAG, the check may report false negatives. Build a
            separate ``GraphInfo`` from the causal structure for an
            accurate check in that case.

        Parameters
        ----------
        treatment : str
            Treatment variable name.
        mediator : str
            Mediator variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        tuple[bool, str]
            ``(identifiable, message)`` where *message* explains the
            result or describes which condition fails.
        """
        return _frontdoor_identifiable(self._graph_info, treatment, mediator, outcome)

    def collider_warnings(
        self,
        adjustment_vars: set[str],
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """Check if any variable in the proposed adjustment set is a
        collider that could introduce bias.

        .. note::

            This function reasons about the DAG structure declared in the
            model specification. It cannot detect omitted variables,
            missing edges, or other forms of misspecification. Use
            ``test_implications()`` to check whether the DAG's structural
            assumptions are consistent with observed data.

        Parameters
        ----------
        adjustment_vars : set[str]
            Proposed adjustment set.
        treatment : str
            Treatment variable name.
        outcome : str
            Outcome variable name.

        Returns
        -------
        list[str]
            Warning strings for problematic variables.
        """
        return _collider_warnings(self._graph_info, adjustment_vars, treatment, outcome)

    def implied_independences(self) -> list[ConditionalIndependence]:
        """List all conditional independences implied by the DAG.

        For each pair of non-adjacent nodes, returns the independence
        statement with the conditioning set derived from the basis set
        method (Shipley, 2000). Works before sampling — only the graph
        structure is needed.

        Returns
        -------
        list[ConditionalIndependence]
            Implied independence statements, sorted alphabetically.
        """
        return _implied_independences(self._graph_info)

    def test_implications(
        self,
        alpha: float = 0.05,
    ) -> ImplicationTestResult:
        """Test all DAG-implied conditional independences against the data.

        For each implied independence X ⊥⊥ Y | Z, computes the partial
        correlation between X and Y controlling for Z and tests whether
        it is significantly different from zero.

        A significant result flags a *violation*: the data show an
        association that the DAG says should not exist, suggesting a
        missing edge or incorrect structure.

        Uses the observed data, not the posterior.

        Parameters
        ----------
        alpha : float
            Significance level for flagging violations (default 0.05).

        Returns
        -------
        ImplicationTestResult
            Test results with ``.violations``, ``.to_dataframe()``, and
            rich display in Jupyter via ``_repr_html_()``.
        """
        self._require_data("test_implications")
        assert self._data is not None
        indeps = self.implied_independences()
        return _test_implications(indeps, self._data, alpha=alpha)

    def _run_do(
        self,
        set: dict[str, float | np.ndarray] | None,
        kind: str,
        *,
        subgroup_indices: np.ndarray | None = None,
    ) -> DoResult:
        """Run a cross-sectional do() via graph surgery on instance state.

        Centralizes the ``run_do_pymc`` call so the instance fields
        (generative model, graph, posterior, data, families) are wired in
        one place. Callers must have passed ``_require_fitted`` first.
        """
        assert self._gen_model is not None
        assert self._data is not None
        assert self._idata is not None
        return run_do_pymc(
            gen_model=self._gen_model,
            graph_info=self._graph_info,
            idata=self._idata,
            data=self._data,
            set=set,
            kind=kind,
            families=self._families,
            subgroup_indices=subgroup_indices,
        )

    def _subgroup_effect(
        self,
        method_name: str,
        treatment: str,
        values: tuple[float, float],
        subgroup_value: float,
        value_param: str,
        kind: str,
    ) -> DoResult:
        """Shared implementation of ``att`` and ``atu``.

        Estimates ``E[Y(hi) - Y(lo) | treatment = subgroup_value]`` by
        empirical integration over the covariate distribution of the
        subgroup whose treatment equals ``subgroup_value``.
        """
        self._require_fitted(method_name)
        assert self._data is not None
        assert self._gen_model is not None
        if self._panel_info is not None:
            raise NotImplementedError(
                f"{method_name}() is not yet supported for panel models. "
                "Use do() with manual subgroup selection instead."
            )

        mask = np.isclose(
            np.asarray(self._data[treatment].to_numpy(), dtype=float),
            subgroup_value,
        )
        subgroup_idx = np.where(mask)[0]
        if len(subgroup_idx) == 0:
            raise ValueError(
                f"No observations with {treatment} ≈ {subgroup_value}. "
                f"Check the {value_param} parameter or data values."
            )

        lo, hi = values
        r_lo = self._run_do({treatment: lo}, kind, subgroup_indices=subgroup_idx)
        r_hi = self._run_do({treatment: hi}, kind, subgroup_indices=subgroup_idx)
        return r_hi - r_lo

    def falsify(
        self,
        n_permutations: int | None = None,
        significance_level: float = 0.05,
        significance_ci: float = 0.05,
        include_unconditional: bool = True,
        random_seed: int | None = None,
    ) -> FalsificationResult:
        """Falsify the whole DAG against the data via a permutation test.

        Grades the entire DAG at once, rather than one missing edge at a
        time as :meth:`test_implications` does. It counts how many
        Local Markov Conditions (implied parental conditional
        independences) the data violate, then compares that count to a
        baseline of randomly relabeled competitor graphs.

        The DAG is reported as *informative* (falsifiable) when few node
        permutations share its Markov equivalence class. An informative
        DAG that violates fewer conditions than the permuted baseline is
        the positive case: *not contradicted and testable*. A
        non-informative DAG is also *not rejected*, but that verdict is
        vacuous — interpret it as *not falsifiable* rather than as
        evidence for the graph. This ports dowhy's
        ``gcm.falsify_graph`` (Eulig et al., 2023) using the same
        partial-correlation conditional independence methodology as
        :meth:`test_implications`. Because that test is linear, purely
        nonlinear dependencies are not detected, so a "not rejected"
        verdict is only as strong as the linear-Gaussian assumption.

        Uses the observed data, not the posterior, and works before
        sampling. Models with residual covariances (``~~``) are not
        supported and raise ``ValueError`` — falsify the directed
        structure without the ``~~`` terms instead.

        Parameters
        ----------
        n_permutations : int | None
            Number of permuted DAGs in the baseline. Defaults to
            ``round(1 / significance_level)``. For small graphs (at most
            7 nodes), if it meets or exceeds the number of distinct node
            relabelings (``n!``), all are enumerated exactly; otherwise
            random relabelings are sampled.
        significance_level : float
            Significance level for the permutation-based verdict
            (default 0.05).
        significance_ci : float
            Significance level for each conditional independence test
            (default 0.05).
        include_unconditional : bool
            Whether to also test unconditional independences implied by
            root nodes (default ``True``).
        random_seed : int | None
            Seed for the permutation sampler, for reproducible results.

        Returns
        -------
        FalsificationResult
            Verdict (``.falsified``, ``.falsifiable``), permutation
            p-values, per-test local violations, and a ``.plot()`` helper.
        """
        self._require_data("falsify")
        assert self._data is not None
        return _falsify_graph(
            self._graph_info,
            self._data,
            n_permutations=n_permutations,
            significance_level=significance_level,
            significance_ci=significance_ci,
            include_unconditional=include_unconditional,
            random_seed=random_seed,
        )

    def do(
        self,
        set: dict[str, float | np.ndarray] | None = None,
        shift: dict[str, float] | None = None,
        kind: str = "mean",
        simulate_over: str | None = None,
    ) -> DoResult:
        """Simulate an intervention using the do-operator.

        Uses PyMC-native graph surgery: ``pm.do()`` on the generative model
        for interventions, then ``pm.sample_posterior_predictive()``
        (kind="predictive") or ``compute_deterministics`` (kind="mean")
        for propagation.

        Parameters
        ----------
        set : dict[str, float | np.ndarray] | None
            Variables to fix at specific values (hard intervention).
            For panel models with ``simulate_over="time"``, values can
            be arrays of shape ``(n_times,)`` for time-varying
            interventions (e.g., a temporary spend increase).
        shift : dict[str, float] | None
            Reserved for soft interventions (not yet implemented).
        kind : str
            ``"mean"`` for deterministic propagation via mu Deterministics,
            ``"predictive"`` to include residual noise.
        simulate_over : str | None
            ``"time"`` to activate time-forward panel simulation.
            Requires the model to have been fitted with ``panel=``.

        Returns
        -------
        DoResult
            Container with ``.mean(var)``, ``.hdi(var)``, and
            contrast arithmetic via ``__sub__``.

        Raises
        ------
        RuntimeError
            If called before ``.fit()``.
        ValueError
            If ``simulate_over="time"`` without panel.
        """
        idata = self._require_fitted("do")
        assert self._data is not None
        assert self._gen_model is not None

        if set:
            for var, val in set.items():
                if var not in self._data.columns:
                    continue
                col_min = self._data[var].min()
                col_max = self._data[var].max()
                if col_min is None or col_max is None:
                    continue
                lo = float(col_min)
                hi = float(col_max)
                if isinstance(val, np.ndarray):
                    val_lo, val_hi = float(val.min()), float(val.max())
                    out_of_range = val_lo < lo or val_hi > hi
                    val_desc = f"[{val_lo:.2f}, {val_hi:.2f}]"
                else:
                    out_of_range = val < lo or val > hi
                    val_desc = f"{val:.2f}"
                if out_of_range:
                    warnings.warn(
                        f"Intervention value {val_desc} for '{var}' is outside "
                        f"the observed data range [{lo:.2f}, {hi:.2f}]. "
                        f"Results are extrapolations and should be interpreted "
                        f"with caution.",
                        UserWarning,
                        stacklevel=2,
                    )

        if simulate_over == "time":
            if self._panel_info is None:
                raise ValueError(
                    "simulate_over='time' requires a panel model. "
                    "Pass panel={...} to model()."
                )

            scan_info = getattr(self._gen_model, "_pathmc_panel_scan", None)
            n_times = (
                scan_info.n_times
                if scan_info is not None
                else len(self._data[self._panel_info.time].unique())
            )

            if set:
                for var, val in set.items():
                    if isinstance(val, np.ndarray) and len(val) != n_times:
                        raise ValueError(
                            f"Intervention array for '{var}' has length "
                            f"{len(val)}, expected {n_times} (one per time "
                            f"step)."
                        )

            if scan_info is not None:
                return run_do_panel_unified(
                    gen_model=self._gen_model,
                    graph_info=self._graph_info,
                    idata=idata,
                    panel_info=self._panel_info,
                    scan_info=scan_info,
                    set=set,
                    kind=kind,
                    families=self._families,
                )
            # Non-scan panel models fall through to the cross-sectional path.

        return self._run_do(set, kind)

    def ate(
        self,
        outcome: str,
        treatment: str,
        values: tuple[float, float] = (0.0, 1.0),
        **do_kwargs: Any,
    ) -> DoResult:
        """Compute the average treatment effect of *treatment* on *outcome*.

        Shorthand for ``do(set={treatment: hi}) - do(set={treatment: lo})``.

        Parameters
        ----------
        outcome : str
            Outcome variable name (used only for documentation;
            the contrast is computed over all variables).
        treatment : str
            Treatment variable to intervene on.
        values : tuple[float, float]
            ``(lo, hi)`` intervention values. Default ``(0.0, 1.0)``.
        **do_kwargs
            Passed to ``do()`` (e.g. ``kind``, ``simulate_over``).

        Returns
        -------
        DoResult
            Contrast ``do(treatment=hi) - do(treatment=lo)``.
        """
        self._require_data("ate")
        lo, hi = values
        r_lo = self.do(set={treatment: lo}, **do_kwargs)
        r_hi = self.do(set={treatment: hi}, **do_kwargs)
        return r_hi - r_lo

    def cate(
        self,
        outcome: str,
        treatment: str,
        values: tuple[float, float] = (0.0, 1.0),
        condition: dict[str, float] | None = None,
        **do_kwargs: Any,
    ) -> DoResult:
        """Compute the conditional average treatment effect.

        Like ``ate()`` but with additional variables fixed in both
        scenarios, enabling effect modification analysis.

        Parameters
        ----------
        outcome : str
            Outcome variable name.
        treatment : str
            Treatment variable to intervene on.
        values : tuple[float, float]
            ``(lo, hi)`` intervention values.
        condition : dict[str, float] | None
            Variables to fix at specific values in both scenarios.
        **do_kwargs
            Passed to ``do()``.

        Returns
        -------
        DoResult
            Contrast with conditioning variables held fixed.
        """
        self._require_data("cate")
        if condition is None:
            condition = {}
        lo, hi = values
        set_lo: dict[str, float | np.ndarray] = {treatment: lo, **condition}
        set_hi: dict[str, float | np.ndarray] = {treatment: hi, **condition}
        r_lo = self.do(set=set_lo, **do_kwargs)
        r_hi = self.do(set=set_hi, **do_kwargs)
        return r_hi - r_lo

    def att(
        self,
        outcome: str,
        treatment: str,
        values: tuple[float, float] = (0.0, 1.0),
        treated_value: float = 1.0,
        kind: str = "mean",
    ) -> DoResult:
        """Compute the average treatment effect on the treated (ATT).

        Estimates ``E[Y(hi) - Y(lo) | T = treated_value]`` using
        subgroup-aware empirical integration over the covariate
        distribution of the treated subgroup.

        Unlike :meth:`ate`, which averages over the full covariate
        distribution, ``att()`` restricts to rows where the treatment
        variable equals *treated_value*. In linear models without
        interactions, ATT equals ATE; they diverge with effect
        modification or nonlinear link functions.

        Parameters
        ----------
        outcome : str
            Outcome variable name (used for documentation; the contrast
            is computed over all endogenous variables).
        treatment : str
            Treatment variable to intervene on.
        values : tuple[float, float]
            ``(lo, hi)`` intervention values. Default ``(0.0, 1.0)``.
        treated_value : float
            Value of the treatment variable identifying the treated
            subgroup (default ``1.0``).
        kind : str
            ``"mean"`` for deterministic propagation, ``"predictive"``
            to include residual noise.

        Returns
        -------
        DoResult
            Contrast ``do(treatment=hi) - do(treatment=lo)`` integrated
            over the treated subgroup's covariate distribution.

        Raises
        ------
        RuntimeError
            If called before ``.fit()``.
        NotImplementedError
            If the model is a panel model (not yet supported).
        ValueError
            If no observations match *treated_value*.

        Examples
        --------
        >>> model = pathmc.model("Y ~ T + X", data=df)
        >>> model.fit(draws=1000, tune=1000)
        >>> att = model.att("Y", "T")
        >>> att.mean("Y")  # E[Y(1) - Y(0) | T=1]
        """
        return self._subgroup_effect(
            "att",
            treatment,
            values,
            subgroup_value=treated_value,
            value_param="treated_value",
            kind=kind,
        )

    def atu(
        self,
        outcome: str,
        treatment: str,
        values: tuple[float, float] = (0.0, 1.0),
        untreated_value: float = 0.0,
        kind: str = "mean",
    ) -> DoResult:
        """Compute the average treatment effect on the untreated (ATU).

        Estimates ``E[Y(hi) - Y(lo) | T = untreated_value]`` using
        subgroup-aware empirical integration over the covariate
        distribution of the untreated subgroup.

        Unlike :meth:`ate`, which averages over the full covariate
        distribution, ``atu()`` restricts to rows where the treatment
        variable equals *untreated_value*. In linear models without
        interactions, ATU equals ATE; they diverge with effect
        modification or nonlinear link functions.

        Parameters
        ----------
        outcome : str
            Outcome variable name (used for documentation; the contrast
            is computed over all endogenous variables).
        treatment : str
            Treatment variable to intervene on.
        values : tuple[float, float]
            ``(lo, hi)`` intervention values. Default ``(0.0, 1.0)``.
        untreated_value : float
            Value of the treatment variable identifying the untreated
            subgroup (default ``0.0``).
        kind : str
            ``"mean"`` for deterministic propagation, ``"predictive"``
            to include residual noise.

        Returns
        -------
        DoResult
            Contrast ``do(treatment=hi) - do(treatment=lo)`` integrated
            over the untreated subgroup's covariate distribution.

        Raises
        ------
        RuntimeError
            If called before ``.fit()``.
        NotImplementedError
            If the model is a panel model (not yet supported).
        ValueError
            If no observations match *untreated_value*.

        Examples
        --------
        >>> model = pathmc.model("Y ~ T + X", data=df)
        >>> model.fit(draws=1000, tune=1000)
        >>> atu = model.atu("Y", "T")
        >>> atu.mean("Y")  # E[Y(1) - Y(0) | T=0]
        """
        return self._subgroup_effect(
            "atu",
            treatment,
            values,
            subgroup_value=untreated_value,
            value_param="untreated_value",
            kind=kind,
        )

    def sensitivity(
        self,
        outcome: str,
        treatment: str,
        gamma_range: tuple[float, float] = (0.0, 1.0),
        delta_range: tuple[float, float] = (0.0, 1.0),
        n_grid: int = 20,
        values: tuple[float, float] = (0.0, 1.0),
        **do_kwargs: Any,
    ) -> SensitivityResult:
        """Assess robustness of a causal effect to unmeasured confounding.

        Hypothesizes an unmeasured confounder U with effect γ on the
        treatment and δ on the outcome. The confounding bias is γ × δ,
        so the adjusted ATE at each grid point is::

            adjusted ATE = observed ATE − γ × δ

        The result includes a contour plot showing which (γ, δ)
        combinations would overturn the causal conclusion.

        Parameters
        ----------
        outcome : str
            Outcome variable name.
        treatment : str
            Treatment variable name.
        gamma_range : tuple[float, float]
            ``(min, max)`` range for γ (confounder → treatment effect).
        delta_range : tuple[float, float]
            ``(min, max)`` range for δ (confounder → outcome effect).
        n_grid : int
            Number of grid points per dimension (default 20).
        values : tuple[float, float]
            ``(lo, hi)`` intervention values for computing the ATE
            (default ``(0.0, 1.0)``).
        **do_kwargs
            Passed to ``ate()`` and thence to ``do()``
            (e.g. ``kind``, ``simulate_over``).

        Returns
        -------
        SensitivityResult
            Sensitivity analysis results with ``.plot()`` method.

        Raises
        ------
        RuntimeError
            If called before ``.fit()``.
        ValueError
            If the ranges are invalid or ``n_grid < 2``.
        """
        self._require_fitted("sensitivity")

        all_vars = self._graph_info.exogenous | self._graph_info.endogenous
        if treatment not in all_vars:
            raise ValueError(
                f"Treatment '{treatment}' not in model. "
                f"Available variables: {sorted(all_vars)}"
            )
        if outcome not in all_vars:
            raise ValueError(
                f"Outcome '{outcome}' not in model. "
                f"Available variables: {sorted(all_vars)}"
            )

        if gamma_range[0] >= gamma_range[1]:
            raise ValueError(
                f"gamma_range must be (min, max) with min < max, got {gamma_range}."
            )
        if delta_range[0] >= delta_range[1]:
            raise ValueError(
                f"delta_range must be (min, max) with min < max, got {delta_range}."
            )
        if n_grid < 2:
            raise ValueError(f"n_grid must be >= 2, got {n_grid}.")

        ate_result = self.ate(outcome, treatment, values=values, **do_kwargs)
        ate_draws = ate_result._values[outcome]

        return compute_sensitivity(
            observed_ate_draws=ate_draws,
            outcome=outcome,
            treatment=treatment,
            gamma_range=gamma_range,
            delta_range=delta_range,
            n_grid=n_grid,
        )

    def prob(
        self,
        expr: str,
        set: dict[str, float | np.ndarray] | None = None,
        kind: str = "predictive",
        **do_kwargs: Any,
    ) -> float:
        """Compute the probability of an expression under an intervention.

        Evaluates ``P(expr | do(set))`` using posterior predictive draws.

        Parameters
        ----------
        expr : str
            Boolean expression over variable names, e.g. ``"Y > 0"``.
        set : dict[str, float] | None
            Intervention values for the do-operator.
        kind : str
            Propagation kind (default ``"predictive"`` to include
            residual noise, which is needed for meaningful probabilities).
        **do_kwargs
            Passed to ``do()``.

        Returns
        -------
        float
            Estimated probability (fraction of draws satisfying *expr*).
        """
        self._require_data("prob")
        result = self.do(set=set, kind=kind, **do_kwargs)
        namespace: dict[str, Any] = {
            var: draws for var, draws in result._values.items()
        }
        namespace["np"] = np
        namespace["__builtins__"] = {}
        mask = eval(expr, namespace)  # noqa: S307
        return float(np.mean(mask))


def model(
    spec_string: str,
    data: IntoFrame | None = None,
    families: dict[str, str] | None = None,
    panel: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: list[str] | None = None,
    priors: dict[str, Any] | None = None,
    **kwargs: Any,
) -> PathModel:
    """Parse a specification and compile a Bayesian path model.

    When *data* is ``None``, returns a data-free model suitable for DAG
    exploration: ``graph()``, ``equations()``, ``priors()``, and all
    identification helpers work immediately. Methods that require data
    (``fit()``, ``do()``, ``design()``, etc.) raise ``RuntimeError``
    with an actionable message.

    Parameters
    ----------
    spec_string : str
        Model specification in the pathmc DSL.
    data : IntoFrame | None
        Observed data as a pandas or polars DataFrame. When ``None``, the
        model is created in data-free mode for DAG exploration and
        identification.
    families : dict[str, str] | None
        Per-variable distribution families (default ``"gaussian"``).
    panel : dict[str, str] | None
        Panel structure with ``"unit"`` and ``"time"`` keys mapping to
        column names. Activates panel mode.
    pooling : str | dict | None
        ``"partial"`` for random intercepts per unit. A dict like
        ``{"intercept": True, "slopes": ["var"]}`` enables random slopes.
        ``None`` (default) means complete pooling (cross-sectional).
    latent : list[str] | None
        Variables to treat as latent deterministic mediators. These must
        appear as LHS of a regression but need not have a data column.
        The model compiles without a likelihood for these variables.
    priors : dict[str, Prior] | None
        Custom prior specifications mapping parameter names to ``Prior``
        objects from ``pymc_extras``. Only the specified parameters are
        overridden; all others use sensible defaults. Call
        ``.equations()`` on the returned model to see all parameter names.

        Example::

            from pymc_extras.prior import Prior

            m = pathmc.model(
                spec,
                data,
                priors={"beta_Y": Prior("Normal", mu=0, sigma=2)},
            )

    **kwargs
        Reserved for future options.

    Returns
    -------
    PathModel
        Compiled model ready for inspection and fitting, or a data-free
        model for exploration when ``data`` is ``None``.

    Raises
    ------
    ValueError
        If a latent variable is not endogenous, or an observed endogenous
        variable is missing from data.
    """
    spec = parse_spec(spec_string)
    latent_set = set(latent) if latent is not None else set()
    graph_info = build_graph(spec, latent=latent_set)

    nw_data = nw.from_native(data, eager_only=True) if data is not None else None

    has_lag_terms = any(
        term.lag_of is not None for reg in spec.regressions for term in reg.terms
    )
    if has_lag_terms and panel is None:
        raise ValueError(
            "lag() terms require a panel model. Pass panel={'unit': ..., "
            "'time': ...} to model()."
        )

    if nw_data is not None:
        endogenous_lhs = {reg.lhs for reg in spec.regressions}
        for var in endogenous_lhs:
            if var not in latent_set and var not in nw_data.columns:
                raise ValueError(
                    f"Endogenous variable '{var}' not found in data columns. "
                    f"If '{var}' is an unobserved mediator, declare it via "
                    f"latent=['{var}']."
                )

    panel_info: PanelInfo | None = None
    if panel is not None:
        if nw_data is None:
            raise ValueError(
                "panel= requires data. Provide data= alongside panel=, "
                "or omit panel= for data-free DAG exploration."
            )
        panel_info = build_panel_info(nw_data, panel)

    return PathModel(
        spec=spec,
        graph_info=graph_info,
        data=nw_data,
        families=families,
        panel_info=panel_info,
        pooling=pooling,
        latent=latent_set,
        priors=priors,
    )


def simulate(
    spec_string: str,
    data: IntoFrameT,
    params: dict[str, Any],
    families: dict[str, str] | None = None,
    latent: list[str] | set[str] | None = None,
    random_seed: int | np.random.Generator | None = None,
) -> IntoFrameT:
    """Simulate data from a pathmc model with known parameter values.

    Builds a generative PyMC model from the specification, fixes all
    parameter random variables at the values in *params* using
    ``pm.do()``, and draws one simulated dataset.

    This is useful for:

    - **Simulate-and-recover** workflows: generate data from known
      parameters, build and fit the model, and verify that the posterior
      concentrates around the truth.
    - **Teaching**: create pedagogical datasets with exact causal
      structure matching the model DAG.
    - **Power analysis**: generate data under hypothesized effect
      sizes and check whether the model can detect them.

    Parameters
    ----------
    spec_string : str
        Model specification in the pathmc DSL. All regressions define
        the generative structure (e.g. ``"M ~ X\\nY ~ M + X"``).
    data : IntoFrame
        pandas or polars DataFrame containing the exogenous variables
        (predictors). Endogenous (outcome) columns need not be present —
        they will be simulated. If present, they are ignored.
    params : dict[str, Any]
        True parameter values keyed by PyMC variable name. Typical
        keys are ``"beta_{var}"`` (coefficient vector) and
        ``"sigma_{var}"`` (residual std). Use ``pathmc.model(...).equations()``
        on a dummy dataset to discover expected names and shapes.
    families : dict[str, str] | None
        Per-variable distribution families (default ``"gaussian"``).
        Supports the same families as :func:`model`: ``"gaussian"``,
        ``"bernoulli"``, ``"poisson"``, ``"negbinomial"``,
        ``"studentt"``, ``"latent_normal"``.
    latent : list[str] | set[str] | None
        Variables to treat as latent (unobserved). These are compiled
        without a likelihood and their simulated values are included
        in the output. Deterministic latent nodes have no ``sigma``
        parameter; stochastic latent nodes (``families={"M":
        "latent_normal"}``) do.
    random_seed : int | np.random.Generator | None
        Random seed for reproducibility.

    Returns
    -------
    IntoFrame
        Copy of *data* (same backend as the input) with simulated
        endogenous columns appended (including latent variables).

    Raises
    ------
    ValueError
        If required parameter values are missing from *params*.
    NotImplementedError
        If the spec contains residual covariances (``~~``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pathmc
    >>> rng = np.random.default_rng(42)
    >>> exog = pd.DataFrame({"X": rng.normal(size=100)})
    >>> df = pathmc.simulate(
    ...     "Y ~ X",
    ...     data=exog,
    ...     params={"beta_Y": [2.0, 0.5], "sigma_Y": 1.0},
    ...     random_seed=42,
    ... )
    >>> list(df.columns)
    ['X', 'Y']
    """
    spec = parse_spec(spec_string)
    latent_set = set(latent) if latent else set()

    if spec.residual_covs:
        raise NotImplementedError(
            "simulate() does not yet support residual covariances (~~). "
            "Use numpy-based simulation for models with correlated residuals."
        )

    graph_info = build_graph(spec, latent=latent_set)

    nw_data = nw.from_native(data, eager_only=True)

    endogenous_lhs = [reg.lhs for reg in spec.regressions]
    endo_set = set(endogenous_lhs)

    data_sim = nw_data
    zero_cols = [var for var in endogenous_lhs if var not in data_sim.columns]
    if zero_cols:
        data_sim = data_sim.with_columns([nw.lit(0.0).alias(var) for var in zero_cols])

    design_matrices: dict[str, nw.DataFrame] = {}
    for reg in spec.regressions:
        design_matrices[reg.lhs] = build_design_matrix(reg, data_sim)

    gen_model = compile_to_pymc(
        spec,
        data_sim,
        design_matrices,
        families=families,
        graph_info=graph_info,
        latent=latent_set,
    )

    all_rv_names = {rv.name for rv in gen_model.free_RVs}
    endo_rv_names = endo_set & all_rv_names
    param_rv_names = all_rv_names - endo_rv_names

    missing = param_rv_names - set(params.keys())
    if missing:
        raise ValueError(
            f"Missing parameter values for: {sorted(missing)}. "
            f"All model parameters must be provided. "
            f"Expected: {sorted(param_rv_names)}"
        )

    extra = set(params.keys()) - param_rv_names
    if extra:
        warnings.warn(
            f"Ignoring unknown parameter names: {sorted(extra)}. "
            f"Expected parameter names: {sorted(param_rv_names)}",
            UserWarning,
            stacklevel=2,
        )

    do_dict = {k: v for k, v in params.items() if k in param_rv_names}
    fixed_model = pm.do(gen_model, do_dict)

    endo_order = [v for v in graph_info.topological_order if v in endo_rv_names]

    det_names = {d.name for d in gen_model.deterministics}
    latent_det_vars = [
        v
        for v in graph_info.topological_order
        if v in latent_set and v not in endo_rv_names and f"mu_{v}" in det_names
    ]

    vars_to_draw = [fixed_model[var] for var in endo_order]
    latent_det_tensors = [fixed_model[f"mu_{v}"] for v in latent_det_vars]

    all_to_draw = vars_to_draw + latent_det_tensors
    drawn = pm.draw(all_to_draw, random_seed=random_seed)
    if not isinstance(drawn, list):
        drawn = [drawn]

    n_endo = len(endo_order)
    new_columns: dict[str, nw.Series] = {}
    for var, values in zip(endo_order, drawn[:n_endo]):
        new_columns[var] = nw.new_series(
            var, np.asarray(values), backend=nw_data.implementation
        )
    for var, values in zip(latent_det_vars, drawn[n_endo:]):
        new_columns[var] = nw.new_series(
            var, np.asarray(values), backend=nw_data.implementation
        )

    result = nw_data.with_columns(list(new_columns.values()))
    return result.to_native()
