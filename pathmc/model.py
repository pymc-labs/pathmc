"""PathModel: the primary user-facing object returned by fit()."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import arviz as az
import graphviz
import numpy as np
import pandas as pd
import pymc as pm

from pathmc.compile import build_design_matrix, compile_to_pymc, get_predictor_columns
from pathmc.effects import (
    EffectResult,
    _has_labeled_terms,
    build_effects_summary,
    build_standardized_effects,
    compute_path_effect,
)
from pathmc.graph import GraphInfo, build_graph
from pathmc.identify import (
    adjustment_sets as _adjustment_sets,
    collider_warnings as _collider_warnings,
    is_identifiable as _is_identifiable,
)
from pathmc.introspect import (
    EquationList,
    PriorTable,
    build_dag_viz,
    build_equations,
    build_priors,
)
from pathmc.panel import PanelInfo, build_panel_info
from pathmc.parse import Spec, parse_spec
from pathmc.simulate import (
    DoResult,
    run_do_panel_unified,
    run_do_pymc,
)


class PathModel:
    """A compiled Bayesian path model.

    Created by :func:`pathmc.fit`. Holds the parsed specification, graph
    structure, design matrices, and the compiled PyMC model.

    Parameters
    ----------
    spec : Spec
        Parsed model specification.
    graph_info : GraphInfo
        DAG with topological order and node classification.
    data : pd.DataFrame
        Observed data used to build design matrices.
    families : dict[str, str] | None
        Per-variable distribution families.
    panel_info : PanelInfo | None
        Panel metadata (unit/time structure).
    pooling : str | dict | None
        Pooling specification for hierarchical panel models.
    latent : set[str] | None
        Endogenous variables with no observed data column.
    """

    def __init__(
        self,
        spec: Spec,
        graph_info: GraphInfo,
        data: pd.DataFrame,
        families: dict[str, str] | None = None,
        panel_info: PanelInfo | None = None,
        pooling: str | dict | None = None,
        latent: set[str] | None = None,
    ) -> None:
        self._spec = spec
        self._graph_info = graph_info
        self._data = data
        self._panel_info = panel_info
        self._pooling = pooling
        self._latent: set[str] = latent if latent is not None else set()

        self._design_matrices: dict[str, pd.DataFrame] = {}
        for reg in spec.regressions:
            missing = [t.variable for t in reg.terms if t.variable not in data.columns]
            if missing:
                cols = get_predictor_columns(reg)
                self._design_matrices[reg.lhs] = pd.DataFrame(
                    columns=cols,
                )
            else:
                self._design_matrices[reg.lhs] = build_design_matrix(reg, data)

        self._families: dict[str, str] = families if families is not None else {}
        self._gen_model: pm.Model = compile_to_pymc(
            spec,
            data,
            self._design_matrices,
            families=families,
            panel_info=panel_info,
            pooling=pooling,
            latent=self._latent,
            graph_info=graph_info,
        )

        block_vars = (
            set().union(*graph_info.residual_blocks)
            if graph_info.residual_blocks
            else set()
        )

        scan_info = getattr(self._gen_model, "_pathmc_panel_scan", None)

        observations: dict[str, Any] = {}
        for reg in spec.regressions:
            var = reg.lhs
            if var in block_vars:
                continue
            if var not in self._latent and var in data.columns:
                family = self._families.get(var, "gaussian")
                vals = data[var].values
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
            self._pymc_model: pm.Model = pm.observe(self._gen_model, observations)
        else:
            self._pymc_model: pm.Model = self._gen_model
        self._idata: az.InferenceData | None = None

    @property
    def pymc_model(self) -> pm.Model:
        """The compiled PyMC model."""
        return self._pymc_model

    def design(self, var: str) -> pd.DataFrame:
        """Return the design matrix for an endogenous variable's equation.

        Parameters
        ----------
        var : str
            Name of the endogenous (LHS) variable.

        Returns
        -------
        pd.DataFrame
            Design matrix with named columns.

        Raises
        ------
        KeyError
            If *var* is not an endogenous variable in the model.
        """
        if var not in self._design_matrices:
            available = ", ".join(sorted(self._design_matrices))
            raise KeyError(
                f"No equation for '{var}'. Available endogenous variables: {available}"
            )
        return self._design_matrices[var]

    def graph(self) -> graphviz.Digraph:
        """Return a graphviz DAG of the structural model.

        Works before sampling. Exogenous nodes are drawn as boxes,
        endogenous nodes as ellipses. Latent nodes get dashed borders.
        Labeled coefficients appear on edges.
        """
        return build_dag_viz(self._spec, self._graph_info)

    def equations(self) -> EquationList:
        """Return human-readable structural equations.

        Works before sampling.
        """
        return build_equations(self._spec, latent=self._latent)

    def priors(self) -> PriorTable:
        """Return a summary of prior distributions for all parameters.

        Works before sampling.
        """
        return build_priors(
            self._spec,
            families=self._families,
            pooling=self._pooling,
            latent=self._latent,
        )

    def summary(self) -> pd.DataFrame:
        """Return a posterior summary table.

        Returns
        -------
        pd.DataFrame
            ArviZ summary of all model parameters.

        Raises
        ------
        RuntimeError
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .summary()."
            )
        return az.summary(self._idata)

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
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. "
                "Call .sample() before .effects_summary()."
            )
        if not _has_labeled_terms(self._spec) and not self._spec.defined_params:
            warnings.warn(
                "No labeled coefficients or defined parameters (:=) in the spec. "
                "effects_summary() only reports labeled terms. "
                "Use labels like 'Y ~ a*X' or add ':= ' definitions to see results here. "
                "For all coefficients, use .summary() instead.",
                UserWarning,
                stacklevel=2,
            )
        return build_effects_summary(self._spec, self._idata)

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
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .standardized()."
            )
        if not _has_labeled_terms(self._spec):
            warnings.warn(
                "No labeled coefficients in the spec. "
                "standardized() only reports labeled terms. "
                "Use labels like 'Y ~ a*X + b*Z' to get standardized effects. "
                "For raw coefficients, use .summary() instead.",
                UserWarning,
                stacklevel=2,
            )
        return build_standardized_effects(
            self._spec, self._idata, self._data, latent=self._latent
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
            If called before ``.sample()``.
        ValueError
            If a node is not endogenous or an edge does not exist.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .effect()."
            )
        return compute_path_effect(path, self._spec, self._idata)

    def sample(self, **kwargs: Any) -> az.InferenceData:
        """Run MCMC sampling and store the resulting InferenceData.

        Parameters
        ----------
        **kwargs
            Passed directly to ``pm.sample()`` (e.g. ``draws``, ``tune``,
            ``chains``, ``random_seed``).

        Returns
        -------
        az.InferenceData
            Posterior samples.
        """
        if sys.platform == "darwin" and "mp_ctx" not in kwargs:
            kwargs.setdefault("mp_ctx", "forkserver")
        with self._pymc_model:
            self._idata = pm.sample(**kwargs)
        return self._idata

    def predict(self, **kwargs: Any) -> az.InferenceData:
        """Run posterior predictive sampling.

        Wraps ``pm.sample_posterior_predictive()`` and extends the
        stored InferenceData with a ``posterior_predictive`` group.

        Parameters
        ----------
        **kwargs
            Passed directly to ``pm.sample_posterior_predictive()``.

        Returns
        -------
        az.InferenceData
            InferenceData with ``posterior_predictive`` group added.

        Raises
        ------
        RuntimeError
            If called before ``.sample()``.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .predict()."
            )
        with self._pymc_model:
            pp = pm.sample_posterior_predictive(self._idata, **kwargs)
        self._idata.extend(pp)
        return self._idata

    def adjustment_sets(
        self,
        treatment: str,
        outcome: str,
    ) -> list[set[str]]:
        """Find valid backdoor adjustment sets for the causal effect
        of *treatment* on *outcome*.

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

    def collider_warnings(
        self,
        adjustment_vars: set[str],
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """Check if any variable in the proposed adjustment set is a
        collider that could introduce bias.

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

    def do(
        self,
        set: dict[str, float | np.ndarray] | None = None,
        shift: dict[str, float] | None = None,
        kind: str = "mean",
        simulate_over: str | None = None,
        panel_engine: str = "numpy",
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
        panel_engine : str
            Deprecated — ignored. Panel do() now uses the scan-compiled
            generative model for all temporal propagation. Passing any
            value other than the default emits a ``DeprecationWarning``.

        Returns
        -------
        DoResult
            Container with ``.mean(var)``, ``.hdi(var)``, and
            contrast arithmetic via ``__sub__``.

        Raises
        ------
        RuntimeError
            If called before ``.sample()``.
        ValueError
            If ``simulate_over="time"`` without panel, or unknown engine.
        """
        if self._idata is None:
            raise RuntimeError(
                "No posterior samples available. Call .sample() before .do()."
            )

        if simulate_over == "time":
            if self._panel_info is None:
                raise ValueError(
                    "simulate_over='time' requires a panel model. "
                    "Pass panel={...} to fit()."
                )

            if panel_engine != "numpy":
                warnings.warn(
                    f"panel_engine='{panel_engine}' is deprecated and ignored. "
                    f"Panel do() now uses the scan-compiled generative model.",
                    DeprecationWarning,
                    stacklevel=2,
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
                    idata=self._idata,
                    panel_info=self._panel_info,
                    scan_info=scan_info,
                    set=set,
                    kind=kind,
                    families=self._families,
                )

            return run_do_pymc(
                gen_model=self._gen_model,
                graph_info=self._graph_info,
                idata=self._idata,
                data=self._data,
                set=set,
                kind=kind,
            )

        return run_do_pymc(
            gen_model=self._gen_model,
            graph_info=self._graph_info,
            idata=self._idata,
            data=self._data,
            set=set,
            kind=kind,
        )

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
        if condition is None:
            condition = {}
        lo, hi = values
        set_lo = {treatment: lo, **condition}
        set_hi = {treatment: hi, **condition}
        r_lo = self.do(set=set_lo, **do_kwargs)
        r_hi = self.do(set=set_hi, **do_kwargs)
        return r_hi - r_lo

    def prob(
        self,
        expr: str,
        set: dict[str, float] | None = None,
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
        result = self.do(set=set, kind=kind, **do_kwargs)
        namespace = {var: draws for var, draws in result._values.items()}
        import numpy as np

        namespace["np"] = np
        namespace["__builtins__"] = {}
        mask = eval(expr, namespace)  # noqa: S307
        return float(np.mean(mask))


def fit(
    spec_string: str,
    data: pd.DataFrame,
    families: dict[str, str] | None = None,
    panel: dict[str, str] | None = None,
    pooling: str | dict | None = None,
    latent: list[str] | None = None,
    **kwargs: Any,
) -> PathModel:
    """Parse a specification and compile a Bayesian path model.

    Parameters
    ----------
    spec_string : str
        Model specification in the pathmc DSL.
    data : pd.DataFrame
        Observed data.
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
    **kwargs
        Reserved for future options.

    Returns
    -------
    PathModel
        Compiled model ready for sampling and introspection.

    Raises
    ------
    ValueError
        If a latent variable is not endogenous, or an observed endogenous
        variable is missing from data.
    """
    spec = parse_spec(spec_string)
    latent_set = set(latent) if latent is not None else set()
    graph_info = build_graph(spec, latent=latent_set)

    endogenous_lhs = {reg.lhs for reg in spec.regressions}
    for var in endogenous_lhs:
        if var not in latent_set and var not in data.columns:
            raise ValueError(
                f"Endogenous variable '{var}' not found in data columns. "
                f"If '{var}' is an unobserved mediator, declare it via "
                f"latent=['{var}']."
            )

    panel_info: PanelInfo | None = None
    if panel is not None:
        panel_info = build_panel_info(data, panel)

    return PathModel(
        spec=spec,
        graph_info=graph_info,
        data=data,
        families=families,
        panel_info=panel_info,
        pooling=pooling,
        latent=latent_set,
    )
