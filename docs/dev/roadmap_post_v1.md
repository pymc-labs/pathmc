# `pathmc` Roadmap (post v1)

Features below are out of scope for v1 but inform architectural decisions. The code should be designed so these can be layered on without major refactors.

See [prd_v1.md](prd_v1.md) for the v1 scope and requirements.

## High priority

- **DAG compatibility checks via implied independences (d-separation)**
  Test conditional independences implied by the DAG using PPC-based diagnostics. Rank the most violated implications to guide DAG refinement. Directly supports the "rapid DAG iteration" workflow.

- **Sensitivity analysis for causal claims**
  Unmeasured confounding sensitivity (parametric bias / latent confounder knobs). High value because any causal claim rests on untestable assumptions; quantifying fragility is essential.

- **DAG comparison and scoring**
  Compare candidate DAGs by predictive criteria (LOO/WAIC, held-out predictive performance). Optional complexity-aware graph priors.

## Medium priority

- **Graph-aware priors and regularization**
  Global-local shrinkage over edges (e.g., horseshoe on coefficients). Tier/time-order shrinkage (shrink long-range or cross-tier edges more aggressively). Useful for large candidate DAGs. Design the prior system to accommodate this from the start.

- **Exploratory misfit diagnostics (opt-in)**
  Bayesian alternative to SEM modification indices. Posterior predictive residual correlation diagnostics. Rank candidate additions: suggested `~~` edges, suggested directed edges (subject to acyclicity). Must be clearly labeled exploratory.

- **Residual structure objects (beyond pairwise `~~`)**
  Phase 1 is complete: a `ResidualStructure` protocol and `LKJResidual` implementation exist in `pathmc/residuals.py`, with `mu_{var}` deterministics and `endogenous_rvs` wiring so `do()` propagates through block variables. Remaining: Phase 2 adds alternative structures (diagonal, low-rank), a `residual_structure=` parameter on `model()`, and full prior customization for LKJ `eta`/`sd_dist`. Phase 3 adds panel `~~` support. See #89 for details.

## Lower priority

- **Policy optimization**
  `optimize_policy()` using `do()` mean simulator + constrained optimization. Budget constraints, smoothness penalties. Particularly compelling for longitudinal applications.

- **Unit-level counterfactuals**
  Distinct from interventional distributions. `model.counterfactual(evidence=..., do=...)` requiring an abduction step to condition on observed evidence before intervening.

- **Soft / mechanism interventions**
  Intervene on a structural mechanism rather than a variable (e.g., scale a coefficient: "reduce price elasticity by 20%") and simulate outcomes.

## Architectural implications

These roadmap items suggest the following design-time considerations:

- The **graph layer** already exposes d-separation queries, adjustment set computation (`identify.py`), and topological utilities. DAG compatibility checks and sensitivity analysis can build on this foundation.
- The **prior system** should accept per-edge or per-group prior specs, making graph-aware priors a configuration change rather than a refactor.
- The **`do()` simulator** is already modular (separate planner via topological order, separate executor in `simulate.py`). Policy optimization and counterfactual queries can layer on top.
- The **residual covariance** uses a `ResidualStructure` protocol so alternative covariance parameterizations (low-rank, group-shock) can slot in without modifying the compiler.
