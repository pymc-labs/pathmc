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
  Low-rank residual factors (a small number of latent shocks inducing residual correlation). Group/time shocks for panel data (shared disturbances by time period or cluster). Useful when many outcomes share a few unmodeled common causes.

## Lower priority

- **Policy optimization**
  `optimize_policy()` using `do()` mean simulator + constrained optimization. Budget constraints, smoothness penalties. Particularly compelling for longitudinal applications.

- **Unit-level counterfactuals**
  Distinct from interventional distributions. `fit.counterfactual(evidence=..., do=...)` requiring an abduction step to condition on observed evidence before intervening.

- **Soft / mechanism interventions**
  Intervene on a structural mechanism rather than a variable (e.g., scale a coefficient: "reduce price elasticity by 20%") and simulate outcomes.

## Architectural implications

These roadmap items suggest the following design-time considerations:

- The **graph layer** should expose d-separation queries, adjustment set computation, and topological utilities — not just compilation.
- The **prior system** should accept per-edge or per-group prior specs, making graph-aware priors a configuration change rather than a refactor.
- The **`do()` simulator** should be modular enough to serve as the engine for optimization and counterfactual queries.
- The **residual covariance** implementation should use an abstraction (e.g., a residual structure object) rather than hardcoding LKJ, so low-rank and group-shock variants can slot in later.
