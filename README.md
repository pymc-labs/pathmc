# pathmc

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_dark.png">
    <img src="docs/assets/logo_light.png" alt="pathmc logo" width="75%">
  </picture>
</p>

[![CI](https://github.com/pymc-labs/pathmc/actions/workflows/ci.yml/badge.svg)](https://github.com/pymc-labs/pathmc/actions/workflows/ci.yml)
[![Docs](https://github.com/pymc-labs/pathmc/actions/workflows/docs.yml/badge.svg)](https://pymc-labs.github.io/pathmc/)
[![Status: beta](https://img.shields.io/badge/status-beta-orange)](https://github.com/pymc-labs/pathmc)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](https://github.com/pymc-labs/pathmc/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

**Structural causal models with Bayesian estimation and interventional simulation via a concise DSL.**

## What is pathmc?

pathmc is a Python package for **Bayesian path analysis and structural causal modeling**. You write your causal assumptions as a small set of structural equations in a [lavaan](https://lavaan.ugent.be/)-inspired formula language, and pathmc compiles them into a generative [PyMC](https://www.pymc.io/) model in which every variable is wired through its structural parents in the DAG.

The result is a single model object that keeps the **directed acyclic graph (DAG) at the center of the workflow**. From that one object you can estimate effects with full posterior uncertainty, inspect the implied graph, check whether a causal effect is identifiable from your assumptions, falsify the graph against the data, run `do()`-operator interventions, and stress-test conclusions for unmeasured confounding, all without rewriting the model for each task.

pathmc is a good fit when you want the interpretability of path / structural equation modeling, the rigor of explicit causal identification, and the uncertainty quantification of full Bayesian inference, all in one place.

## Installation

```bash
pip install pathmc
```

pathmc requires Python ≥ 3.12 and PyMC ≥ 5.22.

## Quickstart

Pass a pandas DataFrame with columns matching the variables in your structural equations:

```python
import pathmc

spec = """
M ~ a*X
Y ~ b*M + c*X
indirect := a*b
"""

m = pathmc.model(spec, data=df)
m.fit(draws=1000, chains=2)

m.effects_summary()             # labeled coefficients + defined parameters
m.ate("Y", "X", values=(0, 1))  # average treatment effect via the do-operator
m.adjustment_sets("X", "Y")     # what to condition on for identification
```

The DSL mirrors lavaan: `~` defines a regression of an outcome on its parents, `label*var` attaches a name to a coefficient, and `:=` defines derived quantities (such as an indirect/mediated effect) that are tracked with full posterior uncertainty.

## Capabilities

**Model specification**
- A compact, lavaan-style formula DSL that compiles directly to a generative PyMC model.
- Labeled coefficients (`a*X`) and defined parameters (`indirect := a*b`) for mediation, total, and custom derived effects.
- Configurable priors (`set_priors`, `pymc_extras.Prior`), prior predictive checks, and a tabular view of every prior in the model.
- Residual covariance structures for correlated errors.

**Bayesian estimation**
- Full posterior inference via PyMC's MCMC samplers (`fit`), returning an ArviZ `InferenceData`.
- Posterior and posterior-predictive summaries, with standardized-effect reporting (`standardized`, `effects_summary`).
- `predict` for posterior-predictive draws on new or counterfactual data.

**Causal identification**
- Derive valid back-door adjustment sets (`adjustment_sets`) and test whether an effect is identifiable from the DAG (`is_identifiable`, `frontdoor_identifiable`).
- Surface collider-bias risks (`collider_warnings`) before you condition on the wrong variable.

**Graph falsification**
- Enumerate the conditional independences implied by your DAG (`implied_independences`) and test them against the data (`test_implications`).
- Run a whole-graph, permutation-based falsification test (`falsify`) adapted from Eulig et al. (2023) / DoWhy's `gcm.falsify_graph`.

**Interventional queries (do-calculus)**
- Simulate interventions by propagating `do()` through the posterior.
- Estimate average and conditional treatment effects: `ate`, `cate`, `att`, `atu`, plus arbitrary intervention probabilities via `prob`.
- Query individual path effects with `effect`.

**Sensitivity analysis**
- Quantify how robust a causal conclusion is to unmeasured confounding (`sensitivity`).

**Transforms & response curves**
- Built-in nonlinear transforms including geometric **adstock** and **logistic saturation**, plus a registry for custom transforms, useful for marketing mix and other media-response models.

**Panel / longitudinal data**
- Build lagged predictors for time-series and panel structures (`add_lags`) and run interventions on longitudinal models.

**Inspection & visualization**
- Render the implied DAG (`graph`, `to_graphviz`), view the symbolic model equations (`equations`, `model_equations`), and inspect the design matrix for any variable (`design`).

**Simulation**
- Generate synthetic data from a specification with `pathmc.simulate` for testing, teaching, and method validation.

## Documentation

Documentation, concepts, and worked examples are available at [pymc-labs.github.io/pathmc](https://pymc-labs.github.io/pathmc/). The user guide covers the Bayesian workflow, model specification, transforms, identification and estimation approaches, standardized effects, and panel data, alongside runnable examples from foundations through applied models.

## Contributing

Development setup, testing, and pull request guidance live in [CONTRIBUTING.md](https://github.com/pymc-labs/pathmc/blob/main/CONTRIBUTING.md).

## Citation

If you use pathmc in academic work, please cite the project using the metadata in the repository's [CITATION.cff](https://github.com/pymc-labs/pathmc/blob/main/CITATION.cff). The citation metadata will be updated before the final 0.1.0 release.

## Thanks to our contributors

<a href="https://github.com/pymc-labs/pathmc/graphs/contributors">
  <img src="docs/assets/contributors.svg" alt="pathmc contributors" />
</a>

The contributor image is regenerated weekly by a GitHub Actions workflow (`.github/workflows/contributors.yml`) that opens a PR when the contributor list changes.
