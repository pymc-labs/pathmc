# pathmc

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo_dark.png">
    <img src="docs/assets/logo_light.png" alt="pathmc logo" width="75%">
  </picture>
</p>

[![CI](https://github.com/pymc-labs/pathmc/actions/workflows/ci.yml/badge.svg)](https://github.com/pymc-labs/pathmc/actions/workflows/ci.yml)
[![Docs](https://github.com/pymc-labs/pathmc/actions/workflows/docs.yml/badge.svg)](https://pathmc.pymc-labs.com/)
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

*For concepts, tutorials, and the full API reference, see the documentation at [pathmc.pymc-labs.com](https://pathmc.pymc-labs.com/).*

| Capability | What it lets you do |
| --- | --- |
| Model specification | Write structural equations in a compact lavaan-style formula language, with labeled coefficients and derived quantities such as mediated effects. |
| Bayesian estimation | Fit with full-posterior MCMC and get coefficient and effect summaries, including standardized effects, all with quantified uncertainty. |
| Causal identification | Check whether an effect is identifiable from your DAG, find valid adjustment sets, and get warned about collider bias. |
| Graph falsification | Test the conditional independences your DAG implies against the data, including a whole-graph falsification test. |
| Interventional queries | Ask "what if?" with the `do`-operator: average and conditional treatment effects and intervention probabilities. |
| Sensitivity analysis | Stress-test causal conclusions against unmeasured confounding. |
| Transforms & response curves | Apply nonlinear transforms such as adstock and saturation for marketing-mix and other media-response models. |
| Panel & longitudinal data | Build lagged predictors and run interventions on time-series and panel structures. |
| Inspection & visualization | Render the implied DAG, view the model equations, and inspect design matrices. |
| Simulation | Generate synthetic data from a specification for testing, teaching, and method validation. |

## Documentation

Documentation, concepts, and worked examples are available at [pathmc.pymc-labs.com](https://pathmc.pymc-labs.com/). The user guide covers the Bayesian workflow, model specification, transforms, identification and estimation approaches, standardized effects, and panel data, alongside runnable examples from foundations through applied models.

## Contributing

Development setup, testing, and pull request guidance live in [CONTRIBUTING.md](https://github.com/pymc-labs/pathmc/blob/main/CONTRIBUTING.md).

## Citation

If you use pathmc in academic work, please cite the project using the metadata in the repository's [CITATION.cff](https://github.com/pymc-labs/pathmc/blob/main/CITATION.cff). The citation metadata will be updated before the final 0.1.0 release.

## Thanks to our contributors

<a href="https://github.com/pymc-labs/pathmc/graphs/contributors">
  <img src="docs/assets/contributors.svg" alt="pathmc contributors" />
</a>

The contributor image is regenerated weekly by a GitHub Actions workflow (`.github/workflows/contributors.yml`) that opens a PR when the contributor list changes.
