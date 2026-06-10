# pathmc

<p align="center">
  <img src="docs/assets/logo.png" alt="pathmc logo" width="75%">
</p>

[![Status: beta](https://img.shields.io/badge/status-beta-orange)](https://pypi.org/project/pathmc/)

**Structural causal models with Bayesian estimation and interventional simulation via a concise DSL.**

## What is pathmc?

pathmc is a Python package for Bayesian path analysis and structural causal modeling. It compiles a lavaan-inspired formula DSL into PyMC models, keeps the DAG at the center of the workflow, and lets you estimate effects with full posterior uncertainty. Use it to specify structural equations, fit with MCMC, inspect the implied graph, and run causal `do()` queries from the same model object.

## Installation

```bash
pip install pathmc
```

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
m.ate("Y", "X", values=(0, 1))
```

The same model object can also render the DAG, summarize labeled coefficients and defined parameters, check identification, and run sensitivity analyses.

## Documentation

Documentation, concepts, and worked examples are available at [pymc-labs.github.io/pathmc](https://pymc-labs.github.io/pathmc/).

## Contributing

Development setup, testing, and pull request guidance live in [CONTRIBUTING.md](https://github.com/pymc-labs/pathmc/blob/main/CONTRIBUTING.md).

## Citation

If you use pathmc in academic work, please cite the project using the metadata in the repository's [CITATION.cff](https://github.com/pymc-labs/pathmc/blob/main/CITATION.cff). The citation metadata will be updated before the final 0.1.0 release.

## Thanks to our contributors

<a href="https://github.com/pymc-labs/pathmc/graphs/contributors">
  <img src="docs/assets/contributors.svg" alt="pathmc contributors" />
</a>

The contributor image is regenerated weekly by a GitHub Actions workflow (`.github/workflows/contributors.yml`) that opens a PR when the contributor list changes.
