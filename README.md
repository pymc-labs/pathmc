# pathmc

<p align="center">
  <img src="docs/assets/logo.png" alt="pathmc logo" width="75%">
</p>

**Structural causal models with Bayesian estimation and interventional simulation via a concise DSL.**

pathmc compiles a lavaan-inspired formula language into PyMC models.
Specify a system of structural equations, fit with MCMC, and reason about causal effects using the do-operator.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
```

Activate it:

```bash
conda activate pathmc
```

Update after changes to `environment.yaml`:

```bash
conda env update -f environment.yaml --prune
```

## Tests

Run fast tests only (no MCMC sampling):

```bash
pytest -x -v -m "not slow"
```

Run all tests including slow integration tests:

```bash
pytest -x -v
```

Run a specific milestone's gate tests:

```bash
pytest tests/test_parse.py -x -v
```

## Docs

Requires [Quarto](https://quarto.org/docs/get-started/) to be installed.

Register the conda environment as a Jupyter kernel (one-time setup):

```bash
conda activate pathmc
python -m ipykernel install --user --name pathmc
```

Preview the docs locally (live-reloads on save):

```bash
cd docs
quarto preview
```

Build the static site to `docs/_site/`:

```bash
quarto render docs/
```

### Rendering after code changes

The docs site uses `freeze: auto` to cache notebook outputs. If you change Python source code that affects notebook results, **you must clear the freeze cache** — otherwise Quarto will serve stale outputs from a previous render.

Clear the cache for a single notebook:

```bash
rm -rf docs/_freeze/examples/<notebook_name> docs/.quarto/_freeze/examples/<notebook_name>
quarto render docs/examples/<notebook_name>.qmd
```

Full rebuild from scratch (clears all caches):

```bash
rm -rf docs/_freeze docs/.quarto && quarto render docs/
```
