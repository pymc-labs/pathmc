# pathmc

<p align="center">
  <img src="docs/assets/logo.png" alt="pathmc logo" width="75%">
</p>

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
cd docs
quarto render
```
