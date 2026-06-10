# Guidelines for Contributing

pathmc welcomes contributions from users, researchers, and developers interested in Bayesian path analysis and structural causal modeling. These guidelines describe how to set up a local development environment, open useful issues, and prepare pull requests that are easy to review.

## Quick Start

After forking this repository on GitHub, get up and running in a few commands:

```bash
git clone git@github.com:<your-github-handle>/pathmc.git
cd pathmc
conda env create -f environment.yml
conda activate pathmc
make setup
make test-fast
```

Common contributor commands are collected in the root `Makefile`; run `make help` to list the available setup, lint, test, docs, environment sync, and build targets.

## Opening issues

Please file bugs, feature requests, and documentation issues in the [GitHub issue tracker](https://github.com/pymc-labs/pathmc/issues). Before opening a new issue, search existing issues and pull requests for related work so discussion stays consolidated.

Usage questions can also start as issues while the project is young; if GitHub Discussions are enabled later, usage questions should move there and the issue tracker should stay focused on bugs and planned enhancements.

## Use of agents

Pull requests with agent-generated code are welcome, but contributors are responsible for understanding, testing, and maintaining the code they submit. See [AGENTS.md](https://github.com/pymc-labs/pathmc/blob/main/AGENTS.md) for repository-specific guidance used by maintainers and coding agents.

The repository ships five [Great Docs Agent Skills](https://posit-dev.github.io/great-docs/) under `.agents/skills/` (`great-docs`, `configure-site`, `write-user-guide`, `revise-docstrings`, `author-skills`) so that AI coding agents working on the documentation site have structured context about Great Docs configuration, build steps, user-guide authoring, docstring conventions, and skill anatomy. The files are checked in and pinned via `skills-lock.json`; you do not need to install anything to use them. To refresh against the latest upstream skills, run `npx skills add https://posit-dev.github.io/great-docs/` from the repo root and commit the result.

## Contributing code via pull requests

The preferred workflow is to fork the repository, clone your fork locally, and develop on a feature branch. Keep pull requests focused on one issue or behavior change, include tests and documentation when appropriate, and explain the user-facing reason for the change in the pull request description.

## Local development steps

Fork the [project repository](https://github.com/pymc-labs/pathmc), clone your fork, and add the upstream repository:

```bash
git clone git@github.com:<your-github-handle>/pathmc.git
cd pathmc
git remote add upstream git@github.com:pymc-labs/pathmc.git
```

Create a feature branch for your work:

```bash
git checkout -b my-feature
```

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate pathmc
```

Update an existing environment after pulling changes to `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Install pathmc in editable mode with development dependencies and hooks:

```bash
make setup
```

Edit dependency metadata in `pyproject.toml`, not `environment.yml`. The conda environment file is generated from `pyproject.toml`; run `make sync-env` after dependency changes, or let the pre-commit hook regenerate it.

Run fast tests only, excluding slow MCMC sampling tests:

```bash
make test-fast
```

Run the full test suite, including slow integration tests:

```bash
make test
```

Run a targeted milestone or module test while iterating:

```bash
pytest tests/test_parse.py -x -v
```

Check formatting, linting, and types before opening a pull request:

```bash
make check_lint
```

To apply automatic lint and format fixes:

```bash
make lint
```

## Pull request checklist

- Link the issue being addressed, preferably with `Closes #<issue-number>` in the pull request description.
- Add or update tests for user-facing behavior changes.
- Update documentation, examples, or README content when behavior or setup instructions change.
- Run the relevant targeted tests, `make test-fast`, and `make check_lint` before requesting review.
- Label the pull request before merge so GitHub's generated release notes place it in the correct category; use labels such as `bug`, `documentation`, or `enhancement` when they apply.
- Mark work-in-progress pull requests as drafts until the implementation and test plan are ready for review.

## Building the documentation locally

The documentation site is built with [Quarto](https://quarto.org/docs/get-started/). Install Quarto separately before running the docs commands.

Register the conda environment as a Jupyter kernel once:

```bash
conda activate pathmc
python -m ipykernel install --user --name pathmc
```

Preview the docs locally with live reload:

```bash
cd docs
quarto preview
```

Build the static site to `docs/_site/` from the project root:

```bash
make docs
```

The docs site uses `freeze: auto` to cache notebook outputs. If you change Python source code that affects notebook results, clear the affected freeze cache before rendering or Quarto may serve stale outputs.

Clear the cache for a single notebook and render it again:

```bash
rm -rf docs/_freeze/examples/<notebook_name> docs/.quarto/_freeze/examples/<notebook_name>
quarto render docs/examples/<notebook_name>.qmd
```

Full rebuild from scratch:

```bash
make cleandocs && make docs
```
