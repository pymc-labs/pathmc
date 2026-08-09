# Guidelines for Contributing

pathmc welcomes contributions from users, researchers, and developers interested in Bayesian path analysis and structural causal modeling. These guidelines describe how to set up a local development environment, open useful issues, and prepare pull requests that are easy to review.

## Quick Start

The development environment is managed with [uv](https://docs.astral.sh/uv/), a fast Python package and project manager that replaces the old conda-based setup. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) first if you don't have it. After forking this repository on GitHub, get up and running in a few commands:

```bash
git clone git@github.com:<your-github-handle>/pathmc.git
cd pathmc
make setup
make test-fast
```

Common contributor commands are collected in the root `Makefile`; run `make help` to list the available setup, lint, test, docs, and build targets. The targets are thin aliases over the underlying `uv` commands (for example `make setup` runs `uv sync --all-extras` plus the hook install, and `make test-fast` runs `uv run pytest -x -v -m "not slow"`), so you can read the `Makefile` to see the exact command behind each alias.

`make setup` runs `uv sync --all-extras`, which reads `pyproject.toml` and `uv.lock`, then creates a project virtual environment at `.venv/` containing the correct Python (per `.python-version`), pathmc installed in editable mode, the default `dev` dependency group, and every dependency from the `docs` and `samplers` extras. It then installs the pre-commit hooks with `uv run prek install -f`. You do not need to `source .venv/bin/activate` or otherwise activate the environment by hand: the `Makefile` targets prefix commands with `uv run`, which runs them inside `.venv/` (syncing first if anything is stale). If you need a command without a target, prefix it with `uv run` yourself (for example `uv run pytest` or `uv run python -c "import pathmc"`).

## Opening issues

Please file bugs, feature requests, and documentation issues in the [GitHub issue tracker](https://github.com/pymc-labs/pathmc/issues). Before opening a new issue, search existing issues and pull requests for related work so discussion stays consolidated.

Usage questions can also start as issues while the project is young; if GitHub Discussions are enabled later, usage questions should move there and the issue tracker should stay focused on bugs and planned enhancements.

## Use of agents

Pull requests with agent-generated code are welcome, but contributors are responsible for understanding, testing, and maintaining the code they submit. See [AGENTS.md](https://github.com/pymc-labs/pathmc/blob/main/AGENTS.md) for repository-specific guidance used by maintainers and coding agents.

The repository ships six Agent Skills under `.agents/skills/`: five [Great Docs Agent Skills](https://posit-dev.github.io/great-docs/) (`great-docs`, `configure-site`, `write-user-guide`, `revise-docstrings`, `author-skills`) plus a repo-local `fix-bug` skill for autonomous bug-fix workflows. The Great Docs skills are checked in and pinned via `skills-lock.json`; you do not need to install anything to use them. To refresh those five against the latest upstream skills, run `npx skills add https://posit-dev.github.io/great-docs/` from the repo root and commit the result — this updates the lockfile-managed skills and is not expected to add or remove `fix-bug`, which is hand-written and maintained in-repo. The Great Docs wheel does not bundle these repository-specific companion skills, so use the skills CLI for the Great Docs refresh.

pathmc itself ships a curated agent skill at `pathmc/skills/pathmc/SKILL.md`, bundled inside the wheel. Users (not contributors — agents working in this repo see the source file directly) can install it into their own projects with `great-docs skill install pathmc` and keep it fresh with `great-docs skill check --update`, which compares the installed copy's content hash against the installed pathmc package. The same file is also published on the docs site via the `skill` section of `great-docs.yml`.

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

Create the development environment and install the pre-commit hooks:

```bash
make setup
```

Update an existing environment after pulling changes that touch dependencies:

```bash
uv sync --all-extras
```

Dependencies and their version constraints are declared in `pyproject.toml` under `[project].dependencies` (the runtime stack, including the `pymc>=6.0,<7` and matching `pytensor>=3.0,<4` pins), `[dependency-groups]` (the `dev` tools), and `[project.optional-dependencies]` (the `docs` and `samplers` extras). Edit those lists to add or bump a dependency. The `uv.lock` lockfile then pins exact resolved versions for reproducible contributor environments; `uv sync` updates it automatically after `pyproject.toml` changes, and the updated lockfile should be committed alongside the metadata edit.

Run fast tests only, excluding slow MCMC sampling tests:

```bash
make test-fast
```

Run the full test suite, including slow integration tests, and report coverage:

```bash
make test
```

`make test` measures line and branch coverage of `pathmc` and fails if total coverage drops below the `fail_under` threshold in `pyproject.toml` (`[tool.coverage.report]`). The same command and gate run in CI on every pull request. Coverage is intentionally not collected by `make test-fast` or by single-file gate runs, so those stay fast and never trip the threshold on a partial run.

Run a targeted milestone or module test while iterating:

```bash
uv run pytest tests/test_parse.py -x -v
```

Check formatting, linting, and types before opening a pull request:

```bash
make check_lint
```

To apply automatic lint and format fixes by running the pre-commit hooks:

```bash
make lint
```

## Pull request checklist

- Link the issue being addressed, preferably with `Closes #<issue-number>` in the pull request description.
- Add or update tests for user-facing behavior changes.
- Update documentation, examples, or README content when behavior or setup instructions change.
- Run the relevant targeted tests, `make test-fast` or `make test`, and `make check_lint` before requesting review. Pull requests also run `make test` in GitHub Actions (the full suite, including slow MCMC tests).
- Label the pull request before merge so GitHub's generated release notes place it in the correct category; use labels such as `bug`, `documentation`, or `enhancement` when they apply.
- Mark work-in-progress pull requests as drafts until the implementation and test plan are ready for review.

## Building the documentation locally

The documentation site is built with [Great Docs](https://posit-dev.github.io/great-docs/), with [Quarto](https://quarto.org/docs/get-started/) as the underlying renderer. Install Quarto separately before running the docs commands.

The `pathmc` Jupyter kernel is registered automatically by `make setup`, `make docs`, and `make refreeze-docs` (executable pages declare `jupyter: pathmc`). To register it manually:

```bash
make jupyter-kernel
```

Build the static site to `great-docs/_site/` from the project root:

```bash
make docs
```

`make docs` renders HTML from the committed `_freeze/` cache. It does not re-execute notebook cells. `make cleandocs` only removes the ephemeral `great-docs/` build directory and leaves `_freeze/` untouched, so `make cleandocs && make docs` still shows the last-frozen outputs.

Preview the docs locally:

```bash
uv run great-docs preview
```

The site uses `freeze: true` to cache notebook outputs in the committed `_freeze/` directory, so ordinary builds never spawn a Jupyter kernel. After editing a single executable page (or changing pathmc behavior that affects that page's rendered output), refresh its cache and commit it:

```bash
make jupyter-kernel
JUPYTER_PATH=$PWD/.venv/share/jupyter uv run great-docs freeze docs/examples/<notebook_name>.qmd
git add _freeze/
```

To re-execute every `.qmd` page after a dependency upgrade or other change that may affect outputs site-wide, run:

```bash
make refreeze-docs
git add _freeze/
```

This wipes `_freeze/`, re-runs all example and user-guide notebooks except the homepage (which needs a separate build step because of an upstream path-mapping quirk), copies the refreshed homepage cache, and prints a reminder to commit `_freeze/`. Expect this to take a long time: many example notebooks run MCMC sampling.

See the "Building the docs" section of [AGENTS.md](https://github.com/pymc-labs/pathmc/blob/main/AGENTS.md) for freeze-cache details and caveats.

## README assets on PyPI

The PyPI project page renders `README.md` and loads its images from absolute `raw.githubusercontent.com/.../main/docs/assets/...` URLs at **view** time, not from the sdist. Do not rename or relocate `docs/assets/logo_light.png`, `logo_dark.png`, or `contributors.svg` without updating `README.md`; if those paths disappear from `main`, images break on PyPI for every previously published release. PyPI currently has no dark-mode toggle, so the README uses a `<picture>` block for GitHub (light/dark logos) with a light-background fallback for PyPI.
