# pathmc — Agent Guide

Developer and agent related notes are in `./docs/dev/`. Product specifications, milestone tracking, and markdown outputs of agent activity should all go there.

## Project Overview

`pathmc` is a Python package for Bayesian path analysis (observed-variable SEM). It compiles a lavaan-inspired formula DSL into PyMC models, provides introspection, and supports a `do()` operator for interventional simulation.

See `docs/dev/prd_v1.md` for the full product requirements document and `docs/dev/milestones.md` for the implementation plan.

## Status

All v1 milestones (M1–M31) are complete. See `docs/dev/roadmap_post_v1.md` for planned future work.

## How to Work

1. Read `docs/dev/milestones.md` to identify the current milestone.
2. Run the milestone's gate tests: `uv run pytest tests/test_<module>.py -x -v`
3. Implement until all gate tests pass.
4. **Do not modify test files.**
5. Run `make lint` before considering a milestone done or creating a commit. This runs `prek run --all-files`, including the configured `ruff`, `ruff-format`, `mypy`, YAML/TOML, and license checks.
6. Move to the next milestone.

## Required Module Structure

The test files import from specific modules. These paths are **fixed**:

```
pathmc/
  __init__.py       # Public API exports: model(), fit() (deprecated alias), add_lags() (deprecated)
  parse.py          # parse_spec(spec_string) -> Spec
  graph.py          # build_graph(spec) -> GraphInfo
  compile.py        # Compiler -> pm.Model (Gaussian, Bernoulli, Poisson, etc.)
  simulate.py       # do() operator logic (cross-sectional + panel)
  effects.py        # Labeled coefficients, defined params, stdyx standardized
  introspect.py     # graph(), equations(), design(), priors()
  transforms.py     # Transform registry (adstock, logistic_saturation)
  identify.py       # Backdoor criterion, adjustment sets, collider warnings
  panel.py          # PanelInfo, add_lags(), panel validation
  exceptions.py     # CycleError, DuplicateEquationError, etc.
  model.py          # PathModel class (returned by model()), model() and fit() entry points
```

Additional internal helpers and submodules can be organized freely, but the imports used in the test files must resolve.

## Environment

The development environment is a [uv](https://docs.astral.sh/uv/)-managed virtualenv at `.venv/`, pinned by `uv.lock` and `.python-version`. Create or update it with:

```bash
make setup        # uv sync --all-extras + pre-commit hook install
```

All commands (tests, scripts, docs builds) **must** run in this environment. The Makefile targets already invoke tools through `uv run`. When running Python snippets to verify behavior, do the same — **not** the system or base conda Python:

```bash
uv run python -c "..."
```

The Jupyter kernel used by Quarto notebooks is named `pathmc` and must point to this environment's Python. Register it once with `uv run python -m ipykernel install --user --name pathmc`.

### Building the docs

The site is built with [Great Docs](https://posit-dev.github.io/great-docs/), driven by `great-docs.yml` at the repo root. Quarto is still the underlying renderer.

```bash
uv run great-docs build                # full build to great-docs/_site/
uv run great-docs build --no-refresh   # faster rebuild — skips API rediscovery
uv run great-docs preview              # local server on http://localhost:3000
```

The `great-docs/` directory is **ephemeral**: it is wiped at the start of every build and listed in `.gitignore`. Never edit files under `great-docs/` directly — change source files (`docs/user_guide/*.qmd`, `docs/examples/*.qmd`, `great-docs.yml`, `pathmc/skills/pathmc/SKILL.md`) instead.

#### Notebooks are frozen — refresh after edits

`great-docs.yml` sets `freeze: true` project-wide. The committed `_freeze/` directory at the repo root stores Quarto's cached cell outputs; `great-docs build` (locally and in CI) restores it before rendering and never spawns a Jupyter kernel. **Local previews show the last-frozen output, not your in-progress edits.**

After editing an executable page, or after a pathmc API change that affects rendered output:

```bash
uv run great-docs freeze docs/examples/my_page.qmd     # or multiple paths
git add _freeze/
git commit -m "Refresh freeze cache for my_page"
```

`great-docs freeze --info` shows per-page cache status; `great-docs freeze --clean <pages>` wipes and regenerates specific entries.

**Homepage caveat.** `docs/user_guide/00-welcome.qmd` is mapped to the site index, and the freeze CLI cannot resolve it (looks for `user-guide/welcome.qmd`, finds `index.qmd`). To refresh the welcome cache: run `great-docs build` once (which executes welcome and writes `great-docs/_freeze/index/`), then `cp -r great-docs/_freeze/index _freeze/` and commit. See `docs/dev/great_docs_migration.md` ("How freeze works for pathmc") for the full rationale, version-pin notes, and upstream issue list.

## Running Tests

```bash
# Fast tests only (no MCMC sampling)
make test-fast

# Specific milestone gate
uv run pytest tests/test_parse.py -x -v

# All tests including slow (sampling) tests
make test

# Single test class
uv run pytest tests/test_compile.py::TestDesignMatrix -x -v
```

## Style Guide

- **Formatter/linter**: `ruff` (config in pyproject.toml, line-length 88)
- **Type hints** on all public functions and methods
- **Docstrings** on all public functions, methods, and classes
- **No global mutable state** — no module-level dicts, lists, or registries that get mutated at import time
- **Error messages** must name the problem AND suggest a fix (e.g., "Duplicate equation for 'Y'. Each variable can appear as LHS in at most one regression.")
- **No narrating comments** — don't write `# parse the spec` above a call to `parse_spec()`. Comments should explain *why*, not *what*.
- **No hard-wrapped prose in `.md` or `.qmd` files.** One paragraph = one line. Do not insert newlines mid-paragraph or mid-list-item to wrap to a column width. Let the editor soft-wrap. Block elements (headings, list markers, table rows, fenced code, blockquotes, blank lines between paragraphs) are unaffected — only mid-paragraph line breaks are forbidden. This applies to all hand-written markdown including `AGENTS.md`, files under `docs/`, `docs/dev/`, `README.md`, and any `.qmd` source. Code inside fenced blocks is exempt.

## Architecture Principles

- **Parser** returns typed dataclass AST nodes, not raw strings or dicts.
- **Graph layer** is independent of the PyMC compiler — it should work with just a `Spec`, no data or PyMC objects.
- **do() planner** is logically separate from the do() executor — plan determines propagation order; execute applies posterior draws.
- **Residual covariance** uses an abstraction layer, not hardcoded LKJ. The roadmap calls for alternative residual structures (low-rank, group shocks); the design should accommodate this without major refactors.
- **Parameter naming** must be predictable, documented, and stable across runs. Use ArviZ/xarray coords for equations, coefficients, and multivariate blocks.
- **Dependencies**: `networkx` (graph/identification) and `graphviz` (DAG rendering) are already in `pyproject.toml`. The adstock/saturation transform kernels are implemented directly in `pathmc/transforms.py`; delegating to `pymc-marketing` again is planned once it supports PyMC 6.

## Do NOT

- Modify test files. If a test seems wrong, flag it for human review.
- Add dependencies without documenting why in a commit message.
- Use global state or module-level mutable variables.
- Suppress warnings without documenting the reason.
- Write "clever" code — prefer clear, boring implementations.
- Commit temporary/scratch files (e.g., draft issue text, PR summary scaffolds). Files intended to be persistent parts of the repo (docs, config, source) are fine to commit.
- Hard-wrap paragraphs in markdown or Quarto files (see Style Guide). Keep each paragraph and list item on a single line.
