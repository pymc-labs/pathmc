# pathmc — Agent Guide

Developer and agent related notes are in `./docs/dev/`. Product specifications, milestone tracking, and markdown outputs of agent activity should all go there.

## Project Overview

`pathmc` is a Python package for Bayesian path analysis (observed-variable SEM). It compiles a lavaan-inspired formula DSL into PyMC models, provides introspection, and supports a `do()` operator for interventional simulation.

See `docs/dev/prd_v1.md` for the full product requirements document and `docs/dev/milestones.md` for the implementation plan.

## Status

All v1 milestones (M1–M31) are complete. See `docs/dev/roadmap_post_v1.md` for planned future work.

## How to Work

1. Read `docs/dev/milestones.md` to identify the current milestone.
2. Run the milestone's gate tests: `pytest tests/test_<module>.py -x -v`
3. Implement until all gate tests pass.
4. **Do not modify test files.**
5. Run `make lint` before considering a milestone done or creating a commit. This runs `prek run --all-files`, including the configured `ruff`, `ruff-format`, `mypy`, YAML/TOML, environment sync, and license checks.
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

All commands (tests, scripts, quarto render) **must** run in the `pathmc` conda environment:

```bash
conda activate pathmc
```

The Jupyter kernel used by Quarto notebooks is also named `pathmc` and points to this environment's Python (`miniforge3/envs/pathmc/bin/python`). When running Python snippets to verify behavior, always use this environment — **not** the base conda env. If using a full path:

```bash
/Users/benjamv/miniforge3/envs/pathmc/bin/python -c "..."
```

### Quarto freeze cache

The docs site uses `execute: freeze: auto` in `docs/_quarto.yml`, which caches notebook outputs. After changing Python source code that affects notebook outputs, **clear the freeze cache** for affected notebooks before re-rendering:

```bash
# Clear cache for a specific notebook
rm -rf docs/_freeze/examples/<notebook_name> docs/.quarto/_freeze/examples/<notebook_name>

# Full rebuild from scratch
make cleandocs && make docs
```

Without this, Quarto will serve stale cached figures/outputs even though the underlying code has changed.

## Running Tests

```bash
# Fast tests only (no MCMC sampling)
make test-fast

# Specific milestone gate
pytest tests/test_parse.py -x -v

# All tests including slow (sampling) tests
make test

# Single test class
pytest tests/test_compile.py::TestDesignMatrix -x -v
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
- **Dependencies**: `networkx` (graph/identification), `graphviz` (DAG rendering), `pymc-marketing` (adstock/saturation transform backends) are already in `pyproject.toml`.

## Do NOT

- Modify test files. If a test seems wrong, flag it for human review.
- Add dependencies without documenting why in a commit message.
- Use global state or module-level mutable variables.
- Suppress warnings without documenting the reason.
- Write "clever" code — prefer clear, boring implementations.
- Commit temporary/scratch files (e.g., draft issue text, PR summary scaffolds). Files intended to be persistent parts of the repo (docs, config, source) are fine to commit.
- Hard-wrap paragraphs in markdown or Quarto files (see Style Guide). Keep each paragraph and list item on a single line.
