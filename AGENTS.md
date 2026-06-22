# pathmc — Agent Guide

`pathmc` is a Python package for Bayesian path analysis (observed-variable SEM). It compiles a lavaan-inspired formula DSL into PyMC models, provides introspection, and supports a `do()` operator for interventional simulation.

Remember, when running into bugs or comple issues, it is not impossible we are finding bugs in external packages. In which case, flag potential issues to the user, suggesting the formation of an issue in the relevant upstream repo.

## Workflow

1. Make your changes. Do not modify test files — if a test seems wrong, flag it for human review. Exceptions are for adding new tests or removing obsolte ones, but never for changing the expected behavior of existing tests.
2. Run the relevant gate tests, e.g. `uv run pytest tests/test_<module>.py -x -v`. `make test-fast` skips MCMC sampling; `make test` runs everything.
3. Run `make lint` before considering work done or committing. It runs `prek run --all-files` (ruff, ruff-format, mypy, YAML/TOML, license checks).

Developer-local and ephemeral notes (specs, drafts, exploration) live in `.scratch/` at the repo root, which is gitignored. Never add scratch files to the tracked tree.

## Required Module Structure

Test files import from these fixed paths; internal helpers can be organized freely, but these imports must resolve.

```
pathmc/
  __init__.py     # Public API: model(), simulate(), Prior (re-export)
  parse.py        # parse_spec(spec_string) -> Spec
  graph.py        # build_graph(spec) -> GraphInfo
  compile.py      # Compiler -> pm.Model (Gaussian, Bernoulli, Poisson, etc.)
  simulate.py     # do() operator (cross-sectional + panel)
  effects.py      # Labeled coefficients, defined params, stdyx standardized
  introspect.py   # graph(), equations(), design(), priors()
  transforms.py   # Transform registry (adstock, logistic_saturation)
  identify.py     # Backdoor criterion, adjustment sets, collider warnings
  falsify.py      # Whole-DAG falsification (permutation test)
  sensitivity.py  # Unmeasured-confounding sensitivity analysis
  panel.py        # PanelInfo and panel validation
  priors.py       # default_priors(), merge_priors(); Prior-based config
  residuals.py    # Residual covariance structures (LKJ Cholesky, pluggable)
  idata.py        # Internal ArviZ InferenceData accessors
  exceptions.py   # CycleError, DuplicateEquationError, etc.
  model.py        # PathModel class; model() and simulate() entry points
```

## Environment

A [uv](https://docs.astral.sh/uv/)-managed virtualenv at `.venv/`, pinned by `uv.lock` and `.python-version`. Create or update it with `make setup`, which also installs the `prek` git hooks (a fast pre-commit runner) so the lint checks run automatically on `git commit`. Run everything through `uv run` (e.g. `uv run pytest`, `uv run python -c "..."`) — never the system or conda Python.

This package targets **PyMC ≥ 6.0, ArviZ ≥ 1.1, PyTensor ≥ 3.0, and NumPy ≥ 2.0** — all recent majors with breaking API changes from earlier versions. Do not assume pre-6 PyMC or pre-1.0 ArviZ APIs; verify against the installed versions (`uv run python -c "import pymc, arviz; print(pymc.__version__, arviz.__version__)"`) or the current upstream docs when unsure.

## Building the docs

The site is built with [Great Docs](https://posit-dev.github.io/great-docs/) via `great-docs.yml`: `uv run great-docs build`. The output dir `great-docs/_site/` is ephemeral and gitignored — never edit it. Edit sources instead: `docs/user_guide/*.qmd`, `docs/examples/**/*.qmd`, `great-docs.yml`, and `pathmc/skills/pathmc/SKILL.md`.

Notebooks are frozen (`freeze: true`): the committed `_freeze/` cache supplies cell outputs and CI never spawns a kernel, so **local previews show the last-frozen output, not in-progress edits.** After editing an executable page, refresh and commit the cache (local freezing needs the kernel once: `uv run python -m ipykernel install --user --name pathmc`):

```bash
uv run great-docs freeze docs/examples/01-foundations/my_page.qmd
git add _freeze/
```

The repo-root symlink `examples -> docs/examples` is required by Great Docs Page Tags — do not delete it.

## Style

- `ruff` formatter/linter, line length 88. Type hints and docstrings on all public functions, methods, and classes.
- No global mutable state (no import-time mutable registries).
- Error messages must name the problem AND suggest a fix.
- No narrating comments — comments explain *why*, not *what*.
- No hard-wrapped prose in `.md`/`.qmd` files: one paragraph = one line (block elements and fenced code are exempt).
- Prefer clear, boring code over clever code.

## Architecture Principles

- Parser returns typed dataclass AST nodes, not raw strings or dicts.
- The graph layer is independent of the PyMC compiler — it works from a `Spec` alone, with no data or PyMC objects.
- The do() planner (propagation order) is logically separate from the do() executor (applies posterior draws).
- Residual covariance uses an abstraction layer, not hardcoded LKJ, so alternative structures can be added without major refactors.
- Parameter naming is predictable and stable across runs, via ArviZ/xarray coords.
