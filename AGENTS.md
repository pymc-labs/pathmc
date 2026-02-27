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
5. Run `ruff check --fix && ruff format` before considering a milestone done.
6. Move to the next milestone.

## Required Module Structure

The test files import from specific modules. These paths are **fixed**:

```
pathmc/
  __init__.py       # Public API exports: fit(), add_lags()
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
  model.py          # PathModel class (returned by fit())
```

Additional internal helpers and submodules can be organized freely, but the imports used in the test files must resolve.

## Running Tests

```bash
# Fast tests only (no MCMC sampling)
pytest -x -v -m "not slow"

# Specific milestone gate
pytest tests/test_parse.py -x -v

# All tests including slow (sampling) tests
pytest -x -v

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
