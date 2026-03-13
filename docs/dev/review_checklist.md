# Review Checklist — pathmc

Use this checklist after each milestone to evaluate quality beyond what gate tests can verify. Each item is binary (pass/fail). A reviewer (human or agent) should provide evidence for each judgment.

## Code Quality

- [ ] No global mutable state (no module-level dicts/lists that get mutated)
- [ ] All public functions and methods have type hints
- [ ] All public functions, methods, and classes have docstrings
- [ ] No comments that merely narrate what code does ("# parse the spec")
- [ ] `ruff check` and `ruff format --check` pass cleanly
- [ ] No `# type: ignore` without an inline justification
- [ ] No bare `except:` clauses — exceptions are caught specifically

## Architecture

- [ ] Parser returns typed AST nodes (dataclasses or similar), not raw strings or dicts
- [ ] Graph layer (`pathmc.graph`) has no dependency on PyMC — it works with `Spec` only
- [ ] Compiler (`pathmc.compile`) is the only module that imports PyMC
- [ ] do() planner is logically separate from do() executor
- [ ] Residual covariance uses an abstraction (not hardcoded LKJ) so alternative structures can slot in later
- [ ] PathModel does not expose internal implementation details through its public API
- [ ] No circular imports between pathmc submodules

## API Ergonomics

- [ ] A user can specify, fit, inspect, and query a model in ≤10 lines of code
- [ ] `pathmc.model()` is the single entry point — no multi-step construction required
- [ ] Introspection methods (`graph()`, `equations()`, `priors()`) return human-readable output
- [ ] Introspection methods work before `.fit()` is called
- [ ] `do()` returns objects that support natural arithmetic (scenario - baseline)
- [ ] `effects_summary()` returns a familiar format (DataFrame with mean, sd, HDI columns)

## Error Handling

- [ ] Invalid spec syntax raises immediately at `parse_spec()` time, not during compilation
- [ ] Cycle in the DAG raises at `build_graph()` time with a message naming the cycle
- [ ] Duplicate LHS raises at parse time with a message naming the duplicated variable
- [ ] Calling `do()` before `fit()` raises a clear error (not a cryptic AttributeError)
- [ ] `~~` between non-Gaussian outcomes raises at compilation time with guidance
- [ ] Unknown variable references in `:=` raise at parse or compile time
- [ ] All error messages name the problem AND suggest a fix or next step

## Parameter Naming & Coordinates

- [ ] Parameter names in the PyMC model follow a documented, predictable convention
- [ ] ArviZ/xarray coords are set for equations, coefficients, and multivariate blocks
- [ ] Parameter names are stable across runs (no randomness or order-dependence)
- [ ] `az.summary()` on the InferenceData produces readable output without renaming

## Documentation (M10)

- [ ] `quarto render` exits 0 with no errors
- [ ] Each example notebook is self-contained (generates its own synthetic data)
- [ ] Each example teaches one primary concept
- [ ] Causal assumptions and limitations are stated explicitly in the intro
- [ ] `do()` documentation includes caveats about causal identification
- [ ] Code in examples follows the same style guide as the package source
