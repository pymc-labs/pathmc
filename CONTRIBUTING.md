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

### AI code attribution with git-ai

We want an honest, measurable picture of how much of pathmc is written by coding agents and how that changes over time. [git-ai](https://usegitai.com/docs/get-started) is an open-source git extension that records this at the line level: supported agents report exactly which lines they wrote as they write them, and on commit the attribution is stored in git notes under `refs/notes/ai`. Because the agents self-report, this is more reliable than trying to guess after the fact, and the data stays local to this repository — there is no account, no hosted service, and no cost.

**Installing git-ai is optional.** It is not a project dependency, it is not part of the test or lint gates, and it will never block a pull request. If you do not install it, your commits simply carry no attribution notes and everything else works exactly as before. Nobody is being asked to install tooling as a condition of contributing. The attribution job is never a required check, and on pull requests from forks it is marked `continue-on-error`, so it cannot turn an outside contributor's pull request red even when it fails outright. On branches pushed directly to this repository it is deliberately allowed to fail visibly, so that maintainers notice when attribution stops being recorded.

If you do want to opt in, install it once per machine (not per repository):

```bash
curl -fsSL https://usegitai.com/install.sh | bash
```

The install is machine-wide rather than per-repository, and it is worth knowing what it touches before you run it: the binary lands in `~/.git-ai/` (symlinked into `~/.local/bin`), `PATH` lines are appended to your shell rc files, `trace2.eventTarget` is set in your global git config so a small background daemon can observe commits, and hooks are installed for whichever supported agents and IDEs it detects (including a VS Code/Cursor extension). Restart any running agent sessions or IDEs afterwards so the hooks are picked up. Nothing else needs configuring, and `git-ai uninstall-hooks` reverses the hook side of it.

**Core devs: check your allow-list.** If you scoped `allow_repositories` to a single repository during the `pymc-marketing` pilot, pathmc will be silently ignored until you add it. This is the most likely reason for a maintainer's attribution to go missing. Widen the glob to cover the org, or add the current repository's remotes directly:

```bash
git-ai config allow_repositories                                    # inspect the current list
git-ai config --add allow_repositories 'https://github.com/pymc-labs/*'
git-ai config --add allow_repositories .                            # or add this repo's remotes
```

Patterns are matched against the repository's remote URLs, and SSH and HTTPS forms are normalised, so `https://github.com/pymc-labs/*` also matches `git@github.com:pymc-labs/pathmc.git`. A repository is allowed if *any* of its remotes matches, so a fork clone that keeps an `upstream` remote pointing at `pymc-labs/pathmc` is covered by the org glob. Note the trade-off between the two commands above: the org glob also enrols every other pymc-labs repository you clone, now and in future, whereas `.` enrols only this one. Prefer `.` if you would rather opt in repository by repository.

**Privacy.** Attribution notes record the agent name, model, session identifiers, token counts, and which lines were AI-written versus human-overridden. Prompt session contents are scanned, redacted, and stored outside git rather than in the notes. Two points on timing are worth being explicit about, because they are easy to miss: notes are synced to the remote by git-ai's own push hook, so they become public **the moment you push a branch**, not when your work is merged — and they stay published even if the pull request is later closed unmerged. Know what is recorded, and when, before you opt in.

Notes do not arrive with a default clone. Fetch them explicitly from whichever remote points at `pymc-labs/pathmc` — that is `origin` if you cloned this repository directly, or `upstream` if you followed the fork workflow in [Local development steps](#local-development-steps):

```bash
git fetch origin 'refs/notes/*:refs/notes/*'     # direct clone
git fetch upstream 'refs/notes/*:refs/notes/*'   # fork workflow
git log --show-notes=ai
git-ai stats <start>..<end>
```

A `.github/workflows/git-ai.yml` job keeps attribution intact across the rewrites GitHub performs server-side. It runs on every push to an open pull request and again when one is closed, but it only does work in the cases that would otherwise lose data — principally squash and rebase merges, which collapse or replace the commits the notes were attached to. Ordinary merge commits already preserve attribution and are skipped.

Two limits on completeness are worth knowing before you read anything into the numbers. Attribution tracks adoption: while only some contributors have git-ai installed, the notes undercount AI authorship and overcount human authorship. And **history before this workflow landed was never consolidated** — every note predating it is still attached to an orphaned pre-squash commit, so `git log --show-notes=ai` and `git-ai stats` will show nothing for that period. That is expected, not a broken install. See the [git-ai docs](https://usegitai.com/docs/get-started) for anything beyond the above.

One limitation to be aware of if you contribute from a fork: GitHub gives `pull_request` events from forks a read-only token, so the job can fetch your fork's notes but cannot push the consolidated result back. Attribution for fork pull requests therefore may not be recorded at all. Nothing about this blocks or slows the pull request; it only means the resulting numbers under-represent outside contributions.

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

The `pathmc` Jupyter kernel is registered automatically by `make setup`, `make docs`, `make freeze-page`, and `make refreeze-docs` (executable pages declare `jupyter: pathmc`). To register it manually:

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
make freeze-page PAGE=docs/examples/01-foundations/my_page.qmd
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
