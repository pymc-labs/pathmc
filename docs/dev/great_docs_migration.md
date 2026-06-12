# Great Docs Migration Plan

Closes [#175](https://github.com/pymc-labs/pathmc/issues/175).

This plan migrates `pathmc`'s documentation from a hand-rolled Quarto website (`docs/_quarto.yml`) to a [Great Docs](https://posit-dev.github.io/great-docs/) build. Great Docs is itself a layer on top of Quarto, so the underlying renderer does not change — what changes is how the site is configured, structured, and deployed, plus the addition of auto-generated API reference, `llms.txt`, `llms-full.txt`, and an Agent Skills file for downstream pathmc users.

## Status

The PR for this migration tracks progress against the phased plan below. Tick items as they land. The intent is for **the PR to be merge-ready before public deployment** — phases 0 through 5 land in the PR; the final flip to public Pages (Phase 4b) is a separate, manually-triggered step taken when we're ready to announce.

- [x] **Phase 0** — branch + skill installed locally + plan written
- [x] **Phase 1** — non-destructive spike with `great-docs init/build/preview`
- [x] **Phase 2** — content migration (`great-docs.yml`, per-page frontmatter); landed in same commits as Phase 1
- [x] **Phase 3** — curated `skills/pathmc/SKILL.md`
- [x] **Phase 4a** — CI build job (`Build Docs`) is enabled and renders the committed `_freeze/` cache without executing notebooks. `Publish Docs` remains guarded until Phase 4b. Per-PR preview deploy was removed (#226); reviewers use the `docs-html` artifact instead. See "How freeze works for pathmc" below.
- [x] **Phase 5** — cleanup (`docs/_quarto.yml`, freeze cache, `AGENTS.md`)
- [ ] **Phase 4b** — *deferred to launch day, not part of this PR.* Flip Settings → Pages → Source → GitHub Actions; add `Documentation` URL to `pyproject.toml`.

### Phase 1 / Phase 2 outcome

**Decisions taken:**

- `homepage: user_guide` (not `index` from `README.md`) — the README is dev-focused; the existing `docs/index.qmd` showcase is much better as a landing page. It moved to `docs/user_guide/00-welcome.qmd` and Great Docs blends it as the homepage.
- `dynamic: false` — auto-detected by `great-docs init`. pathmc imports PyMC at module level which trips up dynamic introspection's cyclic-alias detector. griffe's static AST analysis is sufficient and gives identical output for our docstrings.
- `hero: false` — Great Docs auto-injects a hero block above any homepage; we already have a logo + tagline + showcase in `00-welcome.qmd`, so the hero produces a duplicate. Also: there's a [bug in great-docs 0.10.0](https://github.com/posit-dev/great-docs/issues) where the hero's HTML fenced block is emitted as `` ```{=html}  # pragma: no cover `` — the trailing comment breaks Quarto's attribute parser and the hero renders as literal text. `hero: false` sidesteps both issues.
- `navbar_style: mint`, `dark_mode_toggle: true` — picked `mint` to complement the existing logo palette.
- Explicit `reference:` config: only `model.model`, `model.PathModel`, `model.simulate`, `model.DoResult`, `panel.add_lags`. **Important:** writing a bare `model` in the contents list expands it to *every* class/function in the `pathmc.model` module (Great Docs treats unqualified names as module names). Use the dotted-prefix form (`model.PathModel`) to disambiguate.
- Custom section for examples uses `index: true, index_columns: 3` and Great Docs auto-generates the card grid index. The old `docs/examples/index.qmd` was deleted.

**Content restructure:**

- `docs/concepts/` → `docs/user_guide/` (renamed dir, files renumbered with `10-`–`17-` prefixes for stable order).
- `docs/how-it-works.qmd` → `docs/user_guide/01-how-it-works.qmd`.
- `docs/comparison.qmd` → `docs/user_guide/02-comparison.qmd`.
- `docs/index.qmd` → `docs/user_guide/00-welcome.qmd` (and image path adjusted to `../assets/logo.png`).
- All cross-links updated to point at the new paths.

**Open issues to file upstream against `posit-dev/great-docs`:**

1. **Hero block emits invalid Quarto attribute.** `great_docs/core.py:4886` produces `` ```{=html}  # pragma: no cover `` which Quarto can't parse. The `# pragma: no cover` is a Python coverage marker that has leaked into a doc string. Workaround: `hero: false`.
2. ~~**No way to persist Quarto's freeze cache across builds.**~~ Resolved upstream by [posit-dev/great-docs#158](https://github.com/posit-dev/great-docs/pull/158) (closes [#155](https://github.com/posit-dev/great-docs/issues/155)). pathmc adopts the new `freeze: true` config and the `great-docs freeze` CLI; see "How freeze works for pathmc" below.
3. **Bare module names in `reference.contents` silently expand.** This is documented behaviour from reading the source, but undocumented in the user-facing config reference. Worth a docs improvement upstream.
4. **`great-docs freeze` cannot resolve the homepage.** When a page is mapped to the site index via `homepage:`, the freeze CLI fails with `not found in build directory` because it looks under the page's source-relative path (e.g. `user-guide/welcome.qmd`) rather than `index.qmd`. Workaround for `docs/user_guide/00-welcome.qmd`: run `great-docs build` once to populate `great-docs/_freeze/index/`, then copy that subtree to `_freeze/index/` and commit. Repeat after edits.

### How freeze works for pathmc

The CI `Build Docs` job renders HTML without executing any notebook. The mechanism is upstream's [Freeze & Caching](https://posit-dev.github.io/great-docs/user_guide/freeze.html) feature (added by [posit-dev/great-docs#158](https://github.com/posit-dev/great-docs/pull/158) and released in `great-docs>=0.11.0`), which we configure project-wide:

- [`great-docs.yml`](../../great-docs.yml) sets `freeze: true`, applying to every executable `.qmd` page (examples and user guide).
- The committed `_freeze/` directory at the repo root stores Quarto's cached cell outputs.
- On every `great-docs build`, a built-in pre-render step copies `_freeze/` into the (otherwise wiped) build directory before `quarto render` runs, so Quarto reads the cache instead of spawning a kernel.

`true` (rather than `auto`) is deliberate: `auto` re-executes a page when its source hash drifts from the cached hash, which would let CI fire up a Jupyter kernel for any contributor edit that wasn't paired with a `great-docs freeze` run. `true` makes the rule absolute — *local builds execute and freeze; remote builds only render* — and matches upstream's documented use case for non-deterministic / locally-runnable / draw-stable pages.

**Refresh workflow.** After editing an executable page, or after a pathmc API change that affects any rendered output:

```bash
uv run great-docs freeze docs/examples/my_page.qmd      # or multiple paths
git add _freeze/
git commit -m "Refresh freeze cache for my_page"
```

`great-docs freeze --info` shows per-page cache status. `great-docs freeze --clean <pages>` wipes and regenerates specific entries (useful after a dependency upgrade that changes plot styling).

**Local previews show stale outputs by design.** With `freeze: true`, `great-docs preview` displays the last-frozen output until you re-run `great-docs freeze`. This is the trade-off for CI determinism — explicit refresh, never accidental execution.

**Version pin.** [`pyproject.toml`](../../pyproject.toml)'s `docs` extra requires `great-docs>=0.11.0`, the first PyPI release with the freeze workflow we depend on. This keeps `pip install pathmc[docs]` on tagged releases while allowing compatible Great Docs updates.

## Goals

- Auto-generated API reference for the public surface (`model`, `simulate`, `add_lags`, plus `PathModel` and its methods).
- Preserve all hand-authored narrative content (`how-it-works`, `comparison`, the `concepts/` user guide, and the `examples/` notebook gallery).
- Publish `llms.txt`, `llms-full.txt`, and a curated `SKILL.md` so downstream agents can reason about pathmc without scraping HTML.
- One-command deployment to GitHub Pages on every push to `main`.
- Zero or near-zero edits to existing `.qmd` content.

## Non-goals

- Rewriting prose content. Reorganization and small frontmatter additions only.
- Switching docstring conventions. The current style stays; we just configure Great Docs' parser to match.
- Latent-variable / SEM scope changes (separate roadmap item).
- Changing the package's public API to make it docs-friendly (the API is already small and stable post-v1).

## Current state

```
pathmc/
├── docs/
│   ├── _quarto.yml              ← hand-written Quarto site config
│   ├── index.qmd                ← landing page (executable)
│   ├── how-it-works.qmd         ← architecture overview
│   ├── comparison.qmd           ← comparison with other packages
│   ├── concepts/                ← 8 conceptual pages
│   │   ├── bayesian_workflow.qmd
│   │   ├── model_specification.qmd
│   │   ├── transforms_families.qmd
│   │   ├── causal_inference.qmd
│   │   ├── estimation_approaches.qmd
│   │   ├── panel_data.qmd
│   │   ├── panel_interventions.qmd
│   │   └── standardized_effects.qmd
│   ├── examples/                ← 16 worked examples (.qmd, listing-driven)
│   ├── assets/logo.png
│   ├── references.bib
│   └── dev/                     ← agent + dev notes (excluded from render)
├── pathmc/
│   └── __init__.py              ← exports: model, simulate, add_lags, Prior
└── pyproject.toml               ← name=pathmc, requires-python>=3.11
```

### Public API to document

From `pathmc/__init__.py`:

| Symbol     | Source                          | Notes                                                              |
| ---------- | ------------------------------- | ------------------------------------------------------------------ |
| `model`    | `pathmc.model.model`            | Entry point. Returns a `PathModel` instance.                       |
| `simulate` | `pathmc.model.simulate`         | Standalone simulation helper.                                      |
| `add_lags` | `pathmc.panel.add_lags`         | Panel-data helper.                                                 |
| `Prior`    | `pymc_extras.prior.Prior`       | **Re-exported from upstream.** Should be excluded from auto-docs.  |

`PathModel` (in `pathmc/model.py`) is not in `__all__` but is the class returned by `model()`. Its methods (`fit`, `effects_summary`, `ate`, `do`, `adjustment_sets`, etc.) are the bulk of the user-facing surface and must appear in the reference.

## Target state

```
pathmc/
├── great-docs.yml               ← NEW — single config file (committed)
├── great-docs/                  ← NEW — build directory (gitignored)
├── docs/                        ← retained, repurposed
│   ├── how-it-works.qmd         ← stays as user-guide page
│   ├── comparison.qmd           ← stays as user-guide page
│   ├── concepts/                ← becomes user-guide content
│   ├── examples/                ← becomes a custom section
│   ├── assets/logo.png
│   ├── references.bib
│   └── dev/                     ← unchanged (still excluded)
├── skills/
│   └── pathmc/
│       └── SKILL.md             ← NEW — curated downstream agent skill
└── .github/workflows/
    └── docs.yml                 ← NEW — generated by great-docs setup-github-pages
```

### Removed
- `docs/_quarto.yml` — replaced by `great-docs.yml` and a generated `great-docs/_quarto.yml`.
- `docs/index.qmd` as a separate landing page — replaced by Great Docs' README-driven homepage (or kept as the user-guide first page; see Phase 1 decision below).

## Mapping of existing content

| Current path                                | Great Docs slot         | Frontmatter changes                   |
| ------------------------------------------- | ----------------------- | ------------------------------------- |
| `docs/index.qmd`                            | Homepage / README hero  | Source of truth becomes `README.md`   |
| `docs/how-it-works.qmd`                     | User guide              | Add `guide-section: Overview`, prefix |
| `docs/comparison.qmd`                       | User guide              | Add `guide-section: Overview`, prefix |
| `docs/concepts/*.qmd`                       | User guide              | Add `guide-section: Concepts`, prefix |
| `docs/examples/*.qmd` + `examples/index.qmd`| Custom section          | Section auto-indexes; drop `listing:` |
| `docs/references.bib`                       | Same path, referenced from `great-docs.yml` | none |
| `docs/assets/logo.png`                      | Auto-detected as logo   | none                                  |

**Decision deferred to Phase 1**: should the homepage be derived from `README.md` (`homepage: index`) or from the first user-guide page (`homepage: user_guide`)? Current `docs/index.qmd` has an executable code cell demonstrating mediation analysis. We'll prototype both and pick on look.

## Phased plan

### Phase 0 — branch and skill (this commit)

- [x] Create branch `docs/great-docs-migration`.
- [x] Install Great Docs' own agent skill at `~/.cursor/skills/great-docs/` (user-level, done out-of-band before this PR).
- [x] Write this migration plan.

### Phase 1 — non-destructive spike

Goal: prove Great Docs can produce a usable site for pathmc without deleting or moving anything yet.

```bash
conda activate pathmc
pip install great-docs
quarto --version          # must be installed; install via brew if missing
great-docs scan --verbose # preview discoverable API (no files written)
great-docs init           # writes great-docs.yml + initial reference structure
great-docs build          # full build to great-docs/_site/
great-docs preview        # localhost:3000
```

Artifacts to inspect:
- The auto-generated `great-docs.yml` — which sections it picked, which symbols it discovered, what parser it auto-detected.
- The rendered API reference for `model`, `PathModel`, `simulate`, `add_lags`.
- Whether `Prior` was excluded automatically (it should not appear; we'll add it to `exclude:` if it does).
- Whether `dynamic: true` succeeds. If imports fail in CI, fall back to `dynamic: false`.

Add to `.gitignore`:
```
great-docs/
```

Decision points to resolve in this phase:
1. `homepage: index` vs `homepage: user_guide`.
2. Keep `docs/` as-is and point Great Docs at it, or move `concepts/` → `user_guide/` and `examples/` → `examples/` at the repo root. **Strong default: keep `docs/` to minimize churn and preserve internal links.**
3. Pick a `navbar_style` (e.g. `mint`, `lilac`, `slate`) consistent with the logo.

Exit criteria: `great-docs build` produces a site that, on visual inspection, is at least as informative as the current Quarto site, with a working API reference section.

### Phase 2 — content migration

This phase rewires `great-docs.yml` to match pathmc's actual structure.

Initial `great-docs.yml` sketch (refined during Phase 1):

```yaml
module: pathmc
display_name: pathmc
parser: numpy             # confirm via great-docs init auto-detection
dynamic: true             # fall back to false if import fails

exclude:
  - Prior                 # re-exported from pymc_extras

# Keep all narrative content under docs/ to minimize churn
user_guide: docs/concepts # confirm path option works; otherwise restructure

sections:
  - title: Overview
    dir: docs              # how-it-works.qmd, comparison.qmd
  - title: Examples
    dir: docs/examples
    navbar_after: User Guide

reference:
  - title: Modeling
    contents:
      - model
      - PathModel          # explicit despite not being in __all__
      - simulate
  - title: Panel data
    contents:
      - add_lags

navbar_style: mint         # tentative
dark_mode_toggle: true

bibliography: docs/references.bib  # if supported; else move .bib

logo: docs/assets/logo.png

skill:
  enabled: true
  file: skills/pathmc/SKILL.md     # curated, see Phase 3
```

Per-page changes:
- Add `guide-section:` to user-guide `.qmd` files for sidebar grouping.
- Numeric prefixes (`00-`, `01-`, …) to enforce ordering. **Lazy approach**: use a flat `order:` field in frontmatter if the version of Great Docs supports it; otherwise rename. The skill warns that prefixes are required for "deterministic order".
- Drop the `listing:` block from `examples/index.qmd` — Great Docs custom sections render their own grid index.

Verification after each batch:
```bash
great-docs build --no-refresh   # fast: skip API rediscovery
```

Clear `docs/_freeze` once when switching from the old build path so cached notebook outputs don't go stale (per `AGENTS.md`).

### Phase 3 — curated agent skill

Auto-generated skills are derived from docstrings only; they cannot encode gotchas or a decision table. Hand-write `skills/pathmc/SKILL.md` covering at minimum:

- **Frontmatter**: `name: pathmc`, description that says *what* and *when*, `compatibility: Requires Python >=3.11, PyMC >=5.22.0`.
- **Decision table**: e.g. *"Estimate ATE → `m.ate(Y, X, values=(0,1))`"*, *"Check identification → `m.adjustment_sets(X, Y)`"*, *"Add lags → `pathmc.add_lags(df, ...)`"*.
- **Gotchas**:
  1. `model()` returns a `PathModel`, not a fitted result. Call `.fit()`.
  2. `do()` performs structural intervention via `pm.do()` graph surgery — it is **not** conditioning. `m.ate()` and `m.cate()` are the user-facing wrappers.
  3. The formula DSL is lavaan-inspired but not a 1:1 reimplementation. Refer to the user guide for supported operators.
  4. Latent variables / SEM measurement models are out of scope in v0.1.
  5. Panel data uses `add_lags()` to create lagged columns *before* `model()`; see panel pages.
  6. `Prior` is re-exported from `pymc_extras` for convenience; the canonical reference is upstream.
- **Capabilities and boundaries**: agents can write specs, configure priors, run `fit()`, query `ate`/`cate`/`adjustment_sets`. They cannot (yet) define latent variables or measurement models.
- **Resources**: links to `llms.txt`, `llms-full.txt`, and the docs site.

Cap at ~300 lines per the spec.

### Phase 4a — CI workflow scaffolding (lands with this PR, but disabled)

The repo is **private** and we are **not yet ready to make the documentation public**. Even setting that aside, see "CI build is disabled — why" above for why we cannot build docs on CI today: great-docs has no way to persist Quarto's freeze cache across builds, and re-executing 18 MCMC notebooks on every PR is unworkable.

```bash
great-docs setup-github-pages --python-version 3.12 --main-branch main
```

This generates `.github/workflows/docs.yml`. We modify it before committing:

- **`build-docs`** — enabled; renders from the committed `_freeze/` cache and uploads `docs-html` as a PR artifact.
- **`publish-docs`** — `if: false && github.ref == 'refs/heads/main'`. Re-enable in Phase 4b along with `Settings → Pages → Source → GitHub Actions`.
- **`preview-docs`** — removed (#226). great-docs 0.10.0 generated this job incomplete (it started a `bobheadxi/deployments` deployment but never uploaded or finished). PR reviewers download the `docs-html` artifact from the `Build Docs` job instead.

### Phase 4b — public deployment (deferred to launch day, **not part of this PR**)

When ready to announce:

1. **Re-enable the deploy job** in `.github/workflows/docs.yml` (remove the `if: false` guard or restore the deleted steps). One-line PR.
2. **Configure repo Settings on GitHub**: Settings → Pages → Source → **GitHub Actions**.
3. **Add `Documentation` URL to `pyproject.toml`** so source links and the homepage canonical URL resolve correctly:
   ```toml
   [project.urls]
   Documentation = "https://pymc-labs.github.io/pathmc/"
   Repository    = "https://github.com/pymc-labs/pathmc"
   ```
4. **Trigger the workflow** (push or manual `workflow_dispatch`) and verify the site is live at `https://pymc-labs.github.io/pathmc/`.
5. **Verify visibility**: open the site URL in an incognito window. With repo = private + Pages source = GitHub Actions on a Team-plan org, the *site* is publicly readable; the *repo* remains private. This is the desired end state.

Optional intermediate state for stakeholder review (Team plan supports private GitHub Pages): if you want to dogfood the rendered site internally before launch, you can briefly enable Pages with **Private** visibility (Settings → Pages → Visibility) — only org members will see it. Flip back to Public on launch day. This is strictly optional.

### Phase 5 — cleanup

- Delete `docs/_quarto.yml`.
- Delete `docs/_freeze/` and `docs/.quarto/` (one-time stale-cache flush).
- Delete `docs/dev/pr-summaries/` only if confirmed unreferenced.
- Update `AGENTS.md`:
  - "Quarto freeze cache" section becomes "Great Docs build cache" with the new path (`great-docs/`).
  - The `docs/_quarto.yml` reference disappears.
- Add a `docs` extra to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  docs = ["great-docs"]
  ```
- Verify `dev` extras are still complete; consider `dev = [..., "great-docs"]` for convenience.

## Risks and mitigations

| Risk                                                          | Mitigation                                                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `dynamic: true` fails because pathmc imports PyMC at module-level (slow / heavy) | Set `dynamic: false` and rely on griffe AST analysis. Documented in skill as a fallback.       |
| Auto-discovery picks up `Prior` and other re-exports          | Use `exclude:` (and possibly `auto_include: []`) per the config reference.                       |
| `PathModel` methods don't render because the class isn't in `__all__` | Explicitly list `PathModel` in `reference.contents`; verify members render (they should).      |
| `examples/` notebooks have long execution times in CI         | `freeze: auto` is preserved; first CI build will be slow but subsequent builds reuse cache.      |
| Numeric-prefix file rename breaks external links              | Add Quarto redirect rules or check no external sites link to `concepts/<name>.html` first.       |
| Bibliography path                                             | Confirm Great Docs honours `bibliography:` in frontmatter or `great-docs.yml`; otherwise inline. |
| Per-PR docs preview deploys                     | Removed the incomplete `preview-docs` job (#226); reviewers use the `docs-html` CI artifact.      |
| Accidentally publishing the site before launch                | Phase 4a explicitly disables the deploy step and leaves `Settings → Pages` unconfigured. Phase 4b is the only path to a public URL and is intentionally a separate, manually-triggered change. |

## Acceptance criteria

A reviewer can verify the migration PR is merge-ready by checking:

1. `great-docs build` completes locally with no errors against the branch's `pathmc` package.
2. The rendered site at `great-docs/_site/` (locally **and** as the CI artifact) includes:
   - A landing page (homepage).
   - A reference section listing `model`, `PathModel`, `simulate`, `add_lags`, with rendered docstrings and source links.
   - A user-guide section covering all pages currently under `docs/concepts/`, plus `how-it-works` and `comparison`.
   - An "Examples" section listing all 16 example pages.
   - `llms.txt` and `llms-full.txt` at the site root.
   - `skill.md` at the site root, sourced from `skills/pathmc/SKILL.md`.
3. `docs/_quarto.yml` is deleted.
4. `.github/workflows/docs.yml` exists, runs green on the PR, **and contains no active deploy step**. The deploy step is either commented out, guarded by `if: false`, or absent — Phase 4b will re-enable it.
5. `AGENTS.md` is updated to reflect the new build commands.
6. **No GitHub Pages site is live.** Visiting `https://pymc-labs.github.io/pathmc/` returns 404 (or the previous content if any). The only way to view the rendered site is to download the workflow artifact or build locally.

Phase 4b is **out of scope for this PR's acceptance**. It is tracked as a separate launch-day step (one-line workflow change + repo settings flip).

## Out-of-scope follow-ups

Captured here so they don't block the migration:

- File a bug against `posit-dev/great-docs` re: `npx skills add` failing because `/.well-known/skills/index.json` is not published.
- Consider versioned docs (Great Docs supports multi-version) once the API starts changing post-0.1.
- Hero section + animated navbar — defer until 0.1.0 is released and the homepage copy is finalized.
- API diff / breaking-change detection between releases — Great Docs has this built in; enable on the next release.

## References

- Issue: [#175 — Get docs ready for live docs site](https://github.com/pymc-labs/pathmc/issues/175)
- Great Docs site: <https://posit-dev.github.io/great-docs/>
- Great Docs intro blog post: <https://opensource.posit.co/blog/2026-04-15_great-docs-introduction/>
- Local agent skill: `~/.cursor/skills/great-docs/SKILL.md` (user-level)
- Agent Skills spec: <https://agentskills.io/specification>
