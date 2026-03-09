# Homepage Revision — Brainstorm

Ideas for revising `docs/index.qmd` to better sell pathmc and create excitement.

## The core pitch

pathmc is a **causal orchestration layer around PyMC**. PyMC gives you the engine (MCMC, graph surgery, observe/do). pathmc gives you the steering wheel:

- Write your causal assumptions as equations
- pathmc compiles them into a correct generative model
- Ask causal questions and get answers with full Bayesian uncertainty

The one-liner: **"Specify your DAG. Fit it. Ask 'what if?'"**

## Selling angles

### 1. The gap it fills

PyMC is widely used. People know it *can* do causal inference. But building a multi-equation generative model by hand — wiring structural parents correctly, respecting topological order, handling different likelihoods, getting `pm.do()` and `pm.observe()` to work on the same model — is expert-level work that takes dozens of lines of careful PyMC code. pathmc reduces it to a spec string and one function call. The target audience is **PyMC users who want to do causal inference but don't want to hand-wire generative models**.

### 2. "One spec, five capabilities"

A single spec string unlocks: estimation, introspection, causal intervention, identification checking, and sensitivity analysis. This is the "wow" factor — you write 3 lines of equations and get a full causal analysis toolkit. The homepage should make this tangible with a code example that shows the journey from spec to causal insight in under 20 lines.

### 3. Full posterior uncertainty on causal effects

Most causal inference tools give you point estimates (maybe with bootstrap CIs). pathmc gives you the full posterior distribution on every causal quantity — ATE, CATE, probability queries, path-specific effects. This is a genuine differentiator for anyone who cares about decision-making under uncertainty. Could lead with a visual: a posterior distribution over an ATE, not just a number.

### 4. The guardrails

pathmc doesn't just let you compute causal effects — it warns you when you shouldn't. Identification checks tell you if your effect is even estimable. Collider warnings catch a common mistake. Implied independence tests check if your DAG is consistent with the data. Sensitivity analysis tells you how fragile your conclusions are. This is the "responsible causal inference" angle — the package helps you avoid overconfident claims.

### 5. The applied use cases

The examples gallery shows pathmc solving real industry problems:
- **Marketing mix modeling** (adstock, saturation, geo-level panels)
- **SaaS funnel analysis** (binary outcomes, conversion paths)
- **Vaccine surrogate endpoints** (mediation, do-operator)
- **Dynamic pricing** (intervention simulation)
- **Difference-in-differences** (panel data, quasi-experiments)

The homepage could highlight 3-4 of these to show breadth. The message: this isn't a toy for textbook examples — it solves real problems.

### 6. "lavaan for Python, but Bayesian and causal"

For the SEM/path analysis community (coming from R's lavaan or Python's semopy): pathmc speaks their language (the `~`, `~~`, `:=` syntax) but adds Bayesian inference and a do-operator. This is a compelling upgrade path. The homepage could have a small "Coming from lavaan?" callout.

### 7. The "one object" API

`pathmc.fit()` returns one object. No pipeline. No separate identify-then-estimate workflow. One object with methods for everything: inspect, sample, query, diagnose. This is a developer experience selling point — it's simple and discoverable.

## Structural ideas for the homepage

### Option A: Hero example + feature grid

1. **Hero section**: Logo + one-sentence pitch + a single code example showing spec → fit → ATE in 10 lines
2. **Feature grid**: 4-6 cards with icons/bold titles: "Causal Queries", "Bayesian Uncertainty", "Identification Checks", "Panel Data", "Introspection", "Sensitivity Analysis"
3. **Use case highlights**: 3 cards linking to example notebooks (MMM, mediation, quasi-experiments)
4. **Getting started**: links to concepts and quickstart

### Option B: Problem → solution narrative

1. **The problem**: "PyMC can do causal inference, but wiring multi-equation generative models by hand is hard and error-prone"
2. **The solution**: pathmc compiles your DAG into a correct PyMC model. Show before/after: 40 lines of PyMC → 5 lines of pathmc
3. **What you get**: the four capability bullets (from the Overview page)
4. **See it in action**: 2-3 annotated code blocks showing the journey from spec to causal insight
5. **Getting started**: links

### Option C: "Three things in one" positioning

1. **Pitch**: "pathmc combines three things that are usually separate: (1) a formula language for causal assumptions, (2) full Bayesian inference via PyMC, (3) interventional simulation via the do-operator"
2. **Code example**: show each of the three in action
3. **When to use pathmc**: short positioning table (vs DoWhy, CausalPy, semopy)
4. **Gallery**: curated example links

## Content to carry forward

The current homepage has good bones:
- The code example is clear and shows the core workflow
- The feature bullets are accurate
- The "Getting started" links are useful

What's missing:
- **No emotional hook** — it reads like a README, not a landing page
- **No "why should I care"** — the PyMC positioning (the gap it fills) isn't stated
- **No visuals** — no DAG render, no posterior plot, no equation display
- **No breadth** — the code example shows mediation, but doesn't hint at the range of use cases (MMM, panel data, binary outcomes, quasi-experiments)
- **Features list is flat** — doesn't convey the depth of each capability

## Possible taglines

- "Specify your DAG. Fit it. Ask 'what if?'"
- "Bayesian path analysis with a built-in do-operator"
- "From causal assumptions to causal answers — in PyMC"
- "The causal layer for PyMC"
- "Write equations. Get posteriors. Simulate interventions."
- "Structural causal models, made simple"

## Key decisions before implementing

1. **Tone**: Technical-academic (current) vs. developer-marketing (more energy)?
2. **Length**: Minimal landing page (push detail to Overview) vs. self-contained pitch?
3. **Visuals**: Include rendered DAG / equation / posterior plot on the homepage? (Requires executable code blocks)
4. **Before/after**: Show the PyMC code that pathmc replaces? (Powerful but long)
5. **Target audience emphasis**: PyMC users? R/lavaan migrants? Causal inference practitioners? All three?
