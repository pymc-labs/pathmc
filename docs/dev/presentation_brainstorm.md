# Presentation Brainstorm: Introducing pathmc to PyMC Labs

> Three candidate structures for the Monday presentation, with feedback on strengths, risks, and a recommendation.

## Audience context

The PyMC Labs team knows PyMC deeply, ships Bambi and CausalPy, and thinks in terms of ecosystem coherence. They will immediately ask:

- What does this do that we can't already do with PyMC + Bambi + CausalPy?
- Where does it sit in the ecosystem — complement or competitor?
- Is the implementation sound? (They'll want to see real code, not just slides.)
- What's the adoption story? Who would use this and why?

They are technically sophisticated but also commercially minded (PyMC Labs is a consultancy). Features that unlock new consulting use cases or strengthen the ecosystem story will resonate.

---

## Approach A: Problem-first ("The gap in the ecosystem")

### Structure

1. **The two roads to causal inference in Python** (~5 min)
   - Road 1: Regression adjustment. You have a DAG in your head, you identify the right adjustment set, and you fit a single regression with Bambi. This works well — Bambi is excellent. But the DAG stays informal. You have to manually translate DAG → formula, manually check identification, and manually do g-computation (copy data, set treatment, predict, diff).
   - Road 2: Structural causal models. You write out the full system of structural equations and fit it as a generative model. This is Pearl's SCM framework — it naturally supports `do()`, mediation decomposition, multi-equation systems, and formal identification. But in Python today, this means writing raw PyMC code: wiring equations in topological order, handling multiple families, managing residual covariance, implementing `do()` by hand. Expert-level work.
   - The gap: there is no ergonomic way to go from "I have a DAG" to "I have a fitted structural causal model with `do()` and identification checks" without writing a lot of PyMC boilerplate.

2. **pathmc fills the gap** (~3 min)
   - One spec string → generative PyMC model → estimation, introspection, causal queries, identification, sensitivity. Show the mediation example from the homepage: 6 lines of spec, one `model()` call, then `effects_summary()`, `ate()`, `adjustment_sets()`.
   - Architecture slide: spec → parser → NetworkX DAG → PyMC generative model → `pm.observe()` for estimation, `pm.do()` for intervention.

3. **Live walkthrough of features** (~15 min)
   - Walk through 2–3 examples from the docs site, screen-sharing the rendered Quarto pages. Candidates:
     - **Mediation** — the "hello world." DAG, equations, labeled effects, `indirect := a*b`, `ate()`.
     - **Seeing vs Doing** or **Collider Bias** — shows that pathmc makes the distinction between conditioning and intervening concrete and computational, not just conceptual.
     - **MMM with transforms** — the flagship applied example. Adstock, saturation, panel mode, time-forward `do()` for scenario simulation. This directly relates to PyMC-Marketing's domain and shows the structural advantage.

4. **Ecosystem positioning** (~5 min)
   - Show the comparison table (pathmc vs DoWhy vs CausalPy vs semopy vs EconML vs Bambi).
   - Emphasize complementarity: CausalPy owns quasi-experimental designs, Bambi owns flexible single-equation regression, pathmc owns multi-equation structural causal models.
   - Built on PyMC — uses `pm.do()`, `pm.observe()`, ArviZ, pytensor scan. Not a fork or reimplementation; a compilation layer.

5. **What's next / how you can help** (~2 min)
   - Post-v1 roadmap items: d-separation diagnostics, sensitivity analysis improvements, DAG comparison, policy optimization.
   - Open questions: naming, packaging, where it lives in the org.

### Strengths

- **Intellectually compelling.** The "two roads" framing is clean and immediately positions pathmc as filling a real gap rather than duplicating existing tools. This is critical for an audience that builds the existing tools.
- **Ecosystem-respectful.** By starting with "Bambi is great for Road 1," you avoid any perception of competing with their own products.
- **Builds tension before the reveal.** The audience feels the pain of the gap before seeing the solution.

### Risks

- **Slow start.** Five minutes of setup before showing any code. PyMC Labs folks may already know the landscape and get impatient.
- **Can feel like a pitch.** The gap-filling framing can come across as trying to sell them on a problem they may not feel. If anyone thinks "we already handle that fine with Bambi + manual PyMC," the whole structure weakens.
- **Hard to calibrate the gap description.** If you overstate the difficulty of writing raw PyMC structural models, someone who does it regularly may push back. If you understate it, the motivation for pathmc weakens.

---

## Approach B: Demo-first ("Let me show you something")

### Structure

1. **Cold open with live code** (~8 min)
   - No slides. Open a notebook (or the rendered docs). "I've been building something and I want to show you what it does."
   - Type or reveal the mediation spec. Call `model()`. Show `graph()`, `equations()`, `effects_summary()`. Call `ate()`. Call `adjustment_sets()`. Call `sensitivity()`.
   - Let the output speak. The audience sees: from 6 lines of spec, you get a causal DAG, LaTeX equations, posterior summaries, treatment effects, identification, and sensitivity — all from one object.

2. **"How does this work?"** (~5 min)
   - Now that they've seen the outputs, explain the architecture: parser → graph → compiler → PathModel. Emphasize that it compiles to a standard `pm.Model` — they can inspect it with `pm.model_to_graphviz()`, sample with standard PyMC, get ArviZ InferenceData.
   - Show that `do()` uses `pm.do()` under the hood — not a reimplementation but a compilation target.

3. **Applied example** (~10 min)
   - Walk through the MMM example (or another applied example) from the docs site, screen-sharing the rendered notebook. Show adstock/saturation transforms, panel mode, time-forward `do()` for counterfactual spend scenarios.
   - This demonstrates that pathmc isn't a toy — it handles real problems at the complexity level PyMC Labs consultants encounter.

4. **Where it fits** (~5 min)
   - Brief ecosystem positioning: complement to Bambi (single eq) and CausalPy (quasi-experimental). pathmc fills the multi-equation structural modeling niche.
   - Comparison table if time allows, but by this point the demo has already made the case.

5. **Discussion** (~remaining time)
   - Open it up. What questions do you have? Where could this fit? What would you change?

### Strengths

- **Immediate engagement.** No throat-clearing. The audience sees working code in the first minute and can form their own impressions before you frame the narrative.
- **Authenticity.** "Let me show you something I built" is the most natural register for an internal presentation. It signals confidence without salesmanship.
- **Lets the product sell itself.** If the API is as clean as it looks (and it is — `model.ate("Y", "X", values=(0, 1))` is hard to argue with), the demo does the positioning work for you.
- **Invites engagement.** People can interrupt with questions ("what happens if you add a collider?") and you can answer live by modifying the spec.

### Risks

- **No framing.** If the audience doesn't immediately grasp why this matters, the demo can feel like "cool hack, but why?" You're relying on them to connect the dots to their own work.
- **Technical hiccups.** Live code is risky. MCMC sampling takes time. The rendered docs are safer but less interactive.
- **Harder to control time.** Without a rigid structure, you might spend too long on one example and rush the ecosystem positioning.

### Mitigation

Use the rendered Quarto docs rather than live Jupyter. You get the "code + output" experience without MCMC wait times. You can always drop into a live notebook for one interactive moment if someone asks a good question.

---

## Approach C: Story-driven ("One DAG, three questions")

### Structure

1. **Start with a concrete scenario** (~3 min)
   - "Imagine you're a data scientist at a consumer brand. Marketing spends across TV, digital, and print. You want to know: (a) what's the ROI of each channel? (b) what would happen if we shifted 20% of TV budget to digital? (c) how confident should we be in these answers given possible unmeasured confounders?"
   - These three questions map to estimation, intervention, and sensitivity — pathmc's three core capabilities.

2. **Show how you'd answer today** (~5 min)
   - Question (a): Fit a regression with Bambi or PyMC-Marketing. Works, but you get coefficients, not causal effects. For a structural model you'd need to write raw PyMC.
   - Question (b): With Bambi, you'd do manual g-computation. With PyMC-Marketing, you'd use the budget optimizer, but it's a black box — you can't see the DAG or the structural equations. With raw PyMC, you'd implement `do()` by hand.
   - Question (c): No existing PyMC ecosystem tool does this directly.

3. **Now with pathmc** (~12 min)
   - Walk through the MMM example (or a simplified version). Write the spec, fit, show the DAG, show the equations. Answer question (a) with `effects_summary()`. Answer question (b) with `do(set={"tv_spend": new_value}, simulate_over="time")`. Answer question (c) with `sensitivity()`.
   - Then zoom out: show that the same workflow applies to mediation, SaaS funnels, vaccine surrogates, dynamic pricing. It's not an MMM tool — it's a structural causal modeling tool that happens to do MMM well.

4. **The bigger picture** (~5 min)
   - pathmc as the SCM layer in the PyMC ecosystem. The comparison table. Complementarity with Bambi and CausalPy.
   - 28 worked examples in the docs. The full feature list: 5 families, panel mode, transforms, identification, sensitivity, standardized effects, implied independence tests.

5. **Discussion**

### Strengths

- **Concrete and relatable.** Starting with a business problem that PyMC Labs consultants actually face makes the tool immediately relevant. "I could use this on my next engagement" is the reaction you want.
- **Shows the full loop.** Estimation → intervention → sensitivity in one workflow. This is pathmc's unique selling point and the story structure mirrors it naturally.
- **Anchors in MMM.** PyMC Labs has deep MMM expertise (PyMC-Marketing). Showing that pathmc can do structural MMM — with the DAG visible, with mediation through brand awareness, with `do()` for scenario planning — positions it as a complement that adds causal transparency to the MMM workflow.
- **Natural escalation.** Start with one applied domain, then zoom out to show generality.

### Risks

- **MMM-heavy framing could pigeonhole.** If you lead with MMM, the audience might think "this is an MMM tool" rather than "this is a general structural causal modeling tool." You'd need to zoom out convincingly in section 4.
- **Comparison with PyMC-Marketing could be sensitive.** PyMC-Marketing is a commercial product. Showing that pathmc does something similar (but more transparent) could create tension. Frame it carefully as complementary, not competitive.
- **Less time for the breadth of examples.** The story-driven approach spends more time on one use case and less on the full gallery. The audience might not appreciate the range (28 examples, 5 families, panel mode, etc.) unless you explicitly call it out.

---

## Recommendation

**Lead with Approach B (demo-first), borrow the ecosystem framing from A, and use one applied story from C.**

Here's why:

1. **This audience doesn't need convincing that causal inference matters.** They already know. They don't need 5 minutes on "the two roads" — they've walked both. A demo respects their expertise by showing rather than telling.

2. **The API is genuinely impressive.** The jump from a 6-line spec to `graph()` + `equations()` + `ate()` + `adjustment_sets()` + `sensitivity()` is a "wow" moment that works best when experienced firsthand, not described in bullet points.

3. **Ecosystem framing matters, but later.** After the demo, the audience will naturally ask "how does this relate to Bambi/CausalPy?" — and you'll have a crisp answer ready. Answering a question they're already thinking is more persuasive than preemptively framing.

4. **One applied example grounds it.** After the foundational demo (mediation or seeing-vs-doing), walk through the MMM example to show real-world depth. This addresses "is it a toy?" and connects to PyMC Labs' consulting domain.

### Suggested flow

| Time | Section | Content |
|------|---------|---------|
| 0–2 min | Setup | "I've been building something. Let me show you what it does." Brief context: Bayesian structural causal models, lavaan-inspired DSL, compiles to PyMC. |
| 2–10 min | Core demo | Mediation example from the docs. `model()` → `graph()` → `equations()` → `effects_summary()` → `ate()` → `adjustment_sets()`. Screen-share the rendered Quarto page. |
| 10–12 min | Architecture | How it works: spec → parser → NetworkX DAG → generative `pm.Model`. Uses `pm.do()` and `pm.observe()` — not a reimplementation. Show the architecture diagram from `how-it-works.qmd`. |
| 12–22 min | Applied depth | Walk through the MMM example (or another applied example). Adstock, saturation, panel mode, hierarchical effects, time-forward `do()`. Optionally detour to collider bias or sensitivity to show the guardrails. |
| 22–25 min | Ecosystem fit | Brief comparison table. pathmc = multi-equation SCM with `do()`. Bambi = single-equation regression. CausalPy = quasi-experimental. Complementary, not competitive. Built entirely on PyMC. |
| 25–28 min | Scope & roadmap | 28 examples, 5 families, panel mode, transforms, identification, sensitivity. Post-v1: d-separation diagnostics, DAG comparison, policy optimization. |
| 28–30 min | Open discussion | What questions do you have? Where could this fit in the ecosystem? |

### Presentation format

The Quarto docs site is your best asset. Rather than building a slide deck, screen-share the rendered docs:

- **Homepage** for the elevator pitch and code snippet.
- **How It Works** page for the architecture diagram and the estimation/intervention duality.
- **Example notebooks** for live walkthroughs — they have code, output, DAG renders, equations, and narrative all in one place.
- **Comparison page** for the ecosystem positioning table.

If you want a few slides for transitions (title slide, agenda, "where it fits" diagram), Quarto revealjs is the natural choice — you can embed executable Python in the slides and reuse your existing notebooks. But honestly, walking through the docs site may be more impressive: "here's the documentation that already exists" signals maturity and seriousness.

### Examples to prioritize for the demo

Ranked by demo impact for this audience:

1. **Mediation** — the "hello world." Everyone understands it. Shows the core API surface in 2 minutes. Also naturally shows `:=` defined parameters and `effect("X -> M -> Y")`.

2. **Seeing vs Doing** or **Collider Bias** — pedagogically powerful. Shows that pathmc makes the conditioning-vs-intervention distinction computational, not just theoretical. The `collider_warnings()` output is a concrete differentiator.

3. **MMM with transforms** — the applied flagship. Adstock, saturation, panel mode, `do(simulate_over="time")`. This connects directly to PyMC Labs' MMM consulting practice and shows that pathmc handles real complexity. Also a natural place to show the `comparison.qmd` positioning vs PyMC-Marketing.

4. **DAG testing** (`test_implications()`) — unique feature that no other PyMC ecosystem tool offers. Quick to demo and immediately useful.

5. **Sensitivity analysis** — rounds out the "responsible causal inference" story. Quick to show and directly answers "how robust is this?"

You probably have time for 2–3 of these in detail plus a quick flash of 1–2 others.

### Key messages to land

Regardless of which approach you use, these are the points that need to come through:

1. **The DAG is the model.** In pathmc, you type your causal assumptions as equations — and those assumptions directly become the generative model. There is no translation step where assumptions can get lost.

2. **One spec, five capabilities.** Introspection, estimation, causal queries, identification, sensitivity — all from one `model()` call. No context-switching between packages.

3. **Built on PyMC, not beside it.** pathmc compiles to `pm.Model`, uses `pm.do()` and `pm.observe()`, returns ArviZ `InferenceData`. It's a compilation layer, not a fork. The underlying model is a standard PyMC model that anyone can inspect and extend.

4. **Complementary to Bambi and CausalPy.** Bambi excels at single-equation regression with rich random effects. CausalPy excels at quasi-experimental designs. pathmc excels at multi-equation structural causal models with the `do()` operator. Different tools for different questions.

5. **Not a toy.** 28 worked examples, 5 likelihood families, panel mode with hierarchical effects, transforms with estimable parameters, implied independence tests, sensitivity analysis. This is a complete v1 with documentation.

### Potential tough questions and how to handle them

**"How does this compare to PyMC-Marketing for MMM?"**
Complementary. PyMC-Marketing is a turnkey solution optimized for the common MMM workflow. pathmc lets you specify the causal structure explicitly — which is valuable when you want to model mediation (brand awareness → search → sales), test structural assumptions, or do custom scenario simulation. A consultant might use PyMC-Marketing for standard MMM and pathmc when the client needs structural transparency.

**"Why not just write PyMC directly?"**
You can, and for one-off models you probably should. pathmc is for when you want to iterate rapidly across candidate DAGs, get identification checks and sensitivity analysis for free, and have a clean `do()` operator that propagates through the system. It's the difference between writing raw SQL and using an ORM — both have their place.

**"What about latent variables / full SEM?"**
Out of scope for v1. pathmc is observed-variable path analysis with support for deterministic latent mediators. Full latent factor models (CFA/SEM) are not planned. This is a deliberate scope decision, not a limitation — full SEM is a different beast, and semopy/lavaan handle it.

**"What about scale? Can this handle large models?"**
pathmc compiles to standard PyMC, so the computational bottleneck is MCMC sampling, not pathmc itself. For large DAGs with many equations, the compilation is fast; the sampling scales as any PyMC model would. Panel mode with many units uses hierarchical priors for partial pooling, same as you'd do in Bambi.

**"Why a new package instead of extending Bambi?"**
Bambi's architecture is fundamentally single-equation: one formula, one likelihood, one link function. pathmc needs multi-equation compilation with topological ordering, cross-equation residual covariance, and graph surgery for `do()`. These are different enough to warrant a separate package, like how CausalPy is separate from Bambi despite both using PyMC.
