# PR: Add Pearl's Primer tutorial examples

Closes #21

## Issue Summary

Add four canonical causal inference examples from Pearl, Glymour & Jewell's *Causal Inference in Statistics: A Primer* as pathmc tutorials, bridging causal inference theory to practice.

## Root Cause

The examples gallery lacked coverage of the most widely-known causal inference scenarios (Simpson's Paradox, collider bias, seeing-vs-doing, front-door criterion), which are often the first examples new users encounter in causal inference textbooks.

## Solution

Created four new Quarto tutorial documents in `docs/examples/`, each following the existing tutorial conventions (YAML frontmatter with categories, simulated data with known ground truth, DAG visualization, identification checks, do-operator queries, comparison figures, and reflection prompts).

## Changes Made

- `docs/examples/simpsons_paradox.qmd`: Drug/gender/recovery example. Shows naive (confounded) vs DAG-informed regression, adjustment sets, and do-operator ATE recovery.
- `docs/examples/collider_bias.qmd`: Birth-weight paradox. Demonstrates collider_warnings(), binary outcomes via families={"mortality": "bernoulli"}, and how conditioning on a collider reverses the sign of the estimate.
- `docs/examples/seeing_vs_doing.qmd`: Contrasts P(Y|X=x) with P(Y|do(X=x)) in a confounded model, then shows they agree when X is unconfounded (randomized).
- `docs/examples/front_door.qmd`: Smoking/tar/cancer with unobserved confounder. Shows naive regression failure, front-door identification via mediation analysis (indirect := a*b), and do-operator cross-check.

## Testing

- [x] Existing tests pass (221 passed, 93 deselected)
- [x] No new source code changes — documentation only
- [x] Follows existing tutorial patterns and conventions

## Notes

- All four tutorials use simulated data with known ground truth so readers can verify parameter recovery.
- Each tutorial includes a comparison figure (correct vs biased estimates) with code-fold.
- Categories assigned for automatic listing: [Fundamentals, Causal Inference] and [Causal Inference, Mediation].
