# PR: Docs — explain conditional vs joint modeling, two-graph representations, and cross-links

Closes #2

## Issue Summary

Add documentation explaining conditional-equation fitting vs full-joint generative SCM modeling, the distinction between pathmc's causal DAG and PyMC's estimation graph, and improve cross-linking between concept pages and examples.

## Root Cause

Advanced PyMC users expect interventions to be represented as modified PyMC model graphs. pathmc intentionally separates inference from intervention simulation — this was under-explained, causing confusion when interpreting `pm.model_to_graphviz()` outputs and `do()` behavior.

## Solution

Added two new subsections to the architecture overview (`how-it-works.qmd`) and improved cross-links between concept pages and examples. Several acceptance criteria from the issue were already satisfied by prior work (ATE/ATT/ATU definitions, mediation example contrasts).

## Changes Made

- `docs/how-it-works.qmd`: Added "From structural equations to the generative model" subsection with mathematical equations (structural equations, joint factorization, truncated factorization formula) and a callout explaining why pathmc uses the full joint rather than separate regressions. Added "Causal DAG vs estimation graph" subsection with a "Which graph should I inspect?" callout table.
- `docs/concepts/causal_inference.qmd`: Added cross-link to the architecture overview at the top; added "Worked examples" section at the bottom linking to mediation, do queries, seeing vs doing, moderation, and Simpson's paradox examples.
- `docs/examples/mediation.qmd`: Added cross-link to the new "Causal DAG vs estimation graph" section where the PyMC graph is discussed; added link to the causal inference concepts page from the assumptions section.

## Testing

- [x] Both modified concept pages render successfully with `quarto render`
- [x] No ruff errors introduced
- [x] All cross-links use correct relative paths and anchors

## Notes

Several acceptance criteria from the original issue were already complete before this PR:
- ATE/ATT/ATU definitions are well-documented in `concepts/causal_inference.qmd` (criteria 3)
- Mediation example already demonstrates treatment-only, mediator-only, direct-only contrasts, and ATE helper linkage (criteria 4)
- Architecture diagrams (mermaid) already present in `how-it-works.qmd` (partial criteria 1)

This PR addresses the remaining gaps: the mathematical exposition (criteria 2), the explicit DAG-vs-estimation-graph explanation (criteria 1), cross-links (criteria 5), and the "Which graph should I inspect?" nice-to-have.
