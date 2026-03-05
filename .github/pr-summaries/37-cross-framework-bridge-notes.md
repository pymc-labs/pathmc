# PR: Add cross-framework bridge notes (Pearl ↔ Potential Outcomes)

Closes #37

## Issue Summary

The causal inference concept page had a compact 4-line callout for PO practitioners but lacked a proper cross-framework bridge section. Readers from the potential outcomes tradition could not easily map pathmc's Pearlian concepts to their vocabulary.

## Root Cause

The existing callout briefly mentioned the key equivalences but did not provide a structured term mapping or detailed explanations of each correspondence.

## Solution

Replaced the short callout with a full "Pearl ↔ Potential Outcomes" section containing:
- A term-mapping table (Pearl/SCM ↔ PO/Epidemiology ↔ pathmc API)
- Detailed explanations of three key equivalences: ATE, identification, and estimation
- A DoWhy-specific callout for users coming from that workflow
- A brief framework-positioning note in the comparison page linking to the new section

## Changes Made

- `docs/concepts/causal_inference.qmd`: Replaced the "For potential-outcomes practitioners" callout with a full "Pearl ↔ Potential Outcomes" section including term mapping table and key equivalences subsections
- `docs/comparison.qmd`: Added framework-positioning paragraph to the Philosophy section with a link to the new bridge section

## Testing

- [x] Existing tests pass (222 passed)
- [x] Linting clean (ruff check + format)
- [ ] New tests added — N/A (documentation-only change)

## Notes

This is a documentation-only change addressing Gap 4 from the causal inference audit (#34).
