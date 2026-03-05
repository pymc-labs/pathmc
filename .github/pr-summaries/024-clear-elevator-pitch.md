# PR: Add clear elevator pitch to home page and README

Closes #24

## Issue Summary

The project lacked a clear, prominent elevator pitch describing what pathmc does.

## Root Cause

The docs home page tagline ("Bayesian path analysis via PyMC") was too brief and didn't convey the full scope — structural causal models, Bayesian estimation, interventional simulation, and the DSL.

## Solution

Added the elevator pitch from the issue ("Structural causal models with Bayesian estimation and interventional simulation via a concise DSL") to both the docs home page (`docs/index.qmd`) and the GitHub README (`README.md`).

## Changes Made

- `docs/index.qmd`: Replaced the tagline with the elevator pitch; streamlined the follow-on sentence
- `README.md`: Added the elevator pitch and description below the logo

## Testing

- [x] Existing tests pass (221 passed)
- [x] Changes are docs-only — no code affected

## Notes

The wording aligns with the suggestion in the issue while keeping the follow-on description concise.
