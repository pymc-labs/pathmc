# PR: Add custom transform tutorial

Closes #55

## Issue Summary

The `register_transform()` API exists and works, but there was no documentation showing how to create and use a custom transform. This capability is invisible to users without a tutorial.

## Root Cause

Missing documentation — the transform registry is a powerful extensibility feature with no example showing end-to-end usage.

## Solution

Added a new example notebook (`docs/examples/custom_transforms.qmd`) that walks through the full custom transform lifecycle using a Hill function (dose-response) as the motivating example.

## Changes Made

- `docs/examples/custom_transforms.qmd`: New tutorial covering:
  - Subclassing `Transform` with `name`, `param_specs`, and `apply_pymc()`
  - Registering with `register_transform()`
  - Using the custom transform in the DSL
  - Setting domain-appropriate priors for transform parameters
  - Parameter recovery from simulated data
  - Interventional predictions with `do()` that recompute the transform
  - Causal effect contrasts through the nonlinearity

## Testing

- [x] Existing tests pass (354 passed)
- [x] Notebook renders successfully with Quarto
- [x] ruff check and format clean

## Notes

- The tutorial uses a Hill saturation function (`x^n / (K^n + x^n)`) — a two-parameter nonlinearity common in pharmacology, which is more interesting than a single-parameter example.
- The reflection prompt at the end suggests other domain-specific transforms (Michaelis-Menten, Monod, power law, CES) to encourage users to build their own.
- The notebook auto-appears in the Examples listing page via the existing `listing: contents: "*.qmd"` configuration.
