# PR: Complete PyPI metadata for public release

Closes #177

## Issue Summary

The package metadata in `pyproject.toml` was too sparse for a public PyPI release: the README would not render as the long description, the package had no explicit license metadata, and PyPI would not show useful author, classifier, keyword, or project links.

## Root Cause

The `[project]` table only declared the package name, version, terse description, Python requirement, and dependencies, so build artifacts lacked the standard metadata PyPI uses for package presentation and discoverability.

## Solution

Expand the PEP 621 project metadata with README, SPDX license metadata, organization-only author and maintainer entries, keywords, Trove classifiers, and project URLs.

## Changes Made

- `pyproject.toml`: Add README, Apache-2.0 license metadata, PyMC Labs author and maintainer metadata, keywords, classifiers, and PyPI sidebar URLs.
- `pyproject.toml`: Update the build backend requirement to `setuptools>=77.0` so modern SPDX `license` and `license-files` metadata are supported.

## Testing

- [x] Existing tests pass (`conda run -n pathmc pytest -x -v -m "not slow"`: 391 passed, 136 deselected)
- [x] Lint and format pass (`conda run -n pathmc ruff check --fix && conda run -n pathmc ruff format`)
- [x] Manual verification completed (`conda run -n pathmc python -m build` and `conda run -n pathmc python -m twine check dist/*`)

## Notes

The metadata intentionally omits email addresses for now and uses `PyMC Labs` as the organization-only author and maintainer.
