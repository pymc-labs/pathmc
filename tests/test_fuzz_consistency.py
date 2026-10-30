#   Copyright 2025 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Hypothesis fuzz tests for logp-graph consistency (issue #414).

The per-commit session compiles one example from each covering stratum plus
a small mixed budget (``FAST_MIXED_EXAMPLES``) drawn from
``FAST_PANEL_SHAPES``. The nightly session repeats the strata and draws a
larger mixed budget from the full shape menu. Both fail if a required
family, term kind, pooling mode, or scan bit was never compiled.

``PATHMC_FUZZ_MAX_EXAMPLES`` overrides the mixed budget for local debugging.
It does not skip the covering strata.

Seeds 4141 and 4142 are former scan count-predictor crashes. Current main
compiles them, so they stay in the passing corpus and the generator may
emit that class.
"""

from __future__ import annotations

import os

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

import pathmc
from _consistency import assert_model_consistent
from fuzz.strategies import (
    COVERING_STRATA,
    CROSS_SECTION_N,
    FAST_CROSS_SECTION_N,
    FAST_PANEL_SHAPES,
    PANEL_SHAPES,
    TERM_KINDS,
    Coverage,
    Edge,
    FuzzCase,
    Stratum,
    fuzz_cases,
    realize_frame,
    render_spec,
    stratum_cases,
)
from pathmc.parse import parse_spec

FAST_MIXED_EXAMPLES = 8
NIGHTLY_MIXED_EXAMPLES = 40


# Replay these without Hypothesis. The seed is the parametrize id.
PASSING_CORPUS: tuple[FuzzCase, ...] = (
    FuzzCase(
        seed=4140,
        spec="y ~ x",
        families=(("y", "gaussian"),),
        panel=None,
        pooling_kind="none",
        slope_vars=(),
        layout="cross_section",
        n_units=0,
        n_time=0,
        n_obs=8,
        columns=("x", "y"),
        predictors=("x",),
        term_kinds=("plain",),
        intercept=True,
    ),
    FuzzCase(
        seed=414,
        spec="y ~ lag(x)",
        families=(("y", "gaussian"),),
        panel=(("unit", "unit"), ("time", "time")),
        pooling_kind="none",
        slope_vars=(),
        layout="panel",
        n_units=1,
        n_time=2,
        n_obs=2,
        columns=("x", "y"),
        predictors=("x",),
        term_kinds=("lag",),
        intercept=True,
    ),
    FuzzCase(
        seed=4141,
        spec="y ~ lag(y)",
        families=(("y", "poisson"),),
        panel=(("unit", "unit"), ("time", "time")),
        pooling_kind="none",
        slope_vars=(),
        layout="panel",
        n_units=1,
        n_time=3,
        n_obs=3,
        columns=("y",),
        predictors=("y",),
        term_kinds=("lag",),
        intercept=True,
    ),
    FuzzCase(
        seed=4142,
        spec="m ~ x\ny ~ lag(x) + m",
        families=(("m", "negbinomial"), ("y", "gaussian")),
        panel=(("unit", "unit"), ("time", "time")),
        pooling_kind="none",
        slope_vars=(),
        layout="panel",
        n_units=1,
        n_time=3,
        n_obs=3,
        columns=("x", "m", "y"),
        predictors=("x", "m"),
        term_kinds=("lag", "plain"),
        intercept=True,
    ),
)


def _corpus_id(case: FuzzCase) -> str:
    return f"seed-{case.seed}"


def _mixed_budget(default: int) -> int:
    raw = os.environ.get("PATHMC_FUZZ_MAX_EXAMPLES")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"PATHMC_FUZZ_MAX_EXAMPLES must be a positive integer, got {raw!r}."
        ) from exc
    if value < 1:
        raise ValueError(
            "PATHMC_FUZZ_MAX_EXAMPLES must be >= 1 so the mixed strategy "
            f"still runs, got {value}."
        )
    return value


def _fuzz_settings(max_examples: int) -> settings:
    return settings(
        max_examples=max_examples,
        deadline=None,
        derandomize=True,
        database=None,
        suppress_health_check=[HealthCheck.too_slow],
    )


def _scan_flag(model: pathmc.PathModel) -> bool:
    gen = model._gen_model
    if gen is None:
        raise AssertionError("model() returned no generative model to inspect")
    return getattr(gen, "_pathmc_panel_scan", None) is not None


def _assert_pooling_bound(model: pathmc.PathModel, case: FuzzCase) -> None:
    """Partial and slopes must emit their hierarchical RVs; complete must not.

    Negbinomial dispersion is ``alpha_disp_*`` and is not a random intercept.
    """
    names = set(model.pymc_model.named_vars)
    intercepts = {name for name in names if name.startswith("alpha_")} - {
        name for name in names if name.startswith("alpha_disp_")
    }
    slopes = {name for name in names if name.startswith("slope_")}
    if case.pooling_kind == "partial":
        assert intercepts, (
            f"partial pooling did not emit alpha_* RVs for {case.spec!r}. "
            f"named_vars={sorted(names)}"
        )
    elif case.pooling_kind == "slopes":
        assert slopes, (
            f"slopes pooling did not emit slope_* RVs for {case.spec!r}. "
            f"Expected a raw term in {case.slope_vars}. named_vars={sorted(names)}"
        )
    elif intercepts or slopes or any(name.startswith("mu_alpha_") for name in names):
        raise AssertionError(
            "complete pooling emitted hierarchical RVs "
            f"{sorted(intercepts | slopes)} for {case.spec!r}"
        )


def check_case(case: FuzzCase) -> bool:
    """Compile ``case`` and run the no-sampling consistency oracle.

    Returns the scan flag the compiler actually took. Raises ``AssertionError``
    with the full case so a Hypothesis failure prints the seed and spec.
    """
    try:
        frame = realize_frame(case)
        built = pathmc.model(
            case.spec,
            data=frame,
            families=case.families_dict(),
            panel=case.panel_dict(),
            pooling=case.pooling(),
        )
        scan = _scan_flag(built)
        if scan != case.expects_scan():
            raise AssertionError(
                f"scan flag {scan} != expected {case.expects_scan()} "
                f"for layout={case.layout} terms={case.term_kinds}"
            )
        _assert_pooling_bound(built, case)
        point = built.pymc_model.initial_point(random_seed=case.seed)
        logp = float(built.pymc_model.compile_logp()(point))
        if not np.isfinite(logp):
            raise AssertionError(f"logp(initial_point) is not finite: {logp}")
        assert_model_consistent(built, seed=case.seed)
    except Exception as exc:
        raise AssertionError(f"fuzz case failed: {case}") from exc
    return scan


def _assert_grammar(case: FuzzCase) -> None:
    if "lag(adstock" in case.spec or "lag(logistic_saturation" in case.spec:
        raise AssertionError(f"nested lag is not legal syntax: {case}")
    parse_spec(case.spec)
    for line in case.spec.splitlines():
        child, rhs = (part.strip() for part in line.split("~", 1))
        if f"adstock({child}" in rhs or f"logistic_saturation({child}" in rhs:
            raise AssertionError(
                f"contemporaneous self-edge on {child} is a cycle: {case}"
            )
        body = rhs[3:].strip() if rhs.startswith("0 +") else rhs
        if any(part.strip() == child for part in body.split("+")):
            raise AssertionError(f"plain self-edge on {child} is a cycle: {case}")
    temporal = any(
        kind in {"lag", "adstock", "lag+adstock"} for kind in case.term_kinds
    )
    if (
        case.pooling_kind != "none"
        or "lag" in case.term_kinds
        or "lag+adstock" in case.term_kinds
    ):
        assert case.panel is not None, f"lag or pooling without panel: {case}"
    if temporal and case.layout == "cross_section":
        assert "lag" not in case.term_kinds, f"cross-section spec contains lag: {case}"
    if case.pooling_kind == "slopes":
        missing = set(case.slope_vars) - case.plain_terms()
        assert not missing, f"slope vars {missing} are not raw terms: {case}"


def _run_stratum(stratum: Stratum, coverage: Coverage) -> None:
    @_fuzz_settings(1)
    @given(case=stratum_cases(stratum))
    def _one(case: FuzzCase) -> None:
        _assert_grammar(case)
        if stratum.family not in case.families_dict().values():
            raise AssertionError(f"{stratum.id} lost family {stratum.family}: {case}")
        if stratum.term not in case.term_kinds:
            raise AssertionError(f"{stratum.id} lost term {stratum.term}: {case}")
        if case.pooling_kind != stratum.pooling or case.expects_scan() != stratum.scan:
            raise AssertionError(
                f"{stratum.id} built pooling={case.pooling_kind} "
                f"scan={case.expects_scan()}, expected {stratum.pooling}/"
                f"{stratum.scan}: {case}"
            )
        coverage.record(case, scan=check_case(case))

    try:
        _one()
    except AssertionError as exc:
        raise AssertionError(f"covering stratum {stratum.id} failed") from exc


def _run_mixed(
    max_examples: int,
    coverage: Coverage,
    *,
    panel_shapes: tuple[tuple[int, int], ...] = PANEL_SHAPES,
    cross_section_n: tuple[int, ...] = CROSS_SECTION_N,
) -> None:
    @_fuzz_settings(max_examples)
    @given(case=fuzz_cases(panel_shapes=panel_shapes, cross_section_n=cross_section_n))
    def _mixed(case: FuzzCase) -> None:
        _assert_grammar(case)
        coverage.record(case, scan=check_case(case))

    _mixed()


def _run_session(mixed_examples: int, *, fast_shapes: bool = False) -> None:
    coverage = Coverage()
    for stratum in COVERING_STRATA:
        _run_stratum(stratum, coverage)
    if fast_shapes:
        _run_mixed(
            _mixed_budget(mixed_examples),
            coverage,
            panel_shapes=FAST_PANEL_SHAPES,
            cross_section_n=FAST_CROSS_SECTION_N,
        )
    else:
        _run_mixed(_mixed_budget(mixed_examples), coverage)
    coverage.assert_session_covered()


def test_panel_shape_menu_matches_the_contract() -> None:
    """n_units in {1, 2, 5}, n_time in {3, 10, 30}, plus adversarial (1, 2)."""
    expected = {(1, 2)}
    expected.update(
        (n_units, n_time) for n_units in (1, 2, 5) for n_time in (3, 10, 30)
    )
    assert set(PANEL_SHAPES) == expected
    assert PANEL_SHAPES[0] == (1, 2)
    assert 1 not in CROSS_SECTION_N
    assert min(CROSS_SECTION_N) >= 2


@pytest.mark.parametrize(("n_units", "n_time"), PANEL_SHAPES)
def test_panel_frames_follow_the_shape_menu(n_units: int, n_time: int) -> None:
    case = replace(
        PASSING_CORPUS[1],
        n_units=n_units,
        n_time=n_time,
        n_obs=n_units * n_time,
    )
    frame = realize_frame(case)
    assert len(frame) == n_units * n_time
    assert frame["unit"].nunique() == n_units
    assert frame["time"].nunique() == n_time
    assert set(frame["unit"].unique()) == set(range(n_units))


def test_intercept_off_and_cooccurring_lag_adstock_render() -> None:
    off = render_spec([Edge("y", "plain", ("x",))], {"y": False})
    assert off == "y ~ 0 + x"
    both = render_spec([Edge("y", "lag+adstock", ("x",))], {"y": True})
    assert "lag(adstock" not in both
    assert "lag(x)" in both
    assert "adstock(x, decay=decay_y_x)" in both


def test_plain_terms_keep_x0_when_the_intercept_is_off() -> None:
    """``x0 +`` contains the characters ``0 +``; only the prefix may be stripped."""
    case = replace(PASSING_CORPUS[0], spec="y ~ 0 + x0 + lag(y)", intercept=False)
    assert case.plain_terms() == {"x0"}


def test_grammar_rejects_a_contemporaneous_self_edge() -> None:
    """adstock(y) on y is a cycle; lag(y) is the only legal self-edge."""
    cycled = replace(
        PASSING_CORPUS[1],
        spec="y ~ x + adstock(y, decay=decay_y_y)",
        term_kinds=("adstock", "plain"),
    )
    with pytest.raises(AssertionError, match="self-edge"):
        _assert_grammar(cycled)


def test_coverage_rejects_a_trivial_gaussian_session() -> None:
    coverage = Coverage()
    coverage.record(PASSING_CORPUS[0], scan=False)
    with pytest.raises(AssertionError, match="never hit") as exc:
        coverage.assert_session_covered()
    message = str(exc.value)
    assert "y ~ x" in message
    for term in set(TERM_KINDS) - {"plain"}:
        assert term in message


@pytest.mark.parametrize("case", PASSING_CORPUS, ids=_corpus_id)
def test_frozen_corpus_replays_without_hypothesis(case: FuzzCase) -> None:
    """Passing corpus entries rebuild from the seed and satisfy the oracle."""
    check_case(case)


@_fuzz_settings(30)
@given(case=fuzz_cases())
def test_generated_specs_obey_the_grammar(case: FuzzCase) -> None:
    """Structural invariants, without compiling a PyMC model."""
    _assert_grammar(case)
    assert 2 <= len(case.columns) <= 5
    if case.layout == "panel":
        assert (case.n_units, case.n_time) in PANEL_SHAPES
        assert case.n_obs == case.n_units * case.n_time
    else:
        assert case.n_obs in CROSS_SECTION_N
        assert case.panel is None


def test_fuzz_consistency() -> None:
    """Per-commit budget: every covering cell, plus a small mixed explore.

    Mixed draws stay on ``FAST_PANEL_SHAPES``. A 5×30 scan ``dlogp`` does not
    belong on this path; nightly still draws the full shape menu.
    """
    _run_session(FAST_MIXED_EXAMPLES, fast_shapes=True)


@pytest.mark.nightly
def test_fuzz_consistency_nightly() -> None:
    """Larger mixed budget. Not ``slow``: ``make test-nightly`` is the gate."""
    _run_session(NIGHTLY_MIXED_EXAMPLES)
