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
"""Display contract for a conditioning set that was not the one tested (#376)."""

import warnings
from pathlib import Path

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc._ci import CIResult, named_column_ci, partial_correlation_ci
from pathmc.falsify import FalsificationStatus, falsify_graph
from pathmc.graph import build_graph
from pathmc.identify import ConditionalIndependence
from pathmc.identify import test_implications as score_implications
from pathmc.parse import parse_spec


def _warning_is_from_caller(filename: str, caller: Path) -> bool:
    """Whether this warning frame is the caller file.

    A checkout whose path contains ``/pathmc/`` is still the caller when
    the filename is that file. A ``pathmc`` module is not.
    """
    return Path(filename).resolve() == caller.resolve()


def test_caller_attribution_accepts_a_checkout_named_pathmc() -> None:
    caller = Path("/home/runner/work/pathmc/pathmc/tests/test_ci_rank_collapse.py")
    assert _warning_is_from_caller(str(caller), caller)
    package = Path("/home/runner/work/pathmc/pathmc/pathmc/_ci_report.py")
    assert not _warning_is_from_caller(str(package), caller)
    other_copy = Path("/tmp/elsewhere/tests/test_ci_rank_collapse.py")
    assert not _warning_is_from_caller(str(other_copy), caller)


def _cols(frame, names):
    return frame[list(names)].to_numpy(dtype=float)


def _constant_conditioner_frame() -> pd.DataFrame:
    """C is constant, so any set that includes C is rank-deficient.

    The DAG ``M ~ C`` / ``Z ~ X`` implies two independences that condition
    on C: M ⊥⊥ X | C, and M ⊥⊥ Z | {C, X}.
    """
    rng = np.random.default_rng(11)
    n = 50
    x = rng.normal(size=n)
    c = np.ones(n)
    return pd.DataFrame({
        "C": c,
        "X": x,
        "M": 0.4 * c + rng.normal(size=n),
        "Z": 0.7 * x + rng.normal(size=n),
    })


def _make_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = 0.8 * z1 - 0.4 * z2 + rng.normal(scale=0.7, size=n)
    m = 0.7 * x + rng.normal(scale=0.6, size=n)
    y = 0.6 * m + 0.3 * z1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "M": m, "Y": y, "Z1": z1, "Z2": z2})


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _make_frame()


def test_zero_width_conditioner_is_not_a_named_collapse() -> None:
    """A column-only zero-width block has no name to drop."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    result = partial_correlation_ci(x, y, np.empty((40, 0)))
    assert result.requested_k == 0
    assert result.effective_k == 0
    assert result.rank_collapsed is False
    assert result.dropped_conditioners == ()


class TestRankCollapseIsVisible:
    """A collapsed conditioning set must not look like the one requested (#376)."""

    def test_core_reports_effective_k_without_warning(self, frame):
        x, y = frame["X"].to_numpy(), frame["Y"].to_numpy()
        z = _cols(frame, ["M"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            marginal = partial_correlation_ci(x, y)
            const = partial_correlation_ci(x, y, np.ones((len(x), 1)))
            doubled = partial_correlation_ci(x, y, np.column_stack([z, z]))
        assert marginal.effective_k == 0
        assert marginal.requested_k == 0
        assert marginal.rank_collapsed is False
        assert const.effective_k == 0
        assert const.requested_k == 1
        assert const.rank_collapsed is True
        assert const.df == marginal.df == len(x) - 2
        assert doubled.effective_k == 1
        assert doubled.requested_k == 2
        assert doubled.rank_collapsed is True
        assert doubled.df == len(x) - doubled.effective_k - 2

    def test_skip_has_no_effective_k(self):
        result = partial_correlation_ci(np.array([1.0, 2.0]), np.array([2.0, 1.0]))
        assert result.effective_k is None
        assert result.requested_k == 0
        assert result.rank_collapsed is False

    def test_skip_with_conditioners_is_not_rank_collapsed(self):
        result = partial_correlation_ci(
            np.array([1.0, 2.0]),
            np.array([2.0, 1.0]),
            np.ones((2, 1)),
        )
        assert result.skip_reason == "insufficient_observations"
        assert result.requested_k == 1
        assert result.effective_k is None
        assert result.rank_collapsed is False

    def test_named_column_reports_collapse_without_warning(self):
        data = nw.from_native(_constant_conditioner_frame(), eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            collapsed = named_column_ci(data, "M", "X", ["C"])
            marginal = named_column_ci(data, "M", "X", [])
        assert isinstance(collapsed, CIResult)
        assert collapsed.n == 50
        assert collapsed.rank_collapsed is True
        assert collapsed.dropped_conditioners == ()
        assert collapsed.requested_k == 1
        assert collapsed.effective_k == 0
        assert collapsed.r == pytest.approx(marginal.r)
        assert marginal.rank_collapsed is False

    def test_implications_frame_and_single_warning(self):
        frame = _constant_conditioner_frame()
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.test_implications()
        collapses = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(collapses) == 1
        assert _warning_is_from_caller(collapses[0].filename, Path(__file__))
        assert "rank collapse" in str(collapses[0].message)
        assert "M ⊥⊥ X | {C}" in str(collapses[0].message)
        assert "M ⊥⊥ Z | {C, X}" in str(collapses[0].message)

        by_pair = {
            (row.x, row.y, row.conditioning_set): row
            for row in result.results.itertuples(index=False)
        }
        only_c = by_pair[("M", "X", "C")]
        assert only_c.requested_k == 1
        assert only_c.effective_k == 0
        assert only_c.rank_collapsed
        assert bool(only_c.significant) is False
        assert only_c.df == only_c.n_obs - 2
        both = by_pair[("M", "Z", "C, X")]
        assert both.requested_k == 2
        assert both.effective_k == 1
        assert both.rank_collapsed
        assert both.df == both.n_obs - both.effective_k - 2
        skipped = by_pair[("C", "X", "")]
        assert skipped.requested_k == 0
        assert not skipped.rank_collapsed
        assert np.isnan(skipped.effective_k)
        assert np.isnan(skipped.p_value)
        assert result.n_violations == int(result.results["significant"].sum())
        html = result._repr_html_()
        assert "All implied independences are consistent" not in html
        assert "rank-collapsed" in html
        assert "did not test" in html
        assert "[effective_k=0]" in html
        assert "[effective_k=1]" in html
        assert "rank-collapsed" in repr(result)

    def test_collapsed_reduced_test_is_not_a_violation(self):
        rng = np.random.default_rng(4)
        n = 80
        m = rng.normal(size=n)
        frame = pd.DataFrame({
            "C": np.ones(n),
            "M": m,
            "X": m + rng.normal(scale=0.05, size=n),
            "Z": rng.normal(size=n),
        })
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.test_implications()
        collapsed = result.results[result.results["rank_collapsed"].astype(bool)]
        row = collapsed.loc[collapsed["p_value"] < result.alpha].iloc[0]
        assert {row["x"], row["y"]} == {"M", "X"}
        assert bool(row["significant"]) is False
        assert result.n_violations == 0
        assert int(result.results["significant"].sum()) == result.n_violations
        html = result._repr_html_()
        collapsed_row = next(
            part for part in html.split("<tr") if "M ⊥⊥ X | {C}" in part
        )
        assert "✗" not in collapsed_row
        assert "✓" not in collapsed_row
        assert "—" in collapsed_row
        assert "ffe0e0" not in collapsed_row
        assert "[effective_k=0]" in collapsed_row
        assert "violated" not in html
        assert "0 violations" in repr(result)

    def test_mixed_implication_counts_only_the_written_violation(self):
        rng = np.random.default_rng(1)
        n = 120
        z = rng.normal(size=n)
        x = rng.normal(size=n)
        frame = pd.DataFrame({
            "X": x,
            "Y": z + rng.normal(scale=0.1, size=n),
            "Z": z,
            "W": z + rng.normal(scale=0.1, size=n),
            "M": x + rng.normal(scale=0.1, size=n),
            "C": np.ones(n),
        })
        model = pathmc.model("Y ~ X\nW ~ Z\nM ~ C", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.test_implications()
        collapsed = result.results["rank_collapsed"].fillna(False).astype(bool)
        written = result.results.loc[~collapsed]
        written_violations = int(written["significant"].sum())
        assert collapsed.any()
        assert written_violations >= 1
        assert not result.results.loc[collapsed, "significant"].any()
        assert result.n_violations == int(result.results["significant"].sum())
        assert result.n_violations == written_violations
        assert len(result.violations) == written_violations
        assert not result.violations["rank_collapsed"].any()
        html = result._repr_html_()
        assert f"{written_violations} of {len(written)}" in html
        assert f"{written_violations} violations" in repr(result)
        assert f"{len(written)} tests" in repr(result)

    def test_well_conditioned_implications_do_not_warn(self, frame):
        model = pathmc.model("M ~ X\nY ~ M", data=frame[["X", "M", "Y"]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.test_implications()
        row = result.results.iloc[0]
        assert row["requested_k"] == 1
        assert row["effective_k"] == 1
        assert not row["rank_collapsed"]
        assert row["df"] == row["n_obs"] - row["effective_k"] - 2
        html = result._repr_html_()
        assert "effective_k" not in html
        assert "rank-collapsed" not in html
        assert "rank-collapsed" not in repr(result)

    def test_falsify_warns_once_for_the_given_dag(self):
        frame = _constant_conditioner_frame()
        model = pathmc.model("M ~ C\nZ ~ X", data=frame)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.falsify(n_permutations=24, random_seed=0)
        collapses = [
            w
            for w in caught
            if issubclass(w.category, UserWarning) and "rank collapse" in str(w.message)
        ]
        assert len(collapses) == 1
        assert _warning_is_from_caller(collapses[0].filename, Path(__file__))
        collapsed = result.local_violations[result.local_violations["rank_collapsed"]]
        assert set(collapsed["conditioning_set"]) == {"C"}
        assert (collapsed["requested_k"] == 1).all()
        assert (collapsed["effective_k"] == 0).all()
        assert (collapsed["df"] == len(frame) - 2).all()
        assert not collapsed["violation"].any()
        summary = "\n".join(result._summary_lines())
        assert "rank-collapsed" in summary
        assert "requested parent sets" in summary
        assert "Verdict:" not in summary
        html = result._repr_html_()
        assert "rank-collapsed" in html
        assert "requested parent sets" in html
        assert "LMC violations (given DAG)" in html
        assert "rank-collapsed" in repr(result)

    def test_collapsed_row_does_not_decide_the_lmc_count(self):
        rng = np.random.default_rng(0)
        n = 80
        a = rng.normal(size=n)
        y = rng.normal(size=n)
        frame = pd.DataFrame({
            "A": a,
            "B": 2 * a - 1,
            "M": y + rng.normal(scale=0.05, size=n),
            "X": rng.normal(size=n),
            "Y": y,
        })
        model = pathmc.model("M ~ A + B\nY ~ X", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(n_permutations=8, random_seed=0)
        collapsed = result.local_violations["rank_collapsed"].astype(bool)
        assert collapsed.any()
        scored = result.local_violations.loc[~collapsed]
        scored_viol = int(scored["violation"].sum()) if len(scored) else 0
        scored_tests = len(scored)
        n_collapsed = int(collapsed.sum())
        assert result.given_lmc_violations == scored_viol
        assert result.n_lmc_tests == scored_tests
        assert int(result.local_violations["violation"].sum()) == scored_viol
        assert result.violations["rank_collapsed"].sum() == 0
        text = "\n".join(result._summary_lines())
        noun = "test" if n_collapsed == 1 else "tests"
        assert result.status in {
            FalsificationStatus.REJECTED,
            FalsificationStatus.NOT_FALSIFIABLE,
            FalsificationStatus.NOT_REJECTED,
        }
        assert "rank-collapsed" not in text.split("\n")[0]
        assert f"{scored_viol} of {scored_tests} requested" in text
        assert f"{n_collapsed} {noun} rank-collapsed" in text
        assert text.count("rank-collapsed") == 1
        assert "Permutation fraction" in text
        html = result._repr_html_()
        assert f"{scored_viol} / {scored_tests} requested" in html
        assert f"{n_collapsed} {noun} rank-collapsed" in html

    def test_all_collapsed_parent_sets_are_not_evaluable(self):
        rng = np.random.default_rng(2)
        n = 60
        a = rng.normal(size=n)
        frame = pd.DataFrame({
            "A": a,
            "B": 2 * a - 1,
            "X": a + rng.normal(scale=0.2, size=n),
            "Y": a + rng.normal(scale=0.2, size=n),
        })
        model = pathmc.model("B ~ A\nX ~ A + B\nY ~ A + B", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(n_permutations=6, random_seed=1)
        assert result.local_violations["rank_collapsed"].all()
        assert not result.local_violations["violation"].any()
        assert result.n_lmc_tests == 0
        assert result.given_lmc_violations == 0
        assert result.status is FalsificationStatus.ALL_COLLAPSED
        assert result.can_evaluate is False
        assert result.falsifiable is None
        assert result.falsified is None
        text = "\n".join(result._summary_lines())
        status_sentence, collapse_clause, permutation_clause = text.split("\n")
        assert status_sentence == (
            "DAG falsification: cannot be evaluated. No written parental test remains."
        )
        assert "rank-collapsed" not in status_sentence
        assert collapse_clause == (
            f"{len(result.local_violations)} tests rank-collapsed and did not "
            "test the requested parent sets."
        )
        assert permutation_clause.startswith("Permutation fraction:")
        assert "fully connected" not in text
        html = result._repr_html_()
        assert html.lower().count("rank-collapsed") == 1
        assert "no written parental test remains" in html
        assert "implies no testable" not in html
        compact = repr(result)
        assert "all_collapsed" in compact
        assert "not_evaluable" not in compact
        assert "rank-collapsed" in compact
        assert "\n" not in compact
        with pytest.raises(RuntimeError, match="rank-collapsed"):
            result.plot()

    def test_collapsed_only_permutation_is_not_a_perfect_lmc_score(self):
        rng = np.random.default_rng(0)
        n = 200
        a = rng.normal(size=n)
        x = 0.8 * a + rng.normal(scale=0.3, size=n)
        z = 0.8 * a + rng.normal(scale=0.3, size=n)
        y = 0.5 * a + 0.5 * x + 0.5 * z + rng.normal(scale=0.3, size=n)
        w = 0.5 * a + 0.8 * y + rng.normal(scale=0.3, size=n)
        frame = pd.DataFrame({
            "A": a,
            "X": x,
            "Z": z,
            "Y": y,
            "W": w,
            "C": np.ones(n),
        })
        model = pathmc.model(
            "X ~ A\nZ ~ A\nY ~ A + X + Z\nW ~ Y + A\nC ~ A",
            data=frame,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(
                n_permutations=120,
                random_seed=4,
                include_unconditional=False,
            )
        scored = np.asarray(result.perm_lmc_violation_fractions)
        omitted = result.n_permutations - len(scored)
        assert len(result.perm_tpa_violation_fractions) == result.n_permutations
        assert omitted > 0
        assert result.p_value_tpa == pytest.approx(
            result.n_in_mec / result.n_permutations
        )
        scored_p = float(np.mean(scored <= result.given_lmc_violation_fraction))
        inflated = np.concatenate([scored, np.zeros(omitted)])
        inflated_p = float(np.mean(inflated <= result.given_lmc_violation_fraction))
        assert result.p_value_lmc == pytest.approx(scored_p)
        assert inflated_p > result.significance_level
        assert result.p_value_lmc <= result.significance_level
        assert result.falsified is False
        text = "\n".join(result._summary_lines())
        assert "not in that fraction" in text
        assert f"{inflated_p:.3f}" not in text
        html = result._repr_html_()
        assert f"{omitted} omitted" in html
        assert f"{inflated_p:.3f}" not in html
        fig = result.plot()
        assert fig.axes[0].get_title().startswith("DAG falsification")

    def test_unscored_permutation_baseline_is_not_a_pass(self):
        rng = np.random.default_rng(0)
        n = 80
        b = rng.normal(size=n)
        frame = pd.DataFrame({
            "B": b,
            "Y": b + rng.normal(scale=0.2, size=n),
            "Z": b + rng.normal(scale=0.2, size=n),
            "K": np.ones(n),
        })
        model = pathmc.model("Y ~ B\nZ ~ B\nK ~ B", data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = model.falsify(
                n_permutations=1,
                random_seed=6,
                include_unconditional=False,
            )
        assert len(result.perm_lmc_violation_fractions) == 0
        assert len(result.perm_tpa_violation_fractions) == result.n_permutations
        assert result.status is FalsificationStatus.NOT_COMPARED
        assert result.can_evaluate is False
        assert result.falsifiable is None
        assert result.falsified is None
        text = "\n".join(result._summary_lines())
        assert "LMC comparison was not run" in text
        assert "not rejected" not in text
        assert "not_compared" in repr(result)
        assert "not_rejected" not in repr(result)
        html = result._repr_html_()
        assert "Not compared" in html
        assert "LMC comparison was not run" in html
        assert "Not rejected" not in html
        fig = result.plot()
        assert fig.axes[0].get_title().startswith("DAG falsification")

    def test_falsify_full_rank_parents_do_not_warn(self, frame):
        model = pathmc.model("M ~ X\nY ~ M", data=frame[["X", "M", "Y"]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = model.falsify(n_permutations=6, random_seed=0)
        assert "effective_k" in result.local_violations.columns
        assert "df" in result.local_violations.columns
        assert "requested_k" in result.local_violations.columns
        assert "rank_collapsed" in result.local_violations.columns
        assert not result.local_violations["rank_collapsed"].any()
        assert "rank-collapsed" not in "\n".join(result._summary_lines())
        assert "rank-collapsed" not in result._repr_html_()
        assert "rank-collapsed" not in repr(result)

    def test_single_level_categorical_is_not_the_written_test(self):
        """A named conditioner with no columns is not the written independence."""
        rng = np.random.default_rng(1)
        n = 200
        x = rng.normal(size=n)
        y = 0.6 * x + rng.normal(scale=0.3, size=n)
        statement = ConditionalIndependence("x", "y", frozenset({"g"}))
        categorical = pd.DataFrame({"x": x, "y": y, "g": ["only"] * n})
        numeric = pd.DataFrame({"x": x, "y": y, "g": np.ones(n)})
        coded = pd.DataFrame({"x": x, "y": y, "g": np.ones(n, dtype=int)})
        two_level = pd.DataFrame({
            "x": x,
            "y": y,
            "g": np.array(["a", "b"] * (n // 2)),
        })
        cat_data = nw.from_native(categorical, eager_only=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            engine = named_column_ci(cat_data, "x", "y", ["g"])
        assert engine.requested_k == 0
        assert engine.effective_k == 0
        assert engine.dropped_conditioners == ("g",)
        assert engine.rank_collapsed is True
        assert engine.p is not None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            cat_result = score_implications([statement], cat_data)
        assert len(caught) == 1
        assert "g contributed no columns" in str(caught[0].message)
        row = cat_result.results.iloc[0]
        assert row["requested_k"] == 0
        assert row["effective_k"] == 0
        assert bool(row["rank_collapsed"]) is True
        assert bool(row["significant"]) is False
        assert not np.isnan(row["p_value"])
        assert cat_result.n_violations == 0
        assert int(cat_result.results["significant"].sum()) == 0
        assert cat_result.n_tests == 1
        assert "0 tests, 0 violations, 1 rank-collapsed" in repr(cat_result)
        html = cat_result._repr_html_()
        assert "rank-collapsed" in html
        assert "implied independences violated" not in html
        assert "All implied independences are consistent" not in html

        with warnings.catch_warnings(record=True) as numeric_caught:
            warnings.simplefilter("always", UserWarning)
            num_result = score_implications(
                [statement], nw.from_native(numeric, eager_only=True)
            )
        assert "contributed no columns" not in str(numeric_caught[0].message)
        assert bool(num_result.results.iloc[0]["rank_collapsed"]) is True
        assert row["p_value"] == pytest.approx(num_result.results.iloc[0]["p_value"])

        with warnings.catch_warnings(record=True) as coded_caught:
            warnings.simplefilter("always", UserWarning)
            coded_result = score_implications(
                [statement],
                nw.from_native(coded, eager_only=True),
                categorical_vars=["g"],
            )
        assert "g contributed no columns" in str(coded_caught[0].message)
        assert bool(coded_result.results.iloc[0]["rank_collapsed"]) is True
        assert coded_result.results.iloc[0]["requested_k"] == 0

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            two_result = score_implications(
                [statement], nw.from_native(two_level, eager_only=True)
            )
        assert bool(two_result.results.iloc[0]["rank_collapsed"]) is False

        mixed_y = x + 0.05 * rng.normal(size=n)
        mixed = pd.DataFrame({
            "x": x,
            "y": mixed_y,
            "g": ["only"] * n,
            "m": rng.normal(size=n),
        })
        only_m = ConditionalIndependence("x", "y", frozenset({"m"}))
        mixed_statement = ConditionalIndependence("x", "y", frozenset({"g", "m"}))
        mixed_data = nw.from_native(mixed, eager_only=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m_only = score_implications(
                [only_m], nw.from_native(mixed.drop(columns="g"), eager_only=True)
            )
        assert m_only.n_violations == 1
        with warnings.catch_warnings(record=True) as mixed_caught:
            warnings.simplefilter("always", UserWarning)
            mixed_result = score_implications([mixed_statement], mixed_data)
        mixed_row = mixed_result.results.iloc[0]
        assert mixed_row["requested_k"] == 1
        assert mixed_row["effective_k"] == 1
        assert bool(mixed_row["rank_collapsed"]) is True
        assert bool(mixed_row["significant"]) is False
        assert mixed_result.n_violations == 0
        assert mixed_row["p_value"] == pytest.approx(m_only.results.iloc[0]["p_value"])
        assert "g contributed no columns" in str(mixed_caught[0].message)

        graph = build_graph(parse_spec("Y ~ G\nZ ~ G"))
        cat_frame = pd.DataFrame({
            "G": ["only"] * n,
            "Y": y,
            "Z": y + rng.normal(scale=0.2, size=n),
        })
        num_frame = cat_frame.assign(G=np.ones(n))
        with warnings.catch_warnings(record=True) as falsify_caught:
            warnings.simplefilter("always", UserWarning)
            cat_falsify = falsify_graph(
                graph,
                nw.from_native(cat_frame, eager_only=True),
                n_permutations=2,
                random_seed=0,
                include_unconditional=False,
            )
        assert len(falsify_caught) == 1
        assert "G contributed no columns" in str(falsify_caught[0].message)
        assert cat_falsify.status is FalsificationStatus.ALL_COLLAPSED
        assert cat_falsify.n_lmc_tests == 0
        assert cat_falsify.given_lmc_violations == 0
        assert cat_falsify.can_evaluate is False
        assert cat_falsify.falsified is None
        assert cat_falsify.local_violations["rank_collapsed"].all()
        assert not cat_falsify.local_violations["violation"].any()
        assert cat_falsify.local_violations["requested_k"].eq(0).all()
        assert cat_falsify.local_violations["p_value"].notna().all()
        text = "\n".join(cat_falsify._summary_lines())
        assert "rank-collapsed" in text
        assert text.count("rank-collapsed") == 1
        assert "we do not reject" not in text
        assert "not_rejected" not in repr(cat_falsify)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            num_falsify = falsify_graph(
                graph,
                nw.from_native(num_frame, eager_only=True),
                n_permutations=2,
                random_seed=0,
                include_unconditional=False,
            )
        assert num_falsify.n_lmc_tests == 0
        assert num_falsify.falsified is None
        assert cat_falsify.local_violations["p_value"].to_numpy() == pytest.approx(
            num_falsify.local_violations["p_value"].to_numpy()
        )
