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
"""A defined parameter must not occupy a coefficient label's name.

``effects_summary()`` reports each labeled coefficient and each defined
parameter. ``a := a * 2`` used to replace label ``a`` with the defined
quantity (issue #575). The collision is rejected, naming the clash.
"""

import arviz as az
import numpy as np
import pandas as pd
import pytest

import pathmc
from pathmc.effects import build_effects_summary
from pathmc.exceptions import DuplicateEquationError
from pathmc.parse import DefinedParam, Regression, Spec, Term, TransformCall, parse_spec


def _assert_names_clash(spec: str, *names: str) -> None:
    with pytest.raises(DuplicateEquationError, match="coefficient label") as exc:
        parse_spec(spec)
    message = str(exc.value)
    assert "must not occupy a label's name" in message
    for name in names:
        assert f"'{name}' is already the coefficient label" in message


@pytest.mark.parametrize(
    "spec",
    [
        "Y ~ a*X\na := a * 2",
        "a := a * 2\nY ~ a*X",
        "Y ~ a*X; a := a * 2",
        "a:=a*2\nY ~ a*X",
        "Y ~ a*X\na := 1",
        "M ~ a*X\nY ~ b*M\na := b",
    ],
)
def test_defined_param_label_collision_is_rejected(spec: str) -> None:
    _assert_names_clash(spec, "a")


def test_collision_is_not_special_cased_to_one_label() -> None:
    _assert_names_clash("Y ~ slope*X\nslope := slope + 1", "slope")


def test_every_colliding_name_is_reported() -> None:
    _assert_names_clash("Y ~ a*X + b*Z\na := 1\nb := 2", "a", "b")


def test_collision_message_cites_the_transformed_term() -> None:
    """The owning equation must still show the transform, not only its input."""
    statement = "Y ~ 1 + slope*log(X)"
    with pytest.raises(DuplicateEquationError, match="coefficient label") as exc:
        parse_spec(f"{statement}\nslope := slope * 2")
    message = str(exc.value)
    assert statement in message
    assert "slope*X" not in message


def test_collision_message_cites_a_parameterized_transform() -> None:
    statement = "sales ~ b*adstock(tv, decay=theta)"
    with pytest.raises(DuplicateEquationError, match="coefficient label") as exc:
        parse_spec(f"{statement}\nb := b * 2")
    message = str(exc.value)
    assert statement in message
    assert "b*tv" not in message


def test_distinct_defined_name_still_parses() -> None:
    spec = parse_spec("Y ~ a*X\ntwice := a * 2")
    assert spec.defined_params[0].name == "twice"
    assert spec.regressions[0].terms[0].label == "a"


def test_build_effects_summary_rejects_hand_built_collision() -> None:
    """The summary merge must not replace the label when parse was bypassed."""
    spec = Spec(
        regressions=[Regression("Y", [Term("X", label="a")])],
        defined_params=[DefinedParam("a", "a * 2")],
    )
    idata = az.from_dict(
        {"posterior": {"beta_Y": np.tile([4.0], (1, 4, 1))}},
        coords={"Y_predictors": ["X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )
    with pytest.raises(
        DuplicateEquationError, match="'a' is already the coefficient label"
    ):
        build_effects_summary(spec, idata)


def test_hand_built_transform_collision_names_the_call() -> None:
    spec = Spec(
        regressions=[
            Regression(
                "Y",
                [Term("X", label="slope", transform=TransformCall("log", "X", {}))],
            )
        ],
        defined_params=[DefinedParam("slope", "slope * 2")],
    )
    idata = az.from_dict(
        {"posterior": {"beta_Y": np.tile([4.0], (1, 4, 1))}},
        coords={"Y_predictors": ["X"]},
        dims={"beta_Y": ["Y_predictors"]},
    )
    with pytest.raises(DuplicateEquationError, match="log\\(X\\)") as exc:
        build_effects_summary(spec, idata)
    assert "slope*X" not in str(exc.value)


def test_distinct_name_keeps_label_posterior() -> None:
    """Injected coefficient 4 stays on the label; the defined row is separate."""
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [1.0, 2.0, 3.0, 4.0]})
    model = pathmc.model("Y ~ a*X\ntwice := a * 2", data=df)
    order = [str(v) for v in model.pymc_model.coords["Y_predictors"]]
    vec = [4.0 if name == "X" else 0.0 for name in order]
    model._idata = az.from_dict(
        {"posterior": {"beta_Y": np.tile(vec, (1, 4, 1))}},
        coords={"Y_predictors": order},
        dims={"beta_Y": ["Y_predictors"]},
    )
    summary = model.effects_summary()
    assert summary.loc["a", "mean"] == pytest.approx(4.0)
    assert summary.loc["twice", "mean"] == pytest.approx(8.0)
    assert "a" in summary.index
    assert list(summary.index).count("a") == 1


def test_public_model_rejects_the_issue_probe() -> None:
    df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0], "Y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(
        DuplicateEquationError, match="'a' is already the coefficient label"
    ):
        pathmc.model("Y ~ a*X\na := a * 2", data=df)
