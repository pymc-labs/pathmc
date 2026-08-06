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
"""Check that the freeze rewrite agrees with the real escaper.

``scripts/refreeze_tex.py`` transforms cached TeX textually, which makes it a
second implementation of the escaping in ``pathmc.introspect`` and therefore
free to drift from it. This closes that gap empirically: emit the LaTeX for a
corpus of specs at the unpatched revision, emit it again with the patch
applied, and require that transforming the former reproduces the latter
exactly. If that holds, rewriting the freeze artifacts is equivalent to
re-executing the notebooks as far as the equations are concerned.

Usage::

    python scripts/verify_refreeze_tex.py emit    old.json   # at PR head
    python scripts/verify_refreeze_tex.py emit    new.json   # with the patch
    python scripts/verify_refreeze_tex.py compare old.json new.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from refreeze_tex import rewrite_tex  # noqa: E402

_BARE_UNDERSCORE = re.compile(r"(?<!\\)_(?!\{)")
_DOUBLE_SUBSCRIPT = re.compile(r"_\{[^{}]*_[^{}]*\}")

_FAMILIES = ["gaussian", "bernoulli", "poisson", "negbinomial", "studentt"]

# Names chosen to cover the shapes that reach TeX: single token, one
# underscore (renders wrong but parses), and two (a hard KaTeX error).
_SPECS: list[tuple[str, dict]] = [
    ("y ~ x", {}),
    ("birth_weight ~ tv_spend", {}),
    ("birth_weight ~ tv_spend + sales_q_1", {}),
    ("y_t_1 ~ x_i_j + z", {}),
    ("sales ~ logistic_saturation(tv_spend, lam=lam_tv)", {}),
    ("sales ~ adstock(tv_spend, decay=decay_tv)", {}),
    ("y ~ hsgp(x, m=20, c=1.5)", {}),
    ("outcome_obs ~ treat_flag * region_id", {}),
    ("M_obs ~ X_raw", {}),
    ("awareness_survey ~ search_traffic + display_spend", {}),
]


def _emit() -> dict[str, str]:
    import pathmc

    out: dict[str, str] = {}
    for spec, kwargs in _SPECS:
        lhs = spec.split("~")[0].strip()
        for family in _FAMILIES:
            key = f"{spec}|{family}"
            try:
                model = pathmc.model(spec, families={lhs: family}, **kwargs)
                out[key] = "\n".join((
                    model.equations()._repr_latex_(),
                    model.priors()._repr_latex_(),
                ))
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                out[key] = f"__ERROR__ {type(exc).__name__}: {exc}"
    return out


def _compare(old: dict[str, str], new: dict[str, str]) -> int:
    keys = sorted(set(old) | set(new))
    mismatches: list[str] = []
    skipped = 0
    checked = 0
    still_bad: list[str] = []

    for key in keys:
        a, b = old.get(key), new.get(key)
        if a is None or b is None:
            mismatches.append(f"{key}: present in only one revision")
            continue
        if a.startswith("__ERROR__") or b.startswith("__ERROR__"):
            # A spec the parser rejects at both revisions carries no signal.
            if a != b:
                mismatches.append(f"{key}: error differs\n  old {a}\n  new {b}")
            else:
                skipped += 1
            continue
        checked += 1
        if rewrite_tex(a) != b:
            mismatches.append(
                f"{key}: transform != patched output\n"
                f"  old        {a!r}\n"
                f"  transform  {rewrite_tex(a)!r}\n"
                f"  patched    {b!r}"
            )
        if _BARE_UNDERSCORE.search(b) or _DOUBLE_SUBSCRIPT.search(b):
            still_bad.append(f"{key}: {b!r}")

    print(f"checked {checked} case(s), skipped {skipped} unparseable")
    for m in mismatches:
        print("MISMATCH", m)
    for s in still_bad:
        print("INVARIANT VIOLATED", s)

    if mismatches or still_bad:
        print(f"\nFAIL: {len(mismatches)} mismatch(es), {len(still_bad)} violation(s)")
        return 1
    print("\nOK: textual rewrite reproduces the patched output on every case")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "emit":
        Path(argv[2]).write_text(json.dumps(_emit(), indent=2))
        print(f"wrote {argv[2]}")
        return 0
    if len(argv) >= 4 and argv[1] == "compare":
        return _compare(
            json.loads(Path(argv[2]).read_text()),
            json.loads(Path(argv[3]).read_text()),
        )
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
