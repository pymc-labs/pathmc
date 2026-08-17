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
"""Tests for pathmc.paths — local data dir resolution and GitHub fallback."""

import os
from pathlib import Path

import pandas as pd

from pathmc.paths import URLPath, create_data_url, data_dir


def test_create_data_url_defaults_to_main_branch() -> None:
    url = create_data_url()
    assert isinstance(url, URLPath)
    assert url.url == ("https://raw.githubusercontent.com/pymc-labs/pathmc/main/data")


def test_create_data_url_uses_branch() -> None:
    url = create_data_url("dev")
    assert url.url == ("https://raw.githubusercontent.com/pymc-labs/pathmc/dev/data")


def test_url_path_joins_and_is_path_like() -> None:
    file = create_data_url() / "mediation.csv"
    assert file.url == (
        "https://raw.githubusercontent.com/pymc-labs/pathmc/main/data/mediation.csv"
    )
    assert os.fspath(file) == file.url


def test_data_dir_is_local_path_inside_checkout() -> None:
    assert isinstance(data_dir, Path)
    csv = data_dir / "mediation.csv"
    assert csv.exists()


def test_local_dataset_loads_with_pandas() -> None:
    df = pd.read_csv(data_dir / "mediation.csv")
    assert list(df.columns) == ["X", "M", "Y"]
    assert len(df) == 500


def test_fallback_to_url_when_not_in_checkout(tmp_path: Path) -> None:
    """Isolated copy of paths.py (no data/ ancestor) resolves to URLPath."""
    import importlib.util
    import shutil
    import sys

    src = Path(__file__).parent.parent / "pathmc" / "paths.py"
    shutil.copy(src, tmp_path / "paths.py")
    spec = importlib.util.spec_from_file_location(
        "paths_isolated", str(tmp_path / "paths.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paths_isolated"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    assert isinstance(mod.data_dir, mod.URLPath)
    assert "raw.githubusercontent.com" in mod.data_dir.url


def test_fallback_does_not_trigger_on_stray_data_dir(tmp_path: Path) -> None:
    """A stray data/ folder in a parent must not confuse the resolver."""
    import importlib.util
    import shutil
    import sys

    stray = tmp_path / "data"
    stray.mkdir()
    (stray / "unrelated.csv").write_text("a,b\n1,2\n")

    child = tmp_path / "sub" / "pkg"
    child.mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent.parent / "pathmc" / "paths.py", child / "paths.py"
    )

    spec = importlib.util.spec_from_file_location(
        "paths_isolated_stray", str(child / "paths.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paths_isolated_stray"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    assert isinstance(mod.data_dir, mod.URLPath)
