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
