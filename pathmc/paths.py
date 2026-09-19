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
"""Locate example datasets, locally or from the latest version on GitHub.

:data:`data_dir` resolves to the local ``data/`` directory when this package
lives inside the repository checkout (contributor/development use), and
otherwise to a :class:`URLPath` pointing at ``data/`` on the ``main`` branch
of the upstream GitHub repository. Either result can be combined with a
filename and passed straight to ``pd.read_csv()``:

.. code-block:: python

    import pandas as pd
    from pathmc.paths import data_dir

    df = pd.read_csv(data_dir / "mediation.csv")

Local discovery checks only the direct parent of the ``pathmc`` package
directory and verifies it is the pathmc repository by looking for
``pyproject.toml`` with ``name = "pathmc"``. A pip-installed user always
falls back to GitHub — never to a stray ``data/`` folder in a neighbouring
project.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

URL = "https://raw.githubusercontent.com/pymc-labs/pathmc/{branch}/data"

__all__ = ["URLPath", "create_data_url", "data_dir"]


@dataclass
class URLPath(PathLike[str]):
    """A URL that can be used wherever a file path is expected.

    Implements the ``os.PathLike`` protocol and ``/`` so that code written
    against local paths (``pd.read_csv(data_dir / "mediation.csv")``) works
    unchanged when the data only exists on GitHub. pandas and polars both
    accept the resulting URL string and fetch it transparently.

    Parameters
    ----------
    url : str
        The URL to a data directory or file.
    """

    url: str

    def __fspath__(self) -> str:
        """Return the URL as a string when used as a file path."""
        return self.url

    def __truediv__(self, other: str) -> URLPath:
        """Combine the URL with another path component."""
        return URLPath(f"{self.url}/{other}")


def create_data_url(branch: str = "main") -> URLPath:
    """Return a :class:`URLPath` for the ``data/`` directory on *branch*.

    Parameters
    ----------
    branch : str
        The upstream repository branch to read from (default ``"main"``).

    Returns
    -------
    URLPath
        A URL-path object; combine with a filename using ``/``.
    """
    return URLPath(URL.format(branch=branch))


def _find_local_data_dir() -> Path | None:
    """Return the repository's ``data/`` directory, if this is a checkout.

    Checks only the direct parent of the ``pathmc/`` package directory
    (i.e. the repo root in an editable install) and verifies the
    ``pyproject.toml`` there belongs to pathmc before returning it.
    """
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "data"
    if not candidate.is_dir():
        return None
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file() and 'name = "pathmc"' in pyproject.read_text():
        return candidate
    return None


_local_data_dir = _find_local_data_dir()
data_dir: Path | URLPath = (
    _local_data_dir if _local_data_dir is not None else create_data_url("main")
)
"""Local repository ``data/`` when running from a checkout, else the ``main``
branch data directory on GitHub. Combine with a filename via ``/`` and read
directly (e.g. ``pd.read_csv(data_dir / "mediation.csv")``)."""
