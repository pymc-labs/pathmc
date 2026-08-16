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
"""Type aliases shared across pathmc modules, and their runtime checks.

Aliases live here when more than one module needs them. A type used by a
single module stays in that module, as :data:`pathmc.hsgp.TensorLike` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

BinsLike: TypeAlias = int | str | Sequence[float] | np.ndarray
"""What a histogram ``bins`` argument accepts.

Mirrors ``matplotlib.axes.Axes.hist``: a positive integer bin count, a
binning strategy name such as ``"auto"``, or a sequence of bin edges.
"""


def validate_bins(bins: BinsLike | None) -> None:
    """Raise if *bins* is not something ``Axes.hist`` can use.

    Only the bin count is validated. A strategy name or an edge sequence is
    left to matplotlib, which reports both more precisely.

    Parameters
    ----------
    bins : BinsLike | None
        The value passed by the caller. ``None`` is always allowed.

    Raises
    ------
    ValueError
        If *bins* is neither one of the accepted types nor, for an integer
        bin count, positive.
    """
    if bins is None:
        return
    if isinstance(bins, bool) or not isinstance(
        bins, (int, np.integer, Sequence, np.ndarray)
    ):
        raise ValueError(
            f"bins must be a positive integer, a binning strategy name, "
            f"a sequence of bin edges, or None, got {bins!r} "
            f"of type {type(bins).__name__}."
        )
    if isinstance(bins, (int, np.integer)) and bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins}.")
