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
"""Shared posterior-density rendering for models and result objects.

Centralizes the KDE density plot behind ``plot_dist`` on
:class:`~pathmc._model.PathModel`, :class:`~pathmc.effects.EffectResult`,
:class:`~pathmc.simulate.EstimandResult`, and
:class:`~pathmc.simulate.DoResult`. A leaf module so the model layer and
the result objects can both import it without cycles.

The renderer returns a :class:`matplotlib.figure.Figure`, accepts an
optional ``ax`` for composition, and imports matplotlib lazily inside the
function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import arviz as az
import numpy as np


if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = ["plot_density"]


def plot_density(
    draws: np.ndarray,
    *,
    label: str | None = None,
    ref: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    color: str = "C0",
) -> matplotlib.figure.Figure:
    """Plot a posterior density with an optional reference line.

    Draws a KDE curve with a translucent fill and a legend entry. Optionally
    marks a reference value with a dashed vertical line.

    Parameters
    ----------
    draws : np.ndarray
        Flat ``(n_samples,)`` array of posterior draws.
    label : str | None
        Legend label. Omitted when ``None``.
    ref : float | None
        Reference value marked with a dashed vertical line (e.g. ``0`` for a
        null effect). No line when ``None``.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. Creates a new figure if ``None``.
    color : str
        Line and fill color (default ``"C0"``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the density plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = cast("matplotlib.figure.Figure", ax.get_figure())

    x_kde, y_kde, _ = az.kde(np.asarray(draws))
    ax.plot(x_kde, y_kde, color=color, lw=2, label=label)
    ax.fill_between(x_kde, y_kde, alpha=0.3, color=color)
    if ref is not None:
        ax.axvline(ref, color=color, linestyle="--", linewidth=1.5)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best")
    ax.set_ylabel("Density")
    fig.tight_layout()
    return fig
