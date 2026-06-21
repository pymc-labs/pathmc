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
"""Shared repr infrastructure for pathmc result objects.

Provides :class:`ResultReprMixin` and :class:`ReprSpec` so that every result
object gets a consistent terminal repr (compact, one-liner, never raises) and
a rich HTML notebook view (stats table + optional methods hint) from a single
source of truth.

Usage
-----
Stat-shaped results (numbers to display in a table) implement two hooks:

.. code-block:: python

    class MyResult(ResultReprMixin):
        def _repr_compact(self) -> str:
            return f"MyResult(mean={self.mean:.2f})"

        def _repr_spec(self) -> ReprSpec:
            return ReprSpec(
                title="My Result",
                rows=[["Mean", f"{self.mean:.4f}"]],
                footer="Methods: .hdi() .plot()",
            )

Narrative results (prose verdicts, not stat grids) only implement
``_repr_compact`` and define their own ``_repr_html_``; the mixin's
``_repr_html_`` is never reached for those classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ReprSpec", "ResultReprMixin"]

_CSS = """
<style>
  .pathmc-result {
    font-family: monospace;
    border-collapse: collapse;
    margin: 4px 0;
  }
  .pathmc-result th, .pathmc-result td {
    text-align: left;
    padding: 2px 12px 2px 4px;
    white-space: nowrap;
  }
  .pathmc-result thead th {
    border-bottom: 1px solid #ccc;
    font-weight: bold;
  }
  .pathmc-result-title {
    font-family: monospace;
    font-weight: bold;
    padding-bottom: 4px;
    white-space: nowrap;
  }
  .pathmc-result-footer {
    font-family: monospace;
    font-size: 0.85em;
    color: #666;
    margin-top: 4px;
  }
</style>
"""


@dataclass
class ReprSpec:
    """Structured description of a result for HTML rendering.

    Parameters
    ----------
    title : str
        Display title shown above the table, e.g. ``"ATE of X on Y"``.
    rows : list[list[str]]
        Pre-formatted cell values. When *columns* is ``None`` each row is
        ``[label, value]``; when *columns* is provided the rows must have
        the same length as *columns*.
    columns : list[str] or None
        Column headers. ``None`` selects the label:value two-column layout
        (no visible header row). Supply explicit headers for tabular results
        like :class:`~pathmc.simulate.DoResult`.
    footer : str or None
        Short hint shown below the table in HTML output only, e.g.
        ``"Methods: .hdi() .plot()"``.
    """

    title: str
    rows: list[list[str]] = field(default_factory=list)
    columns: list[str] | None = None
    footer: str | None = None


def _render_html(spec: ReprSpec) -> str:
    """Build an HTML table from a :class:`ReprSpec`.

    Parameters
    ----------
    spec : ReprSpec
        Structured description of the result.

    Returns
    -------
    str
        HTML string suitable for ``_repr_html_`` return values.
    """
    title_html = f'<div class="pathmc-result-title">{spec.title}</div>'

    if spec.columns is not None:
        header_cells = "".join(f"<th>{c}</th>" for c in spec.columns)
        thead = f"<thead><tr>{header_cells}</tr></thead>"
    else:
        thead = ""

    tbody_rows = []
    for row in spec.rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        tbody_rows.append(f"<tr>{cells}</tr>")
    tbody = f"<tbody>{''.join(tbody_rows)}</tbody>"

    table = f'<table class="pathmc-result">{thead}{tbody}</table>'

    if spec.footer:
        footer_html = f'<div class="pathmc-result-footer">{spec.footer}</div>'
    else:
        footer_html = ""

    return f"{_CSS}{title_html}{table}{footer_html}"


class ResultReprMixin:
    """Mixin that provides ``__repr__`` and ``_repr_html_`` for result objects.

    Subclasses must implement :meth:`_repr_compact` (required for all) and
    :meth:`_repr_spec` (required only for stat-shaped results; narrative
    classes that define their own ``_repr_html_`` may omit it).

    ``__repr__`` is guaranteed never to raise — it falls back to
    ``<ClassName>`` if :meth:`_repr_compact` raises.
    """

    def _repr_compact(self) -> str:  # pragma: no cover
        """Return a concise one-line string for use in ``__repr__``."""
        return f"<{type(self).__name__}>"

    def _repr_spec(self) -> ReprSpec:  # pragma: no cover
        """Return structured content for HTML rendering."""
        return ReprSpec(title=type(self).__name__)

    def __repr__(self) -> str:
        try:
            return self._repr_compact()
        except Exception:
            return f"<{type(self).__name__}>"

    def _repr_html_(self) -> str:
        try:
            return _render_html(self._repr_spec())
        except Exception:
            return f"<pre>{self!r}</pre>"
