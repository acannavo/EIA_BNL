"""Backend-independent figure description: a grid of panels with traces."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Label:
    """A label in matplotlib mathtext and in HTML (plotly)."""
    mpl: str
    html: str

    def __str__(self):
        return self.html


def lab(mpl: str, html: str | None = None) -> Label:
    return Label(mpl, html if html is not None else mpl)


def as_mpl(x) -> str:
    return x.mpl if isinstance(x, Label) else ("" if x is None else str(x))


def legend_key(name: str):
    """Legend order: charge states 'z = n+' first, numerically; then the rest as drawn."""
    import re
    m = re.match(r"\s*\$?z\$?\s*=\s*(\d+)", name or "")
    return (0, int(m.group(1))) if m else (1, 0)


def as_html(x) -> str:
    return x.html if isinstance(x, Label) else ("" if x is None else str(x))


@dataclass
class _Trace:
    x: np.ndarray
    y: np.ndarray
    name: str | Label | None = None      # legend entry (None = not in the legend)
    color: str = "black"
    group: str | None = None             # legend group: one toggle for related traces
    y2: bool = False                     # right axis of the panel
    alpha: float = 1.0
    hover: list | None = None


@dataclass
class Line(_Trace):
    width: float = 1.2
    dash: str = "solid"                  # solid | dash | dot | dashdot


@dataclass
class Points(_Trace):
    marker: str = "o"                    # o s ^ v D x
    size: float = 5.0
    filled: bool = True


@dataclass
class ErrorBars(_Trace):
    """Points with asymmetric errors: lower = y - ylo, upper = yhi - y (absolute bounds)."""
    ylo: np.ndarray | None = None
    yhi: np.ndarray | None = None
    marker: str = "o"
    size: float = 5.0
    capsize: float = 2.0
    connect: bool = False


@dataclass
class Band(_Trace):
    """Shaded band between y (lower) and y_upper."""
    y_upper: np.ndarray | None = None


@dataclass
class Text:
    x: float                             # axes fraction
    y: float
    text: str | Label
    size: float = 11
    bold: bool = False
    color: str = "black"


@dataclass
class Panel:
    title: str | Label | None = None
    xlabel: str | Label | None = None
    ylabel: str | Label | None = None
    y2label: str | Label | None = None
    xlog: bool = False
    ylog: bool = False
    y2log: bool = False
    xlim: tuple | None = None
    ylim: tuple | None = None
    y2lim: tuple | None = None
    traces: list = field(default_factory=list)
    texts: list = field(default_factory=list)
    hlines: list = field(default_factory=list)   # (y, color, dash)
    grid: bool = True
    empty_text: str | None = None                # drawn when the panel has no data

    def add(self, *traces):
        self.traces.extend(traces)
        return self

    @property
    def has_y2(self):
        return any(t.y2 for t in self.traces) or self.y2label is not None


@dataclass
class Figure:
    rows: int = 1
    cols: int = 1
    title: str | Label | None = None
    size_in: tuple = (6.4, 4.8)                  # matplotlib size (inches)
    height_px: int | None = None                 # plotly height
    sharex: bool = False
    sharey: bool = False
    panels: dict = field(default_factory=dict)   # (row, col) -> Panel
    legend: str = "figure"                       # figure | panel | none
    letters: bool = False                        # (a), (b), ... row by row
    notes: list = field(default_factory=list)    # lines shown under the HTML figure

    def panel(self, r: int, c: int) -> Panel:
        return self.panels.setdefault((r, c), Panel())
