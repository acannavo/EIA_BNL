"""spec.Figure -> matplotlib Figure (PRL style)."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D

from .spec import Band, ErrorBars, Line, Points, as_mpl, legend_key
from .style import prl_rc

_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}


def to_matplotlib(fig, font_scale: float = 1.0):
    # object API, no pyplot: independent of the GUI backend (MPLBACKEND), no window opened
    with mpl.rc_context(prl_rc(font_scale)):
        mfig = MplFigure(figsize=fig.size_in)
        axs = mfig.subplots(fig.rows, fig.cols, squeeze=False, sharex=fig.sharex,
                            sharey=fig.sharey)
        handles, seen = [], set()
        for (r, c), p in sorted(fig.panels.items()):
            ax = axs[r][c]
            ax2 = ax.twinx() if p.has_y2 else None
            if not p.traces and p.empty_text:
                ax.text(0.5, 0.5, p.empty_text, ha="center", va="center",
                        transform=ax.transAxes)
            for t in p.traces:
                a = ax2 if (t.y2 and ax2 is not None) else ax
                h = _draw(a, t)
                key = as_mpl(t.name) if t.name else None
                if key and key not in seen and h is not None:
                    seen.add(key)
                    handles.append((h, key))
            for y, color, dash in p.hlines:
                ax.axhline(y, color=color, ls=_DASH.get(dash, dash), lw=0.8)
            for tx in p.texts:
                ax.text(tx.x, tx.y, as_mpl(tx.text), transform=ax.transAxes, ha=tx.ha,
                        va="top", fontsize=tx.size * font_scale, color=tx.color,
                        fontweight="bold" if tx.bold else "normal")
            ax.set_xscale("log" if p.xlog else "linear")
            ax.set_yscale("log" if p.ylog else "linear")
            if p.xlim:
                ax.set_xlim(*p.xlim)
            if p.ylim:
                ax.set_ylim(*p.ylim)
            elif p.headroom:
                lo, hi = ax.get_ylim()
                ax.set_ylim(lo, hi * (hi / lo) ** p.headroom if p.ylog and lo > 0
                            else hi + (hi - lo) * p.headroom)
            if p.title:
                ax.set_title(as_mpl(p.title))
            if p.xlabel and (not fig.sharex or r == fig.rows - 1):
                ax.set_xlabel(as_mpl(p.xlabel))
            if p.ylabel and (not fig.sharey or c == 0):
                ax.set_ylabel(as_mpl(p.ylabel))
            if p.grid:
                ax.grid(True, which="both", ls="--", alpha=0.25, lw=0.5)
            if ax2 is not None:
                ax2.set_yscale("log" if p.y2log else "linear")
                if p.y2lim:
                    ax2.set_ylim(*p.y2lim)
                if p.y2label:
                    ax2.set_ylabel(as_mpl(p.y2label))
                ax2.set_xlim(ax.get_xlim())
            if fig.legend == "panel" and p.traces:
                hs = [(_proxy(t), as_mpl(t.name)) for t in p.traces if t.name]
                if hs:
                    ax.legend([h for h, _ in hs], [n for _, n in hs], loc="best")
            if fig.letters:
                k = r * fig.cols + c
                ax.text(0.03, 0.965, f"({'abcdefghijklmnop'[k]})", transform=ax.transAxes,
                        ha="left", va="top", fontweight="bold", zorder=10,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none",
                                  boxstyle="round,pad=0.15"))
        for r in range(fig.rows):
            for c in range(fig.cols):
                if (r, c) not in fig.panels:
                    axs[r][c].set_visible(False)
        handles.sort(key=lambda hn: legend_key(hn[1]))         # stable: others keep order
        h_in = fig.size_in[1]
        top = 1.0
        if fig.title:
            mfig.suptitle(as_mpl(fig.title), y=0.995, va="top")
            top -= 0.26 / h_in
        if fig.legend == "figure" and handles:
            n = len(handles)
            ncol = n if n <= 6 else int(np.ceil(n / 3))
            mfig.legend([h for h, _ in handles], [n_ for _, n_ in handles], loc="upper center",
                        ncol=ncol, frameon=False, bbox_to_anchor=(0.5, top))
            nrow = int(np.ceil(n / ncol))
            top -= nrow * 0.15 / h_in + 0.02 / h_in
            mfig.tight_layout(rect=(0, 0, 1, top))
        else:
            mfig.tight_layout()
        return mfig


def _proxy(t):
    if isinstance(t, Line):
        return Line2D([], [], color=t.color, lw=t.width, ls=_DASH.get(t.dash, t.dash))
    if isinstance(t, (Points, ErrorBars)):
        return Line2D([], [], color=t.color, marker=t.marker, ls="none")
    return Line2D([], [], color=t.color, lw=6, alpha=0.3)


def _draw(ax, t):
    x, y = np.asarray(t.x, float), np.asarray(t.y, float)
    if isinstance(t, Line):
        (h,) = ax.plot(x, y, color=t.color, lw=t.width, ls=_DASH.get(t.dash, t.dash),
                       alpha=t.alpha)
        return h
    if isinstance(t, Points):
        (h,) = ax.plot(x, y, ls="none", marker=t.marker, ms=t.size, color=t.color,
                       mfc=t.color if t.filled else "none", alpha=t.alpha)
        return h
    if isinstance(t, ErrorBars):
        lo = y - np.asarray(t.ylo, float) if t.ylo is not None else None
        hi = np.asarray(t.yhi, float) - y if t.yhi is not None else None
        yerr = None if lo is None else np.vstack([np.abs(lo), np.abs(hi)])
        h = ax.errorbar(x, y, yerr=yerr, fmt=("-" if t.connect else "") + t.marker,
                        color=t.color, ms=t.size, capsize=t.capsize, elinewidth=0.8,
                        lw=0.8, alpha=t.alpha)
        return h
    if isinstance(t, Band):
        return ax.fill_between(x, y, np.asarray(t.y_upper, float), color=t.color,
                               alpha=t.alpha if t.alpha < 1 else 0.25, lw=0)
    return None
