"""spec.Figure -> plotly Figure (interactive: zoom, pan, click legend to hide / show)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .spec import Band, ErrorBars, Line, Points, as_html, legend_key

_DASH = {"solid": "solid", "dash": "dash", "dot": "dot", "dashdot": "dashdot"}
_MARK = {"o": "circle", "s": "square", "^": "triangle-up", "v": "triangle-down",
         "D": "diamond", "x": "x"}


def _clean(a):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), a, None)


def to_plotly(fig, max_points: int = 1500):
    specs = [[{"secondary_y": bool(fig.panels.get((r, c)) and fig.panels[(r, c)].has_y2)}
              for c in range(fig.cols)] for r in range(fig.rows)]
    titles = [as_html(fig.panels[(r, c)].title) if (r, c) in fig.panels else ""
              for r in range(fig.rows) for c in range(fig.cols)]
    if fig.letters:
        k, out = 0, []
        for t in titles:
            out.append(f"({'abcdefghijklmnop'[k]}) {t}" if t else "")
            k += bool(t)
        titles = out
    height = fig.height_px or int(330 * fig.rows + 140)
    pf = make_subplots(rows=fig.rows, cols=fig.cols, specs=specs, subplot_titles=titles,
                       shared_xaxes=fig.sharex, shared_yaxes=fig.sharey,
                       horizontal_spacing=0.07 if fig.cols > 1 else 0.02,
                       vertical_spacing=(min(0.14, (55 if fig.height_ratios else 95) / height)
                                         if fig.rows > 1 else 0.02),
                       row_heights=[h / sum(fig.height_ratios) for h in fig.height_ratios]
                       if fig.height_ratios else None)
    shown = set()
    for (r, c), p in sorted(fig.panels.items()):
        for t in p.traces:
            name = as_html(t.name) if t.name else None
            group = t.group or name
            first = bool(name) and name not in shown
            if name:
                shown.add(name)
            for tr in _traces(t, name, group, first, max_points):
                pf.add_trace(tr, row=r + 1, col=c + 1,
                             secondary_y=bool(t.y2) if specs[r][c]["secondary_y"] else None)
        kw = dict(row=r + 1, col=c + 1)
        pf.update_xaxes(type="log" if p.xlog else "linear", showgrid=p.grid,
                        dtick=1 if p.xlog else None,
                        title_text=as_html(p.xlabel) if (not fig.sharex or r == fig.rows - 1)
                        else None, **kw)
        if p.xlim:
            pf.update_xaxes(range=_rng(p.xlim, p.xlog), **kw)
        yk = dict(secondary_y=False) if specs[r][c]["secondary_y"] else {}
        pf.update_yaxes(type="log" if p.ylog else "linear", showgrid=p.grid,
                        dtick=1 if p.ylog else None,
                        title_text=as_html(p.ylabel) if (not fig.sharey or c == 0) else None,
                        exponentformat="power", **kw, **yk)
        if p.ylim:
            pf.update_yaxes(range=_rng(p.ylim, p.ylog), **kw, **yk)
        elif p.headroom:
            ys = [np.asarray(v, float) for t in p.traces if not t.y2
                  for v in (t.y, getattr(t, "yhi", None), getattr(t, "y_upper", None))
                  if v is not None]
            ys = np.concatenate([v[np.isfinite(v)] for v in ys]) if ys else np.array([])
            if ys.size:
                lo, hi = float(ys.min()), float(ys.max())
                pad = (hi - lo) * 0.05
                rng = (lo - pad, hi + (hi - lo) * p.headroom) if not p.ylog else \
                    (lo / 1.2, hi * (hi / lo) ** p.headroom)
                pf.update_yaxes(range=_rng(rng, p.ylog), **kw, **yk)
        if specs[r][c]["secondary_y"]:
            pf.update_yaxes(type="log" if p.y2log else "linear", showgrid=False,
                            tickvals=[1, 2, 3, 5, 10, 20, 30, 50] if p.y2log else None,
                            title_text=as_html(p.y2label), secondary_y=True, **kw)
            if p.y2lim:
                pf.update_yaxes(range=_rng(p.y2lim, p.y2log), secondary_y=True, **kw)
        for y, color, dash in p.hlines:
            pf.add_hline(y=y, line_color=color, line_dash=_DASH.get(dash, "solid"),
                         line_width=1, **kw)
        if not p.traces and p.empty_text:
            pf.add_annotation(text=p.empty_text, showarrow=False, xref="x domain",
                              yref="y domain", x=0.5, y=0.5, **kw)
        for tx in p.texts:
            pf.add_annotation(text=("<b>%s</b>" if tx.bold else "%s") % as_html(tx.text),
                              showarrow=False, xref="x domain", yref="y domain", x=tx.x,
                              y=tx.y, xanchor=tx.ha, yanchor="top",
                              font=dict(color=tx.color), **kw)
    pf.update_layout(template="simple_white",
                     title=as_html(fig.title) if fig.title else None,
                     height=height,
                     legend=dict(itemclick="toggle", itemdoubleclick="toggleothers",
                                 groupclick="togglegroup"),
                     hovermode="closest", showlegend=fig.legend != "none",
                     margin=dict(l=70, r=70, t=90 if fig.title else 60, b=60),
                     font=dict(family="Times New Roman, serif", size=13))
    pf.update_xaxes(ticks="inside", mirror=True, showline=True)
    pf.update_yaxes(ticks="inside", showline=True)
    return pf


def _rng(lim, log):
    lo, hi = lim
    return [np.log10(lo), np.log10(hi)] if log else [lo, hi]


def _decimate(x, y, n):
    """Keep at most ~n points for the browser (min / max per bin keeps the peaks)."""
    if len(x) <= n:
        return x, y
    k = int(np.ceil(len(x) / (n / 2)))
    m = len(x) // k * k
    xs, ys = x[:m].reshape(-1, k), y[:m].reshape(-1, k)
    with np.errstate(invalid="ignore"):
        ya = np.nan_to_num(ys, nan=-np.inf)
        i1, i2 = np.argmax(ya, 1), np.argmin(np.nan_to_num(ys, nan=np.inf), 1)
    r = np.arange(len(xs))
    idx = np.sort(np.stack([i1, i2], 1), 1)
    xo = xs[r[:, None], idx].ravel()
    yo = ys[r[:, None], idx].ravel()
    return np.concatenate([xo, x[m:]]), np.concatenate([yo, y[m:]])


def _traces(t, name, group, first, max_points):
    x, y = np.asarray(t.x, float), np.asarray(t.y, float)
    k = legend_key(name)
    common = dict(name=name or "", legendgroup=group, showlegend=first, opacity=t.alpha,
                  legendrank=1000 + k[1] if k[0] == 0 else 2000)
    if isinstance(t, Line):
        x, y = _decimate(x, y, max_points)
        yield go.Scattergl(x=_clean(x), y=_clean(y), mode="lines",
                           line=dict(color=t.color, width=t.width * 1.4,
                                     dash=_DASH.get(t.dash, "solid")), **common)
    elif isinstance(t, Points):
        yield go.Scattergl(x=_clean(x), y=_clean(y), mode="markers",
                           marker=dict(color=t.color if t.filled else "rgba(0,0,0,0)",
                                       symbol=_MARK.get(t.marker, "circle"), size=t.size * 1.3,
                                       line=dict(color=t.color, width=1)), **common)
    elif isinstance(t, ErrorBars):
        err = {}
        if t.ylo is not None:
            err = dict(error_y=dict(type="data", symmetric=False,
                                    array=_clean(np.asarray(t.yhi, float) - y),
                                    arrayminus=_clean(y - np.asarray(t.ylo, float)),
                                    color=t.color, thickness=1, width=3))
        yield go.Scatter(x=_clean(x), y=_clean(y),
                         mode="lines+markers" if t.connect else "markers",
                         marker=dict(color=t.color, symbol=_MARK.get(t.marker, "circle"),
                                     size=t.size * 1.3),
                         line=dict(color=t.color, width=1), text=t.hover, **err, **common)
    elif isinstance(t, Band):
        yield go.Scatter(x=_clean(np.concatenate([x, x[::-1]])), mode="lines",
                         y=_clean(np.concatenate([y, np.asarray(t.y_upper, float)[::-1]])),
                         fill="toself", fillcolor=t.color, line=dict(width=0),
                         **{**common, "opacity": t.alpha if t.alpha < 1 else 0.25})
