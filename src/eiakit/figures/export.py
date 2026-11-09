"""Write a figure as PDF + PNG (matplotlib, PRL style) and HTML (plotly, interactive)."""
from __future__ import annotations

import html as _html
from pathlib import Path

import matplotlib.pyplot as plt

from .render_mpl import to_matplotlib
from .render_plotly import to_plotly


def save(fig, base: Path, formats=("pdf", "png", "html"), dpi: int = 200,
         font_scale: float = 1.0) -> list[Path]:
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    if {"pdf", "png"} & set(formats):
        m = to_matplotlib(fig, font_scale)
        for f in ("pdf", "png"):
            if f in formats:
                p = base.with_suffix(f".{f}")
                m.savefig(p, dpi=dpi, bbox_inches="tight")
                out.append(p)
        plt.close(m)
    if "html" in formats:
        p = base.with_suffix(".html")
        write_html_page(p, str(base.name), [fig])
        out.append(p)
    return out


def figure_div(fig, include_js: bool) -> str:
    pf = to_plotly(fig)
    cfg = {"displaylogo": False, "responsive": True,
           "toImageButtonOptions": {"format": "svg", "filename": "figure"}}
    return pf.to_html(full_html=False, include_plotlyjs="inline" if include_js else False,
                      config=cfg)


_CSS = """body{font-family:'Times New Roman',serif;margin:24px auto;max-width:1400px;
color:#222} h1{font-size:24px;margin-bottom:4px} h2{font-size:19px;margin-top:34px;
border-bottom:1px solid #ccc;padding-bottom:3px} .meta{color:#666;font-size:13px}
table{border-collapse:collapse;font-size:12px;margin:8px 0} td,th{border:1px solid #ddd;
padding:2px 6px;text-align:right} th{background:#f4f4f4} .note{color:#555;font-size:13px}
details{margin:6px 0} summary{cursor:pointer;color:#245}"""


def write_html_page(path: Path, title: str, items: list, meta: str | None = None) -> Path:
    """
    One self-contained HTML page (plotly.js included once, works offline). items: spec
    Figures, ('h2', text), ('p', text), ('html', raw) or ('table', DataFrame, caption).
    """
    parts, js = [], True
    for it in items:
        if isinstance(it, tuple):
            kind = it[0]
            if kind == "h2":
                parts.append(f"<h2>{_html.escape(it[1])}</h2>")
            elif kind == "p":
                parts.append(f"<p class='note'>{_html.escape(it[1])}</p>")
            elif kind == "html":
                parts.append(it[1])
            elif kind == "table":
                cap = f"<summary>{_html.escape(it[2])}</summary>" if len(it) > 2 else ""
                tab = it[1].to_html(index=False, na_rep="", float_format=lambda v: f"{v:.6g}")
                parts.append(f"<details open>{cap}{tab}</details>")
        else:
            parts.append(figure_div(it, js))
            js = False
            for n in it.notes:
                parts.append(f"<p class='note'>{_html.escape(n)}</p>")
    head = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{_html.escape(title)}"
            f"</title><style>{_CSS}</style></head><body><h1>{_html.escape(title)}</h1>")
    if meta:
        head += f"<div class='meta'>{meta}</div>"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + "\n".join(parts) + "</body></html>", encoding="utf-8")
    return path
