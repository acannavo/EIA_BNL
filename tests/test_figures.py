"""Figure spec and both renderers."""
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from eiakit.figures import Band, ErrorBars, Figure, Line, Points, Text, lab  # noqa: E402
from eiakit.figures.export import save  # noqa: E402
from eiakit.figures.render_plotly import _decimate, to_plotly  # noqa: E402
from eiakit.figures.spec import legend_key  # noqa: E402


def _fig():
    x = np.linspace(1, 100, 500)
    f = Figure(rows=1, cols=2, letters=True, title=lab(r"$\bar z$ test", "z̄ test"))
    p = f.panel(0, 0)
    p.xlog = p.ylog = p.y2log = True
    p.add(Line(x, x ** 0.4, "z = 2+", "red"), Line(x, x ** 0.3, "z = 10+", "blue"),
          Line(x, x, "total", "black", dash="dash"), Line(x, 1 + x / 50, "zbar", "red", y2=True),
          Points(x[::50], x[::50] ** 0.4, "points", "gray"))
    q = f.panel(0, 1)
    q.add(ErrorBars(np.arange(5.0), np.arange(5.0), "err", "green", ylo=np.arange(5.0) - 0.5,
                    yhi=np.arange(5.0) + 1),
          Band(np.arange(5.0), np.zeros(5), "band", "orange", y_upper=np.ones(5)))
    q.texts.append(Text(0.1, 0.9, "note"))
    q.hlines.append((1.0, "black", "dot"))
    return f


def test_render_and_save(tmp_path):
    out = save(_fig(), tmp_path / "t")
    assert {p.suffix for p in out} == {".pdf", ".png", ".html"}
    html = (tmp_path / "t.html").read_text(encoding="utf-8")
    assert "plotly" in html and ("z̄ test" in html or "z\\u0304 test" in html)


def test_plotly_legend_order_and_secondary_axis():
    pf = to_plotly(_fig())
    names = {t.name: t.legendrank for t in pf.data if t.showlegend}
    assert names["z = 2+"] < names["z = 10+"] < names["total"]
    assert any(t.yaxis not in (None, "y", "y1") for t in pf.data if t.name == "zbar")


def test_legend_key():
    names = ["total", "z = 10+", r"$z$ = 3+", "z̄(E)", "z = 1+"]
    assert sorted(names, key=legend_key) == ["z = 1+", r"$z$ = 3+", "z = 10+", "total", "z̄(E)"]


def test_decimate_keeps_extremes():
    x = np.arange(100000.0)
    y = np.zeros_like(x)
    y[54321] = 7.0
    y[777] = -3.0
    xd, yd = _decimate(x, y, 1000)
    assert len(xd) <= 1100 and yd.max() == 7.0 and yd.min() == -3.0


def test_errorbar_bounds_like_legacy():
    from eiakit.figures.cbs_study import _bounds
    u = np.array([100.0, 100.0, 100.0])
    lo, hi = _bounds(u, np.array([90.0, np.nan, 120.0]), np.array([110.0, 105.0, 95.0]))
    assert list(lo) == [90.0, 100.0, 95.0] and list(hi) == [110.0, 105.0, 120.0]
