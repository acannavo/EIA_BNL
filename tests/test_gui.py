"""The eiakit window, headless (QT_QPA_PLATFORM=offscreen); skipped without PySide6 or data."""
import os
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("EIA_DATA_ROOT", "/nonexistent"))
pytest.importorskip("PySide6")
pytestmark = pytest.mark.skipif(not (ROOT / "20250611").is_dir(), reason="needs the data")


def test_window_tabs_and_cbs_fitter(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIA_PROCESSED_ROOT", str(tmp_path / "processed"))
    monkeypatch.setenv("EIA_STUDIES_ROOT", str(tmp_path / "studies"))
    monkeypatch.setenv("EIA_FITS_ROOT", str(tmp_path / "fits"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from eiakit.gui.app import MainWindow
    w = MainWindow()
    assert w.runs.table.rowCount() == len(list(w.db.runs()))
    assert w.runs.table.item(0, 3).text() == "missing"        # nothing processed yet
    f = w.fitter
    f.select_run("20250611")
    assert f.s is not None and f.s.source.startswith("legacy")
    f.zlist.setCurrentRow(2)
    f.table.selectRow(0)
    app.processEvents()
    z = f.z()
    # hover on a measured point shows its voltage
    import numpy as np
    from matplotlib.backend_bases import MouseEvent
    f.draw()
    f.canvas.draw()
    te, ye = f.s.exp[z]
    j = int(np.flatnonzero(np.isfinite(f.s.exp_U[z]))[0])
    x, y = f.ax.transData.transform((te[j], ye[j]))
    f.on_hover(MouseEvent("motion_notify_event", f.canvas, x, y))
    assert f._hover.get_visible() and f._hover.get_text().startswith(f"U = {f.s.exp_U[z][j]:g} V")
    f.s.fit(z, 0)
    p = f.s.save(w.paths.fits, "test")
    assert p.is_file()
    w.runs.refresh()
    col = [w.runs.table.horizontalHeaderItem(c).text()
           for c in range(w.runs.table.columnCount())].index("CBS fit")
    row = [w.runs.table.item(r, 0).text() for r in range(w.runs.table.rowCount())] \
        .index("20250611")
    assert w.runs.table.item(row, col).text() == "store"
    w.peaks.run_box.setCurrentText("20250611")
    w.peaks.ulist.setCurrentRow(0)
    assert w.peaks.table.rowCount() > 0
    w.baseline.run_box.setCurrentText("20250611")
    w.baseline.load()
    assert abs(w.baseline.value.value() - 0.4636) < 0.01
    # Peaks tab: species colours, zoom in x and y, hover, click-to-lock
    pk = w.peaks
    pk.load_U(pk.ulist.currentRow())
    i = 3
    pk.table.selectRow(i)
    app.processEvents()
    r = pk.s.fits.iloc[i]
    full_y = (pk.s.y.min(), pk.s.y.max())
    lo, hi = pk.ax.get_ylim()
    assert hi - lo < 0.8 * (full_y[1] - full_y[0])             # y follows the zoom
    pk.canvas.draw()
    x, y = pk.ax.transData.transform((r.TOF * 1e6, r.B))
    pk.on_hover(MouseEvent("motion_notify_event", pk.canvas, x, y))
    assert pk._hover.get_visible() and pk._hover.get_text().startswith("Nb")
    pk.mode.setCurrentIndex(1)                                # lock the peak position
    pk.on_click(MouseEvent("button_press_event", pk.canvas, x, y, button=1))
    assert pk.s.fits.iloc[i].locked == "μ"
    pk.mode.setCurrentIndex(2)                                # lock the background
    pk.on_click(MouseEvent("button_press_event", pk.canvas, x, y, button=1))
    assert pk.s.fits.iloc[i].locked == "μB"
    assert pk.table.item(i, pk.COLS.index("locked")).text() == "μB"
    pk.unlock()
    assert pk.s.fits.iloc[i].locked == ""
    # drag the peak bottom: position and height locked
    pk.table.selectRow(i)
    app.processEvents()
    pk.canvas.draw()
    r = pk.s.fits.iloc[i]
    bx, by = pk.ax.transData.transform((r.TOF * 1e6, r.Yield + r.B))
    pk.mode.setCurrentIndex(0)
    pk.on_click(MouseEvent("button_press_event", pk.canvas, bx, by, button=1))
    assert pk._drag is not None
    tx, ty = pk.ax.transData.transform((r.TOF * 1e6 + 0.01, r.B + 1.05 * r.Yield))
    pk.on_hover(MouseEvent("motion_notify_event", pk.canvas, tx, ty))
    pk.on_release(MouseEvent("button_release_event", pk.canvas, tx, ty, button=1))
    r2 = pk.s.fits.iloc[i]
    assert r2.locked == "μhB" and abs(r2.Yield / r.Yield - 1.05) < 0.02
    assert abs(r2.TOF * 1e6 - r.TOF * 1e6 - 0.01) < 1e-3
    # recap: one bar per good line, click selects
    assert len(pk._rbars) == int((pk.s.fits.success & ~pk.s.fits.excluded).sum())
    pk.rcanvas.draw()
    b, k = pk._rbars[0]
    cx, cy = pk.rax.transData.transform((b.get_x() + b.get_width() / 2, b.get_height() / 2))
    pk.recap_click(MouseEvent("button_press_event", pk.rcanvas, cx, cy, button=1))
    assert pk.sel() == k
    pk.s.unlock(i)
    pk.show_table()
    labels = [ln.get_label() for ln in pk.ax.get_lines()]
    assert "total (sum of all lines)" in labels
    tot = pk.s.total_curve()
    assert tot is not None and tot.shape == pk.s.tof.shape
    # common background: 2 points, a tuning point, remove it; negative Corr
    def click(t_us, v):
        X, Y = pk.ax.transData.transform((t_us, v))
        pk.on_click(MouseEvent("button_press_event", pk.canvas, X, Y, button=1))
    pk.full()
    pk.canvas.draw()
    pk.mode.setCurrentIndex(3)
    t1, t2 = pk.ax.get_xlim()
    click(t1 + 0.2 * (t2 - t1), -0.002)
    assert pk.s.bg is None and pk._bg_click is not None
    click(t1 + 0.8 * (t2 - t1), -0.001)
    assert pk.s.bg_on and abs(pk.s.bg_level + 0.0015) < 2e-4
    assert pk.s.fits.locked.str.contains("Bc").all() and pk.bg_use.isChecked()
    pk.mode.setCurrentIndex(4)
    tt = r.TOF * 1e6
    click(tt, -0.0005)
    assert len(pk.s.bg["tune"]) == 1
    pk.canvas.draw()
    pk.mode.setCurrentIndex(5)
    click(tt, -0.0005)
    assert pk.s.bg["tune"] == []
    pk.bg_use.setChecked(False)
    assert not pk.s.fits.locked.str.contains("Bc").any()
    pk.corr.setValue(-0.01)
    pk.refit()
    rr = pk.s.fits.iloc[i]
    assert abs((rr.win_hi - rr.win_lo) / rr.t0 - 0.02) < 1e-6
    w.close()
