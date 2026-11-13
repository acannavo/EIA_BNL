"""
The eiakit window (Steps 9-10).

    Runs       every run with the state of its stages (ok / stale / missing / failed and
               why); run what is stale, open a run report, start the CBS fit store
    Studies    figures of a study, method studies, snapshots and diffs, the HTML pages
    Settings   the switches of db/pipeline.toml; measure one with "eiakit compare"
    Peaks      raw SEM shots of one voltage: refit every line at the file voltage, move a
               fit window, drop a line; save -> replaces that voltage in the calibration
    FC baseline  the FC average and its pre-trigger region: pick the pedestal per run
    CBS fitter components per charge state on the current calibration: pick the peak,
               Fit (kT scan), kT from FWHM, add / remove labelled components, windows;
               save to the fit store (fits/<run>/cbs_fit.json); then "Run" updates the
               cbs and report stages and the studies

Every long action runs `python -m eiakit ...` in a separate process (QProcess): the window
stays responsive, the output goes to the log at the bottom, exactly as in the terminal.
Start: eiakit gui
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..db import load_database
from ..paths import get_paths

STATE_COLOR = {"ok": "#c8e6c9", "stale": "#ffe0b2", "missing": "#eeeeee", "failed": "#ffcdd2"}
LABEL_COLOR = {"ultra_fast": "#9467bd", "fast": "#d62728", "slow": "#1f77b4",
               "component_4": "#8c564b"}


def _mj(run) -> str:
    e = getattr(run.laser, "energy_mJ", None) if getattr(run, "laser", None) else None
    return "?" if e is None else f"{e:g}"


def _open(path: Path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _btn(text, slot, tip=None):
    b = QPushButton(text)
    b.clicked.connect(slot)
    if tip:
        b.setToolTip(tip)
    return b


# ------------------------------------------------------------------------------ runner

class Runner:
    """Runs eiakit commands one at a time in a QProcess; output to the log widget."""

    def __init__(self, log: QPlainTextEdit, on_done=None):
        self.log, self.on_done = log, on_done
        self.proc: QProcess | None = None
        self.queue: list[list[str]] = []

    def busy(self) -> bool:
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def run(self, *args: str):
        self.queue.append([str(a) for a in args])
        if not self.busy():
            self._next()

    def stop(self):
        self.queue.clear()
        if self.busy():
            self.proc.kill()
            self.log.appendPlainText("[stopped]")

    def _next(self):
        if not self.queue:
            return
        args = self.queue.pop(0)
        self.log.appendPlainText(f"\n$ eiakit {' '.join(args)}")
        p = QProcess()
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._finished)
        env = QProcessEnvironment.systemEnvironment()      # EIA_* roots of this session
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("MPLBACKEND", "Agg")
        p.setProcessEnvironment(env)
        self.proc = p
        p.start(sys.executable, ["-m", "eiakit", *args])

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.rstrip("\n").splitlines():
            self.log.appendPlainText(line)

    def _finished(self, code, _status):
        self.log.appendPlainText(f"[exit code {code}]")
        if self.on_done:
            self.on_done()
        self._next()


# -------------------------------------------------------------------------------- runs

class RunsTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellDoubleClicked.connect(lambda r, c: self.open_report())
        lay.addWidget(self.table)
        row = QHBoxLayout()
        for b in (_btn("Refresh", self.refresh),
                  _btn("Run selected", lambda: self.run(False),
                       "eiakit run KEY: stages that are missing or stale"),
                  _btn("Run all", lambda: win.runner.run("run")),
                  _btn("Force rerun selected", lambda: self.run(True)),
                  _btn("Open report", self.open_report, "processed/<run>/40_report/report.html"),
                  _btn("Open processed folder", self.open_folder),
                  _btn("Start CBS fit store", self.fit_import,
                       "eiakit fit-import KEY: copy the legacy CBS fit into fits/<run>/"),
                  _btn("CBS fitter", self.to_fitter)):
            row.addWidget(b)
        row.addStretch()
        lay.addLayout(row)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.detail)
        self.table.itemSelectionChanged.connect(self.show_detail)
        self.status = {}

    def keys(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return [self.table.item(r, 0).text() for r in rows]

    def refresh(self):
        from ..cbs_store import store_path
        from ..pipeline import ORDER, stage_status
        db, paths = self.win.db, self.win.paths
        runs = list(db.runs())
        cols = ["Run", "Target", "Laser [mJ]", *ORDER, "CBS fit"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(runs))
        cache, self.status = {}, {}
        for r, run in enumerate(runs):
            vals = [run.key, run.target, _mj(run)]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
            for j, name in enumerate(ORDER):
                st = stage_status(db, run, name, paths.data, paths.processed, cache)
                self.status[(run.key, name)] = st
                it = QTableWidgetItem(st["state"])
                it.setBackground(QColor(STATE_COLOR.get(st["state"], "#ffffff")))
                it.setToolTip("\n".join(st["reasons"][:6]) or "up to date")
                self.table.setItem(r, 3 + j, it)
            sp = store_path(paths.fits, run)
            it = QTableWidgetItem("store" if sp.is_file() else
                                  ("none yet" if run.is_new else "legacy file"))
            it.setToolTip(str(sp) if sp.is_file() else "no fit store: the legacy "
                          "CBS_fit_results file is used")
            self.table.setItem(r, len(cols) - 1, it)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.show_detail()

    def show_detail(self):
        from ..pipeline import ORDER
        ks = self.keys()
        if not ks:
            self.detail.setText("Select a run: the reasons of every stale stage are shown "
                                "here (hover a cell for the same).")
            return
        out = []
        for k in ks[:3]:
            for name in ORDER:
                st = self.status.get((k, name))
                if st and st["state"] != "ok":
                    out.append(f"<b>{k} / {name}</b>: {st['state']} — "
                               + "; ".join(st["reasons"][:3]))
        self.detail.setText("<br>".join(out) or "selected runs are up to date")

    def run(self, force):
        ks = self.keys()
        if not ks:
            return
        self.win.runner.run("run", *ks, *(["--force"] if force else []))

    def open_report(self):
        for k in self.keys()[:1]:
            p = self.win.paths.processed / k / "40_report" / "report.html"
            if p.is_file():
                _open(p)
            else:
                QMessageBox.information(self, "Report", f"No report yet: run {k} first.")

    def open_folder(self):
        for k in self.keys()[:1]:
            _open(self.win.paths.processed / k)

    def fit_import(self):
        ks = self.keys()
        if ks:
            self.win.runner.run("fit-import", *ks)

    def to_fitter(self):
        ks = self.keys()
        if ks:
            self.win.fitter.select_run(ks[0])
            self.win.tabs.setCurrentWidget(self.win.fitter)


# ----------------------------------------------------------------------------- studies

class StudiesTab(QWidget):
    PARTS = ["powerlaw", "cbs", "reconstruction", "collapse"]

    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.study = QComboBox()
        self.study.addItems(sorted(win.db.studies))
        self.study.currentTextChanged.connect(self.refresh)
        top.addWidget(QLabel("Study"))
        top.addWidget(self.study)
        top.addStretch()
        lay.addLayout(top)

        g = QGroupBox("Figures")
        gl = QHBoxLayout(g)
        self.parts = {p: QCheckBox(p) for p in self.PARTS}
        for cb in self.parts.values():
            cb.setChecked(True)
            gl.addWidget(cb)
        self.html_only = QCheckBox("HTML only (faster)")
        gl.addWidget(self.html_only)
        gl.addWidget(_btn("Make figures", self.make))
        gl.addWidget(_btn("Method studies (Step 12)", self.methods,
                          "eiakit methods: z̄ systematics, calibration, sensitivity, FC, shots"))
        gl.addStretch()
        lay.addWidget(g)

        g = QGroupBox("Paper numbers")
        gl = QHBoxLayout(g)
        self.label = QLineEdit()
        self.label.setPlaceholderText("snapshot label")
        gl.addWidget(self.label)
        gl.addWidget(_btn("Snapshot", self.snapshot))
        self.a, self.b = QComboBox(), QComboBox()
        gl.addWidget(QLabel("diff"))
        gl.addWidget(self.a)
        gl.addWidget(QLabel("→"))
        gl.addWidget(self.b)
        self.title = QLineEdit()
        self.title.setPlaceholderText("changelog title (optional)")
        gl.addWidget(self.title)
        gl.addWidget(_btn("Diff", self.diff))
        lay.addWidget(g)

        split = QSplitter()
        self.state = QTreeWidget()
        self.state.setHeaderLabels(["part", "state", "why"])
        split.addWidget(self.state)
        self.pages = QListWidget()
        self.pages.itemDoubleClicked.connect(lambda it: _open(Path(it.data(Qt.UserRole))))
        split.addWidget(self.pages)
        lay.addWidget(QLabel("Double-click a page to open it in the browser."))
        lay.addWidget(split, 1)

    def name(self):
        return self.study.currentText()

    def refresh(self):
        from ..study import study_status
        db, paths = self.win.db, self.win.paths
        self.state.clear()
        for part in self.PARTS:
            try:
                st = study_status(db, self.name(), part, paths.data, paths.processed,
                                  paths.studies)
            except Exception as e:                     # noqa: BLE001 - shown, not raised
                st = {"state": "error", "reasons": [str(e)]}
            it = QTreeWidgetItem([part, st["state"], "; ".join(st["reasons"][:3])])
            it.setBackground(1, QColor(STATE_COLOR.get(st["state"], "#ffffff")))
            self.state.addTopLevelItem(it)
        self.pages.clear()
        root = paths.studies / self.name()
        for p in sorted(root.glob("*/*.html")) + sorted(root.glob("*.md")):
            it = QListWidgetItem(str(p.relative_to(root)))
            it.setData(Qt.UserRole, str(p))
            self.pages.addItem(it)
        snaps = sorted(p.stem for p in (root / "snapshots").glob("*.csv"))
        for cb in (self.a, self.b):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(snaps)
            if cur in snaps:
                cb.setCurrentText(cur)

    def make(self):
        parts = [p for p, cb in self.parts.items() if cb.isChecked()]
        self.win.runner.run("study", self.name(), "--part", *parts,
                            *(["--html-only"] if self.html_only.isChecked() else []))

    def methods(self):
        self.win.runner.run("methods", self.name(),
                            *(["--html-only"] if self.html_only.isChecked() else []))

    def snapshot(self):
        lab = self.label.text().strip()
        if lab:
            self.win.runner.run("snapshot", self.name(), "--label", lab)

    def diff(self):
        a, b = self.a.currentText(), self.b.currentText()
        if a and b and a != b:
            t = self.title.text().strip()
            self.win.runner.run("diff", self.name(), a, b, *(["--changelog", t] if t else []))


# ---------------------------------------------------------------------------- settings

class SettingsTab(QWidget):
    SECTIONS = ["corrections", "calibration", "analysis", "cbs", "peaks"]

    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Switches of db/pipeline.toml (edit the file to change the "
                             "defaults; every stage that reads a changed value becomes "
                             "stale). To measure a switch without editing: Compare."))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["setting", "value"])
        self.tree.itemClicked.connect(self.pick)
        lay.addWidget(self.tree, 1)
        g = QGroupBox("Compare one setting (eiakit compare): run + snapshot with A, then B")
        f = QFormLayout(g)
        self.key, self.va, self.vb = QLineEdit(), QLineEdit(), QLineEdit()
        self.title = QLineEdit()
        self.key.setPlaceholderText("section.key, e.g. calibration.method")
        self.title.setPlaceholderText("changelog title (optional)")
        f.addRow("setting", self.key)
        f.addRow("A (before)", self.va)
        f.addRow("B (after)", self.vb)
        f.addRow("changelog", self.title)
        row = QHBoxLayout()
        row.addWidget(_btn("Compare", self.compare))
        row.addWidget(_btn("Open db folder", lambda: _open(self.win.db.root)))
        row.addStretch()
        f.addRow(row)
        lay.addWidget(g)

    def refresh(self):
        self.tree.clear()
        for sec in self.SECTIONS:
            d = self.win.db.pipeline.get(sec, {})
            top = QTreeWidgetItem([f"[{sec}]", ""])
            for k, v in d.items():
                if isinstance(v, dict):
                    v = ", ".join(f"{a}={b}" for a, b in v.items())
                QTreeWidgetItem(top, [k, str(v)])
            self.tree.addTopLevelItem(top)
            top.setExpanded(sec in ("corrections", "calibration", "analysis"))
        self.tree.resizeColumnToContents(0)

    def pick(self, item, _col):
        if item.parent() is not None:
            sec = item.parent().text(0).strip("[]")
            self.key.setText(f"{sec}.{item.text(0)}")
            self.va.setText(item.text(1))

    def compare(self):
        k, a, b = self.key.text().strip(), self.va.text().strip(), self.vb.text().strip()
        if not (k and a and b):
            return
        study = self.win.studies.name()
        t = self.title.text().strip()
        self.win.runner.run("compare", study, k, a, b, *(["--changelog", t] if t else []))


# -------------------------------------------------------------------------- CBS fitter

class FitterTab(QWidget):
    COLS = ["label", "t_peak [µs]", "FWHM [µs]", "kT [eV]", "V0 [V]", "A", "u [km/s]",
            "t_fit [µs]", "FWHM fit [µs]"]
    EDIT = {1: "peak", 2: "fwhm", 3: "kT", 5: "A"}

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.s = None
        self._busy = False
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        row = QHBoxLayout()
        self.run_box = QComboBox()
        self.run_box.addItems([r.key for r in win.db.runs()])
        row.addWidget(self.run_box)
        row.addWidget(_btn("Load", self.load))
        left.addLayout(row)
        self.src = QLabel()
        self.src.setWordWrap(True)
        left.addWidget(self.src)
        self.zlist = QListWidget()
        self.zlist.currentRowChanged.connect(self.show_z)
        left.addWidget(QLabel("Charge state"))
        left.addWidget(self.zlist, 1)
        self.logy = QCheckBox("log y")
        self.logy.toggled.connect(self.draw)
        left.addWidget(self.logy)
        g = QGroupBox("Click on the plot sets")
        gl = QVBoxLayout(g)
        self.click = QButtonGroup(self)
        for i, t in enumerate(("nothing", "peak of the selected component",
                               "left window edge", "right window edge")):
            rb = QRadioButton(t)
            rb.setChecked(i == 1)
            self.click.addButton(rb, i)
            gl.addWidget(rb)
        left.addWidget(g)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(260)
        lay.addWidget(lw)

        right = QVBoxLayout()
        self.fig = Figure(figsize=(8, 5.5), constrained_layout=True)
        self.ax, self.axr = self.fig.subplots(2, 1, sharex=True,
                                              gridspec_kw={"height_ratios": [3, 1]})
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        right.addWidget(NavigationToolbar2QT(self.canvas, self))
        right.addWidget(self.canvas, 3)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemChanged.connect(self.edited)
        self.table.itemSelectionChanged.connect(self.draw)
        right.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.kT_hi = QDoubleSpinBox()
        self.kT_hi.setRange(0, 1e6)
        self.kT_hi.setDecimals(0)
        self.kT_hi.setValue(0)
        self.kT_hi.setSpecialValueText("kT range: auto")
        self.kT_hi.setPrefix("kT ≤ ")
        self.kT_hi.setSuffix(" eV")
        self.kT_hi.setToolTip("upper end of the kT scan; auto = max(kT_max, 3 x the current "
                              "kT or the kT from the FWHM)")
        for b in (_btn("Fit", self.fit, "scan kT against the data minus the other components; "
                       "V0 from kT and the picked peak; A analytic"),
                  self.kT_hi,
                  _btn("kT from FWHM", self.from_fwhm),
                  _btn("Add component", self.add),
                  _btn("Remove", self.remove),
                  _btn("Default window", self.default_window),
                  _btn("Seed this z", self.seed_z,
                       "replace the components of this z by ones proposed from the peaks of "
                       "its J_z(t) (labelled by arrival time), then Fit each"),
                  _btn("Seed all z", self.seed_all, "Seed every z that has no component yet"),
                  _btn("Save to store", self.save,
                       "fits/<run>/cbs_fit.json (previous version kept)")):
            row.addWidget(b)
        row.addStretch()
        right.addLayout(row)
        self.info = QLabel()
        right.addWidget(self.info)
        rw = QWidget()
        rw.setLayout(right)
        lay.addWidget(rw, 1)

    # ------------------------------------------------------------------ data
    def select_run(self, key):
        self.run_box.setCurrentText(key)
        self.load()

    def load(self):
        from ..cbs_edit import CBSSession
        if self.s is not None and self.s.dirty and QMessageBox.question(
                self, "CBS fitter", "Discard the unsaved changes?") != QMessageBox.Yes:
            return
        key = self.run_box.currentText()
        try:
            self.s = CBSSession.load(self.win.db, self.win.db.run(key), self.win.paths.data,
                                     self.win.paths.processed)
        except Exception as e:                         # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "CBS fitter", str(e))
            return
        self.src.setText(f"<small>{self.s.source}</small>")
        self.fill_z()

    def fill_z(self):
        cur = self.zlist.currentRow()
        self.zlist.blockSignals(True)
        self.zlist.clear()
        for z in self.s.zs:
            x2, dof = self.s.chi2(z)
            n = len(self.s.comps.get(z, []))
            mark = "*" if z in self.s.dirty else ""
            self.zlist.addItem(f"z = {z}{mark}   {n} comp."
                               + (f"   χ²/dof {x2 / dof:.3g}" if dof > 0 else ""))
        self.zlist.blockSignals(False)
        self.zlist.setCurrentRow(max(cur, 0))
        self.show_z(self.zlist.currentRow())

    def z(self):
        r = self.zlist.currentRow()
        return self.s.zs[r] if self.s and 0 <= r < len(self.s.zs) else None

    def sel(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        return min(rows) if rows else None

    def show_z(self, _row):
        z = self.z()
        if z is None:
            return
        self._busy = True
        tab = self.s.table(z)
        self.table.setRowCount(len(tab))
        for r, rec in enumerate(tab.to_dict("records")):
            vals = [rec["label"], rec["t_peak_us"], rec["FWHM_us"], rec["kT_eV"], rec["V0_V"],
                    rec["A"], rec["u_kms"], rec["t_peak_fit_us"], rec["FWHM_fit_us"]]
            for c, v in enumerate(vals):
                txt = v if isinstance(v, str) else ("" if v is None or not np.isfinite(v)
                                                    else f"{v:.6g}")
                it = QTableWidgetItem(txt)
                if c not in self.EDIT and c != 0:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 0:
                    it.setBackground(QColor(LABEL_COLOR.get(rec["label"], "#cccccc")))
                    it.setForeground(QColor("white"))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self._busy = False
        if len(tab) and self.sel() is None:
            self.table.selectRow(0)
        self.draw()

    # ------------------------------------------------------------------ plot
    def draw(self):
        z = self.z()
        if z is None or self._busy:
            return
        s, ax, axr = self.s, self.ax, self.axr
        ax.clear()
        axr.clear()
        t = s.t_us()
        if z in s.curves:
            ax.plot(t, s.curves[z], color="0.4", lw=1, label="data (interpolated)")
        te, ye = s.exp.get(z, (np.array([]), np.array([])))
        if len(te):
            ax.plot(te, ye, "o", ms=3, color="black", label="measured points")
        sel = self.sel()
        for i, c in enumerate(s.comps.get(z, [])):
            y = s.component_curve(z, i)
            col = LABEL_COLOR.get(c.extra.get("label"), "#8c564b")
            if y is not None:
                ax.plot(t, y, color=col, lw=2.5 if i == sel else 1.5,
                        label=c.extra.get("label"))
            if c.t_peak:
                ax.axvline(c.t_peak, color=col, ls=":", lw=1)
            if i == sel and c.t_peak:
                wl, wr = s.model.edges(c)
                ax.axvspan(wl, wr, color=col, alpha=0.08)
        tot = s.total(z)
        if np.any(tot) and z in s.curves:
            ax.plot(t, tot, color="#2ca02c", lw=1.2, ls="--", label="total")
            pk = float(np.nanmax(s.curves[z])) or 1.0
            axr.plot(t, (s.curves[z] - tot) / pk * 100, color="0.3", lw=0.8)
        axr.axhline(0, color="0.6", lw=0.6)
        axr.set_ylabel("residual [% of peak]")
        axr.set_xlabel("t [µs]")
        ax.set_ylabel("signal per ion")
        ax.set_title(f"{s.run.key}: {s.run.target}, {_mj(s.run)} mJ, z = {z}")
        if self.logy.isChecked():
            ax.set_yscale("log")
            pk = float(np.nanmax(s.curves.get(z, [1]))) or 1.0
            ax.set_ylim(pk * 1e-4, pk * 2)
        cs = [c for c in s.comps.get(z, []) if c.t_peak]
        if cs:
            lo = min(s.model.edges(c)[0] for c in cs)
            hi = max(s.model.edges(c)[1] for c in cs)
            ax.set_xlim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.legend(fontsize=8, loc="upper right")
        self.canvas.draw_idle()
        x2, dof = s.chi2(z)
        self.info.setText(f"χ² = {x2:.4g} for {dof} dof" if dof > 0 else "no active component")

    # ------------------------------------------------------------------ edits
    def _after(self):
        self.fill_z()

    def on_click(self, ev):
        z, i, mode = self.z(), self.sel(), self.click.checkedId()
        if z is None or i is None or ev.inaxes is not self.ax or ev.xdata is None or mode == 0:
            return
        if self.win.fitter_toolbar_active():
            return
        c = self.s.comps[z][i]
        if mode == 1:
            self.s.set_peak(z, i, ev.xdata)
        else:
            wl, wr = self.s.model.edges(c)
            self.s.set_window(z, i, ev.xdata if mode == 2 else wl, ev.xdata if mode == 3 else wr)
        self._after()

    def edited(self, item):
        if self._busy or self.s is None:
            return
        z, r, c = self.z(), item.row(), item.column()
        txt = item.text().strip()
        try:
            if c == 0:
                self.s.comps[z][r].extra["label"] = txt
                self.s.dirty.add(z)
            else:
                v = float(txt)
                {1: self.s.set_peak, 2: self.s.set_fwhm, 3: self.s.set_kT,
                 5: self.s.set_A}[c](z, r, v)
        except (ValueError, KeyError):
            pass
        self._after()

    def fit(self):
        z, i = self.z(), self.sel()
        if z is not None and i is not None:
            out = self.s.fit(z, i, self.kT_hi.value() or None)
            if out is None:
                QMessageBox.information(self, "Fit", "No valid kT on the grid (V0 or A not "
                                        "positive): check the peak and the window.")
            self._after()

    def from_fwhm(self):
        z, i = self.z(), self.sel()
        if z is not None and i is not None:
            self.s.kT_from_fwhm(z, i)
            self._after()

    def add(self):
        from ..cbs_edit import LABEL_ORDER, default_label
        z = self.z()
        if z is None:
            return
        have = [c.extra.get("label") for c in self.s.comps.get(z, [])]
        lab, ok = QInputDialog.getItem(self, "Add component", "label", LABEL_ORDER,
                                       LABEL_ORDER.index(default_label(have))
                                       if default_label(have) in LABEL_ORDER else 0, True)
        if not ok:
            return
        lo, hi = self.ax.get_xlim()
        i = self.s.add(z, lab, 0.5 * (lo + hi), 0.05 * (hi - lo))
        self._after()
        self.table.selectRow(i)

    def remove(self):
        z, i = self.z(), self.sel()
        if z is not None and i is not None:
            self.s.remove(z, i)
            self._after()

    def default_window(self):
        z, i = self.z(), self.sel()
        if z is not None and i is not None:
            self.s.set_window(z, i, None, None)
            self._after()

    def seed_z(self):
        z = self.z()
        if z is None:
            return
        if self.s.comps.get(z) and QMessageBox.question(
                self, "Seed", f"Replace the {len(self.s.comps[z])} components of z = {z}?") \
                != QMessageBox.Yes:
            return
        self.s.seed(z)
        self._after()

    def seed_all(self):
        if self.s is None:
            return
        r = self.s.seed_all()
        self.win.log.appendPlainText(f"seeded {len(r)} charge states: " + ", ".join(
            f"z{z} {'+'.join(v) or '-'}" for z, v in r.items()))
        self._after()

    def save(self):
        if self.s is None:
            return
        note, ok = QInputDialog.getText(self, "Save CBS fit", "note (optional)")
        if not ok:
            return
        p = self.s.save(self.win.paths.fits, note)
        self.win.log.appendPlainText(f"CBS fit saved: {p}. 'Run' on the Runs tab updates "
                                     f"the cbs and report stages.")
        self.src.setText(f"<small>store {p}</small>")
        self.fill_z()
        self.win.runs.refresh()


# --------------------------------------------------------------------- peak picker

class PeaksTab(QWidget):
    COLS = ["m", "z", "t0 [µs]", "TOF [µs]", "Yield [V]", "σ [ns]", "ok", "excluded",
            "ref. TOF [µs]", "ref. Yield [V]"]

    def __init__(self, win):
        super().__init__()
        self.win, self.s, self._loading = win, None, False
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.run_box = QComboBox()
        self.run_box.addItems([r.key for r in win.db.runs()])
        self.run_box.currentTextChanged.connect(self.fill_U)
        left.addWidget(self.run_box)
        self.ulist = QListWidget()
        self.ulist.currentRowChanged.connect(self.load_U)
        left.addWidget(QLabel("Voltage (with raw SEM files; * = refitted)"))
        left.addWidget(self.ulist, 1)
        self.all_lines = QCheckBox("all lines (every z to z max)")
        self.all_lines.toggled.connect(lambda _: self.load_U(self.ulist.currentRow()))
        left.addWidget(self.all_lines)
        self.corr = QDoubleSpinBox()
        self.corr.setRange(0, 0.1)
        self.corr.setSingleStep(0.005)
        self.corr.setDecimals(3)
        self.corr.setPrefix("Corr (window +) ")
        left.addWidget(self.corr)
        g = QGroupBox("Expected lines (as in untitled28)")
        gl = QFormLayout(g)
        self.uclk = QSpinBox()
        self.uclk.setRange(-60, 60)
        self.uclk.setToolTip("arrow clicks on U: each moves U by U/500 (0.2 %); all expected "
                             "lines of this voltage move, then every line is refitted")
        self.lclk = QSpinBox()
        self.lclk.setRange(-30, 30)
        self.lclk.setToolTip("arrow clicks on LOF: each moves LOF by LOF/100 (1 %)")
        self.uval, self.lval = QLabel(), QLabel()
        gl.addRow("U clicks", self.uclk)
        gl.addRow("", self.uval)
        gl.addRow("LOF clicks", self.lclk)
        gl.addRow("", self.lval)
        self.auto = QCheckBox("automatic centring (new runs)")
        self.auto.setChecked(True)
        self.auto.setToolTip("off: the expected lines come from U and LOF only, as in "
                             "untitled28; on: corrected by the factor found from the "
                             "strongest lines (peaks stage)")
        gl.addRow(self.auto)
        self.uclk.valueChanged.connect(lambda v: self.clicks(u=v))
        self.lclk.valueChanged.connect(lambda v: self.clicks(lof=v))
        self.auto.toggled.connect(self.auto_centre)
        left.addWidget(g)
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(260)
        lay.addWidget(lw)
        right = QVBoxLayout()
        self.fig = Figure(figsize=(8, 4.5), constrained_layout=True)
        self.ax = self.fig.subplots()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 3)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.draw)
        right.addWidget(self.table, 2)
        row = QHBoxLayout()
        for b in (_btn("Full spectrum", self.full,
                       "back to the whole trace (also: Esc, or click the selected row again)"),
                  _btn("Refit", self.refit, "fit every line at the file voltage (+ Corr)"),
                  _btn("Exclude / include line", self.toggle),
                  _btn("Reset window centre", self.reset),
                  _btn("Save voltage", self.save,
                       "fits/<run>/peaks/p<U>.txt: replaces this voltage's Output rows in "
                       "the calibration")):
            row.addWidget(b)
        row.addStretch()
        right.addLayout(row)
        right.addWidget(QLabel("Orange rows: the fit ended at the edge of its ± 150 ns search "
                               "(missed peak or no peak). Select the row, click on the real "
                               "peak (new window centre, refit), or Exclude the line; then "
                               "Save voltage."))
        rw = QWidget()
        rw.setLayout(right)
        lay.addWidget(rw, 1)
        self.fill_U(self.run_box.currentText())

    def run(self):
        return self.win.db.run(self.run_box.currentText())

    def fill_U(self, _key):
        from ..peaks_edit import PeakSession, override_files
        run = self.run()
        ov = override_files(self.win.paths.fits, run)
        self.Us = PeakSession.voltages(run, self.win.paths.data)
        edges = {}
        fits = self.win.paths.processed / run.key / "10_peaks" / "fits.csv"
        if run.is_new and fits.is_file():
            import pandas as pd
            f = pd.read_csv(fits)
            edges = (f.success & f.mu_at_bound).groupby(f.U).sum().to_dict()
        self.ulist.blockSignals(True)
        self.ulist.clear()
        for U in self.Us:
            n = int(edges.get(U, 0))
            self.ulist.addItem(f"U = {U:g} V" + (" *" if U in ov else "")
                               + (f"   ({n} at edge)" if n else ""))
        self.ulist.blockSignals(False)
        self.s = None
        if self.Us:
            self.ulist.setCurrentRow(0)
        else:
            self.table.setRowCount(0)
            self.ax.clear()
            self.ax.set_title(f"{run.key}: no raw SEM files in the data folder")
            self.canvas.draw_idle()

    def load_U(self, r):
        from ..peaks_edit import PeakSession
        if not (0 <= r < len(self.Us)):
            return
        try:
            self.s = PeakSession.load(self.win.db, self.run(), self.win.paths.data,
                                      self.Us[r], self.all_lines.isChecked(),
                                      self.win.paths.processed)
        except Exception as e:                         # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "Peaks", str(e))
            return
        saved = self.s.restore(self.win.paths.fits)    # a saved voltage: as it was saved
        self._loading = True
        self.uclk.setValue(self.s.u_clicks)
        self.lclk.setValue(self.s.lof_clicks)
        self.corr.setValue(self.s.corr)
        self.auto.setChecked(self.s.k != 1.0 or not self.s.run.is_new)
        self._loading = False
        if saved:
            self.win.log.appendPlainText(f"U = {self.s.U:g} V: the saved fit (clicks "
                                         f"{self.s.u_clicks}, LOF clicks {self.s.lof_clicks}, "
                                         f"Corr {self.s.corr:g})")
        self.show_table()

    def sel(self):
        rows = {i.row() for i in self.table.selectedIndexes()}
        return min(rows) if rows else None

    def full(self):
        self.table.clearSelection()
        self.draw()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.full()
        else:
            super().keyPressEvent(ev)

    def clicks(self, u=None, lof=None):
        if self.s is None or self._loading:
            return
        i = self.sel()
        self.s.set_clicks(u, lof)
        self.show_table()
        if i is not None:
            self.table.selectRow(i)

    def auto_centre(self, on):
        if self.s is None or self._loading:
            return
        self.s.set_auto_centre(on)
        self.show_table()

    def show_table(self):
        s = self.s
        self.uval.setText(f"U = {s.U_used:.6g} V (file {s.U:g} V)")
        self.lval.setText(f"LOF = {s.L_used:.6g} m (db {s.L_m:g} m)")
        self.auto.setEnabled(bool(s.run.is_new))
        f, leg = self.s.fits, self.s.legacy
        li = {(int(a), int(b)): (t, y) for a, b, t, y in
              zip(leg.m, leg.z, leg.TOF, leg.YieldAveFit, strict=True)} if len(leg) else {}
        self.table.blockSignals(True)
        self.table.setRowCount(len(f))
        for r, rec in enumerate(f.to_dict("records")):
            lt, ly = li.get((int(rec["m"]), int(rec["z"])), (np.nan, np.nan))
            vals = [rec["m"], rec["z"], rec["t0"] * 1e6, rec["TOF"] * 1e6, rec["Yield"],
                    rec["sigma"] * 1e9,
                    ("at edge" if rec.get("at_edge") else "yes") if rec["success"] else "no",
                    "yes" if rec["excluded"] else "", lt * 1e6, ly]
            for c, v in enumerate(vals):
                txt = v if isinstance(v, str) else ("" if not np.isfinite(v) else f"{v:.6g}")
                it = QTableWidgetItem(txt)
                if rec["excluded"]:
                    it.setForeground(QColor("#999999"))
                elif rec.get("at_edge") or not rec["success"]:
                    it.setBackground(QColor("#ffe0b2"))
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.draw()

    def draw(self):
        if self.s is None:
            return
        s, ax = self.s, self.ax
        ax.clear()
        ax.plot(s.tof * 1e6, s.y, color="0.35", lw=0.8, label=f"average of {len(s.traces)} shots")
        sel = self.sel()
        for i, r in enumerate(s.fits.itertuples()):
            col = LABEL_COLOR["fast"] if i == sel else ("#bbbbbb" if r.excluded else "#1f77b4")
            ax.axvline(r.t0 * 1e6, color=col, ls=":", lw=0.8)
            g = s.gaussian(i)
            if g is not None and not r.excluded:
                w = np.abs(s.tof - r.TOF) < 5 * r.sigma
                ax.plot(s.tof[w] * 1e6, g[w], color=col, lw=2 if i == sel else 1.2)
        for k, t_leg in enumerate(s.legacy.TOF if len(s.legacy) else []):
            ax.axvline(t_leg * 1e6, ymax=0.08, color="#ff7f0e", lw=2,
                       label=("pipeline fit" if s.run.is_new else "untitled28 fit")
                       + " position" if k == 0 else None)
        ts = s.fits.t0
        ax.set_xlim(ts.min() * 0.9e6, ts.max() * 1.1e6)
        if sel is not None:
            r = s.fits.iloc[sel]
            ax.set_xlim(r.t0 * 0.95e6, r.t0 * 1.05e6)
        ax.set_xlabel("t [µs]")
        ax.set_ylabel("SEM [V]")
        ax.set_title(f"{s.run.key}, U = {s.U:g} V (windows at {s.U_used:.6g} V, "
                     f"LOF {s.L_used:.6g} m), Corr {s.corr:g}")
        ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def on_click(self, ev):
        i = self.sel()
        if self.s is None or i is None or ev.inaxes is not self.ax or ev.xdata is None \
                or self.toolbar.mode:
            return
        self.s.set_center(i, ev.xdata * 1e-6)
        self.show_table()
        self.table.selectRow(i)

    def refit(self):
        if self.s:
            self.s.refit(self.corr.value())
            self.show_table()

    def toggle(self):
        i = self.sel()
        if self.s and i is not None:
            self.s.exclude(i, i not in self.s.excluded)
            self.show_table()
            self.table.selectRow(i)

    def reset(self):
        i = self.sel()
        if self.s and i is not None:
            self.s.set_center(i, None)
            self.show_table()
            self.table.selectRow(i)

    def save(self):
        if self.s is None:
            return
        p = self.s.save(self.win.paths.fits)
        self.win.log.appendPlainText(
            f"voltage saved: {p}\n  It is used from the next run of {self.s.run.key}: Runs "
            f"tab -> Run selected (the calibration line then lists "
            f"'voltages_from_peaks_tab'). To undo: delete that file and run again.")
        it = self.ulist.currentItem()                  # mark it saved; keep the session
        if it is not None and "*" not in it.text():
            t = it.text()
            j = t.find(" V") + 2
            it.setText(t[:j] + " *" + t[j:])
        self.win.runs.refresh()


# ------------------------------------------------------------------- FC baseline

class BaselineTab(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win, self.b, self.clicks = win, None, []
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.run_box = QComboBox()
        self.run_box.addItems([r.key for r in win.db.runs()])
        row.addWidget(self.run_box)
        row.addWidget(_btn("Load FC traces", self.load))
        self.value = QDoubleSpinBox()
        self.value.setDecimals(5)
        self.value.setSuffix(" mV")
        self.value.setRange(-1e4, 1e4)
        self.value.valueChanged.connect(self.draw)
        row.addWidget(QLabel("pedestal"))
        row.addWidget(self.value)
        row.addWidget(_btn("= pre-trigger mean", lambda: self.set(self.b and self.b.pretrigger)))
        row.addWidget(_btn("= legacy (recovered)", lambda: self.set(self.b and self.b.legacy)))
        row.addWidget(_btn("Save for this run", self.save,
                           "fits/<run>/fc_baseline.json; used with [calibration] "
                           "fc_baseline = \"picked\""))
        row.addStretch()
        lay.addLayout(row)
        self.fig = Figure(figsize=(8, 5), constrained_layout=True)
        self.ax, self.axz = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.info = QLabel("Two clicks on the left plot: the pedestal is the mean between "
                           "them. The right plot zooms on the pre-trigger region.")
        lay.addWidget(self.info)

    def load(self):
        from ..peaks_edit import BaselineSession
        try:
            self.b = BaselineSession.load(self.win.db, self.win.db.run(self.run_box.currentText()),
                                          self.win.paths.data, self.win.paths.fits)
        except Exception as e:                         # noqa: BLE001 - shown to the user
            QMessageBox.warning(self, "FC baseline", str(e))
            return
        v = self.b.picked if self.b.picked is not None else (
            self.b.legacy if self.b.legacy is not None else self.b.pretrigger)
        self.set(v)

    def set(self, v):
        if v is not None:
            self.value.setValue(v * 1e3)
        self.draw()

    def draw(self):
        if self.b is None:
            return
        b = self.b
        P = self.value.value() * 1e-3
        for ax in (self.ax, self.axz):
            ax.clear()
            ax.plot(b.t * 1e6, b.v * 1e3, color="0.3", lw=0.7, label="FC average (native grid)")
            for val, lab_, col in ((b.pretrigger, "pre-trigger mean", "#1f77b4"),
                                   (b.legacy, "legacy (recovered)", "#ff7f0e"),
                                   (P, "pedestal", "#d62728")):
                if val is not None:
                    ax.axhline(val * 1e3, color=col, lw=1.2, ls="--" if col != "#d62728" else "-",
                               label=f"{lab_}: {val * 1e3:.4f} mV")
            ax.set_xlabel("t [µs]")
            ax.set_ylabel("FC [mV]")
        pre = b.t < 0
        if pre.any():
            self.axz.set_xlim(b.t[pre].min() * 1e6, 0)
            vv = b.v[pre] * 1e3
            self.axz.set_ylim(np.percentile(vv, 1) - 1, np.percentile(vv, 99) + 1)
        self.ax.legend(fontsize=8)
        self.ax.set_title(f"{b.run.key}: FC peak after the pedestal "
                          f"{(b.v.max() - P) * 1e3:.3f} mV")
        self.canvas.draw_idle()

    def on_click(self, ev):
        if self.b is None or ev.inaxes not in (self.ax, self.axz) or ev.xdata is None \
                or self.toolbar.mode:
            return
        self.clicks.append(ev.xdata * 1e-6)
        if len(self.clicks) == 2:
            self.set(self.b.window_mean(*self.clicks))
            self.info.setText(f"pedestal = mean between {self.clicks[0] * 1e6:.3f} and "
                              f"{self.clicks[1] * 1e6:.3f} µs")
            self.clicks = []

    def save(self):
        if self.b is None:
            return
        p = self.b.save(self.win.paths.fits, self.value.value() * 1e-3)
        mode = self.win.db.pipeline["calibration"].get("fc_baseline")
        self.win.log.appendPlainText(
            f"FC pedestal saved: {p}." + ("" if mode == "picked" else
                                         " It is used when [calibration] fc_baseline = "
                                         "\"picked\" (now: " + str(mode) + ")."))
        self.win.runs.refresh()


# ------------------------------------------------------------------------------ window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"eiakit {__version__}")
        self.paths = get_paths()
        self.db = load_database()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas" if os.name == "nt" else "monospace", 9))
        self.log.setMaximumBlockCount(20000)
        self.runner = Runner(self.log, self.after_command)
        self.tabs = QTabWidget()
        self.runs = RunsTab(self)
        self.studies = StudiesTab(self)
        self.settings = SettingsTab(self)
        self.fitter = FitterTab(self)
        self.peaks = PeaksTab(self)
        self.baseline = BaselineTab(self)
        for w, t in ((self.runs, "Runs"), (self.studies, "Studies"),
                     (self.settings, "Settings"), (self.fitter, "CBS fitter"),
                     (self.peaks, "Peaks"), (self.baseline, "FC baseline")):
            self.tabs.addTab(w, t)
        self.setCentralWidget(self.tabs)
        dock = QDockWidget("Log", self)
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(2, 2, 2, 2)
        bl.addWidget(self.log)
        row = QHBoxLayout()
        row.addWidget(_btn("Stop", self.runner.stop))
        row.addWidget(_btn("Clear", self.log.clear))
        row.addWidget(_btn("Reload database", self.reload_db,
                           "after editing db/*.toml"))
        row.addStretch()
        bl.addLayout(row)
        dock.setWidget(box)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resize(1400, 900)
        self.log.appendPlainText(f"eiakit {__version__}: data {self.paths.data}, processed "
                                 f"{self.paths.processed}, studies {self.paths.studies}, "
                                 f"fits {self.paths.fits}")
        self.refresh_all()

    def fitter_toolbar_active(self) -> bool:
        tb = self.fitter.findChild(NavigationToolbar2QT)
        return bool(tb is not None and tb.mode)

    def reload_db(self):
        self.db = load_database()
        self.log.appendPlainText("database reloaded")
        self.refresh_all()

    def refresh_all(self):
        self.runs.refresh()
        self.studies.refresh()
        self.settings.refresh()

    def after_command(self):
        self.runs.refresh()
        self.studies.refresh()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
