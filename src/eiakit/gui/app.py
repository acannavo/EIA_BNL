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
from PySide6.QtGui import QColor, QDesktopServices, QFont, QKeySequence, QShortcut
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
    QSlider,
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
# Peaks tab: one colour per species (others: the matplotlib cycle, stable per name)
SPECIES_COLOR = {"Al": "#1f77b4", "Nb": "#2ca02c", "H": "#d62728", "C": "#9467bd",
                 "O": "#ff7f0e"}


def species_color(sp: str) -> str:
    if sp in SPECIES_COLOR:
        return SPECIES_COLOR[sp]
    cyc = ["#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
    return cyc[sum(map(ord, sp)) % len(cyc)]


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
            "t_fit [µs]", "FWHM fit [µs]", "locked"]
    EDIT = {1: "peak", 2: "fwhm", 3: "kT", 4: "V0", 5: "A"}

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.s = None
        self._busy = False
        self._full = False
        esc = QShortcut(QKeySequence(Qt.Key_Escape), self, self.full)
        esc.setContext(Qt.WidgetWithChildrenShortcut)
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
                               "left window edge", "right window edge",
                               "half-maximum left (FWHM)", "half-maximum right (FWHM)")):
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
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self._hover = None
        right.addWidget(NavigationToolbar2QT(self.canvas, self))
        right.addWidget(self.canvas, 3)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemChanged.connect(self.edited)
        self.table.itemSelectionChanged.connect(self.row_changed)
        right.addWidget(self.table, 1)
        sl = QHBoxLayout()
        self.sA, self.skT = QSlider(Qt.Horizontal), QSlider(Qt.Horizontal)
        self.lA, self.lkT = QLabel("A"), QLabel("kT")
        for sld, lab_, tip in ((self.sA, self.lA, "amplitude A of the selected component: "
                                "÷10 … ×10 around its value when you grab the slider"),
                               (self.skT, self.lkT, "kT of the selected component, ÷10 … ×10: "
                                "the peak stays where it is (V0 follows) and the peak height "
                                "is kept (A follows)")):
            sld.setRange(-100, 100)
            sld.setValue(0)
            sld.setToolTip(tip)
            sl.addWidget(lab_)
            sl.addWidget(sld, 1)
        self.sA.valueChanged.connect(lambda v: self.slide("A", v))
        self.skT.valueChanged.connect(lambda v: self.slide("kT", v))
        for sld in (self.sA, self.skT):
            sld.sliderPressed.connect(self.grab_base)
            sld.sliderReleased.connect(self.release_slider)
        self._base = None
        right.addLayout(sl)
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
                  self._lock_btn(),
                  _btn("Full spectrum", self.full, "whole time range (also Esc)"),
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
            lk = self.s.locked(z, r)
            vals = [rec["label"] + ("  🔒" if lk else ""), rec["t_peak_us"], rec["FWHM_us"],
                    rec["kT_eV"], rec["V0_V"],
                    rec["A"], rec["u_kms"], rec["t_peak_fit_us"], rec["FWHM_fit_us"],
                    "locked" if lk else ""]
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
            ax.plot(te, ye, "o", ms=3, color="black",
                    label="measured points (hover: voltage)")
        self._hover = ax.annotate("", (0, 0), xytext=(10, 10), textcoords="offset points",
                                  fontsize=8, bbox={"boxstyle": "round", "fc": "lightyellow",
                                                    "ec": "0.5"}, zorder=10)
        self._hover.set_visible(False)
        sel = self.sel()
        for i, c in enumerate(s.comps.get(z, [])):
            y = s.component_curve(z, i)
            col = LABEL_COLOR.get(c.extra.get("label"), "#8c564b")
            if y is not None:
                ax.plot(t, y, color=col, lw=2.5 if i == sel else 1.5,
                        ls="-" if not c.extra.get("locked") else (0, (6, 2)),
                        label=c.extra.get("label") + (" (locked)" if c.extra.get("locked")
                                                      else ""))
            if c.t_peak:
                ax.axvline(c.t_peak, color=col, ls=":", lw=1)
            if i == sel and c.t_peak:
                wl, wr = s.model.edges(c)
                ax.axvspan(wl, wr, color=col, alpha=0.08)
                h = s.data_at(z, c.t_peak)              # guides for the FWHM clicks
                if np.isfinite(h) and h > 0:
                    ax.axhline(h, color=col, ls=":", lw=0.9)
                    ax.axhline(h / 2, color=col, ls="--", lw=1.1)
                    ax.annotate("½ max", (c.t_peak, h / 2), color=col, fontsize=8,
                                xytext=(4, 3), textcoords="offset points")
                for tt in (c.t_left, c.t_right):           # picked half-maximum edges
                    if tt is not None and np.isfinite(tt):
                        ax.axvline(tt, color=col, ls="--", lw=0.8)
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
        if self._full and len(te):
            pad = 0.05 * (te.max() - te.min())
            ax.set_xlim(te.min() - pad, te.max() + pad)
        elif cs:
            lo = min(s.model.edges(c)[0] for c in cs)
            hi = max(s.model.edges(c)[1] for c in cs)
            ax.set_xlim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.legend(fontsize=8, loc="upper right")
        self.canvas.draw_idle()
        x2, dof = s.chi2(z)
        self.info.setText(f"χ² = {x2:.4g} for {dof} dof" if dof > 0 else "no active component")

    # ------------------------------------------------------------------ sliders, lock
    def row_changed(self):
        self._full = False
        self._base = None
        self.update_lock_btn()
        self.draw()

    def grab_base(self):
        z, i = self.z(), self.sel()
        if z is None or i is None:
            self._base = None
            return
        c = self.s.comps[z][i]
        self._base = (z, i, c.A, c.kT)

    def slide(self, which, v):
        if self._busy or self.s is None:
            return
        if self._base is None or self._base[:2] != (self.z(), self.sel()):
            self.grab_base()
        if self._base is None:
            return
        z, i, A0, kT0 = self._base
        f = 10 ** (v / 100.0)
        if which == "A" and A0:
            self.s.set_A(z, i, A0 * f)
        ok = True
        if which == "A" and A0:
            pass
        elif which == "kT" and kT0:
            ok = self.s.set_kT_keep_height(z, i, kT0 * f)
        c = self.s.comps[z][i]
        self.lA.setText(f"A {c.A:.4g}" if c.A else "A")
        self.lkT.setText(f"kT {c.kT:.4g} eV" if ok else
                         f"kT {kT0 * f:.4g} eV: peak not reachable (V0 < 0)")
        self.draw()

    def release_slider(self):
        """Slider back to the middle; the new values become the base of the next move."""
        if self.s is None:
            return
        self._busy = True
        self.sA.setValue(0)
        self.skT.setValue(0)
        self._busy = False
        self._base = None
        r = self.sel()
        self.show_z(self.zlist.currentRow())
        if r is not None:
            self.table.selectRow(r)

    def _lock_btn(self):
        self.lock_btn = _btn("Lock", self.toggle_lock,
                             "locked: Fit and Seed leave this component as you set it "
                             "(sliders, typed values); it still counts when the others are "
                             "fitted. Select a component (table row) first.")
        return self.lock_btn

    def update_lock_btn(self):
        z, i = self.z(), self.sel()
        ok = self.s is not None and z is not None and i is not None
        self.lock_btn.setEnabled(ok)
        self.lock_btn.setText("Unlock" if ok and self.s.locked(z, i) else "Lock")

    def toggle_lock(self):
        z, i = self.z(), self.sel()
        if z is None or i is None:
            QMessageBox.information(self, "Lock", "Select a component first: click its row "
                                    "in the table.")
            return
        new = not self.s.locked(z, i)
        self.s.lock(z, i, new)
        lab = self.s.comps[z][i].extra.get("label")
        self.win.log.appendPlainText(f"z = {z} {lab}: {'locked' if new else 'unlocked'}"
                                     " (Save to store keeps it)")
        self._after()
        self.table.selectRow(i)
        self.update_lock_btn()

    def full(self):
        self._full = True
        self.draw()

    def on_hover(self, ev):
        """Measured point under the mouse: its deflector voltage (and t, signal)."""
        z = self.z()
        if self.s is None or z is None or self._hover is None:
            return
        te, ye = self.s.exp.get(z, (np.array([]), np.array([])))
        show = False
        if ev.inaxes is self.ax and len(te):
            pts = self.ax.transData.transform(np.column_stack([te, ye]))
            d = np.hypot(pts[:, 0] - ev.x, pts[:, 1] - ev.y)
            j = int(np.argmin(d))
            if d[j] < 7:
                Us = self.s.exp_U.get(z)
                U = Us[j] if Us is not None and j < len(Us) else np.nan
                self._hover.xy = (te[j], ye[j])
                self._hover.set_text((f"U = {U:g} V" if np.isfinite(U) else "U = ?")
                                     + f"\nt = {te[j]:.4g} µs\nsignal {ye[j]:.3g}")
                show = True
        if show or self._hover.get_visible():
            self._hover.set_visible(show)
            self.canvas.draw_idle()

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
            self.click.button(4).setChecked(True)          # next: the half-maximum edges
        elif mode in (4, 5):
            self.s.set_half(z, i, "left" if mode == 4 else "right", ev.xdata)
            if mode == 4:
                self.click.button(5).setChecked(True)
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
                self.s.comps[z][r].extra["label"] = txt.replace("🔒", "").strip()
                self.s.dirty.add(z)
            else:
                v = float(txt)
                {1: self.s.set_peak, 2: self.s.set_fwhm, 3: self.s.set_kT,
                 4: self.s.set_V0, 5: self.s.set_A}[c](z, r, v)
        except (ValueError, KeyError):
            pass
        self._after()

    def fit(self):
        z, i = self.z(), self.sel()
        if z is not None and i is not None:
            if self.s.locked(z, i):
                QMessageBox.information(self, "Fit", "This component is locked (your manual "
                                        "values are kept). Unlock it to fit.")
                return
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
    COLS = ["species", "m", "z", "t0 [µs]", "TOF [µs]", "Yield [V]", "σ [ns]", "B [V]", "ok",
            "locked", "overlap", "excluded", "ref. TOF [µs]", "ref. Yield [V]"]
    MODES = ["window centre", "peak position (lock μ)", "background level (lock B)",
             "common background: 2 points", "common background: add tuning point",
             "common background: remove tuning point"]
    BG_COLOR = "#8c564b"

    def __init__(self, win):
        super().__init__()
        self.win, self.s, self._loading = win, None, False
        esc = QShortcut(QKeySequence(Qt.Key_Escape), self, self.full)
        esc.setContext(Qt.WidgetWithChildrenShortcut)
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
        wf = float(win.db.pipeline["peaks"].get("window_frac", 0.02))
        self.corr.setRange(-wf + 0.001, 0.1)
        self.corr.setSingleStep(0.005)
        self.corr.setDecimals(3)
        self.corr.setPrefix("Corr (window ±) ")
        self.corr.setToolTip(f"added to the half window of every line ({wf:g} x t0): "
                             f"positive widens, negative narrows (down to {0.001:g} x t0); "
                             "press Refit")
        left.addWidget(self.corr)
        gb = QGroupBox("Common background (all lines)")
        gbl = QFormLayout(gb)
        self.bg_use = QCheckBox("use it")
        self.bg_use.setToolTip("every line is fitted on this background (a line's own "
                               "locked B wins); define it with the click mode '2 points'")
        self.bg_use.toggled.connect(self.bg_toggle)
        self.bg_reach = QDoubleSpinBox()
        self.bg_reach.setRange(0.2, 20)
        self.bg_reach.setSingleStep(0.5)
        self.bg_reach.setValue(3.0)
        self.bg_reach.setSuffix(" % of t")
        self.bg_reach.setToolTip("how far a tuning point bends the background on each side "
                                 "(adjacent charge states are ~5 % apart)")
        self.bg_reach.valueChanged.connect(self.bg_reach_changed)
        self.bg_info = QLabel("not defined")
        self.bg_info.setWordWrap(True)
        gbl.addRow(self.bg_use)
        gbl.addRow("tuning reach", self.bg_reach)
        gbl.addRow(self.bg_info)
        gbl.addRow(_btn("Clear common background", self.bg_clear))
        left.addWidget(gb)
        self._bg_click = None
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
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self._drag = None
        self._hover = None
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 3)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.draw)
        tr = QHBoxLayout()
        tr.addWidget(self.table, 3)
        self.rfig = Figure(figsize=(5, 3), constrained_layout=True)
        self.rax = self.rfig.subplots()
        self.rcanvas = FigureCanvasQTAgg(self.rfig)
        self.rcanvas.setMinimumWidth(380)
        self.rcanvas.mpl_connect("button_press_event", self.recap_click)
        self.rcanvas.mpl_connect("motion_notify_event", self.recap_hover)
        self._rbars, self._rhover = [], None
        tr.addWidget(self.rcanvas, 2)
        right.addLayout(tr, 2)
        row = QHBoxLayout()
        for b in (_btn("Full spectrum", self.full,
                       "back to the whole trace (also: Esc, or click the selected row again)"),
                  _btn("Refit", self.refit, "fit every line at the file voltage (+ Corr)"),
                  _btn("Exclude / include line", self.toggle),
                  _btn("Reset window centre", self.reset),
                  _btn("Unlock line", self.unlock,
                       "free the hand-set peak position and background of the selected "
                       "line again (the fit finds them)"),
                  _btn("Save voltage", self.save,
                       "fits/<run>/peaks/p<U>.txt: replaces this voltage's Output rows in "
                       "the calibration")):
            row.addWidget(b)
        row.addStretch()
        right.addLayout(row)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Click on the plot sets:"))
        self.mode = QComboBox()
        self.mode.addItems(self.MODES)
        self.mode.setToolTip(
            "window centre: the fit window moves there, everything is fitted again\n"
            "peak position (lock μ): the peak is fixed at the clicked time; depth and width "
            "are fitted around it\n"
            "background level (lock B): the baseline is fixed at the clicked height\n"
            "Both locks together: only depth and width are fitted. Locks are saved with "
            "the voltage.\n"
            "common background, 2 points: click 2 baseline points anywhere; a horizontal "
            "line at their mean becomes the background of every line\n"
            "add / remove tuning point: bends the common background locally through the "
            "click (within the tuning reach) / removes the nearest tuning point")
        mrow.addWidget(self.mode)
        mrow.addStretch()
        right.addLayout(mrow)
        right.addWidget(QLabel(
            "Orange rows: the fit ended at the edge of its ± 150 ns search (missed peak or "
            "no peak). Select the row (or click a peak in the full spectrum), then drag the "
            "white circle to the real peak bottom (locks position and height), or click with "
            "the mode above, or Exclude the line; then Save voltage. Right: heights by "
            "species; hatched = may be blended with another line."))
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
        self._bg_click = None
        if self.s.bg:
            self.bg_reach.setValue(self.s.bg.get("reach", 0.03) * 100)
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

    def bg_sync(self):
        s = self.s
        self._loading = True
        self.bg_use.setEnabled(bool(s.bg))
        self.bg_use.setChecked(s.bg_on)
        self._loading = False
        if s.bg:
            n = len(s.bg["tune"])
            self.bg_info.setText(f"level {s.bg_level:.4g} V, {n} tuning point"
                                 + ("s" if n != 1 else "") + ("" if s.bg_on else " (off)"))
        else:
            self.bg_info.setText("not defined: click mode 'common background: 2 points'")

    def bg_toggle(self, on):
        if self.s is None or self._loading:
            return
        i = self.sel()
        self.s.set_bg_on(on)
        self.show_table()
        if i is not None:
            self.table.selectRow(i)

    def bg_reach_changed(self, v):
        if self.s is None or self._loading or not self.s.bg:
            return
        i = self.sel()
        self.s.set_bg_reach(v / 100)
        self.show_table()
        if i is not None:
            self.table.selectRow(i)

    def bg_clear(self):
        if self.s is None:
            return
        i = self.sel()
        self._bg_click = None
        self.s.clear_common_bg()
        self.show_table()
        if i is not None:
            self.table.selectRow(i)

    def show_table(self):
        s = self.s
        self.bg_sync()
        self.uval.setText(f"U = {s.U_used:.6g} V (file {s.U:g} V)")
        self.lval.setText(f"LOF = {s.L_used:.6g} m (db {s.L_m:g} m)")
        self.auto.setEnabled(bool(s.run.is_new))
        f, leg = self.s.fits, self.s.legacy
        li = {(int(a), int(b)): (t, y) for a, b, t, y in
              zip(leg.m, leg.z, leg.TOF, leg.YieldAveFit, strict=True)} if len(leg) else {}
        ov = s.overlaps()
        self.table.blockSignals(True)
        self.table.setRowCount(len(f))
        for r, rec in enumerate(f.to_dict("records")):
            ovt = ", ".join(("≡ " if kind == "same" else "~ ") + name
                            for _, name, kind, _ in ov.get(r, []))
            lt, ly = li.get((int(rec["m"]), int(rec["z"])), (np.nan, np.nan))
            sp = s.species(r)
            vals = [f"{sp}{int(rec['z'])}+", rec["m"], rec["z"], rec["t0"] * 1e6,
                    rec["TOF"] * 1e6, rec["Yield"], rec["sigma"] * 1e9, rec["B"],
                    ("at edge" if rec.get("at_edge") else "yes") if rec["success"] else "no",
                    rec.get("locked", ""), ovt, "yes" if rec["excluded"] else "", lt * 1e6, ly]
            for c, v in enumerate(vals):
                txt = v if isinstance(v, str) else ("" if not np.isfinite(v) else f"{v:.6g}")
                it = QTableWidgetItem(txt)
                if rec["excluded"]:
                    it.setForeground(QColor("#999999"))
                elif rec.get("at_edge") or not rec["success"]:
                    it.setBackground(QColor("#ffe0b2"))
                if c == 0 and not rec["excluded"]:
                    it.setForeground(QColor(species_color(sp)))
                if c == self.COLS.index("overlap") and ovt:
                    same = "≡" in ovt
                    it.setForeground(QColor("#c62828" if same else "#e65100"))
                    it.setToolTip("≡ same time (cannot be separated); ~ inside the fit window")
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
        seen = set()
        for i, r in enumerate(s.fits.itertuples()):
            sp = s.species(i)
            col = "#bbbbbb" if r.excluded else species_color(sp)
            ax.axvline(r.t0 * 1e6, color=col, ls=":", lw=1.4 if i == sel else 0.7)
            g = s.gaussian(i)
            if g is not None and not r.excluded:
                w = np.abs(s.tof - r.TOF) < 5 * r.sigma
                ax.plot(s.tof[w] * 1e6, g[w], color=col, lw=2.4 if i == sel else 1.2,
                        label=sp if sp not in seen else None)
                seen.add(sp)
        if s.bg:
            ax.plot(s.tof * 1e6, s.bg_curve(), color=self.BG_COLOR, ls="--",
                    lw=1.2 if s.bg_on else 0.8, alpha=1 if s.bg_on else 0.5,
                    label="common background" + ("" if s.bg_on else " (off)"))
            ax.plot([p[0] * 1e6 for p in s.bg["pts"]], [s.bg_level] * 2, "s", ms=6,
                    color=self.BG_COLOR)
            if s.bg["tune"]:
                ax.plot([p[0] * 1e6 for p in s.bg["tune"]], [p[1] for p in s.bg["tune"]],
                        "D", ms=6, mfc="white", color=self.BG_COLOR, label="tuning points")
        if self._bg_click is not None:
            ax.plot(self._bg_click[0] * 1e6, self._bg_click[1], "s", ms=7, mfc="white",
                    color=self.BG_COLOR, label="1st background point")
        for k, t_leg in enumerate(s.legacy.TOF if len(s.legacy) else []):
            ax.axvline(t_leg * 1e6, ymax=0.08, color="0.2", lw=2,
                       label=("pipeline fit" if s.run.is_new else "untitled28 fit")
                       + " position" if k == 0 else None)
        ts = s.fits.t0
        ax.set_xlim(ts.min() * 0.9e6, ts.max() * 1.1e6)
        if sel is not None:
            self._zoom(sel)
        ax.set_xlabel("t [µs]")
        ax.set_ylabel("SEM [V]")
        ax.set_title(f"{s.run.key}, U = {s.U:g} V (windows at {s.U_used:.6g} V, "
                     f"LOF {s.L_used:.6g} m), Corr {s.corr:g}")
        ax.legend(fontsize=8, loc="lower right")
        self.draw_recap()
        self._hover = ax.annotate("", (0, 0), xytext=(12, 12), textcoords="offset points",
                                  fontsize=8, zorder=10, multialignment="left",
                                  bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "0.5"})
        self._hover.set_visible(False)
        self.canvas.draw_idle()

    def _zoom(self, i):
        """Selected line: its window, locks, and both axes fitted to the peak."""
        s, ax = self.s, self.ax
        r = s.fits.iloc[i]
        col = species_color(s.species(i))
        x0, x1 = r.t0 * 0.95, r.t0 * 1.05
        ax.set_xlim(x0 * 1e6, x1 * 1e6)
        ax.axvspan(r.win_lo * 1e6, r.win_hi * 1e6, color=col, alpha=0.08, label="fit window")
        lk = s.lock(i)
        if "B" in lk:
            ax.hlines(lk["B"], r.win_lo * 1e6, r.win_hi * 1e6, color=col, ls="--", lw=1.5,
                      label="background (locked)")
        if "mu" in lk:
            ax.axvline(lk["mu"] * 1e6, color=col, ls="-.", lw=1.5, label="peak (locked)")
        if r.success:                                  # handle: drag to set the peak
            ax.plot([r.TOF * 1e6], [r.Yield + r.B], "o", ms=8, mfc="white", color=col,
                    zorder=5, label="drag: peak position + height"
                    + (" (locked)" if "A" in lk else ""))
        m = (s.tof >= x0) & (s.tof <= x1)
        vals = [s.y[m]]
        g = s.gaussian(i)
        if g is not None:
            vals.append(g[m])
        if "B" in lk:
            vals.append(np.array([lk["B"]]))
        if s.bg_on:
            vals.append(s.bg_curve(s.tof[m]))
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        if len(v):
            lo, hi = float(v.min()), float(v.max())
            pad = 0.08 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)

    def _nearest_line(self, ev, px=8):
        """Index of the line whose peak (fit TOF, else window centre) is within px pixels
        of the mouse in x, or None."""
        s = self.s
        t = np.where(s.fits.success, s.fits.TOF, s.fits.t0).astype(float) * 1e6
        xs = self.ax.transData.transform(np.column_stack([t, np.zeros_like(t)]))[:, 0]
        d = np.abs(xs - ev.x)
        j = int(np.nanargmin(d)) if np.isfinite(d).any() else None
        return j if j is not None and d[j] < px else None

    def on_hover(self, ev):
        if self.s is None or self._hover is None:
            return
        if self._drag is not None:                     # dragging a peak: live preview
            if ev.inaxes is self.ax and ev.xdata is not None:
                d, t = self._drag, ev.xdata * 1e-6
                B = self._drag_B(d["i"], t)
                A = min(ev.ydata - B, 0.0)
                sg = d["sigma"]
                tt = np.linspace(t - 5 * sg, t + 5 * sg, 200)
                Bt = self.s.bg_curve(tt) if self.s.bg_on and "B" not in self.s.lock(d["i"]) \
                    else B
                d["line"].set_data(tt * 1e6, A * np.exp(-0.5 * ((tt - t) / sg) ** 2) + Bt)
                d["dot"].set_data([ev.xdata], [B + A])
                self.canvas.draw_idle()
            return
        j = self._nearest_line(ev) if ev.inaxes is self.ax and ev.xdata is not None else None
        if j is not None:
            r = self.s.fits.iloc[j]
            x = (r.TOF if r.success else r.t0) * 1e6
            right = ev.x > self.canvas.width() * 0.55 * self.canvas.device_pixel_ratio
            self._hover.xy = (x, ev.ydata)
            self._hover.set_text(self.s.info(j))
            self._hover.set_position((-12, 12) if right else (12, 12))
            self._hover.set_ha("right" if right else "left")
            self._hover.set_visible(True)
            self.canvas.draw_idle()
        elif self._hover.get_visible():
            self._hover.set_visible(False)
            self.canvas.draw_idle()

    def on_click(self, ev):
        if self.s is None or ev.inaxes is not self.ax or ev.xdata is None or self.toolbar.mode:
            return
        i = self.sel()
        mode = self.mode.currentIndex()
        if mode >= 3:
            self.bg_click(mode, ev.xdata * 1e-6, ev.ydata)
            if i is not None:
                self.table.selectRow(i)
            return
        if i is None:                                  # full spectrum: pick the line
            j = self._nearest_line(ev)
            if j is not None:
                self.table.selectRow(j)
            return
        r = self.s.fits.iloc[i]
        if r.success and ev.button == 1:               # press on the peak bottom: drag it
            bx, by = self.ax.transData.transform((r.TOF * 1e6, r.Yield + r.B))
            if np.hypot(bx - ev.x, by - ev.y) < 12:
                col = species_color(self.s.species(i))
                self._drag = {"i": i, "sigma": float(r.sigma),
                              "line": self.ax.plot([], [], color=col, ls="--", lw=2)[0],
                              "dot": self.ax.plot([r.TOF * 1e6], [r.Yield + r.B], "o", ms=9,
                                                  mfc="white", color=col)[0]}
                return
        if mode == 0:
            self.s.set_center(i, ev.xdata * 1e-6)
        elif mode == 1:
            self.s.lock_mu(i, ev.xdata * 1e-6)
        else:
            self.s.lock_B(i, ev.ydata)
        r = self.s.fits.iloc[i]
        self.win.log.appendPlainText(
            f"U = {self.s.U:g} V, {self.s.species(i)}{int(r.z)}+: {self.MODES[mode]} -> "
            f"TOF {r.TOF * 1e6:.4f} µs, Yield {r.Yield:.4g} V, B {r.B:.4g} V"
            + (f" (locked {r.locked})" if r.locked else ""))
        self.show_table()
        self.table.selectRow(i)

    def _drag_B(self, i, t_s):
        lk = self.s.lock(i)
        if "B" in lk:
            return lk["B"]
        if self.s.bg_on:
            return float(self.s.bg_curve(np.array([t_s]))[0])
        return float(self.s.fits.iloc[i].B)

    def on_release(self, ev):
        d, self._drag = self._drag, None
        if d is None or self.s is None:
            return
        if ev.inaxes is not self.ax or ev.xdata is None:
            self.draw()
            return
        i = d["i"]
        self.s.lock_peak(i, ev.xdata * 1e-6, ev.ydata)
        r = self.s.fits.iloc[i]
        self.win.log.appendPlainText(
            f"U = {self.s.U:g} V, {self.s.name(i)}: peak dragged -> TOF {r.TOF * 1e6:.4f} µs, "
            f"height {r.Yield:.4g} V, B {r.B:.4g} V, σ {r.sigma * 1e9:.3g} ns "
            f"(locked {r.locked})")
        self.show_table()
        self.table.selectRow(i)

    # ------------------------------------------------------------ yield recap
    def draw_recap(self):
        """Peak heights by species and charge state; hatched: may be blended with another
        line (red xx: same time, cannot be separated; orange //: inside the fit window)."""
        from matplotlib.patches import Patch
        s, ax = self.s, self.rax
        ax.clear()
        self._rbars = []
        self._rhover = None
        if s is None or s.fits is None:
            self.rcanvas.draw_idle()
            return
        ov = s.overlaps()
        sel = self.sel()
        sps = list(dict.fromkeys(s.species(i) for i in range(len(s.lines))))
        w = 0.8 / max(len(sps), 1)
        ymin, ymax = np.inf, 0.0
        for i, r in enumerate(s.fits.itertuples()):
            sp = s.species(i)
            if not r.success or r.excluded or not np.isfinite(r.Yield):
                continue
            h = max(-float(r.Yield), 1e-6)
            kinds = {k for _, _, k, _ in ov.get(i, [])}
            hatch, ec = ("xxx", "#c62828") if "same" in kinds else \
                (("///", "#e65100") if "near" in kinds else (None, species_color(sp)))
            x = int(r.z) + (sps.index(sp) - (len(sps) - 1) / 2) * w
            b = ax.bar(x, h, w * 0.95, color=species_color(sp), alpha=0.75, hatch=hatch,
                       edgecolor="black" if i == sel else ec, lw=2 if i == sel else 0.8)[0]
            self._rbars.append((b, i))
            ymin, ymax = min(ymin, h), max(ymax, h)
        ax.set_yscale("log")
        if np.isfinite(ymin):
            ax.set_ylim(ymin / 3, ymax * 40)               # room for the legend
        ax.set_xlabel("charge state z")
        ax.set_ylabel("peak height |Yield| [V]")
        ax.set_title(f"U = {s.U:g} V: heights by species", fontsize=9)
        zs = sorted({int(z) for _, _, z in s.lines})
        ax.set_xticks(zs)
        ax.tick_params(labelsize=7)
        hand = [Patch(color=species_color(sp), label=sp) for sp in sps]
        hand += [Patch(facecolor="white", edgecolor="#c62828", hatch="xxx",
                       label="same time as another line"),
                 Patch(facecolor="white", edgecolor="#e65100", hatch="///",
                       label="another line in its window")]
        ax.legend(handles=hand, fontsize=6, ncol=2, loc="upper right")
        self._rhover = ax.annotate("", (0, 0), xytext=(8, 8), textcoords="offset points",
                                   fontsize=7, zorder=10, multialignment="left",
                                   bbox={"boxstyle": "round", "fc": "lightyellow", "ec": "0.5"})
        self._rhover.set_visible(False)
        self.rcanvas.draw_idle()

    def _recap_bar(self, ev):
        if ev.inaxes is not self.rax:
            return None
        for b, i in self._rbars:
            if b.contains(ev)[0]:
                return b, i
        return None

    def recap_click(self, ev):
        hit = self._recap_bar(ev)
        if hit is not None:
            self.table.selectRow(hit[1])

    def recap_hover(self, ev):
        if self._rhover is None:
            return
        hit = self._recap_bar(ev)
        if hit is not None:
            b, i = hit
            right = ev.x > self.rcanvas.width() * 0.5 * self.rcanvas.device_pixel_ratio
            self._rhover.xy = (b.get_x() + b.get_width() / 2, b.get_height())
            self._rhover.set_text(self.s.info(i))
            self._rhover.set_position((-8, 8) if right else (8, 8))
            self._rhover.set_ha("right" if right else "left")
            self._rhover.set_visible(True)
            self.rcanvas.draw_idle()
        elif self._rhover.get_visible():
            self._rhover.set_visible(False)
            self.rcanvas.draw_idle()

    def bg_click(self, mode, t, v):
        s, log = self.s, self.win.log.appendPlainText
        if mode == 3:
            if self._bg_click is None:
                self._bg_click = (t, v)
                log(f"common background: 1st point {t * 1e6:.3f} µs, {v:.4g} V; "
                    "click the 2nd")
                self.draw()
                return
            p1, self._bg_click = self._bg_click, None
            s.set_common_bg(p1, (t, v), self.bg_reach.value() / 100)
            log(f"common background: level {s.bg_level:.4g} V for every line (mean of the "
                "2 points); refitted")
        elif not s.bg:
            QMessageBox.information(self, "Peaks", "Define the common background first: "
                                    "click mode 'common background: 2 points'.")
            return
        elif mode == 4:
            s.add_tune(t, v)
            log(f"common background: tuning point {t * 1e6:.3f} µs, {v:.4g} V "
                f"({v - s.bg_level:+.3g} V from the level)")
        else:
            xs = self.ax.transData.transform([(p[0] * 1e6, p[1]) for p in s.bg["tune"]]) \
                if s.bg["tune"] else np.empty((0, 2))
            ex, ey = self.ax.transData.transform((t * 1e6, v))
            d = np.hypot(xs[:, 0] - ex, xs[:, 1] - ey) if len(xs) else np.array([])
            if not len(d) or d.min() > 15:
                log("common background: no tuning point under the click")
                return
            s.remove_tune(s.bg["tune"][int(np.argmin(d))][0])
            log("common background: tuning point removed")
        self.show_table()

    def unlock(self):
        i = self.sel()
        if self.s and i is not None:
            self.s.unlock(i)
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
