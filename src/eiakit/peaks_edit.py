"""
Peak picker and FC baseline picker (Step 9): the logic behind the GUI tools, without Qt.

Peak picker (replaces the per-voltage work of untitled28):
    s = PeakSession.load(db, run, data_root, U)      # raw SEM shots of one voltage
    s.refit(corr=0.0)                                # every line of the mass list, file U
    s.set_center(i, t_s) / s.exclude(i, True)        # move a fit window / drop a line
    s.save(fits_root)        # fits/<run>/peaks/p<U>.txt (Output columns) + _detail.txt
The fit uses the file voltage (no arrow clicks, N16), the LOF of the database, the run's
mass list (labels and z max from its Output rows) and keeps all shot fits (N12).
The calibration replaces the Output rows of that voltage by the saved ones
(calibration.prepare_rows: peak_overrides), so a refitted voltage enters the spectra
like a clean Output file: one save, no duplicates (N1), the right voltage (N14).

FC baseline picker (N7: untitled27 did not store the pedestal):
    b = BaselineSession.load(db, run, data_root)
    b.save(fits_root, volts)  # fits/<run>/fc_baseline.json, used with fc_baseline = "picked"
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .peaks import average_repeats, fit_peak, gui_lof, gui_voltage, predicted_tof


def _ustr(U: float) -> str:
    """p<U>.txt name part as untitled28 wrote it (10.0 -> '10', 200.01 -> '200.01')."""
    return f"{U:g}" if float(U) != int(U) else str(int(U))


def peaks_dir(fits_root, run) -> Path:
    return Path(fits_root) / run.key / "peaks"


def override_files(fits_root, run) -> dict[float, Path]:
    """{U: fits/<run>/peaks/p<U>.txt} of the saved voltages."""
    from .legacy import voltage_from_output_name
    d = peaks_dir(fits_root, run) if fits_root is not None else None
    if d is None or not d.is_dir():
        return {}
    out = {}
    for p in sorted(d.glob("p*.txt")):
        if p.stem.endswith("_detail"):
            continue
        U = voltage_from_output_name(p.name)
        if U is not None:
            out[U] = p
    return out


@dataclass
class PeakSession:
    db: object
    run: object
    U: float
    tof: np.ndarray                  # s
    y: np.ndarray                    # averaged SEM signal (V)
    traces: dict                     # rep -> (t, v)
    lines: list                      # (m_amu, label, z)
    legacy: pd.DataFrame             # Output_detail rows of this U (empty if none)
    L_m: float
    profile: object
    settings: dict
    k: float = 1.0                   # window-centre factor (new runs: from the peaks stage)
    k_auto: float = 1.0              # the stage's factor (auto centring on), kept to restore
    u_clicks: int = 0                # untitled28 arrow clicks on U (+/- U/500 each)
    lof_clicks: int = 0              # untitled28 arrow clicks on LOF (+/- LOF/100 each)
    centers: dict = field(default_factory=dict)      # i -> t0 override (s)
    excluded: set = field(default_factory=set)
    corr: float = 0.0
    fits: pd.DataFrame | None = None

    @classmethod
    def voltages(cls, run, data_root) -> list[float]:
        from .acquisition import scan_folder
        sem = run.folder(data_root) / run.acquisition.sem.folder
        return sorted({u for _, u, _ in scan_folder(sem, run.acquisition.sem).matched
                       if u is not None})

    @classmethod
    def load(cls, db, run, data_root, U: float, all_lines: bool = False,
             processed=None) -> PeakSession:
        """all_lines: every z up to the run's z max for every mass; default: the (m, z)
        the Output of this voltage holds (what was fitted then), else every line."""
        from .acquisition import read_trace, scan_folder
        from .legacy import read_output_folder
        fmt = run.acquisition.sem
        sem = run.folder(data_root) / fmt.folder
        files = {rep: p for p, u, rep in scan_folder(sem, fmt).matched if u == U}
        if not files:
            raise FileNotFoundError(f"{run.key}: no raw SEM files for U = {U:g} V")
        traces = {rep: read_trace(p, fmt) for rep, p in sorted(files.items())}
        tof, y = average_repeats([traces[r] for r in sorted(traces)])
        settings = dict(db.pipeline["peaks"])
        gui = {int(k): float(v) for k, v in settings.pop("gui_masses", {}).items()}
        if run.is_new:
            # new run: the fits of the peaks stage are the reference; same lines, same
            # window-centre factor k of this voltage
            from .peaks import new_run_lines
            stage = Path(processed) / run.key / "10_peaks" if processed is not None else None
            det = read_output_folder(stage / "output_detail") if stage else pd.DataFrame()
            k = 1.0
            if stage is not None and (stage / "fits.csv").is_file():
                f = pd.read_csv(stage / "fits.csv")
                f = f[np.isclose(f.U, U)]
                k = float(f.k.iloc[0]) if len(f) else 1.0
            prof = db.profile(db.profile_for(run, "peaks"))
            s = cls(db, run, U, tof, y, traces, new_run_lines(db, run),
                    det[det.U == U] if len(det) else det, run.eia.L_m, prof, settings, k=k,
                    k_auto=k)
            s.refit()
            return s
        rows = read_output_folder(sem / "Output")
        det = read_output_folder(sem / "Output_detail")
        zmax = (rows.groupby("m").z.max().to_dict() if len(rows)
                else {int(m): 1 for m in run.mass_labels})
        lines = [(gui.get(int(m), float(m)), int(m), z) for m, zm in sorted(zmax.items())
                 for z in range(1, int(zm) + 1)]
        leg = det[det.U == U] if len(det) else det
        here = rows[rows.U == U] if len(rows) else rows
        if not all_lines and len(here):
            have = set(zip(here.m.astype(int), here.z.astype(int), strict=True))
            lines = [ln for ln in lines if (ln[1], ln[2]) in have]
        prof = db.profile(run.legacy.profile_peaks if run.legacy else "legacy_untitled28")
        L = run.legacy.peaks_lof_m if run.legacy and run.legacy.peaks_lof_m else run.eia.L_m
        s = cls(db, run, U, tof, y, traces, lines, leg, L, prof, settings)
        s.refit()
        return s

    def restore(self, fits_root) -> bool:
        """Load the settings of a saved voltage (fits/<run>/peaks/p<U>.json): clicks, Corr,
        moved windows, excluded lines, centring; refit. True if there was one."""
        p = peaks_dir(fits_root, self.run) / f"p{_ustr(self.U)}.json"
        if fits_root is None or not p.is_file():
            return False
        m = json.loads(p.read_text(encoding="utf-8"))
        self.u_clicks = int(m.get("u_clicks", 0))
        self.lof_clicks = int(m.get("lof_clicks", 0))
        self.corr = float(m.get("corr", 0.0))
        self.k = float(m.get("auto_centre_k", self.k))
        self.centers = {int(i): float(t) for i, t in m.get("centers_s", {}).items()}
        ex = {tuple(x) for x in m.get("excluded", [])}
        self.excluded = {i for i, (_, lab, z) in enumerate(self.lines) if (lab, z) in ex}
        self.refit()
        return True

    @property
    def U_used(self) -> float:
        """U of the fit windows: the file voltage after the arrow clicks (untitled28)."""
        return gui_voltage(self.U, self.u_clicks, self.settings.get("u_step_frac", 0.002))

    @property
    def L_used(self) -> float:
        return gui_lof(self.L_m, self.lof_clicks, self.settings.get("lof_step_frac", 0.01))

    def set_clicks(self, u_clicks: int | None = None, lof_clicks: int | None = None):
        """Arrow clicks on U and LOF as in untitled28: every expected line moves; refit."""
        if u_clicks is not None:
            self.u_clicks = int(u_clicks)
        if lof_clicks is not None:
            self.lof_clicks = int(lof_clicks)
        self.refit()

    def set_auto_centre(self, on: bool):
        """Automatic window centring of new runs on (factor k of the peaks stage) or off
        (expected lines from U and LOF only, as in untitled28)."""
        self.k = self.k_auto if on else 1.0
        self.refit()

    def t0(self, i) -> float:
        m_amu, _, z = self.lines[i]
        return self.centers.get(i, predicted_tof(m_amu, z, self.U_used, self.L_used,
                                                 self.run.eia, self.profile) * self.k)

    def refit(self, corr: float | None = None):
        if corr is not None:
            self.corr = corr
        out = []
        reps = sorted(self.traces)
        for i, (_m_amu, label, z) in enumerate(self.lines):
            t0 = self.t0(i)
            f = fit_peak(self.tof, self.y, t0, self.corr, self.settings.get("window_frac", 0.02),
                         self.settings.get("mu_bound_s", 150e-9),
                         self.settings.get("min_points", 1))
            row = {"i": i, "m": label, "z": z, "t0": t0, "TOF": f["mu"], "Yield": f["A"],
                   "sigma": f["sigma"], "B": f["B"], "success": bool(f["success"]),
                   "at_edge": bool(f["success"] and f.get("mu_at_bound", False)),
                   "excluded": i in self.excluded}
            if f["success"]:
                vals = [float(np.interp(f["mu"], *self.traces[r])) for r in reps]
                row.update(YieldAveFit=f["A"], YieldAve=float(np.mean(vals)),
                           YieldSTDV=float(np.std(vals)))
                row.update({f"Yield{k + 1}": v for k, v in enumerate(vals)})
            out.append(row)
        self.fits = pd.DataFrame(out)
        return self.fits

    def set_center(self, i, t_s: float | None):
        if t_s is None:
            self.centers.pop(i, None)
        else:
            self.centers[i] = float(t_s)
        self.refit()

    def exclude(self, i, flag: bool = True):
        (self.excluded.add if flag else self.excluded.discard)(i)
        self.refit()

    def gaussian(self, i, t=None):
        r = self.fits.iloc[i]
        t = self.tof if t is None else t
        if not r.success:
            return None
        return r.Yield * np.exp(-0.5 * ((t - r.TOF) / r.sigma) ** 2) + r.B

    def save(self, fits_root, note: str = "") -> Path:
        ok = self.fits[self.fits.success & ~self.fits.excluded]
        d = peaks_dir(fits_root, self.run)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"p{_ustr(self.U)}.txt"
        ok[["m", "z", "TOF", "Yield"]].to_csv(p, index=False)
        cols = ["m", "z", "TOF", "YieldAveFit", "YieldAve", "YieldSTDV",
                *[c for c in ok.columns if c.startswith("Yield") and c[5:].isdigit()]]
        ok[[c for c in cols if c in ok]].to_csv(d / f"p{_ustr(self.U)}_detail.txt", index=False)
        meta = {"U": self.U, "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "note": note,
                "L_m": self.L_m, "corr": self.corr, "centers_s": self.centers,
                "u_clicks": self.u_clicks, "U_windows": self.U_used,
                "lof_clicks": self.lof_clicks, "LOF_windows": self.L_used,
                "auto_centre_k": self.k,
                "excluded": [list(self.lines[i][1:]) for i in sorted(self.excluded)],
                "files": len(self.traces)}
        (d / f"p{_ustr(self.U)}.json").write_text(json.dumps(meta, indent=1, default=str),
                                                  encoding="utf-8")
        return p


def baseline_path(fits_root, run) -> Path:
    return Path(fits_root) / run.key / "fc_baseline.json"


def picked_baseline(fits_root, run) -> float | None:
    p = baseline_path(fits_root, run) if fits_root is not None else None
    if p is None or not p.is_file():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["baseline_V"])


@dataclass
class BaselineSession:
    run: object
    t: np.ndarray                    # s, FC average on the native grid
    v: np.ndarray                    # V, raw (pedestal not subtracted)
    pretrigger: float
    legacy: float | None             # pedestal recovered from the legacy checkpoint
    picked: float | None

    @classmethod
    def load(cls, db, run, data_root, fits_root=None) -> BaselineSession:
        from .acquisition import read_trace, scan_folder
        from .calibration import average_fc, fc_dirs, recover_fc_baseline
        from .legacy import load_checkpoint
        folder = run.folder(data_root)
        src = db.pipeline["calibration"].get("fc_source", "start")
        traces = [read_trace(p, run.acquisition.fc)
                  for d in fc_dirs(run, src) if d and (folder / d).is_dir()
                  for p, _, _ in scan_folder(folder / d, run.acquisition.fc).matched]
        if not traces:
            raise FileNotFoundError(f"{run.key}: no FC traces ({src})")
        t, v = average_fc(traces, "native")
        pre = float(np.mean(np.concatenate([vv[tt < 0] for tt, vv in traces])))
        ck = folder / "EIA-Results" / "pipeline_checkpoint.npy"
        leg = recover_fc_baseline(load_checkpoint(ck), traces, eia=run.eia, fc=run.fc) \
            if ck.is_file() else None
        return cls(run, t, v, pre, leg, picked_baseline(fits_root, run))

    def window_mean(self, t0: float, t1: float) -> float:
        m = (self.t >= min(t0, t1)) & (self.t <= max(t0, t1))
        return float(self.v[m].mean()) if m.any() else float("nan")

    def save(self, fits_root, volts: float, how: str = "picked") -> Path:
        p = baseline_path(fits_root, self.run)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"baseline_V": float(volts), "how": how,
                                 "pretrigger_V": self.pretrigger, "legacy_V": self.legacy,
                                 "saved": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1),
                     encoding="utf-8")
        self.picked = float(volts)
        return p
