"""
Peak picker and FC baseline picker (Step 9): the logic behind the GUI tools, without Qt.

Peak picker (replaces the per-voltage work of untitled28):
    s = PeakSession.load(db, run, data_root, U)      # raw SEM shots of one voltage
    s.refit(corr=0.0)                                # every line of the mass list, file U
    s.set_center(i, t_s) / s.exclude(i, True)        # move a fit window / drop a line
    s.lock_mu(i, t_s) / s.lock_B(i, volts) / s.unlock(i)   # hand-set peak position / background
    s.lock_peak(i, t_s, volts)            # dragged peak: position and height locked
    s.overlaps()                          # lines that may be blended with others
    s.set_common_bg((t1, v1), (t2, v2))   # one background for every line: horizontal at
    s.add_tune(t, v) / s.remove_tune(t)   # the mean of 2 clicks, + local tuning points
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

from .peaks import average_repeats, fit_peak_locked, gui_lof, gui_voltage, predicted_tof


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
    locks: dict = field(default_factory=dict)        # (label, z) -> {"mu": s, "B": V}
    # common background of this voltage: None or {"pts": [(t, V), (t, V)], "tune":
    # [(t, V)], "reach": frac, "on": bool}; a per-line B lock wins over it
    bg: dict | None = None

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
        self.locks = {(int(a), int(b)): {k: float(v) for k, v in d.items()}
                      for a, b, d in m.get("locks", [])}
        cb = m.get("common_bg")
        self.bg = None if not cb else {"pts": [tuple(p) for p in cb["pts"]],
                                       "tune": [tuple(p) for p in cb.get("tune", [])],
                                       "reach": float(cb.get("reach", 0.03)),
                                       "on": bool(cb.get("on", True))}
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

    def predicted(self, i) -> float:
        m_amu, _, z = self.lines[i]
        return predicted_tof(m_amu, z, self.U_used, self.L_used, self.run.eia,
                             self.profile) * self.k

    def t0(self, i) -> float:
        """Window centre: moved by hand, else the locked peak position, else predicted."""
        if i in self.centers:
            return self.centers[i]
        mu = self.lock(i).get("mu")
        return mu if mu is not None else self.predicted(i)

    # ---------------------------------------------------------------- hand locks
    def key(self, i) -> tuple[int, int]:
        return int(self.lines[i][1]), int(self.lines[i][2])

    def lock(self, i) -> dict:
        return self.locks.get(self.key(i), {})

    def lock_mu(self, i, t_s: float | None):
        """Peak position of line i fixed at t_s (None: free again); A, sigma (and B unless
        locked) are fitted around it."""
        self._set_lock(i, "mu", t_s)

    def lock_peak(self, i, t_s: float, volts: float):
        """Dragged peak: the bottom of line i at (t_s, volts). Position and height (depth
        below the background) are locked; the background is the line's locked B, else the
        common one, else the current fitted B, which is then locked too so the bottom
        stays where it was put. Only sigma is fitted afterwards."""
        lk = dict(self.lock(i))
        if "B" in lk:
            B = lk["B"]
        elif self.bg_on:
            B = float(self.bg_curve(np.array([t_s]))[0])
        else:
            r = self.fits.iloc[i]
            B = float(r.B) if np.isfinite(r.B) else float(np.median(self.y))
            lk["B"] = B
        lk.update(mu=float(t_s), A=min(float(volts) - B, 0.0))
        self.locks[self.key(i)] = lk
        self.refit()

    def lock_B(self, i, volts: float | None):
        """Background (baseline) of line i fixed at volts (None: free again)."""
        self._set_lock(i, "B", volts)

    def unlock(self, i):
        self.locks.pop(self.key(i), None)
        self.refit()

    def _set_lock(self, i, what, v):
        d = dict(self.lock(i))
        if v is None:
            d.pop(what, None)
        else:
            d[what] = float(v)
        if d:
            self.locks[self.key(i)] = d
        else:
            self.locks.pop(self.key(i), None)
        self.refit()

    # ------------------------------------------------------------ common background
    @property
    def bg_on(self) -> bool:
        return bool(self.bg and self.bg.get("on", True))

    def set_common_bg(self, p1, p2, reach: float | None = None):
        """Common background from two clicked points (t s, V): horizontal at their mean,
        for every line of this voltage. Tuning points are kept."""
        old = self.bg or {}
        self.bg = {"pts": [tuple(map(float, p1)), tuple(map(float, p2))],
                   "tune": old.get("tune", []),
                   "reach": float(reach if reach is not None else old.get("reach", 0.03)),
                   "on": True}
        self.refit()

    def set_bg_on(self, on: bool):
        if self.bg:
            self.bg["on"] = bool(on)
            self.refit()

    def set_bg_reach(self, reach: float):
        if self.bg:
            self.bg["reach"] = float(reach)
            self.refit()

    def clear_common_bg(self):
        self.bg = None
        self.refit()

    @property
    def bg_level(self) -> float:
        return float(np.mean([v for _, v in self.bg["pts"]]))

    def add_tune(self, t: float, v: float):
        """Tuning point: the background goes through (t, v); it changes only within
        +/- reach x t of it (or up to the next tuning point, if closer)."""
        if not self.bg:
            raise ValueError("define the common background first (2 points)")
        self.bg["tune"] = sorted([p for p in self.bg["tune"] if p[0] != t] +
                                 [(float(t), float(v))])
        self.refit()

    def remove_tune(self, t: float) -> bool:
        """Remove the tuning point nearest to t (s). False if there is none."""
        if not self.bg or not self.bg["tune"]:
            return False
        j = int(np.argmin([abs(p[0] - t) for p in self.bg["tune"]]))
        del self.bg["tune"][j]
        self.refit()
        return True

    def bg_curve(self, t=None) -> np.ndarray | None:
        """Common background on t (default: the tof grid): level + offsets of the tuning
        points, linear between neighbours and back to the level within reach of each."""
        if not self.bg:
            return None
        t = self.tof if t is None else np.asarray(t, float)
        lvl, w = self.bg_level, self.bg.get("reach", 0.03)
        tune = sorted(self.bg["tune"])
        if not tune:
            return np.full(t.shape, lvl)
        xs, ds = [tune[0][0] * (1 - w)], [0.0]
        for k, (tk, vk) in enumerate(tune):
            xs.append(tk)
            ds.append(vk - lvl)
            nxt = tune[k + 1][0] if k + 1 < len(tune) else None
            if nxt is None or nxt - tk > w * (tk + nxt):
                xs.append(tk * (1 + w))
                ds.append(0.0)
                if nxt is not None:
                    xs.append(nxt * (1 - w))
                    ds.append(0.0)
        return lvl + np.interp(t, xs, ds, left=0.0, right=0.0)

    def species(self, i) -> str:
        """Species of line i, from the mass label (species.toml data_labels)."""
        lab = self.lines[i][1]
        for name, sp in self.db.species.items():
            if lab in sp.data_labels or lab == sp.file_label:
                return name
        return str(lab)

    def info(self, i) -> str:
        """Text for the hover label of line i."""
        r = self.fits.iloc[i]
        m_amu, lab, z = self.lines[i]
        sp = self.species(i)
        lk = self.lock(i)
        out = [f"{sp}{z}+  (m {lab}, z {z}, m/z {m_amu / z:.3g})",
               f"expected t {self.predicted(i) * 1e6:.4f} µs, window centre {r.t0 * 1e6:.4f} µs"]
        if r.success:
            dev = (r.TOF / self.predicted(i) - 1) * 100
            out += [f"fit TOF {r.TOF * 1e6:.4f} µs ({dev:+.2f} % from expected)",
                    f"Yield {r.Yield:.4g} V, σ {r.sigma * 1e9:.3g} ns, B {r.B:.4g} V"]
        else:
            out.append("fit failed")
        if r.get("common_bg", False):
            out.append("background: common (all lines)")
        if lk:
            fmt = {"mu": lambda v: f"μ {v * 1e6:.4f} µs", "A": lambda v: f"height {v:.4g} V",
                   "B": lambda v: f"B {v:.4g} V"}
            out.append("locked: " + ", ".join(fmt[k](v) for k, v in lk.items()))
        ov = self.overlaps().get(i, [])
        if ov:
            out.append("overlap: " + ", ".join(
                f"{'same time as' if kind == 'same' else 'near'} {name} "
                f"({dt * 1e9:+.0f} ns)" for _, name, kind, dt in ov))
        if r.get("at_edge"):
            out.append("at the edge of the ± search")
        if r.excluded:
            out.append("EXCLUDED")
        return "\n".join(out)

    def name(self, i) -> str:
        return f"{self.species(i)}{int(self.lines[i][2])}+"

    def sigma_rel(self) -> float:
        """Typical peak width sigma / t of this voltage (median of the good fits)."""
        f = self.fits
        if f is None:
            return 0.005
        ok = f.success & ~f.excluded & ~f.at_edge & (f.sigma > 0)
        v = (f.sigma[ok] / f.TOF[ok]).to_numpy(float)
        return float(np.median(v)) if len(v) else 0.005

    def overlaps(self) -> dict:
        """{i: [(j, name_j, kind, t_j - t_i)]} from the expected times: kind 'same' when
        the two lines are less than one typical sigma apart (cannot be separated: same
        m/z within the resolution, e.g. Al9+ / C4+), 'near' when line j is inside line i's
        fit window or within 2 sigma of it (its tail is in the fit, e.g. Al8+ / O5+)."""
        n = len(self.lines)
        t = np.array([self.predicted(i) for i in range(n)])
        half = max(self.settings.get("window_frac", 0.02) + self.corr, 1e-3)
        sr = self.sigma_rel()
        out = {}
        for i in range(n):
            for j in range(n):
                if i == j or j in self.excluded:
                    continue
                d = t[j] - t[i]
                tm = max(t[i], t[j])                   # symmetric: both lines are affected
                if abs(d) < sr * tm:
                    kind = "same"
                elif abs(d) < (half + 2 * sr) * tm:
                    kind = "near"
                else:
                    continue
                out.setdefault(i, []).append((j, self.name(j), kind, float(d)))
        return out

    def refit(self, corr: float | None = None):
        if corr is not None:
            self.corr = corr
        out = []
        reps = sorted(self.traces)
        Bt = self.bg_curve() if self.bg_on else None
        for i, (_m_amu, label, z) in enumerate(self.lines):
            t0 = self.t0(i)
            lk = self.lock(i)
            f = fit_peak_locked(self.tof, self.y, t0, self.corr,
                                self.settings.get("window_frac", 0.02),
                                self.settings.get("mu_bound_s", 150e-9),
                                self.settings.get("min_points", 1), lk.get("mu"), lk.get("B"),
                                Bt, lk.get("A"))
            common = Bt is not None and "B" not in lk
            row = {"i": i, "m": label, "z": z, "t0": t0, "TOF": f["mu"], "Yield": f["A"],
                   "sigma": f["sigma"], "B": f["B"], "success": bool(f["success"]),
                   "win_lo": f["win_lo"], "win_hi": f["win_hi"],
                   "locked": "".join(x for x, k in (("μ", "mu"), ("h", "A"), ("B", "B"))
                                     if k in lk)
                   + ("Bc" if common else ""),
                   "common_bg": common,
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
        B = self.bg_curve(t) if r.get("common_bg", False) else r.B
        return r.Yield * np.exp(-0.5 * ((t - r.TOF) / r.sigma) ** 2) + B

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
                "locks": [[a, b, d] for (a, b), d in sorted(self.locks.items())],
                "common_bg": self.bg,
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
