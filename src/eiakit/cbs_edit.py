"""
CBS fitting session (Step 9): the logic behind the GUI CBS fitter, without Qt.

    s = CBSSession.load(db, run, data_root, processed)   # store if present, else legacy
    s.fit(z, i)                 # Fit button of cbs_manual_fitter_v12 (kT scan, V0 from u, A)
    s.set_peak(z, i, t_us)      # pick the peak time (V0 follows from kT and u = L / t)
    s.set_kT(z, i, kT)          # kT by hand (V0 from u, A analytic)
    s.kT_from_fwhm(z, i)        # kT from the picked FWHM
    s.add(z, "slow", t_peak, fwhm) / s.remove(z, i)
    s.save(fits_root)           # -> fits/<run>/cbs_fit.json (physical labels, every z)

Components carry their physical label in Component.extra["label"]; the CBS number of the
store is derived from it (cbs_store), so a component can never change name by pick order.
Spectra: the current calibration (processed/<run>/20_calibration/spectra.h5) when it
exists, else the legacy checkpoint.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cbs import CBSModel, Component, run_model
from .cbs_store import TABLE_COLS, write_store
from .legacy import CBSComponent, CBSFitFile, FileHeader

LABEL_ORDER = ["ultra_fast", "fast", "slow", "component_4"]


@dataclass
class CBSSession:
    db: object
    run: object
    model: CBSModel
    curves: dict                     # z -> per-ion signal on model.tof
    exp: dict                        # z -> (t_us, y) measured points
    comps: dict                      # z -> [Component]
    source: str
    dirty: set = field(default_factory=set)

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, db, run, data_root, processed=None) -> CBSSession:
        ck = None
        if processed is not None:
            from pathlib import Path

            from .store import load_calibration_h5
            p = Path(processed) / run.key / "20_calibration" / "spectra.h5"
            if p.is_file():
                ck = load_calibration_h5(p)
        rm = run_model(db, run, data_root, ck)
        if rm["status"] != "ok":
            raise FileNotFoundError(f"{run.key}: {rm['status']} {rm.get('files')}")
        names = run.legacy.cbs_components if run.legacy else {}
        comps = copy.deepcopy(rm["components"])
        for cs in comps.values():
            for c in cs:
                c.extra["label"] = names.get(c.extra.get("cbs"), f"component_{c.extra.get('cbs')}")
        src = ("store " if rm.get("from_store") else "legacy ") + str(rm["fit_path"])
        return cls(db, run, rm["model"], rm["curves"], rm["exp"], comps, src)

    # ---------------------------------------------------------------- views
    @property
    def zs(self) -> list[int]:
        return sorted(set(self.curves) | set(self.comps))

    def t_us(self):
        return self.model.tus

    def component_curve(self, z, i):
        c = self.comps[z][i]
        if not c.active:
            return None
        return c.A * self.model.shape(c.kT, c.V0, z)

    def total(self, z):
        ys = [self.component_curve(z, i) for i in range(len(self.comps.get(z, [])))]
        ys = [y for y in ys if y is not None]
        return sum(ys) if ys else np.zeros_like(self.model.tof)

    def table(self, z) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(self.comps.get(z, [])):
            m = self.model.metrics(c, z) if c.active else {}
            rows.append({"i": i, "label": c.extra.get("label"), "t_peak_us": c.t_peak,
                         "FWHM_us": c.fwhm, "kT_eV": c.kT, "V0_V": c.V0, "A": c.A,
                         "u_kms": (self.model.L_m / (c.t_peak * 1e-6) / 1e3
                                   if c.t_peak else np.nan),
                         "t_peak_fit_us": m.get("t_peak", np.nan),
                         "FWHM_fit_us": m.get("fwhm", np.nan),
                         "win_left": c.win_left, "win_right": c.win_right})
        return pd.DataFrame(rows)

    def chi2(self, z) -> tuple[float, int]:
        """(chi2, dof) of all active components of z inside their windows."""
        cs = self.comps.get(z, [])
        if not any(c.active for c in cs) or z not in self.curves:
            return np.nan, 0
        return self.model.chi2_total(cs, self.curves[z], z)

    def areas(self, z):
        cs = self.comps.get(z, [])
        return self.model.areas(cs, self.curves[z], z) if z in self.curves else None

    # ---------------------------------------------------------------- edits
    def _u(self, c):
        return self.model.L_m / (c.t_peak * 1e-6)

    def _refresh(self, z, i, keep_A=False):
        c = self.comps[z][i]
        if c.kT is None or c.t_peak is None:
            return
        V0 = self.model.V0_from_u(c.kT, self._u(c), z)
        c.V0 = V0
        if V0 is None:
            c.A = None
            return
        if not keep_A:
            r = self.model.residual(self.comps[z], i, self.curves[z], z)
            A = self.model.analytic_A(self.model.shape(c.kT, V0, z), r, self.model.mask(c))
            c.A = None if not np.isfinite(A) or A <= 0 else float(A)
        self.dirty.add(z)

    def auto_kT_hi(self, z, i) -> float:
        """Upper end of the kT scan: max(kT_max, 3 x the current kT or the kT from the FWHM)."""
        c = self.comps[z][i]
        k = c.kT
        if k is None and c.fwhm and c.t_peak:
            k = self.model.kT_from_fwhm(c.fwhm, self._u(c), z)[0]
        return max(self.model.kT_max, 3.0 * k) if k else self.model.kT_guess_max

    def fit(self, z, i, kT_hi=None):
        """Fit button: kT scan (up to kT_hi, default auto_kT_hi), V0 from u, A analytic."""
        kT_hi = kT_hi or self.auto_kT_hi(z, i)
        out = self.model.fit_component(self.comps[z], i, self.curves[z], z, kT_hi)
        self.dirty.add(z)
        return out

    def set_peak(self, z, i, t_us: float):
        c = self.comps[z][i]
        c.t_peak = float(t_us)
        if c.win_src == "default":
            c.win_left = c.win_right = None
        self._refresh(z, i)

    def set_kT(self, z, i, kT: float):
        self.comps[z][i].kT = float(kT)
        self._refresh(z, i)

    def set_A(self, z, i, A: float):
        self.comps[z][i].A = float(A)
        self.dirty.add(z)

    def set_fwhm(self, z, i, fwhm_us: float):
        self.comps[z][i].fwhm = float(fwhm_us)
        self.dirty.add(z)

    def set_window(self, z, i, left_us: float | None, right_us: float | None):
        c = self.comps[z][i]
        c.win_left, c.win_right = left_us, right_us
        c.win_src = "default" if left_us is None else "picked"
        self._refresh(z, i)

    def kT_from_fwhm(self, z, i):
        c = self.comps[z][i]
        kT, V0 = self.model.kT_from_fwhm(c.fwhm, self._u(c), z)
        if kT is not None:
            c.kT = kT
            self._refresh(z, i)
        return kT

    def add(self, z, label: str, t_peak_us: float, fwhm_us: float | None = None,
            kT: float | None = None) -> int:
        c = Component(t_peak=float(t_peak_us), fwhm=float(fwhm_us or 0.05 * t_peak_us),
                      extra={"label": label})
        self.comps.setdefault(z, []).append(c)
        i = len(self.comps[z]) - 1
        k0 = kT if kT is not None else self.model.kT_from_fwhm(c.fwhm, self._u(c), z)[0]
        if k0 is not None:
            c.kT = float(k0)
            self._refresh(z, i)
        self.dirty.add(z)
        return i

    def remove(self, z, i):
        del self.comps[z][i]
        self.dirty.add(z)

    # ---------------------------------------------------------------- save
    def _triplet(self, z, t_fit):
        te, ye = self.exp.get(z, (np.array([]), np.array([])))
        if not len(te) or not np.isfinite(t_fit):
            return {}
        j = int(np.argmin(np.abs(te - t_fit)))
        out = {}
        for key, k in (("prev", j - 1), ("PEAK", j), ("next", j + 1)):
            if 0 <= k < len(te):
                out[key] = {"t": float(te[k]), "I": float(ye[k]),
                            "u": float(self.model.L_m / (te[k] * 1e-6) / 1e3)}
        return out

    def to_fit_file(self) -> CBSFitFile:
        names = self.run.legacy.cbs_components if self.run.legacy else {}
        lab2num = {v: k for k, v in names.items()}
        nxt = max(lab2num.values(), default=0) + 1
        rows, blocks = [], []
        for z in self.zs:
            for c in self.comps.get(z, []):
                lab = c.extra.get("label", "fast")
                if lab not in lab2num:
                    lab2num[lab], nxt = nxt, nxt + 1
                n = lab2num[lab]
                m = self.model.metrics(c, z) if c.active else {}
                wl, wr = self.model.edges(c) if c.t_peak is not None else (np.nan, np.nan)
                row = {"z": z, "CBS": n, "t_peak_pick": c.t_peak,
                       "t_peak_fit": m.get("t_peak", np.nan), "FWHM_pick": c.fwhm,
                       "FWHM_fit": m.get("fwhm", np.nan), "t_left_pick": c.t_left,
                       "t_left_fit": m.get("t_left", np.nan), "t_right_pick": c.t_right,
                       "t_right_fit": m.get("t_right", np.nan), "I_peak_click": np.nan,
                       "I_peak_exp": np.nan, "I_peak_fit": np.nan, "win_left": wl,
                       "win_right": wr, "win_src": c.win_src.replace(" ", "_"),
                       "kT_eV": np.nan if c.kT is None else c.kT,
                       "V0_V": np.nan if c.V0 is None else c.V0,
                       "A": np.nan if c.A is None else c.A}
                rows.append(row)
                blocks.append(CBSComponent(
                    z=z, cbs=n, picks={"t_peak": c.t_peak, "t_left": c.t_left,
                                       "t_right": c.t_right, "FWHM": c.fwhm},
                    window={"win_left": wl, "win_right": wr, "src": c.win_src},
                    triplet=self._triplet(z, m.get("t_peak", np.nan)),
                    kT=row["kT_eV"], V0=row["V0_V"], A=row["A"]))
        h = FileHeader(generated=None, L_EIA_m=self.model.L_m, M_ION_kg=self.model.m_kg,
                       signal="per_ion" if self.model.per_ion else "charge")
        return CBSFitFile(path=None, header=h, z_totals={}, components=blocks,
                          table=pd.DataFrame(rows, columns=["z", "CBS", *TABLE_COLS]))

    def save(self, fits_root, note: str = ""):
        """Write the store (previous version kept as cbs_fit.<time>.json)."""
        p = write_store(fits_root, self.run, self.to_fit_file(), self.model.e_C,
                        source=f"eiakit CBS fitter (from {self.source})", note=note)
        self.dirty.clear()
        return p


def default_label(existing: list[str]) -> str:
    for lab in LABEL_ORDER:
        if lab not in existing:
            return lab
    return f"component_{len(existing) + 1}"
