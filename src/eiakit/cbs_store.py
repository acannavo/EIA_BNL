"""
CBS fit store (Step 9): the CBS components of a run as edited in the eiakit CBS fitter,
in <fits>/<run>/cbs_fit.json (paths.fits; default NewEIASoftware/fits).

Unlike the legacy CBS_fit_results file, the store
  - names every component by its physical label (fast, slow, ultra_fast, ...), never by
    pick order (N3); the CBS number is derived from the run's legacy map, so every
    consumer of the legacy file works unchanged;
  - always holds every charge state (the fitter overwrote the file with the z of the
    current session only, N2);
  - records the constants its kT, V0 belong to (M_ION, e, L_EIA, signal), so a change of
    the constants rescales them exactly (I1).

load_fit(db, run, data_root, fits_root) returns the store when it exists, else the legacy
file, as the same CBSFitFile structure (legacy.CBSComponent list + header).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .legacy import CBSComponent, CBSFitFile, FileHeader, parse_cbs_fit_results


@dataclass
class StoreHeader(FileHeader):
    e_C: float | None = None             # elementary charge of the stored kT, V0


def store_path(fits_root, run) -> Path:
    return Path(fits_root) / run.key / "cbs_fit.json"


def _labels(run) -> tuple[dict, dict]:
    num2lab = dict(run.legacy.cbs_components) if run.legacy else {}
    lab2num = {v: k for k, v in num2lab.items()}
    return num2lab, lab2num


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


TABLE_COLS = ["t_peak_pick", "t_peak_fit", "FWHM_pick", "FWHM_fit", "t_left_pick",
              "t_left_fit", "t_right_pick", "t_right_fit", "I_peak_click", "I_peak_exp",
              "I_peak_fit", "win_left", "win_right", "win_src", "kT_eV", "V0_V", "A"]


def _clean(v):
    if isinstance(v, str):
        return v
    try:
        return _f(float(v))
    except (TypeError, ValueError):
        return None


def write_store(fits_root, run, fit: CBSFitFile, e_C: float, source: str,
                note: str = "") -> Path:
    """fit: a CBSFitFile (legacy or edited): table rows (full precision) + text blocks."""
    num2lab, _ = _labels(run)
    p = store_path(fits_root, run)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_file():                                  # keep the previous version
        bak = p.with_name(f"cbs_fit.{time.strftime('%Y%m%d-%H%M%S')}.json")
        p.replace(bak)
    comps = []
    for r in fit.table.sort_values(["z", "CBS"]).to_dict("records"):
        z, n = int(r["z"]), int(r["CBS"])
        blk = fit.component(z, n)
        comps.append({
            "z": z, "label": num2lab.get(n, f"component_{n}"),
            "table": {k: _clean(r.get(k)) for k in TABLE_COLS},
            "picks": {k: _clean(v) for k, v in (blk.picks if blk else {}).items()},
            "window": {k: _clean(v) for k, v in (blk.window if blk else {}).items()},
            "triplet": {k: {a: _clean(b) for a, b in d.items()}
                        for k, d in (blk.triplet if blk else {}).items()},
            "chi2": _clean(blk.chi2) if blk else None})
    h = fit.header
    data = {"run": run.key, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eiakit": __version__, "source": source, "note": note,
            "constants": {"M_ION_kg": h.M_ION_kg, "e_C": e_C, "L_EIA_m": h.L_EIA_m,
                          "signal": h.signal or "per_ion"},
            "components": comps}
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return p


def read_store(fits_root, run) -> CBSFitFile | None:
    p = store_path(fits_root, run)
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    _, lab2num = _labels(run)
    nxt = max(lab2num.values(), default=0) + 1
    k = d["constants"]
    h = StoreHeader(generated=d.get("saved"), L_EIA_m=k.get("L_EIA_m"),
                    M_ION_kg=k.get("M_ION_kg"), signal=k.get("signal"), e_C=k.get("e_C"))
    nan = np.nan

    def un(v):
        return nan if v is None else v
    comps, rows = [], []
    for c in d["components"]:
        lab = c["label"]
        if lab not in lab2num:
            lab2num[lab], nxt = nxt, nxt + 1
        t = c["table"]
        comps.append(CBSComponent(
            z=int(c["z"]), cbs=lab2num[lab],
            picks={kk: un(v) for kk, v in c.get("picks", {}).items()},
            window={kk: un(v) for kk, v in c.get("window", {}).items()},
            triplet={kk: {a: un(b) for a, b in tt.items()}
                     for kk, tt in c.get("triplet", {}).items()},
            kT=un(t["kT_eV"]), V0=un(t["V0_V"]), A=un(t["A"]), chi2=un(c.get("chi2"))))
        rows.append({"z": int(c["z"]), "CBS": lab2num[lab],
                     **{kk: (v if isinstance(v, str) else un(v)) for kk, v in t.items()}})
    tab = pd.DataFrame(rows, columns=["z", "CBS", *TABLE_COLS])
    return CBSFitFile(path=p, header=h, z_totals={}, components=comps, table=tab)


def legacy_fit_path(run, data_root):
    from .audit import _find_cbs_file
    res = run.folder(data_root) / "EIA-Results"
    return _find_cbs_file(res, "CBS_fit_results", run.fit_file_labels, run.target)


def default_fits_root():
    """paths.fits when the paths are configured, else None (legacy files only)."""
    from .paths import PathsNotConfigured, get_paths
    try:
        return get_paths().fits
    except PathsNotConfigured:
        return None


def load_fit(db, run, data_root, fits_root=None) -> CBSFitFile | None:
    """The CBS fit of a run: the store if it exists (fits_root given), else the legacy file."""
    if fits_root is not None:
        st = read_store(fits_root, run)
        if st is not None:
            return st
    p = legacy_fit_path(run, data_root)
    return parse_cbs_fit_results(p) if p else None


def import_legacy(db, run, data_root, fits_root) -> Path:
    """Start the store from the legacy CBS_fit_results file (values and constants as is)."""
    p = legacy_fit_path(run, data_root)
    if p is None:
        raise FileNotFoundError(f"{run.key}: no CBS_fit_results file")
    fit = parse_cbs_fit_results(p)
    e_old = db.e_C(run.legacy.profile_cbs)
    if fit.header.M_ION_kg is None:
        fit.header.M_ION_kg = db.mass_kg(run.target, run.legacy.profile_cbs)
    return write_store(fits_root, run, fit, e_old, source=f"imported from {p.name}")
