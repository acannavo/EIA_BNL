"""
Step 3: audit of the legacy results of one run.

    report = audit_run(db, run, data_root)      # Report with findings + tidy tables
    write_report(report, out_dir)               # audit.txt, audit.json, CSV tables

Every check is a Finding(level, code, title, details). Codes refer to the plan:
I1-I8 (math audit sheet), N1-N15 (found while porting). Levels:
    error  the legacy result is wrong or inconsistent at this point
    warn   needs a look by a human (may be fine)
    info   documented fact, no action needed
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import physics as ph
from .layout import scan_run
from .legacy import (
    load_checkpoint,
    parse_cbs_areas,
    parse_cbs_fit_results,
    read_output_detail_folder,
    read_output_folder,
)

_trapz = getattr(np, "trapezoid", None) or np.trapz
LEVELS = ("error", "warn", "info")
_ORDER = {"ultra_fast": 0, "fast": 1, "slow": 2, "component_4": 3}


@dataclass
class Finding:
    level: str
    code: str
    title: str
    details: list = field(default_factory=list)


@dataclass
class Report:
    run_key: str
    label: str
    files: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def add(self, level, code, title, details=None):
        self.findings.append(Finding(level, code, title, list(details or [])))

    def count(self, level):
        return sum(f.level == level for f in self.findings)


# ================================================================ helpers

def _find_cbs_file(results_dir: Path, prefix: str, labels: list[int], target: str):
    for lab in labels:
        p = results_dir / f"{prefix}_{lab}_{target}.txt"
        if p.is_file():
            return p
    return None


def _match_profile(db, species: str, M_header: float | None):
    """Name of the constants profile whose ion mass matches the file header (6 digits)."""
    if M_header is None:
        return None
    for name in db.constants.profiles:
        if abs(db.mass_kg(species, name) / M_header - 1) < 2e-6:
            return name
    return None


# ================================================================ Output (untitled28)

def _audit_output(rep: Report, out: pd.DataFrame, detail: pd.DataFrame | None, run):
    if out.empty:
        rep.add("warn", "OUT", "2_SEM/Output is empty or missing")
        return
    rep.facts["output"] = {
        "files": int(out.file.nunique()), "rows": len(out),
        "U_min": float(out.U.min()), "U_max": float(out.U.max()),
        "masses": sorted(int(m) for m in out.m.unique()),
        "z_max_per_mass": {int(m): int(g.z.max()) for m, g in out.groupby("m")},
    }
    # N1: the same (m, z) written more than once in one file (append on every save)
    dup = out[out.duplicated(["file", "m", "z"], keep=False)]
    if len(dup):
        det = [f"{f}: (m, z) = {sorted(set(zip(g.m, g.z, strict=True)))}"
               for f, g in dup.groupby("file")]
        rep.add("error", "N1", f"{dup.groupby('file').ngroups} Output file(s) contain the same "
                f"(m, z) more than once ({len(dup)} rows); all copies enter the interpolation",
                det)
    # N14: identical rows in two different files (fit of one U saved under another U)
    key = ["m", "z", "TOF", "Yield"]
    cp = out[out.duplicated(key, keep=False)]
    cp = cp.groupby(key).filter(lambda g: g.file.nunique() > 1)
    if len(cp):
        det = []
        for _, g in cp.groupby(key):
            det.append(f"(m={g.m.iloc[0]}, z={g.z.iloc[0]}) TOF={g.TOF.iloc[0]:.6e} s in "
                       + ", ".join(sorted(g.file.unique())))
        pairs = sorted({tuple(sorted(g.file.unique())) for _, g in cp.groupby(key)})
        rep.add("error", "N14", f"identical Gaussian results stored under different deflector "
                f"voltages {pairs}: 'Save Fitted Points' saved the fit held in memory "
                f"(previous U) -> duplicated points in the spectra", det)
    # nearly equal voltages (p200 and p200.01): two files for one U
    Us = np.sort(out.U.unique())
    close = [(a, b) for a, b in zip(Us[:-1], Us[1:], strict=True) if (b - a) / b < 1e-3]
    if close:
        rep.add("warn", "OUT-U", "deflector voltages closer than 0.1 % (one U split over two "
                "files?)", [f"p{a:g}.txt and p{b:g}.txt" for a, b in close])
    # N8: two (m, z) Gaussians on the same peak in one spectrum
    locks = []
    for f, g in out.groupby("file"):
        g = g.sort_values("TOF")
        t = g.TOF.to_numpy()
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if (a.m, a.z) != (b.m, b.z) and abs(t[i + 1] - t[i]) / t[i] < 1e-3:
                locks.append(f"{f}: (m={a.m}, z={a.z}) and (m={b.m}, z={b.z}) at "
                             f"{t[i] * 1e6:.4f} / {t[i + 1] * 1e6:.4f} us")
    if locks:
        rep.add("error", "N8", "different (m, z) fitted on the same peak (TOF within 0.1 %): "
                "the same current is counted twice in the FC calibration", locks)
    # Output_detail: repeats per U
    if detail is not None and not detail.empty:
        ycols = [c for c in detail.columns if c.startswith("Yield") and c[5:].isdigit()]
        n_rep = detail[ycols].notna().sum(axis=1)
        per_U = detail.assign(n=n_rep).groupby("U").n.max()
        rep.facts["output_detail"] = {"files": int(detail.file.nunique()),
                                      "repeats_per_U": {f"{u:g}": int(n) for u, n in per_U.items()}}
        missing = sorted(set(out.file) - set(detail.file))
        if missing:
            rep.add("info", "OUT-DETAIL", "Output files without Output_detail (saved with "
                    "'Save Picked Points', no Gaussian fit)", missing)


# ================================================================ checkpoint (untitled27)

def _audit_checkpoint(rep: Report, ck: dict, out: pd.DataFrame, run, db):
    eia, fc = run.eia, run.fc
    spectra = ck.get("corrected_spectra", {})
    keys = sorted(spectra)
    rep.facts["checkpoint"] = {
        "keys": sorted(ck), "pairs": len(keys),
        "masses": sorted({m for m, _ in keys}),
        "z": sorted({z for _, z in keys}),
        "tof_us": [float(ck["corrected_tof"].min() * 1e6), float(ck["corrected_tof"].max() * 1e6)],
        "n_grid": int(len(ck["corrected_tof"])),
    }
    # geometry and unit factors against the database
    checks = [("L_EIA", ck.get("L_EIA"), eia.L_m), ("L_FC", ck.get("L_FC"), fc.L_m),
              ("CFArea", ck.get("CFArea"), 1 / fc.aperture_area_cm2),
              ("CFCurrent", ck.get("CFCurrent"), 1 / fc.R_load_ohm)]
    bad = [f"{n}: checkpoint {v!r} vs database {ref:.6g}" for n, v, ref in checks
           if v is None or abs(v / ref - 1) > 1e-9]
    if bad:
        rep.add("error", "CK-GEOM", "checkpoint geometry / unit factors differ from the database",
                bad)
    else:
        rep.add("info", "CK-GEOM", f"checkpoint made with L_EIA = {eia.L_m} m, L_FC = {fc.L_m} m, "
                f"r = {fc.aperture_radius_cm} cm, R = {fc.R_load_ohm} ohm (as in the database)")
    # internal consistency of the projection chain
    L = ck.get("L_EIA", eia.L_m)
    r = ck["CFY_EIA"] / ck["CFY_1m"]
    ok_ratio = abs(r / (L / ck.get("L_FC", fc.L_m)) ** 3 - 1) < 1e-9
    i1 = np.array([ck["integrals"][k] for k in keys])
    ie = np.array([ck["integrals_eia"][k] for k in keys])
    ok_int = np.allclose(i1, L ** 2 * ie, rtol=1e-9)
    s1 = sum(ck["corrected_spectra_1m"].values())
    se = sum(spectra.values())
    ok_sp = np.allclose(s1, se * L ** 3, rtol=1e-12, atol=0)
    if ok_ratio and ok_int and ok_sp:
        rep.add("info", "CK-CHAIN", "projection chain consistent: CFY_EIA/CFY_1m = (L_EIA/L_FC)^3, "
                "integrals = L_EIA^2 integrals_eia, spectra_1m = spectra L_EIA^3")
    else:
        rep.add("error", "CK-CHAIN", "projection chain inconsistent",
                [f"CFY ratio ok: {ok_ratio}", f"integrals ok: {ok_int}", f"spectra ok: {ok_sp}"])
    # FC peak used for the calibration (recovered): max of the summed EIA at 1 m
    fc_peak_V = float(s1.max() / (ck["CFArea"] * ck["CFCurrent"]) / ck["L_FC"] ** 3)
    rep.facts["checkpoint"]["calibration"] = {
        "CFY_1m": ck["CFY_1m"], "CFY_EIA": ck["CFY_EIA"],
        "FC_peak_after_baseline_V": fc_peak_V,
        "total_ions_cm2_at_L_EIA": float(sum(ck["integrals_eia"][k] / (k[1] * 1.602e-19)
                                             for k in keys)),
        "note": "FC baseline not stored (N7); recover it from FC_Ave.txt and the raw FC traces",
    }
    # duplicated raw points (N1 / N14 seen from the checkpoint)
    et = ck.get("corrected_exp_tof", {})
    dup = {k: int(len(v) - len(np.unique(v))) for k, v in et.items() if len(v) != len(np.unique(v))}
    if dup:
        rep.add("error", "CK-DUP", f"{len(dup)} (m, z) pair(s) have repeated raw TOF points "
                "(interpolation through duplicated points)",
                [f"(m={m}, z={z}): {n} repeated" for (m, z), n in sorted(dup.items())])
    # checkpoint built from this Output folder?
    if not out.empty and et:
        cnt = out.groupby(["m", "z"]).size()
        mism = [f"(m={m}, z={z}): checkpoint {len(et[(m, z)])} points, "
                f"Output {int(cnt.get((m, z), 0))}"
                for (m, z) in sorted(et) if len(et[(m, z)]) != int(cnt.get((m, z), 0))]
        extra = [f"(m={m}, z={z})" for (m, z) in cnt.index if (m, z) not in et]
        if mism or extra:
            rep.add("warn", "CK-OUT", "checkpoint raw points do not match the Output rows "
                    "(checkpoint made from another Output version?)", mism + extra)
        else:
            rep.add("info", "CK-OUT", "checkpoint raw points match the Output rows one to one "
                    f"({sum(len(v) for v in et.values())} points)")
    # I4: zero fill outside each (m, z) measured range
    tof = ck["corrected_tof"]
    span = tof[-1] - tof[0]
    zero = {k: 1 - (et[k].max() - et[k].min()) / span for k in et if len(et[k]) > 1}
    rep.facts["checkpoint"]["zero_filled_fraction"] = {f"{m},{z}": round(v, 3)
                                                       for (m, z), v in sorted(zero.items())}
    rep.add("info", "I4", "outside its measured TOF range each (m, z) spectrum is set to 0 "
            "(interp1d fill_value=0); fraction of the grid zero-filled, median "
            f"{np.median(list(zero.values())):.2f}")
    # target masses present
    labels = set(run.mass_labels)
    if not labels & {m for m, _ in keys}:
        rep.add("error", "CK-MASS", f"no spectra with the target mass label(s) {sorted(labels)}; "
                f"masses in the checkpoint: {sorted({m for m, _ in keys})}")


# ================================================================ CBS files (fitter)

def _audit_cbs(rep: Report, fit, areas, ck, run, db):
    h = fit.header
    prof = _match_profile(db, run.target, h.M_ION_kg)
    rep.facts["cbs"] = {"generated": h.generated, "L_EIA": h.L_EIA_m, "M_ION": h.M_ION_kg,
                        "signal": h.signal, "mass_profile": prof,
                        "components": len(fit.components),
                        "z": sorted(fit.z_totals)}
    if prof is None:
        rep.add("error", "I1", f"M_ION = {h.M_ION_kg} kg in the fit file matches no constants "
                f"profile for {run.target} -> fitted with another species' mass: refit needed")
    else:
        level = "info" if prof == run.legacy.profile_cbs else "warn"
        rep.add(level, "I1", f"fit file made with the '{prof}' mass "
                f"({db.mass_amu(run.target, prof)} amu x {db.profile(prof).amu_kg} kg); "
                f"corrected/legacy mass ratio = "
                f"{db.mass_kg(run.target) / db.mass_kg(run.target, prof):.5f} (scales kT and V0)")
    if h.L_EIA_m is not None and abs(h.L_EIA_m - run.eia.L_m) > 1e-9:
        rep.add("error", "CBS-L", f"fit file L_EIA = {h.L_EIA_m} m, database {run.eia.L_m} m")
    if h.signal is None:
        rep.add("warn", "CBS-SIG", "no '# Signal:' header: charge-current fit (pre-v12) assumed")

    # numbering: the fitter numbers the active components 1..n in every z
    edited = []
    for z in sorted(fit.z_totals):
        nums = sorted(c.cbs for c in fit.components if c.z == z)
        if nums != list(range(1, len(nums) + 1)):
            edited.append(f"z={z}: CBS {nums}")
    if edited:
        rep.add("warn", "N3", "CBS numbers not 1..n in some z: the file was renumbered by hand "
                "(the fitter always writes 1..n)", edited)

    # physical labels and ordering in time
    comp_map = run.legacy.cbs_components
    tab = fit.table.copy()
    tab["component"] = tab.CBS.map(comp_map)
    bad_order = []
    for z, g in tab.groupby("z"):
        g = g.dropna(subset=["component"]).sort_values("t_peak_fit")
        ranks = [_ORDER[c] for c in g.component]
        if ranks != sorted(ranks):
            bad_order.append(f"z={z}: " + ", ".join(
                f"CBS {r.CBS} ({r.component}) t={r.t_peak_fit:.3f} us" for r in g.itertuples()))
    if bad_order:
        rep.add("error", "N3", "component labels of runs.toml contradict the arrival times "
                "(expected ultra_fast < fast < slow) -> wrong fast/slow in the figures", bad_order)
    else:
        rep.add("info", "N3", f"legacy CBS numbering -> {comp_map} agrees with the arrival times "
                "in every z")

    # Picks t_peak (text block, after 'u=' / 't=' overrides) vs clicked peak (table)
    over = []
    for c in fit.components:
        r = tab[(tab.z == c.z) & (tab.CBS == c.cbs)]
        if len(r) and "t_peak" in c.picks:
            d = c.picks["t_peak"] - float(r.t_peak_pick.iloc[0])
            if abs(d) > 1e-3:
                over.append(f"z={c.z} CBS {c.cbs}: Picks t_peak = {c.picks['t_peak']:.4f} us, "
                            f"clicked {float(r.t_peak_pick.iloc[0]):.4f} us")
    if over:
        rep.add("warn", "N15", "peak time changed after the click (t= / u= typed in the text "
                "box): the text block and the table disagree; the fit used the typed value", over)

    # triplet duplicates
    trip = [f"z={c.z} CBS {c.cbs}: " + ("prev" if c.triplet["prev"]["t"] == c.triplet["PEAK"]["t"]
                                         else "next") + " = PEAK"
            for c in fit.components if {"prev", "PEAK", "next"} <= set(c.triplet)
            and (c.triplet["prev"]["t"] == c.triplet["PEAK"]["t"]
                 or c.triplet["next"]["t"] == c.triplet["PEAK"]["t"])]
    if trip:
        rep.add("warn", "CBS-TRIP", "velocity error bar from a repeated raw point (prev or next "
                "equals PEAK): the bar is one-sided (consequence of N1 / N14)", trip)

    v0 = [f"z={r.z} CBS {r.CBS} ({r.component})" for r in tab.itertuples() if r.V0_V < 0.01]
    if v0:
        rep.add("info", "CBS-V0", "V0 ~ 0 (purely thermal component, peak constraint at its "
                "limit)", v0)

    # reproduce the fitted model on the checkpoint grid
    if ck is not None and prof is not None:
        tof = ck["corrected_tof"]
        tus = tof * 1e6
        m, e, L = db.mass_kg(run.target, prof), db.e_C(prof), h.L_EIA_m or run.eia.L_m
        dt = float(np.median(np.diff(tus)))
        area_tab = areas.table.set_index(["z", "CBS"]) if areas is not None else None
        worst = {"t_peak": 0.0, "FWHM": 0.0, "Area": 0.0}
        rows_bad = []
        for r in tab.itertuples():
            y = r.A * ph.cbs_shape(tof, r.kT_eV, r.V0_V, r.z, m, L, e)
            tpk = tus[int(np.argmax(y))]
            tL, tR = ph.half_maximum_edges(tus, y)
            d_t = abs(tpk - r.t_peak_fit)
            d_f = abs((tR - tL) / r.FWHM_fit - 1)
            d_a = 0.0
            if area_tab is not None and (r.z, r.CBS) in area_tab.index:
                a_ref = float(area_tab.loc[(r.z, r.CBS), "Area_model_full"])
                d_a = abs(float(_trapz(y, tus)) / a_ref - 1)
            worst = {"t_peak": max(worst["t_peak"], d_t), "FWHM": max(worst["FWHM"], d_f),
                     "Area": max(worst["Area"], d_a)}
            if d_t > 3 * dt or d_f > 2e-3 or d_a > 2e-3:
                rows_bad.append(f"z={r.z} CBS {r.CBS}: t_peak {tpk:.4f} vs {r.t_peak_fit:.4f} us, "
                                f"FWHM rel {d_f:.1e}, area rel {d_a:.1e}")
        rep.facts["cbs"]["reproduction"] = {"max_dt_peak_us": worst["t_peak"],
                                            "grid_step_us": dt, "max_rel_FWHM": worst["FWHM"],
                                            "max_rel_area": worst["Area"]}
        if rows_bad:
            rep.add("error", "CBS-REPRO", "fitted kT, V0, A do not reproduce the stored peak / "
                    "FWHM / area on the checkpoint grid", rows_bad)
        else:
            rep.add("info", "CBS-REPRO", f"kT, V0, A of every component reproduce t_peak "
                    f"(max {worst['t_peak']:.1e} us, grid {dt:.1e} us), FWHM (rel "
                    f"{worst['FWHM']:.0e}) and area (rel {worst['Area']:.0e}) with the "
                    f"'{prof}' constants")

    # areas file
    if areas is not None:
        at, ab = areas.table, areas.blocks
        if set(zip(at.z, at.CBS, strict=True)) != set(zip(tab.z, tab.CBS, strict=True)):
            rep.add("error", "AREAS", "areas file and fit file list different (z, CBS) components")
        typos = []
        if not ab.empty:
            mrg = at.merge(ab, on=["z", "CBS"], suffixes=("", "_blk"))
            for r in mrg.itertuples():
                if abs(r.t_peak_us - r.t_peak_us_blk) > 1e-3:
                    typos.append(f"z={r.z} CBS {r.CBS}: table t_peak {r.t_peak_us} vs block "
                                 f"{r.t_peak_us_blk}")
            nocolon = ab[~ab.colon]
            if len(nocolon):
                rep.add("warn", "AREAS-EDIT", f"{len(nocolon)} 'CBS n' block header(s) without "
                        "colon: the areas file was edited by hand",
                        [f"z={r.z} CBS {r.CBS}" for r in nocolon.itertuples()])
        if typos:
            rep.add("error", "AREAS-TYPO", "summary table differs from the readable blocks "
                    "(hand-edit typo?) -- the plot scripts read the table", typos)
        if "Ions_model_full_cm2" in at:
            ratio = (at.Ions_model_full_cm2 / (at.Area_model_full * 1e-6)).median()
            rep.facts["cbs"]["e_used_for_ions"] = float(1 / ratio)
            rep.add("info", "CBS-IONS", f"ions cm^-2 = area x 1e-6 / e with e = {1 / ratio:.4e} C "
                    f"({'per-ion signal' if h.per_ion else 'charge signal, /z'})")

    keep = ["z", "CBS", "component", "t_peak_pick", "t_peak_fit", "FWHM_pick", "FWHM_fit",
            "win_left", "win_right", "win_src", "kT_eV", "V0_V", "A"]
    t = tab[[c for c in keep if c in tab]].copy()
    trip = {(c.z, c.cbs): c.triplet for c in fit.components}
    for k in ("prev", "PEAK", "next"):
        t[f"t_exp_{k}_us"] = [trip.get((r.z, r.CBS), {}).get(k, {}).get("t", np.nan)
                              for r in t.itertuples()]
    rep.tables["cbs_components"] = t


# ================================================================ main entry

def audit_run(db, run, data_root: Path, study=None) -> Report:
    folder = run.folder(data_root)
    rep = Report(run.key, run.label)
    if not folder.is_dir():
        rep.add("error", "FOLDER", f"run folder missing: {folder}")
        return rep
    rf = scan_run(folder)
    rep.facts["folders"] = {"canonical": sorted(rf.canonical), "variants_ignored":
                            sorted(str(p.relative_to(folder)) for p in rf.variants),
                            "extra": sorted(str(p.relative_to(folder)) for p in rf.extras)}
    sem = folder / "2_SEM"
    res = folder / "EIA-Results"

    out = read_output_folder(sem / "Output") if (sem / "Output").is_dir() else pd.DataFrame()
    det = read_output_detail_folder(sem / "Output_detail") if (sem / "Output_detail").is_dir() \
        else None
    rep.files["output"] = str(sem / "Output")
    _audit_output(rep, out, det, run)
    if not out.empty:
        rep.tables["output_rows"] = out[["file", "U", "row", "m", "z", "TOF", "Yield"]]

    ck = None
    ck_path = res / "pipeline_checkpoint.npy"
    if ck_path.is_file():
        rep.files["checkpoint"] = str(ck_path)
        ck = load_checkpoint(ck_path)
        _audit_checkpoint(rep, ck, out, run, db)
    else:
        rep.add("error", "CK", f"no checkpoint: {ck_path}")

    fit_p = _find_cbs_file(res, "CBS_fit_results", run.fit_file_labels, run.target)
    area_p = _find_cbs_file(res, "CBS_areas", run.fit_file_labels, run.target)
    if fit_p is None:
        cands = sorted(p.name for p in res.glob("CBS_fit_results*.txt")) if res.is_dir() else []
        rep.add("error", "CBS", "no CBS_fit_results file with labels "
                f"{run.fit_file_labels}", [f"found: {c}" for c in cands])
    else:
        rep.files["cbs_fit"] = str(fit_p)
        rep.files["cbs_areas"] = str(area_p) if area_p else None
        fit = parse_cbs_fit_results(fit_p)
        areas = parse_cbs_areas(area_p) if area_p else None
        _audit_cbs(rep, fit, areas, ck, run, db)

    if study is not None:
        ex = study.exclude.get(run.key, {})
        if ex:
            rep.add("info", "EXCL", f"study {study.name} excludes {ex}")
    return rep


def format_report(rep: Report) -> str:
    lines = [f"Legacy audit -- {rep.run_key} ({rep.label})",
             f"errors {rep.count('error')}   warnings {rep.count('warn')}   "
             f"info {rep.count('info')}", ""]
    for k, v in rep.files.items():
        lines.append(f"  {k:<11} {v}")
    lines.append("")
    for level in LEVELS:
        for f in (x for x in rep.findings if x.level == level):
            lines.append(f"[{level.upper():5}] {f.code:<10} {f.title}")
            for d in f.details[:40]:
                lines.append(f"            - {d}")
            if len(f.details) > 40:
                lines.append(f"            ... {len(f.details) - 40} more")
    lines += ["", "Facts:", json.dumps(rep.facts, indent=2, default=str)]
    return "\n".join(lines)


def write_report(rep: Report, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / "audit.txt"
    p.write_text(format_report(rep), encoding="utf-8")
    paths.append(p)
    p = out_dir / "audit.json"
    p.write_text(json.dumps({"run": rep.run_key, "label": rep.label, "files": rep.files,
                             "facts": rep.facts,
                             "findings": [asdict(f) for f in rep.findings]},
                            indent=2, default=str), encoding="utf-8")
    paths.append(p)
    for name, df in rep.tables.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
