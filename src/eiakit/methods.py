"""
Step 12: method studies. Each study varies one analysis choice and shows what the paper
numbers do; the choices are switches of db/pipeline.toml (defaults = legacy), so a study
result can be adopted by flipping the switch (and measured with eiakit compare).

    zbar         I4 zero fill outside the measured ranges (bracket with the edge
                 continuation, mask), I5 noise floor scan, clipping of negative J
    calibration  I2 peak vs area matching of the summed EIA to the Faraday cup, and the
                 EIA / FC overlay over the whole trace (needs FC traces)
    sensitivity  flight length (LOF, +/- 1 %) and time zero (+/- 50 ns): zbar fits and z*
    fc           FC start / stop / both: CFY and the shot-to-shot FC peaks (drift)
    shots        N12: shot-to-shot scatter of the untitled28 fits (Yield1..3 of
                 Output_detail): block bootstrap over shots of the zbar fit

eiakit methods NAME [--only P ...] [--boot N]  ->  studies/NAME/methods/
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .figures.spec import ErrorBars, Figure, Line, Points, lab
from .figures.style import ELEMENT_DASH, energy_color
from .pipeline import git_commit

PARTS = ["zbar", "calibration", "sensitivity", "fc", "shots"]

ZBAR_VARIANTS = [                              # (name, [analysis] overrides)
    ("legacy", {}),
    ("fill = extrapolated (I4)", {"zbar_fill": "extrapolate"}),
    ("fill = edge (I4 bound)", {"zbar_fill": "edge"}),
    ("mask, 5 % (I4)", {"zbar_mask": "measured"}),
    ("no clip (I5)", {"zbar_clip_negative": False}),
    ("floor 1e-3", {"zbar_min_frac": 1e-3}),
    ("floor 3e-3", {"zbar_min_frac": 3e-3}),
    ("floor 1e-2", {"zbar_min_frac": 1e-2}),
    ("floor 3e-2", {"zbar_min_frac": 3e-2}),
]
FLOORS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2]


@contextlib.contextmanager
def _pipe(db, section: str, **kv):
    """Temporarily change db.pipeline[section] (restored afterwards)."""
    old = copy.deepcopy(db.pipeline[section])
    db.pipeline[section].update(kv)
    try:
        yield db
    finally:
        db.pipeline[section] = old


def _datasets(db, st, data_root, processed, source):
    from .analysis import load_dataset
    from .study import _load_ck
    out = []
    for key in st.all_runs():
        run = db.run(key)
        ck, _ = _load_ck(db, run, source, data_root, processed)
        out.append((run, ck, load_dataset(db, run, ck, st)))
    return out


def _style(ds):
    return energy_color(int(round(ds.energy_mJ or 0))), ELEMENT_DASH.get(ds.species, "solid")


# ------------------------------------------------------------------------- zbar (I4, I5)

def study_zbar(db, st, dsets) -> dict:
    from .analysis import analyze_zbar, compute_zbar, fit_power_law, zbar_options
    base = dict(db.pipeline.get("analysis", {}))
    rows, curves = [], {}
    for _, _, ds in dsets:
        ref = analyze_zbar(ds, st.zbar, base)["fit"]          # the window of the current analysis
        win = (ref["E_lo"], ref["E_hi"]) if ref else None
        for name, ov in ZBAR_VARIANTS:
            zb = compute_zbar(ds, **zbar_options({**base, **ov}))
            a = {"zb": zb, "fit": fit_power_law(zb["E_eV"], zb["zbar"], zb["valid"], win)}
            f = a["fit"] or {}
            rows.append({"dataset": ds.name, "variant": name, **{k: f.get(k, np.nan) for k in
                         ("C", "b", "sig_b", "n", "r2", "E_lo", "E_hi", "npts")}})
            if name in ("legacy", "fill = extrapolated (I4)", "floor 1e-2"):
                curves[(ds.name, name)] = a
    tab = pd.DataFrame(rows)
    leg = tab[tab.variant == "legacy"].set_index("dataset")
    tab["d_b"] = tab.b - tab.dataset.map(leg.b)
    tab["C_rel"] = tab.C / tab.dataset.map(leg.C) - 1

    scan = Figure(rows=1, cols=2, size_in=(12, 4.6), legend="figure", letters=True,
                  title=lab(r"$\bar z = C E^b$: noise floor (I5)",
                            "z̄ = C E^b: noise floor (I5)"))
    pb, pc = scan.panel(0, 0), scan.panel(0, 1)
    x = np.array([3e-4] + FLOORS[1:])                   # floor 0 drawn at 3e-4
    for p, col, yl in ((pb, "b", lab("exponent $b$", "exponent b")),
                       (pc, "C", lab("prefactor $C$", "prefactor C"))):
        p.xlog = True
        p.xlabel = lab(r"floor: $N_{tot} > f\,\max N_{tot}$  ($f = 0$ drawn at $3\times10^{-4}$)",
                       "floor f: N_tot > f max N_tot (f = 0 drawn at 3e-4)")
        p.ylabel = yl
        for _, _, ds in dsets:
            sub = tab[(tab.dataset == ds.name)]
            y = [sub[sub.variant == ("legacy" if f == 0 else f"floor {f:.0e}".replace("e-0", "e-"))]
                 [col].iloc[0] for f in FLOORS]
            c, d = _style(ds)
            p.add(Line(x, np.array(y, float), ds.name if p is pb else None, c, dash=d,
                       group=ds.name, width=1.6),
                  Points(x, np.array(y, float), None, c, group=ds.name))

    grid = Figure(rows=2, cols=3, size_in=(16, 9), sharex=True, sharey=True, letters=True,
                  legend="panel", title=lab(r"$\bar z(E)$: zero fill vs extrapolation (I4), "
                                            r"noise floor $10^{-2}$ (I5)",
                                            "z̄(E): zero fill vs extrapolation (I4), noise floor "
                                            "1e-2 (I5)"))
    look = {(ds.species, int(round(ds.energy_mJ or 0))): ds for _, _, ds in dsets}
    for i, el in enumerate(("Al", "Nb")):
        for j, mj in enumerate((180, 370, 780)):
            p = grid.panel(i, j)
            p.xlog = p.ylog = True
            p.xlabel = lab(r"$E$  [eV]", "E [eV]")
            p.ylabel = lab(r"$\bar z$", "z̄")
            ds = look.get((el, mj))
            if ds is None:
                p.empty_text = f"{el}, {mj} mJ: not available"
                continue
            p.title = ds.title
            for name, color in (("legacy", "black"), ("fill = extrapolated (I4)", "#ff7f0e"),
                                ("floor 1e-2", "#2ca02c")):
                a = curves[(ds.name, name)]
                zb, f = a["zb"], a["fit"]
                v = zb["valid"]
                p.add(Line(zb["E_eV"][v], zb["zbar"][v], f"{name} (data)", color, width=0.7,
                           alpha=0.6, group=name))
                if f:
                    Ef = np.logspace(np.log10(f["E_lo"]), np.log10(f["E_hi"]), 60)
                    p.add(Line(Ef, f["C"] * Ef ** f["b"], f"{name}: b = {f['b']:.3f}", color,
                               width=2, dash="dash", group=name))
    return {"table": tab, "figures": {"zbar_floor_scan": scan, "zbar_fill": grid}}


# ------------------------------------------------------------------- calibration (I2)

def study_calibration(db, st, data_root) -> dict:
    from .calibration import run_calibration
    rows, overlays = [], []
    fcs = db.pipeline["calibration"].get("fc_source", "start")
    base = db.pipeline["calibration"].get("fc_baseline", "legacy")
    for key in st.all_runs():
        run = db.run(key)
        res = {}
        for m in ("peak", "area"):
            with _pipe(db, "calibration", method=m, fc_missing="error"):
                res[m] = run_calibration(db, run, data_root, fcs, base)
        if "error" in res["peak"]:
            rows.append({"run": key, "note": res["peak"]["error"]})
            continue
        rp, ra = res["peak"]["res"], res["area"]["res"]
        rows.append({"run": key, "CFY_peak": rp["CFY_1m"], "CFY_area": ra["CFY_1m"],
                     "area_over_peak": ra["CFY_1m"] / rp["CFY_1m"],
                     "fc_peak_V": rp["fc_peak_V"], "note": ""})
        L_FC = rp["L_FC"]
        k = rp["CFArea"] * rp["CFCurrent"]
        fc_t, fc_v = rp["fc_average"]
        eia = sum(rp["corrected_spectra_1m"].values()) / k            # = CFY sum_EIA
        overlays.append((key, run, fc_t / L_FC * 1e6, fc_v * L_FC ** 3,
                         rp["corrected_tof_1m"] * 1e6, eia,
                         eia * ra["CFY_1m"] / rp["CFY_1m"]))
    tab = pd.DataFrame(rows)
    figs = {}
    if overlays:
        n = len(overlays)
        f = Figure(rows=n, cols=1, size_in=(10, 3.6 * n), legend="panel", letters=n > 1,
                   title=lab("EIA (summed) vs Faraday cup at 1 m (I2)",
                             "EIA (summed) vs Faraday cup at 1 m (I2)"))
        for i, (key, run, tf, jf, te, je, ja) in enumerate(overlays):
            p = f.panel(i, 0)
            p.title = f"{key}: {run.target}, {run.laser.energy_mJ:g} mJ"
            p.xlabel = lab(r"$t / L$  [$\mu$s/m]", "t / L [µs/m]")
            p.ylabel = lab(r"$V_{FC}(L\tau)\,L_{FC}^3$  [V m$^3$]", "V_FC(Lτ) L_FC³ [V m³]")
            p.add(Line(tf, jf, "Faraday cup", "black", width=1.0),
                  Line(te, je, "EIA, peak matched (legacy)", "#1f77b4", width=1.4),
                  Line(te, ja, "EIA, area matched (I2)", "#d62728", width=1.4, dash="dash"))
            p.xlog = True
            p.xlim = (max(float(np.nanmin(te)), 0.3), float(np.nanmax(tf)) * 1.5)
        figs["eia_fc_overlay"] = f
    return {"table": tab, "figures": figs}


# ---------------------------------------------------------- sensitivity (LOF, time zero)

PERTURB = [("L +1 %", 0.01, 0.0), ("L -1 %", -0.01, 0.0),
           ("t0 +50 ns", 0.0, 0.05), ("t0 -50 ns", 0.0, -0.05)]


def study_sensitivity(db, st, dsets, data_root) -> dict:
    from . import collapse as C
    from .analysis import analyze_zbar
    an = db.pipeline.get("analysis", {})
    rows = []
    for _, _, ds in dsets:
        b0 = analyze_zbar(ds, st.zbar, an)["fit"]
        for name, dl, dt in PERTURB:
            d2 = dataclasses.replace(ds, L_m=ds.L_m * (1 + dl), tof_us=ds.tof_us - dt)
            f = analyze_zbar(d2, st.zbar, an)["fit"]
            for q in ("C", "b"):
                rows.append({"quantity": f"zbar {q}", "dataset": ds.name, "perturbation": name,
                             "base": b0[q], "value": f[q], "rel": f[q] / b0[q] - 1})
    cs = st.collapse
    prof = {k: db.profile_for(db.run(k), "analysis") for k in st.runs}
    masses = {k: db.mass_kg(db.run(k).target, prof[k]) for k in st.runs}
    e_C = db.e_C(prof[st.runs[0]])
    pk = pd.concat([C.cbs_peaks(db, db.run(k), data_root, cs.cbs_t_source) for k in st.runs],
                   ignore_index=True)
    pk = C.drop_excluded(pk, st.exclude)

    def zstars(p):
        ser = C.collapse_series(p, cs.runs, masses, e_C, cs.min_fast_points)
        out = {s["run"]: s["zstar"] for s in ser}
        j = C.weighted_joint([s["zstar"] for s in ser], [s["se_zstar"] for s in ser])
        out["joint"] = j["zstar"]
        return out
    z0 = zstars(pk)
    for name, dl, dt in PERTURB:
        p2 = pk.copy()
        for c in ("t_before", "t_peak", "t_after"):
            p2[c] = p2[c] - dt
        p2["distance_m"] = p2.distance_m * (1 + dl)
        z1 = zstars(p2)
        for k, v in z0.items():
            rows.append({"quantity": "z* (CBS)", "dataset": k, "perturbation": name,
                         "base": v, "value": z1.get(k, np.nan),
                         "rel": z1.get(k, np.nan) / v - 1})
    tab = pd.DataFrame(rows)
    f = Figure(rows=1, cols=3, size_in=(16, 4.8), legend="figure", letters=True,
               title=lab("Sensitivity to the flight length and the time zero",
                         "Sensitivity to the flight length and the time zero"))
    marks = {"L +1 %": ("^", "#1f77b4"), "L -1 %": ("v", "#1f77b4"),
             "t0 +50 ns": ("^", "#d62728"), "t0 -50 ns": ("v", "#d62728")}
    for j, q in enumerate(("zbar C", "zbar b", "z* (CBS)")):
        p = f.panel(0, j)
        sub = tab[tab.quantity == q]
        names = list(dict.fromkeys(sub.dataset))
        p.title = q
        p.ylabel = lab("relative change  [%]", "relative change [%]")
        p.xlabel = lab("dataset (index)", "dataset (index)")
        for pert, (mk, col) in marks.items():
            s = sub[sub.perturbation == pert].set_index("dataset").reindex(names)
            p.add(Points(np.arange(len(names)), 100 * s.rel.to_numpy(float),
                         pert if j == 0 else None, col, marker=mk, size=7, group=pert,
                         hover=names))
        p.hlines.append((0.0, "gray", "dot"))
    f.notes.append("Datasets (x index): " + "; ".join(
        f"{q}: " + ", ".join(f"{i} {n}" for i, n in
                              enumerate(dict.fromkeys(tab[tab.quantity == q].dataset)))
        for q in ("zbar b", "z* (CBS)")))
    return {"table": tab, "figures": {"sensitivity": f}}


# ---------------------------------------------------------------- FC start / stop drift

def study_fc(db, st, data_root) -> dict:
    from .calibration import run_calibration
    rows, series = [], []
    base = db.pipeline["calibration"].get("fc_baseline", "legacy")
    for key in st.all_runs():
        run = db.run(key)
        cfy = {}
        for src in ("start", "stop", "both"):
            with _pipe(db, "calibration", fc_missing="error"):
                r = run_calibration(db, run, data_root, src, base)
            if "error" in r:
                continue
            i = r["info"]
            pk = np.array(i["fc_per_shot_peak_V"])
            cfy[src] = i["CFY_1m"]
            rows.append({"run": key, "source": src, "shots": len(pk), "CFY_1m": i["CFY_1m"],
                         "fc_peak_avg_V": i["fc_peak_V"], "shot_peak_mean_V": pk.mean(),
                         "shot_peak_relstd": pk.std() / pk.mean(),
                         "last_over_first": pk[-1] / pk[0]})
            if src in ("start", "stop"):
                series.append((key, src, pk))
        if not cfy:
            rows.append({"run": key, "source": "-", "note": "no FC traces here"})
    tab = pd.DataFrame(rows)
    if "CFY_1m" in tab:
        ref = tab[tab.source == "start"].set_index("run").CFY_1m
        tab["CFY_over_start"] = tab.CFY_1m / tab.run.map(ref)
    figs = {}
    if series:
        f = Figure(rows=1, cols=1, size_in=(9, 4.8), legend="figure",
                   title=lab("Faraday cup peak shot by shot", "Faraday cup peak shot by shot"))
        p = f.panel(0, 0)
        p.xlabel = lab("shot (start, then stop)", "shot (start, then stop)")
        p.ylabel = lab("FC peak / mean of the start shots", "FC peak / mean of the start shots")
        starts = {k: pk.mean() for k, s, pk in series if s == "start"}
        n_start = {k: len(pk) for k, s, pk in series if s == "start"}
        for key, src, pk in series:
            run = db.run(key)
            x0 = 0 if src == "start" else n_start.get(key, 0)
            m = starts.get(key, pk.mean())
            col = energy_color(int(run.laser.energy_mJ))
            p.add(Line(np.arange(len(pk)) + x0, pk / m,
                       f"{key} ({run.target}, {run.laser.energy_mJ:g} mJ)"
                       if src == "start" or key not in starts else None, col,
                       dash=ELEMENT_DASH.get(run.target, "solid"), group=key),
                  Points(np.arange(len(pk)) + x0, pk / m, None, col, group=key,
                         marker="o" if src == "start" else "s"))
        p.hlines.append((1.0, "gray", "dot"))
        figs["fc_shots"] = f
    return {"table": tab, "figures": figs}


# --------------------------------------------------------------------- shots (N12)

def _shot_table(db, run, data_root):
    from .calibration import prepare_rows
    from .legacy import read_output_folder
    prep = prepare_rows(db, run, data_root)
    if "error" in prep:
        return None
    rows = prep["rows"].reset_index(drop=True)
    det = read_output_folder(run.folder(data_root) / run.acquisition.sem.folder
                             / "Output_detail")
    if det.empty or not {"Yield1", "YieldAve"} <= set(det.columns):
        return rows, None
    shots = sorted(c for c in det.columns if c.startswith("Yield") and c[5:].isdigit())
    key = rows[["file", "m", "z"]].assign(TOF=rows.TOF.round(15))
    d = det[["file", "m", "z", "TOF", "YieldAve", *shots]].assign(TOF=det.TOF.round(15))
    merged = key.merge(d, on=["file", "m", "z", "TOF"], how="left")
    return rows, merged[["YieldAve", *shots]].to_numpy(float)


def study_shots(db, st, dsets, data_root, n_boot: int = 30, seed: int = 1) -> dict:
    from .analysis import analyze_zbar, load_dataset
    from .calibration import calibrate
    an = db.pipeline.get("analysis", {})
    pipe = db.pipeline["calibration"]
    rng = np.random.default_rng(seed)
    rows, boots = [], {}
    for run, _, ds in dsets:
        t = _shot_table(db, run, data_root)
        if t is None or t[1] is None:
            rows.append({"dataset": ds.name, "note": "no Output_detail shot columns"})
            continue
        r0, sh = t
        ave, ind = sh[:, 0], sh[:, 1:]
        nrow = np.isfinite(ind).sum(axis=1)                 # shots stored for each row
        files = r0.file.to_numpy()
        ufiles = np.unique(files)
        nfile = {f: int(nrow[files == f].max()) for f in ufiles}
        has = nrow >= 2                                     # rows with a shot-to-shot scatter
        with np.errstate(divide="ignore", invalid="ignore"):
            relsc = np.nanstd(ind, axis=1, ddof=1) / np.sqrt(nrow) / np.abs(ave)
        f0 = analyze_zbar(ds, st.zbar, an)["fit"]
        Cs, bs = [], []
        for _ in range(n_boot):
            # the same resampled shots for all z of a U (they come from the same shots)
            pick = {f: rng.integers(0, n, n) for f, n in nfile.items() if n >= 2}
            delta = np.zeros(len(r0))
            for i in np.flatnonzero(has):
                p = pick.get(files[i])
                if p is not None and np.isfinite(ind[i, p]).all():
                    delta[i] = ind[i, p].mean() - ave[i]
            rb = r0.assign(Yield=r0.Yield + np.nan_to_num(delta))
            ck = calibrate(rb, None, 0.0, run.eia, run.fc, pipe, fc_peak_override=1.0)
            f = analyze_zbar(load_dataset(db, run, ck, st), st.zbar, an)["fit"]
            if f:
                Cs.append(f["C"])
                bs.append(f["b"])
        boots[ds.name] = (np.array(Cs), np.array(bs))
        rows.append({"dataset": ds.name, "rows_with_shots": int(has.sum()), "rows": len(r0),
                     "U_with_1_shot": int(sum(n < 2 for n in nfile.values())),
                     "U_with_2plus": int(sum(n >= 2 for n in nfile.values())),
                     "median_shots": float(np.median(list(nfile.values()))),
                     "point_rel_scatter_median": float(np.nanmedian(relsc[has])),
                     "C": f0["C"], "C_boot_std_rel": float(np.std(Cs) / f0["C"]),
                     "b": f0["b"], "sig_b_fit": f0["sig_b"], "b_boot_std": float(np.std(bs)),
                     "boot_over_fit": float(np.std(bs) / f0["sig_b"]), "n_boot": len(bs),
                     "note": ""})
    tab = pd.DataFrame(rows)
    figs = {}
    ok = tab.dropna(subset=["b"]) if "b" in tab else tab.iloc[:0]
    if len(ok):
        f = Figure(rows=1, cols=1, size_in=(8, 4.8), legend="figure",
                   title=lab(r"$\bar z$ exponent: fit error vs shot-to-shot bootstrap (N12)",
                             "z̄ exponent: fit error vs shot-to-shot bootstrap (N12)"))
        p = f.panel(0, 0)
        x = np.arange(len(ok))
        p.xlabel = lab("dataset (index)", "dataset (index)")
        p.ylabel = lab("exponent $b$", "exponent b")
        b = ok.b.to_numpy(float)
        p.add(ErrorBars(x - 0.08, b, "least-squares error", "black",
                        ylo=b - ok.sig_b_fit.to_numpy(float),
                        yhi=b + ok.sig_b_fit.to_numpy(float),
                        hover=list(ok.dataset)),
              ErrorBars(x + 0.08, b, "bootstrap over shots", "#d62728",
                        ylo=b - ok.b_boot_std.to_numpy(float),
                        yhi=b + ok.b_boot_std.to_numpy(float), hover=list(ok.dataset)))
        f.notes.append("Datasets: " + ", ".join(f"{i} {n}" for i, n in enumerate(ok.dataset)))
        figs["shots_bootstrap"] = f
    return {"table": tab, "figures": figs}


# ------------------------------------------------------------------------------ driver

INTRO = {
    "zbar": "I4 / I5: z̄ fit with the legacy zero fill; with every z extrapolated beyond its "
            "measured range (log-log slope of its 3 outer points, decaying at least as t^±1); "
            "with every z continued flat at its edge value (an extreme upper bound of what the "
            "zero fill drops); with a mask where that bound exceeds 5 % of the ions; without "
            "clipping negative J; and with a noise floor "
            "N_tot > f max N_tot. All variants are fitted in the same window (the one of the "
            "current analysis), so they differ only by the choice. Switches: [analysis] "
            "zbar_fill, zbar_mask, zbar_clip_negative, zbar_min_frac.",
    "calibration": "I2: CFY from the peak (legacy) or from the area of the summed EIA "
                   "matched to the FC over the time range both cover (both scaled to 1 m). The "
                   "overlay shows the FC and the EIA over the whole trace. Switch: "
                   "[calibration] method.",
    "sensitivity": "Flight length +/- 1 % and time zero +/- 50 ns (t_true = t - t0): z̄ "
                   "fits and z* of the collapse (CBS peaks).",
    "fc": "Faraday cup start / stop / both: CFY and the peak of every shot (moving average "
          "fc_smooth_s), normalised to the mean of the start shots. Switch: [calibration] "
          "fc_source.",
    "shots": "N12: J(t) is assembled from different shots at every U. Bootstrap: at every "
             "U the shots are resampled (the same shots for all z of that U, keeping their "
             "correlation), the spectra rebuilt and z̄ refitted; compared with the "
             "least-squares error of b.",
}


def run_methods(db, study_name, data_root, processed, studies_root, only=None,
                source="pipeline", formats=("pdf", "png"), n_boot: int = 30, log=print) -> dict:
    from .figures.export import save, write_html_page
    st = db.study(study_name)
    out = Path(studies_root) / study_name / "methods"
    out.mkdir(parents=True, exist_ok=True)
    parts = only or PARTS
    dsets = _datasets(db, st, data_root, processed, source) \
        if {"zbar", "sensitivity", "shots"} & set(parts) else []
    results = {}
    for part in parts:
        t = time.time()
        if part == "zbar":
            r = study_zbar(db, st, dsets)
        elif part == "calibration":
            r = study_calibration(db, st, data_root)
        elif part == "sensitivity":
            r = study_sensitivity(db, st, dsets, data_root)
        elif part == "fc":
            r = study_fc(db, st, data_root)
        elif part == "shots":
            r = study_shots(db, st, dsets, data_root, n_boot)
        else:
            raise ValueError(f"unknown methods part {part!r}: {PARTS}")
        r["table"].to_csv(out / f"{part}.csv", index=False)
        for name, fig in r["figures"].items():
            if formats:
                save(fig, out / name, formats=formats)
        results[part] = r
        log(f"  methods / {part}: {len(r['table'])} rows, {len(r['figures'])} figures "
            f"in {time.time() - t:.0f} s")
    items = [("p", f"Study {study_name}, source {source}. Defaults of db/pipeline.toml = the "
                   "analysis as it is; each section varies one choice.")]
    for part, r in results.items():
        items.append(("h2", part))
        items.append(("p", INTRO[part]))
        items.append(("table", r["table"].round(6), f"{part}.csv"))
        items.extend(r["figures"].values())
    write_html_page(out / "methods.html", f"{study_name}: method studies (Step 12)", items,
                    f"eiakit {__version__}, git {git_commit()}, "
                    f"{time.strftime('%Y-%m-%d %H:%M')}")
    man = {"study": study_name, "parts": list(results), "source": source,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
           "analysis": db.pipeline.get("analysis", {}),
           "calibration": db.pipeline.get("calibration", {})}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return {"out": out, "results": results}
