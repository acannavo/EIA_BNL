"""
Study outputs (cross-run figures and tables), written to studies/<name>/<part>/.

Each part records in manifest.json the study settings it used and the run_id of every
run output it read (pipeline source), so `study_status` can tell when a figure is out of
date after a db change or a rerun.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from . import __version__
from .pipeline import STAGES, git_commit, read_manifest, sha256_file, sha256_json


def _settings_hash(db, st) -> str:
    return sha256_json({"study": st.model_dump(mode="json"), "corrections": db.corrections(),
                        "analysis": db.pipeline.get("analysis", {})})


def _load_ck(db, run, source, data_root, processed):
    """Checkpoint-like dict of a run + the run_id it came from."""
    if source == "legacy":
        from .legacy import load_checkpoint
        p = run.folder(data_root) / "EIA-Results" / "pipeline_checkpoint.npy"
        return load_checkpoint(p), f"legacy:{sha256_file(p)[:16]}"
    from .store import load_calibration_h5
    st = STAGES["calibration"]
    man = read_manifest(processed, run, st)
    if man is None or not man.get("ok"):
        raise RuntimeError(f"{run.key}: calibration stage not run (eiakit run {run.key!r})")
    return load_calibration_h5(Path(processed) / run.key / st.folder / "spectra.h5"), \
        man["run_id"]


def powerlaw(db, study_name, data_root, processed, studies_root, source="pipeline",
             formats=("pdf", "png"), log=print) -> dict:
    """zbar(E) power laws, Sheil-style spectra, per-z summaries (plot_eia_powerLaw_test)."""
    from .analysis import analyze_zbar, compute_sheil, load_dataset, reference_prefactor, summarize
    from .figures import powerlaw as pw
    from .figures.export import save, write_html_page

    st = db.study(study_name)
    out = Path(studies_root) / study_name / "powerlaw"
    out.mkdir(parents=True, exist_ok=True)
    recs, upstream, fits, summ = [], {}, [], []
    for key in st.all_runs():
        run = db.run(key)
        ck, rid = _load_ck(db, run, source, data_root, processed)
        upstream[key] = rid
        ds = load_dataset(db, run, ck, st)
        a = analyze_zbar(ds, st.zbar, db.pipeline.get("analysis"))
        S = compute_sheil(ds)
        recs.append({"ds": ds, "zb": a["zb"], "fit": a["fit"], "S": S})
        f = a["fit"] or {}
        fits.append({"run": key, "dataset": ds.name, "E_start_z2_eV": a["E_start"], **f,
                     "sheil_peak_total": S["peak_total"], "sheil_E_peak_eV": S["E_peak_total"],
                     "yield_ions_sr_shot": S["yield_sr"], "check_E_vs_t": S["check_E_vs_t"],
                     "check_sec9": S["check_sec9"], "jacobian_check": a["jacobian_check_rel"],
                     "excluded_z": " ".join(map(str, ds.excluded_z))})
        summ.append(summarize(ds).assign(run=key, dataset=ds.name))
        if f:
            log(f"  {ds.name:<10} zbar = {f['C']:.4g} E^{f['b']:.4f} (+/- {f['sig_b']:.4f}), "
                f"n = {f['n']:.3f}, R2 = {f['r2']:.4f}, E in [{f['E_lo']:.1f}, "
                f"{f['E_hi']:.1f}] eV ({f['npts']} pts); yield {S['yield_sr']:.3e} /sr/shot")
    ref = reference_prefactor(recs)
    log(f"  reference line zbar = {ref['C']:.4g} E^{ref['exponent']} ({ref['npts']} points)")
    fits_df = pd.DataFrame(fits)
    summ_df = pd.concat(summ, ignore_index=True)
    fits_df.to_csv(out / "zbar_fits.csv", index=False)
    summ_df.to_csv(out / "summary_per_z.csv", index=False)
    figs = {"zbar_grid": pw.zbar_grid(recs), "zbar_all": pw.zbar_all(recs, ref),
            "sheil_grid": pw.sheil_grid(recs)}
    for dom in ("tof", "velocity", "dlnE", "energy"):
        figs[f"spectra_{dom}"] = pw.spectra_grid(recs, dom)
    for name, f in figs.items():
        if formats:
            save(f, out / name, formats=formats)
    cols = ["dataset", "C", "b", "sig_b", "n", "sig_n", "r2", "E_lo", "E_hi", "npts",
            "yield_ions_sr_shot", "excluded_z"]
    items = [("p", f"Study {study_name}, source: {source}. Charge states in excluded_z are "
                   f"removed everywhere. Click a legend entry to hide it, double-click to "
                   f"show only it."),
             ("table", fits_df[[c for c in cols if c in fits_df]], "zbar = C E^b fits"),
             ("h2", "Average charge state"), figs["zbar_grid"], figs["zbar_all"],
             ("h2", "Energy spectra (Sheil et al. style)"), figs["sheil_grid"],
             ("h2", "Per-charge-state spectra"), figs["spectra_tof"], figs["spectra_velocity"],
             figs["spectra_dlnE"], figs["spectra_energy"]]
    meta = (f"eiakit {__version__}, git {git_commit()}, {time.strftime('%Y-%m-%d %H:%M')}"
            f"; runs: {', '.join(st.all_runs())}")
    write_html_page(out / "powerlaw.html", f"{study_name}: zbar(E) and spectra", items, meta)
    man = {"study": study_name, "part": "powerlaw", "source": source,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "eiakit": __version__,
           "git": git_commit(), "settings_hash": _settings_hash(db, st),
           "upstream": upstream, "reference_line": ref,
           "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str),
                                       encoding="utf-8")
    return {"out": out, "fits": fits_df, "summary": summ_df, "reference": ref}


def study_status(db, study_name, part, data_root, processed, studies_root) -> dict:
    """ok / missing / stale (reasons) for a study part."""
    st = db.study(study_name)
    p = Path(studies_root) / study_name / part / "manifest.json"
    if not p.is_file():
        return {"state": "missing", "reasons": ["never made"]}
    man = json.loads(p.read_text(encoding="utf-8"))
    reasons = []
    if man.get("settings_hash") != _settings_hash(db, st):
        reasons.append("study settings or [corrections] changed")
    if man.get("source") == "pipeline":
        stage = STAGES[man.get("upstream_stage", "calibration")]
        for key in man.get("upstream", {}):
            run = db.run(key)
            m = read_manifest(processed, run, stage)
            if m is None or m.get("run_id") != man.get("upstream", {}).get(key):
                reasons.append(f"{stage.name} of {key} changed or missing")
        expected = (st.all_runs() if part == "powerlaw" else st.runs if part == "collapse" else
                    [p.run for p in st.reconstruction.panels] if part == "reconstruction"
                    else st.runs)
        if set(man.get("upstream", {})) != set(expected):
            reasons.append("runs of the study changed")
    return {"state": "stale" if reasons else "ok", "reasons": reasons}


def cbs(db, study_name, data_root, processed, studies_root, source="pipeline",
        formats=("pdf", "png"), log=print) -> dict:
    """CBS component figures (plot_cbs_multipanel_v2): velocity, picks, area_fwhm per laser
    energy, fit_summary; tables of every component and of the mean +/- std lines."""
    from .cbs_study import component_means, study_components
    from .figures import cbs_study as cf
    from .figures.export import save, write_html_page

    st = db.study(study_name)
    out = Path(studies_root) / study_name / "cbs"
    out.mkdir(parents=True, exist_ok=True)
    tab = study_components(db, study_name, data_root, processed, source)
    upstream = {}
    for key in st.runs:
        m = read_manifest(processed, db.run(key), STAGES["cbs"]) if source == "pipeline" \
            else None
        upstream[key] = m["run_id"] if m else f"legacy:{source}"
    means = pd.concat([component_means(tab, st.avg_components, c, nan_aware=(c != "u_fit"))
                       for c in ("u_peak", "u_pick", "u_fit")], ignore_index=True)
    tab.to_csv(out / "components.csv", index=False)
    means.to_csv(out / "component_means.csv", index=False)
    for r in means.itertuples():
        log(f"  {r.species} {r.energy_mJ:>5g} mJ  {r.column:<7} {r.component:<10} "
            f"{r.mean:7.1f} +/- {r.std:5.1f} km/s ({r.n} z)")
    energies = sorted({float(e) for e in tab.energy_mJ}, reverse=True)
    figs = {}
    for mj in energies:
        figs[f"velocity_{mj:g}mJ"] = cf.velocity(tab, mj, st.avg_components, "velocity")
        figs[f"picks_{mj:g}mJ"] = cf.velocity(tab, mj, st.avg_components, "picks")
        figs[f"area_fwhm_{mj:g}mJ"] = cf.area_fwhm(tab, mj)
    figs["fit_summary"] = cf.fit_summary(tab, st.avg_components,
                                        sorted(int(e) for e in energies))
    for name, f in figs.items():
        if formats:
            save(f, out / name, formats=formats)
    n_ex = int(tab.excluded.sum())
    items = [("p", f"Study {study_name}, areas from: {source}. Component names from the db "
                   f"(legacy.cbs_components); {n_ex} (z, component) points excluded "
                   f"(study [exclude]). Velocities u = L/t with L from each fit-file header."),
             ("table", means, "mean +/- std of the averaged components")]
    for mj in energies:
        items += [("h2", f"{mj:g} mJ"), figs[f"velocity_{mj:g}mJ"], figs[f"picks_{mj:g}mJ"],
                  figs[f"area_fwhm_{mj:g}mJ"]]
    items += [("h2", "Fit peak and width, all conditions"), figs["fit_summary"]]
    meta = (f"eiakit {__version__}, git {git_commit()}, {time.strftime('%Y-%m-%d %H:%M')}"
            f"; runs: {', '.join(st.runs)}")
    write_html_page(out / "cbs.html", f"{study_name}: CBS components", items, meta)
    man = {"study": study_name, "part": "cbs", "source": source,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "eiakit": __version__,
           "git": git_commit(), "settings_hash": _settings_hash(db, st),
           "upstream": upstream, "upstream_stage": "cbs",
           "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str),
                                       encoding="utf-8")
    return {"out": out, "components": tab, "means": means}


def reconstruction(db, study_name, data_root, processed, studies_root, source="pipeline",
                   formats=("pdf", "png"), log=print) -> dict:
    """Representative charge states with their CBS decomposition and residuals
    (plot_reconstructed_charge_states_v2); component names from the db."""
    from .cbs import run_model
    from .figures.export import save, write_html_page
    from .figures.run_report import reconstruction as recon_fig

    st = db.study(study_name)
    out = Path(studies_root) / study_name / "reconstruction"
    out.mkdir(parents=True, exist_ok=True)
    cols, upstream, rows = [], {}, []
    for pnl in st.reconstruction.panels:
        run = db.run(pnl.run)
        ck, rid = _load_ck(db, run, source, data_root, processed)
        upstream[run.key] = rid
        rm = run_model(db, run, data_root, ck)
        if rm["status"] != "ok":
            raise RuntimeError(f"{run.key}: {rm}")
        title = f"{run.target}, {run.laser.energy_mJ:g} mJ"
        cols.append((title, rm, run.cbs_names, list(pnl.z)))
        for z in pnl.z:
            for c in rm["components"].get(z, []):
                if c.active:
                    rows.append({"run": run.key, "z": z, "cbs_file": c.extra.get("cbs"),
                                 "component": run.cbs_names.get(c.extra.get("cbs")),
                                 "kT_eV": c.kT, "V0_V": c.V0, "A": c.A,
                                 "win_left_us": c.win_left, "win_right_us": c.win_right})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "components.csv", index=False)
    fig = recon_fig(cols)
    if formats:
        save(fig, out / "TOF_reconstruction", formats=formats)
    write_html_page(out / "reconstruction.html", f"{study_name}: TOF reconstruction",
                    [("p", f"Source: {source}. Components named from the db "
                           f"(legacy.cbs_components)."), fig, ("table", tab, "components")],
                    f"eiakit {__version__}, git {git_commit()}")
    for r in tab.itertuples():
        log(f"  {r.run} z = {r.z:>2} {r.component:<10} kT = {r.kT_eV:9.3f} eV  "
            f"V0 = {r.V0_V:9.3f} V  A = {r.A:.4e}")
    man = {"study": study_name, "part": "reconstruction", "source": source,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "eiakit": __version__,
           "git": git_commit(), "settings_hash": _settings_hash(db, st),
           "upstream": upstream,
           "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str),
                                       encoding="utf-8")
    return {"out": out, "components": tab}


def collapse(db, study_name, data_root, processed, studies_root, source="pipeline",
             formats=("pdf", "png"), log=print) -> dict:
    """Velocity collapse u2/u1 = z/z* from the J_z(t) peaks and from the CBS fits
    (plot_collapse_v2), slow-branch energy scaling, literature, peak-check figures."""
    from . import collapse as C
    from .analysis import load_dataset
    from .figures import collapse as cf
    from .figures.export import save, write_html_page

    st = db.study(study_name)
    cs = st.collapse
    out = Path(studies_root) / study_name / "collapse"
    out.mkdir(parents=True, exist_ok=True)
    upstream, dsets, pk = {}, [], []
    for key in st.runs:
        run = db.run(key)
        ck, rid = _load_ck(db, run, source, data_root, processed)
        upstream[key] = rid
        ds = load_dataset(db, run, ck, st)
        dsets.append(ds)
        pk.append(C.eia_peaks(ds, cs.peak_min_prom_frac, cs.peak_err_mode))
    peaks = {"eia": C.drop_excluded(pd.concat(pk, ignore_index=True), st.exclude),
             "cbs": C.drop_excluded(pd.concat([C.cbs_peaks(db, db.run(k), data_root,
                                                           cs.cbs_t_source)
                                               for k in st.runs], ignore_index=True),
                                    st.exclude)}
    prof = {k: db.profile_for(db.run(k), "analysis") for k in st.runs}
    masses = {k: db.mass_kg(db.run(k).target, prof[k]) for k in st.runs}
    e_C = db.e_C(prof[st.runs[0]])
    lit = C.literature_series({k: db.literature[k] for k in cs.literature})
    fits, efits, figs = [], [], {}
    for src, title in (("eia", "peaks from the J_z(t) curves"), ("cbs", "CBS fit files")):
        peaks[src].to_csv(out / f"peaks_{src}.csv", index=False)
        ser = C.collapse_series(peaks[src], cs.runs, masses, e_C, cs.min_fast_points, log)
        j = C.weighted_joint([s["zstar"] for s in ser], [s["se_zstar"] for s in ser])
        log(f"  [{title}]")
        for s in ser:
            log(f"    {s['element']} {s['energy_mJ']:>4g} mJ: z* = {s['zstar']:.2f} +/- "
                f"{s['se_zstar']:.2f}  s_phys = {s['s'] * s['u1']:.2f} km/s  u1 = "
                f"{s['u1']:.1f} km/s  R2 = {s['R2']:.3f} ({len(s['z'])} slow)")
            fits.append({"source": src, "run": s["run"], "element": s["element"],
                         "energy_mJ": s["energy_mJ"], "zstar": s["zstar"],
                         "se_zstar": s["se_zstar"], "s_dim": s["s"], "se_s": s["se_s"],
                         "s_phys_kms": s["s"] * s["u1"], "u1_kms": s["u1"], "R2": s["R2"],
                         "n_slow": len(s["z"]), "n_fast": s["n_fast"]})
            ef = s["efit"]
            efits.append({"source": src, "run": s["run"],
                          **{k: v for k, v in ef.items() if k not in ("z", "E2_keV")}})
        log(f"    joint z* = {j['zstar']:.2f} +/- {j['se']:.2f} ({j['rms_pct']:.0f}% rms, "
            f"{j['n']} datasets)")
        fits.append({"source": src, "run": "joint", "zstar": j["zstar"], "se_zstar": j["se"],
                     "rms_pct": j["rms_pct"], "n_datasets": j["n"]})
        figs[f"collapse_{src}"] = cf.collapse(ser, j, lit, f"Collapse: {title}")
        figs[f"peak_check_{src}"] = cf.peak_check(dsets, peaks[src])
    for L in lit:
        log(f"  {L['label']}: z* = {L['zstar']:.2f} +/- {L['se_zstar']:.2f}, R2 = {L['R2']:.3f}")
        fits.append({"source": "literature", "run": L["key"], "zstar": L["zstar"],
                     "se_zstar": L["se_zstar"], "s_dim": L["s"], "se_s": L["se_s"],
                     "s_phys_kms": L["s"] * L["u1"], "u1_kms": L["u1"], "R2": L["R2"]})
    fits_df, ef_df = pd.DataFrame(fits), pd.DataFrame(efits)
    fits_df.to_csv(out / "collapse_fits.csv", index=False)
    ef_df.to_csv(out / "energy_fits.csv", index=False)
    for name, f in figs.items():
        if formats:
            save(f, out / name, formats=formats)
    items = [("p", f"Source: {source}. Collapse u2/u1 = z/z* (fit through the origin); u1 = "
                   f"mean fast velocity (>= {cs.min_fast_points} fast peaks), u2 = slow peak. "
                   f"Bars: {cs.peak_err_mode} (curves), prev/next samples (CBS)."),
             ("table", fits_df, "z* fits"), ("table", ef_df, "slow-branch energy fits (keV)"),
             ("h2", "Collapse"), figs["collapse_eia"], figs["collapse_cbs"],
             ("h2", "Peak check"), figs["peak_check_eia"], figs["peak_check_cbs"]]
    write_html_page(out / "collapse.html", f"{study_name}: velocity collapse", items,
                    f"eiakit {__version__}, git {git_commit()}")
    man = {"study": study_name, "part": "collapse", "source": source,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"), "eiakit": __version__,
           "git": git_commit(), "settings_hash": _settings_hash(db, st),
           "upstream": upstream,
           "outputs": sorted(p.name for p in out.iterdir() if p.name != "manifest.json")}
    (out / "manifest.json").write_text(json.dumps(man, indent=2, default=str),
                                       encoding="utf-8")
    return {"out": out, "fits": fits_df, "energy_fits": ef_df}
