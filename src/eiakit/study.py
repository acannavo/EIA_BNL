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
        a = analyze_zbar(ds, st.zbar)
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
           "git": git_commit(), "settings_hash": sha256_json(st.model_dump(mode="json")),
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
    if man.get("settings_hash") != sha256_json(st.model_dump(mode="json")):
        reasons.append("study settings changed")
    if man.get("source") == "pipeline":
        for key in st.all_runs():
            run = db.run(key)
            m = read_manifest(processed, run, STAGES["calibration"])
            if m is None or m.get("run_id") != man.get("upstream", {}).get(key):
                reasons.append(f"calibration of {key} changed or missing")
        if set(man.get("upstream", {})) != set(st.all_runs()):
            reasons.append("runs of the study changed")
    return {"state": "stale" if reasons else "ok", "reasons": reasons}
