"""
Paper numbers of a study, snapshots and diffs (Step 11: every correction is followed by
a diff table and a changelog entry).

A snapshot is a long table  group | item | dataset | value  with every number the paper
can quote:
    zbar            C, b, n, R2, E_lo, E_hi per dataset; Sheil yield (ions / sr / shot)
    cbs_velocity    mean / std of the averaged components (u_peak, u_pick, u_fit)
    cbs_params      kT, V0 (and their scale) per (run, z, component)
    cbs_ions        ions cm^-2 per (run, z, component): model on the whole grid (figures),
                    data in the component window (shows a stale fit after a data change)
    collapse        z*, se, s_phys, u1 per dataset and joint, for both peak sources
    energy          slow-branch energy fits (n, A, intercepts) per dataset and source
Snapshots live in studies/<name>/snapshots/<label>.csv (+ .json: corrections, git, time).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .pipeline import git_commit


def _rows(group, item, dataset, value):
    return {"group": group, "item": item, "dataset": dataset, "value": float(value)}


def paper_numbers(db, study_name, data_root, processed, source="pipeline") -> pd.DataFrame:
    from . import collapse as C
    from .analysis import analyze_zbar, compute_sheil, load_dataset
    from .cbs import run_model
    from .cbs_study import component_means, study_components
    from .study import _load_ck

    st = db.study(study_name)
    out = []
    cks = {}
    for key in st.all_runs():
        run = db.run(key)
        ck, _ = _load_ck(db, run, source, data_root, processed)
        cks[key] = ck
        ds = load_dataset(db, run, ck, st)
        a = analyze_zbar(ds, st.zbar, db.pipeline.get("analysis"))
        f = a["fit"] or {}
        for k in ("C", "b", "n", "r2", "E_lo", "E_hi"):
            if k in f:
                out.append(_rows("zbar", k, ds.name, f[k]))
        out.append(_rows("zbar", "yield_ions_sr_shot", ds.name, compute_sheil(ds)["yield_sr"]))
    # CBS: parameters (after any I1 rescaling) and ions on the current calibration
    for key in st.runs:
        run = db.run(key)
        rm = run_model(db, run, data_root, cks[key])
        if rm["status"] != "ok":
            continue
        names = run.legacy.cbs_components
        mo = rm["model"]
        for z, cs in rm["components"].items():
            act = [c for c in cs if c.active]
            if not act or z not in rm["curves"]:
                continue
            a = mo.areas(act, rm["curves"][z], z)
            for c, r in zip(act, a.itertuples(), strict=True):
                lbl = f"{key} z{z} {names.get(c.extra.get('cbs'), c.extra.get('cbs'))}"
                out += [_rows("cbs_params", "kT_eV", lbl, c.kT),
                        _rows("cbs_params", "V0_V", lbl, c.V0),
                        _rows("cbs_ions", "ions_model_full_cm2", lbl, r.Ions_model_full_cm2),
                        _rows("cbs_ions", "ions_data_win_cm2", lbl, r.Ions_data_win_cm2)]
    tab = study_components(db, study_name, data_root, processed,
                           "pipeline" if source == "pipeline" else "legacy")
    for col in ("u_peak", "u_pick", "u_fit"):
        m = component_means(tab, st.avg_components, col, nan_aware=(col != "u_fit"))
        for r in m.itertuples():
            lbl = f"{r.species}_{r.energy_mJ:g}mJ {r.component}"
            out += [_rows("cbs_velocity", f"{col}_mean", lbl, r.mean),
                    _rows("cbs_velocity", f"{col}_std", lbl, r.std)]
    # collapse
    cs = st.collapse
    prof = {k: db.profile_for(db.run(k), "analysis") for k in st.runs}
    masses = {k: db.mass_kg(db.run(k).target, prof[k]) for k in st.runs}
    e_C = db.e_C(prof[st.runs[0]])
    pk = {"eia": pd.concat([C.eia_peaks(load_dataset(db, db.run(k), cks[k], st),
                                        cs.peak_min_prom_frac, cs.peak_err_mode)
                            for k in st.runs], ignore_index=True),
          "cbs": pd.concat([C.cbs_peaks(db, db.run(k), data_root, cs.cbs_t_source)
                            for k in st.runs], ignore_index=True)}
    for src, p in pk.items():
        ser = C.collapse_series(C.drop_excluded(p, st.exclude), cs.runs, masses, e_C,
                                cs.min_fast_points)
        for s in ser:
            lbl = f"{src} {s['run']}"
            out += [_rows("collapse", "zstar", lbl, s["zstar"]),
                    _rows("collapse", "se_zstar", lbl, s["se_zstar"]),
                    _rows("collapse", "s_phys_kms", lbl, s["s"] * s["u1"]),
                    _rows("collapse", "u1_kms", lbl, s["u1"])]
            ef = s["efit"]
            for k in ("n_pow", "A_pow", "m1", "b1", "a2", "b2", "c2"):
                out.append(_rows("energy", k, lbl, ef[k]))
        j = C.weighted_joint([s["zstar"] for s in ser], [s["se_zstar"] for s in ser])
        out += [_rows("collapse", "zstar", f"{src} joint", j["zstar"]),
                _rows("collapse", "se_zstar", f"{src} joint", j["se"])]
    return pd.DataFrame(out)


def snapshot_dir(studies_root, study_name) -> Path:
    return Path(studies_root) / study_name / "snapshots"


def write_snapshot(db, study_name, label, data_root, processed, studies_root,
                   source="pipeline", note: str = "") -> Path:
    tab = paper_numbers(db, study_name, data_root, processed, source)
    d = snapshot_dir(studies_root, study_name)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{label}.csv"
    tab.to_csv(p, index=False, float_format="%.10g")
    meta = {"label": label, "study": study_name, "source": source, "note": note,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "eiakit": __version__,
            "git": git_commit(), "corrections": db.corrections(),
            "calibration": dict(db.pipeline["calibration"]),
            "n_numbers": int(len(tab))}
    p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def diff(studies_root, study_name, a: str, b: str) -> dict:
    d = snapshot_dir(studies_root, study_name)
    A, B = pd.read_csv(d / f"{a}.csv"), pd.read_csv(d / f"{b}.csv")
    ma = json.loads((d / f"{a}.json").read_text(encoding="utf-8"))
    mb = json.loads((d / f"{b}.json").read_text(encoding="utf-8"))
    m = A.merge(B, on=["group", "item", "dataset"], how="outer", suffixes=("_a", "_b"),
                indicator=True)
    m["delta"] = m.value_b - m.value_a
    with np.errstate(divide="ignore", invalid="ignore"):
        m["rel"] = np.where(m.value_a != 0, m.delta / m.value_a.abs(), np.nan)
    summ = []
    for (g, it), grp in m[m._merge == "both"].groupby(["group", "item"], sort=False):
        r = grp.rel.abs()
        i = r.idxmax() if r.notna().any() else grp.index[0]
        summ.append({"group": g, "item": it, "n": len(grp),
                     "changed": int((grp.delta.abs() > 1e-12 * grp.value_a.abs()
                                     .clip(lower=1e-300)).sum()),
                     "median_rel": float(r.median()) if r.notna().any() else np.nan,
                     "max_rel": float(r.max()) if r.notna().any() else np.nan,
                     "at": grp.loc[i, "dataset"], "from": grp.loc[i, "value_a"],
                     "to": grp.loc[i, "value_b"]})
    return {"table": m.drop(columns="_merge"), "summary": pd.DataFrame(summ),
            "only_a": m[m._merge == "left_only"], "only_b": m[m._merge == "right_only"],
            "meta_a": ma, "meta_b": mb}


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.6g}"


def changelog_entry(res: dict, a: str, b: str, title: str, text: str = "") -> str:
    ma, mb = res["meta_a"], res["meta_b"]
    sw = []
    for k in sorted(set(ma.get("corrections", {})) | set(mb.get("corrections", {}))):
        va, vb = ma.get("corrections", {}).get(k), mb.get("corrections", {}).get(k)
        if va != vb:
            sw.append(f"`{k}`: {va!r} → {vb!r}")
    for k in sorted(set(ma.get("calibration", {})) | set(mb.get("calibration", {}))):
        va, vb = ma.get("calibration", {}).get(k), mb.get("calibration", {}).get(k)
        if va != vb:
            sw.append(f"`calibration.{k}`: {va!r} → {vb!r}")
    lines = [f"## {title}", "",
             f"*{time.strftime('%Y-%m-%d %H:%M')}* — snapshots `{a}` → `{b}` "
             f"(git {ma.get('git')} → {mb.get('git')}, source {mb.get('source')})", ""]
    if sw:
        lines += ["Switches: " + "; ".join(sw), ""]
    if text:
        lines += [text, ""]
    s = res["summary"]
    s = s[s.changed > 0]
    if s.empty:
        lines += ["No paper number changed.", ""]
    else:
        lines += ["| group | item | changed / n | median rel. | max rel. | at | from | to |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            lines.append(f"| {r['group']} | {r['item']} | {r['changed']} / {r['n']} | "
                         f"{_fmt(r['median_rel'])} | {_fmt(r['max_rel'])} | {r['at']} | "
                         f"{_fmt(r['from'])} | {_fmt(r['to'])} |")
        lines.append("")
    if len(res["only_a"]) or len(res["only_b"]):
        lines += [f"Numbers only in `{a}`: {len(res['only_a'])}; only in `{b}`: "
                  f"{len(res['only_b'])}.", ""]
    return "\n".join(lines) + "\n"
