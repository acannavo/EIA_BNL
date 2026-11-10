"""
Pipeline stages, provenance manifests, status and incremental runs.

Each stage of a run writes into processed/<run>/<folder>/ and leaves a manifest.json:
    stage, run, created, eiakit version, git commit,
    code     - sha256 of the source modules the stage uses,
    params   - every database value the stage reads (full copy, JSON),
    inputs   - every file it reads: size, mtime, sha256 (data files relative to the
               data root, database files as 'db:<path>'),
    upstream - run_id of the stage outputs it reads,
    outputs  - sha256 of the files it wrote,
    run_id   - hash of code + params + inputs + upstream (what the outputs depend on),
    summary, ok.

`status` recomputes code, params and the input list and compares them with the manifest:
a stage is 'ok', 'missing', 'failed' or 'stale' (with the reasons: which db values,
which files, which code, or an upstream stage that was rerun or is stale itself).
`run` executes the stages that are not ok, in dependency order.

Legacy mode (Phase A): calibration reads the legacy peak tables (2_SEM/Output) and the
FC traces; cbs reads the stored legacy CBS parameters and the *new* calibration output,
so a change of e.g. the FC aperture in the db makes calibration and cbs stale.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__

SRC = Path(__file__).resolve().parent
MANIFEST = "manifest.json"


# ---- hashing ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_hash(modules: list[str]) -> str:
    h = hashlib.sha256()
    for m in sorted(modules):
        h.update(m.encode())
        h.update((SRC / f"{m}.py").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=SRC, text=True,
                           capture_output=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def flatten(d, prefix=""):
    """{'a': {'b': 1}} -> {'a.b': 1} (lists kept as values)."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def _jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return json.loads(json.dumps(obj, default=str))


# ---- inputs ----------------------------------------------------------------------------

class Inputs:
    """File list of a stage, with names relative to the data root or 'db:...'."""

    def __init__(self, data_root: Path, db_root: Path):
        self.data_root, self.db_root = Path(data_root), Path(db_root)
        self.files: dict[str, Path] = {}

    def add(self, path: Path):
        path = Path(path)
        if not path.is_file():
            return
        try:
            name = path.relative_to(self.data_root).as_posix()
        except ValueError:
            try:
                name = "db:" + path.relative_to(self.db_root).as_posix()
            except ValueError:
                name = path.as_posix()
        self.files[name] = path

    def add_many(self, paths):
        for p in paths:
            self.add(p)

    def signatures(self, previous: dict | None = None) -> dict:
        """size, mtime_ns, sha256 per file; sha256 reused when size and mtime match."""
        previous = previous or {}
        out = {}
        for name, p in sorted(self.files.items()):
            st = p.stat()
            old = previous.get(name)
            if old and old["size"] == st.st_size and old["mtime_ns"] == st.st_mtime_ns:
                sha = old["sha256"]
            else:
                sha = sha256_file(p)
            out[name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}
        return out


# ---- stages ----------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    folder: str
    depends: list[str]
    modules: list[str]
    params: Callable          # (db, run) -> dict
    inputs: Callable          # (db, run, data_root, ins: Inputs) -> None
    execute: Callable         # (db, run, ctx: Context) -> (ok: bool, summary: dict)
    doc: str = ""


@dataclass
class Context:
    db: object
    run: object
    data_root: Path
    processed: Path
    out: Path
    log: Callable = print
    upstream_out: dict = field(default_factory=dict)


def _sem_dir(run, data_root):
    return run.folder(data_root) / run.acquisition.sem.folder


def _results_dir(run, data_root):
    return run.folder(data_root) / "EIA-Results"


# audit --------------------------------------------------------------------------------

def _audit_params(db, run):
    return {"run": _jsonable(run.rec), "constants": _jsonable(db.constants),
            "species": {k: _jsonable(v) for k, v in db.species.items()}}


def _audit_inputs(db, run, data_root, ins):
    sem = _sem_dir(run, data_root)
    for sub in ("Output", "Output_detail"):
        d = sem / sub
        if d.is_dir():
            ins.add_many(sorted(d.glob("*.txt")))
    res = _results_dir(run, data_root)
    if res.is_dir():
        ins.add_many(sorted(p for p in res.iterdir() if p.suffix in (".txt", ".npy")))


def _audit_execute(db, run, ctx):
    from .audit import audit_run, write_report
    rep = audit_run(db, run, ctx.data_root)
    write_report(rep, ctx.out)
    n = {lvl: sum(f.level == lvl for f in rep.findings) for lvl in ("error", "warning", "info")}
    return True, {"findings": n}


# peaks --------------------------------------------------------------------------------

def _peaks_params(db, run):
    leg = run.legacy
    return {"pipeline.peaks": db.pipeline["peaks"], "eia": _jsonable(run.eia),
            "acquisition.sem": _jsonable(run.acquisition.sem),
            "legacy": {"profile_peaks": leg.profile_peaks, "peaks_lof_m": leg.peaks_lof_m,
                       "peaks_grid": leg.peaks_grid},
            "constants": _jsonable(db.profile(leg.profile_peaks))}


def _peaks_inputs(db, run, data_root, ins):
    from .acquisition import scan_folder
    from .peaks import peak_state_path
    sem = _sem_dir(run, data_root)
    if sem.is_dir():
        ins.add_many(p for p, _, _ in scan_folder(sem, run.acquisition.sem).matched)
        if (sem / "Output_detail").is_dir():
            ins.add_many(sorted((sem / "Output_detail").glob("*.txt")))
    ins.add(peak_state_path(db, run.key))


def _peaks_execute(db, run, ctx):
    import pandas as pd

    from .peaks import load_peak_state, replay_run
    res = replay_run(db, run, ctx.data_root, state=load_peak_state(db, run.key))
    pd.DataFrame(res["summary"]).to_csv(ctx.out / "replay_summary.csv", index=False)
    if res["rows"]:
        pd.concat(res["rows"]).to_csv(ctx.out / "replay_rows.csv", index=False)
    c = res["counts"]
    return c["partial"] == c["known_partial"], c


# calibration --------------------------------------------------------------------------

def _cal_params(db, run):
    return {"pipeline.calibration": db.pipeline["calibration"], "eia": _jsonable(run.eia),
            "fc": _jsonable(run.fc), "acquisition.fc": _jsonable(run.acquisition.fc),
            "legacy.profile_calibration": run.legacy.profile_calibration,
            "peaks.gui_masses": db.pipeline["peaks"]["gui_masses"],
            "mass_labels": list(run.mass_labels)}


def _cal_inputs(db, run, data_root, ins):
    from .acquisition import scan_folder
    from .calibration import fc_dirs
    pipe = db.pipeline["calibration"]
    sem = _sem_dir(run, data_root)
    if (sem / "Output").is_dir():
        ins.add_many(sorted((sem / "Output").glob("*.txt")))
    if (pipe.get("output_duplicates", "all") in ("last_per_mz", "last_save")
            and (sem / "Output_detail").is_dir()):
        ins.add_many(sorted((sem / "Output_detail").glob("*.txt")))
    fmt = run.acquisition.fc
    for d in fc_dirs(run, pipe.get("fc_source", "start")):
        if d and (run.folder(data_root) / d).is_dir():
            ins.add_many(p for p, _, _ in scan_folder(run.folder(data_root) / d, fmt).matched)
    if pipe.get("fc_baseline", "legacy") == "legacy":
        ins.add(_results_dir(run, data_root) / "pipeline_checkpoint.npy")


def _cal_execute(db, run, ctx):
    from .calibration import run_calibration
    from .store import save_calibration_h5
    pipe = db.pipeline["calibration"]
    r = run_calibration(db, run, ctx.data_root, pipe.get("fc_source", "start"),
                        pipe.get("fc_baseline", "legacy"))
    if "error" in r:
        return False, {"error": r["error"]}
    save_calibration_h5(ctx.out / "spectra.h5", r["res"],
                        {"run": run.key, "baseline_method": r["info"]["baseline_method"]})
    (ctx.out / "calibration.json").write_text(json.dumps(r["info"], indent=2, default=str),
                                              encoding="utf-8")
    lc = r["info"].get("legacy_comparison", {})
    return True, {"CFY_1m": float(r["info"]["CFY_1m"]),
                  "fc_baseline_V": float(r["info"]["fc_baseline_V"]),
                  "CFY_ratio_new_over_legacy": lc.get("CFY_ratio_new_over_legacy"),
                  "identical_to_legacy": lc.get("identical_1e-9")}


# cbs ----------------------------------------------------------------------------------

def _cbs_params(db, run):
    prof = db.profile_for(run, "cbs")
    return {"pipeline.cbs": db.pipeline["cbs"], "constants": _jsonable(db.profile(prof)),
            "profile": prof, "corrections.mass_constants":
                db.corrections().get("mass_constants", "legacy"),
            "mass_amu": db.mass_amu(run.target, prof), "mass_labels": run.mass_labels,
            "fit_file_labels": run.fit_file_labels, "target": run.target,
            "eia.L_m": run.eia.L_m}


def _cbs_inputs(db, run, data_root, ins):
    from .audit import _find_cbs_file
    res = _results_dir(run, data_root)
    for prefix in ("CBS_fit_results", "CBS_areas"):
        ins.add(_find_cbs_file(res, prefix, run.fit_file_labels, run.target) or res / "-")


def _cbs_execute(db, run, ctx):
    from .cbs import replay_run_cbs
    from .store import load_calibration_h5
    ck = load_calibration_h5(ctx.upstream_out["calibration"] / "spectra.h5")
    r = replay_run_cbs(db, run, ctx.data_root, ck=ck)
    if r["status"] == "missing files":
        return False, {"error": f"missing files {r['files']}"}
    r["areas"].to_csv(ctx.out / "cbs_areas.csv", index=False)
    r["components"].to_csv(ctx.out / "replay_components.csv", index=False)
    r["areas_cmp"].drop(columns="_merge", errors="ignore").to_csv(
        ctx.out / "replay_areas.csv", index=False)
    r["totals"].to_csv(ctx.out / "replay_totals.csv", index=False)
    s = r["summary"]
    (ctx.out / "replay_summary.json").write_text(json.dumps(
        {"status": r["status"], "files": r["files"], **s}, indent=2, default=str),
        encoding="utf-8")
    return r["status"].startswith(("reproduced", "model reproduced")), {
        "status": r["status"], "components": s["n_components"],
        "data_area_scale_vs_legacy_fit": s.get("data_area_scale")}


# report -------------------------------------------------------------------------------

def study_of(db, run):
    """The first study (alphabetical) that lists the run, or None."""
    for name in sorted(db.studies):
        st = db.study(name)
        if run.key in st.runs or run.key in st.extra_runs:
            return st
    return None


def _report_params(db, run):
    st = study_of(db, run)
    return {"pipeline.analysis": db.pipeline.get("analysis", {}),
            "constants": _jsonable(db.profile(db.profile_for(run, "analysis"))),
            "corrections": db.corrections(),
            "study": None if st is None else {"name": st.name,
                                              "zbar": _jsonable(st.zbar),
                                              "exclude": st.exclude.get(run.key, {})},
            "cbs_components": {str(k): v for k, v in run.legacy.cbs_components.items()},
            "run": {"target": run.target, "label": run.label,
                    "laser": _jsonable(run.laser), "mass_labels": run.mass_labels}}


def _report_inputs(db, run, data_root, ins):
    _cbs_inputs(db, run, data_root, ins)


def _report_execute(db, run, ctx):
    import pandas as pd

    from .analysis import analyze_zbar, compute_sheil, load_dataset, summarize
    from .cbs import run_model
    from .figures import run_report as rr
    from .figures.export import write_html_page
    from .store import load_calibration_h5

    ck = load_calibration_h5(ctx.upstream_out["calibration"] / "spectra.h5")
    st = study_of(db, run)
    ds = load_dataset(db, run, ck, st)
    summ = summarize(ds)
    a = analyze_zbar(ds, st.zbar, db.pipeline.get("analysis")) if st is not None else {
        "zb": None, "fit": None}
    items = []
    meta = (f"{run.target}, laser {run.laser.energy_mJ} mJ, L_EIA = {run.eia.L_m} m, "
            f"L_FC = {run.fc.L_m} m; profile {db.profile_for(run, 'analysis')}"
            + (f"; excluded z {ds.excluded_z} (study {st.name})" if st and ds.excluded_z else "")
            + (f"; <b>provisional: {', '.join(run.provisional)}</b>" if run.provisional else "")
            + f"<br>eiakit {__version__}, git {git_commit()}, {time.strftime('%Y-%m-%d %H:%M')}")
    items.append(rr.spectra_overview(ds))
    rm = run_model(db, run, ctx.data_root, ck)
    areas = None
    if rm["status"] == "ok":
        items += [("h2", "CBS decomposition per charge state"),
                  ("p", "Stored legacy CBS parameters (kT, V0, A, windows) on the current "
                        "calibration. Shaded: fit windows. Toggle a component in the legend."),
                  rr.cbs_panels(rm, run.legacy.cbs_components)]
        p = ctx.upstream_out["cbs"] / "cbs_areas.csv"
        if p.is_file():
            areas = pd.read_csv(p)
            cols = ["z", "CBS", "t_peak_fit_us", "FWHM_fit_us", "Area_model", "Fraction",
                    "Ions_model_full_cm2", "Ions_data_win_cm2", "Win_coverage", "win_src"]
            items.append(("table", areas[[c for c in cols if c in areas]], "CBS areas"))
    if a["zb"] is not None:
        S = compute_sheil(ds)
        items += [("h2", "Average charge state and energy spectra"),
                  rr.zbar_single(ds, a["zb"], a["fit"]), rr.sheil_single(ds, S, a["zb"])]
    items += [("h2", "Per charge state"), ("table", summ, "peak, centroid, area per z")]
    write_html_page(ctx.out / "report.html", f"{run.key}: {run.label}", items, meta)
    summ.to_csv(ctx.out / "summary_per_z.csv", index=False)
    f = a.get("fit") or {}
    return True, {"zbar_fit": {k: f.get(k) for k in ("C", "b", "n", "r2")} if f else None,
                  "charge_states": len(ds.all_z), "cbs_components": 0 if areas is None
                  else int(len(areas))}


STAGES: dict[str, Stage] = {s.name: s for s in [
    Stage("audit", "00_legacy_import", [], ["audit", "legacy", "db"], _audit_params,
          _audit_inputs, _audit_execute, "audit of the legacy results"),
    Stage("peaks", "10_peaks", [], ["peaks", "acquisition", "physics"], _peaks_params,
          _peaks_inputs, _peaks_execute, "untitled28: replay of the peak fits"),
    Stage("calibration", "20_calibration", [], ["calibration", "legacy", "acquisition",
                                                "store"],
          _cal_params, _cal_inputs, _cal_execute, "untitled27: FC calibration -> spectra.h5"),
    Stage("cbs", "30_cbs", ["calibration"], ["cbs", "physics", "legacy", "store"],
          _cbs_params, _cbs_inputs, _cbs_execute,
          "cbs_manual_fitter_v12: legacy CBS parameters on the current calibration"),
    Stage("report", "40_report", ["calibration", "cbs"],
          ["analysis", "cbs", "store", "figures/run_report", "figures/export",
           "figures/render_plotly", "figures/spec", "figures/style", "figures/powerlaw"],
          _report_params, _report_inputs, _report_execute, "per-run HTML report"),
]}
ORDER = ["audit", "peaks", "calibration", "cbs", "report"]


# ---- status ----------------------------------------------------------------------------

def out_dir(processed: Path, run, stage: Stage) -> Path:
    return Path(processed) / run.key / stage.folder


def read_manifest(processed: Path, run, stage: Stage) -> dict | None:
    p = out_dir(processed, run, stage) / MANIFEST
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _current(db, run, stage: Stage, data_root: Path, previous_inputs=None):
    ins = Inputs(data_root, db.root)
    stage.inputs(db, run, data_root, ins)
    params = _jsonable(stage.params(db, run))
    return {"code": code_hash(stage.modules), "params": params,
            "params_hash": sha256_json(params),
            "inputs": ins.signatures(previous_inputs)}


def stage_status(db, run, stage_name: str, data_root: Path, processed: Path,
                 _cache: dict | None = None) -> dict:
    """{'state': ok|missing|failed|stale, 'reasons': [...], 'manifest': ...}"""
    _cache = {} if _cache is None else _cache
    key = (run.key, stage_name)
    if key in _cache:
        return _cache[key]
    stage = STAGES[stage_name]
    man = read_manifest(processed, run, stage)
    if man is None:
        st = {"state": "missing", "reasons": ["never run"], "manifest": None}
        _cache[key] = st
        return st
    cur = _current(db, run, stage, data_root, man.get("inputs"))
    reasons = []
    if not man.get("ok", False):
        reasons.append(f"last run failed: {man.get('summary', {}).get('error', '')}".strip())
    if cur["code"] != man.get("code"):
        reasons.append(f"code changed ({', '.join(stage.modules)})")
    if cur["params_hash"] != man.get("params_hash"):
        new, old = flatten(cur["params"]), flatten(man.get("params", {}))
        ch = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
        for k in ch[:8]:
            reasons.append(f"db value {k}: {old.get(k, '<none>')!r} -> {new.get(k, '<none>')!r}")
        if len(ch) > 8:
            reasons.append(f"... and {len(ch) - 8} more db values")
    oi, ni = man.get("inputs", {}), cur["inputs"]
    added = sorted(set(ni) - set(oi))
    removed = sorted(set(oi) - set(ni))
    changed = sorted(k for k in set(ni) & set(oi) if ni[k]["sha256"] != oi[k]["sha256"])
    for label, lst in (("new input", added), ("input removed", removed),
                       ("input changed", changed)):
        if lst:
            more = f" (+{len(lst) - 3} more)" if len(lst) > 3 else ""
            reasons.append(f"{label}: {', '.join(lst[:3])}{more}")
    od = out_dir(processed, run, stage)
    bad = []
    for name, sig in (man.get("outputs") or {}).items():
        p = od / name
        if not p.is_file():
            bad.append(f"{name} missing")
        elif isinstance(sig, dict) and (p.stat().st_size != sig["size"]
                                        or p.stat().st_mtime_ns != sig["mtime_ns"]):
            if sha256_file(p) != sig["sha256"]:
                bad.append(f"{name} modified")
    if bad:
        reasons.append("outputs changed outside the pipeline: " + ", ".join(bad[:3]))
    for dep in stage.depends:
        ds = stage_status(db, run, dep, data_root, processed, _cache)
        dman = ds["manifest"]
        if ds["state"] != "ok":
            reasons.append(f"upstream '{dep}' is {ds['state']}")
        elif dman and dman.get("run_id") != man.get("upstream", {}).get(dep):
            reasons.append(f"upstream '{dep}' was rerun since")
    state = "ok" if not reasons else ("failed" if not man.get("ok", False) else "stale")
    st = {"state": state, "reasons": reasons, "manifest": man}
    _cache[key] = st
    return st


# ---- run -------------------------------------------------------------------------------

def run_stage(db, run, stage_name: str, data_root: Path, processed: Path,
              log=print) -> dict:
    stage = STAGES[stage_name]
    out = out_dir(processed, run, stage)
    out.mkdir(parents=True, exist_ok=True)
    prev = read_manifest(processed, run, stage)
    cur = _current(db, run, stage, data_root, (prev or {}).get("inputs"))
    upstream, up_out = {}, {}
    for dep in stage.depends:
        dman = read_manifest(processed, run, STAGES[dep])
        upstream[dep] = dman.get("run_id") if dman else None
        up_out[dep] = out_dir(processed, run, STAGES[dep])
    ctx = Context(db, run, Path(data_root), Path(processed), out, log, up_out)
    t0 = time.time()
    try:
        ok, summary = stage.execute(db, run, ctx)
    except Exception as exc:                      # recorded in the manifest
        ok, summary = False, {"error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc()}
    for p in [q for q in out.rglob("*") if q.is_file() and q.name != MANIFEST]:
        if p.stat().st_mtime < t0 - 1:            # left over from an earlier run
            p.unlink()
    outputs = {p.relative_to(out).as_posix(): {"size": p.stat().st_size,
                                               "mtime_ns": p.stat().st_mtime_ns,
                                               "sha256": sha256_file(p)}
               for p in sorted(out.rglob("*")) if p.is_file() and p.name != MANIFEST}
    man = {"stage": stage.name, "run": run.key, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "seconds": round(time.time() - t0, 2), "eiakit": __version__, "git": git_commit(),
           **cur, "upstream": upstream, "outputs": outputs, "summary": summary, "ok": bool(ok)}
    man["run_id"] = sha256_json({"code": man["code"], "params": man["params_hash"],
                                 "inputs": {k: v["sha256"] for k, v in man["inputs"].items()},
                                 "upstream": upstream})
    (out / MANIFEST).write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man


def plan(db, runs, data_root, processed, stages=None, force=False):
    """(run, stage, status) in execution order for the stages that need to run."""
    stages = stages or ORDER
    todo = []
    for run in runs:
        cache: dict = {}
        will = set()
        for name in ORDER:
            if name not in stages:
                continue
            st = stage_status(db, run, name, data_root, processed, cache)
            deps_rerun = any(d in will for d in STAGES[name].depends)
            if force or st["state"] != "ok" or deps_rerun:
                todo.append((run, name, st))
                will.add(name)
    return todo
