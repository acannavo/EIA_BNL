"""
Command line entry point:  ``eiakit <command>``

    eiakit info                     environment, paths, packages, run folders found
    eiakit checksum [--root DIR] [--out FILE] [--skip-variants]
                                    SHA-256 manifest of a tree (default: the data root)
    eiakit checksum-compare A B     compare two manifests (exit code 1 if they differ)
    eiakit runs [--all]             runs of the database with their conditions
    eiakit db-check                 validate the database (exit code 1 on errors)
    eiakit files KEY [--list]       raw files a run's acquisition format picks up / skips
    eiakit audit [KEY ...] [--study NAME]
                                    audit of the legacy results (all included runs if no KEY);
                                    reports in processed/<KEY>/00_legacy_import/
    eiakit calibrate KEY [--fc start|stop|both] [--baseline auto|pretrigger|<V>] [--legacy-cfy]
                                    calibration stage (untitled27); compares with the legacy
                                    checkpoint; outputs in processed/<KEY>/20_calibration/
    eiakit peaks KEY [--U V ...] [--max-clicks N] [--deep] [--write-state] [--search] [-v]
                                    peak stage (untitled28): refits every voltage with raw SEM
                                    files and an Output_detail file, recovers the U arrow
                                    clicks; outputs in processed/<KEY>/10_peaks/
    eiakit cbs [KEY ...]             CBS stage (cbs_manual_fitter_v12): recomputes areas, fit
                                    metrics and chi2 from the stored parameters and the
                                    checkpoint; outputs in processed/<KEY>/30_cbs/
    eiakit status [KEY ...] [--stage S ...] [-v]
                                    state of each stage: ok / stale (with the reasons: db
                                    values, input files, code, upstream) / missing / failed
    eiakit run [KEY ...] [--stage S ...] [--force] [--dry-run]
                                    runs what is missing or stale, in dependency order;
                                    each stage writes processed/<KEY>/<stage>/manifest.json
    eiakit study NAME [--part P ...] [--source pipeline|legacy] [--html-only] [--status]
                                    cross-run figures (PDF, PNG, one HTML page) and tables
                                    in studies/<NAME>/<part>/
    eiakit snapshot NAME --label L [--source pipeline|legacy] [--note TEXT]
                                    every paper number of a study -> studies/<NAME>/snapshots
    eiakit diff NAME A B [--changelog TITLE] [--text TEXT]
                                    compare two snapshots; append the table to changelog.md
    eiakit compare NAME KEY A B [--changelog TITLE] [--text TEXT]
                                    effect of one setting: run + snapshot with KEY = A, then
                                    B, diff (+ changelog); no file to edit
    eiakit methods NAME [--only P ...] [--boot N]
                                    method studies (Step 12) -> studies/<NAME>/methods/:
                                    zbar (I4/I5), calibration (I2), sensitivity (LOF, t0),
                                    fc (start/stop drift), shots (N12 bootstrap)
    eiakit note NAME TITLE [--text TEXT]
                                    changelog section for a check that changed no number
    eiakit --set SECTION.KEY=VALUE <command>
                                    override a db/pipeline.toml setting for one command
    eiakit import-exclusions FILE [--study NAME]
                                    legacy ExclusionList.txt -> the study's [exclude] table
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from . import __version__
from .acquisition import scan_folder
from .checksum import build_manifest, compare_manifests, read_manifest, write_manifest
from .db import DatabaseError, format_exclude_toml, import_exclusion_list, load_database
from .layout import discover_runs
from .paths import PathsNotConfigured, get_paths

PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "plotly", "h5py", "pydantic", "PySide6")


def cmd_info(_args) -> int:
    print(f"eiakit {__version__}   Python {platform.python_version()}   ({sys.executable})")
    for p in PACKAGES:
        try:
            v = md.version(p)
        except md.PackageNotFoundError:
            v = "MISSING"
        print(f"  {p:<11} {v}")
    try:
        from PySide6 import QtCore
        print(f"  Qt check    ok (Qt {QtCore.qVersion()})")
    except Exception as exc:                       # noqa: BLE001 - report any failure
        print(f"  Qt check    FAILED: {exc}  (GUI only; see the note at the top of env.ps1)")
    print(f"  base Python {sys.base_prefix}")
    try:
        paths = get_paths()
    except PathsNotConfigured as exc:
        print(f"\nPaths: NOT CONFIGURED -- {exc}")
        return 1
    print("\nPaths:")
    for name, ok in paths.check().items():
        print(f"  {name:<10} {'ok     ' if ok else 'missing'} {getattr(paths, name)}")
    if not paths.data.is_dir():
        return 1

    runs = discover_runs(paths.data)
    print(f"\nRun folders in data root: {len(runs)}")
    for r in runs:
        need = ["1_FC_Start", "2_SEM", "3_FC_Stop"]
        miss = [n for n in need if not r.has(n)]
        legacy = [n for n in ("2_SEM/Output", "2_SEM/Output_detail", "EIA-Results") if r.has(n)]
        print(f"  {r.name:<24} legacy: {', '.join(legacy) or '-':<45}"
              f" variants ignored: {len(r.variants):<2} extra: {len(r.extras):<2}"
              + (f"  MISSING: {miss}" if miss else ""))
    return 0


def cmd_checksum(args) -> int:
    if args.root:
        root = Path(args.root)
        default_dir = Path.cwd()
    else:
        paths = get_paths()
        root = paths.data
        paths.ensure_output_dirs()
        default_dir = paths.processed / "_manifests"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else default_dir / f"sha256_{root.name}_{stamp}.csv"
    print(f"Hashing {root} ...")
    rows = build_manifest(root, skip_variants=args.skip_variants,
                          progress=lambda n: print(f"  {n} files", flush=True))
    write_manifest(rows, out)
    total = sum(r["size"] for r in rows)
    print(f"{len(rows)} files, {total / 1e9:.2f} GB  ->  {out}")
    return 0


def cmd_checksum_compare(args) -> int:
    d = compare_manifests(read_manifest(Path(args.a)), read_manifest(Path(args.b)))
    print(f"identical files: {d.n_same}")
    for label, items in (("only in A", d.only_a), ("only in B", d.only_b),
                         ("different content", d.different)):
        print(f"{label}: {len(items)}")
        for k in items[:50]:
            print(f"    {k}")
        if len(items) > 50:
            print(f"    ... {len(items) - 50} more")
    print("RESULT:", "IDENTICAL" if d.identical else "DIFFERENT")
    return 0 if d.identical else 1


def _fmt(v, f="{:g}"):
    return "-" if v is None else f.format(v)


# ------------------------------------------------------------------ --set overrides

_OVERRIDES: list[tuple[str, str, object]] = []


def _parse_value(v: str):
    import tomllib
    try:
        return tomllib.loads(f"x = {v}")["x"]
    except tomllib.TOMLDecodeError:
        return v                                   # bare word -> string


def _parse_set(item: str) -> tuple[str, str, object]:
    """'calibration.output_duplicates=all' -> ('calibration', 'output_duplicates', 'all')."""
    if "=" not in item or "." not in item.split("=", 1)[0]:
        raise DatabaseError(f"--set {item!r}: use SECTION.KEY=VALUE of db/pipeline.toml, "
                            "e.g. calibration.output_duplicates=all")
    k, v = item.split("=", 1)
    sec, key = k.strip().split(".", 1)
    return sec, key, _parse_value(v.strip())


def _apply_overrides(db, overrides):
    for sec, key, val in overrides:
        if sec not in db.pipeline or key not in db.pipeline[sec]:
            raise DatabaseError(f"--set {sec}.{key}: no such setting in db/pipeline.toml")
        db.pipeline[sec][key] = val
    return db


def _load_db(strict=True):
    return _apply_overrides(load_database(strict=strict), _OVERRIDES)


def cmd_runs(args) -> int:
    db = _load_db()
    try:
        data_root = get_paths().data
    except PathsNotConfigured:
        data_root = None
    hdr = (f"{'key':<24}{'label':<22}{'tgt':<4}{'laser':<20}{'L_EIA':>6}{'L_FC':>6}"
           f"{'r_FC':>6}  {'acquisition':<15}{'folder':<7} provisional / notes")
    print(hdr)
    print("-" * len(hdr))
    for r in db.runs(include_only=not args.all):
        exists = "-" if data_root is None else ("ok" if r.folder(data_root).is_dir() else "MISSING")
        if not r.include:
            print(f"{r.key:<24}{r.label[:21]:<22}{'':<4}{'(not analysed)':<20}{'':>18}  {'':<15}"
                  f"{exists:<7} {r.notes}")
            continue
        las = r.laser
        laser = " ".join(x for x in (
            las.pulse or "",
            f"{las.energy_mJ:g} mJ" if las.energy_mJ else "",
            f"{'~' if las.approximate else ''}{las.intensity_Wcm2:.0e} W/cm2"
            if las.intensity_Wcm2 else "") if x) or "-"
        prov = ", ".join(r.provisional)
        print(f"{r.key:<24}{r.label:<22}{r.target:<4}{laser:<20}{r.eia.L_m:>6.2f}{r.fc.L_m:>6.2f}"
              f"{r.fc.aperture_radius_cm:>6.2f}  {r.rec.acquisition:<15}{exists:<7} {prov}")
    print()
    print("Ion mass of the target per constants profile [kg]:")
    for sp in sorted({r.target for r in db.runs()}):
        line = "  ".join(f"{p}={db.mass_kg(sp, p):.6e}" for p in db.constants.profiles)
        ratio = db.mass_kg(sp, "corrected") / db.mass_kg(sp, "legacy_proton")
        print(f"  {sp:<3} {line}   corrected/legacy_proton = {ratio:.5f}")
    print(f"\nStudies: {', '.join(sorted(db.studies)) or '-'}   (database: {db.root})")
    return 0


def cmd_db_check(_args) -> int:
    db = _load_db(strict=False)
    errors = db.check()
    n_inc = len(db.runs())
    print(f"database {db.root}: {len(db.run_keys)} runs ({n_inc} included), "
          f"{len(db.species)} species, {len(db.instruments.eia)} EIA / {len(db.instruments.fc)} FC "
          f"configurations, {len(db.constants.profiles)} constants profiles, "
          f"{len(db.studies)} studies")
    for e in errors:
        print(f"  ERROR: {e}")
    print("RESULT:", "OK" if not errors else f"{len(errors)} error(s)")
    return 0 if not errors else 1


def cmd_files(args) -> int:
    db = _load_db()
    run = db.run(args.key)
    if not run.include:
        print(f"{run.key}: include = false in runs.toml (no acquisition format)")
        return 1
    folder = run.folder(get_paths().data)
    acq = run.acquisition
    print(f"{run.key} ({run.label})   acquisition: {run.rec.acquisition}"
          + ("  + overrides" if run.rec.acquisition_overrides else ""))
    status = 0
    blocks = [("SEM", acq.sem, acq.sem.folder), ("FC start", acq.fc, acq.fc.folder)]
    if acq.fc.folder_stop:
        blocks.append(("FC stop", acq.fc, acq.fc.folder_stop))
    for title, fmt, sub in blocks:
        sc = scan_folder(folder / sub, fmt)
        print(f"\n[{title}]  {folder / sub}")
        print(f"  patterns {fmt.patterns}   ignore {fmt.ignore}")
        if sc.missing_folder:
            print("  FOLDER MISSING")
            status = 1
            continue
        print(f"  matched {len(sc.matched)}   ignored {len(sc.ignored)}   "
              f"unmatched {len(sc.unmatched)}")
        if title == "SEM":
            byU = sc.by_voltage()
            if byU:
                from collections import Counter
                counts = Counter(len(r) for r in byU.values())
                usual = counts.most_common(1)[0][0]
                Us = list(byU)
                print(f"  {len(byU)} deflector voltages U = {Us[0]:g} ... {Us[-1]:g} V; "
                      f"usual repeats per U: {usual} ({counts[usual]} voltages)")
                other = {U: len(r) for U, r in byU.items() if len(r) != usual}
                if other:
                    print("  other repeat counts: "
                          + ", ".join(f"{U:g} V: {n}" for U, n in other.items()))
                gaps = {U: r for U, r in byU.items() if r != list(range(1, len(r) + 1))}
                if gaps:
                    print(f"  repeat numbers not 1..n: {gaps}")
            if args.list:
                for U, r in byU.items():
                    print(f"    U = {U:<8g} repeats {r}")
            dup = sc.duplicates()
            if dup:
                print(f"  DUPLICATE (U, repeat): {dup}")
                status = 1
        for label, files in (("unmatched", sc.unmatched), ("ignored", sc.ignored)):
            shown = files if args.list else files[:10]
            for f in shown:
                print(f"    {label}: {f.name}")
            if len(files) > len(shown):
                print(f"    ... {len(files) - len(shown)} more {label} (use --list)")
        if not sc.matched:
            status = 1
    return status


def cmd_audit(args) -> int:
    from .audit import audit_run, write_report
    db = _load_db()
    paths = get_paths()
    paths.ensure_output_dirs()
    study = db.study(args.study) if args.study else None
    keys = args.keys or [r.key for r in db.runs()]
    print(f"{'run':<24}{'label':<22}{'errors':>7}{'warn':>6}{'info':>6}  report")
    worst = 0
    for key in keys:
        run = db.run(key)
        if not run.include:
            print(f"{key:<24}(include = false, skipped)")
            continue
        rep = audit_run(db, run, paths.data, study)
        out = write_report(rep, paths.processed / key / "00_legacy_import")
        print(f"{key:<24}{run.label:<22}{rep.count('error'):>7}{rep.count('warn'):>6}"
              f"{rep.count('info'):>6}  {out[0]}")
        for f in rep.findings:
            if f.level == "error":
                print(f"{'':<24}  ERROR {f.code}: {f.title[:110]}")
        worst = max(worst, 1 if rep.count("error") else 0)
    return worst


def cmd_calibrate(args) -> int:
    import json

    from .calibration import run_calibration
    from .store import save_calibration_h5

    db = _load_db()
    paths = get_paths()
    run = db.run(args.key)
    baseline = "legacy" if args.baseline == "auto" else args.baseline
    r = run_calibration(db, run, paths.data, args.fc, baseline, legacy_cfy=args.legacy_cfy)
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return 2
    res, info = r["res"], r["info"]
    out_dir = paths.processed / run.key / "20_calibration_check"   # not the tracked stage
    save_calibration_h5(out_dir / "spectra.h5", res,
                        {"run": run.key, "baseline_method": info["baseline_method"]})
    (out_dir / "calibration.json").write_text(json.dumps(info, indent=2, default=str),
                                              encoding="utf-8")
    B = info["fc_baseline_V"]
    print(f"{run.key} ({run.label}): {info['n_output_rows']} peak rows, {info['pairs']} "
          f"(m, z) spectra, {len(info['fc_files'])} FC traces")
    print(f"  FC baseline {B * 1e3:.4f} mV ({info['baseline_method']}); FC peak "
          f"{info['fc_peak_V'] * 1e3:.4f} mV; CFY_1m = {info['CFY_1m']:.6e}")
    lc = info.get("legacy_comparison")
    if lc:
        print(f"  legacy checkpoint: max relative difference {lc['max_rel_diff']:.1e} -> "
              f"{'IDENTICAL (< 1e-9)' if lc['identical_1e-9'] else 'DIFFERENT'}")
        for k, v in lc["per_key"].items():
            if not isinstance(v, float) or v >= 1e-9:
                print(f"    {k}: {v}")
    print(f"  written: {out_dir / 'spectra.h5'}, calibration.json (a check: the "
          f"tracked stage is 'eiakit run --stage calibration')")
    return 0 if not lc or lc["identical_1e-9"] else 1


def cmd_peaks(args) -> int:
    import pandas as pd

    from .peaks import load_peak_state, replay_run, update_peak_state

    db = _load_db()
    paths = get_paths()
    run = db.run(args.key)
    out = paths.processed / run.key / "10_peaks"
    out.mkdir(parents=True, exist_ok=True)
    state = load_peak_state(db, run.key)
    if state and not args.search:
        print(f"using the stored GUI state for {len(state)} voltages "
              f"(db/legacy_peaks/{run.key}.toml; --search to ignore it)")
    res = replay_run(db, run, paths.data, voltages=args.U, max_clicks=args.max_clicks,
                     deep=args.deep, deep_lof=args.deep_lof, log=print, verbose=args.verbose,
                     state=state, search=args.search)
    if args.write_state:
        p, n = update_peak_state(db, run.key, res["summary"])
        print(f"state: {n} voltage(s) added/improved -> {p}")
    pd.DataFrame(res["summary"]).to_csv(out / "replay_summary.csv", index=False)
    if res["rows"]:
        pd.concat(res["rows"]).to_csv(out / "replay_rows.csv", index=False)
    c = res["counts"]
    print(f"{run.key}: {c['exact']} exact, {c['equivalent']} equivalent, {c['partial']} partial "
          f"({c['known_partial']} known), {c['skipped']} skipped -> {out}")
    return 0 if c["partial"] == c["known_partial"] else 1


def cmd_cbs(args) -> int:
    import json

    from .cbs import replay_run_cbs

    db = _load_db()
    paths = get_paths()
    keys = args.keys or [r.key for r in db.runs()]
    worst = 0
    for key in keys:
        run = db.run(key)
        res = replay_run_cbs(db, run, paths.data)
        if res["status"] == "missing files":
            print(f"{key}: missing files {res['files']}")
            worst = max(worst, 2)
            continue
        s = res["summary"]
        out = paths.processed / run.key / "30_cbs"
        out.mkdir(parents=True, exist_ok=True)
        res["components"].to_csv(out / "replay_components.csv", index=False)
        res["areas_cmp"].drop(columns="_merge", errors="ignore").to_csv(
            out / "replay_areas.csv", index=False)
        res["totals"].to_csv(out / "replay_totals.csv", index=False)
        (out / "replay_summary.json").write_text(json.dumps(
            {"status": res["status"], "files": res["files"], **s}, indent=2, default=str))
        print(f"{key}: {res['status']}")
        print(f"   {s['n_components']} components ({s['profile']}, M = {s['M_ION_kg']:.6e} kg, "
              f"L = {s['L_EIA_m']} m); data areas x {s.get('data_area_scale', float('nan')):.7f}"
              f" (spread {s.get('data_area_spread', float('nan')):.1e}); model areas "
              f"{s.get('model_area_over_bound', float('nan')):.2f} x rounding bound; FWHM "
              f"{s['fwhm_over_bound']:.2f} x bound; t_peak_fit <= "
              f"{s['t_peak_fit_max_steps']:.0f} grid steps")
        print(f"   origin of kT {s['kT_origin']}, of A {s['A_origin']}  -> {out}")
        if not res["status"].startswith(("reproduced", "model reproduced")):
            worst = max(worst, 1)
    return worst


def _select_runs(db, keys):
    return [db.run(k) for k in keys] if keys else list(db.runs())


def cmd_status(args) -> int:
    from .pipeline import ORDER, STAGES, stage_status

    db = _load_db()
    paths = get_paths()
    stages = args.stage or ORDER
    mark = {"ok": "ok", "missing": "--", "failed": "FAILED", "stale": "STALE"}
    w = max(len(r.key) for r in _select_runs(db, args.keys)) + 2
    print(f"{'run':<{w}}" + "".join(f"{s:<14}" for s in stages))
    details, n_bad = [], 0
    for run in _select_runs(db, args.keys):
        cache: dict = {}
        line = f"{run.key:<{w}}"
        for name in stages:
            st = stage_status(db, run, name, paths.data, paths.processed, cache)
            line += f"{mark[st['state']]:<14}"
            if st["state"] != "ok":
                n_bad += 1
                if st["state"] != "missing" or args.verbose:
                    details.append((run.key, name, st["reasons"]))
            elif args.verbose and st["manifest"]:
                m = st["manifest"]
                details.append((run.key, name, [f"run {m['created']} (git {m.get('git')}), "
                                                f"{len(m['inputs'])} inputs, summary "
                                                f"{m.get('summary')}"]))
        if run.provisional:
            line += "  provisional: " + ", ".join(run.provisional)
        print(line)
    for key, name, reasons in details:
        print(f"\n{key} / {name} ({STAGES[name].folder}):")
        for r in reasons:
            print(f"   - {r}")
    if n_bad:
        print(f"\n{n_bad} stage(s) to run: eiakit run" + (" " + " ".join(
            f'"{k}"' for k in args.keys) if args.keys else ""))
    return 0


def cmd_run(args) -> int:
    from .pipeline import plan, run_stage

    db = _load_db()
    paths = get_paths()
    todo = plan(db, _select_runs(db, args.keys), paths.data, paths.processed,
                stages=args.stage, force=args.force)
    if not todo:
        print("everything is up to date")
        return 0
    for run, name, st in todo:
        why = "; ".join(st["reasons"][:2]) if st["reasons"] else "upstream rerun"
        print(f"{run.key} / {name}: {st['state']} ({why})")
    if args.dry_run:
        return 0
    from .pipeline import STAGES
    worst, failed = 0, set()
    for run, name, _ in todo:
        blocked = [d for d in STAGES[name].depends if (run.key, d) in failed]
        if blocked:
            print(f"  {run.key} / {name}: skipped (upstream {', '.join(blocked)} failed)")
            failed.add((run.key, name))
            continue
        t = time.time()
        man = run_stage(db, run, name, paths.data, paths.processed)
        flag = "ok" if man["ok"] else "FAILED"
        summ = {k: v for k, v in man["summary"].items() if k != "traceback"}
        print(f"  {run.key} / {name}: {flag} in {time.time() - t:.1f} s  {summ}")
        if not man["ok"]:
            worst = 1
            failed.add((run.key, name))
    return worst


def cmd_study(args) -> int:
    from . import study as S

    db = _load_db()
    paths = get_paths()
    if args.name not in db.studies:
        print(f"ERROR: no study {args.name!r}; known: {', '.join(sorted(db.studies))} "
              f"(db/studies/<name>.toml)")
        return 2
    parts = args.part or ["powerlaw", "cbs", "reconstruction", "collapse"]
    if args.status:
        for part in parts:
            st = S.study_status(db, args.name, part, paths.data, paths.processed, paths.studies)
            print(f"{args.name} / {part}: {st['state']}")
            for r in st["reasons"]:
                print(f"   - {r}")
        return 0
    fmts = () if args.html_only else ("pdf", "png")
    for part in parts:
        print(f"{args.name} / {part} (source: {args.source})")
        res = getattr(S, part)(db, args.name, paths.data, paths.processed, paths.studies,
                               source=args.source, formats=fmts)
        print(f"  -> {res['out']}")
    return 0


def cmd_snapshot(args) -> int:
    from .paper import write_snapshot

    db = _load_db()
    paths = get_paths()
    if args.name not in db.studies:
        print(f"ERROR: no study {args.name!r}")
        return 2
    p = write_snapshot(db, args.name, args.label, paths.data, paths.processed, paths.studies,
                       args.source, args.note or "")
    print(f"snapshot {args.label!r}: {sum(1 for _ in open(p)) - 1} numbers "
          f"(corrections {db.corrections()}) -> {p}")
    return 0


def cmd_diff(args) -> int:
    import pandas as pd

    from .paper import changelog_entry, diff

    paths = get_paths()
    res = diff(paths.studies, args.name, args.a, args.b)
    s = res["summary"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(s.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    out = paths.studies / args.name / "diffs"
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / f"{args.a}__{args.b}.csv", index=False)
    switches = [k for k in ("corrections", "calibration")
                if res["meta_a"].get(k) != res["meta_b"].get(k)]
    if args.changelog and not switches and not (s.changed > 0).any() and not args.force:
        print(f"\nNOT written to changelog.md: '{args.a}' and '{args.b}' have the same "
              f"switches ({res['meta_b'].get('corrections')}) and the same numbers.\n"
              "Probably the switch was set before the first snapshot: set it back, "
              "eiakit run, snapshot again (same label overwrites). --force to write anyway.")
        return 1
    if args.changelog:
        entry = changelog_entry(res, args.a, args.b, args.changelog, args.text or "")
        print(f"changelog entry appended -> {_append_changelog(paths, args.name, entry)}")
    print(f"full table -> {out / f'{args.a}__{args.b}.csv'}")
    return 0


def cmd_import_exclusions(args) -> int:
    db = _load_db()
    excl = import_exclusion_list(Path(args.file), db, args.study)
    print(f"# from {args.file}  ->  replace the [exclude] table at the end of "
          f"db/studies/{args.study}.toml with:\n")
    print(format_exclude_toml(excl))
    return 0


def cmd_methods(args) -> int:
    from .methods import run_methods
    db = _load_db()
    paths = get_paths()
    if args.name not in db.studies:
        print(f"ERROR: no study {args.name!r}")
        return 2
    print(f"{args.name} / methods (source: {args.source})")
    r = run_methods(db, args.name, paths.data, paths.processed, paths.studies, args.only,
                    args.source, () if args.html_only else ("pdf", "png"), args.boot)
    print(f"-> {r['out'] / 'methods.html'}")
    return 0


def _append_changelog(paths, name: str, entry: str):
    log = paths.studies / name / "changelog.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.is_file():
        log.write_text(f"# {name}: changes to the paper numbers\n\n"
                       "Each section: one correction, the switches that changed, and "
                       "the paper numbers it moved (eiakit diff).\n\n", encoding="utf-8")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(entry)
    return log


def cmd_note(args) -> int:
    """A changelog section without numbers (a correction checked to change nothing)."""
    import time as _t
    paths = get_paths()
    entry = (f"## {args.title}\n\n*{_t.strftime('%Y-%m-%d %H:%M')}* — note, no paper "
             f"number changed\n\n{args.text or ''}\n\n")
    print(f"note appended -> {_append_changelog(paths, args.name, entry)}")
    return 0


def cmd_compare(args) -> int:
    """Run the pipeline and snapshot the study with KEY = A, then KEY = B, then diff."""
    global _OVERRIDES
    sec, key, _ = _parse_set(f"{args.key}=x")
    base = _OVERRIDES
    labels = []
    for val in (args.a, args.b):
        _OVERRIDES = [*base, (sec, key, _parse_value(val))]
        lab = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{key}-{val}")
        print(f"\n=== {sec}.{key} = {val}  (snapshot '{lab}')")
        rc = cmd_run(argparse.Namespace(keys=[], stage=None, force=False, dry_run=False))
        if rc:
            print("pipeline failed: stopping")
            return rc
        rc = cmd_snapshot(argparse.Namespace(name=args.name, label=lab, source="pipeline",
                                             note=f"eiakit compare {args.key} {args.a} {args.b}"))
        if rc:
            return rc
        labels.append(lab)
    _OVERRIDES = base
    print("\n=== diff")
    rc = cmd_diff(argparse.Namespace(name=args.name, a=labels[0], b=labels[1],
                                     changelog=args.changelog, text=args.text, force=False))
    print("\n=== back to db/pipeline.toml as it is")
    cmd_run(argparse.Namespace(keys=[], stage=None, force=False, dry_run=False))
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eiakit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a db/pipeline.toml setting for this command only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="environment, paths, run folders").set_defaults(func=cmd_info)
    p = sub.add_parser("checksum", help="SHA-256 manifest of a folder tree")
    p.add_argument("--root", help="tree to hash (default: EIA_DATA_ROOT)")
    p.add_argument("--out", help="output CSV")
    p.add_argument("--skip-variants", action="store_true",
                   help="do not hash variant folders (Output~, EIA-Results_ ...)")
    p.set_defaults(func=cmd_checksum)
    p = sub.add_parser("checksum-compare", help="compare two manifests")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_checksum_compare)
    p = sub.add_parser("runs", help="runs of the database")
    p.add_argument("--all", action="store_true", help="also runs with include = false")
    p.set_defaults(func=cmd_runs)
    sub.add_parser("db-check", help="validate the database").set_defaults(func=cmd_db_check)
    p = sub.add_parser("files", help="raw files picked up by a run's acquisition format")
    p.add_argument("key", help="run key (data folder name)")
    p.add_argument("--list", action="store_true", help="list every voltage and skipped file")
    p.set_defaults(func=cmd_files)
    p = sub.add_parser("audit", help="audit of the legacy results of one or more runs")
    p.add_argument("keys", nargs="*", help="run keys (default: every included run)")
    p.add_argument("--study", default="PRL_AlNb", help="study whose exclusions are listed")
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser("calibrate", help="calibration stage (untitled27)")
    p.add_argument("key")
    p.add_argument("--fc", choices=["start", "stop", "both"], default="start",
                   help="FC traces to average (default: 1_FC_Start, as untitled27)")
    p.add_argument("--baseline", default="auto",
                   help="FC pedestal: auto (from the legacy checkpoint), pretrigger, or volts")
    p.add_argument("--legacy-cfy", action="store_true",
                   help="use CFY_1m of the legacy checkpoint instead of the FC")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("peaks", help="peak stage (untitled28): replay the legacy fits")
    p.add_argument("key")
    p.add_argument("--U", type=float, nargs="+", help="only these file voltages")
    p.add_argument("--max-clicks", type=int, default=None,
                   help="search the U arrow clicks within +/- N (default from pipeline.toml)")
    p.add_argument("--deep", action="store_true",
                   help="also search LOF arrow clicks and Corr (slow; for 'partial' voltages)")
    p.add_argument("--deep-lof", type=int, default=3, help="LOF clicks searched with --deep")
    p.add_argument("-v", "--verbose", action="store_true", help="also list exact voltages")
    p.add_argument("--write-state", action="store_true",
                   help="merge the recovered GUI state into db/legacy_peaks/KEY.toml")
    p.add_argument("--search", action="store_true", help="ignore the stored state and search")
    p.set_defaults(func=cmd_peaks)
    p = sub.add_parser("cbs", help="CBS stage (cbs_manual_fitter_v12): replay the legacy fits")
    p.add_argument("keys", nargs="*", help="run keys (default: all included runs)")
    p.set_defaults(func=cmd_cbs)
    p = sub.add_parser("status", help="state of every stage of every run (ok / stale / ...)")
    p.add_argument("keys", nargs="*")
    p.add_argument("--stage", nargs="+", choices=["audit", "peaks", "calibration", "cbs", "report"])
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("run", help="run the stages that are missing or stale")
    p.add_argument("keys", nargs="*")
    p.add_argument("--stage", nargs="+", choices=["audit", "peaks", "calibration", "cbs", "report"])
    p.add_argument("--force", action="store_true", help="rerun even if up to date")
    p.add_argument("--dry-run", action="store_true", help="only list what would run")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("study", help="cross-run figures and tables of a study")
    p.add_argument("name", help="study name (db/studies/<name>.toml)")
    p.add_argument("--part", nargs="+", choices=["powerlaw", "cbs", "reconstruction", "collapse"])
    p.add_argument("--source", choices=["pipeline", "legacy"], default="pipeline",
                   help="pipeline: calibration stage outputs (default); legacy: checkpoints")
    p.add_argument("--html-only", action="store_true", help="skip PDF / PNG")
    p.add_argument("--status", action="store_true", help="only say whether it is up to date")
    p.set_defaults(func=cmd_study)
    p = sub.add_parser("snapshot", help="save the paper numbers of a study")
    p.add_argument("name")
    p.add_argument("--label", required=True)
    p.add_argument("--source", choices=["pipeline", "legacy"], default="pipeline")
    p.add_argument("--note")
    p.set_defaults(func=cmd_snapshot)
    p = sub.add_parser("diff", help="compare two snapshots; optionally write a changelog entry")
    p.add_argument("name")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--changelog", metavar="TITLE", help="append a section to changelog.md")
    p.add_argument("--text", help="explanation for the changelog entry")
    p.add_argument("--force", action="store_true",
                   help="write the changelog entry even if nothing changed")
    p.set_defaults(func=cmd_diff)
    p = sub.add_parser("methods", help="method studies (Step 12): zbar, calibration, ...")
    p.add_argument("name")
    p.add_argument("--only", nargs="+",
                   choices=["zbar", "calibration", "sensitivity", "fc", "shots"])
    p.add_argument("--boot", type=int, default=30, help="bootstrap samples (shots)")
    p.add_argument("--source", choices=["pipeline", "legacy"], default="pipeline")
    p.add_argument("--html-only", action="store_true")
    p.set_defaults(func=cmd_methods)
    p = sub.add_parser("note", help="append a changelog section without numbers")
    p.add_argument("name")
    p.add_argument("title")
    p.add_argument("--text")
    p.set_defaults(func=cmd_note)
    p = sub.add_parser("compare", help="effect of one setting: run + snapshot with A, then B, diff")
    p.add_argument("name", help="study, e.g. PRL_AlNb")
    p.add_argument("key", help="setting of db/pipeline.toml, e.g. calibration.output_duplicates")
    p.add_argument("a", help="value before (e.g. the legacy one)")
    p.add_argument("b", help="value after")
    p.add_argument("--changelog", metavar="TITLE", help="append the entry to changelog.md")
    p.add_argument("--text", help="explanation for the changelog entry")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("import-exclusions", help="legacy ExclusionList.txt -> study [exclude]")
    p.add_argument("file")
    p.add_argument("--study", default="PRL_AlNb")
    p.set_defaults(func=cmd_import_exclusions)
    args = ap.parse_args(argv)
    try:
        global _OVERRIDES
        _OVERRIDES = [_parse_set(x) for x in args.set]
        return args.func(args)
    except (PathsNotConfigured, DatabaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
