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
    eiakit import-exclusions FILE [--study NAME]
                                    legacy ExclusionList.txt -> the study's [exclude] table
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import sys
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


def cmd_runs(args) -> int:
    db = load_database()
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
    db = load_database(strict=False)
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
    db = load_database()
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
    db = load_database()
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

    import numpy as np

    from .acquisition import read_trace, scan_folder
    from .calibration import calibrate, compare_checkpoints, recover_fc_baseline
    from .legacy import load_checkpoint, read_output_folder
    from .store import save_calibration_h5

    db = load_database()
    paths = get_paths()
    run = db.run(args.key)
    folder = run.folder(paths.data)
    pipe = db.pipeline["calibration"]
    rows = read_output_folder(folder / "2_SEM" / "Output")
    if rows.empty:
        print(f"ERROR: no peak table in {folder / '2_SEM' / 'Output'}")
        return 2
    fmt = run.acquisition.fc
    dirs = {"start": [fmt.folder], "stop": [fmt.folder_stop],
            "both": [fmt.folder, fmt.folder_stop]}[args.fc]
    traces, used = [], []
    for d in dirs:
        if d is None:
            continue
        for p, _, _ in scan_folder(folder / d, fmt).matched:
            traces.append(read_trace(p, fmt))
            used.append(f"{d}/{p.name}")
    ck_path = folder / "EIA-Results" / "pipeline_checkpoint.npy"
    old = load_checkpoint(ck_path) if ck_path.is_file() else None

    cfy = None
    if args.legacy_cfy:
        if old is None:
            print("ERROR: --legacy-cfy needs the legacy checkpoint")
            return 2
        cfy, B, how = old["CFY_1m"], 0.0, "CFY_1m taken from the legacy checkpoint"
    elif args.baseline == "auto":
        if old is None or not traces:
            print("ERROR: --baseline auto needs the legacy checkpoint and FC traces")
            return 2
        B = recover_fc_baseline(old, traces, int(pipe["fc_grid_points"]), run.eia, run.fc)
        how = "recovered from the legacy checkpoint (N7)"
    elif args.baseline == "pretrigger":
        pre = np.concatenate([v[t < 0] for t, v in traces])
        B, how = float(np.mean(pre)), f"mean of the FC signal before t = 0 ({pre.size} samples)"
    else:
        B, how = float(args.baseline), "given on the command line"
    if not traces and cfy is None:
        print(f"ERROR: no FC traces found in {dirs}")
        return 2

    res = calibrate(rows, traces, B, run.eia, run.fc, pipe, cfy_1m_override=cfy)
    out_dir = paths.processed / run.key / "20_calibration"
    info = {"run": run.key, "fc_files": used, "fc_baseline_V": B, "baseline_method": how,
            "fc_peak_V": res["fc_peak_V"], "CFY_1m": res["CFY_1m"], "CFY_EIA": res["CFY_EIA"],
            "n_output_rows": len(rows), "pairs": len(res["corrected_spectra"]),
            "settings": {k: pipe[k] for k in ("n_interp", "fc_grid_points", "method",
                                              "interp_fill")}}
    if old is not None:
        diff = compare_checkpoints(res, old)
        num = [v for v in diff.values() if isinstance(v, float)]
        info["legacy_comparison"] = {"max_rel_diff": max(num), "per_key": diff,
                                     "identical_1e-9": all(isinstance(v, float) and v < 1e-9
                                                           for v in diff.values())}
    save_calibration_h5(out_dir / "spectra.h5", res, {"run": run.key, "baseline_method": how})
    (out_dir / "calibration.json").write_text(json.dumps(info, indent=2, default=str),
                                              encoding="utf-8")
    print(f"{run.key} ({run.label}): {len(rows)} peak rows, {len(res['corrected_spectra'])} "
          f"(m, z) spectra, {len(traces)} FC traces")
    print(f"  FC baseline {B * 1e3:.4f} mV ({how}); FC peak {res['fc_peak_V'] * 1e3:.4f} mV; "
          f"CFY_1m = {res['CFY_1m']:.6e}")
    if old is not None:
        lc = info["legacy_comparison"]
        print(f"  legacy checkpoint: max relative difference {lc['max_rel_diff']:.1e} -> "
              f"{'IDENTICAL (< 1e-9)' if lc['identical_1e-9'] else 'DIFFERENT'}")
        for k, v in lc["per_key"].items():
            if not isinstance(v, float) or v >= 1e-9:
                print(f"    {k}: {v}")
    print(f"  written: {out_dir / 'spectra.h5'}, calibration.json")
    return 0 if old is None or info["legacy_comparison"]["identical_1e-9"] else 1


def cmd_peaks(args) -> int:
    import pandas as pd

    from .peaks import load_peak_state, replay_run, update_peak_state

    db = load_database()
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

    db = load_database()
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
        if not res["status"].startswith("reproduced"):
            worst = max(worst, 1)
    return worst


def cmd_import_exclusions(args) -> int:
    db = load_database()
    excl = import_exclusion_list(Path(args.file), db, args.study)
    print(f"# from {args.file}  ->  replace the [exclude] table at the end of "
          f"db/studies/{args.study}.toml with:\n")
    print(format_exclude_toml(excl))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eiakit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
    p = sub.add_parser("import-exclusions", help="legacy ExclusionList.txt -> study [exclude]")
    p.add_argument("file")
    p.add_argument("--study", default="PRL_AlNb")
    p.set_defaults(func=cmd_import_exclusions)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (PathsNotConfigured, DatabaseError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
