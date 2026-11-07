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
