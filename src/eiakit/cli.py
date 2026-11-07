"""
Command line entry point:  ``eiakit <command>``

    eiakit info                     environment, paths, packages, run folders found
    eiakit checksum [--root DIR] [--out FILE] [--skip-variants]
                                    SHA-256 manifest of a tree (default: the data root)
    eiakit checksum-compare A B     compare two manifests (exit code 1 if they differ)
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .checksum import build_manifest, compare_manifests, read_manifest, write_manifest
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
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except PathsNotConfigured as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
