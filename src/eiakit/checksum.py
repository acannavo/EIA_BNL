"""
SHA-256 manifest of a folder tree, and comparison of two manifests.

Used in Step 0 to prove that NewEIASoftware/data is an exact copy of the
original tree, and later to detect any change in the read-only data.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .layout import iter_files

FIELDS = ("relpath", "size", "sha256")
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, skip_variants: bool = False, progress=None) -> list[dict]:
    """One row per file: path relative to root (forward slashes), size, sha256."""
    rows = []
    for i, f in enumerate(iter_files(root, skip_variants=skip_variants), 1):
        rows.append({"relpath": f.relative_to(root).as_posix(),
                     "size": f.stat().st_size,
                     "sha256": sha256_file(f)})
        if progress and i % 500 == 0:
            progress(i)
    rows.sort(key=lambda r: r["relpath"])
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["relpath"]: r for r in csv.DictReader(fh)}


@dataclass
class ManifestDiff:
    only_a: list[str]
    only_b: list[str]
    different: list[str]
    n_same: int

    @property
    def identical(self) -> bool:
        return not (self.only_a or self.only_b or self.different)


def compare_manifests(a: dict[str, dict], b: dict[str, dict]) -> ManifestDiff:
    ka, kb = set(a), set(b)
    common = ka & kb
    diff = sorted(k for k in common if a[k]["sha256"] != b[k]["sha256"])
    return ManifestDiff(only_a=sorted(ka - kb), only_b=sorted(kb - ka),
                        different=diff, n_same=len(common) - len(diff))
