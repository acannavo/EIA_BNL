"""
Layout of the data folder and the rule that separates canonical folders from
temporary variants.

Every run lives in one top-level folder of the data root (``20250513``,
``2026_1103 Nb analysis``, ...). Inside it, the canonical sub-folders are::

    0_FocalPosition | 0_FocalSearch
    1_FC_Start
    2_SEM/            raw SEM CSVs
        Output/        legacy peak table per deflector voltage
        Output_detail/
    3_FC_Stop
    EIA-Results/      legacy checkpoint + CBS fit files
        QC_Interp/

Variants of a canonical name (``Output~``, ``Output_``, ``Output_0``,
``Output_detail~``, ``EIA-Results_b``, ``0_FocalPosition_1``, ...) and anything
with ``wrong`` in its name are temporary folders with wrong results: they are
ignored together with their whole subtree. Other folders (``LowStateCharge``,
``4_MultiShotSameSpot``, ``Full``, ``Miscellaneous``, ``Data Backup``, ...) are
'extra': kept, listed, not used by the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

CANONICAL = (
    "0_FocalPosition", "0_FocalSearch", "1_FC_Start", "2_SEM", "3_FC_Stop",
    "Output", "Output_detail", "EIA-Results", "QC_Interp",
)
# longest first, so 'Output_detail_' is matched against 'Output_detail', not 'Output'
_BASES = sorted(CANONICAL, key=len, reverse=True)
_VARIANT_SUFFIX = re.compile(r"^(~+|_[0-9A-Za-z]{0,2}~*)$")


class Kind(StrEnum):
    CANONICAL = "canonical"
    VARIANT = "variant"      # temporary copy with wrong results: ignored
    EXTRA = "extra"          # other data, kept but not used yet


def classify(name: str) -> Kind:
    """Kind of a folder name (see module docstring)."""
    if name in CANONICAL:
        return Kind.CANONICAL
    if "wrong" in name.lower():
        return Kind.VARIANT
    for base in _BASES:
        if name.startswith(base) and _VARIANT_SUFFIX.match(name[len(base):]):
            return Kind.VARIANT
    return Kind.EXTRA


@dataclass
class RunFolder:
    """One top-level run folder of the data root."""

    path: Path
    canonical: dict[str, Path] = field(default_factory=dict)   # relative name -> path
    variants: list[Path] = field(default_factory=list)
    extras: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    def has(self, rel: str) -> bool:
        return rel in self.canonical


def scan_run(path: Path) -> RunFolder:
    """Walk one run folder, classifying every directory; variant subtrees are not entered."""
    run = RunFolder(path=path)
    stack = [(path, "")]
    while stack:
        folder, rel = stack.pop()
        for child in sorted(p for p in folder.iterdir() if p.is_dir()):
            kind = classify(child.name)
            child_rel = f"{rel}/{child.name}" if rel else child.name
            if kind is Kind.VARIANT:
                run.variants.append(child)
                continue
            if kind is Kind.CANONICAL:
                run.canonical[child_rel] = child
            else:
                run.extras.append(child)
            stack.append((child, child_rel))
    return run


def discover_runs(data_root: Path) -> list[RunFolder]:
    """Every top-level folder of the data root that contains a 2_SEM folder."""
    return [scan_run(p) for p in sorted(data_root.iterdir())
            if p.is_dir() and (p / "2_SEM").is_dir()]


def iter_files(root: Path, skip_variants: bool = True):
    """All files below root, optionally skipping variant subtrees. Yields Paths."""
    stack = [root]
    while stack:
        folder = stack.pop()
        for child in sorted(folder.iterdir()):
            if child.is_dir():
                if skip_variants and classify(child.name) is Kind.VARIANT:
                    continue
                stack.append(child)
            elif child.is_file():
                yield child
