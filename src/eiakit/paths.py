"""
Locations of the data, processed results and studies.

They come from environment variables, set by ``env.ps1`` (terminal) and by the
``.env`` file it writes (VS Code run/debug and tests):

    EIA_DATA_ROOT       read-only raw data + legacy results  (NewEIASoftware/data)
    EIA_PROCESSED_ROOT  everything the software writes per run
    EIA_STUDIES_ROOT    cross-run comparisons (figures, dashboards)

No path is hard-coded in the package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_VARS = ("EIA_DATA_ROOT", "EIA_PROCESSED_ROOT", "EIA_STUDIES_ROOT")


class PathsNotConfigured(RuntimeError):
    """Raised when a required environment variable is missing."""


@dataclass(frozen=True)
class Paths:
    data: Path
    processed: Path
    studies: Path

    @property
    def fits(self) -> Path:
        """Hand-made fits (CBS fitter, Step 9): EIA_FITS_ROOT, default <processed>/../fits.
        Kept outside processed/ so that clearing the pipeline outputs never deletes them."""
        v = os.environ.get("EIA_FITS_ROOT")
        return Path(v).expanduser() if v else self.processed.parent / "fits"

    def check(self) -> dict[str, bool]:
        """Existence of each root (data must exist; the others are created on demand)."""
        return {"data": self.data.is_dir(),
                "processed": self.processed.is_dir(),
                "studies": self.studies.is_dir()}

    def ensure_output_dirs(self) -> None:
        """Create processed/ and studies/ if missing. data/ is never created or written."""
        self.processed.mkdir(parents=True, exist_ok=True)
        self.studies.mkdir(parents=True, exist_ok=True)


def _env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise PathsNotConfigured(
            f"environment variable {name} is not set. In PowerShell run  . .\\env.ps1  "
            f"from the repository folder (or select the 'EIA PowerShell' terminal in VS Code).")
    return Path(value).expanduser()


def get_paths() -> Paths:
    """Paths from the environment. Raises PathsNotConfigured if a variable is missing."""
    paths = Paths(data=_env("EIA_DATA_ROOT"),
                  processed=_env("EIA_PROCESSED_ROOT"),
                  studies=_env("EIA_STUDIES_ROOT"))
    if _is_inside(paths.processed, paths.data) or _is_inside(paths.studies, paths.data):
        raise PathsNotConfigured(
            "processed/ and studies/ must not be inside the read-only data/ folder")
    return paths


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
