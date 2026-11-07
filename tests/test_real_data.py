"""
Checks on the real legacy results. Skipped unless EIA_DATA_ROOT points to the data folder
(run them in VS Code after  . .\\env.ps1 ).
"""
import os
from pathlib import Path

import pytest

from eiakit.audit import audit_run
from eiakit.db import load_database

ROOT = Path(os.environ.get("EIA_DATA_ROOT", "/nonexistent"))
db = load_database()
RUNS = [r.key for r in db.runs() if (ROOT / r.key / "EIA-Results").is_dir()]

pytestmark = pytest.mark.skipif(not RUNS, reason="no data (EIA_DATA_ROOT not set or empty)")


@pytest.mark.parametrize("key", RUNS)
def test_legacy_fits_reproduce_and_labels_are_consistent(key):
    rep = audit_run(db, db.run(key), ROOT)
    errors = {f.code for f in rep.findings if f.level == "error"}
    assert "CBS-REPRO" not in errors, "stored kT/V0/A do not reproduce the stored peaks"
    assert "N3" not in errors, "fast/slow labels of runs.toml contradict the arrival times"
    assert "CK-GEOM" not in errors and "CK-CHAIN" not in errors
    assert "I1" not in errors
