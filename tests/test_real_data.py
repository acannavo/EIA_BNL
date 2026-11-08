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


@pytest.mark.parametrize("key", [k for k in RUNS if (ROOT / k / "2_SEM" / "Output").is_dir()])
def test_checkpoint_rebuilt_from_output(key):
    """untitled27 port: the legacy checkpoint is rebuilt from 2_SEM/Output (CFY taken from it)."""
    from eiakit.calibration import calibrate, compare_checkpoints
    from eiakit.legacy import load_checkpoint, read_output_folder
    run = db.run(key)
    old = load_checkpoint(ROOT / key / "EIA-Results" / "pipeline_checkpoint.npy")
    rows = read_output_folder(ROOT / key / "2_SEM" / "Output")
    new = calibrate(rows, None, 0.0, run.eia, run.fc, db.pipeline["calibration"],
                    cfy_1m_override=old["CFY_1m"])
    d = compare_checkpoints(new, old)
    bad = {k: v for k, v in d.items() if not isinstance(v, float) or v > 1e-9}
    assert not bad, bad


SLOW = os.environ.get("EIA_SLOW") == "1"


@pytest.mark.skipif(not SLOW, reason="slow (minutes): set EIA_SLOW=1")
@pytest.mark.parametrize("key", [k for k in RUNS
                                 if (ROOT / k / "2_SEM" / "Output_detail").is_dir()])
def test_legacy_peak_fits_are_replayed(key):
    """untitled28 port: every Output_detail row refitted (exact or within curve_fit noise)."""
    from eiakit.peaks import replay_run
    res = replay_run(db, db.run(key), ROOT)
    partial = [s["U_file"] for s in res["summary"] if s.get("status") == "partial"]
    assert not partial, f"voltages not reproduced: {partial} (run eiakit peaks {key})"


@pytest.mark.parametrize("key", [k for k in RUNS
                                 if (ROOT / k / "2_SEM" / "Output_detail").is_dir()])
def test_legacy_peak_fits_with_stored_state(key):
    """Refit with the GUI state in db/legacy_peaks/<key>.toml: nothing may get worse."""
    from eiakit.peaks import _ukey, load_peak_state, replay_run
    state = load_peak_state(db, key)
    if not state:
        pytest.skip(f"no db/legacy_peaks/{key}.toml (eiakit peaks {key} --write-state)")
    Us = [float(k) for k in state]
    res = replay_run(db, db.run(key), ROOT, voltages=Us, state=state)
    worse = []
    for s in res["summary"]:
        if s.get("status") in ("no raw files", "no Output_detail"):
            continue                                  # data not present on this machine
        old = state[_ukey(s["U_file"])]
        if s.get("n_match", 0) < old["n_match"] or s.get("n_exact", 0) < old["n_exact"]:
            worse.append((s["U_file"], old["status"], s.get("status")))
    assert not worse, worse
