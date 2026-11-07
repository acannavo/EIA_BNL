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


def _peak_cases():
    from eiakit.acquisition import scan_folder
    out = []
    for r in db.runs():
        sem = ROOT / r.key / r.acquisition.sem.folder
        if not (sem / "Output_detail").is_dir():
            continue
        volts = {u for _, u, _ in scan_folder(sem, r.acquisition.sem).matched}
        out += [(r.key, u) for u in sorted(volts)]
    return out


@pytest.mark.parametrize("key,U", _peak_cases())
def test_legacy_peak_fits_are_replayed_exactly(key, U):
    """untitled28 port: every Output_detail row is refitted bit for bit (U clicks recovered)."""
    from eiakit.acquisition import read_trace, scan_folder
    from eiakit.legacy import read_output_detail_folder
    from eiakit.peaks import replay_voltage
    run = db.run(key)
    fmt = run.acquisition.sem
    sem = ROOT / key / fmt.folder
    detail = read_output_detail_folder(sem / "Output_detail")
    d = detail[detail.U == U]
    if d.empty:
        pytest.skip("no Output_detail for this voltage")
    traces = {rep: read_trace(p, fmt) for p, u, rep in scan_folder(sem, fmt).matched if u == U}
    settings = dict(db.pipeline["peaks"])
    masses = {int(k): float(v) for k, v in settings.pop("gui_masses").items()}
    r = replay_voltage(traces, U, d, masses, run.eia.L_m, run.eia,
                       db.profile(run.legacy.profile_peaks), settings)
    assert r["status"] == "exact", r["comparison"][~r["comparison"].match]
