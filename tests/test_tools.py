"""Step 9 tools on the real data: peak picker, baseline picker (skipped without data)."""
import os
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("EIA_DATA_ROOT", "/nonexistent"))
pytestmark = pytest.mark.skipif(not (ROOT / "20250611" / "2_SEM").is_dir(),
                                reason="needs 20250611 raw SEM / FC files")


def test_peak_picker_and_baseline_merge_into_calibration(tmp_path, monkeypatch):
    from eiakit.calibration import prepare_rows, run_calibration
    from eiakit.db import load_database
    from eiakit.peaks_edit import BaselineSession, PeakSession
    monkeypatch.setenv("EIA_FITS_ROOT", str(tmp_path))
    monkeypatch.setenv("EIA_PROCESSED_ROOT", str(tmp_path / "p"))
    monkeypatch.setenv("EIA_STUDIES_ROOT", str(tmp_path / "s"))
    db = load_database()
    run = db.run("20250611")
    assert 35.0 in PeakSession.voltages(run, ROOT)
    s = PeakSession.load(db, run, ROOT, 35.0)
    assert len(s.fits) == len(s.legacy) and s.fits.success.all()
    # amplitudes as in Output_detail (same shots); times differ by the N16 window shift
    m = s.fits.merge(s.legacy, on=["m", "z"], suffixes=("", "_leg"))
    assert (abs(m.TOF / m.TOF_leg - 1) < 0.02).all()
    s.exclude(len(s.fits) - 1)
    s.save(tmp_path)
    pr = prepare_rows(db, run, ROOT)
    r = pr["rows"][pr["rows"].U == 35.0]
    assert pr["peak_overrides"] == [35.0] and len(r) == len(s.fits) - 1
    assert set(r.file) == {"fits:p35.txt"}
    b = BaselineSession.load(db, run, ROOT, tmp_path)
    assert abs(b.legacy - 4.636e-4) < 2e-6 and abs(b.pretrigger - 4.303e-4) < 2e-6
    b.save(tmp_path, 4.5e-4)
    db.pipeline["calibration"]["fc_baseline"] = "picked"
    c = run_calibration(db, run, ROOT, "start", "picked")
    assert c["info"]["fc_baseline_V"] == 4.5e-4 and c["info"]["peak_overrides_U"] == [35.0]
