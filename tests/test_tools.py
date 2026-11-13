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


def test_legacy_pedestal_with_other_fc_sources(monkeypatch, tmp_path):
    """fc_source != start with the legacy pedestal: per-trace pre-trigger + the legacy offset.
    Here 3_FC_Stop of 20250611 may be empty: 'both' = the start traces -> the legacy CFY."""
    from eiakit.calibration import run_calibration
    from eiakit.db import load_database
    monkeypatch.setenv("EIA_FITS_ROOT", str(tmp_path))
    db = load_database()
    run = db.run("20250611")
    db.pipeline["calibration"]["output_duplicates"] = "all"
    db.pipeline["calibration"]["overlaps"] = "keep_both"
    a = run_calibration(db, run, ROOT, "start", "legacy")
    stop_dir = ROOT / "20250611" / "3_FC_Stop"
    if stop_dir.is_dir() and any(stop_dir.iterdir()):
        pytest.skip("stop traces present: 'both' differs from 'start' by design")
    b = run_calibration(db, run, ROOT, "both", "legacy")
    assert abs(a["info"]["CFY_1m"] / b["info"]["CFY_1m"] - 1) < 1e-9
    assert "per trace" in b["info"]["baseline_method"]
    assert a["info"]["fc_peak_t_s"] > 1e-6 and not a["info"].get("fc_peak_warning")
    assert all(p > 0.02 for p in a["info"]["fc_per_shot_peak_V"])     # ion peaks, not pickup


def test_peak_picker_u_clicks_reproduce_untitled28():
    """20250611, U = 35 V was fitted with -7 arrow clicks on U (N16): same clicks, same fits."""
    import numpy as np

    from eiakit.db import load_database
    from eiakit.peaks_edit import PeakSession
    db = load_database()
    s = PeakSession.load(db, db.run("20250611"), ROOT, 35.0)
    s.set_clicks(u_clicks=-7)
    assert abs(s.U_used - 34.51) < 1e-9
    m = s.fits[s.fits.success].merge(s.legacy, on=["m", "z"], suffixes=("", "_leg"))
    assert len(m) == len(s.legacy) and np.allclose(m.TOF, m.TOF_leg, rtol=1e-12)
    s.set_clicks(lof_clicks=1)
    assert abs(s.L_used / s.L_m - 1.01) < 1e-9
