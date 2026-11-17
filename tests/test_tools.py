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


def test_saved_voltage_reopens_as_saved(tmp_path, monkeypatch):
    from eiakit.db import load_database
    from eiakit.peaks_edit import PeakSession
    monkeypatch.setenv("EIA_FITS_ROOT", str(tmp_path))
    db = load_database()
    run = db.run("20250611")
    s = PeakSession.load(db, run, ROOT, 35.0)
    s.set_clicks(u_clicks=-7)
    s.refit(0.005)
    s.exclude(1)
    want = s.fits[["m", "z", "TOF", "Yield"]].copy()
    s.save(tmp_path)
    s2 = PeakSession.load(db, run, ROOT, 35.0)
    assert s2.restore(tmp_path)
    assert (s2.u_clicks, s2.corr, s2.excluded) == (-7, 0.005, {1})
    assert (s2.fits[["m", "z", "TOF", "Yield"]].to_numpy() == want.to_numpy()).all()


def test_extra_condition_column_and_pick_bars():
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from eiakit.figures import cbs_study as cf
    from eiakit.figures.powerlaw import _grid
    recs = [{"ds": SimpleNamespace(species=el, energy_mJ=mj)}
            for el in ("Al", "Nb") for mj in (180.0, 370.0, 780.0, 3.3)]
    g, els, ens = _grid(recs)
    assert ens == (180, 370, 780, 3.3) and g[0][3]["ds"].energy_mJ == 3.3
    tab = pd.DataFrame({"species": "Al", "energy_mJ": 3.3, "component": "fast", "z": [1, 2],
                        "u_fit": [300.0, 310.0], "u_left": [340.0, np.nan],
                        "u_right": [270.0, np.nan], "t_peak_fit_us": [10.0, 9.0],
                        "FWHM_fit_us": [2.0, 2.0], "excluded": False})
    lo, hi = cf._pick_bounds(tab)
    assert list(lo) == [270.0, 310.0] and list(hi) == [340.0, 310.0]   # no pick: no bar
    f = cf.fit_summary(tab, {"Al": ["fast"]}, [180.0, 3.3], "picks", {3.3: "3.3 mJ (ps)"})
    assert f.panel(0, 1).title == "Al, 3.3 mJ (ps)"


def test_study_cbs_status_counts_extra_runs(tmp_path, monkeypatch):
    """cbs records every run incl. the extra ones: it must not stay 'stale' forever."""
    import json

    from eiakit import study as S
    from eiakit.db import load_database
    db = load_database()
    st = db.study("PRL_AlNb")
    d = tmp_path / "PRL_AlNb" / "cbs"
    d.mkdir(parents=True)
    monkeypatch.setattr(S, "read_manifest", lambda *a, **k: {"run_id": "x"})
    man = {"source": "pipeline", "upstream_stage": "cbs",
           "settings_hash": S._settings_hash(db, st),
           "upstream": {k: "x" for k in st.all_runs()}}
    (d / "manifest.json").write_text(json.dumps(man))
    assert S.study_status(db, "PRL_AlNb", "cbs", None, None, tmp_path)["state"] == "ok"


def test_peak_check_columns_and_locked_save(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from eiakit.figures import export
    from eiakit.figures.collapse import peak_check
    t = np.geomspace(5, 500, 200)
    ds = [SimpleNamespace(key=f"{el}{mj}", species=el, energy_mJ=mj, title=f"{el}, {mj:g} mJ",
                          all_z=[1, 2], tof_us=t,
                          z_curves={1: np.exp(-(t - 50) ** 2 / 50), 2: np.exp(-(t - 30) ** 2 / 50)})
          for el in ("Al", "Nb") for mj in (3.3, 180.0, 370.0, 780.0)]
    pk = pd.DataFrame(columns=["run", "z", "component", "t_peak"])
    f = peak_check(ds, pk, columns=[180.0, 370.0, 780.0, 3.3])
    assert (f.rows, f.cols) == (2, 4)
    assert f.panel(0, 3).title == "Al, 3.3 mJ" and f.panel(1, 0).title == "Nb, 180 mJ"
    # a locked target (OneDrive / viewer): kept as .new, no crash
    calls = {"n": 0}
    real = export.os.replace

    def locked(src, dst):
        if str(dst).endswith("fig.png"):
            calls["n"] += 1
            raise OSError(22, "Invalid argument")
        return real(src, dst)
    monkeypatch.setattr(export.os, "replace", locked)
    monkeypatch.setattr(export.time, "sleep", lambda s: None)
    out = export.save(f, tmp_path / "fig", formats=("png",))
    assert out[0].name == "fig.new.png" and out[0].is_file() and calls["n"] == 6


def test_peak_check_time_axis_follows_the_data():
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from eiakit.figures.collapse import peak_check
    t = np.geomspace(0.5, 900, 400)
    ps = SimpleNamespace(key="ps", species="Nb", energy_mJ=3.3, title="Nb, 3.3 mJ", all_z=[1],
                         tof_us=t, z_curves={1: np.exp(-(np.log(t / 2.0)) ** 2 / 0.05)})
    ns = SimpleNamespace(key="ns", species="Nb", energy_mJ=780.0, title="Nb, 780 mJ",
                         all_z=[1], tof_us=t, z_curves={1: np.exp(-(np.log(t / 60)) ** 2 / 0.1)})
    pk = pd.DataFrame({"run": ["ps"], "z": [1], "component": ["slow"], "t_peak": [2.0]})
    f = peak_check([ps, ns], pk, columns=[780.0, 3.3])
    lo, hi = f.panel(0, 1).xlim                 # ps: signal around 2 us, not cut at 5 us
    assert lo < 1.5 and hi < 10
    lo2, hi2 = f.panel(0, 0).xlim
    assert 10 < lo2 < 60 < hi2
