"""A new dataset (raw files only, no legacy results) through the whole pipeline:
add-run, run (audit, peaks fit, calibration, cbs, report), cbs-seed, run, study."""
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("EIA_DATA_ROOT", "/nonexistent"))
SRC = ROOT / "20250611"
pytestmark = pytest.mark.skipif(not (SRC / "2_SEM").is_dir(), reason="needs 20250611 raw files")


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    run = data / "20251120"
    (run / "2_SEM").mkdir(parents=True)
    for p in (SRC / "2_SEM").glob("*.csv"):
        shutil.copy(p, run / "2_SEM")
    shutil.copytree(SRC / "1_FC_Start", run / "1_FC_Start")
    db = tmp_path / "db"
    shutil.copytree(Path(__file__).parents[1] / "db", db)
    for k, v in {"EIA_DB_ROOT": db, "EIA_DATA_ROOT": data, "EIA_PROCESSED_ROOT":
                 tmp_path / "processed", "EIA_STUDIES_ROOT": tmp_path / "studies",
                 "EIA_FITS_ROOT": tmp_path / "fits"}.items():
        monkeypatch.setenv(k, str(v))
    return tmp_path


def test_new_dataset_end_to_end(env, capsys):
    from eiakit.cli import main
    from eiakit.db import load_database
    assert main(["add-run", "20251120", "--target", "Nb", "--energy", "370",
                 "--like", "20250611"]) == 0
    run = load_database().run("20251120")
    assert run.is_new and run.cbs_names[1] == "fast"
    assert main(["run", "20251120"]) == 0
    out = capsys.readouterr().out
    assert "peaks: ok" in out and "calibration: ok" in out and "no CBS fit yet" in out
    assert "report: ok" in out
    p = env / "processed" / "20251120"
    assert len(list((p / "10_peaks" / "output").glob("p*.txt"))) == 2
    assert (p / "40_report" / "report.html").is_file()
    assert main(["cbs-seed", "20251120"]) == 0
    assert (env / "fits" / "20251120" / "cbs_fit.json").is_file()
    assert main(["run", "20251120"]) == 0
    out = capsys.readouterr().out
    assert "fit store" in out
    (env / "db" / "studies" / "NEW.toml").write_text(
        'name = "NEW"\nruns = ["20251120"]\nrows = ["Nb"]\ncolumn_energy_mJ = [370.0]\n'
        '[zbar]\ne_max_eV = { Nb = 3.1e4 }\nlower_from_z = 2\n', encoding="utf-8")
    assert main(["study", "NEW", "--html-only"]) == 0
    assert (env / "studies" / "NEW" / "powerlaw").is_dir()


def test_peak_fit_matches_untitled28_on_strong_lines(env):
    """The automatic window centring reproduces the hand-adjusted untitled28 fits."""
    import pandas as pd

    from eiakit.cli import main
    from eiakit.db import load_database
    from eiakit.legacy import read_output_folder
    from eiakit.peaks import fit_run
    main(["add-run", "20251120", "--target", "Nb", "--energy", "780", "--like", "20250611"])
    db = load_database()
    r = fit_run(db, db.run("20251120"), env / "data", env / "fit")
    assert r["voltages"] == 2
    f = pd.read_csv(env / "fit" / "fits.csv")
    leg = read_output_folder(SRC / "2_SEM" / "Output_detail")
    m = f[f.success].merge(leg, on=["U", "m", "z"], suffixes=("", "_leg"))
    strong = m[m.YieldAveFit_leg.abs() > 0.01]
    assert len(strong) >= 6
    assert (abs(strong.TOF / strong.TOF_leg - 1) < 1e-3).all()
    assert (abs(strong.Yield / strong.YieldAveFit_leg - 1) < 0.03).all()


def test_peak_picker_on_a_new_run_reproduces_the_pipeline(env):
    from eiakit.cli import main
    from eiakit.db import load_database
    from eiakit.peaks_edit import PeakSession
    main(["add-run", "20251120", "--target", "Nb", "--energy", "780", "--like", "20250611"])
    assert main(["run", "20251120", "--stage", "peaks"]) == 0
    db = load_database()
    s = PeakSession.load(db, db.run("20251120"), env / "data", 35.0,
                         processed=env / "processed")
    # every line is tried (> 20), the stage keeps those above the noise cut, the same
    # ones the session keeps
    kept = s.fits[s.fits.success & ~s.fits.excluded]
    assert len(s.fits) > 20 and s.k != 1.0 and len(kept) == len(s.legacy) >= 6
    m = kept.merge(s.legacy, on=["m", "z"], suffixes=("", "_ref"))
    assert len(m) == len(kept) and (abs(m.TOF / m.TOF_ref - 1) < 1e-12).all()


def test_line_choice_noise_cut_and_run_defaults(env, tmp_path):
    import numpy as np

    from eiakit.cli import main
    from eiakit.db import load_database
    from eiakit.peaks import fit_run, new_run_lines
    from eiakit.peaks_edit import PeakSession
    main(["add-run", "20251120", "--target", "Nb", "--energy", "780", "--like", "20250611"])
    db = load_database()
    run = db.run("20251120")
    assert max(z for _, lab, z in new_run_lines(db, run) if lab == 12) == 6   # C: Z = 6
    s = PeakSession.load(db, run, env / "data", 35.0)
    s.set_selection({92: 12, 12: 99})                  # capped at Z
    assert s.selection() == {92: 12, 12: 6}
    f = s.fits[s.fits.success]
    half = f.t0 * 0.02
    assert ((f.TOF - f.t0).abs() <= half * (1 + 1e-9)).all()   # never out of the window
    s.mu_bound = 20e-9
    s.refit()
    f = s.fits[s.fits.success]
    assert ((f.TOF - f.t0).abs() <= 20e-9 * (1 + 1e-6)).all()
    weak = s.fits.weak & s.fits.excluded
    assert weak.any() and not s.fits[s.fits.m == 12].snr.gt(3).any()   # no carbon here
    i = int(np.flatnonzero(weak)[0])
    s.exclude(i, False)                                # kept by hand
    assert not s.fits.excluded.iloc[i]
    fits = tmp_path / "fits"
    s.save_defaults(fits)
    s2 = PeakSession.load(db, run, env / "data", 30.0)
    assert not s2.restore(fits) and s2.selection() == {92: 12, 12: 6} and s2.mu_bound == 20e-9
    r = fit_run(db, run, env / "data", tmp_path / "out",
                defaults=json.loads((fits / run.key / "peaks" / "defaults.json").read_text()))
    assert r["lines_from"] == "Peaks-tab defaults" and r["lines_total"] == 2 * 18
