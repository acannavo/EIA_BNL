"""A new dataset (raw files only, no legacy results) through the whole pipeline:
add-run, run (audit, peaks fit, calibration, cbs, report), cbs-seed, run, study."""
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
