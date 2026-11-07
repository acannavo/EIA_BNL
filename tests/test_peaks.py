"""Peak stage (untitled28 port): GUI arithmetic, fit, replay of the U arrow clicks."""
import numpy as np
import pandas as pd
import pytest

from eiakit.db import load_database
from eiakit.peaks import (
    average_repeats,
    fit_peak,
    fit_voltage,
    gaussian,
    gui_lof,
    gui_voltage,
    legacy_grid_lengths,
    predicted_tof,
    replay_voltage,
)

db = load_database()
RUN = db.run("20250611")
PROF = db.profile("legacy_untitled28")
SETTINGS = dict(db.pipeline["peaks"])
MASSES = {int(k): float(v) for k, v in SETTINGS.pop("gui_masses").items()}


def _gui_reference(p, n):
    """Literal untitled28: step_entry = str(p/500); U = get_U() +/- float(step); str(U)."""
    entry, step = str(p), str(p / 500)
    for _ in range(abs(n)):
        U = float(entry) + float(step) if n > 0 else float(entry) - float(step)
        entry = str(U)
    return float(entry)


@pytest.mark.parametrize("p", [18.0, 24.0, 30.0, 35.0, 0.6, 250.01])
@pytest.mark.parametrize("n", [-9, -7, -2, 0, 1, 5])
def test_gui_voltage_matches_entry_round_trip(p, n):
    assert gui_voltage(p, n) == _gui_reference(p, n)


def test_gui_voltage_known_values():
    assert gui_voltage(30.0, -7) == 29.58000000000001
    assert gui_voltage(18.0, 5) == 18.180000000000007
    assert gui_voltage(24.0, -2) == 23.904000000000003


def test_gui_masses_cover_the_mass_labels():
    for sp in db.species.values():
        assert any(lbl in MASSES for lbl in sp.data_labels), sp.name
    assert MASSES[92] == 92.906 and MASSES[26] == 26.98


def _synthetic(U_true, lines, rng):
    """Three shots with Gaussian dips at the TOF predicted for U_true."""
    t = np.linspace(-8e-6, 1.9e-4, 9999)
    shots = {}
    for rep in (1, 2, 3):
        y = 1e-3 * rng.standard_normal(t.size)
        for m_amu, _, z in lines:
            t0 = predicted_tof(m_amu, z, U_true, RUN.eia.L_m, RUN.eia, PROF)
            y = y + gaussian(t, -0.05 / z, t0 * (1 + 2e-3 * rng.standard_normal()), 6e-8, 0)
        shots[rep] = (t, y)
    return shots


def test_fit_peak_recovers_a_clean_gaussian():
    t = np.linspace(20e-6, 40e-6, 20000)
    y = gaussian(t, -0.04, 30.01e-6, 5e-8, 1e-4)
    f = fit_peak(t, y, 30e-6)
    assert f["success"]
    assert f["mu"] == pytest.approx(30.01e-6, rel=1e-9)
    # legacy bound A >= -(max - min) of the window data: the depth can sit at that bound
    assert f["A"] == pytest.approx(-0.04, rel=1e-4)


@pytest.mark.parametrize("clicks", [-7, 0, 4])
def test_replay_recovers_the_click_count(clicks):
    rng = np.random.default_rng(1 + clicks + 10)
    lines = [(92.906, 92, z) for z in range(1, 6)]
    U_file = 30.0
    U_used = gui_voltage(U_file, clicks)
    shots = _synthetic(U_used * 0.999, lines, rng)
    legacy = fit_voltage(shots, U_used, lines, RUN.eia.L_m, RUN.eia, PROF, SETTINGS)
    detail = legacy[legacy.success].assign(U=U_file)
    r = replay_voltage(shots, U_file, detail, MASSES, RUN.eia.L_m, RUN.eia, PROF, SETTINGS)
    assert r["status"] == "exact"
    assert r["clicks"] == clicks and r["U_used"] == U_used


def test_replay_reports_missing_gui_mass():
    detail = pd.DataFrame({"m": [7], "z": [1], "TOF": [1e-5], "YieldAveFit": [-0.1]})
    r = replay_voltage({}, 30.0, detail, MASSES, RUN.eia.L_m, RUN.eia, PROF, SETTINGS)
    assert r["status"] == "no-mass" and r["missing_masses"] == [7]


def test_average_repeats_on_a_common_grid_is_the_plain_mean():
    t = np.linspace(0, 1, 11)
    g, y = average_repeats([(t, t * 0 + 1), (t, t * 0 + 3)])
    assert np.allclose(g, t) and np.allclose(y, 2)


def test_gui_lof_round_trip():
    entry, step = str(3.8499999999999996), str(3.8499999999999996 / 100)
    for _ in range(2):
        entry = str(round(float(entry) + float(step), 6))
    assert gui_lof(3.8499999999999996, 2) == float(entry) == 3.927


def test_legacy_grid_lengths_previous_voltage_rule(tmp_path):
    from pathlib import Path
    names = {"10v1.csv": 100, "10v2.csv": 110, "5p5v1.csv": 200, "5p5v2.csv": 210,
             "20v1.csv": 300}
    files = [(Path(n), float(n.split("v")[0].replace("p", ".")), int(n[-5])) for n in names]
    lens = legacy_grid_lengths(files, lambda p: names[p.name])
    # glob order 10v1, 10v2, 20v1, 5p5v1, 5p5v2: first U takes the last file read (5p5v2),
    # the next ones the last shot of the previous U
    assert lens == {10.0: 210, 20.0: 110, 5.5: 300}


def test_match_levels():
    from eiakit.peaks import match_rows
    old = pd.DataFrame({"m": [92, 92, 92], "z": [1, 2, 3], "TOF": [1e-5, 2e-5, 3e-5],
                        "YieldAveFit": [-0.1, -0.2, -0.3]})
    new = pd.DataFrame({"m": [92, 92, 92], "z": [1, 2, 3], "TOF": [1e-5, 2e-5, 3.03e-5],
                        "YieldAveFit": [-0.1, -0.2 * (1 + 3e-7), -0.3], "success": True})
    assert list(match_rows(new, old).level) == ["exact", "equivalent", "different"]
