import legacy_reference as legacy
import numpy as np
import pandas as pd
import pytest

from eiakit.calibration import average_fc, calibrate, compare_checkpoints, recover_fc_baseline
from eiakit.db import load_database
from eiakit.store import load_calibration_h5, save_calibration_h5


@pytest.fixture(scope="module")
def db():
    return load_database()


def _rows():
    """A small untitled28 peak table: two masses, z = 1..3, eight voltages."""
    out = []
    for i, U in enumerate([5, 10, 20, 40, 80, 160, 320, 640]):
        for m, zmax in ((92, 3), (16, 1)):
            for z in range(1, zmax + 1):
                t = 3.85 / np.sqrt(2 * z * 1.6e-19 * 20 * U / (m * 1.66e-27))
                out.append(dict(file=f"p{U}.txt", U=float(U), row=z, m=m, z=z, TOF=t,
                                Yield=-0.01 * np.exp(-((i - 3) / 2) ** 2) / z))
    return pd.DataFrame(out)


def _fc():
    t = np.linspace(-2e-6, 40e-6, 3001)
    rng = np.random.default_rng(0)
    return [(t, 0.002 + 0.03 * np.exp(-0.5 * ((t - 12e-6) / 3e-6) ** 2)
             + 1e-4 * rng.normal(size=t.size)) for _ in range(3)]


def test_calibration_chain_and_baseline_recovery(db, tmp_path):
    run = db.run("20250611")
    pipe = {**db.pipeline["calibration"], "n_interp": 5000}
    rows, traces = _rows(), _fc()
    res = calibrate(rows, traces, 0.002, run.eia, run.fc, pipe)
    L, Lf = run.eia.L_m, run.fc.L_m
    # projection chain of untitled27
    assert res["CFY_EIA"] / res["CFY_1m"] == pytest.approx((L / Lf) ** 3, rel=1e-12)
    for k in res["integrals"]:
        assert res["integrals"][k] == pytest.approx(L ** 2 * res["integrals_eia"][k], rel=1e-12)
    I_1m, I_eia = legacy.untitled27_integrals(
        res["corrected_tof"], res["corrected_spectra_1m"][(92, 2)]
        / (res["CFY_1m"] * res["CFArea"] * res["CFCurrent"]),
        res["CFY_1m"], res["CFArea"], res["CFCurrent"], L)
    assert (I_1m, I_eia) == pytest.approx((res["integrals"][(92, 2)],
                                           res["integrals_eia"][(92, 2)]), rel=1e-12)
    # the FC peak after baseline equals the calibrated EIA peak at 1 m x L_FC^3 (peak matching)
    s1 = sum(res["corrected_spectra_1m"].values())
    assert s1.max() == pytest.approx(res["fc_peak_V"] * Lf ** 3 * res["CFArea"] * res["CFCurrent"],
                                     rel=1e-12)
    # zero fill outside each (m, z) measured range (I4) and the x z factor
    seg_t = rows[(rows.m == 16)].TOF
    outside = (res["corrected_tof"] < seg_t.min()) | (res["corrected_tof"] > seg_t.max())
    assert np.all(res["corrected_spectra"][(16, 1)][outside] == 0)
    # the hand-picked pedestal comes back from the checkpoint (N7)
    B = recover_fc_baseline(res, traces, pipe["fc_grid_points"], run.eia, run.fc)
    assert B == pytest.approx(0.002, abs=1e-15)
    # CFY override rebuilds the same result
    res2 = calibrate(rows, None, 0.0, run.eia, run.fc, pipe, cfy_1m_override=res["CFY_1m"])
    legacy_keys = [k for k in res if k not in ("fc_average", "fc_baseline_V")]
    d = compare_checkpoints(res2, {k: res[k] for k in legacy_keys})
    assert max(v for v in d.values() if isinstance(v, float)) < 1e-12
    # HDF5 round trip
    p = save_calibration_h5(tmp_path / "spectra.h5", res, {"run": run.key})
    back = load_calibration_h5(p)
    assert np.array_equal(back["corrected_spectra"][(92, 3)], res["corrected_spectra"][(92, 3)])
    assert back["CFY_1m"] == res["CFY_1m"] and back["integrals"] == res["integrals"]


def test_average_fc_grid():
    t = np.linspace(0, 1, 11)
    gt, gv = average_fc([(t, t), (t, 2 * t)], 5)
    assert np.allclose(gt, np.linspace(0, 1, 5)) and np.allclose(gv, 1.5 * gt)


def test_compare_detects_differences():
    a = {"x": np.ones(3), "d": {(1, 1): np.ones(2)}, "l": [1, 2]}
    b = {"x": np.ones(3) * (1 + 1e-6), "d": {(1, 2): np.ones(2)}, "l": [1, 2]}
    d = compare_checkpoints(a, b)
    assert d["x"] == pytest.approx(1e-6, rel=1e-5) and "different" in d["d"] and d["l"] == 0.0
