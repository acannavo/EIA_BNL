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
# these tests compare with the legacy scripts: legacy constants (I1 off)
db.pipeline.setdefault("corrections", {})["mass_constants"] = "legacy"
# ... and the FC aperture the 2026 legacy analysis assumed (0.50 cm; real: 20 mm diameter)
db.instruments.fc["FC_2026_ps"] = db.instruments.fc["FC_2026_ps"].model_copy(
    update={"aperture_radius_cm": 0.50})
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
    # the FC peak recovered from the checkpoint gives the same CFY (fc_missing = legacy_fc_peak)
    from eiakit.calibration import legacy_fc_peak
    new2 = calibrate(rows, None, 0.0, run.eia, run.fc, db.pipeline["calibration"],
                     fc_peak_override=legacy_fc_peak(old, run.fc))
    assert abs(new2["CFY_1m"] / old["CFY_1m"] - 1) < 1e-12


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


@pytest.mark.parametrize("key", [k for k in RUNS if (ROOT / k / "EIA-Results").is_dir()])
def test_legacy_cbs_replay(key):
    """CBS port: areas, fractions, fitted peak / FWHM recomputed within the stored rounding."""
    from eiakit.cbs import replay_run_cbs
    res = replay_run_cbs(db, db.run(key), ROOT)
    if res["status"] == "missing files":
        pytest.skip("no CBS files / checkpoint")
    assert res["status"].startswith("reproduced"), res["summary"]


@pytest.mark.parametrize("key", [k for k in RUNS if (ROOT / k / "EIA-Results").is_dir()])
def test_cbs_replay_invariant_under_codata_constants(key):
    """I1: CODATA masses / charge only rescale kT, V0; the CBS curves are unchanged."""
    from eiakit.cbs import replay_run_cbs, run_model
    d = load_database()
    d.pipeline.setdefault("corrections", {})["mass_constants"] = "codata"
    res = replay_run_cbs(d, d.run(key), ROOT)
    if res["status"] == "missing files":
        pytest.skip("no CBS files / checkpoint")
    assert res["status"].startswith("reproduced"), res["summary"]
    assert run_model(d, d.run(key), ROOT)["profile"] == "corrected"


# printed by plot_eia_powerLaw_test.py on the legacy checkpoints (scale-invariant numbers)
LEGACY_ZBAR = {   # run: (C, b, sig_b, n, R2, E_lo, E_hi, npts)
    "20250513": (0.3482, 0.3374, 0.0002, 2.964, 0.9709, 20.0, 9999.7, 67765),
    "20250602": (0.2332, 0.3949, 0.0002, 2.532, 0.9865, 41.3, 9999.6, 66659),
    "20250606": (0.2353, 0.3844, 0.0002, 2.601, 0.9733, 40.5, 9996.3, 66713),
    "20250611": (0.1754, 0.3905, 0.0003, 2.561, 0.9836, 98.9, 30974.3, 30124),
    "20250623": (0.2795, 0.3314, 0.0002, 3.017, 0.9660, 27.3, 30988.4, 68861),
    "20250617": (0.2809, 0.3416, 0.0003, 2.928, 0.9725, 41.1, 25541.8, 48331),
}


@pytest.mark.parametrize("key", [k for k in LEGACY_ZBAR if (ROOT / k / "EIA-Results").is_dir()])
def test_zbar_power_law_matches_legacy_script(key):
    from eiakit.analysis import analyze_zbar, load_dataset
    from eiakit.legacy import load_checkpoint
    st = db.study("PRL_AlNb")
    ck = load_checkpoint(ROOT / key / "EIA-Results" / "pipeline_checkpoint.npy")
    f = analyze_zbar(load_dataset(db, db.run(key), ck, st), st.zbar)["fit"]
    new = (f"{f['C']:.4g}", f"{f['b']:.4f}", f"{f['sig_b']:.4f}", f"{f['n']:.3f}",
           f"{f['r2']:.4f}", f"{f['E_lo']:.1f}", f"{f['E_hi']:.1f}", f["npts"])
    assert tuple(float(v) for v in new) == LEGACY_ZBAR[key]


# printed by plot_cbs_multipanel_v2.py (mean +/- std km/s, n z) after its remap + exclusions
LEGACY_CBS_MEANS = {
    ("u_peak", "20250513", "ultra_fast"): (593.0, 36.5, 7),
    ("u_peak", "20250513", "fast"): (377.3, 23.5, 11),
    ("u_peak", "20250611", "fast"): (286.6, 62.2, 13),
    ("u_pick", "20250513", "fast"): (378.8, 24.0, 11),
    ("u_pick", "20250611", "fast"): (286.9, 61.9, 13),
    ("u_peak", "20250602", "ultra_fast"): (513.8, 23.9, 6),
    ("u_peak", "20250602", "fast"): (337.1, 22.3, 11),
    ("u_peak", "20250623", "fast"): (211.1, 22.2, 8),
    ("u_pick", "20250623", "fast"): (210.4, 23.5, 8),
    ("u_peak", "20250606", "fast"): (288.9, 37.4, 9),
    ("u_peak", "20250617", "fast"): (151.4, 0.0, 1),
    ("u_fit", "20250606", "fast"): (286.8, 39.3, 9),
    ("u_fit", "20250513", "fast"): (379.5, 23.9, 11),
    ("u_fit", "20250602", "ultra_fast"): (514.6, 23.0, 6),
}


def test_cbs_component_means_match_legacy_script():
    from eiakit.cbs_study import component_means, study_components
    st = db.study("PRL_AlNb")
    if not all((ROOT / k / "EIA-Results").is_dir() for k in st.runs):
        pytest.skip("not all PRL runs present")
    tab = study_components(db, "PRL_AlNb", ROOT)
    for (col, key, comp), (m, s, n) in LEGACY_CBS_MEANS.items():
        r = component_means(tab, st.avg_components, col, nan_aware=(col != "u_fit"))
        r = r[(r.run == key) & (r.component == comp)].iloc[0]
        assert (round(r["mean"], 1), round(r["std"], 1), r["n"]) == (m, s, n), (col, key, comp)


# printed by plot_collapse_v2.py: (z*, se) per dataset and joint, for both peak sources
LEGACY_COLLAPSE = {
    "eia": {"20250513": (10.79, 0.53), "20250602": (8.52, 0.24), "20250606": (6.82, 0.24),
            "20250611": (13.63, 0.25), "20250623": (9.92, 0.36), "joint": (9.65, 0.13)},
    "cbs": {"20250513": (13.96, 0.34), "20250602": (12.83, 0.17), "20250606": (10.67, 0.13),
            "20250611": (14.66, 0.18), "20250623": (12.88, 0.38), "joint": (12.36, 0.08)},
}


def test_collapse_matches_legacy_script():
    import pandas as pd

    from eiakit import collapse as C
    from eiakit.analysis import load_dataset
    from eiakit.legacy import load_checkpoint
    st = db.study("PRL_AlNb")
    if not all((ROOT / k / "EIA-Results").is_dir() for k in st.runs):
        pytest.skip("not all PRL runs present")
    cs = st.collapse
    prof = {k: db.run(k).legacy.profile_analysis for k in st.runs}
    masses = {k: db.mass_kg(db.run(k).target, prof[k]) for k in st.runs}
    e_C = db.e_C(prof[st.runs[0]])
    pk = []
    for k in st.runs:
        ck = load_checkpoint(ROOT / k / "EIA-Results" / "pipeline_checkpoint.npy")
        pk.append(C.eia_peaks(load_dataset(db, db.run(k), ck, st), cs.peak_min_prom_frac,
                              cs.peak_err_mode))
    peaks = {"eia": pd.concat(pk, ignore_index=True),
             "cbs": pd.concat([C.cbs_peaks(db, db.run(k), ROOT, cs.cbs_t_source)
                               for k in st.runs], ignore_index=True)}
    for src, exp in LEGACY_COLLAPSE.items():
        ser = C.collapse_series(C.drop_excluded(peaks[src], st.exclude), cs.runs, masses, e_C,
                                cs.min_fast_points)
        got = {s["run"]: (round(s["zstar"], 2), round(s["se_zstar"], 2)) for s in ser}
        j = C.weighted_joint([s["zstar"] for s in ser], [s["se_zstar"] for s in ser])
        got["joint"] = (round(j["zstar"], 2), round(j["se"], 2))
        assert got == exp, src


def test_cbs_store_roundtrip_reproduces_the_legacy_fit(tmp_path):
    """Step 9: importing a legacy fit into the store changes no number of the CBS model."""
    from eiakit.cbs import CBSModel, components_from_fit_file  # noqa: F401
    from eiakit.cbs_store import import_legacy, legacy_fit_path, read_store
    from eiakit.legacy import parse_cbs_fit_results
    for key in [k for k in RUNS if (ROOT / k / "EIA-Results").is_dir()][:3]:
        run = db.run(key)
        p = import_legacy(db, run, ROOT, tmp_path)
        st = read_store(tmp_path, run)
        leg = parse_cbs_fit_results(legacy_fit_path(run, ROOT))
        a, b = components_from_fit_file(leg), components_from_fit_file(st)
        assert sorted(a) == sorted(b)
        for z in a:
            for ca, cb in zip(a[z], b[z], strict=True):
                assert (ca.kT, ca.V0, ca.A, ca.t_peak, ca.fwhm, ca.win_left, ca.win_right) == \
                       (cb.kT, cb.V0, cb.A, cb.t_peak, cb.fwhm, cb.win_left, cb.win_right)
                assert ca.extra["cbs"] == cb.extra["cbs"]
        assert p.is_file() and st.header.M_ION_kg == (leg.header.M_ION_kg or st.header.M_ION_kg)


def test_cbs_session_edit_save_reload(tmp_path):
    """Step 9: a fitter session saved unchanged reproduces the legacy model; edits persist."""
    import numpy as np

    from eiakit.cbs_edit import CBSSession
    key = next(k for k in RUNS if (ROOT / k / "EIA-Results").is_dir())
    run = db.run(key)
    s = CBSSession.load(db, run, ROOT)
    z = next(z for z in s.zs if s.comps.get(z) and s.comps[z][0].active)
    ref = s.total(z).copy()
    a0 = s.areas(z)
    p = s.save(tmp_path)
    assert p.is_file()
    import os
    os.environ["EIA_FITS_ROOT"], old = str(tmp_path), os.environ["EIA_FITS_ROOT"]
    try:
        os.environ.setdefault("EIA_PROCESSED_ROOT", str(tmp_path / "p"))
        os.environ.setdefault("EIA_STUDIES_ROOT", str(tmp_path / "s"))
        s2 = CBSSession.load(db, run, ROOT)
        assert s2.source.startswith("store")
        assert np.allclose(s2.total(z), ref, rtol=1e-10, atol=0)
        assert np.allclose(s2.areas(z).Ions_model_full_cm2, a0.Ions_model_full_cm2, rtol=1e-10)
        # edit: refit the first component, add a slow one, save, reload
        s2.fit(z, 0)
        i = s2.add(z, "slow", float(s2.comps[z][0].t_peak) * 1.5)
        assert s2.comps[z][i].extra["label"] == "slow"
        s2.save(tmp_path, note="test")
        s3 = CBSSession.load(db, run, ROOT)
        labs = [c.extra["label"] for c in s3.comps[z]]
        assert "slow" in labs and len(labs) == len(s2.comps[z])
        assert len(list(tmp_path.glob(f"{key}/cbs_fit.*.json"))) >= 1     # backup kept
    finally:
        os.environ["EIA_FITS_ROOT"] = old


def test_cbs_half_maximum_clicks_set_fwhm_and_kT():
    from eiakit.cbs_edit import CBSSession
    key = next(k for k in RUNS if (ROOT / k / "EIA-Results").is_dir())
    s = CBSSession.load(db, db.run(key), ROOT)
    z = next(z for z in s.zs if s.comps.get(z) and s.comps[z][0].active)
    c = s.comps[z][0]
    t0, kT0 = c.t_peak, c.kT
    s.set_half(z, 0, "left", t0 - 0.1 * t0)
    s.set_half(z, 0, "right", t0 + 0.12 * t0)
    assert abs(c.fwhm - 0.22 * t0) < 1e-12 and c.kT != kT0 and c.A is not None
