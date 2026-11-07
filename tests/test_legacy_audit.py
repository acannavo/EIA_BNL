"""Legacy readers and audit, on small synthetic files (real data: test_real_data.py)."""
from pathlib import Path

import numpy as np
import pytest

from eiakit import physics as ph
from eiakit.audit import audit_run
from eiakit.db import load_database
from eiakit.legacy import (
    load_checkpoint,
    parse_cbs_areas,
    parse_cbs_fit_results,
    read_output_folder,
    voltage_from_output_name,
)


@pytest.fixture(scope="module")
def db():
    return load_database()


@pytest.mark.parametrize("name,U", [("p10.txt", 10.0), ("p0.5.txt", 0.5),
                                    ("p200.01.txt", 200.01), ("notes.txt", None)])
def test_voltage_from_name(name, U):
    assert voltage_from_output_name(name) == U


def _write_output(folder: Path):
    folder.mkdir(parents=True)
    hdr = "m,z,TOF,Yield\n"
    (folder / "p30.txt").write_text(hdr + "92,1,1.0e-4,-0.01\n92,2,7.0e-5,-0.02\n")
    # p35: own fit + a copy of p30 (N14) ; z=2 twice (N1)
    (folder / "p35.txt").write_text(hdr + "92,1,9.3e-5,-0.03\n92,2,6.6e-5,-0.04\n"
                                    "92,1,1.0e-4,-0.01\n92,2,7.0e-5,-0.02\n")
    # two (m, z) on the same peak (N8) and a near-duplicate voltage
    (folder / "p40.txt").write_text(hdr + "92,6,4.000e-5,-0.1\n16,1,4.0001e-5,-0.1\n")
    (folder / "p40.01.txt").write_text(hdr + "92,3,5.0e-5,-0.1\n")


def _fit_text(M, rows, renumber=False):
    """A CBS_fit_results file in the v12 format, for rows (z, cbs, kT, V0, A, t_pk_us[, fwhm])."""
    out = ["# CBS fit results  —  generated 2026-10-22 17:53:56",
           f"# L_EIA = 3.85 m   M_ION = {M:.6e} kg", "# Signal: per-ion (checkpoint spectra / z)",
           "#", ""]
    table = []
    for z in sorted({r[0] for r in rows}):
        out.append(f"z={z}  Total chi2=1.0e-04  dof=100  chi2/dof=1.0e-06")
        for (zz, k, kT, V0, A, tpk, *fw) in rows:
            fw = fw[0] if fw else 2.0
            if zz != z:
                continue
            out += [f"    CBS {k}:",
                    f"      Picks:   t_peak={tpk:.4f} µs  I_peak=1.0e-08 A·cm⁻²  "
                    f"t_left={tpk - 1:.4f} µs  t_right={tpk + 1:.4f} µs  FWHM=2.0000 µs",
                    f"      Window:  win_left={tpk - 5:.4f} µs  "
                    f"win_right={tpk + 5:.4f} µs  (picked)",
                    "      Exp peak triplet  [raw exp.]:",
                    f"        prev : t={tpk - 1:.6f} µs  I=1.0e-09 A·cm⁻²",
                    f"        PEAK : t={tpk:.6f} µs  I=1.0e-08 A·cm⁻²  u=100.0 km/s",
                    f"        next : t={tpk:.6f} µs  I=1.0e-08 A·cm⁻²",
                    f"      kT = {kT:.2f} +1.00 / -2.00 eV",
                    f"      V0 = {V0:.2f} +3.00 / -4.00 V",
                    f"      A  = {A:.4e} +1.0e-20 / -2.0e-20 A·cm⁻²",
                    "      chi2=1.0e-05  chi2/dof=1.0e-07"]
            table.append((z, k, tpk, kT, V0, A, fw))
    out.append("# ── Pick vs fit summary (tab-separated; times in µs, currents in A·cm⁻²) ──")
    out.append("# z\tCBS\tt_peak_pick\tt_peak_fit\tFWHM_pick\tFWHM_fit\tt_left_pick\tt_left_fit\t"
               "t_right_pick\tt_right_fit\tI_peak_click\tI_peak_exp\tI_peak_fit\twin_left\t"
               "win_right\twin_src\tkT_eV\tV0_V\tA")
    for z, k, tpk, kT, V0, A, fw in table:
        out.append(f"{z}\t{k}\t{tpk:.6f}\t{tpk:.6f}\t{fw:.6f}\t{fw:.6f}\t1\t1\t1\t1\t1e-8\t1e-8\t1e-8\t"
                   f"{tpk - 5:.6f}\t{tpk + 5:.6f}\tpicked\t{kT:.4f}\t{V0:.4f}\t{A:.6e}")
    return "\n".join(out) + "\n"


def test_output_checks(db, tmp_path):
    run_dir = tmp_path / "data" / "20250611"
    _write_output(run_dir / "2_SEM" / "Output")
    out = read_output_folder(run_dir / "2_SEM" / "Output")
    assert len(out) == 9 and set(out.U) == {30.0, 35.0, 40.0, 40.01}
    rep = audit_run(db, db.run("20250611"), tmp_path / "data")
    codes = {f.code: f for f in rep.findings}
    assert "N1" in codes and "p35.txt" in codes["N1"].details[0]
    assert "N14" in codes and "('p30.txt', 'p35.txt')" in codes["N14"].title
    assert "N8" in codes and "(m=92, z=6) and (m=16, z=1)" in codes["N8"].details[0]
    assert "OUT-U" in codes


def test_fit_file_parser_tolerant(tmp_path, db):
    M = db.mass_kg("Nb", "legacy_proton")
    f = tmp_path / "fit.txt"
    f.write_text(_fit_text(M, [(1, 2, 30.0, 0.0, 1e-16, 200.0), (7, 1, 29.5, 1427.0, 3e-20, 25.3),
                               (7, 2, 101.1, 931.6, 5e-19, 28.8)]), encoding="utf-8")
    p = parse_cbs_fit_results(f)
    assert p.header.per_ion and p.header.L_EIA_m == 3.85
    c = p.component(7, 1)
    assert (c.kT, c.V0, c.A) == (29.5, 1427.0, 3e-20) and c.errors["kT"] == (1.0, 2.0)
    assert c.window["src"] == "picked" and c.triplet["next"]["t"] == c.triplet["PEAK"]["t"]
    assert list(p.table.CBS) == [2, 1, 2]


def test_areas_parser_typo_and_colon(tmp_path):
    f = tmp_path / "areas.txt"
    f.write_text(
        "# CBS area integrals  —  generated x\n# L_EIA = 3.85 m   M_ION = 1.5e-25 kg\n"
        "# Signal: per-ion (checkpoint spectra / z)\n#\n"
        "z=1  Area_data_FULL  = 3.9e-06 A·cm⁻²·µs  (entire tof grid)\n"
        "    CBS 2 t_peak=208.3273 µs  FWHM=258.9882 µs  window=[30.4205, 856.9784] µs  (picked)\n"
        "      Area_model  = 2.8e-06 A·cm⁻²·µs\n      Area_model_FULL = 2.9e-06 A·cm⁻²·µs\n\n"
        "# z\tCBS\tt_peak_us\tArea_model_full\n1\t2\t08.327260\t2.9e-06\n", encoding="utf-8")
    a = parse_cbs_areas(f)
    assert a.blocks.t_peak_us.iloc[0] == 208.3273 and not a.blocks.colon.iloc[0]
    assert a.table.t_peak_us.iloc[0] == pytest.approx(8.32726)


def test_full_run_audit_reproduces_model(db, tmp_path):
    """Synthetic run: checkpoint + fit + areas made with the legacy fitter formulas."""
    run = db.run("20250623")                     # Nb 370: CBS 1 = slow, CBS 2 = fast
    res = tmp_path / "data" / run.key / "EIA-Results"
    res.mkdir(parents=True)
    m, e, L = db.mass_kg("Nb", "legacy_proton"), 1.602e-19, 3.85
    tof = np.linspace(5e-6, 400e-6, 20000)
    comps = [(3, 1, 150.0, 30.0, 1e-16), (3, 2, 300.0, 900.0, 1e-19)]
    rows, area_lines = [], []
    for z, k, kT, V0, A in comps:
        y = A * ph.cbs_shape(tof, kT, V0, z, m, L, e)
        tpk = tof[np.argmax(y)] * 1e6
        tL, tR = ph.half_maximum_edges(tof * 1e6, y)
        rows.append((z, k, kT, V0, A, tpk, tR - tL))
        area = float(np.trapezoid(y, tof * 1e6))
        area_lines.append(f"{z}\t{k}\t{tpk:.6f}\t{area:.6e}")
    (res / "CBS_fit_results_92_Nb.txt").write_text(_fit_text(m, rows), encoding="utf-8")
    (res / "CBS_areas_92_Nb.txt").write_text(
        "# CBS area integrals\n# L_EIA = 3.85 m   M_ION = 1.5e-25 kg\n"
        "# Signal: per-ion (checkpoint spectra / z)\n"
        "# z\tCBS\tt_peak_us\tArea_model_full\n" + "\n".join(area_lines) + "\n", encoding="utf-8")
    spec = {(92, 3): np.exp(-0.5 * ((tof - 60e-6) / 8e-6) ** 2) * 1e-6}
    ck = {"corrected_spectra": spec, "corrected_tof": tof,
          "corrected_spectra_1m": {k: v * L ** 3 for k, v in spec.items()},
          "corrected_exp_points": {(92, 3): -np.ones(4)},
          "corrected_exp_tof": {(92, 3): np.array([40e-6, 50e-6, 60e-6, 70e-6])},
          "integrals": {(92, 3): 2.0 * L ** 2}, "integrals_eia": {(92, 3): 2.0},
          "CFY_EIA": 0.059, "CFY_1m": 0.059 / (L / 2.38) ** 3, "CFArea": 1 / (np.pi * 0.25),
          "CFCurrent": 1 / 51, "L_EIA": L, "L_FC": 2.38}
    np.save(res / "pipeline_checkpoint.npy", ck, allow_pickle=True)
    assert load_checkpoint(res / "pipeline_checkpoint.npy")["L_EIA"] == L
    rep = audit_run(db, run, tmp_path / "data")
    by = {}
    for f in rep.findings:
        by.setdefault((f.level, f.code), f)
    assert ("info", "CBS-REPRO") in by, [(f.level, f.code, f.title) for f in rep.findings]
    assert ("info", "I1") in by and "legacy_proton" in by[("info", "I1")].title
    assert ("info", "N3") in by                     # slow (CBS 1) arrives after fast (CBS 2)
    assert ("info", "CK-CHAIN") in by and ("info", "CK-GEOM") in by
    assert ("warn", "CBS-TRIP") in by
    t = rep.tables["cbs_components"]
    assert list(t.component) == ["slow", "fast"]
    # labels against the arrival times: with the Nb 780 mapping this run would be wrong
    rep2 = audit_run(db, db.run("20250611"), tmp_path / "data")
    assert not any(f.code == "N3" and f.level == "error" for f in rep2.findings)  # no files there
