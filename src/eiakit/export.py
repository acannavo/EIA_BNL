"""
CSV export of the charge-state-resolved time-of-flight distributions of a study's runs
(from processed/<run>/20_calibration/spectra.h5), for use outside eiakit.

Per run, in studies/<study>/export/:

  <run>_Jz_vs_time.csv   the calibrated distributions J_z(t) on a common time grid
  <run>_points.csv       the measured points (one per deflector voltage) with the
                         voltage, velocity, energy, raw EIA yield and the per-eV value

What the signal is (untitled27, kept by the calibration stage):
  signal = EIA yield x z x CFY_1m x CFArea x CFCurrent        ("yield x z", charge-weighted)
  per ion = signal / z                                         (what the CBS fitter uses)
Flight length: the measurement is at the EIA flight length L_EIA; the files give the
distributions SCALED TO 1 m (t_1m = t / L_EIA, signal_1m = signal(L_EIA) x L_EIA^3), as the
z-bar, energy-spectrum and CBS analyses use them, plus the real time t for reference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .store import load_calibration_h5


def _species_of(db, label: int) -> str:
    for name, sp in db.species.items():
        if label in sp.data_labels or label == sp.file_label:
            return name
    return f"m{label}"


def _header(run, L, ck, what: list[str]) -> str:
    las = run.laser
    lines = [
        f"run: {run.key}  target: {run.target}  laser: "
        + ", ".join(f"{k}={v}" for k, v in las.model_dump().items()
                    if v is not None and v is not False),
        f"EIA flight length L_EIA = {L:.6g} m (the real measurement distance)",
        "signal = EIA yield x z x CFY_1m x CFArea x CFCurrent = 'yield x z' "
        "(charge-weighted, as untitled27); per ion = signal / z",
        f"calibration: CFY_1m = {ck.get('CFY_1m', float('nan')):.6g}, "
        f"CFArea = {ck.get('CFArea', float('nan')):.6g}, "
        f"CFCurrent = {ck.get('CFCurrent', float('nan')):.6g}",
        "scaled to a 1 m flight path: t_1m = t / L_EIA, signal_1m = signal(L_EIA) x L_EIA^3",
        *what,
    ]
    return "".join(f"# {s}\n" for s in lines)


def export_run(db, run, ck: dict, out_dir: Path, n_grid: int = 2000) -> list[Path]:
    """The two CSV files of one run from its calibration output ck."""
    out_dir.mkdir(parents=True, exist_ok=True)
    L = float(ck.get("L_EIA", run.eia.L_m))
    prof = db.profile_for(run, "analysis")
    amu, e = db.profile(prof).amu_kg, db.e_C(prof)
    keys = sorted(ck["corrected_spectra_1m"], key=lambda k: (k[0], k[1]))
    target = {int(m) for m in run.mass_labels}
    names = {k: f"{_species_of(db, k[0])}{k[1]}+ (m{k[0]})" for k in keys}

    # 1) distributions on a common grid (the stored curve is the linear interpolation
    #    between the measured points; resampled to n_grid points)
    t1 = np.asarray(ck["corrected_tof_1m"], float)
    g = np.linspace(t1.min(), t1.max(), n_grid)
    cols = {"t_1m_us_per_m": g * 1e6, "t_us_at_L_EIA": g * L * 1e6}
    for k in keys:
        cols[f"{names[k]} signal_1m (yield x z)"] = np.interp(g, t1, ck["corrected_spectra_1m"][k])
    for k in keys:
        cols[f"{names[k]} per_ion_1m (yield)"] = cols[f"{names[k]} signal_1m (yield x z)"] / k[1]
    f1 = out_dir / f"{run.key}_Jz_vs_time.csv"
    head = _header(run, L, ck, [
        "columns: t_1m in us per metre of flight (= t / L_EIA), t at L_EIA in us; then for each "
        "charge state the signal at 1 m ('yield x z') and the same / z ('per ion')",
        f"curves: linear interpolation between the measured points, {n_grid} points; "
        "target species masses: " + ", ".join(f"m{m}" for m in sorted(target)),
        "density per unit time: integrate over t_1m (in s) for the yield of a charge state",
    ])
    f1.write_text(head + pd.DataFrame(cols).to_csv(index=False, float_format="%.8g"),
                  encoding="utf-8")

    # 2) measured points
    corr = ck.get("CFY_1m", np.nan) * ck.get("CFArea", np.nan) * ck.get("CFCurrent", np.nan)
    rows = []
    for k in keys:
        if k not in ck.get("corrected_exp_tof", {}):
            continue
        m_lab, z = k
        sp = _species_of(db, m_lab)
        m_kg = db.mass_kg(sp, prof) if sp in db.species else m_lab * amu
        t = np.asarray(ck["corrected_exp_tof"][k], float)          # s at L_EIA
        sig_L = -np.asarray(ck["corrected_exp_points"][k], float)   # signal at L_EIA (>0)
        sig_1m = sig_L * L ** 3
        U = np.asarray(ck.get("corrected_exp_U", {}).get(k, np.full(len(t), np.nan)), float)
        v = L / t
        E = 0.5 * m_kg * v ** 2 / e
        t1m = t / L
        per_ion = sig_1m / z
        rows.append(pd.DataFrame({
            "species": sp, "m_label": m_lab, "z": z, "target": m_lab in target, "U_V": U,
            "t_us_at_L_EIA": t * 1e6, "t_1m_us_per_m": t1m * 1e6, "v_km_s": v / 1e3,
            "E_eV": E, "E_per_z_eV": E / z,
            "raw_EIA_yield_V": sig_1m / corr / z if np.isfinite(corr) else np.nan,
            "signal_1m (yield x z)": sig_1m, "per_ion_1m (yield)": per_ion,
            "per_ion_per_eV_1m (dN/dE)": per_ion * t1m / (2 * E)}))
    pts = (pd.concat(rows, ignore_index=True).sort_values(["m_label", "z", "t_us_at_L_EIA"])
           if rows else pd.DataFrame())
    f2 = out_dir / f"{run.key}_points.csv"
    head = _header(run, L, ck, [
        "one row per measured point (one fitted peak of one deflector voltage U)",
        "raw_EIA_yield_V: size of the EIA (SEM) peak, no calibration, not x z",
        "v = L_EIA / t, E = m v^2 / 2 (eV, per ion), E_per_z = E / z",
        "per_ion_per_eV_1m = per ion x |dt/dE| = per ion x t_1m[s] / (2 E[eV]) "
        "(the Jacobian: the same ions per eV instead of per unit time)",
        "U_V empty: the calibration predates the voltage record - run the run again",
    ])
    f2.write_text(head + pts.to_csv(index=False, float_format="%.8g"), encoding="utf-8")
    return [f1, f2]


def export_study(db, study_name: str, processed: Path, studies_root: Path,
                 n_grid: int = 2000, log=print) -> list[Path]:
    """CSV files of every run of the study (studies/<study>/export/)."""
    st = db.study(study_name)
    out = Path(studies_root) / study_name / "export"
    files = []
    keys = st.all_runs() if st.include_extra else st.runs
    for key in keys:
        run = db.run(key)
        h5 = Path(processed) / key / "20_calibration" / "spectra.h5"
        if not h5.is_file():
            log(f"  {key}: no calibration output ({h5}) - run the run first; skipped")
            continue
        fs = export_run(db, run, load_calibration_h5(h5), out, n_grid)
        files += fs
        log(f"  {key}: {fs[0].name}, {fs[1].name}")
    (out / "README.txt").write_text(__doc__.strip() + "\n", encoding="utf-8")
    log(f"  -> {out}")
    return files
