"""
CBS component table of a study (port of plot_cbs_multipanel_v2 data handling).

One row per (run, z, component) with the physical component name from the run's
legacy.cbs_components (the legacy scripts' REMAP), velocities u = L / t with L from the
fit-file header:
    u_peak, u_prev, u_next   raw sample triplet prev / PEAK / next of the text block
                             ('velocity' figure; error bar = shift to a neighbour sample)
    u_pick, u_left, u_right  picked peak and picked half-maximum points ('picks')
    u_fit, t_peak_fit_us     peak of the fitted model ('fit_summary'), with FWHM_fit_us
    Ions_model_full_cm2, FWHM_fit_us   from the areas table ('area_fwhm')
Exclusions of the study: 'all' and per-component lists (physical names).
Areas: source 'legacy' reads the CBS_areas file, 'pipeline' the cbs stage output
(current calibration); velocities do not depend on the calibration.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _u(L, t_us):
    t = np.asarray(t_us, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(t) & (t > 0), L / (t * 1e-6) / 1e3, np.nan)


def run_components(db, run, data_root, areas: pd.DataFrame | None = None,
                   study=None) -> pd.DataFrame:
    from .audit import _find_cbs_file
    from .cbs_store import default_fits_root, load_fit
    from .legacy import parse_cbs_areas

    res = run.folder(data_root) / "EIA-Results"
    fit = load_fit(db, run, data_root, default_fits_root())
    if fit is None:
        raise FileNotFoundError(f"{run.key}: no CBS_fit_results file")
    L = fit.header.L_EIA_m or run.eia.L_m
    if areas is None:
        ap = _find_cbs_file(res, "CBS_areas", run.fit_file_labels, run.target)
        areas = parse_cbs_areas(ap).table if ap else pd.DataFrame(columns=["z", "CBS"])
    names = run.cbs_names
    excl = (study.exclude.get(run.key, {}) if study is not None else {})
    prof = db.profile_for(run, "cbs")
    m_kg, e_C = db.mass_kg(run.target, prof), db.e_C(prof)
    gamma = float(db.pipeline["cbs"].get("gamma", 5 / 3))
    a_idx = areas.set_index(["z", "CBS"]) if len(areas) else None
    rows = []
    for r in fit.table.itertuples():
        z, n = int(r.z), int(r.CBS)
        comp = names.get(n, f"cbs{n}")
        blk = fit.component(z, n)
        tr = blk.triplet if blk else {}
        t_pk = tr.get("PEAK", {}).get("t", np.nan)
        row = {"run": run.key, "species": run.target, "energy_mJ": run.laser.energy_mJ,
               "z": z, "cbs_file": n, "component": comp, "L_m": L,
               "u_peak": float(_u(L, t_pk)), "u_prev": float(_u(L, tr.get("prev", {})
                                                                .get("t", np.nan))),
               "u_next": float(_u(L, tr.get("next", {}).get("t", np.nan))),
               "u_pick": float(_u(L, r.t_peak_pick)), "u_left": float(_u(L, r.t_left_pick)),
               "u_right": float(_u(L, r.t_right_pick)), "t_peak_fit_us": float(r.t_peak_fit),
               "u_fit": float(_u(L, r.t_peak_fit))}
        row.update(_peak_velocities(getattr(r, "kT_eV", np.nan), getattr(r, "V0_V", np.nan),
                                    z, m_kg, e_C, gamma))
        if a_idx is not None and (z, n) in a_idx.index:
            ar = a_idx.loc[(z, n)]
            row["FWHM_fit_us"] = float(ar["FWHM_fit_us"])
            row["Ions_model_full_cm2"] = float(ar["Ions_model_full_cm2"])
        row["excluded"] = bool(z in excl.get("all", []) or z in excl.get(comp, []))
        rows.append(row)
    return pd.DataFrame(rows)


def _peak_velocities(kT, V0, z, m_kg, e_C, gamma) -> dict:
    """km/s of a fitted component: peak of dN/dt, dN/dv, dN/dE, mean velocity, and the
    drift v_s of the shifted Maxwellian (NaN without kT / V0)."""
    from .physics import cbs_density_peak_velocity, cbs_mean_velocity, shift_velocity
    keys = ("u_dNdt", "u_dNdv", "u_dNdE", "u_mean", "v_shift")
    try:
        kT, V0 = float(kT), float(V0)
    except (TypeError, ValueError):
        return dict.fromkeys(keys, np.nan)
    if not (np.isfinite(kT) and np.isfinite(V0) and kT > 0):
        return dict.fromkeys(keys, np.nan)
    pk = {d: float(cbs_density_peak_velocity(kT, V0, z, m_kg, e_C, d, gamma)) / 1e3
          for d in ("t", "v", "E")}
    return {"u_dNdt": pk["t"], "u_dNdv": pk["v"], "u_dNdE": pk["E"],
            "u_mean": float(cbs_mean_velocity(kT, V0, z, m_kg, e_C, gamma)) / 1e3,
            "v_shift": float(shift_velocity(kT, V0, z, m_kg, e_C, gamma)) / 1e3}


def study_components(db, study_name, data_root, processed=None, source="legacy",
                     include_extra=False) -> pd.DataFrame:
    st = db.study(study_name)
    keys = st.runs + (st.extra_runs if include_extra else [])
    tabs = []
    for key in keys:
        run = db.run(key)
        areas = None
        if source == "pipeline":
            p = Path(processed) / key / "30_cbs" / "cbs_areas.csv"
            if not p.is_file():
                raise RuntimeError(f"{key}: no cbs stage output (eiakit run {key!r})")
            areas = pd.read_csv(p)
        tabs.append(run_components(db, run, data_root, areas, st))
    return pd.concat(tabs, ignore_index=True)


def component_means(tab: pd.DataFrame, avg_components: dict, column: str,
                    nan_aware: bool = True) -> pd.DataFrame:
    """mean +/- std (population) of `column` per (run, averaged component), over the
    charge states kept after the exclusions."""
    rows = []
    for (key, sp), g in tab[~tab.excluded].groupby(["run", "species"], sort=False):
        for comp in avg_components.get(sp, []):
            v = g.loc[g.component == comp, column].to_numpy(float)
            if not len(v):
                continue
            m = np.nanmean(v) if nan_aware else np.mean(v)
            s = np.nanstd(v) if nan_aware else np.std(v)
            rows.append({"run": key, "species": sp, "energy_mJ": g.energy_mJ.iloc[0],
                         "component": comp, "column": column, "mean": float(m),
                         "std": float(s), "n": int(len(v))})
    return pd.DataFrame(rows)
