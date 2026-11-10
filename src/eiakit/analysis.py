"""
Spectra analysis: port of plot_eia_powerLaw_test.py (computations only, no plotting).

Input: a checkpoint-like dict (the calibration stage spectra.h5, or a legacy
pipeline_checkpoint.npy) with J_z(t) per (m, z) in A cm^-2 at L_EIA, charge-weighted.
    N_z(t) = J_z / (z e)                         number flux per charge state
    zbar(t) = sum z N_z / sum N_z                at fixed t all z share E = M (L/t)^2 / 2
    zbar = C E^b fitted in log-log inside the study window (E_max per element, lower
           edge raised to where z = lower_from_z starts)
    Sheil-style spectra d2N/(dE dOmega) = N_z |dt/dE| L^2   [ions eV^-1 sr^-1 shot^-1]
Constants and mass: db.profile_for(run, 'analysis') (legacy: amu 1.66053907e-27,
e = 1.602e-19; 'corrected' with [corrections] mass_constants = "codata");
exclusions ('all' lists of the study) remove charge states everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class Dataset:
    key: str                     # run key
    name: str                    # e.g. 'Al_780mJ'
    species: str
    energy_mJ: float | None
    m_kg: float
    e_C: float
    L_m: float
    tof_us: np.ndarray
    z_curves: dict               # z -> J_z (A cm^-2), excluded z removed
    z_exp: dict = field(default_factory=dict)   # z -> (t_us, J) raw points, sign-flipped
    all_z: list = field(default_factory=list)
    excluded_z: list = field(default_factory=list)
    masses_used: list = field(default_factory=list)
    N_sec9: dict | None = None   # z -> ions cm^-2 from integrals_eia (cross-check)

    @property
    def title(self):
        mj = f", {self.energy_mJ:g} mJ" if self.energy_mJ is not None else ""
        return f"{self.species}{mj}"


def load_dataset(db, run, ck: dict, study=None, profile: str | None = None,
                 mass_labels=None) -> Dataset:
    """Per-z curves from a checkpoint, as load_species() of the legacy script."""
    prof = profile or db.profile_for(run, "analysis")
    labels = mass_labels if mass_labels is not None else run.mass_labels
    keep = None if labels is None else {int(m) for m in labels}
    spectra = ck["corrected_spectra"]
    found = sorted({int(m) for (m, _) in spectra})
    if keep is not None:
        spectra = {k: v for k, v in spectra.items() if int(k[0]) in keep}
    zs = sorted({int(z) for (_, z) in spectra})
    excl = []
    if study is not None:
        excl = sorted(study.exclude.get(run.key, {}).get("all", []))
    curves = {}
    for z in zs:
        if z in excl:
            continue
        c = sum(np.asarray(a, float) for (m, zz), a in spectra.items() if zz == z)
        curves[z] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    e_C = db.e_C(prof)
    z_exp = {}
    pts, ptof = ck.get("corrected_exp_points"), ck.get("corrected_exp_tof")
    if pts is not None and ptof is not None:
        for z in curves:
            t = [np.asarray(ptof[k], float) for k in pts if k[1] == z
                 and (keep is None or int(k[0]) in keep)]
            y = [np.asarray(pts[k], float) for k in pts if k[1] == z
                 and (keep is None or int(k[0]) in keep)]
            if t:
                t, y = np.concatenate(t), np.concatenate(y)
                o = np.argsort(t)
                z_exp[z] = (t[o] * 1e6, -y[o])
    sec9 = None
    ie = ck.get("integrals_eia")
    if isinstance(ie, dict):
        sec9 = {}
        for (m, z), v in ie.items():
            if keep is None or int(m) in keep:
                sec9[int(z)] = sec9.get(int(z), 0.0) + float(v) / (int(z) * e_C)
    mj = run.laser.energy_mJ if getattr(run, "laser", None) else None
    return Dataset(key=run.key, name=f"{run.target}_{mj:g}mJ" if mj else run.key,
                   species=run.target, energy_mJ=mj, m_kg=db.mass_kg(run.target, prof),
                   e_C=e_C, L_m=run.eia.L_m, tof_us=np.asarray(ck["corrected_tof"]) * 1e6,
                   z_curves=curves, z_exp=z_exp, all_z=sorted(curves), excluded_z=excl,
                   masses_used=sorted(keep) if keep is not None else found, N_sec9=sec9)


# ---- axes ------------------------------------------------------------------------------

def velocity_kms(ds: Dataset, t_us):
    return ds.L_m / (np.asarray(t_us) * 1e-6) / 1e3


def energy_keV(ds: Dataset, u_kms):
    u = np.asarray(u_kms) * 1e3
    return 0.5 * ds.m_kg * u ** 2 / ds.e_C / 1e3


def energy_eV(ds: Dataset, t_us):
    return energy_keV(ds, velocity_kms(ds, t_us)) * 1e3


# ---- per-z summary ---------------------------------------------------------------------

def summarize(ds: Dataset) -> pd.DataFrame:
    rows = []
    t = ds.tof_us
    for z in ds.all_z:
        y = ds.z_curves[z]
        i = int(np.argmax(y))
        area = float(_trapz(y, t))
        u = float(velocity_kms(ds, t[i]))
        E = float(energy_keV(ds, u))
        rows.append({"z": z, "t_peak_us": float(t[i]), "J_peak": float(y[i]), "u_peak_kms": u,
                     "E_peak_keV": E, "EperQ_peak_kV": E / z,
                     "t_centroid_us": float(_trapz(y * t, t) / area) if area else np.nan,
                     "area": area})
    return pd.DataFrame(rows)


# ---- average charge state ---------------------------------------------------------------

def missing_ions(ds: Dataset, t_us, zs=None, mode: str = "edge"):
    """I4: signal of each z outside its measured range, which the zero fill of untitled27
    drops (same units as the curves). mode 'edge': continued at its value at the first /
    last measured point (upper bound, the spectra decay outwards); 'extrapolate': power law
    in t through the 3 outer points, decaying at least as t^+1 (early side) / t^-1 (late)."""
    t_us = np.asarray(t_us)
    miss = np.zeros(t_us.shape)
    for z, (te, ye) in ds.z_exp.items():
        if z not in ds.z_curves or len(te) < 2 or (zs is not None and z not in zs):
            continue
        y = np.abs(np.asarray(ye, float))
        for side in ("lo", "hi"):
            sel = slice(0, 3) if side == "lo" else slice(-3, None)
            t0, y0 = (te[0], y[0]) if side == "lo" else (te[-1], y[-1])
            out = t_us < t0 if side == "lo" else t_us > t0
            if not out.any() or y0 <= 0:
                continue
            if mode == "edge":
                miss[out] += y0
                continue
            tt, yy = np.asarray(te[sel], float), y[sel]
            ok = (yy > 0) & (tt > 0)
            s = np.polyfit(np.log(tt[ok]), np.log(yy[ok]), 1)[0] if ok.sum() >= 2 else 0.0
            s = max(s, 1.0) if side == "lo" else min(s, -1.0)
            miss[out] += y0 * (t_us[out] / t0) ** s
    return miss


def compute_zbar(ds: Dataset, jacobian: bool = False, z_list=None, min_frac: float = 0.0,
                 clip: bool = True, mask: str = "none", mask_tol: float = 0.05,
                 fill: str = "zero"):
    """zbar(E) = sum z N_z / sum N_z.  min_frac: noise floor (I5, valid where N_tot >
    min_frac max N_tot); clip: negative J set to 0 (legacy, I5); mask = 'measured':
    invalid where the ions possibly dropped by the zero fill outside the measured ranges
    exceed mask_tol of the total (I4); fill = 'extrapolate' / 'edge': every z continued
    beyond its measured range (missing_ions) instead of the zero fill (I4)."""
    t = ds.tof_us
    zs = [z for z in ds.all_z if z_list is None or z in z_list]
    E_keV = energy_keV(ds, velocity_kms(ds, t))
    jac = (t * 1e-6) / (2.0 * E_keV) if jacobian else 1.0
    Nz = {z: (np.clip(ds.z_curves[z], 0.0, None) if clip else ds.z_curves[z])
          / (z * ds.e_C) * jac for z in zs}
    if fill in ("edge", "extrapolate"):
        Nz = {z: Nz[z] + missing_ions(ds, t, [z], fill) / (z * ds.e_C) * jac for z in zs}
    elif fill != "zero":
        raise ValueError(f"zbar_fill = {fill!r}: use 'zero', 'extrapolate' or 'edge'")
    Ntot = sum(Nz.values())
    Nw = sum(z * Nz[z] for z in zs)
    with np.errstate(divide="ignore", invalid="ignore"):
        zbar = np.where(Ntot > 0, Nw / Ntot, np.nan)
    E = E_keV * 1e3
    valid = Ntot > min_frac * np.nanmax(Ntot)
    if mask == "measured":
        miss = sum(missing_ions(ds, t, [z]) / (z * ds.e_C) * jac for z in zs)
        with np.errstate(divide="ignore", invalid="ignore"):
            valid &= miss <= mask_tol * (Ntot + miss)
    elif mask != "none":
        raise ValueError(f"zbar_mask = {mask!r}: use 'none' or 'measured'")
    o = np.argsort(E)
    return {"E_eV": E[o], "zbar": zbar[o], "Ntot": Ntot[o], "valid": valid[o], "zs": zs,
            "Nz": {z: Nz[z][o] for z in zs}}


def zbar_options(analysis: dict | None) -> dict:
    """compute_zbar keyword arguments from db [analysis] (defaults = legacy)."""
    a = analysis or {}
    return {"min_frac": float(a.get("zbar_min_frac", 0.0)),
            "clip": bool(a.get("zbar_clip_negative", True)),
            "mask": a.get("zbar_mask", "none"),
            "mask_tol": float(a.get("zbar_mask_tol", 0.05)),
            "fill": a.get("zbar_fill", "zero")}


def fit_power_law(E_eV, zbar, valid, e_range=None):
    m = valid & np.isfinite(zbar) & (zbar > 0) & (E_eV > 0)
    if e_range is not None:
        lo, hi = e_range
        if lo is not None:
            m &= E_eV >= lo
        if hi is not None:
            m &= E_eV <= hi
    if m.sum() < 5:
        return None
    x, y = np.log10(E_eV[m]), np.log10(zbar[m])
    (b, a), cov = np.polyfit(x, y, 1, cov=True)
    res = y - (a + b * x)
    r2 = 1.0 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2)
    sb = float(np.sqrt(cov[0, 0]))
    return {"C": float(10 ** a), "b": float(b), "sig_b": sb, "n": float(1 / b),
            "sig_n": sb / b ** 2, "r2": float(r2), "E_lo": float(E_eV[m].min()),
            "E_hi": float(E_eV[m].max()), "npts": int(m.sum())}


def zbar_fit_range(ds: Dataset, zb: dict, settings) -> tuple:
    """(E_lo, E_hi, E_start): study window, lower edge raised to where z = lower_from_z
    first has N_z > 0."""
    lo = settings.e_min_eV.get(ds.name, settings.e_min_eV.get(ds.species))
    hi = settings.e_max_eV.get(ds.name, settings.e_max_eV.get(ds.species))
    E_start = None
    zf = settings.lower_from_z
    if zf is not None and zf in zb["Nz"]:
        pos = zb["Nz"][zf] > 0
        if pos.any():
            E_start = float(zb["E_eV"][pos].min())
            lo = E_start if lo is None else max(lo, E_start)
    return lo, hi, E_start


def analyze_zbar(ds: Dataset, settings, analysis: dict | None = None) -> dict:
    opt = zbar_options(analysis)
    zb = compute_zbar(ds, **opt)
    zj = compute_zbar(ds, jacobian=True, **opt)
    both = zb["valid"] & zj["valid"] & np.isfinite(zb["zbar"]) & np.isfinite(zj["zbar"])
    jac_rel = float(np.max(np.abs(zj["zbar"][both] / zb["zbar"][both] - 1))) if both.any() \
        else np.nan
    lo, hi, E_start = zbar_fit_range(ds, zb, settings)
    fit = fit_power_law(zb["E_eV"], zb["zbar"], zb["valid"], (lo, hi))
    return {"zb": zb, "fit": fit, "E_start": E_start, "jacobian_check_rel": jac_rel}


def reference_prefactor(records, exponent: float = 0.4, default: float = 0.3) -> dict:
    """Sheil-type line C E^exponent through all conditions: C = geometric mean of
    zbar / E^exponent over the points inside each condition's fit window."""
    pE, pz = [], []
    for r in records:
        zb, f = r["zb"], r["fit"]
        if f is None:
            continue
        v = zb["valid"]
        E, z = zb["E_eV"][v], zb["zbar"][v]
        m = (E >= f["E_lo"]) & (E <= f["E_hi"]) & np.isfinite(z) & (z > 0)
        pE.append(E[m])
        pz.append(z[m])
    if not pE:
        return {"C": default, "exponent": exponent, "npts": 0}
    E, z = np.concatenate(pE), np.concatenate(pz)
    return {"C": float(10 ** np.mean(np.log10(z) - exponent * np.log10(E))),
            "exponent": exponent, "npts": int(E.size)}


# ---- Sheil-style spectra ---------------------------------------------------------------

def compute_sheil(ds: Dataset, zs=None) -> dict:
    zs = ds.all_z if zs is None else zs
    t_s = ds.tof_us * 1e-6
    E_eV = energy_eV(ds, ds.tof_us)
    jac = t_s / (2.0 * E_eV)
    L2 = (ds.L_m * 100.0) ** 2
    o = np.argsort(E_eV)
    E = E_eV[o]
    f, n_t = {}, {}
    for z in zs:
        N = np.clip(ds.z_curves[z], 0.0, None) / (z * ds.e_C)
        f[z] = (N * jac * L2)[o]
        n_t[z] = float(_trapz(N, t_s)) * L2 * (1.0 if t_s[-1] > t_s[0] else -1.0)
    n_E = {z: float(_trapz(f[z], E)) for z in zs}
    total = sum(f.values())
    tot_sr = sum(n_E.values())
    dt_rel = max((abs(n_E[z] / n_t[z] - 1) for z in zs
                  if n_t[z] > 0 and n_E[z] >= 0.01 * tot_sr), default=np.nan)
    d9 = np.nan
    if ds.N_sec9:
        d9 = max((abs(n_E[z] / L2 / ds.N_sec9[z] - 1) for z in zs
                  if ds.N_sec9.get(z, 0) > 0 and n_E[z] >= 0.01 * tot_sr), default=np.nan)
    return {"E": E, "f": f, "total": total, "n_E": n_E, "n_t": n_t, "L2": L2,
            "peak_total": float(total.max()), "E_peak_total": float(E[np.argmax(total)]),
            "yield_sr": float(tot_sr), "check_E_vs_t": float(dt_rel), "check_sec9": float(d9)}
