"""
Velocity collapse u2 / u1 = z / z*  (port of plot_collapse_v2.py computations).

Two peak sources per dataset (run):
  'eia'  fast / slow / ultra-fast peaks located on the calibrated J_z(t) curves
         (scipy find_peaks, prominence > peak_min_prom_frac * max; slow = latest peak,
         ultra-fast = earliest of >= 2 earlier peaks, fast = highest of the rest;
         bounds at half maximum ('hwhm', searched up to the adjacent peak) or the
         neighbour samples ('triplet'))
  'cbs'  the fitted components of the CBS file (physical names from the db), t from the
         raw PEAK sample ('exp') or the picks ('fit'), bounds = prev / next samples
u1 = mean fast velocity of the dataset (>= min_fast_points fast peaks), u2 = slow peak
velocity; fit through the origin u2/u1 = s z, z* = 1/s; joint z* = inverse-variance
weighted mean. Slow-branch energy E2 = m u2^2 / 2 fitted as A z^n, linear, quadratic.
Exclusions: study [exclude] ('all' and per-component lists), applied once after the peaks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

COLS = ["run", "element", "energy_mJ", "z", "component", "t_before", "t_peak", "t_after",
        "distance_m"]


def _peak_bounds(t, y, i, left, right, mode):
    n = len(t)
    if mode == "triplet":
        return t[max(i - 1, 0)], t[min(i + 1, n - 1)]
    half = 0.5 * y[i]

    def walk(step, limit):
        j = i
        while j != limit and y[j + step] > half:
            j += step
        k = j + step if j != limit else j
        if j != limit and y[k] <= half and y[j] > y[k]:
            f = (y[j] - half) / (y[j] - y[k])
            return t[j] + f * (t[k] - t[j])
        return t[j]

    return walk(-1, left), walk(+1, right)


def eia_peaks(ds, prom_frac: float = 0.05, err_mode: str = "hwhm") -> pd.DataFrame:
    """Peaks of every J_z(t) curve of an analysis.Dataset (already mass-filtered)."""
    t = ds.tof_us
    rows = []
    for z in ds.all_z:
        y = np.clip(ds.z_curves[z], 0.0, None)
        ymax = float(y.max())
        if ymax <= 0:
            continue
        idx, _ = find_peaks(y, prominence=prom_frac * ymax)
        if len(idx) == 0:
            continue
        idx = np.sort(idx)
        slow_i = int(idx[-1])
        earlier = idx[:-1]
        ultra_i = None
        if len(earlier) >= 2:
            ultra_i, earlier = int(earlier[0]), earlier[1:]
        fast_i = int(earlier[np.argmax(y[earlier])]) if len(earlier) else None
        for comp, i in (("slow", slow_i), ("fast", fast_i), ("ultra_fast", ultra_i)):
            if i is None:
                continue
            k = int(np.searchsorted(idx, i))
            lo = int(idx[k - 1]) if k > 0 else 0
            hi = int(idx[k + 1]) if k + 1 < len(idx) else len(t) - 1
            tb, ta = _peak_bounds(t, y, i, lo, hi, err_mode)
            rows.append(dict(run=ds.key, element=ds.species, energy_mJ=ds.energy_mJ, z=z,
                             component=comp, t_before=float(tb), t_peak=float(t[i]),
                             t_after=float(ta), distance_m=ds.L_m))
    return pd.DataFrame(rows, columns=COLS)


def cbs_peaks(db, run, data_root, t_source: str = "exp") -> pd.DataFrame:
    """Fitted components of the run's CBS file: PEAK sample ('exp') or picked t_peak."""
    from .audit import _find_cbs_file
    from .legacy import parse_cbs_fit_results

    res = run.folder(data_root) / "EIA-Results"
    p = _find_cbs_file(res, "CBS_fit_results", run.fit_file_labels, run.target)
    fit = parse_cbs_fit_results(p)
    L = fit.header.L_EIA_m or run.eia.L_m
    names = run.legacy.cbs_components
    rows = []
    for c in fit.components:
        comp = names.get(c.cbs)
        if comp not in ("fast", "slow", "ultra_fast") or not np.isfinite(c.kT):
            continue
        tr = c.triplet
        t_pk = tr.get("PEAK", {}).get("t", np.nan) if t_source == "exp" else \
            c.picks.get("t_peak", np.nan)
        if not np.isfinite(t_pk):
            continue
        rows.append(dict(run=run.key, element=run.target, energy_mJ=run.laser.energy_mJ,
                         z=c.z, component=comp,
                         t_before=tr.get("prev", {}).get("t", np.nan), t_peak=t_pk,
                         t_after=tr.get("next", {}).get("t", np.nan), distance_m=L))
    return pd.DataFrame(rows, columns=COLS)


def drop_excluded(df: pd.DataFrame, exclude: dict) -> pd.DataFrame:
    """exclude: {run: {'all' | component: [z, ...]}} (study [exclude])."""
    drop = pd.Series(False, index=df.index)
    for key, ex in exclude.items():
        sel = df.run == key
        for comp, zs in ex.items():
            if comp == "all":
                drop |= sel & df.z.isin(zs)
            else:
                drop |= sel & (df.component == comp) & df.z.isin(zs)
    return df[~drop].reset_index(drop=True)


def fit_through_origin(z, ratio):
    z, ratio = np.asarray(z, float), np.asarray(ratio, float)
    n = len(z)
    s = np.sum(z * ratio) / np.sum(z ** 2)
    pred = s * z
    resid = ratio - pred
    R2 = 1 - np.sum(resid ** 2) / np.sum((ratio - ratio.mean()) ** 2)
    if n > 1:
        se_s = np.sqrt(np.sum(resid ** 2) / (n - 1) / np.sum(z ** 2))
        se_z = se_s / s ** 2
    else:
        se_s = se_z = np.nan
    return {"zstar": 1 / s, "R2": R2, "se_zstar": se_z, "s": s, "se_s": se_s}


def weighted_joint(zstars, ses):
    z, se = np.asarray(zstars, float), np.asarray(ses, float)
    v = np.isfinite(z) & np.isfinite(se) & (se > 0)
    z, se = z[v], se[v]
    if not len(z):
        return {"zstar": np.nan, "se": np.nan, "rms_pct": np.nan, "n": 0}
    w = 1 / se ** 2
    zj = np.sum(w * z) / np.sum(w)
    rms = 100 * np.sqrt(np.mean((z - zj) ** 2)) / zj if len(z) > 1 else 0.0
    return {"zstar": zj, "se": 1 / np.sqrt(np.sum(w)), "rms_pct": rms, "n": int(len(z))}


def energy_fits(z, u2_kms, m_kg, e_C):
    z, u2 = np.asarray(z, float), np.asarray(u2_kms, float)
    E = 0.5 * m_kg * (u2 * 1e3) ** 2 / e_C / 1e3            # keV

    def r2(y, p):
        ss = np.sum((y - y.mean()) ** 2)
        return 1 - np.sum((y - p) ** 2) / ss if ss > 0 else np.nan

    lz, lE = np.log(z), np.log(E)
    n, lA = np.polyfit(lz, lE, 1)
    m1, b1 = np.polyfit(z, E, 1)
    out = {"z": z, "E2_keV": E, "n_pow": n, "A_pow": np.exp(lA), "R2_pow": r2(lE, n * lz + lA),
           "m1": m1, "b1": b1, "R2_lin": r2(E, m1 * z + b1)}
    if len(z) >= 3:
        a2, b2, c2 = np.polyfit(z, E, 2)
        out.update(a2=a2, b2=b2, c2=c2, R2_quad=r2(E, a2 * z ** 2 + b2 * z + c2))
    else:
        out.update(a2=np.nan, b2=np.nan, c2=np.nan, R2_quad=np.nan)
    return out


def collapse_series(peaks: pd.DataFrame, runs, masses: dict, e_C: float,
                    min_fast_points: int = 2, log=None) -> list[dict]:
    """Per run: z, u2/u1 with asymmetric bars, z* fit and energy fits.
    masses: run -> ion mass (kg) for the energy fits."""
    out = []
    for key in runs:
        sub = peaks[peaks.run == key]
        if sub.empty:
            log and log(f"  [collapse] {key}: no peaks, skipped")
            continue
        d = sub.distance_m.iloc[0]

        def u(t, d=d):
            return 1000.0 * d / np.asarray(t, float)

        fast = sub[sub.component == "fast"].sort_values("z")
        u1 = float(u(fast.t_peak).mean()) if len(fast) else np.nan
        if not np.isfinite(u1) or len(fast) < min_fast_points:
            log and log(f"  [collapse] {key}: {len(fast)} fast point(s) "
                        f"(< {min_fast_points}) -> u1 not reliable, skipped")
            continue
        slow = sub[sub.component == "slow"].sort_values("z")
        if len(slow) < 2:
            log and log(f"  [collapse] {key}: fewer than 2 slow points, skipped")
            continue
        z = slow.z.to_numpy(float)
        u2, ub, ua = u(slow.t_peak), u(slow.t_before), u(slow.t_after)
        r = u2 / u1
        f = fit_through_origin(z, r)
        out.append({"run": key, "element": sub.element.iloc[0],
                    "energy_mJ": sub.energy_mJ.iloc[0], "z": z, "ratio": r, "u1": u1,
                    "u2": u2, "lo": np.nan_to_num((u2 - ua) / u1, nan=0.0),
                    "hi": np.nan_to_num((ub - u2) / u1, nan=0.0), "n_fast": len(fast),
                    **f, "efit": energy_fits(z, u2, masses[key], e_C)})
    return out


def literature_series(lit: dict) -> list[dict]:
    """Poirier (Sn) and Kondrashev (Nb) collapse fits from literature.toml entries."""
    out = []
    for key, L in lit.items():
        if "E_slow_eV" in L:
            c = L["velocity_const_kms_per_sqrt_eV"]
            u2 = c * np.sqrt(np.asarray(L["E_slow_eV"]))
            u1 = float((c * np.sqrt(np.asarray(L["E_fast_eV"]))).mean())
        elif "E_slow_keV" in L:
            c, A = L["velocity_const_kms"], L["species_mass_amu"]
            u2 = c * np.sqrt(np.asarray(L["E_slow_keV"]) / A)
            u1 = float((c * np.sqrt(np.asarray(L["E_fast_keV"]) / A)).mean())
        else:
            continue
        z = np.asarray(L["z"], float)
        out.append({"key": key, "label": L.get("label", key), "z": z, "ratio": u2 / u1,
                    "u1": u1, **fit_through_origin(z, u2 / u1)})
    return out
