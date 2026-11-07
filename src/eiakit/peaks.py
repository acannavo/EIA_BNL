"""
Step 5: peak stage (port of untitled28_EIA_Analysis_2, without the GUI).

For one deflector voltage U:
    tof, y = average_repeats(traces)                 shots averaged on a common TOF grid
    t0     = predicted_tof(m_amu, z, U, L, eia, prof) analyzer law (the GUI's vertical lines)
    fit    = fit_peak(tof, y, t0, corr, settings)    bounded Gaussian, as 'Fit Gaussians'
    rows   = fit_voltage(...)                        one Output + Output_detail row per (m, z)

The Gaussian g(t) = A exp(-((t - mu)/sigma)^2 / 2) + B is fitted in t0 (1 +/- (0.02 + corr))
with A in [A_guess, 0], mu in t0 +/- 150 ns, sigma in [0, window/6], B <= B_guess (the
bounds of untitled28). The Output row is (m, z, TOF = mu, Yield = A): the peak height,
not the area (I6).

Not stored by untitled28 and therefore parameters here: the U actually used (arrow
buttons), LOF, Corr, and the mass / z-max lists typed in the GUI.

Replay (replay_voltage): the GUI pre-fills U with the file voltage p and moves it by
p/500 per arrow click; the fit windows follow the U in the box, not the file voltage.
The number of clicks is recovered by refitting with n = 0, -1, +1, -2, ... until every
Output_detail row is reproduced (TOF and YieldAveFit to 1e-9 relative). On 20250611
(U = 30, 35: -7 clicks) and 2026_1103 (U = 18: +5, U = 24: -2) every row is reproduced
bit for bit. Only TOF = mu and Yield = A enter the later stages, and the energy is
still computed from the file voltage, so the clicks shift only the fit windows (N16).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import physics as ph


def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B


def average_repeats(traces, n_points: int | None = None):
    """
    Repeats of one U interpolated on a common grid (min..max of all their times) and
    averaged. n_points defaults to the longest trace (untitled28 PSexp; the 2025 version
    used the length of the last file read, identical when all records have 10000 points).
    """
    t_all = np.concatenate([t for t, _ in traces])
    n = n_points or max(len(t) for t, _ in traces)
    grid = np.linspace(t_all.min(), t_all.max(), n)
    ys = [np.interp(grid, t, y) for t, y in traces]
    return grid, np.vstack(ys).mean(axis=0)


def predicted_tof(m_amu: float, z: int, U: float, L_m: float, eia, profile) -> float:
    """Time of flight where untitled28 draws the line for (m, z) at deflector voltage U."""
    return float(ph.tof_from_deflector(U, z, m_amu * profile.amu_kg, L_m, eia.R_out,
                                       eia.R_in, profile.e_C, eia.R_orbit_m))


def fit_peak(tof, y, t0: float, corr: float = 0.0, window_frac: float = 0.02,
             mu_bound_s: float = 150e-9, min_points: int = 1) -> dict:
    """One bounded Gaussian fit around t0 (fit_gaussians_to_data of untitled28)."""
    half = t0 * (window_frac + corr)
    lo, hi = t0 - half, t0 + half
    mask = (tof >= lo) & (tof <= hi)
    x, yy = tof[mask], y[mask]
    res = {"t0": t0, "win_lo": lo, "win_hi": hi, "n_points": int(mask.sum()),
           "A": np.nan, "mu": np.nan, "sigma": np.nan, "B": np.nan, "success": False,
           "A_at_bound": False, "sigma_at_bound": False, "mu_at_bound": False}
    if len(x) < max(min_points, 1):
        res["reason"] = "too few points"
        return res
    B_guess = float(np.max(yy))
    A_guess = -abs(float(np.min(yy)) - B_guess)
    sigma_guess = (hi - lo) / 6
    p0 = (A_guess, t0, sigma_guess, B_guess)
    lower = [A_guess, t0 - mu_bound_s, 0, -np.inf]
    upper = [0, t0 + mu_bound_s, sigma_guess, B_guess]
    try:
        popt, _ = curve_fit(gaussian, x, yy, p0=p0, bounds=(lower, upper))
    except (RuntimeError, ValueError, TypeError) as exc:
        res["reason"] = f"fit failed: {exc}"
        return res
    A, mu, sigma, B = (float(v) for v in popt)
    res.update(A=A, mu=mu, sigma=sigma, B=B, success=True,
               A_at_bound=bool(np.isclose(A, A_guess, rtol=1e-6)),
               sigma_at_bound=bool(np.isclose(sigma, sigma_guess, rtol=1e-6)),
               mu_at_bound=bool(abs(abs(mu - t0) - mu_bound_s) < 1e-12))
    return res


def fit_voltage(traces_by_rep: dict[int, tuple], U: float, lines: list[tuple[float, int, int]],
                L_m: float, eia, profile, settings: dict, corr: float = 0.0,
                n_points: int | None = None) -> pd.DataFrame:
    """
    All Gaussian fits of one deflector voltage.
    lines: (m_amu typed in the GUI, mass label of the files, z).
    Returns one row per line with the Output columns (m, z, TOF, Yield), the Output_detail
    columns (YieldAveFit, YieldAve, YieldSTDV, Yield1..n) and the fit diagnostics.
    """
    reps = sorted(traces_by_rep)
    tof, y = average_repeats([traces_by_rep[r] for r in reps], n_points)
    out = []
    for m_amu, label, z in lines:
        t0 = predicted_tof(m_amu, z, U, L_m, eia, profile)
        f = fit_peak(tof, y, t0, corr, settings.get("window_frac", 0.02),
                     settings.get("mu_bound_s", 150e-9), settings.get("min_points", 1))
        row = {"m": label, "z": z, "TOF": f["mu"], "Yield": f["A"], **{
            k: f[k] for k in ("t0", "sigma", "B", "n_points", "success", "A_at_bound",
                              "sigma_at_bound", "mu_at_bound")}}
        if f["success"]:
            vals = [float(np.interp(f["mu"], *traces_by_rep[r])) for r in reps]
            row.update(YieldAveFit=f["A"], YieldAve=float(np.mean(vals)),
                       YieldSTDV=float(np.std(vals)))
            row.update({f"Yield{i + 1}": v for i, v in enumerate(vals)})
        out.append(row)
    return pd.DataFrame(out)


def compare_with_output(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """
    Fitted rows of one voltage vs the legacy Output rows of the same file. When a (m, z)
    appears several times in the legacy file (N1), the closest copy is used and the
    number of copies is reported.
    """
    rows = []
    for r in new.itertuples():
        cand = old[(old.m == r.m) & (old.z == r.z)]
        if cand.empty or not np.isfinite(r.TOF):
            rows.append({"m": r.m, "z": r.z, "copies": len(cand), "dTOF_rel": np.nan,
                         "dYield_rel": np.nan, "match": False})
            continue
        d = (cand.TOF / r.TOF - 1).abs() + (cand.Yield / r.Yield - 1).abs()
        best = cand.loc[d.idxmin()]
        dt, dy = abs(best.TOF / r.TOF - 1), abs(best.Yield / r.Yield - 1)
        rows.append({"m": r.m, "z": r.z, "copies": len(cand), "dTOF_rel": dt,
                     "dYield_rel": dy, "match": bool(dt < 1e-6 and dy < 1e-4),
                     "row_in_file": int(best.row)})
    return pd.DataFrame(rows)


def gui_voltage(p_value: float, clicks: int, step_frac: float = 0.002) -> float:
    """
    Value in the GUI U box after `clicks` arrow clicks (negative = down), reproducing the
    float round trip of untitled28: step = float(str(p/500)); U = float(entry) +/- step;
    entry = str(U).
    """
    text = str(p_value)
    step = float(str(p_value / round(1 / step_frac)))   # p/500 for the default 0.002
    for _ in range(abs(int(clicks))):
        text = str(float(text) + (step if clicks > 0 else -step))
    return float(text)


def _click_order(max_clicks: int):
    yield 0
    for n in range(1, max_clicks + 1):
        yield -n
        yield n


def match_rows(new: pd.DataFrame, old: pd.DataFrame, rtol: float = 1e-9) -> pd.DataFrame:
    """Row-by-row comparison with an Output_detail table (same (m, z) order or lookup)."""
    rows = []
    for r in old.itertuples():
        cand = new[(new.m == r.m) & (new.z == r.z)]
        if cand.empty or not bool(cand.iloc[0].success):
            rows.append({"m": r.m, "z": r.z, "TOF_old": r.TOF, "TOF_new": np.nan,
                         "A_old": r.YieldAveFit, "A_new": np.nan, "dTOF_rel": np.nan,
                         "dA_rel": np.nan, "match": False})
            continue
        c = cand.iloc[0]
        dt = abs(c.TOF / r.TOF - 1)
        da = abs(c.YieldAveFit - r.YieldAveFit) / max(abs(r.YieldAveFit), 1e-300)
        rows.append({"m": r.m, "z": r.z, "TOF_old": r.TOF, "TOF_new": c.TOF,
                     "A_old": r.YieldAveFit, "A_new": c.YieldAveFit, "dTOF_rel": dt,
                     "dA_rel": da, "match": bool(dt < rtol and da < rtol)})
    return pd.DataFrame(rows)


def replay_voltage(traces_by_rep: dict[int, tuple], U_file: float, detail: pd.DataFrame,
                   gui_masses: dict[int, float], L_m: float, eia, profile, settings: dict,
                   corr: float = 0.0, rtol: float = 1e-9) -> dict:
    """
    Reproduce the legacy fits of one voltage, searching the number of U arrow clicks.
    detail: the Output_detail rows of this voltage (columns m, z, TOF, YieldAveFit, ...).
    Returns {clicks, U_used, n_rows, n_match, status, fits, comparison}; status is
    'exact' (all rows), 'partial' (best click count, some rows differ) or 'no-mass'.
    """
    keys = detail[["m", "z"]].drop_duplicates()
    missing = sorted({int(m) for m in keys.m if int(m) not in gui_masses})
    if missing:
        return {"clicks": None, "U_used": None, "n_rows": len(detail), "n_match": 0,
                "status": "no-mass", "missing_masses": missing}
    lines = [(gui_masses[int(m)], int(m), int(z)) for m, z in keys.itertuples(index=False)]
    step = settings.get("u_step_frac", 0.002)
    best = None
    for n in _click_order(int(settings.get("max_u_clicks", 15))):
        U = gui_voltage(U_file, n, step)
        fits = fit_voltage(traces_by_rep, U, lines, L_m, eia, profile, settings, corr)
        cmp = match_rows(fits, detail, rtol)
        k = int(cmp.match.sum())
        if best is None or k > best["n_match"]:
            best = {"clicks": n, "U_used": U, "n_rows": len(detail), "n_match": k,
                    "fits": fits, "comparison": cmp}
        if k == len(detail):
            break
    best["status"] = "exact" if best["n_match"] == best["n_rows"] else "partial"
    return best
