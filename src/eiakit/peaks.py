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

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from . import physics as ph


def gaussian(x, A, mu, sigma, B):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B


def average_repeats(traces, n_points: int | None = None):
    """
    Repeats of one U interpolated on a common grid (min..max of all their times) and
    averaged. n_points defaults to the longest trace (untitled28 PSexp); the 2025 version
    used another length, see legacy_grid_lengths (N9).
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
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
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


def legacy_grid_lengths(files: list[tuple], length_of) -> dict[float, int]:
    """
    Grid length of each voltage in the 2025 untitled28 (N9). The script reads the files
    in glob order, then averages with  np.linspace(min, max, len(data))  where `data` is
    the last DataFrame bound: for the first voltage the last file read, for the others
    the last shot of the previous voltage (the averaging loop rebinds `data`).
    files: (path, U, rep) of the SEM folder; length_of(path) -> number of points (only
    called for the files that set a grid).
    The glob order on Windows (NTFS) is the case-insensitive name order.
    """
    order = sorted(files, key=lambda f: f[0].name.upper())
    groups: dict[float, list] = {}
    for f in order:
        groups.setdefault(f[1], []).append(f[0])
    out, prev = {}, (order[-1][0] if order else None)
    for U, paths in groups.items():
        out[U] = length_of(prev)
        prev = paths[-1]
    return out


def gui_lof(lof0: float, clicks: int, step_frac: float = 0.01) -> float:
    """LOF box after `clicks` arrow clicks: step = float(str(LOF/100)), str(round(lof, 6))."""
    text = str(lof0)
    step = float(str(lof0 / round(1 / step_frac)))
    for _ in range(abs(int(clicks))):
        text = str(round(float(text) + (step if clicks > 0 else -step), 6))
    return float(text)


def _click_order(max_clicks: int):
    yield 0
    for n in range(1, max_clicks + 1):
        yield -n
        yield n


# match levels: 'exact' = same numbers up to rounding of the last digits; 'equivalent' =
# the same fit within the convergence noise of curve_fit (1e-9 .. 1e-6 seen between
# machines / scipy versions); anything else is a different fit.
EXACT_RTOL = 1e-9
EQUIV_TOF_RTOL = 1e-8
EQUIV_A_RTOL = 1e-6


def match_rows(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row comparison with the Output_detail rows of one voltage."""
    rows = []
    for r in old.itertuples():
        cand = new[(new.m == r.m) & (new.z == r.z)]
        base = {"m": r.m, "z": r.z, "TOF_old": r.TOF, "A_old": r.YieldAveFit}
        if cand.empty or not bool(cand.iloc[0].success):
            rows.append({**base, "TOF_new": np.nan, "A_new": np.nan, "dTOF_rel": np.nan,
                         "dA_rel": np.nan, "level": "different", "match": False})
            continue
        c = cand.iloc[0]
        dt = abs(c.TOF / r.TOF - 1)
        da = abs(c.YieldAveFit - r.YieldAveFit) / max(abs(r.YieldAveFit), 1e-300)
        if dt < EXACT_RTOL and da < EXACT_RTOL:
            level = "exact"
        elif dt < EQUIV_TOF_RTOL and da < EQUIV_A_RTOL:
            level = "equivalent"
        else:
            level = "different"
        rows.append({**base, "TOF_new": c.TOF, "A_new": c.YieldAveFit, "dTOF_rel": dt,
                     "dA_rel": da, "level": level, "match": level != "different"})
    return pd.DataFrame(rows)


def replay_voltage(traces_by_rep: dict[int, tuple], U_file: float, detail: pd.DataFrame,
                   gui_masses: dict[int, float], L_m: float, eia, profile, settings: dict,
                   grids: tuple = (None,), lof_clicks: tuple = (0,),
                   corrs: tuple = (0.0,), clicks: tuple | None = None) -> dict:
    """
    Reproduce the legacy fits of one voltage by searching the GUI state that was not
    saved: U arrow clicks (always), averaging-grid length (`grids`, None = longest
    trace), LOF arrow clicks and Corr (deep search). L_m is the LOF pre-filled in the GUI.
    Returns {clicks, U_used, grid_n, lof_clicks, LOF_used, corr, n_rows, n_match,
    n_exact, status, fits, comparison}; status 'exact' | 'equivalent' | 'partial' |
    'no-mass'. The search stops at the first state that reproduces every row.
    """
    keys = detail[["m", "z"]].drop_duplicates()
    missing = sorted({int(m) for m in keys.m if int(m) not in gui_masses})
    if missing:
        return {"clicks": None, "U_used": None, "n_rows": len(detail), "n_match": 0,
                "n_exact": 0, "status": "no-mass", "missing_masses": missing}
    lines = [(gui_masses[int(m)], int(m), int(z)) for m, z in keys.itertuples(index=False)]
    step = settings.get("u_step_frac", 0.002)
    lstep = settings.get("lof_step_frac", 0.01)
    reps = sorted(traces_by_rep)
    best, score_best = None, (-1, -1)
    for grid_n in dict.fromkeys(grids):
        avg = average_repeats([traces_by_rep[r] for r in reps], grid_n)
        for corr in corrs:
            for nl in lof_clicks:
                lof = gui_lof(L_m, nl, lstep)
                for n in (clicks if clicks is not None
                          else _click_order(int(settings.get("max_u_clicks", 15)))):
                    U = gui_voltage(U_file, n, step)
                    fits = _fit_lines(avg, traces_by_rep, reps, U, lines, lof, eia, profile,
                                      settings, corr)
                    cmp = match_rows(fits, detail)
                    score = (int(cmp.match.sum()), int((cmp.level == "exact").sum()))
                    if score > score_best:
                        score_best = score
                        best = {"clicks": n, "U_used": U, "grid_n": len(avg[0]),
                                "lof_clicks": nl, "LOF_used": lof, "corr": corr,
                                "n_rows": len(detail), "n_match": score[0],
                                "n_exact": score[1], "fits": fits, "comparison": cmp}
                    if score[1] == len(detail):
                        best["status"] = "exact"
                        return best
    if best["n_match"] == best["n_rows"]:
        best["status"] = "equivalent"
    else:
        best["status"] = "partial"
    return best


def _fit_lines(avg, traces_by_rep, reps, U, lines, L_m, eia, profile, settings, corr):
    tof, y = avg
    out = []
    for m_amu, label, z in lines:
        t0 = predicted_tof(m_amu, z, U, L_m, eia, profile)
        f = fit_peak(tof, y, t0, corr, settings.get("window_frac", 0.02),
                     settings.get("mu_bound_s", 150e-9), settings.get("min_points", 1))
        row = {"m": label, "z": z, "TOF": f["mu"], "Yield": f["A"], "success": f["success"]}
        if f["success"]:
            row["YieldAveFit"] = f["A"]
        out.append(row)
    return pd.DataFrame(out)


def shot_check(traces_by_rep: dict[int, tuple], detail: pd.DataFrame) -> dict:
    """
    Are the raw shots the ones that were fitted? Compares the stored per-shot values
    Yield1..n (np.interp of each shot at the fitted TOF) with the raw files.
    Returns {n_stored, n_files, max_rel (same order), max_rel_any_order}.
    """
    cols = sorted((c for c in detail.columns if c.startswith("Yield") and c[5:].isdigit()),
                  key=lambda c: int(c[5:]))
    stored = detail[cols].to_numpy(float)
    n_stored = int(np.isfinite(stored).all(axis=0).sum()) if len(stored) else 0
    reps = sorted(traces_by_rep)
    raw = np.array([[np.interp(t, *traces_by_rep[r]) for r in reps] for t in detail.TOF])
    scale = np.nanmax(np.abs(stored[:, :n_stored])) if n_stored else 1.0
    out = {"n_stored": n_stored, "n_files": len(reps), "max_rel": np.nan,
           "max_rel_any_order": np.nan}
    if n_stored == len(reps) and len(stored):
        st = stored[:, :n_stored]
        out["max_rel"] = float(np.nanmax(np.abs(st - raw)) / scale)
        out["max_rel_any_order"] = float(np.nanmax(np.abs(np.sort(st, 1) - np.sort(raw, 1)))
                                         / scale)
    return out


def replay_run(db, run, data_root, voltages=None, max_clicks=None, deep=False, deep_lof=3,
               log=None, verbose=False, state=None, search=False) -> dict:
    """
    Replay every voltage of a run that has raw SEM files and an Output_detail file.
    state: recovered GUI state per voltage (load_peak_state); voltages in it are refitted
    with that state only (fast), unless search=True or the stored state no longer works.
    Grid candidates: the run's legacy rule first, then the other one. With deep=True
    also LOF clicks within +/- deep_lof and Corr in {0, 0.005, 0.01, 0.02, 0.03, 0.05}.
    Returns {summary: [dict per voltage], rows: [comparison DataFrames], counts}.
    """
    from functools import cache

    from .acquisition import read_trace, scan_folder
    from .legacy import read_output_detail_folder

    log = log or (lambda *_: None)
    fmt = run.acquisition.sem
    sem = run.folder(data_root) / fmt.folder
    detail = read_output_detail_folder(sem / "Output_detail")
    scan = scan_folder(sem, fmt)
    settings = dict(db.pipeline["peaks"])
    if max_clicks is not None:
        settings["max_u_clicks"] = max_clicks
    gui_masses = {int(k): float(v) for k, v in settings.pop("gui_masses", {}).items()}
    leg = run.legacy
    rule = leg.peaks_grid if leg else "max"
    profile_name = leg.profile_peaks if leg else "legacy_untitled28"
    profile = db.profile(profile_name)
    lof0 = leg.peaks_lof_m if leg and leg.peaks_lof_m is not None else run.eia.L_m

    @cache
    def length_of(path):
        return len(read_trace(path, fmt)[0])

    prev_len = legacy_grid_lengths(scan.matched, length_of)

    @cache
    def all_lengths():
        return tuple(sorted({length_of(p) for p, _, _ in scan.matched}))
    lof_clicks = tuple(sorted(range(-deep_lof, deep_lof + 1), key=abs)) if deep else (0,)
    corrs = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05) if deep else (0.0,)
    log(f"{run.key}: LOF in GUI = {lof0!r} m, profile {profile_name}, grid rule {rule},"
        f" U step = U/{round(1 / settings['u_step_frac'])}"
        + (f", deep search (LOF +/-{deep_lof} clicks, Corr {corrs})" if deep else ""))
    Us = sorted(set(detail.U)) if not voltages else sorted(voltages)
    summary, rows = [], []
    for U in Us:
        files = {rep: p for p, u, rep in scan.matched if u == U}
        d = detail[detail.U == U] if not detail.empty else detail
        rec = {"U_file": U, "n_files": len(files), "n_rows": len(d)}
        if not files or d.empty:
            rec["status"] = "no raw files" if not files else "no Output_detail"
            summary.append(rec)
            continue
        traces = {rep: read_trace(p, fmt) for rep, p in sorted(files.items())}
        n_max = max(len(t[0]) for t in traces.values())
        g_prev = prev_len.get(U)
        grids = (g_prev, None) if rule == "previous" else (None, g_prev)
        st = (state or {}).get(_ukey(U))
        r = None
        # a stored partial is searched again when a deeper search is asked for
        use_state = st and not search and (st.get("status") in ("exact", "equivalent")
                                           or (st.get("status") == "partial" and not deep))
        if use_state:
            r = replay_voltage(traces, U, d, gui_masses, lof0, run.eia, profile, settings,
                               grids=(st.get("grid_n"),), lof_clicks=(st["lof_clicks"],),
                               corrs=(st["corr"],), clicks=(st["clicks"],))
            if (r.get("n_match", 0), r.get("n_exact", 0)) < (st["n_match"], st["n_exact"]):
                r = None                       # stored state no longer reproduces: search
            elif st["status"] == "partial":
                rec["known"] = st.get("note", "known partial")
        if r is None:
            r = replay_voltage(traces, U, d, gui_masses, lof0, run.eia, profile, settings,
                               grids=grids, lof_clicks=lof_clicks, corrs=corrs)
        if r.get("status") == "partial" and "known" not in rec:
            others = tuple(n for n in all_lengths() if n not in (g_prev, n_max))
            if others:
                r2 = replay_voltage(traces, U, d, gui_masses, lof0, run.eia, profile,
                                    settings, grids=others, lof_clicks=lof_clicks, corrs=corrs)
                if (r2["n_match"], r2["n_exact"]) > (r["n_match"], r["n_exact"]):
                    r = r2
        sc = shot_check(traces, d)
        rec.update(shots_stored=sc["n_stored"], shots_max_rel=sc["max_rel"],
                   shots_max_rel_any_order=sc["max_rel_any_order"])
        keys = ("clicks", "U_used", "grid_n", "lof_clicks", "LOF_used", "corr",
                "n_match", "n_exact", "status")
        rec.update({k: r.get(k) for k in keys}, n_max=n_max, grid_previous=g_prev)
        if r["status"] == "no-mass":
            rec["status"] = f"no GUI mass for labels {r['missing_masses']}"
        else:
            rows.append(r["comparison"].assign(U_file=U, clicks=r["clicks"],
                                               U_used=r["U_used"], grid_n=r["grid_n"]))
        summary.append(rec)
        if (rec["status"] == "exact" or "known" in rec) and not verbose:
            continue
        extra = ""
        if r.get("grid_n") is not None and r["grid_n"] != n_max:
            extra += f"  grid {r['grid_n']} (longest shot {n_max})"
        if r.get("lof_clicks"):
            extra += f"  LOF {r['LOF_used']!r} ({r['lof_clicks']:+d})"
        if r.get("corr"):
            extra += f"  Corr {r['corr']}"
        log(f"  U = {U:>8g}: {len(files)} files, {rec['n_rows']:>3} rows -> "
            f"{rec['status']:<10} clicks {rec.get('clicks')!s:>4}  U_used {rec.get('U_used')!r}"
            f"  ({rec.get('n_match', 0)}/{rec['n_rows']}){extra}")
        if rec["status"] == "partial":
            if sc["n_stored"] != sc["n_files"]:
                log(f"        shots: {sc['n_stored']} stored in Output_detail, "
                    f"{sc['n_files']} raw files")
            else:
                same = "yes" if sc["max_rel"] < 1e-9 else (
                    "yes, other order" if sc["max_rel_any_order"] < 1e-9 else "NO")
                log(f"        raw shots = fitted shots: {same} "
                    f"(max rel {sc['max_rel']:.1e})")
            c = r["comparison"]
            for x in c[~c.match].itertuples():
                log(f"        m={x.m} z={x.z}: TOF {x.TOF_old:.6e} -> {x.TOF_new:.6e} "
                    f"(rel {x.dTOF_rel:.1e}), A {x.A_old:.4e} -> {x.A_new:.4e} "
                    f"(rel {x.dA_rel:.1e})")
    st = [s_.get("status") for s_ in summary]
    counts = {k: st.count(k) for k in ("exact", "equivalent", "partial")}
    counts["skipped"] = len(st) - sum(counts.values())
    counts["known_partial"] = sum(1 for s_ in summary if "known" in s_)
    return {"summary": summary, "rows": rows, "counts": counts}


# ---- recovered GUI state, versioned in db/legacy_peaks/<key>.toml ----------------------
_RANK = {"exact": 3, "equivalent": 2, "partial": 1}


def _ukey(U: float) -> str:
    return repr(float(U))


def peak_state_path(db, key: str) -> Path:
    return Path(db.root) / "legacy_peaks" / f"{key}.toml"


def load_peak_state(db, key: str) -> dict:
    import tomllib
    p = peak_state_path(db, key)
    if not p.is_file():
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh).get("voltages", {})


def update_peak_state(db, key: str, summary: list[dict]) -> tuple[Path, int]:
    """Merge the replay results into the state file (keeps the better result per U)."""
    state = load_peak_state(db, key)
    notes = {k: v.get("note") for k, v in state.items() if v.get("note")}
    n = 0
    for rec in summary:
        if rec.get("status") not in _RANK:
            continue
        k = _ukey(rec["U_file"])
        old = state.get(k)
        new = {f: rec[f] for f in ("status", "clicks", "U_used", "grid_n", "lof_clicks",
                                   "LOF_used", "corr", "n_rows", "n_match", "n_exact")}
        if old and (_RANK[old["status"]], old["n_match"], old["n_exact"]) >= \
                (_RANK[new["status"]], new["n_match"], new["n_exact"]):
            continue
        state[k] = new
        n += 1
    for k, v in notes.items():
        state[k]["note"] = v
    p = peak_state_path(db, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# GUI state of untitled28 recovered by `eiakit peaks {key} --write-state`.",
             "# clicks: U arrow clicks (U/500 each); lof_clicks: LOF arrow clicks (LOF/100);",
             "# corr: Corr box; grid_n: points of the averaging grid. Generated - edit only",
             "# the 'note' entries.", ""]
    for k in sorted(state, key=float):
        v = state[k]
        lines.append(f'[voltages."{k}"]')
        for f in ("status", "clicks", "U_used", "grid_n", "lof_clicks", "LOF_used", "corr",
                  "n_rows", "n_match", "n_exact", "note"):
            if f in v and v[f] is not None:
                val = v[f]
                if isinstance(val, str):
                    lines.append(f'{f} = "{val}"')
                elif isinstance(val, float) and not float(val).is_integer() or f in (
                        "U_used", "LOF_used", "corr"):
                    lines.append(f"{f} = {float(val)!r}")
                else:
                    lines.append(f"{f} = {int(val)}")
        lines.append("")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p, n
