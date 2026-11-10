"""
Step 4: calibration stage (port of untitled27_REFACTORED_4, Sections 1-9, without plots).

    result = calibrate(output_rows, fc_traces, fc_baseline_V, eia, fc, pipeline)

Inputs
    output_rows    DataFrame of the untitled28 peak table (legacy.read_output_folder):
                   one row per (deflector voltage file, m, z): TOF [s], Yield [V] (negative)
    fc_traces      list of (t [s], V [V]) Faraday-cup traces (1_FC_Start or 3_FC_Stop)
    fc_baseline_V  pedestal subtracted from the averaged FC trace (picked by hand in
                   untitled27; recover_fc_baseline() gets it back from a legacy checkpoint)
    eia, fc        instrument configurations (db), pipeline = db.pipeline["calibration"]

Output: a dict with the same keys and meaning as pipeline_checkpoint.npy, plus
'fc_baseline_V', 'fc_peak_V', 'eia_sum_peak'. With legacy settings the order of
the floating-point operations is that of untitled27, so the result is identical to
the legacy checkpoint (see compare_checkpoints).

Legacy behaviour kept here on purpose (switches come in Phase C):
    duplicated rows enter the interpolation (N1, N14), zero fill outside each (m, z)
    range (I4), calibration on the peak of the summed EIA (I2), FC averaged on a
    1000-point grid (N7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _segments(rows: pd.DataFrame, m: int, z: int) -> pd.DataFrame | None:
    """get_segments_for_mz of untitled27: all rows of (m, z), Yield x z, sorted by TOF."""
    seg = rows[(rows.m == m) & (rows.z == z)].copy()
    if seg.empty:
        return None
    seg["Yield_scaled"] = seg["Yield"] * seg["z"]
    seg = seg.dropna(subset=["TOF", "Yield_scaled"])
    # untitled27 sorted each file's rows, concatenated them in file order, then sorted again
    parts = [g.sort_values("TOF") for _, g in seg.groupby("file", sort=False)]
    return pd.concat(parts, ignore_index=True).sort_values("TOF")


def interpolate_spectra(rows: pd.DataFrame, n_points: int, fill: str = "zero"):
    """
    Common TOF grid (min..max of every TOF of every file) and, for every (m, z),
    -(Yield x z) linearly interpolated on it. fill = 'zero' (legacy, I4) or 'nan'.
    """
    tof_all = rows["TOF"].dropna().to_numpy()
    tof_common = np.linspace(tof_all.min(), tof_all.max(), n_points)
    fill_value = 0.0 if fill == "zero" else np.nan
    m_values = sorted(int(m) for m in rows.m.unique())
    z_values = sorted(int(z) for z in rows.z.unique())
    data, segs = {}, {}
    for m in m_values:
        for z in z_values:
            seg = _segments(rows, m, z)
            if seg is None:
                continue
            f = interp1d(seg["TOF"], -seg["Yield_scaled"], kind="linear",
                         bounds_error=False, fill_value=fill_value)
            data[(m, z)] = f(tof_common)
            segs[(m, z)] = seg
    return tof_common, data, segs, m_values, z_values


LEGACY_FC_GRID = 1000          # untitled27: FC traces averaged on 1000 points (N7)


def average_fc(traces, n_points):
    """Section 6: every FC trace interpolated on a common grid of n_points, then averaged.
    n_points = 'native': the grid of the traces themselves (smallest sample step)."""
    t_min = min(float(np.min(t)) for t, _ in traces)
    t_max = max(float(np.max(t)) for t, _ in traces)
    if n_points == "native":
        dt = min(float(np.median(np.diff(t))) for t, _ in traces)
        n_points = int(round((t_max - t_min) / dt)) + 1
    grid = np.linspace(t_min, t_max, num=int(n_points))
    frames = []
    for t, v in traces:
        f = interp1d(t, v, kind="linear", bounds_error=False, fill_value=0.0)
        frames.append(pd.DataFrame({"TIME": grid, "CH2": f(grid)}))
    avg = pd.concat(frames).groupby("TIME", as_index=False).mean()
    return avg["TIME"].to_numpy(), avg["CH2"].to_numpy()


def calibrate(rows: pd.DataFrame, fc_traces, fc_baseline_V: float, eia, fc, pipeline: dict,
              cfy_1m_override: float | None = None, fc_peak_override: float | None = None) -> dict:
    """
    The untitled27 chain. cfy_1m_override uses a given CFY_1m instead of matching the FC
    (to rebuild a legacy checkpoint when its FC traces are not available);
    fc_peak_override a given FC peak [V] after the pedestal (legacy_fc_peak()).
    """
    L_EIA, L_FC = eia.L_m, fc.L_m
    tof_common, interp, segs, m_values, z_values = interpolate_spectra(
        rows, int(pipeline["n_interp"]), pipeline.get("interp_fill", "zero"))
    summed = sum(interp.values())
    max_val_sum = summed.max()

    method = pipeline.get("method", "peak")
    if method not in ("peak", "area"):
        raise ValueError(f"[calibration] method = {method!r}: use 'peak' or 'area'")

    L1 = (1.0 / L_FC) ** (-3)          # same expressions as untitled27 (bit-for-bit)
    L2 = (1.0 / L_EIA) ** (-3)
    CFArea = 1.0 / (np.pi * fc.aperture_radius_cm ** 2)
    CFCurrent = 1.0 / fc.R_load_ohm

    def _grid(v):
        return v if v == "native" else int(v)

    def _peak(t, v):
        """FC peak: 'max' of the average (legacy) or max after a moving average of
        fc_smooth_s, searched at t >= fc_peak_t_min_s (the laser pickup at t ~ 0 can be
        as high as the ion peak; legacy: whole trace)."""
        if pipeline.get("fc_peak", "max") == "smoothed":
            w = max(1, int(round(float(pipeline.get("fc_smooth_s", 5e-8))
                                 / float(np.median(np.diff(t))))))
            v = np.convolve(v, np.ones(w) / w, mode="same")
        tmin = pipeline.get("fc_peak_t_min_s")
        sel = t >= float(tmin) if tmin is not None else np.ones(t.shape, bool)
        k = int(np.flatnonzero(sel)[np.argmax(v[sel])])
        return float(v[k]), float(t[k])

    fc_peak_t = None
    if fc_peak_override is not None:
        fc_t = fc_v = None
        max_val_avg = fc_peak_override
        CFY_1m = max_val_avg * L1 / max_val_sum
        CFY_EIA = max_val_avg * L2 / max_val_sum
    elif cfy_1m_override is None:
        fc_t, fc_v = average_fc(fc_traces, _grid(pipeline["fc_grid_points"]))
        fc_v = fc_v - fc_baseline_V
        max_val_avg, fc_peak_t = _peak(fc_t, fc_v)
        CFY_1m = max_val_avg * L1 / max_val_sum
        if method == "area":
            # I2: match the charge per unit area over the time range both cover: at 1 m,
            # j(tau) = FC(L_FC tau) L_FC^3 = CFY sum_EIA(L_EIA tau), integrated over tau
            tau0 = max(tof_common.min() / L_EIA, fc_t.min() / L_FC)
            tau1 = min(tof_common.max() / L_EIA, fc_t.max() / L_FC)
            sf = (fc_t >= tau0 * L_FC) & (fc_t <= tau1 * L_FC)
            se = (tof_common >= tau0 * L_EIA) & (tof_common <= tau1 * L_EIA)
            fc_int = _trapz(fc_v[sf], fc_t[sf])
            eia_int = _trapz(summed[se], tof_common[se])
            CFY_1m = L_FC ** 2 * L_EIA * fc_int / eia_int
            CFY_EIA = CFY_1m * L2 / L1
        else:
            CFY_EIA = max_val_avg * L2 / max_val_sum          # untitled27 expression
    else:
        fc_t = fc_v = None
        CFY_1m = cfy_1m_override
        max_val_avg = CFY_1m * max_val_sum / L1
        CFY_EIA = max_val_avg * L2 / max_val_sum

    CFT_1m = 1.0 / L_EIA
    tof_1m = tof_common * CFT_1m

    def corr(y):                          # apply_corrections of untitled27
        return y * CFY_1m * CFArea * CFCurrent

    exp_points = {k: corr(s["Yield_scaled"].values) / L_EIA ** 3 for k, s in segs.items()}
    exp_tof = {k: s["TOF"].values for k, s in segs.items()}
    integrals = {k: _trapz(corr(y), tof_1m) for k, y in interp.items()}
    integrals_eia = {k: _trapz(corr(y) / L_EIA ** 3, tof_common) for k, y in interp.items()}

    return {
        "corrected_spectra": {k: corr(y) / L_EIA ** 3 for k, y in interp.items()},
        "corrected_tof": tof_common,
        "corrected_spectra_1m": {k: corr(y) for k, y in interp.items()},
        "corrected_tof_1m": tof_1m,
        "corrected_exp_points": exp_points,
        "corrected_exp_tof": exp_tof,
        "integrals": integrals,
        "total_integral": sum(integrals.values()),
        "integrals_eia": integrals_eia,
        "total_integral_eia": sum(integrals_eia.values()),
        "CFY_EIA": CFY_EIA, "CFY_1m": CFY_1m, "CFArea": CFArea, "CFCurrent": CFCurrent,
        "CFT_1m": CFT_1m, "L_EIA": L_EIA, "L_FC": L_FC,
        "m_values": m_values, "z_values_list": z_values,
        # new: what untitled27 did not store (N7)
        "fc_baseline_V": fc_baseline_V, "fc_peak_V": float(max_val_avg),
        "fc_peak_t_s": fc_peak_t,
        "eia_sum_peak": float(max_val_sum),
        "fc_average": (fc_t, fc_v),
    }


def legacy_fc_peak(legacy_ck: dict, fc) -> float:
    """FC peak after the pedestal [V] used by untitled27: CFY_1m * max(sum EIA) / L_FC^3."""
    s1 = sum(legacy_ck["corrected_spectra_1m"].values())
    max_val_sum = s1.max() / (legacy_ck["CFY_1m"] * legacy_ck["CFArea"] * legacy_ck["CFCurrent"])
    L_fc = float(legacy_ck.get("L_FC", fc.L_m))
    return float(legacy_ck["CFY_1m"] * max_val_sum / (1.0 / L_fc) ** (-3))


def _per_shot_peaks(traces, baseline_V: float, pipe: dict) -> list:
    """Peak of every FC shot after the pedestal (moving average of fc_smooth_s), searched
    at t >= fc_peak_t_min_s (at least 1 us: the laser pickup at t ~ 0); drift check."""
    tmin = max(float(pipe.get("fc_peak_t_min_s") or 0.0), 1e-6)
    out = []
    for t, v in traces:
        w = max(1, int(round(float(pipe.get("fc_smooth_s", 5e-8)) / float(np.median(np.diff(t))))))
        sm = np.convolve(v - baseline_V, np.ones(w) / w, mode="same")
        out.append(float(sm[t >= tmin].max()) if (t >= tmin).any() else float("nan"))
    return out


def recover_fc_baseline(legacy_ck: dict, fc_traces, fc_grid_points: int = LEGACY_FC_GRID,
                        eia=None, fc=None) -> float:
    """
    The FC pedestal picked by hand in untitled27 was not saved (N7). It follows from the
    legacy checkpoint: FC peak after baseline = CFY_1m * max(sum EIA) / L_FC^3, and
    baseline = max(raw FC average) - that peak.
    """
    # L_FC of the legacy checkpoint, not of the db: the pedestal is a property of the FC
    # traces, so a later correction of L_FC / aperture in the db must not be absorbed here
    fc_peak = legacy_fc_peak(legacy_ck, fc)
    _, v = average_fc(fc_traces, fc_grid_points)
    return float(v.max() - fc_peak)


# ================================================================ comparison

def compare_checkpoints(new: dict, old: dict) -> dict:
    """Largest relative difference per checkpoint key (0.0 = identical)."""
    def rel(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            return np.inf
        scale = np.max(np.abs(b)) or 1.0
        return float(np.max(np.abs(a - b)) / scale) if a.size else 0.0

    out = {}
    for key, ov in old.items():
        if key not in new:
            out[key] = "missing"
            continue
        nv = new[key]
        if isinstance(ov, dict):
            if set(ov) != set(nv):
                out[key] = f"different (m, z) sets: only old {sorted(set(ov) - set(nv))}, " \
                           f"only new {sorted(set(nv) - set(ov))}"
                continue
            out[key] = max((rel(nv[k], ov[k]) for k in ov), default=0.0)
        elif isinstance(ov, list):
            out[key] = 0.0 if [int(x) for x in ov] == [int(x) for x in nv] else "different"
        else:
            out[key] = rel(nv, ov)
    return out


# ================================================================ run-level stage

def fc_dirs(run, source: str) -> list:
    fmt = run.acquisition.fc
    return {"start": [fmt.folder], "stop": [fmt.folder_stop],
            "both": [fmt.folder, fmt.folder_stop]}[source]


def _picked(run):
    from .cbs_store import default_fits_root
    from .peaks_edit import picked_baseline
    return picked_baseline(default_fits_root(), run)


def prepare_rows(db, run, data_root) -> dict:
    """The untitled28 peak rows the calibration uses: Output/p<U>.txt after the N1
    (output_duplicates) and N8 (overlaps) settings. {rows, n_rows_file, N1, N8} or {error}."""
    from .legacy import keep_last_save, read_output_folder, resolve_overlaps

    pipe = db.pipeline["calibration"]
    sem = run.folder(data_root) / run.acquisition.sem.folder
    rows = read_output_folder(sem / "Output")
    if rows.empty:
        return {"error": f"no peak table in {sem / 'Output'}"}
    n_rows_file, n1 = len(rows), []
    # Step 9: voltages refitted in the peak picker replace their Output rows
    from .cbs_store import default_fits_root
    from .peaks_edit import override_files
    ovr = override_files(default_fits_root(), run)
    for U, f in ovr.items():
        new = pd.read_csv(f)
        new.insert(0, "row", range(len(new)))
        new.insert(0, "U", U)
        new.insert(0, "file", f"fits:{f.name}")
        rows = pd.concat([rows[rows.U != U], new.astype({"m": int, "z": int})],
                         ignore_index=True)
    dup_mode = pipe.get("output_duplicates", "all")
    if dup_mode in ("last_per_mz", "last_save"):                   # N1
        det = sem / "Output_detail"
        rows, n1 = keep_last_save(rows, read_output_folder(det) if det.is_dir() else None,
                                  dup_mode)
    elif dup_mode != "all":
        return {"error": f"[calibration] output_duplicates = {dup_mode!r}: "
                         "use 'all', 'last_per_mz' or 'last_save'"}
    n8, ov = [], pipe.get("overlaps", "keep_both")
    if ov == "nearest_line":                                       # N8
        masses = {int(k): float(v) for k, v in db.pipeline["peaks"]["gui_masses"].items()}
        rows, n8 = resolve_overlaps(rows, masses, run.mass_labels,
                                    float(pipe.get("overlap_res_floor", 0.002)))
    elif ov != "keep_both":
        return {"error": f"[calibration] overlaps = {ov!r}: use 'keep_both' or 'nearest_line'"}
    return {"rows": rows, "n_rows_file": n_rows_file, "N1": n1, "N8": n8,
            "peak_overrides": sorted(ovr)}


def run_calibration(db, run, data_root, fc_source="start", baseline="legacy",
                    legacy_cfy=False) -> dict:
    """
    Calibration stage (untitled27): the FC pedestal is 'legacy' (recovered from the
    legacy checkpoint, N7), 'pretrigger' (mean before t = 0) or a number in volts.
    Returns {res, info, legacy} or {error}.
    """
    import numpy as np

    from .acquisition import read_trace, scan_folder
    from .legacy import load_checkpoint

    folder = run.folder(data_root)
    pipe = db.pipeline["calibration"]
    prep = prepare_rows(db, run, data_root)
    if "error" in prep:
        return prep
    rows, n_rows_file, n1, n8 = prep["rows"], prep["n_rows_file"], prep["N1"], prep["N8"]
    fmt = run.acquisition.fc
    traces, used = [], []
    for d in fc_dirs(run, fc_source):
        if d is None or not (folder / d).is_dir():
            continue
        for p, _, _ in scan_folder(folder / d, fmt).matched:
            traces.append(read_trace(p, fmt))
            used.append(f"{d}/{p.name}")
    ck_path = folder / "EIA-Results" / "pipeline_checkpoint.npy"
    old = load_checkpoint(ck_path) if ck_path.is_file() else None
    cfy = fc_peak = None
    missing = pipe.get("fc_missing", "error")
    if not traces and not legacy_cfy and missing == "legacy_cfy":
        legacy_cfy = True                     # explicit db choice; recorded in the summary
    if not traces and not legacy_cfy and missing == "legacy_fc_peak":
        if old is None:
            return {"error": "fc_missing = legacy_fc_peak needs the legacy checkpoint"}
        fc_peak, B = legacy_fc_peak(old, run.fc), 0.0
        how = ("no FC traces: FC peak from the legacy checkpoint (exact for corrections "
               "that do not touch the FC)")
    elif legacy_cfy:
        if old is None:
            return {"error": "legacy CFY needs the legacy checkpoint"}
        cfy, B, how = old["CFY_1m"], 0.0, "CFY_1m taken from the legacy checkpoint"
    elif baseline == "picked" and traces and _picked(run) is not None:
        B, how = _picked(run), "picked in the baseline tool (fits/<run>/fc_baseline.json)"
    elif baseline in ("legacy", "picked"):
        if old is None or not traces:
            if not traces:
                return {"error": f"no FC traces in {fc_dirs(run, fc_source)}"}
            pre = np.concatenate([v[t < 0] for t, v in traces])
            B, how = float(np.mean(pre)), "pretrigger mean (no legacy checkpoint to recover)"
        elif fc_source == "start":
            # on the legacy grid: the pedestal is what untitled27 used; a finer grid or
            # another peak estimator (N7) must change the FC peak, not the pedestal
            B = recover_fc_baseline(old, traces, LEGACY_FC_GRID, run.eia, run.fc)
            how = "recovered from the legacy checkpoint (N7)"
        else:
            # the legacy pedestal belongs to the start traces (untitled27 used only those):
            # recovered there, its offset from their pre-trigger mean is applied on top of
            # each trace's own pre-trigger mean (otherwise the recovery would absorb every
            # start / stop difference into the pedestal)
            start = [read_trace(p, fmt) for d in fc_dirs(run, "start")
                     if d and (folder / d).is_dir()
                     for p, _, _ in scan_folder(folder / d, fmt).matched]
            if not start:
                return {"error": "legacy pedestal with fc_source != start needs the start "
                                 "traces"}
            B0 = recover_fc_baseline(old, start, LEGACY_FC_GRID, run.eia, run.fc)
            delta = B0 - float(np.mean(np.concatenate([v[t < 0] for t, v in start])))
            traces = [(t, v - (float(np.mean(v[t < 0])) + delta)) for t, v in traces]
            B = 0.0
            how = (f"per trace: pre-trigger mean + {delta * 1e3:.4f} mV (offset of the legacy "
                   f"pedestal on the start traces)")
    elif baseline == "pretrigger":
        if not traces:
            return {"error": f"no FC traces in {fc_dirs(run, fc_source)}"}
        pre = np.concatenate([v[t < 0] for t, v in traces])
        B, how = float(np.mean(pre)), f"mean of the FC signal before t = 0 ({pre.size} samples)"
    else:
        B, how = float(baseline), "fixed value"
    if not traces and cfy is None and fc_peak is None:
        return {"error": f"no FC traces in {fc_dirs(run, fc_source)}"}
    res = calibrate(rows, traces, B, run.eia, run.fc, pipe, cfy_1m_override=cfy,
                    fc_peak_override=fc_peak)
    info = {"run": run.key, "fc_source": fc_source, "fc_files": used, "fc_baseline_V": B,
            "baseline_method": how, "fc_peak_V": res["fc_peak_V"], "CFY_1m": res["CFY_1m"],
            "CFY_EIA": res["CFY_EIA"], "n_output_rows": len(rows),
            "n_output_rows_in_files": n_rows_file, "N1_last_save": n1,
            "N8_overlaps_dropped": n8, "peak_overrides_U": prep["peak_overrides"],
            "pairs": len(res["corrected_spectra"]),
            "fc_peak_t_s": res.get("fc_peak_t_s"),
            "fc_peak_warning": (None if res.get("fc_peak_t_s") is None
                                or res["fc_peak_t_s"] > 1e-6 else
                                "FC peak within 1 us of the trigger: laser pickup? set "
                                "[calibration] fc_peak_t_min_s"),
            "fc_per_shot_peak_V": _per_shot_peaks(traces, B, pipe),
            "settings": {k: pipe.get(k) for k in ("n_interp", "fc_grid_points", "fc_peak",
                                                  "fc_smooth_s", "fc_peak_t_min_s", "method",
                                                  "interp_fill", "output_duplicates", "overlaps")}}
    if old is not None:
        diff = compare_checkpoints(res, old)
        num = [v for v in diff.values() if isinstance(v, float)]
        info["legacy_comparison"] = {
            "max_rel_diff": max(num), "per_key": diff,
            "CFY_ratio_new_over_legacy": float(res["CFY_1m"] / old["CFY_1m"]),
            "identical_1e-9": all(isinstance(v, float) and v < 1e-9 for v in diff.values())}
    return {"res": res, "info": info, "legacy": old}
