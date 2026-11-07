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


def average_fc(traces, n_points: int):
    """Section 6: every FC trace interpolated on a common grid of n_points, then averaged."""
    t_min = min(float(np.min(t)) for t, _ in traces)
    t_max = max(float(np.max(t)) for t, _ in traces)
    grid = np.linspace(t_min, t_max, num=n_points)
    frames = []
    for t, v in traces:
        f = interp1d(t, v, kind="linear", bounds_error=False, fill_value=0.0)
        frames.append(pd.DataFrame({"TIME": grid, "CH2": f(grid)}))
    avg = pd.concat(frames).groupby("TIME", as_index=False).mean()
    return avg["TIME"].to_numpy(), avg["CH2"].to_numpy()


def calibrate(rows: pd.DataFrame, fc_traces, fc_baseline_V: float, eia, fc, pipeline: dict,
              cfy_1m_override: float | None = None) -> dict:
    """
    The untitled27 chain. cfy_1m_override uses a given CFY_1m instead of matching the FC
    (to rebuild a legacy checkpoint when its FC traces are not available).
    """
    L_EIA, L_FC = eia.L_m, fc.L_m
    tof_common, interp, segs, m_values, z_values = interpolate_spectra(
        rows, int(pipeline["n_interp"]), pipeline.get("interp_fill", "zero"))
    summed = sum(interp.values())
    max_val_sum = summed.max()

    if pipeline.get("method", "peak") != "peak":
        raise NotImplementedError("only the legacy 'peak' calibration is implemented (I2)")

    L1 = (1.0 / L_FC) ** (-3)          # same expressions as untitled27 (bit-for-bit)
    L2 = (1.0 / L_EIA) ** (-3)
    CFArea = 1.0 / (np.pi * fc.aperture_radius_cm ** 2)
    CFCurrent = 1.0 / fc.R_load_ohm

    if cfy_1m_override is None:
        fc_t, fc_v = average_fc(fc_traces, int(pipeline["fc_grid_points"]))
        fc_v = fc_v - fc_baseline_V
        max_val_avg = fc_v.max()
        CFY_1m = max_val_avg * L1 / max_val_sum
        CFY_EIA = max_val_avg * L2 / max_val_sum
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
        "eia_sum_peak": float(max_val_sum),
        "fc_average": (fc_t, fc_v),
    }


def recover_fc_baseline(legacy_ck: dict, fc_traces, fc_grid_points: int, eia, fc) -> float:
    """
    The FC pedestal picked by hand in untitled27 was not saved (N7). It follows from the
    legacy checkpoint: FC peak after baseline = CFY_1m * max(sum EIA) / L_FC^3, and
    baseline = max(raw FC average) - that peak.
    """
    s1 = sum(legacy_ck["corrected_spectra_1m"].values())
    max_val_sum = s1.max() / (legacy_ck["CFY_1m"] * legacy_ck["CFArea"] * legacy_ck["CFCurrent"])
    fc_peak = legacy_ck["CFY_1m"] * max_val_sum / (1.0 / fc.L_m) ** (-3)
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
