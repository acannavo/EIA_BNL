"""
Stage outputs on disk: HDF5 for arrays, JSON for parameters (Step 7 will add the
manifest / provenance). Keys (m, z) are stored as "m{m}_z{z}".
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _k(key):
    return f"m{int(key[0])}_z{int(key[1])}"


def _unk(name):
    m, z = name[1:].split("_z")
    return int(m), int(z)


def save_calibration_h5(path: Path, res: dict, attrs: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        for name in ("corrected_tof", "corrected_tof_1m"):
            h.create_dataset(name, data=res[name])
        for group in ("corrected_spectra", "corrected_spectra_1m", "corrected_exp_points",
                      "corrected_exp_tof"):
            g = h.create_group(group)
            for key, arr in res[group].items():
                g.create_dataset(_k(key), data=np.asarray(arr))
        for group in ("integrals", "integrals_eia"):
            g = h.create_group(group)
            for key, v in res[group].items():
                g.attrs[_k(key)] = float(v)
        t, v = res.get("fc_average", (None, None))
        if t is not None:
            g = h.create_group("fc_average")
            g.create_dataset("t_s", data=t)
            g.create_dataset("V_minus_baseline", data=v)
        for k in ("CFY_EIA", "CFY_1m", "CFArea", "CFCurrent", "CFT_1m", "L_EIA", "L_FC",
                  "fc_baseline_V", "fc_peak_V", "eia_sum_peak", "total_integral",
                  "total_integral_eia"):
            h.attrs[k] = float(res[k])
        h.attrs["signal"] = "charge"            # J_z with the factor z inside (legacy)
        for k, v in (attrs or {}).items():
            h.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return path


def load_calibration_h5(path: Path) -> dict:
    out = {}
    with h5py.File(path, "r") as h:
        for name in ("corrected_tof", "corrected_tof_1m"):
            out[name] = h[name][()]
        for group in ("corrected_spectra", "corrected_spectra_1m", "corrected_exp_points",
                      "corrected_exp_tof"):
            out[group] = {_unk(n): h[group][n][()] for n in h[group]}
        for group in ("integrals", "integrals_eia"):
            out[group] = {_unk(n): float(v) for n, v in h[group].attrs.items()}
        out.update({k: v for k, v in h.attrs.items()})
        if "fc_average" in h:
            out["fc_average"] = (h["fc_average/t_s"][()], h["fc_average/V_minus_baseline"][()])
    return out
