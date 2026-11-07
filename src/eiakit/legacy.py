"""
Readers for the results of the legacy scripts (never written by the new code).

    read_output_folder(dir)       untitled28   2_SEM/Output/p<U>.txt          (m, z, TOF, Yield)
    read_output_detail_folder(dir) untitled28  2_SEM/Output_detail/p<U>.txt
    load_checkpoint(path)         untitled27   EIA-Results/pipeline_checkpoint.npy
    parse_cbs_fit_results(path)   cbs fitter   EIA-Results/CBS_fit_results_<label>_<El>.txt
    parse_cbs_areas(path)         cbs fitter   EIA-Results/CBS_areas_<label>_<El>.txt

The parsers are tolerant (hand-edited files, missing colons, old formats) and keep
both the human-readable blocks and the tab-separated summary table, so the audit
can compare them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_KV = re.compile(rf"([A-Za-z][\w/]*)\s*=\s*({NUM})")
_U_FROM_NAME = re.compile(r"p([0-9]+\.?[0-9]*)")      # untitled27, Section 1


def _kv(line: str) -> dict[str, float]:
    return {k: float(v) for k, v in _KV.findall(line)}


# ================================================================ untitled28 output

def voltage_from_output_name(name: str) -> float | None:
    """U [V] from p<U>.txt exactly as untitled27 does (p10.txt -> 10.0, p200.01.txt -> 200.01)."""
    m = _U_FROM_NAME.search(name)
    return float(m.group(1).rstrip(".")) if m else None


def read_output_folder(folder: Path) -> pd.DataFrame:
    """
    All rows of all p<U>.txt files: columns file, U, row (0-based line in the file),
    m, z, TOF [s], Yield [V] (+ any extra column of Output_detail files).
    """
    frames = []
    for f in sorted(Path(folder).glob("*.txt")):
        U = voltage_from_output_name(f.name)
        if U is None:
            continue
        df = pd.read_csv(f)
        df.insert(0, "row", range(len(df)))
        df.insert(0, "U", U)
        df.insert(0, "file", f.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["file", "U", "row", "m", "z", "TOF", "Yield"])
    out = pd.concat(frames, ignore_index=True)
    for c in ("m", "z"):
        out[c] = out[c].astype(int)
    return out


def read_output_detail_folder(folder: Path) -> pd.DataFrame:
    return read_output_folder(folder)


# ================================================================ untitled27 checkpoint

def load_checkpoint(path: Path) -> dict:
    """The pickled dict of untitled27, with (m, z) keys turned into plain int tuples."""
    obj = np.load(path, allow_pickle=True)
    ck = obj.item() if obj.dtype == object else obj
    if not isinstance(ck, dict):
        raise ValueError(f"{path}: not a pickled dict")
    out = {}
    for k, v in ck.items():
        if isinstance(v, dict) and v and all(isinstance(kk, tuple) for kk in v):
            v = {(int(kk[0]), int(kk[1])): vv for kk, vv in v.items()}
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            v = v.item()
        elif isinstance(v, list):
            v = [x.item() if isinstance(x, np.generic) else x for x in v]
        out[k] = v
    return out


# ================================================================ CBS files: header + table

@dataclass
class FileHeader:
    generated: str | None = None
    L_EIA_m: float | None = None
    M_ION_kg: float | None = None
    signal: str | None = None           # "per_ion" | "charge" | None (old files: charge)

    @property
    def per_ion(self) -> bool:
        return self.signal == "per_ion"


def _parse_header(lines: list[str]) -> FileHeader:
    h = FileHeader()
    for line in lines[:12]:
        if "generated" in line:
            m = re.search(r"generated\s+(.+?)\s*$", line)
            h.generated = m.group(1) if m else None
        m = re.search(rf"L_EIA\s*=\s*({NUM})", line)
        if m and line.startswith("#"):
            h.L_EIA_m = float(m.group(1))
        m = re.search(rf"M_ION\s*=\s*({NUM})", line)
        if m and line.startswith("#"):
            h.M_ION_kg = float(m.group(1))
        if line.startswith("# Signal:"):
            h.signal = "per_ion" if "per-ion" in line else "charge"
    return h


def _parse_table(lines: list[str]) -> pd.DataFrame:
    """The tab-separated table that follows a '# z<TAB>CBS<TAB>...' header line."""
    header, rows, raw = None, [], []
    for line in lines:
        s = line.rstrip("\n")
        if s.startswith("# z\t"):
            header = s[2:].split("\t")
            continue
        if header is None or s.startswith("#") or not s.strip():
            continue
        parts = s.split("\t")
        if len(parts) != len(header):
            continue
        rows.append(parts)
        raw.append(s)
    if header is None:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().all():
            df[c] = conv
    df["z"] = df["z"].astype(int)
    df["CBS"] = df["CBS"].astype(int)
    df["_raw"] = raw
    return df


# ================================================================ CBS fit results

@dataclass
class CBSComponent:
    z: int
    cbs: int                      # number as written in the file
    picks: dict = field(default_factory=dict)      # t_peak, I_peak, t_left, t_right, FWHM
    window: dict = field(default_factory=dict)     # win_left, win_right, src
    triplet: dict = field(default_factory=dict)    # prev/PEAK/next: {"t": .., "I": .., "u": ..}
    kT: float = np.nan
    V0: float = np.nan
    A: float = np.nan
    errors: dict = field(default_factory=dict)     # kT/V0/A: (plus, minus)
    chi2: float = np.nan
    chi2_dof: float = np.nan

    @property
    def fitted(self) -> bool:
        return np.isfinite(self.kT) and np.isfinite(self.V0) and np.isfinite(self.A)


@dataclass
class CBSFitFile:
    path: Path
    header: FileHeader
    z_totals: dict[int, dict]                 # z -> {"chi2", "dof", "chi2/dof"}
    components: list[CBSComponent]
    table: pd.DataFrame                       # pick-vs-fit summary table

    def component(self, z: int, cbs: int) -> CBSComponent | None:
        for c in self.components:
            if c.z == z and c.cbs == cbs:
                return c
        return None


_ERR = re.compile(rf"^(kT|V0|A)\s*=\s*({NUM})\s*\+\s*({NUM})\s*/\s*-\s*({NUM})")


def parse_cbs_fit_results(path: Path) -> CBSFitFile:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header = _parse_header(lines)
    z_totals, comps = {}, []
    cur_z, cur = None, None
    for raw in lines:
        line = raw.strip()
        if line.startswith("#"):
            continue
        m = re.match(r"z=(\d+)\s+Total", line)
        if m:
            cur_z, cur = int(m.group(1)), None
            kv = _kv(line)
            z_totals[cur_z] = {"chi2": kv.get("chi2", np.nan), "dof": kv.get("dof", np.nan),
                               "chi2/dof": kv.get("chi2/dof", np.nan)}
            continue
        m = re.match(r"CBS\s+(\d+)\s*:?", line)
        if m and cur_z is not None:
            cur = CBSComponent(z=cur_z, cbs=int(m.group(1)))
            comps.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("Picks:"):
            cur.picks = _kv(line)
        elif line.startswith("Window:"):
            cur.window = _kv(line)
            src = re.search(r"\(([^)]*)\)\s*$", line)
            s = src.group(1) if src else "default"
            cur.window["src"] = "default" if s.startswith("default") else s
        elif re.match(r"(prev|PEAK|next)\s*:", line):
            key = line.split(":")[0].strip()
            kv = _kv(line)
            cur.triplet[key] = {"t": kv.get("t", np.nan), "I": kv.get("I", np.nan),
                                "u": kv.get("u", np.nan)}
        elif _ERR.match(line):
            g = _ERR.match(line)
            setattr(cur, g.group(1), float(g.group(2)))
            cur.errors[g.group(1)] = (float(g.group(3)), float(g.group(4)))
        elif re.match(r"kT\s*=", line):
            kv = _kv(line)
            cur.kT, cur.V0, cur.A = (kv.get(k, np.nan) for k in ("kT", "V0", "A"))
        elif line.startswith("chi2=") or line.startswith("chi2 ="):
            kv = _kv(line)
            cur.chi2, cur.chi2_dof = kv.get("chi2", np.nan), kv.get("chi2/dof", np.nan)
    return CBSFitFile(Path(path), header, z_totals, comps, _parse_table(lines))


# ================================================================ CBS areas

@dataclass
class CBSAreasFile:
    path: Path
    header: FileHeader
    blocks: pd.DataFrame          # one row per (z, CBS) from the readable blocks
    table: pd.DataFrame           # summary table


def parse_cbs_areas(path: Path) -> CBSAreasFile:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    header = _parse_header(lines)
    rows, cur_z, cur = [], None, None
    for raw in lines:
        line = raw.strip()
        if line.startswith("#"):
            continue
        m = re.match(r"z=(\d+)\s+Area_data_FULL", line)
        if m:
            cur_z, cur = int(m.group(1)), None
            continue
        m = re.match(r"CBS\s+(\d+)\s*:?\s*(.*)$", line)
        if m and cur_z is not None:
            kv = _kv(m.group(2))
            w = re.search(rf"window=\[\s*({NUM})\s*,\s*({NUM})\s*\]", line)
            src = re.search(r"\(([^)]*)\)\s*$", line)
            cur = {"z": cur_z, "CBS": int(m.group(1)), "t_peak_us": kv.get("t_peak", np.nan),
                   "FWHM_us": kv.get("FWHM", np.nan),
                   "win_left_us": float(w.group(1)) if w else np.nan,
                   "win_right_us": float(w.group(2)) if w else np.nan,
                   "win_src": src.group(1) if src else None,
                   "colon": ":" in line.split("t_peak")[0]}
            rows.append(cur)
            continue
        if cur is None:
            continue
        for key in ("Area_model_FULL", "Area_model", "Area_data_i"):
            m = re.match(rf"{key}\s*=\s*({NUM})", line)
            if m:
                cur[key] = float(m.group(1))
                break
    return CBSAreasFile(Path(path), header, pd.DataFrame(rows), _parse_table(lines))
