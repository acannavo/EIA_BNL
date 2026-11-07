"""
Acquisition file formats: how raw SEM and Faraday-cup files are named and laid out.

Everything is configured in ``db/instruments.toml`` ([acquisition.<name>.sem] and
[acquisition.<name>.fc]) and can be overridden per run (``acquisition_overrides``
in runs.toml), so a new naming scheme needs no code change.

File-name templates (case-insensitive, matched against the whole name)
----------------------------------------------------------------------
    {int}   digits: integer part of the deflector voltage U
    {dec}   digits: decimal part, U = float("int.dec")     000p6  -> 0.6,  12p25 -> 12.25
    {U}     a plain decimal number, U = float(text)         12.5   -> 12.5
    {rep}   digits: repeat / shot number
    {*}     any text (also empty)
    anything else is literal text.
Several templates may be listed; the first that matches wins. A template starting
with ``re:`` is a raw regular expression with named groups U / int / dec / rep.

Files matching an ``ignore`` glob (e.g. "*first*.csv") are skipped before matching.

Header
------
    "skiprows:<n>"     skip n lines; then a header line if columns are names,
                       no header line if columns are 0-based indices
    "detect:<WORD>"    the header is the first line whose first field is WORD (e.g. TIME)
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

_PLACEHOLDERS = {
    "int": r"(?P<int>\d+)",
    "dec": r"(?P<dec>\d+)",
    "U": r"(?P<U>\d+(?:\.\d+)?)",
    "rep": r"(?P<rep>\d+)",
    "*": r".*?",
}
_TOKEN = re.compile(r"\{(int|dec|U|rep|\*)\}")


def template_to_regex(template: str) -> re.Pattern:
    """Compile a file-name template (see module docstring) to a regex."""
    if template.startswith("re:"):
        return re.compile(template[3:], re.IGNORECASE)
    parts, pos = [], 0
    for m in _TOKEN.finditer(template):
        parts.append(re.escape(template[pos:m.start()]))
        parts.append(_PLACEHOLDERS[m.group(1)])
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    rx = "".join(parts)
    names = re.findall(r"\?P<(\w+)>", rx)
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f"template {template!r}: placeholder(s) {sorted(dup)} used twice")
    if "dec" in names and "int" not in names:
        raise ValueError(f"template {template!r}: {{dec}} needs {{int}}")
    return re.compile(rf"^{rx}$", re.IGNORECASE)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FileFormat(_Model):
    folder: str
    patterns: list[str] = Field(min_length=1)
    ignore: list[str] = []
    header: str
    delimiter: str = ","
    encoding: str = "utf-8"
    time_column: str | int
    signal_column: str | int
    time_scale: float = 1.0          # multiply the time column to get seconds
    signal_scale: float = 1.0        # multiply the signal column (e.g. -1 to flip polarity)

    @field_validator("header")
    @classmethod
    def _header(cls, v):
        if not re.fullmatch(r"skiprows:\d+|detect:\S+", v):
            raise ValueError(f"header must be 'skiprows:<n>' or 'detect:<WORD>', got {v!r}")
        return v

    @field_validator("patterns")
    @classmethod
    def _patterns(cls, v):
        for t in v:
            template_to_regex(t)
        return v

    def compiled(self) -> list[re.Pattern]:
        return [template_to_regex(t) for t in self.patterns]

    def is_ignored(self, filename: str) -> bool:
        low = filename.lower()
        return any(fnmatch.fnmatchcase(low, g.lower()) for g in self.ignore)

    def match(self, filename: str) -> re.Match | None:
        if self.is_ignored(filename):
            return None
        for rx in self.compiled():
            m = rx.match(filename)
            if m:
                return m
        return None


class SEMFormat(FileFormat):
    u_scale: float = 1.0             # U [V] = value from the name x u_scale

    @field_validator("patterns")
    @classmethod
    def _needs_u(cls, v):
        for t in v:
            groups = set(template_to_regex(t).groupindex)
            if not ({"U", "int"} & groups):
                raise ValueError(f"SEM template {t!r} must contain {{int}} or {{U}}")
        return v

    def parse_name(self, filename: str) -> tuple[float, int] | None:
        """(deflector voltage U [V], repeat number) of a SEM file, or None (no match / ignored)."""
        m = self.match(filename)
        if m is None:
            return None
        g = m.groupdict()
        if g.get("U") is not None:
            U = float(g["U"])
        elif g.get("dec"):
            U = float(f"{int(g['int'])}.{g['dec']}")
        else:
            U = float(int(g["int"]))
        rep = int(g["rep"]) if g.get("rep") is not None else 1
        return U * self.u_scale, rep


class FCFormat(FileFormat):
    folder_stop: str | None = None   # FC traces recorded after the scan (e.g. 3_FC_Stop)


class Acquisition(_Model):
    description: str = ""
    sem: SEMFormat
    fc: FCFormat


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        nested = isinstance(v, dict) and isinstance(out.get(k), dict)
        out[k] = deep_merge(out[k], v) if nested else v
    return out


# ---------------------------------------------------------------- scanning

@dataclass
class FileScan:
    folder: Path
    matched: list[tuple[Path, float | None, int]] = field(default_factory=list)  # path, U, rep
    ignored: list[Path] = field(default_factory=list)
    unmatched: list[Path] = field(default_factory=list)
    missing_folder: bool = False

    def by_voltage(self) -> dict[float, list[int]]:
        out: dict[float, list[int]] = {}
        for _, U, rep in self.matched:
            out.setdefault(U, []).append(rep)
        return {U: sorted(r) for U, r in sorted(out.items())}

    def duplicates(self) -> list[tuple[float | None, int]]:
        seen, dup = set(), []
        for _, U, rep in self.matched:
            if (U, rep) in seen:
                dup.append((U, rep))
            seen.add((U, rep))
        return dup


def scan_folder(folder: Path, fmt: FileFormat) -> FileScan:
    """Classify the files directly inside folder (sub-folders are not entered)."""
    scan = FileScan(folder=folder)
    if not folder.is_dir():
        scan.missing_folder = True
        return scan
    for f in sorted(p for p in folder.iterdir() if p.is_file()):
        if fmt.is_ignored(f.name):
            scan.ignored.append(f)
            continue
        if isinstance(fmt, SEMFormat):
            parsed = fmt.parse_name(f.name)
            if parsed is None:
                scan.unmatched.append(f)
            else:
                scan.matched.append((f, parsed[0], parsed[1]))
        else:
            m = fmt.match(f.name)
            if m is None:
                scan.unmatched.append(f)
            else:
                rep = m.groupdict().get("rep")
                scan.matched.append((f, None, int(rep) if rep else 1))
    return scan


# ---------------------------------------------------------------- reading

def _header_row(path: Path, fmt: FileFormat) -> tuple[int, bool]:
    """(lines to skip, file has a header line)."""
    kind, _, arg = fmt.header.partition(":")
    names = isinstance(fmt.time_column, str) or isinstance(fmt.signal_column, str)
    if kind == "skiprows":
        return int(arg), names
    word = arg.strip().upper()
    with open(path, encoding=fmt.encoding, errors="ignore") as fh:
        for n, line in enumerate(fh):
            if line.split(fmt.delimiter)[0].strip().strip('"').upper() == word:
                return n, True
    raise ValueError(f"{path.name}: no header line starting with {arg!r}")


def read_trace(path: Path, fmt: FileFormat) -> tuple[np.ndarray, np.ndarray]:
    """Time [s] and signal of one oscilloscope file, with the format's scales applied."""
    import pandas as pd

    skip, has_header = _header_row(Path(path), fmt)
    cols = [fmt.time_column, fmt.signal_column]
    if has_header:
        df = pd.read_csv(path, skiprows=skip, sep=fmt.delimiter, encoding=fmt.encoding,
                         encoding_errors="ignore", skipinitialspace=True)
        df.columns = [str(c).strip() for c in df.columns]
        try:
            t, y = (df[c] if isinstance(c, str) else df.iloc[:, c] for c in cols)
        except KeyError as exc:
            raise ValueError(f"{Path(path).name}: column {exc} not found; "
                             f"columns are {list(df.columns)}") from None
    else:
        df = pd.read_csv(path, skiprows=skip, header=None, sep=fmt.delimiter,
                         encoding=fmt.encoding, encoding_errors="ignore")
        t, y = df.iloc[:, cols[0]], df.iloc[:, cols[1]]
    t = pd.to_numeric(t, errors="coerce").to_numpy(float) * fmt.time_scale
    y = pd.to_numeric(y, errors="coerce").to_numpy(float) * fmt.signal_scale
    ok = np.isfinite(t) & np.isfinite(y)
    return t[ok], y[ok]
