"""
The experimental-conditions database: TOML files in ``db/`` (version-controlled).

    constants.toml     physical constants by profile (corrected, legacy_*)
    species.toml       ion species, masses, legacy mass labels
    instruments.toml   EIA and FC configurations, acquisition (file-format) profiles
    runs.toml          runs, keyed by the data folder name
    pipeline.toml      default stage parameters
    literature.toml    literature data for the comparison figures
    studies/*.toml     cross-run comparisons

Location: ``EIA_DB_ROOT`` if set, else the ``db/`` folder of the repository.

    db = load_database()
    run = db.run("20250623")
    run.eia.L_m, db.mass_kg("Nb", "legacy_proton"), db.e_C()
"""

from __future__ import annotations

import datetime as dt
import math
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .acquisition import Acquisition, deep_merge

COMPONENTS = ("fast", "slow", "ultra_fast", "component_4")
STAGES = ("peaks", "calibration", "cbs", "analysis")


class DatabaseError(ValueError):
    """Inconsistent or invalid database content."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- constants

class ConstantsProfile(_Model):
    description: str = ""
    amu_kg: float = Field(gt=0)
    e_C: float = Field(gt=0)
    mass_amu_override: dict[str, float] = {}


class Constants(_Model):
    default_profile: str
    profiles: dict[str, ConstantsProfile]


# ---------------------------------------------------------------- species

class Species(_Model):
    name: str
    mass_amu: float = Field(gt=0)
    data_labels: list[int]
    file_label: int


# ---------------------------------------------------------------- instruments

class EIAConfig(_Model):
    description: str = ""
    L_m: float = Field(gt=0)
    R_out: float = Field(gt=0)
    R_in: float = Field(gt=0)
    R_orbit_m: float = Field(gt=0)
    t0_s: float = 0.0
    provisional: list[str] = []

    @model_validator(mode="after")
    def _radii(self):
        if self.R_out <= self.R_in:
            raise ValueError("R_out must be larger than R_in")
        return self

    @property
    def energy_per_charge_factor(self) -> float:
        """E/q [V] per volt of deflector voltage U: 1 / ln(R_out / R_in)."""
        return 1.0 / math.log(self.R_out / self.R_in)


class FCConfig(_Model):
    description: str = ""
    L_m: float = Field(gt=0)
    aperture_radius_cm: float = Field(gt=0)
    R_load_ohm: float = Field(gt=0)
    provisional: list[str] = []

    @property
    def aperture_area_cm2(self) -> float:
        return math.pi * self.aperture_radius_cm ** 2


class Instruments(_Model):
    eia: dict[str, EIAConfig]
    fc: dict[str, FCConfig]
    acquisition: dict[str, Acquisition]


# ---------------------------------------------------------------- runs

class Laser(_Model):
    pulse: str | None = None
    energy_mJ: float | None = None
    intensity_Wcm2: float | None = None
    spot_um: float | None = None
    approximate: bool = False


class LegacyInfo(_Model):
    profile_peaks: str
    profile_calibration: str
    profile_cbs: str
    profile_analysis: str
    cbs_components: dict[int, Literal["fast", "slow", "ultra_fast", "component_4"]]
    # untitled28 state that is not in the eia record: the LOF pre-filled in the GUI
    # (2025 script: 3.86 - 0.01 = 3.8499999999999996) and the averaging-grid rule
    # ("previous": length of the last shot of the previous voltage, N9; "max": PSexp)
    peaks_lof_m: float | None = None
    peaks_grid: Literal["max", "previous"] = "max"

    def profile(self, stage: str) -> str:
        if stage not in STAGES:
            raise KeyError(f"unknown stage {stage!r}; use one of {STAGES}")
        return getattr(self, f"profile_{stage}")


class RunRecord(_Model):
    include: bool = True
    target: str | None = None
    label: str
    date: dt.date | None = None
    laser: Laser = Laser()
    eia: str | None = None
    fc: str | None = None
    acquisition: str | None = None
    acquisition_overrides: dict = {}     # e.g. { sem = { signal_column = "CH2" } }
    mass_labels: list[int] = []
    spectra_species: list[str] = []
    fit_file_labels: list[int] = []
    notes: str = ""
    legacy: LegacyInfo | None = None

    @model_validator(mode="after")
    def _complete_if_included(self):
        if self.include:
            missing = [k for k in ("target", "eia", "fc", "acquisition", "legacy")
                       if getattr(self, k) is None]
            missing += [k for k in ("mass_labels", "fit_file_labels") if not getattr(self, k)]
            if missing:
                raise ValueError(f"included run lacks {missing}")
        return self


class Run:
    """A run record resolved against the database (instrument objects attached)."""

    def __init__(self, key: str, rec: RunRecord, db: Database):
        self.key, self.rec, self._db = key, rec, db

    def __getattr__(self, name):          # record fields: run.label, run.target, ...
        return getattr(self.rec, name)

    @property
    def eia(self) -> EIAConfig:
        return self._db.instruments.eia[self.rec.eia]

    @property
    def fc(self) -> FCConfig:
        return self._db.instruments.fc[self.rec.fc]

    @property
    def acquisition(self) -> Acquisition:
        """The run's acquisition profile with its acquisition_overrides applied."""
        base = self._db.instruments.acquisition[self.rec.acquisition]
        if not self.rec.acquisition_overrides:
            return base
        return Acquisition(**deep_merge(base.model_dump(), self.rec.acquisition_overrides))

    @property
    def provisional(self) -> list[str]:
        """Provisional values that affect this run, e.g. 'fc.aperture_radius_cm'."""
        if not self.include:
            return []
        return ([f"eia.{p}" for p in self.eia.provisional]
                + [f"fc.{p}" for p in self.fc.provisional])

    def mass_kg(self, profile: str | None = None) -> float:
        return self._db.mass_kg(self.rec.target, profile)

    def folder(self, data_root: Path) -> Path:
        return data_root / self.key

    def __repr__(self):
        return f"Run({self.key!r}, {self.label!r})"


# ---------------------------------------------------------------- studies

class ZbarSettings(_Model):
    e_min_eV: dict[str, float] = {}
    e_max_eV: dict[str, float] = {}
    lower_from_z: int | None = None
    reference: str | None = None


class CollapseSettings(_Model):
    runs: list[str]
    literature: list[str] = []
    min_fast_points: int = 2
    peak_min_prom_frac: float = 0.05
    peak_err_mode: Literal["hwhm", "triplet"] = "hwhm"
    cbs_t_source: Literal["exp", "fit"] = "exp"


class ReconstructionPanel(_Model):
    run: str
    z: list[int]


class ReconstructionSettings(_Model):
    panels: list[ReconstructionPanel] = []


class Study(_Model):
    name: str
    description: str = ""
    runs: list[str]
    extra_runs: list[str] = []
    include_extra: bool = False
    rows: list[str]
    column_energy_mJ: list[float]
    avg_components: dict[str, list[Literal["fast", "slow", "ultra_fast", "component_4"]]] = {}
    zbar: ZbarSettings = ZbarSettings()
    collapse: CollapseSettings | None = None
    reconstruction: ReconstructionSettings = ReconstructionSettings()
    # run key -> component (or "all") -> charge states
    exclude: dict[str, dict[str, list[int]]] = {}

    @field_validator("exclude")
    @classmethod
    def _exclude(cls, v):
        for run, comps in v.items():
            bad = [c for c in comps if c not in (*COMPONENTS, "all")]
            if bad:
                raise ValueError(f"exclude[{run!r}]: unknown component(s) {bad}")
        return v

    def all_runs(self) -> list[str]:
        return self.runs + (self.extra_runs if self.include_extra else [])


# ---------------------------------------------------------------- database

class Database:
    def __init__(self, root: Path, constants: Constants, species: dict[str, Species],
                 instruments: Instruments, runs: dict[str, RunRecord], pipeline: dict,
                 literature: dict, studies: dict[str, Study]):
        self.root = root
        self.constants = constants
        self.species = species
        self.instruments = instruments
        self._runs = runs
        self.pipeline = pipeline
        self.literature = literature
        self.studies = studies

    # -- runs
    @property
    def run_keys(self) -> list[str]:
        return sorted(self._runs)

    def run(self, key: str) -> Run:
        try:
            return Run(key, self._runs[key], self)
        except KeyError:
            raise KeyError(f"no run {key!r} in runs.toml; known: {self.run_keys}") from None

    def runs(self, include_only: bool = True) -> list[Run]:
        return [self.run(k) for k in self.run_keys
                if self._runs[k].include or not include_only]

    # -- constants
    def profile(self, name: str | None = None) -> ConstantsProfile:
        name = name or self.constants.default_profile
        try:
            return self.constants.profiles[name]
        except KeyError:
            raise KeyError(f"no constants profile {name!r}; known: "
                           f"{sorted(self.constants.profiles)}") from None

    def e_C(self, profile: str | None = None) -> float:
        return self.profile(profile).e_C

    def mass_amu(self, species: str, profile: str | None = None) -> float:
        p = self.profile(profile)
        return p.mass_amu_override.get(species, self.species[species].mass_amu)

    def mass_kg(self, species: str, profile: str | None = None) -> float:
        """Ion mass in kg as used by the given constants profile (legacy or corrected)."""
        return self.mass_amu(species, profile) * self.profile(profile).amu_kg

    def study(self, name: str) -> Study:
        try:
            return self.studies[name]
        except KeyError:
            raise KeyError(f"no study {name!r}; known: {sorted(self.studies)}") from None

    # -- checks
    def check(self) -> list[str]:
        """Cross-reference errors (empty list = consistent)."""
        err = []
        prof = self.constants.profiles
        if self.constants.default_profile not in prof:
            err.append(f"default_profile {self.constants.default_profile!r} not defined")
        for name, p in prof.items():
            for sp in p.mass_amu_override:
                if sp not in self.species:
                    err.append(f"profile {name}: override for unknown species {sp!r}")
        ins = self.instruments
        for key, r in self._runs.items():
            where = f"run {key!r}"
            if r.target and r.target not in self.species:
                err.append(f"{where}: unknown target {r.target!r}")
            for sp in r.spectra_species:
                if sp not in self.species:
                    err.append(f"{where}: unknown species {sp!r} in spectra_species")
            if not r.include:
                continue
            for field, table in (("eia", ins.eia), ("fc", ins.fc),
                                 ("acquisition", ins.acquisition)):
                if getattr(r, field) not in table:
                    err.append(f"{where}: unknown {field} {getattr(r, field)!r}")
            if r.acquisition in ins.acquisition and r.acquisition_overrides:
                try:
                    _ = self.run(key).acquisition
                except ValueError as exc:
                    err.append(f"{where}: acquisition_overrides invalid: {exc}")
            for st in STAGES:
                if r.legacy.profile(st) not in prof:
                    err.append(f"{where}: legacy profile_{st} {r.legacy.profile(st)!r} not defined")
            bad = [m for m in r.mass_labels if m not in self.species[r.target].data_labels]
            if bad:
                err.append(f"{where}: mass_labels {bad} not in species {r.target} data_labels")
        for sname, s in self.studies.items():
            where = f"study {sname!r}"
            listed = s.runs + s.extra_runs
            for k in listed:
                if k not in self._runs:
                    err.append(f"{where}: unknown run {k!r}")
                elif not self._runs[k].include:
                    err.append(f"{where}: run {k!r} has include = false")
            for k in s.exclude:
                if k not in listed:
                    err.append(f"{where}: exclude refers to run {k!r} not in the study")
            if s.collapse:
                err += [f"{where}: collapse run {k!r} not in the study"
                        for k in s.collapse.runs if k not in listed]
                err += [f"{where}: unknown literature {k!r}"
                        for k in s.collapse.literature if k not in self.literature]
            if s.zbar.reference and s.zbar.reference not in self.literature:
                err.append(f"{where}: unknown zbar reference {s.zbar.reference!r}")
            err += [f"{where}: reconstruction run {p.run!r} not in the study"
                    for p in s.reconstruction.panels if p.run not in listed]
            for el in s.rows:
                if el not in self.species:
                    err.append(f"{where}: unknown row species {el!r}")
        return err


# ---------------------------------------------------------------- loading

def default_db_root() -> Path:
    env = os.environ.get("EIA_DB_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "db"


def _read(path: Path) -> dict:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise DatabaseError(f"missing database file {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise DatabaseError(f"{path.name}: TOML syntax error: {exc}") from None


def _int_keys(d: dict) -> dict:
    """TOML keys are strings; cbs_components needs integer keys."""
    return {int(k): v for k, v in d.items()}


def load_database(root: Path | str | None = None, strict: bool = True) -> Database:
    """Load and validate the database. strict=True raises on cross-reference errors."""
    root = Path(root) if root else default_db_root()
    try:
        constants = Constants(**_read(root / "constants.toml"))
        species = {k: Species(**v) for k, v in _read(root / "species.toml")["species"].items()}
        instruments = Instruments(**_read(root / "instruments.toml"))
        runs = {}
        for key, rec in _read(root / "runs.toml")["runs"].items():
            leg = rec.get("legacy")
            if leg and "cbs_components" in leg:
                rec = {**rec, "legacy": {**leg, "cbs_components": _int_keys(leg["cbs_components"])}}
            try:
                runs[key] = RunRecord(**rec)
            except ValueError as exc:
                raise DatabaseError(f"runs.toml, run {key!r}: {exc}") from None
        pipeline = _read(root / "pipeline.toml")
        literature = _read(root / "literature.toml")
        studies = {}
        for f in sorted((root / "studies").glob("*.toml")):
            s = Study(**_read(f))
            studies[s.name] = s
    except DatabaseError:
        raise
    except ValueError as exc:            # pydantic ValidationError is a ValueError
        raise DatabaseError(str(exc)) from None
    db = Database(root, constants, species, instruments, runs, pipeline, literature, studies)
    errors = db.check()
    if strict and errors:
        raise DatabaseError("database inconsistent:\n  " + "\n  ".join(errors))
    return db


# ---------------------------------------------------------------- ExclusionList import

_LEGACY_COMP = {1: "fast", 2: "slow", 3: "ultra_fast", 4: "component_4", "all": "all"}


def import_exclusion_list(path: Path, db: Database, study: str) -> dict[str, dict[str, list[int]]]:
    """
    Convert a legacy ExclusionList.txt (EXCLUDE[mJ][element][1|2|3|4|'all'] = [z],
    post-remap numbering 1 = fast, 2 = slow, 3 = ultra-fast) into the study's
    exclude table keyed by run key and physical component. Non-numeric condition
    keys (e.g. '2026_1103') are matched to a run key starting with them.
    """
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "EXCLUDE" for t in node.targets):
            value = ast.literal_eval(node.value)
    if not isinstance(value, dict):
        raise DatabaseError(f"{path}: no 'EXCLUDE = {{...}}' dictionary found")

    s = db.study(study)
    candidates = [db.run(k) for k in s.runs + s.extra_runs]
    out: dict[str, dict[str, list[int]]] = {}
    for cond, els in value.items():
        for el, comps in els.items():
            if isinstance(cond, (int, float)):
                match = [r for r in candidates
                         if r.target == el and r.laser.energy_mJ == float(cond)]
            else:
                match = [r for r in candidates if r.target == el and r.key.startswith(str(cond))]
            if len(match) != 1:
                raise DatabaseError(f"ExclusionList entry ({cond}, {el}) matches "
                                    f"{[r.key for r in match]} in study {study}")
            for comp, zs in comps.items():
                if not zs:
                    continue
                name = _LEGACY_COMP.get(comp)
                if name is None:
                    raise DatabaseError(f"bad component key {comp!r} ({cond}, {el})")
                out.setdefault(match[0].key, {})[name] = sorted(int(z) for z in zs)
    return out


def format_exclude_toml(exclude: dict[str, dict[str, list[int]]]) -> str:
    lines = ["[exclude]"]
    for run in sorted(exclude):
        items = ", ".join(f"{c} = {zs}" for c, zs in sorted(exclude[run].items()))
        lines.append(f'"{run}" = {{ {items} }}')
    return "\n".join(lines)
