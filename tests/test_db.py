import math
import shutil
from pathlib import Path

import pytest

from eiakit.db import (
    DatabaseError,
    default_db_root,
    format_exclude_toml,
    import_exclusion_list,
    load_database,
)


@pytest.fixture(scope="module")
def db():
    return load_database()


def test_loads_and_is_consistent(db):
    assert db.check() == []
    keys = {r.key for r in db.runs()}                  # new runs may be added: the 7 base runs
    assert keys >= {"20250513", "20250602", "20250606", "20250611", "20250617", "20250623",
                    "2026_1103 Nb analysis"}
    assert {r.key for r in db.runs(include_only=False)} >= {"20250506", "20250508"}


def test_run_key_is_folder_name(db):
    r = db.run("20250623")
    assert (r.target, r.laser.energy_mJ, r.eia.L_m, r.fc.L_m) == ("Nb", 370.0, 3.85, 2.38)
    assert r.folder(Path("/data")) == Path("/data/20250623")
    new = db.run("2026_1103 Nb analysis")
    assert (new.eia.L_m, new.fc.L_m, new.fc.R_load_ohm) == (1.22, 0.58, 51.0)
    assert new.fc.aperture_radius_cm == 1.0 and "fc.aperture_radius_cm" not in new.provisional
    with pytest.raises(KeyError):
        db.run("nope")


def test_mass_profiles_reproduce_issue_I1(db):
    # TODO file: rescale factor s = M_new / M_old, Al 0.99278, Nb 1.00031
    def ratio(sp):
        return db.mass_kg(sp, "corrected") / db.mass_kg(sp, "legacy_proton")
    assert ratio("Al") == pytest.approx(0.99278, abs=5e-6)
    assert ratio("Nb") == pytest.approx(1.00031, abs=5e-6)
    assert db.mass_kg("Nb", "legacy_proton") == 92.206 * 1.6726219e-27       # fitter v12
    assert db.mass_kg("Al", "legacy_amu_e1602") == 26.9815 * 1.66053907e-27  # powerLaw script
    assert db.e_C() == 1.602176634e-19 and db.e_C("legacy_proton") == 1.602e-19


def test_analyzer_factor(db):
    # E/q = U / ln(102.5/97.5) = 19.996 U  (the audit sheet's "19.99" is truncated)
    assert db.run("20250513").eia.energy_per_charge_factor == pytest.approx(19.9958, abs=1e-4)
    assert db.run("20250513").fc.aperture_area_cm2 == pytest.approx(math.pi * 0.25)


def test_legacy_component_labels(db):
    # remap tables of plot_cbs_multipanel_v2 / plot_collapse_v2
    assert db.run("20250513").legacy.cbs_components[1] == "slow"      # Al: {1:2, 2:1}
    assert db.run("20250606").legacy.cbs_components[1] == "fast"      # Al 180: None
    assert db.run("20250611").legacy.cbs_components[1] == "fast"      # Nb: None
    assert db.run("20250623").legacy.cbs_components[1] == "slow"      # Nb 370: swap


def _copy_db(tmp_path):
    dst = tmp_path / "db"
    shutil.copytree(default_db_root(), dst)
    return dst


def test_broken_reference_is_caught(tmp_path):
    dst = _copy_db(tmp_path)
    p = dst / "runs.toml"
    p.write_text(p.read_text(encoding="utf-8").replace('fc              = "FC_2025"',
                                                       'fc              = "FC_typo"', 1),
                 encoding="utf-8")
    with pytest.raises(DatabaseError, match="FC_typo"):
        load_database(dst)
    assert any("FC_typo" in e for e in load_database(dst, strict=False).check())


def test_syntax_error_is_caught(tmp_path):
    dst = _copy_db(tmp_path)
    (dst / "species.toml").write_text("[species.Al\nmass_amu = 1", encoding="utf-8")
    with pytest.raises(DatabaseError, match="species.toml"):
        load_database(dst)


def test_import_exclusion_list(db, tmp_path):
    f = tmp_path / "ExclusionList.txt"
    f.write_text("EXCLUDE = {780: {'Al': {1: [3], 2: [], 3: [9, 8], 'all': [1]},"
                 " 'Nb': {2: [24]}}, '2026_1103': {'Nb': {1: [5]}}}\n", encoding="utf-8")
    ex = import_exclusion_list(f, db, "PRL_AlNb")
    assert ex == {"20250513": {"fast": [3], "ultra_fast": [8, 9], "all": [1]},
                  "20250611": {"slow": [24]},
                  "2026_1103 Nb analysis": {"fast": [5]}}
    text = format_exclude_toml(ex)
    assert text.splitlines()[0] == "[exclude]" and '"20250611" = { slow = [24] }' in text


def test_profile_for_follows_corrections_switch():
    d = load_database()
    r = d.run("20250513")
    d.pipeline.setdefault("corrections", {})["mass_constants"] = "legacy"
    assert d.profile_for(r, "cbs") == r.legacy.profile("cbs")
    d.pipeline.setdefault("corrections", {})["mass_constants"] = "codata"
    assert d.profile_for(r, "cbs") == "corrected"
    assert d.profile_for(r, "analysis") == "corrected"
    assert d.profile_for(r, "peaks") == r.legacy.profile("peaks")   # peaks untouched by I1


def test_cli_set_overrides():
    from eiakit.cli import _apply_overrides, _parse_set
    assert _parse_set("calibration.output_duplicates=all") == ("calibration",
                                                               "output_duplicates", "all")
    assert _parse_set("calibration.n_interp=5000")[2] == 5000
    d = _apply_overrides(load_database(), [_parse_set("corrections.mass_constants=legacy")])
    assert d.corrections()["mass_constants"] == "legacy"
    with pytest.raises(DatabaseError):
        _apply_overrides(load_database(), [_parse_set("calibration.nope=1")])
    with pytest.raises(DatabaseError):
        _parse_set("output_duplicates=all")
