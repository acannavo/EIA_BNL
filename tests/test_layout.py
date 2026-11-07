from pathlib import Path

import pytest

from eiakit.checksum import build_manifest, compare_manifests
from eiakit.layout import Kind, classify, discover_runs
from eiakit.paths import PathsNotConfigured, get_paths


@pytest.mark.parametrize("name", [
    "Output", "Output_detail", "EIA-Results", "QC_Interp", "2_SEM",
    "1_FC_Start", "3_FC_Stop", "0_FocalPosition", "0_FocalSearch"])
def test_canonical(name):
    assert classify(name) is Kind.CANONICAL


@pytest.mark.parametrize("name", [
    "Output~", "Output~~", "Output_", "Output_0", "Output_detail~", "Output_detail_",
    "Output_detail_0", "EIA-Results~", "EIA-Results_", "EIA-Results_0", "EIA-Results_b",
    "0_FocalPosition_1", "wrongolo", "wrongolo1", "wrong"])
def test_variant(name):
    assert classify(name) is Kind.VARIANT


@pytest.mark.parametrize("name", [
    "LowStateCharge", "4_MultiShotSameSpot", "Full", "Miscellaneous", "0_FC",
    "1_InSlitWideScan", "Slicing", "Slicing_limited", "Data Backup", "test", "Data3 only"])
def test_extra(name):
    assert classify(name) is Kind.EXTRA


def _make_tree(root: Path):
    run = root / "20250513"
    for d in ["1_FC_Start", "2_SEM/Output", "2_SEM/Output_detail", "2_SEM/Output~",
              "2_SEM/Output_0/wrongolo", "3_FC_Stop", "EIA-Results/QC_Interp",
              "EIA-Results_b", "Miscellaneous/0_FC"]:
        (run / d).mkdir(parents=True)
    (run / "2_SEM/Output/p0.5.txt").write_text("m,z,TOF,Yield\n")
    (run / "2_SEM/Output~/p0.5.txt").write_text("wrong\n")
    (root / "notes").mkdir()          # no 2_SEM: not a run
    return run


def test_discover_and_scan(tmp_path):
    _make_tree(tmp_path)
    runs = discover_runs(tmp_path)
    assert [r.name for r in runs] == ["20250513"]
    r = runs[0]
    assert r.has("2_SEM/Output") and r.has("EIA-Results/QC_Interp")
    assert {p.name for p in r.variants} == {"Output~", "Output_0", "EIA-Results_b"}
    assert {p.name for p in r.extras} == {"Miscellaneous", "0_FC"}


def test_manifest_skip_variants_and_compare(tmp_path):
    _make_tree(tmp_path)
    full = {r["relpath"]: r for r in build_manifest(tmp_path)}
    clean = {r["relpath"]: r for r in build_manifest(tmp_path, skip_variants=True)}
    assert "20250513/2_SEM/Output~/p0.5.txt" in full
    assert "20250513/2_SEM/Output~/p0.5.txt" not in clean
    assert compare_manifests(full, full).identical
    d = compare_manifests(full, clean)
    assert d.only_a == ["20250513/2_SEM/Output~/p0.5.txt"] and not d.different


def test_paths_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EIA_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("EIA_PROCESSED_ROOT", str(tmp_path / "processed"))
    monkeypatch.setenv("EIA_STUDIES_ROOT", str(tmp_path / "studies"))
    assert get_paths().data == tmp_path / "data"
    monkeypatch.setenv("EIA_PROCESSED_ROOT", str(tmp_path / "data" / "processed"))
    with pytest.raises(PathsNotConfigured):
        get_paths()
    monkeypatch.delenv("EIA_DATA_ROOT")
    with pytest.raises(PathsNotConfigured):
        get_paths()
