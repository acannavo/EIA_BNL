"""Snapshot diff and changelog entry (Step 11), on synthetic snapshots."""
import json

import pandas as pd

from eiakit.paper import changelog_entry, diff, snapshot_dir


def _snap(root, label, values, corr):
    d = snapshot_dir(root, "S")
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"group": g, "item": i, "dataset": ds, "value": v}
                  for (g, i, ds), v in values.items()]).to_csv(d / f"{label}.csv", index=False)
    (d / f"{label}.json").write_text(json.dumps(
        {"git": "x", "source": "pipeline", "corrections": corr,
         "calibration": {"fc_source": "start"}}))


def test_diff_and_changelog(tmp_path):
    a = {("zbar", "C", "Al"): 0.3482, ("zbar", "C", "Nb"): 0.2795,
         ("cbs_params", "kT_eV", "r z2 fast"): 100.0, ("energy", "m1", "x"): 1.0}
    b = dict(a)
    b[("cbs_params", "kT_eV", "r z2 fast")] = 99.0
    b[("zbar", "C", "Nb")] = 0.2796
    del b[("energy", "m1", "x")]
    b[("energy", "m2", "x")] = 2.0
    _snap(tmp_path, "a", a, {"mass_constants": "legacy"})
    _snap(tmp_path, "b", b, {"mass_constants": "codata"})
    res = diff(tmp_path, "S", "a", "b")
    s = res["summary"].set_index(["group", "item"])
    assert s.loc[("cbs_params", "kT_eV"), "changed"] == 1
    assert abs(s.loc[("cbs_params", "kT_eV"), "max_rel"] - 0.01) < 1e-12
    assert s.loc[("zbar", "C"), "changed"] == 1 and s.loc[("zbar", "C"), "at"] == "Nb"
    assert len(res["only_a"]) == 1 and len(res["only_b"]) == 1
    txt = changelog_entry(res, "a", "b", "I1: test", "why")
    assert "## I1: test" in txt and "`mass_constants`: 'legacy' → 'codata'" in txt
    assert "| cbs_params | kT_eV | 1 / 1 |" in txt and "| 100 | 99 |" in txt
    assert "only in `b`: 1" in txt


def test_keep_last_save_modes():
    """N1: p600 of 20250611 - first save z = 1..3, last save only a z = 1 refit."""
    from eiakit.legacy import keep_last_save
    rows = pd.DataFrame({"file": ["p600.txt"] * 4 + ["p10.txt"],
                         "m": [92] * 5, "z": [1, 2, 3, 1, 1],
                         "TOF": [2.48e-5, 1.76e-5, 1.44e-5, 2.45e-5, 1e-4],
                         "Yield": [-1.0, -2.0, -3.0, -1.5, -9.0]})
    detail = pd.DataFrame({"file": ["p600.txt"], "TOF": [2.45e-5], "YieldAveFit": [-1.5]})
    per, notes = keep_last_save(rows, detail, "last_per_mz")
    p6 = per[per.file == "p600.txt"].sort_values("z")
    assert list(p6.z) == [1, 2, 3] and list(p6.Yield) == [-1.5, -2.0, -3.0]
    assert len(per[per.file == "p10.txt"]) == 1 and len(notes) == 1
    last, notes = keep_last_save(rows, detail, "last_save")
    assert list(last[last.file == "p600.txt"].z) == [1]
    assert notes[0]["method"] == "last save = Output_detail"


def test_resolve_overlaps():
    """N8: O+ fitted on the Nb6+ peak (one peak, two rows); C3+/O4+ unresolvable."""
    import numpy as np

    from eiakit.legacy import resolve_overlaps
    M = {92: 92.906, 15: 15.99, 12: 12.01}
    C = 1e-5

    def t(m, z, off=0.0):
        return C * np.sqrt(M[m] / z) * (1 + off)
    rows = pd.DataFrame({
        "file": ["p.txt"] * 7,
        "m": [92, 92, 92, 92, 15, 12, 15], "z": [5, 6, 7, 8, 1, 3, 4],
        "TOF": [t(92, 5), t(92, 6, 1e-4), t(92, 7), t(92, 8), t(92, 6, 2e-4),
                t(12, 3), t(15, 4, 1e-4)],
        "Yield": [-1.0, -2.0, -1.0, -1.0, -2.0, -0.1, -0.1]})
    new, notes = resolve_overlaps(rows, M, [92])
    got = {(n["m"], n["z"]) for n in notes}
    assert got == {(15, 1), (15, 4)}, notes       # O+ on Nb6+; O4+ vs C3+ -> lower z kept
    assert len(new) == 5 and not ((new.m == 92) & (new.z == 6)).sum() == 0
    # an O+ fitted on its own line (resolved from Nb6+) is kept
    rows.loc[4, "TOF"] = t(15, 1)
    _, notes = resolve_overlaps(rows, M, [92])
    assert {(n["m"], n["z"]) for n in notes} == {(15, 4)}
