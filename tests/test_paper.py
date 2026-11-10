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
