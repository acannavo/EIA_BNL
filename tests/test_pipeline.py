"""Provenance manifests, status and incremental runs (with a toy two-stage pipeline)."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from eiakit import pipeline as pl


class ToyRun:
    def __init__(self, key):
        self.key = key


@pytest.fixture
def toy(tmp_path, monkeypatch):
    data, processed, dbroot = tmp_path / "data", tmp_path / "processed", tmp_path / "db"
    for d in (data / "R1", processed, dbroot):
        d.mkdir(parents=True)
    (data / "R1" / "a.txt").write_text("1 2 3")
    db = SimpleNamespace(root=dbroot, params={"x": 1.0, "nested": {"y": "a"}}, fail=False)
    calls = []

    def a_exec(db_, run, ctx):
        calls.append("A")
        v = sum(map(float, (ctx.data_root / run.key / "a.txt").read_text().split()))
        (ctx.out / "a.json").write_text(json.dumps({"sum": v * db_.params["x"]}))
        return True, {"sum": v}

    def b_exec(db_, run, ctx):
        calls.append("B")
        if db_.fail:
            raise RuntimeError("boom")
        a = json.loads((ctx.upstream_out["A"] / "a.json").read_text())
        (ctx.out / "b.txt").write_text(str(a["sum"] * 2))
        return True, {"b": a["sum"] * 2}

    stages = {
        "A": pl.Stage("A", "10_a", [], ["physics"], lambda d, r: dict(d.params),
                      lambda d, r, root, ins: ins.add(root / r.key / "a.txt"), a_exec),
        "B": pl.Stage("B", "20_b", ["A"], ["cbs"], lambda d, r: {"k": 2},
                      lambda d, r, root, ins: None, b_exec),
    }
    monkeypatch.setattr(pl, "STAGES", stages)
    monkeypatch.setattr(pl, "ORDER", ["A", "B"])
    return SimpleNamespace(db=db, run=ToyRun("R1"), data=data, processed=processed,
                           calls=calls)


def _status(t, name):
    return pl.stage_status(t.db, t.run, name, t.data, t.processed)


def _run_all(t, **kw):
    for run, name, _ in pl.plan(t.db, [t.run], t.data, t.processed, **kw):
        pl.run_stage(t.db, run, name, t.data, t.processed)


def test_missing_then_ok(toy):
    assert _status(toy, "A")["state"] == "missing"
    _run_all(toy)
    assert toy.calls == ["A", "B"]
    assert _status(toy, "A")["state"] == "ok" and _status(toy, "B")["state"] == "ok"
    man = json.loads((toy.processed / "R1" / "10_a" / "manifest.json").read_text())
    assert set(man) >= {"code", "params", "inputs", "outputs", "run_id", "upstream", "summary"}
    assert "R1/a.txt" in man["inputs"] and "a.json" in man["outputs"]
    assert pl.plan(toy.db, [toy.run], toy.data, toy.processed) == []


def test_param_change_makes_stage_and_downstream_stale(toy):
    _run_all(toy)
    toy.db.params["nested"]["y"] = "b"
    st = _status(toy, "A")
    assert st["state"] == "stale"
    assert any("nested.y: 'a' -> 'b'" in r for r in st["reasons"])
    assert _status(toy, "B")["state"] == "stale"
    names = [n for _, n, _ in pl.plan(toy.db, [toy.run], toy.data, toy.processed)]
    assert names == ["A", "B"]


def test_input_content_vs_touch(toy):
    _run_all(toy)
    f = toy.data / "R1" / "a.txt"
    t = time.time() + 5
    os.utime(f, (t, t))                               # touched, same content
    assert _status(toy, "A")["state"] == "ok"
    f.write_text("1 2 4")
    st = _status(toy, "A")
    assert st["state"] == "stale" and any("input changed: R1/a.txt" in r for r in st["reasons"])


def test_upstream_rerun_marks_downstream(toy):
    _run_all(toy)
    pl.run_stage(toy.db, toy.run, "A", toy.data, toy.processed)     # same inputs: same run_id
    assert _status(toy, "B")["state"] == "ok"
    (toy.data / "R1" / "a.txt").write_text("5")
    pl.run_stage(toy.db, toy.run, "A", toy.data, toy.processed)
    st = _status(toy, "B")
    assert st["state"] == "stale" and "upstream 'A' was rerun since" in st["reasons"]


def test_failure_is_recorded(toy):
    toy.db.fail = True
    _run_all(toy)
    st = _status(toy, "B")
    assert st["state"] == "failed" and "boom" in " ".join(st["reasons"])
    toy.db.fail = False
    _run_all(toy)
    assert _status(toy, "B")["state"] == "ok"


def test_code_hash_tracks_module_source():
    assert pl.code_hash(["cbs"]) != pl.code_hash(["physics"])
    assert pl.code_hash(["cbs", "physics"]) == pl.code_hash(["physics", "cbs"])


def test_flatten():
    assert pl.flatten({"a": {"b": 1, "c": [1, 2]}, "d": 3}) == {"a.b": 1, "a.c": [1, 2], "d": 3}


def test_outputs_modified_outside_the_pipeline(toy):
    _run_all(toy)
    (toy.processed / "R1" / "10_a" / "a.json").write_text("{}")
    st = _status(toy, "A")
    assert st["state"] == "stale" and "a.json modified" in " ".join(st["reasons"])
    (toy.processed / "R1" / "10_a" / "a.json").unlink()
    assert "a.json missing" in " ".join(_status(toy, "A")["reasons"])
