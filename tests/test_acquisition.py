import numpy as np
import pytest

from eiakit.acquisition import FCFormat, SEMFormat, read_trace, scan_folder, template_to_regex
from eiakit.db import load_database


@pytest.fixture(scope="module")
def db():
    return load_database()


def sem(**kw):
    base = dict(folder="2_SEM", patterns=["{int}p{dec}v{rep}.csv"], header="skiprows:0",
                time_column=0, signal_column=1)
    return SEMFormat(**{**base, **kw})


@pytest.mark.parametrize("key,name,expected", [
    ("20250513", "0p5v1.csv", (0.5, 1)),
    ("20250513", "12p25v3.csv", (12.25, 3)),
    ("20250513", "0p05v2.CSV", (0.05, 2)),        # leading zeros of the decimals kept
    ("20250513", "10v2.csv", (10.0, 2)),          # second template, no decimals
    ("20250513", "notes.csv", None),
    ("2026_1103 Nb analysis", "000p6Data1.csv", (0.6, 1)),
    ("2026_1103 Nb analysis", "091p0Data4.csv", (91.0, 4)),
    ("2026_1103 Nb analysis", "000p6Datafirst.csv", None),     # ignored
    ("2026_1103 Nb analysis", "000p6second.csv", None),        # ignored
])
def test_db_formats(db, key, name, expected):
    assert db.run(key).acquisition.sem.parse_name(name) == expected


@pytest.mark.parametrize("pattern,name,expected", [
    ("U{U}_shot{rep}.txt", "U12.5_shot3.txt", (12.5, 3)),
    ("{*}_{U}V.csv", "Nb_run7_0.75V.csv", (0.75, 1)),               # no {rep} -> 1
    ("{int}p{dec}mV_{rep}.csv", "250p0mV_2.csv", (0.25, 2)),        # with u_scale = 1e-3
    ("re:^scan_(?P<U>[0-9.]+)_(?P<rep>\\d+)\\.dat$", "scan_3.5_07.dat", (3.5, 7)),
])
def test_custom_templates(pattern, name, expected):
    fmt = sem(patterns=[pattern], u_scale=1e-3 if "mV" in pattern else 1.0)
    assert fmt.parse_name(name) == pytest.approx(expected)


@pytest.mark.parametrize("bad", ["{dec}v{rep}.csv", "{int}_{int}.csv", "v{rep}.csv"])
def test_bad_templates(bad):
    with pytest.raises(ValueError):
        sem(patterns=[bad])


def test_template_is_literal_elsewhere():
    rx = template_to_regex("a.b{rep}(x).csv")
    assert rx.match("a.b1(x).csv") and not rx.match("aXb1(x).csv")


def test_run_overrides(db, tmp_path):
    import shutil

    from eiakit.db import default_db_root
    dst = tmp_path / "db"
    shutil.copytree(default_db_root(), dst)
    p = dst / "runs.toml"
    text = p.read_text(encoding="utf-8").replace(
        '[runs."20250513".legacy]',
        '[runs."20250513".acquisition_overrides]\n'
        'sem = { signal_column = 2, ignore = ["*bad*"] }\n\n[runs."20250513".legacy]')
    p.write_text(text, encoding="utf-8")
    r = load_database(dst).run("20250513")
    assert r.acquisition.sem.signal_column == 2 and r.acquisition.sem.ignore == ["*bad*"]
    assert r.acquisition.sem.patterns == db.run("20250513").acquisition.sem.patterns
    assert load_database(dst).run("20250602").acquisition.sem.signal_column == 1


def test_scan_folder(db, tmp_path):
    fmt = db.run("2026_1103 Nb analysis").acquisition.sem
    for n in ["000p6Data1.csv", "000p6Data2.csv", "001p2Data1.csv", "000p6first.csv",
              "readme.txt"]:
        (tmp_path / n).write_text("x")
    sc = scan_folder(tmp_path, fmt)
    assert sc.by_voltage() == {0.6: [1, 2], 1.2: [1]}
    assert [f.name for f in sc.ignored] == ["000p6first.csv"]
    assert [f.name for f in sc.unmatched] == ["readme.txt"]
    assert scan_folder(tmp_path / "nope", fmt).missing_folder


def test_read_2025_style(tmp_path):
    f = tmp_path / "0p5v1.csv"
    lines = [f"meta{i},x" for i in range(22)] + ["1e-6,-0.5", "2e-6,-0.25", "bad,1"]
    f.write_text("\n".join(lines))
    t, y = read_trace(f, sem(header="skiprows:22"))
    assert np.allclose(t, [1e-6, 2e-6]) and np.allclose(y, [-0.5, -0.25])


def test_read_2026_style_with_scales(tmp_path):
    f = tmp_path / "000p6Data1.csv"
    f.write_text("Model,MDO\nRecord Length,3\n\nTIME,CH1,CH3\n1,9,0.1\n2,9,0.2\n3,9,0.3\n")
    fmt = sem(patterns=["{int}p{dec}Data{rep}.csv"], header="detect:TIME", time_column="TIME",
              signal_column="CH3", time_scale=1e-6, signal_scale=-1.0)
    t, y = read_trace(f, fmt)
    assert np.allclose(t, [1e-6, 2e-6, 3e-6]) and np.allclose(y, [-0.1, -0.2, -0.3])
    with pytest.raises(ValueError, match="CH4"):
        read_trace(f, sem(header="detect:TIME", time_column="TIME", signal_column="CH4"))


def test_read_fc_style(tmp_path):
    f = tmp_path / "FC1Data1.csv"
    f.write_text("\n".join([f"m{i}" for i in range(20)] + ["TIME,CH1,CH2", "0,5,1", "1,5,2"]))
    fmt = FCFormat(folder="1_FC_Start", patterns=["{*}Data{rep}.csv"], header="skiprows:20",
                   time_column="TIME", signal_column="CH2")
    t, y = read_trace(f, fmt)
    assert np.allclose(y, [1, 2]) and fmt.match("FC1Data3.csv")["rep"] == "3"
