"""Spectra analysis (plot_eia_powerLaw_test port) on synthetic data."""
import numpy as np
import pytest

from eiakit.analysis import (
    Dataset,
    compute_sheil,
    compute_zbar,
    energy_eV,
    fit_power_law,
    summarize,
)

E_C = 1.602e-19
M = 26.9815 * 1.66053907e-27


def _ds(curves):
    t = np.linspace(2.0, 400.0, 20000)
    return Dataset(key="R", name="Al_1mJ", species="Al", energy_mJ=1.0, m_kg=M, e_C=E_C,
                   L_m=3.85, tof_us=t, z_curves={z: f(t) for z, f in curves.items()},
                   all_z=sorted(curves))


def test_zbar_of_two_states_with_constant_number_ratio():
    # N_2 / N_1 = 3 everywhere -> zbar = (1 + 2*3) / 4 = 1.75; J_z carries the factor z
    g = lambda t: np.exp(-((t - 100) / 30) ** 2) * 1e-6  # noqa: E731
    ds = _ds({1: lambda t: g(t), 2: lambda t: 2 * 3 * g(t)})
    zb = compute_zbar(ds)
    v = zb["valid"] & np.isfinite(zb["zbar"])
    assert np.allclose(zb["zbar"][v], 1.75)
    zj = compute_zbar(ds, jacobian=True)
    assert np.allclose(zj["zbar"][v], zb["zbar"][v])


def test_power_law_fit_recovers_exponent():
    E = np.logspace(1, 4, 400)
    f = fit_power_law(E, 0.3 * E ** 0.4, np.ones(E.size, bool), (None, 5e3))
    assert f["b"] == pytest.approx(0.4, abs=1e-10) and f["C"] == pytest.approx(0.3, rel=1e-9)
    assert f["E_hi"] <= 5e3 and f["r2"] == pytest.approx(1.0)


def test_sheil_integrals_energy_equals_time():
    ds = _ds({1: lambda t: np.exp(-((t - 150) / 40) ** 2) * 1e-6,
              3: lambda t: 3 * np.exp(-((t - 60) / 10) ** 2) * 1e-6})
    S = compute_sheil(ds)
    assert S["check_E_vs_t"] < 1e-4
    assert S["yield_sr"] == pytest.approx(sum(S["n_t"].values()), rel=1e-4)


def test_summary_peak_and_energy():
    ds = _ds({2: lambda t: np.exp(-((t - 50) / 5) ** 2)})
    r = summarize(ds).iloc[0]
    assert r.t_peak_us == pytest.approx(50, abs=0.05)
    assert r.E_peak_keV * 1e3 == pytest.approx(float(energy_eV(ds, r.t_peak_us)), rel=1e-12)
    assert r.EperQ_peak_kV == pytest.approx(r.E_peak_keV / 2)


def test_collapse_fit_through_origin_and_joint():
    from eiakit.collapse import fit_through_origin, weighted_joint
    z = np.arange(1.0, 11.0)
    f = fit_through_origin(z, z / 12.5)
    assert f["zstar"] == pytest.approx(12.5) and f["R2"] == pytest.approx(1.0)
    j = weighted_joint([10.0, 12.0], [1.0, 1.0])
    assert j["zstar"] == pytest.approx(11.0) and j["se"] == pytest.approx(1 / np.sqrt(2))
    assert j["rms_pct"] == pytest.approx(100 / 11)


def test_eia_peaks_finds_slow_fast_ultra():
    from eiakit.collapse import eia_peaks
    g = lambda t, c, w: np.exp(-((t - c) / w) ** 2)  # noqa: E731
    ds = _ds({3: lambda t: 0.3 * g(t, 20, 2) + 1.0 * g(t, 40, 3) + 0.8 * g(t, 120, 15)})
    p = eia_peaks(ds, 0.05, "hwhm").set_index("component")
    assert p.loc["slow", "t_peak"] == pytest.approx(120, abs=0.1)
    assert p.loc["fast", "t_peak"] == pytest.approx(40, abs=0.1)
    assert p.loc["ultra_fast", "t_peak"] == pytest.approx(20, abs=0.1)
    hw = 3 * np.sqrt(np.log(2))                     # half width of exp(-(x/3)^2)
    assert p.loc["fast", "t_after"] - p.loc["fast", "t_before"] == pytest.approx(2 * hw, rel=1e-2)


def test_zbar_fill_floor_and_mask_options():
    """I4 / I5: z = 1 measured only for t >= 150 us, zero-filled before (legacy)."""
    from eiakit.analysis import missing_ions
    g = lambda t: np.exp(-((t - 100) / 40) ** 2) * 1e-6  # noqa: E731
    ds = _ds({1: lambda t: np.where(t >= 150, g(t), 0.0), 2: lambda t: 2 * g(t)})
    te = np.linspace(150, 400, 50)
    ds.z_exp = {1: (te, g(te)), 2: (np.linspace(2, 400, 50), 2 * g(np.linspace(2, 400, 50)))}
    # zero fill: only z = 2 before 150 us -> zbar = 2 there
    zb = compute_zbar(ds)
    E_early = zb["E_eV"] > energy_eV(ds, 140.0)
    assert np.allclose(zb["zbar"][E_early & zb["valid"]], 2.0)
    # extrapolated / edge fill: z = 1 continued before 150 us -> zbar < 2
    for fill in ("extrapolate", "edge"):
        zf = compute_zbar(ds, fill=fill)
        assert np.nanmax(zf["zbar"][E_early & zf["valid"]]) < 2.0
    # mask: points where the possibly missing z = 1 exceeds 5 % are invalid
    zm = compute_zbar(ds, mask="measured")
    assert zm["valid"].sum() < zb["valid"].sum()
    # noise floor
    zn = compute_zbar(ds, min_frac=0.1)
    assert zn["valid"].sum() < zb["valid"].sum()
    m = missing_ions(ds, ds.tof_us, [1], "extrapolate")
    assert m[ds.tof_us < 150].max() <= g(150.0) + 1e-18 and m[ds.tof_us > 160].max() == 0

