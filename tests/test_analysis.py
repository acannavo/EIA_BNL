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
