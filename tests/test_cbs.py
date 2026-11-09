"""CBS stage (cbs_manual_fitter_v12 port): physics, fit scan, FWHM guess, areas, replay."""
import numpy as np
import pytest
from legacy_reference import fitter_cbs_shape, fitter_V0_from_degen

from eiakit.cbs import CBSModel, Component, rounding_bounds

M = 92.206 * 1.6726219e-27          # legacy_proton Nb
L = 3.85
E = 1.602e-19
TOF = np.linspace(2e-6, 400e-6, 40000)


def model(**kw):
    return CBSModel(tof=TOF, m_kg=M, L_m=L, e_C=E, **kw)


@pytest.mark.parametrize("kT,u,z", [(8.0, 95e3, 1), (300.0, 2.2e5, 5), (35.0, 1.8e4, 1)])
def test_V0_and_shape_match_the_fitter(kT, u, z):
    mo = model()
    V0 = mo.V0_from_u(kT, u, z)
    ref = fitter_V0_from_degen(kT, u, z, M)
    assert (V0 is None) == (ref is None)
    if V0 is not None:
        assert V0 == pytest.approx(ref, rel=1e-14)
        assert np.allclose(mo.shape(kT, V0, z), fitter_cbs_shape(TOF, kT, V0, z, M, L),
                           rtol=1e-13, atol=0)
        assert mo.u_of(kT, V0, z) == pytest.approx(u, rel=1e-12)


def _synthetic(mo, comps, z, noise=0.0, seed=0):
    y = sum(c.A * mo.shape(c.kT, c.V0, z) for c in comps)
    if noise:
        y = y * (1 + noise * np.random.default_rng(seed).standard_normal(y.size))
    return y


def test_fit_scan_recovers_a_grid_kT_and_A():
    mo = model()
    grid = mo.scan_grid(100.0)
    kT_true = float(grid[187])
    u = 2.0e5
    z = 3
    V0 = mo.V0_from_u(kT_true, u, z)
    t_pk = L / u * 1e6
    c = Component(kT=50.0, V0=mo.V0_from_u(50.0, u, z), A=1.0, t_peak=t_pk,
                  fwhm=mo.fwhm(kT_true, V0, z), win_left=t_pk * 0.8, win_right=t_pk * 1.3)
    curve = 2.5e-18 * mo.shape(kT_true, V0, z)
    r = mo.fit_component([c], 0, curve, z)
    assert r["kT"] == kT_true
    assert c.A == pytest.approx(2.5e-18, rel=1e-10)
    assert c.V0 == pytest.approx(V0, rel=1e-12)


def test_two_component_fit_with_subtraction():
    mo = model()
    z = 2
    grid = mo.scan_grid(100.0)
    k1, k2 = float(grid[40]), float(grid[400])
    u1, u2 = 2.4e5, 1.2e5
    c1 = Component(kT=10.0, V0=mo.V0_from_u(10.0, u1, z), A=1e-18, t_peak=L / u1 * 1e6,
                   fwhm=1.0)
    c2 = Component(kT=20.0, V0=mo.V0_from_u(20.0, u2, z), A=1e-17, t_peak=L / u2 * 1e6,
                   fwhm=5.0)
    for c, kt in ((c1, k1), (c2, k2)):
        V = mo.V0_from_u(kt, L / (c.t_peak * 1e-6), z)
        c.win_left, c.win_right = mo.half_edges(kt, V, z)
        hw = c.win_right - c.win_left
        c.win_left, c.win_right = c.win_left - hw, c.win_right + hw
    truth = [Component(kT=k1, V0=mo.V0_from_u(k1, u1, z), A=3e-18),
             Component(kT=k2, V0=mo.V0_from_u(k2, u2, z), A=4e-17)]
    curve = _synthetic(mo, truth, z)
    for _ in range(3):
        mo.fit_component([c1, c2], 0, curve, z)
        mo.fit_component([c1, c2], 1, curve, z)
    assert (c1.kT, c2.kT) == (k1, k2)
    assert c1.A == pytest.approx(3e-18, rel=1e-3) and c2.A == pytest.approx(4e-17, rel=1e-3)


def test_kT_from_fwhm_inverts_the_model_fwhm():
    mo = model()
    u, z = 1.5e5, 4
    for kT in (5.0, 120.0, 900.0):
        V0 = mo.V0_from_u(kT, u, z)
        if V0 is None:
            continue
        fw = mo.fwhm(kT, V0, z, TOF[::4])
        kT_g, _ = mo.kT_from_fwhm(fw, u, z)
        assert kT_g == pytest.approx(kT, rel=2e-3)


def test_areas_fractions_and_bounds():
    mo = model()
    z = 1
    u1, u2 = 9.5e4, 1.8e4
    comps = []
    for kT, u, A in ((8.0, u1, 1e-21), (35.0, u2, 5e-15)):
        V0 = mo.V0_from_u(kT, u, z) or 0.0
        t = L / u * 1e6
        comps.append(Component(kT=kT, V0=V0, A=A, t_peak=t, fwhm=0.2 * t,
                               win_left=0.7 * t, win_right=1.6 * t, extra={"cbs": len(comps) + 1}))
    curve = _synthetic(mo, comps, z, noise=0.01)
    a = mo.areas(comps, curve, z)
    assert list(a.CBS) == [1, 2]
    assert a.Fraction.sum() == pytest.approx(1.0) and a.Fraction_full.sum() == pytest.approx(1.0)
    assert (a.Win_coverage <= 1 + 1e-12).all()
    assert np.allclose(a.Ions_model_full_cm2, a.Area_model_full * 1e-6 / E)
    b = rounding_bounds(mo, comps, 0, curve, z)
    assert all(v > 0 for v in b.values())
