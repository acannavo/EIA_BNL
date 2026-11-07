import legacy_reference as legacy
import numpy as np
import pytest

from eiakit import physics as ph
from eiakit.db import load_database

_trapz = getattr(np, "trapezoid", None) or np.trapz


@pytest.fixture(scope="module")
def db():
    return load_database()


@pytest.fixture(scope="module")
def nb(db):
    """Nb, 2025 geometry, legacy fitter constants (proton mass, Nb 92.206, e 1.602e-19)."""
    return dict(m=db.mass_kg("Nb", "legacy_proton"), e=db.e_C("legacy_proton"),
                L=db.run("20250623").eia.L_m)


# ------------------------------------------------------------ analyzer

def test_analyzer_factor():
    assert ph.energy_per_charge_factor(102.5, 97.5) == pytest.approx(19.99583, abs=1e-5)


@pytest.mark.parametrize("m_amu,z,U", [(92.906, 1, 0.5), (92.906, 7, 12.25), (26.98, 3, 91.0),
                                       (1.008, 1, 3.0)])
def test_tof_matches_untitled28_bit_for_bit(db, m_amu, z, U):
    prof = db.profile("legacy_untitled28")
    t_new = ph.tof_from_deflector(U, z, m_amu * prof.amu_kg, 3.85, 102.5, 97.5, prof.e_C)
    assert t_new == legacy.untitled28_TOF(m_amu, z, U, 3.85)


def test_tof_is_consistent_with_energy_per_charge(db):
    m, e = db.mass_kg("Nb"), db.e_C()
    U, z, L = 12.0, 5, 3.85
    t = ph.tof_from_deflector(U, z, m, L, 102.5, 97.5, e)
    assert ph.energy_per_charge_from_tof(t, z, L, m, e) == pytest.approx(
        ph.energy_per_charge_V(U, 102.5, 97.5), rel=1e-13)


# ------------------------------------------------------------ kinematics

def test_kinematics_round_trip_and_powerlaw_reference(db):
    m, e, L = db.mass_kg("Al", "legacy_amu_e1602"), 1.602e-19, 3.85
    t = np.geomspace(5e-6, 800e-6, 50)
    E = ph.kinetic_energy_eV(t, L, m, e)
    assert np.allclose(ph.tof_from_energy(E, L, m, e), t, rtol=1e-13)
    assert np.allclose(ph.tof_from_velocity(ph.velocity(t, L), L), t, rtol=1e-15)
    E_legacy = legacy.powerlaw_velocity_to_energy_keV(
        legacy.powerlaw_tof_to_velocity_kms(t * 1e6), m) * 1e3
    assert np.allclose(E, E_legacy, rtol=1e-13)


# ------------------------------------------------------------ jacobians (audit section 3)

def _bimodal(t):
    """Two components: slow at 100 us, fast at 50 us (velocity ratio r = 2)."""
    return (np.exp(-0.5 * ((t - 100e-6) / 4e-6) ** 2)
            + np.exp(-0.5 * ((t - 50e-6) / 2e-6) ** 2))


def test_jacobian_against_numerical_derivative(nb):
    t = np.geomspace(10e-6, 500e-6, 2001)
    E = ph.kinetic_energy_eV(t, nb["L"], nb["m"], nb["e"])
    assert np.allclose(ph.dt_dE(t, E), np.abs(np.gradient(t, E)), rtol=2e-3)
    assert np.allclose(ph.dt_dlnE(t), np.abs(np.gradient(t, np.log(E))), rtol=2e-3)
    assert np.allclose(ph.dt_du(t, nb["L"]),
                       np.abs(np.gradient(t, ph.velocity(t, nb["L"]))), rtol=2e-3)


def test_check_A_number_conserved_in_energy_domain(nb):
    # audit check A: integral f dE = integral N dt
    t = np.linspace(20e-6, 300e-6, 200001)
    n_t = _bimodal(t)
    E = ph.kinetic_energy_eV(t, nb["L"], nb["m"], nb["e"])
    n_E = n_t * ph.dt_dE(t, E)
    order = np.argsort(E)
    assert _trapz(n_E[order], E[order]) == pytest.approx(_trapz(n_t, t), rel=1e-6)
    n_lnE = n_t * ph.dt_dlnE(t)
    assert _trapz(n_lnE[order], np.log(E)[order]) == pytest.approx(_trapz(n_t, t), rel=1e-6)


@pytest.mark.parametrize("density", ["t", "lnE", "u", "E"])
def test_weight_table(density, nb):
    # the density is j(t) * t^k: the fast (t = 50 us) over slow (t = 100 us) weight is 2^-k
    t = np.array([50e-6, 100e-6])
    E = ph.kinetic_energy_eV(t, nb["L"], nb["m"], nb["e"])
    w = {"t": np.ones(2), "lnE": ph.dt_dlnE(t), "u": ph.dt_du(t, nb["L"]),
         "E": ph.dt_dE(t, E)}[density]
    assert w[0] / w[1] == pytest.approx(ph.relative_suppression(2.0, density), rel=1e-12)
    assert ph.relative_suppression(2.0, "E") == 1 / 8


def test_dNdE_peak_moves_to_lower_energy(nb):
    # audit section 3: the maximum of dN/dE sits at later t (lower E) than the maximum of j(t)
    t = np.linspace(60e-6, 160e-6, 100001)
    j = np.exp(-0.5 * ((t - 100e-6) / 10e-6) ** 2)
    E = ph.kinetic_energy_eV(t, nb["L"], nb["m"], nb["e"])
    assert t[np.argmax(j * ph.dt_dE(t, E))] > t[np.argmax(j)]


# ------------------------------------------------------------ plume (audit section 5, check 2)

def test_distance_scaling_matches_untitled27():
    L = 3.85
    t = np.linspace(5e-6, 400e-6, 100000)
    y = np.exp(-0.5 * ((t - 80e-6) / 9e-6) ** 2)
    I_1m, I_eia = legacy.untitled27_integrals(t, y, 2.0, 1.2732, 1 / 51, L)
    assert I_1m == pytest.approx(L ** 2 * I_eia, rel=1e-12)          # I_1m = L^2 I_EIA = 14.82 I
    # same with the library: the 1 m spectrum on the 1 m axis, projected to L
    t1, j1 = t / L, y * 2.0 * 1.2732 / 51
    tL, jL = ph.project_distance(t1, j1, 1.0, L)
    assert np.allclose(tL, t, rtol=1e-15)
    assert _trapz(jL, tL) == pytest.approx(I_eia, rel=1e-12)
    assert _trapz(j1, t1) * ph.integral_scale(1.0, L) == pytest.approx(I_eia, rel=1e-12)
    t_back, j_back = ph.project_distance(tL, jL, L, 1.0)
    assert np.allclose(t_back, t1) and np.allclose(j_back, j1)


# ------------------------------------------------------------ charge states (check C)

def test_zbar_matches_legacy_and_is_jacobian_invariant(nb):
    t = np.linspace(20e-6, 300e-6, 5000)
    rng = np.random.default_rng(1)
    J = {z: z * np.exp(-0.5 * ((t - 200e-6 / np.sqrt(z)) / 8e-6) ** 2)
         + 1e-3 * rng.normal(size=t.size) for z in range(1, 9)}
    zb = ph.average_charge(J)
    assert np.allclose(zb, legacy.powerlaw_zbar(J), rtol=1e-14, equal_nan=True)
    E = ph.kinetic_energy_eV(t, nb["L"], nb["m"], nb["e"])
    zb_jac = ph.average_charge({z: c * ph.dt_dE(t, E) for z, c in J.items()})
    ok = np.isfinite(zb)
    assert np.max(np.abs(zb_jac[ok] - zb[ok])) < 1e-12              # audit check 6: ~1e-15
    assert np.allclose(ph.average_charge({5: np.ones(3)}), 5.0)


def test_number_flux():
    assert ph.number_flux(np.array([3.204e-19, -1.0]), 2, 1.602e-19,
                          clip_negative=True).tolist() == [1.0, 0.0]


# ------------------------------------------------------------ CBS (audit section 4)

@pytest.mark.parametrize("kT,V0,z", [(30.0, 400.0, 3), (80.0, 2500.0, 8), (5.0, 50.0, 1)])
def test_cbs_shape_matches_fitter(nb, kT, V0, z):
    t = np.linspace(5e-6, 600e-6, 20001)
    new = ph.cbs_shape(t, kT, V0, z, nb["m"], nb["L"], nb["e"])
    old = legacy.fitter_cbs_shape(t, kT, V0, z, nb["m"], nb["L"])
    assert np.array_equal(new, old)


@pytest.mark.parametrize("kT,V0,z", [(30.0, 400.0, 3), (80.0, 2500.0, 8), (5.0, 50.0, 1)])
def test_cbs_peak_and_V0_round_trip(nb, kT, V0, z):
    m, L, e = nb["m"], nb["L"], nb["e"]
    u_pk = ph.cbs_peak_velocity(kT, V0, z, m, e)
    # analytic maximum agrees with the maximum of the shape on a fine grid
    t_pk = L / u_pk
    t = np.linspace(0.9 * t_pk, 1.1 * t_pk, 400001)
    assert t[np.argmax(ph.cbs_shape(t, kT, V0, z, m, L, e))] == pytest.approx(t_pk, rel=1e-6)
    # V0_from_peak inverts it (the fitter's peak constraint), and equals the fitter's function
    assert ph.V0_from_peak(kT, u_pk, z, m, e) == pytest.approx(V0, rel=1e-10)
    assert ph.V0_from_peak(kT, u_pk, z, m, e) == pytest.approx(
        legacy.fitter_V0_from_degen(kT, u_pk, z, m), rel=1e-15)


def test_V0_from_peak_unphysical_is_nan(nb):
    assert np.isnan(ph.V0_from_peak(1e4, 1e3, 1, nb["m"], nb["e"]))
    assert legacy.fitter_V0_from_degen(1e4, 1e3, 1, nb["m"]) is None


def test_half_maximum_edges_match_fitter(nb):
    tof = np.linspace(5e-6, 600e-6, 100000)
    for kT, V0, z in [(30.0, 400.0, 3), (80.0, 2500.0, 8)]:
        tL, tR = ph.half_maximum_edges(tof, ph.cbs_shape(tof, kT, V0, z, nb["m"], nb["L"],
                                                         nb["e"]))
        oL, oR = legacy.fitter_cbs_half_edges_us(kT, V0, z, tof, nb["m"], nb["L"])
        assert (tL * 1e6, tR * 1e6) == pytest.approx((oL, oR), rel=1e-12)
        assert ph.cbs_fwhm(tof, kT, V0, z, nb["m"], nb["L"], nb["e"]) == pytest.approx(
            (oR - oL) * 1e-6, rel=1e-10)


def test_recombination_marker():
    assert ph.recombination_tof(40e-6, 4, 1) * 1e6 == pytest.approx(
        legacy.fitter_recomb_tof_us(40.0, 4, 1), rel=1e-15)


def test_mass_fix_changes_kT_V0_not_times(db):
    # issue I1: with the same picked peak time and FWHM, kT and V0 scale with the mass
    # and every time / FWHM stays the same
    L, e = 3.85, 1.602e-19
    m_old, m_new = db.mass_kg("Al", "legacy_proton"), db.mass_kg("Al", "legacy_amu_e1602")
    s = m_new / m_old
    kT, V0, z = 40.0, 900.0, 4
    t = np.linspace(5e-6, 300e-6, 50001)
    y_old = ph.cbs_shape(t, kT, V0, z, m_old, L, e)
    y_new = ph.cbs_shape(t, kT * s, V0 * s, z, m_new, L, e)
    assert np.allclose(y_new / y_new.max(), y_old / y_old.max(), rtol=1e-9, atol=1e-12)
