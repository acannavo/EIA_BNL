"""
Verbatim copies of the legacy functions (only global constants turned into arguments),
used as references by the tests. Do not edit: they define what "legacy" means.
"""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz


# untitled28_EIA_Analysis_2.py -- TOF()
def untitled28_TOF(m, z, U, lof, Rout=102.5, Rin=97.5, mp=1.66054e-27, e=1.60217663e-19):
    R_orb = 0.1
    E = 2 * U / (R_orb * np.log(Rout / Rin))
    v = (z * e * R_orb * E / (m * mp)) ** 0.5
    t = lof / v
    return t


# cbs_manual_fitter_v12.py -- V0_from_degen(), cbs_shape(), cbs_half_edges_us(), recomb_tof_us()
def fitter_V0_from_degen(kT_eV, u_ms, z_ion, M_ION, E_CHARGE=1.602e-19, GAMMA=5.0 / 3.0):
    kT_J = kT_eV * E_CHARGE
    v_th = np.sqrt(GAMMA * kT_J / M_ION)
    v_s = u_ms - 5.0 * kT_J / (M_ION * u_ms)
    v_ac = v_s - v_th
    if v_ac <= 0:
        return None
    return M_ION * v_ac**2 / (2.0 * E_CHARGE * z_ion)


def fitter_cbs_shape(t_s, kT_eV, V0, z_ion, M_ION, L_EIA, E_CHARGE=1.602e-19, GAMMA=5.0 / 3.0):
    kT_J = kT_eV * E_CHARGE
    v_th = np.sqrt(GAMMA * kT_J / M_ION)
    v_ac = np.sqrt(2.0 * E_CHARGE * V0 * z_ion / M_ION)
    v_s = v_th + v_ac
    pre = (M_ION / (2.0 * np.pi * kT_J))**1.5 * L_EIA**4 * t_s**(-5)
    expo = np.exp(np.clip(-(M_ION / (2.0 * kT_J)) * (L_EIA / t_s - v_s)**2,
                          -708, 0))
    return pre * expo


def fitter_cbs_half_edges_us(kT_eV, V0, z_ion, tof_s, M_ION, L_EIA, E_CHARGE=1.602e-19):
    y = fitter_cbs_shape(tof_s, kT_eV, V0, z_ion, M_ION, L_EIA, E_CHARGE)
    tus = tof_s * 1e6
    pk = np.argmax(y)
    half = y[pk] / 2.0
    lc = np.where(np.diff(np.sign(y[:pk] - half)))[0]
    rc = np.where(np.diff(np.sign(y[pk:] - half)))[0]
    if len(lc) == 0 or len(rc) == 0:
        return np.nan, np.nan
    i = lc[-1]
    tL = tus[i] + (half - y[i]) / (y[i+1] - y[i]) * (tus[i+1] - tus[i])
    j = pk + rc[0]
    tR = tus[j] + (half - y[j]) / (y[j+1] - y[j]) * (tus[j+1] - tus[j])
    return tL, tR


def fitter_recomb_tof_us(t_peak_us, z_ion, dz):
    if t_peak_us is None or t_peak_us <= 0 or dz <= 0:
        return None
    return t_peak_us * np.sqrt(z_ion / (z_ion + dz))


# plot_eia_powerLaw_test.py -- conversions and compute_zbar() core
def powerlaw_velocity_to_energy_keV(u_kms, m_ion, E_CHARGE=1.602e-19):
    u = np.asarray(u_kms) * 1e3
    return 0.5 * m_ion * u**2 / E_CHARGE / 1e3


def powerlaw_tof_to_velocity_kms(t_us, L_EIA=3.85):
    return L_EIA / (np.asarray(t_us) * 1e-6) / 1e3


def powerlaw_zbar(z_curves, E_CHARGE=1.602e-19):
    Nz = {}
    for z, c in z_curves.items():
        J = np.clip(c, 0.0, None)
        Nz[z] = J / (z * E_CHARGE)
    Ntot = sum(Nz.values())
    Nz_w = sum(z * Nz[z] for z in Nz)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(Ntot > 0, Nz_w / Ntot, np.nan)


# untitled27_REFACTORED_4.py -- projection chain and integrals (Sections 7, 8, 8b)
def untitled27_integrals(tof_common, yield_interp, CFY_1m, CFArea, CFCurrent, L_EIA):
    tof_1m = tof_common * (1.0 / L_EIA)
    cy = yield_interp * CFY_1m * CFArea * CFCurrent
    I_1m = _trapz(cy, tof_1m)
    I_eia = _trapz(cy / L_EIA**3, tof_common)
    return I_1m, I_eia
