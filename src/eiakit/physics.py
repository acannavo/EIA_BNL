"""
Physics of the EIA analysis: one implementation of every formula used by the
legacy scripts. Pure functions on numpy arrays, SI units, no global constants:
the mass m_kg, the charge e_C and the geometry come from the database
(``db.mass_kg(species, profile)``, ``db.e_C(profile)``, ``run.eia.L_m``), so the
same code reproduces legacy results (legacy profile) and corrected ones.

Units: t [s], L [m], u [m/s], energies [eV], potentials [V], kT [eV], V0 [V].

Sections
    analyzer        E/q = U / ln(R_out/R_in); time of flight of (m, z) at deflector voltage U
    kinematics      t <-> u <-> E <-> E/q at fixed flight path
    jacobians       |dt/dE|, |dt/dlnE|, |dt/du|; weight of each density (t^k)
    plume           distance scaling j(L, t) = L^-3 j(1 m, t / L)
    charge states   number flux N_z = J_z / (z e); average charge z_bar(t)
    CBS             shifted-Maxwellian (Centre-of-mass Boltzmann Shift) time-of-flight shape
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

GAMMA_DEFAULT = 5.0 / 3.0


# ============================================================ analyzer

def energy_per_charge_factor(R_out: float, R_in: float) -> float:
    """E/q [V] per volt of deflector voltage: 1 / ln(R_out / R_in)  (19.996 for 102.5/97.5)."""
    return 1.0 / np.log(R_out / R_in)


def energy_per_charge_V(U, R_out: float, R_in: float):
    """Kinetic energy per charge [V] selected by the analyzer at deflector voltage U [V]."""
    return np.asarray(U, dtype=float) * energy_per_charge_factor(R_out, R_in)


def tof_from_deflector(U, z, m_kg: float, L_m: float, R_out: float, R_in: float,
                       e_C: float, R_orbit_m: float = 0.1):
    """
    Time of flight [s] of an ion (m, z) passed by the analyzer at deflector voltage U.
    Same operations, in the same order, as TOF() in untitled28_EIA_Analysis_2:
        E_field = 2 U / (R_orb ln(R_out/R_in)),   v = sqrt(z e R_orb E_field / m),   t = L / v
    (R_orbit cancels: (1/2) m v^2 = z e U / ln(R_out/R_in)).
    """
    E_field = 2 * np.asarray(U, dtype=float) / (R_orbit_m * np.log(R_out / R_in))
    v = (np.asarray(z) * e_C * R_orbit_m * E_field / m_kg) ** 0.5
    return L_m / v


# ============================================================ kinematics

def velocity(t_s, L_m: float):
    """u = L / t  [m/s]."""
    return L_m / np.asarray(t_s, dtype=float)


def tof_from_velocity(u_ms, L_m: float):
    return L_m / np.asarray(u_ms, dtype=float)


def kinetic_energy_eV(t_s, L_m: float, m_kg: float, e_C: float):
    """E = (1/2) m (L/t)^2 / e  [eV]; the same for every charge state at a given t."""
    u = velocity(t_s, L_m)
    return 0.5 * m_kg * u ** 2 / e_C


def tof_from_energy(E_eV, L_m: float, m_kg: float, e_C: float):
    return L_m / np.sqrt(2.0 * np.asarray(E_eV, dtype=float) * e_C / m_kg)


def energy_per_charge_from_tof(t_s, z, L_m: float, m_kg: float, e_C: float):
    """E/q [V] = E [eV] / z."""
    return kinetic_energy_eV(t_s, L_m, m_kg, e_C) / np.asarray(z, dtype=float)


# ============================================================ jacobians

#: exponent k of the time weight t^k that each density applies to j(t):
#: a fast component at velocity ratio r = u_fast/u_slow is suppressed by r^-k
DENSITY_WEIGHT_EXPONENT = {"t": 0, "lnE": 1, "u": 2, "E": 3}


def dt_dE(t_s, E_eV):
    """|dt/dE| = t / (2E)  [s/eV]   (E proportional to t^-2)."""
    return np.asarray(t_s, dtype=float) / (2.0 * np.asarray(E_eV, dtype=float))


def dt_dlnE(t_s):
    """|dt/dlnE| = E |dt/dE| = t / 2  [s]."""
    return np.asarray(t_s, dtype=float) / 2.0


def dt_du(t_s, L_m: float):
    """|dt/du| = t^2 / L  [s^2/m]."""
    return np.asarray(t_s, dtype=float) ** 2 / L_m


def relative_suppression(r, density: str):
    """Factor by which a fast component (velocity ratio r > 1) is reduced, relative to the
    slow one, when j(t) is plotted as the given density ('t', 'lnE', 'u', 'E')."""
    return np.asarray(r, dtype=float) ** (-DENSITY_WEIGHT_EXPONENT[density])


# ============================================================ plume

def project_distance(t_s, j, L_from_m: float, L_to_m: float):
    """
    Plume law j(L, t) = L^-3 j(1 m, t/L): the signal measured at L_from, expressed at L_to.
    Returns (t at L_to, j at L_to). Time integrals scale by (L_from/L_to)^2.
    """
    k = L_to_m / L_from_m
    return np.asarray(t_s, dtype=float) * k, np.asarray(j, dtype=float) / k ** 3


def integral_scale(L_from_m: float, L_to_m: float) -> float:
    """Factor of a time integral (C/cm^2, ions/cm^2) moved from L_from to L_to: (L_from/L_to)^2."""
    return (L_from_m / L_to_m) ** 2


# ============================================================ charge states

def number_flux(J, z, e_C: float, clip_negative: bool = False):
    """N_z = J_z / (z e): charge current density [A/cm^2] -> ions cm^-2 s^-1."""
    J = np.asarray(J, dtype=float)
    if clip_negative:
        J = np.clip(J, 0.0, None)
    return J / (np.asarray(z, dtype=float) * e_C)


def average_charge(J_by_z: Mapping[int, np.ndarray], clip_negative: bool = True):
    """
    z_bar(t) = sum_z z N_z / sum_z N_z = sum_z J_z / sum_z (J_z / z), with J_z the charge
    current density of each charge state on a common time grid. NaN where sum N_z <= 0.
    Any factor common to all z at fixed t (e.g. a Jacobian) cancels.
    """
    num = den = 0.0
    for z, J in J_by_z.items():
        J = np.asarray(J, dtype=float)
        if clip_negative:
            J = np.clip(J, 0.0, None)
        num = num + J
        den = den + J / z
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ============================================================ CBS

def thermal_velocity(kT_eV, m_kg: float, e_C: float, gamma: float = GAMMA_DEFAULT):
    """v_th = sqrt(gamma kT / m)  [m/s]."""
    return np.sqrt(gamma * np.asarray(kT_eV, dtype=float) * e_C / m_kg)


def acceleration_velocity(V0, z, m_kg: float, e_C: float):
    """v_ac = sqrt(2 e V0 z / m)  [m/s]."""
    return np.sqrt(2.0 * e_C * np.asarray(V0, dtype=float) * np.asarray(z) / m_kg)


def shift_velocity(kT_eV, V0, z, m_kg: float, e_C: float, gamma: float = GAMMA_DEFAULT):
    """v_s = v_th + v_ac  [m/s]."""
    return thermal_velocity(kT_eV, m_kg, e_C, gamma) + acceleration_velocity(V0, z, m_kg, e_C)


def cbs_shape(t_s, kT_eV: float, V0: float, z, m_kg: float, L_m: float, e_C: float,
              gamma: float = GAMMA_DEFAULT):
    """
    Unit-amplitude CBS time-of-flight shape (number flux per unit time); the fitted signal
    is A * cbs_shape. Same expression as cbs_manual_fitter_v12:
        F = (m / 2 pi kT)^1.5 L^4 t^-5 exp(-(m / 2kT) (L/t - v_s)^2)
    i.e. a Maxwellian shifted by v_s, f(v) v^3 |dv/dt| with v = L/t.
    """
    t_s = np.asarray(t_s, dtype=float)
    kT_J = kT_eV * e_C
    v_s = shift_velocity(kT_eV, V0, z, m_kg, e_C, gamma)
    pre = (m_kg / (2.0 * np.pi * kT_J)) ** 1.5 * L_m ** 4 * t_s ** (-5)
    expo = np.exp(np.clip(-(m_kg / (2.0 * kT_J)) * (L_m / t_s - v_s) ** 2, -708, 0))
    return pre * expo


def cbs_peak_velocity(kT_eV, V0, z, m_kg: float, e_C: float, gamma: float = GAMMA_DEFAULT):
    """
    Velocity at the maximum of cbs_shape in time: d/dv [v^5 exp(-m (v - v_s)^2 / 2kT)] = 0
    gives  m u^2 - m v_s u - 5 kT = 0,  u = (v_s + sqrt(v_s^2 + 20 kT/m)) / 2.
    """
    v_s = shift_velocity(kT_eV, V0, z, m_kg, e_C, gamma)
    kT_J = np.asarray(kT_eV, dtype=float) * e_C
    return 0.5 * (v_s + np.sqrt(v_s ** 2 + 20.0 * kT_J / m_kg))


def cbs_peak_tof(kT_eV, V0, z, m_kg: float, L_m: float, e_C: float,
                 gamma: float = GAMMA_DEFAULT):
    return L_m / cbs_peak_velocity(kT_eV, V0, z, m_kg, e_C, gamma)


def V0_from_peak(kT_eV, u_peak_ms, z, m_kg: float, e_C: float, gamma: float = GAMMA_DEFAULT):
    """
    Accelerating potential V0 [V] that puts the CBS maximum at velocity u_peak for a given kT
    (the 'degeneracy' constraint of the fitter, V0_from_degen):
        v_s = u - 5kT/(m u),  v_ac = v_s - v_th,  V0 = m v_ac^2 / (2 e z).
    NaN where v_ac <= 0 (no physical V0; the fitter returned None).
    """
    kT_J = np.asarray(kT_eV, dtype=float) * e_C
    u = np.asarray(u_peak_ms, dtype=float)
    v_s = u - 5.0 * kT_J / (m_kg * u)
    v_ac = v_s - thermal_velocity(kT_eV, m_kg, e_C, gamma)
    V0 = m_kg * v_ac ** 2 / (2.0 * e_C * np.asarray(z, dtype=float))
    return np.where(v_ac > 0, V0, np.nan)


def half_maximum_edges(t_s, y):
    """
    (t_left, t_right) where y crosses half its maximum, by linear interpolation on the grid
    (same algorithm as cbs_half_edges_us of the fitter, in seconds). (nan, nan) if a side
    never crosses.
    """
    t_s, y = np.asarray(t_s, dtype=float), np.asarray(y, dtype=float)
    pk = int(np.argmax(y))
    half = y[pk] / 2.0
    lc = np.where(np.diff(np.sign(y[:pk] - half)))[0]
    rc = np.where(np.diff(np.sign(y[pk:] - half)))[0]
    if len(lc) == 0 or len(rc) == 0:
        return np.nan, np.nan
    i = lc[-1]
    tL = t_s[i] + (half - y[i]) / (y[i + 1] - y[i]) * (t_s[i + 1] - t_s[i])
    j = pk + rc[0]
    tR = t_s[j] + (half - y[j]) / (y[j + 1] - y[j]) * (t_s[j + 1] - t_s[j])
    return float(tL), float(tR)


def cbs_fwhm(t_s, kT_eV, V0, z, m_kg: float, L_m: float, e_C: float,
             gamma: float = GAMMA_DEFAULT) -> float:
    """FWHM [s] of the CBS shape evaluated on the grid t_s."""
    tL, tR = half_maximum_edges(t_s, cbs_shape(t_s, kT_eV, V0, z, m_kg, L_m, e_C, gamma))
    return tR - tL


def recombination_tof(t_peak_s, z, dz):
    """Recombination marker drawn by the CBS fitter: t_peak sqrt(z / (z + dz))."""
    return np.asarray(t_peak_s, dtype=float) * np.sqrt(z / (z + np.asarray(dz, dtype=float)))
