"""Study figures of plot_eia_powerLaw_test.py: zbar grid, zbar all, Sheil grid, spectra grids."""
from __future__ import annotations

import numpy as np

from ..analysis import energy_eV, velocity_kms
from .spec import Figure, Line, lab
from .style import ELEMENT_DASH, energy_color, z_color

ZBAR = lab(r"Average charge state  $\bar z$", "Average charge state z̄")
E_EV = lab(r"Kinetic energy  $E$  [eV]", "Kinetic energy E [eV]")
SHEIL_Y = lab(r"$d^2N/dE\,d\Omega$  (/eV/sr/shot)", "d²N/dE dΩ (/eV/sr/shot)")
REF = (0.3, 0.4)                                   # Sheil et al., PRL 133, 125101 (Sn)


def _grid(records, elements=("Al", "Nb"), energies=(180, 370, 780)):
    look = {(r["ds"].species, int(round(r["ds"].energy_mJ or 0))): r for r in records}
    return [[look.get((el, mj)) for mj in energies] for el in elements], elements, energies


def zbar_grid(records, ref=REF) -> Figure:
    """2 x 3: zbar(E) data, power-law fit, Sheil reference (rows Al / Nb)."""
    g, els, ens = _grid(records)
    fig = Figure(rows=2, cols=3, size_in=(16, 9), sharex=True, sharey=True, letters=True,
                 title=lab(r"Average charge state $\bar z$ vs kinetic energy",
                           "Average charge state z̄ vs kinetic energy"), legend="panel")
    vr = [r["zb"]["E_eV"][r["zb"]["valid"]] for r in records if r["zb"]["valid"].any()]
    Er = np.logspace(np.log10(min(v.min() for v in vr)), np.log10(max(v.max() for v in vr)),
                     100)
    for i, el in enumerate(els):
        for j, mj in enumerate(ens):
            p = fig.panel(i, j)
            p.xlog = p.ylog = True
            p.xlabel, p.ylabel = E_EV, ZBAR
            rec = g[i][j]
            if rec is None:
                p.empty_text = f"{el}, {mj} mJ: not available"
                continue
            zb, fit = rec["zb"], rec["fit"]
            v = zb["valid"]
            p.title = rec["ds"].title
            p.add(Line(zb["E_eV"][v], zb["zbar"][v], lab(r"$\bar z(E)$ (data)", "z̄(E) data"),
                       energy_color(mj), width=0.8))
            if fit is not None:
                Ef = np.logspace(np.log10(fit["E_lo"]), np.log10(fit["E_hi"]), 100)
                p.add(Line(Ef, fit["C"] * Ef ** fit["b"],
                           lab(rf"fit: ${fit['C']:.3g}\,E^{{{fit['b']:.2f}}}$  "
                               rf"($n={fit['n']:.2f}$)",
                               f"fit: {fit['C']:.3g} E^{fit['b']:.2f} (n = {fit['n']:.2f})"),
                           "red", width=2))
            p.add(Line(Er, ref[0] * Er ** ref[1],
                       lab(rf"Sheil et al. (Sn): ${ref[0]}\,E^{{{ref[1]}}}$",
                           f"Sheil et al. (Sn): {ref[0]} E^{ref[1]}"), "black", dash="dash"))
    return fig


def zbar_all(records, ref_line: dict) -> Figure:
    """All conditions in one panel, colour = laser energy, line style = element."""
    fig = Figure(rows=1, cols=1, size_in=(6.4, 6.2), legend="figure")
    p = fig.panel(0, 0)
    p.xlog = p.ylog = True
    p.xlabel, p.ylabel = E_EV, ZBAR
    lo, hi = np.inf, 0.0
    for r in sorted(records, key=lambda r: (r["ds"].species, r["ds"].energy_mJ or 0)):
        zb = r["zb"]
        v = zb["valid"]
        if not v.any():
            continue
        E = zb["E_eV"][v]
        lo, hi = min(lo, E.min()), max(hi, E.max())
        p.add(Line(E, zb["zbar"][v], r["ds"].title, energy_color(r["ds"].energy_mJ),
                   dash=ELEMENT_DASH.get(r["ds"].species, "solid"), width=1.0, alpha=0.85))
    Er = np.logspace(np.log10(lo), np.log10(hi), 100)
    C, b = ref_line["C"], ref_line["exponent"]
    p.add(Line(Er, C * Er ** b, lab(rf"Sheil-type: ${C:.2f}\,E^{{{b}}}$",
                                    f"Sheil-type: {C:.2f} E^{b}"), "black", dash="dash",
               width=1.4))
    fig.notes.append(f"Reference line: exponent {b} kept (Sheil et al., Sn), prefactor "
                     f"fitted to all conditions inside their fit windows "
                     f"({ref_line['npts']} points): C = {C:.4g}.")
    return fig


def sheil_grid(records, ylim=(1e5, 1e12)) -> Figure:
    """2 x 3 in the style of Sheil et al. Fig. 1b: d2N/dE dOmega per z, total (dashed),
    zbar (red, right axis)."""
    g, els, ens = _grid(records)
    fig = Figure(rows=2, cols=3, size_in=(17, 10), letters=True, legend="figure")
    for i, el in enumerate(els):
        for j, mj in enumerate(ens):
            p = fig.panel(i, j)
            p.xlog = p.ylog = p.y2log = True
            p.xlabel, p.ylabel = lab(r"Ion energy  $E$  (eV)", "Ion energy E (eV)"), SHEIL_Y
            p.y2label = ZBAR if j == len(ens) - 1 else None
            rec = g[i][j]
            if rec is None:
                p.empty_text = f"{el}, {mj} mJ: not available"
                continue
            S, zb = rec["S"], rec["zb"]
            p.title = rec["ds"].title
            E = S["E"]
            for z in sorted(S["f"]):
                f = S["f"][z]
                if f.max() <= 0:
                    continue
                p.add(Line(E, np.where(f > 0, f, np.nan), f"z = {z}+", z_color(z), width=1.5,
                           alpha=0.55))
            tot = S["total"]
            p.add(Line(E, np.where(tot > 0, tot, np.nan),
                       lab(r"total  $d^2N/dE\,d\Omega$", "total d²N/dE dΩ"), "black",
                       dash="dash", width=1.5))
            v = zb["valid"]
            p.add(Line(zb["E_eV"][v], zb["zbar"][v], lab(r"$\bar z(E)$", "z̄(E)"), "red",
                       width=2.2, y2=True))
            p.ylim = ylim
            sel = tot > ylim[0]
            if sel.any():
                p.xlim = (E[sel].min() / 1.3, E[sel].max() * 1.3)
            zmax = max(S["f"]) if S["f"] else 1
            p.y2lim = (1, max(10.0, 1.25 * max(zmax, float(np.nanmax(zb["zbar"][v])))))
    return fig


def spectra_grid(records, domain: str = "tof") -> Figure:
    """2 x 3 per-charge-state spectra. domain: 'tof' (J_z vs t), 'velocity' (J_z vs u),
    'dlnE' ((J_z / z) t / 2 vs E: area = ion number), 'energy' ((J_z / z) |dt/dE| vs E)."""
    g, els, ens = _grid(records)
    lbl = {"tof": (lab(r"Time of flight  $t$  [$\mu$s]", "Time of flight t [µs]"),
                   lab(r"$J_z$  [A cm$^{-2}$]", "J_z [A cm⁻²]"), False),
           "velocity": (lab(r"Velocity  $u$  [km/s]", "Velocity u [km/s]"),
                        lab(r"$J_z$  [A cm$^{-2}$]", "J_z [A cm⁻²]"), False),
           "dlnE": (E_EV, lab(r"$dN/d\ln E$  [ions cm$^{-2}$]", "dN/dlnE [ions cm⁻²]"), True),
           "energy": (E_EV, lab(r"$dN/dE$  [ions cm$^{-2}$ eV$^{-1}$]",
                                "dN/dE [ions cm⁻² eV⁻¹]"), True)}[domain]
    fig = Figure(rows=2, cols=3, size_in=(16, 9), letters=True, legend="figure")
    for i, el in enumerate(els):
        for j, mj in enumerate(ens):
            p = fig.panel(i, j)
            p.xlabel, p.ylabel = lbl[0], lbl[1]
            p.ylog = True
            p.xlog = lbl[2]
            rec = g[i][j]
            if rec is None:
                p.empty_text = f"{el}, {mj} mJ: not available"
                continue
            ds = rec["ds"]
            p.title = ds.title
            t = ds.tof_us
            E = energy_eV(ds, t)
            peak = 0.0
            for z in ds.all_z:
                J = ds.z_curves[z]
                if domain == "tof":
                    x, y = t, J
                elif domain == "velocity":
                    x, y = velocity_kms(ds, t), J
                elif domain == "dlnE":
                    x, y = E, J / (z * ds.e_C) * (t * 1e-6) / 2
                else:
                    x, y = E, J / (z * ds.e_C) * (t * 1e-6) / (2 * E)
                y = np.where(y > 0, y, np.nan)
                peak = max(peak, float(np.nanmax(y)) if np.isfinite(y).any() else 0)
                p.add(Line(x, y, f"z = {z}+", z_color(z), width=1.0, alpha=0.6))
            if peak > 0:
                p.ylim = (peak * 1e-5, peak * 2)
    return fig
