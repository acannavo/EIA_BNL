"""Figures of the per-run HTML report."""
from __future__ import annotations

import numpy as np

from .powerlaw import E_EV, SHEIL_Y, ZBAR
from .spec import Band, Figure, Line, Points, lab
from .style import COMPONENT_COLORS, COMPONENT_NAMES, z_color

T_US = lab(r"Time of flight  $t$  [$\mu$s]", "Time of flight t [µs]")


def spectra_overview(ds) -> Figure:
    fig = Figure(rows=1, cols=1, size_in=(8, 5), height_px=520, legend="figure",
                 title=f"{ds.title}: calibrated spectra J_z(t) at L_EIA = {ds.L_m} m")
    p = fig.panel(0, 0)
    p.xlabel, p.ylabel, p.ylog = T_US, lab(r"$J_z$  [A cm$^{-2}$]", "J_z [A cm⁻²]"), True
    peak = 0.0
    for z in ds.all_z:
        y = np.where(ds.z_curves[z] > 0, ds.z_curves[z], np.nan)
        if np.isfinite(y).any():
            peak = max(peak, float(np.nanmax(y)))
        p.add(Line(ds.tof_us, y, f"z = {z}+", z_color(z), width=1.0))
    if peak:
        p.ylim = (peak * 1e-5, peak * 2)
    return fig


def cbs_panels(rm: dict, comp_names: dict, cols: int = 4) -> Figure:
    """One panel per charge state: measured points, calibrated curve, every stored CBS
    component (legacy parameters) and their sum, fit windows shaded."""
    model, curves, exp, comps = rm["model"], rm["curves"], rm["exp"], rm["components"]
    zs = sorted(z for z in curves if z in comps)
    rows = int(np.ceil(len(zs) / cols)) or 1
    unit = "per ion" if rm["per_ion"] else "charge"
    fig = Figure(rows=rows, cols=cols, size_in=(4 * cols, 2.8 * rows),
                 height_px=270 * rows + 120, legend="figure",
                 title=f"CBS components on the current calibration ({unit} signal)")
    tus = model.tus
    for k, z in enumerate(zs):
        p = fig.panel(k // cols, k % cols)
        p.title = f"z = {z}+"
        p.xlabel, p.ylabel = T_US, (lab(r"$J_z/z$", "J_z / z") if rm["per_ion"]
                                    else lab(r"$J_z$", "J_z"))
        cs = comps[z]
        lo = min(c.win_left for c in cs if c.win_left is not None) if cs else tus.min()
        hi = max(c.win_right for c in cs if c.win_right is not None) if cs else tus.max()
        sel = (tus >= lo / 1.4) & (tus <= hi * 1.3)
        y = curves[z]
        if z in exp:
            te, ye = exp[z]
            m = (te >= lo / 1.4) & (te <= hi * 1.3)
            p.add(Points(te[m], ye[m], "measured points", "#9a9a9a", size=2.5))
        p.add(Line(tus[sel], y[sel], "calibrated curve", "black", width=1.0))
        tot = np.zeros(sel.sum())
        for c in cs:
            if not c.active:
                continue
            name = comp_names.get(c.extra.get("cbs"), f"CBS {c.extra.get('cbs')}")
            col = COMPONENT_COLORS.get(name, "#7f7f7f")
            f = c.A * model.shape(c.kT, c.V0, z)[sel]
            tot += f
            p.add(Line(tus[sel], f, COMPONENT_NAMES.get(name, name), col, width=1.4,
                       group=name))
            wl, wr = model.edges(c)
            ymax = float(np.nanmax(y[sel])) if sel.any() else 1.0
            p.add(Band(np.array([wl, wr]), np.zeros(2), name=None, color=col, group=name,
                       alpha=0.08, y_upper=np.full(2, ymax)))
        if len(cs) > 1:
            p.add(Line(tus[sel], tot, "sum of components", "#e377c2", width=1.2, dash="dash"))
    return fig


def zbar_single(ds, zb: dict, fit: dict | None) -> Figure:
    fig = Figure(rows=1, cols=1, size_in=(6.4, 4.8), height_px=480, legend="panel",
                 title=f"{ds.title}: average charge state")
    p = fig.panel(0, 0)
    p.xlog = p.ylog = True
    p.xlabel, p.ylabel = E_EV, ZBAR
    v = zb["valid"]
    p.add(Line(zb["E_eV"][v], zb["zbar"][v], "z̄(E) data", "#1f77b4", width=1.0))
    if fit:
        Ef = np.logspace(np.log10(fit["E_lo"]), np.log10(fit["E_hi"]), 100)
        p.add(Line(Ef, fit["C"] * Ef ** fit["b"],
                   f"fit {fit['C']:.3g} E^{fit['b']:.3f} (n = {fit['n']:.2f}, "
                   f"R² = {fit['r2']:.3f})", "red", width=2))
    return fig


def sheil_single(ds, S: dict, zb: dict) -> Figure:
    fig = Figure(rows=1, cols=1, size_in=(7, 5), height_px=520, legend="figure",
                 title=f"{ds.title}: d²N/dE dΩ per charge state")
    p = fig.panel(0, 0)
    p.xlog = p.ylog = p.y2log = True
    p.xlabel, p.ylabel, p.y2label = E_EV, SHEIL_Y, ZBAR
    for z in sorted(S["f"]):
        f = S["f"][z]
        if f.max() > 0:
            p.add(Line(S["E"], np.where(f > 0, f, np.nan), f"z = {z}+", z_color(z), width=1.3))
    tot = S["total"]
    p.add(Line(S["E"], np.where(tot > 0, tot, np.nan), "total", "black", dash="dash",
               width=1.5))
    v = zb["valid"]
    p.add(Line(zb["E_eV"][v], zb["zbar"][v], "z̄(E)", "red", width=2.2, y2=True))
    if tot.max() > 0:
        p.ylim = (tot.max() * 1e-6, tot.max() * 3)
    return fig
