"""Figures of plot_collapse_v2: collapse u2/u1 vs z, peak check on the J_z(t) curves."""
from __future__ import annotations

import numpy as np

from .spec import ErrorBars, Figure, Line, Points, lab
from .style import energy_color, z_color

EL_MARKER = {"Al": "o", "Nb": "s"}
LIT_STYLE = {"poirier_2023_sn": ("#2255a4", "D"), "kondrashev_2006_nb": ("#e07b1a", "^")}


def collapse(series: list, joint: dict, literature: list, title: str) -> Figure:
    fig = Figure(rows=1, cols=1, size_in=(5.4, 6.4), height_px=640, legend="figure",
                 title=title)
    p = fig.panel(0, 0)
    p.xlabel = lab(r"Charge state  $z$", "Charge state z")
    p.ylabel = lab(r"$u_2/u_1$", "u2 / u1")
    zmax = 1.0
    for s in series:
        c = energy_color(s["energy_mJ"])
        name = (f"{s['element']}, {s['energy_mJ']:g} mJ (z* = {s['zstar']:.1f}, "
                f"R² = {s['R2']:.2f})")
        p.add(ErrorBars(s["z"], s["ratio"], name, c, ylo=s["ratio"] - s["lo"],
                        yhi=s["ratio"] + s["hi"], marker=EL_MARKER.get(s["element"], "o"),
                        size=4.5, group=s["run"]))
        zz = np.linspace(0, s["z"].max() * 1.05, 50)
        p.add(Line(zz, s["s"] * zz, None, c, width=0.8, dash="dash", group=s["run"]))
        zmax = max(zmax, float(s["z"].max()))
    for L in literature:
        col, mk = LIT_STYLE.get(L["key"], ("#555555", "v"))
        p.add(Points(L["z"], L["ratio"], f"{L['label']} (z* = {L['zstar']:.1f})", col,
                     marker=mk, size=5, filled=False, group=L["key"]))
        zmax = max(zmax, float(L["z"].max()))
    if joint.get("n"):
        zz = np.linspace(0, zmax * 1.05, 50)
        p.add(Line(zz, zz / joint["zstar"],
                   f"joint z* = {joint['zstar']:.2f} ± {joint['se']:.2f}", "black", width=1.6))
    p.xlim = (0, zmax * 1.08)
    return fig


def peak_check(datasets: list, peaks, height: float = 2.5, xlim=(5, 850)) -> Figure:
    """Stacked J_z(t)^(1/2), normalised per z and offset by z, with the peaks used."""
    order = sorted(datasets, key=lambda d: (d.species, d.energy_mJ or 0))
    ncol = 3
    rows = int(np.ceil(len(order) / ncol))
    fig = Figure(rows=rows, cols=ncol, size_in=(15, 6 * rows), height_px=650 * rows,
                 legend="figure", letters=True)
    mk = {"fast": ("#e6194b", "o"), "slow": ("#3cb44b", "s"), "ultra_fast": ("#4363d8", "^")}
    for k, ds in enumerate(order):
        p = fig.panel(k // ncol, k % ncol)
        p.title, p.xlog = ds.title, True
        p.xlabel = lab(r"Time of flight  $t$  [$\mu$s]", "Time of flight t [µs]")
        p.ylabel = lab(r"$z$ + $\sqrt{J_z/J_{z,max}}$ (offset)", "z + sqrt(J_z / max)")
        p.xlim = xlim
        sub = peaks[peaks.run == ds.key]
        for z in ds.all_z:
            y = np.clip(ds.z_curves[z], 0, None)
            if y.max() <= 0:
                continue
            yn = np.sqrt(y / y.max()) * height / 2.5
            p.add(Line(ds.tof_us, z + yn, None, z_color(z), width=0.8, alpha=0.8))
            for comp, (col, m) in mk.items():
                r = sub[(sub.z == z) & (sub.component == comp)]
                if len(r):
                    i = np.searchsorted(ds.tof_us, r.t_peak.to_numpy())
                    i = np.clip(i, 0, len(ds.tof_us) - 1)
                    p.add(Points(r.t_peak.to_numpy(), z + yn[i], comp.replace("_", "-"), col,
                                 marker=m, size=4, group=comp))
    return fig
