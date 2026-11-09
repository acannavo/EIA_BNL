"""Study figures of plot_cbs_multipanel_v2: velocity, picks, area_fwhm, fit_summary."""
from __future__ import annotations

import numpy as np

from .spec import ErrorBars, Figure, Points, Text, lab
from .style import COMPONENT_NAMES

# legacy colours / markers (plot_cbs_multipanel_v2: 1 fast, 2 slow, 3 ultra-fast, 4)
COLORS = {"fast": "#e6194b", "slow": "#3cb44b", "ultra_fast": "#4363d8",
          "component_4": "#f58231"}
MARKERS = {"fast": "o", "slow": "s", "ultra_fast": "^", "component_4": "D"}
ORDER = ["fast", "slow", "ultra_fast", "component_4"]
U_LABEL = lab(r"Peak velocity  $u$  (km/s)", "Peak velocity u (km/s)")
Z_LABEL = lab(r"Charge state  $z$", "Charge state z")


def _bounds(u, lo, hi):
    """errorbar_bounds of the legacy script: neighbours as min / max, missing -> u."""
    lo = np.where(np.isfinite(lo), lo, u)
    hi = np.where(np.isfinite(hi), hi, u)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def _mean_lines(p, g, comps, col, unit="km/s", fmt=".0f"):
    info = []
    for c in comps:
        v = g.loc[g.component == c, col].to_numpy(float)
        if len(v):
            info.append((c, float(np.nanmean(v)), float(np.nanstd(v))))
    info.sort(key=lambda t: -t[1])
    for i, (c, m, s) in enumerate(info):
        p.hlines.append((m, COLORS.get(c, "gray"), "dot"))
        p.texts.append(Text(0.95, 0.95 - i * 0.115, f"{m:{fmt}} ± {s:{fmt}} {unit}",
                            size=8, color=COLORS.get(c, "gray"), ha="right"))
    if info:
        p.headroom = 0.12 + 0.12 * len(info)


def velocity(tab, energy_mJ, avg_components: dict, kind: str = "velocity") -> Figure:
    """kind 'velocity': u of the PEAK sample, bars to the neighbour samples;
    'picks': picked peak, bars to the picked half-maximum points."""
    ucol, lo, hi = (("u_peak", "u_prev", "u_next") if kind == "velocity"
                    else ("u_pick", "u_right", "u_left"))
    t = tab[(tab.energy_mJ == energy_mJ) & ~tab.excluded]
    fig = Figure(rows=1, cols=2, size_in=(4.7, 2.65), height_px=470, letters=True,
                 legend="figure", title=f"{energy_mJ:g} mJ" + (" (picked peaks)"
                                                               if kind == "picks" else ""))
    for j, el in enumerate(["Al", "Nb"]):
        p = fig.panel(0, j)
        p.title, p.xlabel = el, Z_LABEL
        p.ylabel = U_LABEL if j == 0 else None
        g = t[t.species == el]
        if g.empty:
            p.empty_text = f"{el}, {energy_mJ:g} mJ: no data"
            continue
        for c in ORDER:
            r = g[g.component == c].sort_values("z")
            if r.empty:
                continue
            u = r[ucol].to_numpy(float)
            a, b = _bounds(u, r[lo].to_numpy(float), r[hi].to_numpy(float))
            hov = [f"z = {z}, u = {v:.1f} km/s" for z, v in zip(r.z, u, strict=True)]
            p.add(ErrorBars(r.z.to_numpy(float), u, COMPONENT_NAMES.get(c, c),
                            COLORS.get(c, "gray"), ylo=a, yhi=b, marker=MARKERS.get(c, "o"),
                            size=4.5, group=c, hover=hov))
        _mean_lines(p, g, avg_components.get(el, []), ucol)
    return fig


def area_fwhm(tab, energy_mJ) -> Figure:
    """Top: ions per component (model, whole grid) [cm^-2]; bottom: FWHM of the fit [us]."""
    t = tab[(tab.energy_mJ == energy_mJ) & ~tab.excluded]
    fig = Figure(rows=2, cols=2, size_in=(4.7, 4.1), height_px=700, letters=True,
                 legend="figure", title=f"{energy_mJ:g} mJ")
    for j, el in enumerate(["Al", "Nb"]):
        g = t[t.species == el]
        for i, (col, ylab) in enumerate([
                ("Ions_model_full_cm2", lab(r"Ions  (cm$^{-2}$)", "Ions (cm⁻²)")),
                ("FWHM_fit_us", lab(r"FWHM  ($\mu$s)", "FWHM (µs)"))]):
            p = fig.panel(i, j)
            p.ylog = True
            p.headroom = 0.25
            p.title = el if i == 0 else None
            p.xlabel = Z_LABEL if i == 1 else None
            p.ylabel = ylab if j == 0 else None
            for c in ORDER:
                r = g[g.component == c].sort_values("z")
                if r.empty or col not in r:
                    continue
                p.add(Points(r.z.to_numpy(float), r[col].to_numpy(float),
                             COMPONENT_NAMES.get(c, c), COLORS.get(c, "gray"),
                             marker=MARKERS.get(c, "o"), size=4.5, group=c))
    return fig


def fit_summary(tab, avg_components: dict, energies=(180, 370, 780)) -> Figure:
    """All conditions: fitted peak velocity per z, bar = FWHM of the fitted distribution
    (half-maximum points at t -/+ FWHM/2, asymmetric in u)."""
    fig = Figure(rows=2, cols=len(energies), size_in=(6.8, 4.4), height_px=760, letters=True,
                 legend="figure")
    t = tab[~tab.excluded]
    for i, el in enumerate(["Al", "Nb"]):
        for j, mj in enumerate(energies):
            p = fig.panel(i, j)
            p.title = f"{el}, {mj} mJ"
            p.xlabel = Z_LABEL
            p.ylabel = lab(r"Fit peak  $u$  (km/s)", "Fit peak u (km/s)") if j == 0 else None
            g = t[(t.species == el) & (t.energy_mJ == mj)]
            if g.empty:
                p.empty_text = "no data"
                continue
            for c in ORDER:
                r = g[g.component == c].sort_values("z")
                if r.empty:
                    continue
                u, tt = r.u_fit.to_numpy(float), r.t_peak_fit_us.to_numpy(float)
                half = r.FWHM_fit_us.to_numpy(float) / 2 if "FWHM_fit_us" in r else 0 * u
                with np.errstate(divide="ignore", invalid="ignore"):
                    ul = np.where(tt - half > 0, u * tt / (tt - half), np.nan)
                    ur = u * tt / (tt + half)
                p.add(ErrorBars(r.z.to_numpy(float), u, COMPONENT_NAMES.get(c, c),
                                COLORS.get(c, "gray"), ylo=np.nan_to_num(ur, nan=u),
                                yhi=np.nan_to_num(ul, nan=u), marker=MARKERS.get(c, "o"),
                                size=4, group=c))
            _mean_lines(p, g, avg_components.get(el, []), "u_fit")
    return fig
