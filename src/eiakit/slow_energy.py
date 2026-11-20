"""
Slow-component kinetic energy versus charge state, E(z), fitted with three laws:

    poirier    E = A (z + 1)^B           (Poirier et al. 2023, Sn)
    linear     E = a + b z
    quadratic  E = a + b z + c z^2

E = m u^2 / 2 from the fitted peak velocity of each slow CBS component (u_fit); the
bars come from the half-maximum points picked in the fitter (u_left / u_right).
Unweighted least squares on E (as a fit to the published points); compared by R^2, the
rms relative residual and AIC = n ln(SS/n) + 2k (lower = better, penalises parameters).
The same fits on the literature E_slow points (Poirier Sn, Kondrashev Nb) check the
procedure.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

MODELS = {
    "poirier": ("A (z+1)^B", lambda z, A, B: A * (z + 1.0) ** B, 2),
    "linear": ("a + b z", lambda z, a, b: a + b * z, 2),
    "quadratic": ("a + b z + c z²", lambda z, a, b, c: a + b * z + c * z * z, 3),
}
PARAM_NAMES = {"poirier": ("A", "B"), "linear": ("a", "b"), "quadratic": ("a", "b", "c")}


def energy_eV(u_kms, m_amu: float, amu_kg: float, e_C: float):
    """Kinetic energy (eV) of an ion of mass m_amu moving at u (km/s)."""
    return 0.5 * m_amu * amu_kg * (np.asarray(u_kms, float) * 1e3) ** 2 / e_C


def fit_models(z, E) -> list[dict]:
    """The three laws on (z, E); a law is skipped when it has as many parameters as
    points (no degree of freedom)."""
    z, E = np.asarray(z, float), np.asarray(E, float)
    ok = np.isfinite(z) & np.isfinite(E)
    z, E = z[ok], E[ok]
    out = []
    for name, (formula, f, k) in MODELS.items():
        r = {"model": name, "formula": formula, "n": len(z), "k": k}
        if len(z) <= k:
            out.append({**r, "status": f"needs > {k} points"})
            continue
        if name == "poirier":
            pos = E > 0
            if pos.sum() < 2:
                out.append({**r, "status": "needs E > 0"})
                continue
            B0, lnA0 = np.polyfit(np.log(z[pos] + 1), np.log(E[pos]), 1)
            p0 = (float(np.exp(lnA0)), float(B0))
        else:
            p0 = np.polyfit(z, E, k - 1)[::-1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, cov = curve_fit(f, z, E, p0=p0, maxfev=20000)
        except (RuntimeError, ValueError) as exc:
            out.append({**r, "status": f"fit failed: {exc}"})
            continue
        res = E - f(z, *p)
        ss = float((res ** 2).sum())
        tot = float(((E - E.mean()) ** 2).sum())
        err = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) \
            else np.full(len(p), np.nan)
        r.update(status="ok", params=[float(v) for v in p], errors=[float(v) for v in err],
                 R2=1 - ss / tot if tot > 0 else np.nan,
                 rms_rel=float(np.sqrt(np.mean((res / np.where(E != 0, E, np.nan)) ** 2))),
                 AIC=float(len(z) * np.log(ss / len(z)) + 2 * k) if ss > 0 else -np.inf)
        out.append(r)
    return out


def curve(model: str, params, z):
    return MODELS[model][1](np.asarray(z, float), *params)


def slow_points(tab: pd.DataFrame, db) -> pd.DataFrame:
    """E(z) of the slow components of the study table (not excluded), per run."""
    t = tab[(tab.component == "slow") & ~tab.excluded].copy()
    if t.empty:
        return pd.DataFrame(columns=["run", "species", "energy_mJ", "z", "E_eV",
                                     "E_lo_eV", "E_hi_eV"])
    prof = db.profile(None)
    m = t.species.map(lambda s: db.species[s].mass_amu)
    E = energy_eV(t.u_fit, 1.0, prof.amu_kg, prof.e_C) * m
    # picked half-maximum points: earlier time = faster = higher E
    lo = energy_eV(t.u_right, 1.0, prof.amu_kg, prof.e_C) * m
    hi = energy_eV(t.u_left, 1.0, prof.amu_kg, prof.e_C) * m
    return pd.DataFrame({"run": t.run, "species": t.species, "energy_mJ": t.energy_mJ,
                         "z": t.z.astype(int), "E_eV": E.to_numpy(float),
                         "E_lo_eV": np.where(np.isfinite(lo), lo, E),
                         "E_hi_eV": np.where(np.isfinite(hi), hi, E)}).sort_values(
        ["species", "energy_mJ", "z"]).reset_index(drop=True)


def literature_points(lit: dict, names) -> dict:
    """{label: (z, E_eV)} of the literature slow energies (Poirier eV, Kondrashev keV)."""
    out = {}
    for key in names or []:
        L = lit.get(key, {})
        if "E_slow_eV" in L:
            out[L.get("label", key)] = (np.asarray(L["z"], float),
                                        np.asarray(L["E_slow_eV"], float))
        elif "E_slow_keV" in L:
            out[L.get("label", key)] = (np.asarray(L["z"], float),
                                        np.asarray(L["E_slow_keV"], float) * 1e3)
    return out


def fit_table(pts: pd.DataFrame, lit: dict) -> pd.DataFrame:
    """One row per (dataset, law): parameters +/- errors, R^2, rms_rel, AIC, best."""
    rows = []
    sets = [(f"{sp}, {mj:g} mJ", g.z.to_numpy(), g.E_eV.to_numpy(), sp, mj)
            for (sp, mj), g in pts.groupby(["species", "energy_mJ"], sort=False)]
    sets += [(lab, z, E, "literature", np.nan) for lab, (z, E) in lit.items()]
    for name, z, E, sp, mj in sets:
        fits = fit_models(z, E)
        good = [f for f in fits if f.get("status") == "ok" and np.isfinite(f["AIC"])]
        best = min(good, key=lambda f: f["AIC"])["model"] if good else None
        for f in fits:
            r = {"dataset": name, "species": sp, "energy_mJ": mj, "model": f["model"],
                 "formula": f["formula"], "n": f["n"], "status": f.get("status")}
            for j, pn in enumerate(PARAM_NAMES[f["model"]]):
                r[pn] = f["params"][j] if "params" in f else np.nan
                r[f"{pn}_err"] = f["errors"][j] if "errors" in f else np.nan
            r.update(R2=f.get("R2", np.nan), rms_rel=f.get("rms_rel", np.nan),
                     AIC=f.get("AIC", np.nan), best_by_AIC=f["model"] == best)
            rows.append(r)
    return pd.DataFrame(rows)
