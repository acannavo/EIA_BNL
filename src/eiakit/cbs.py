"""
CBS stage: port of cbs_manual_fitter_v12 without the GUI.

The legacy fitter works per charge state z on the calibrated spectra of the checkpoint:
    signal(z) = sum over the fitted masses of corrected_spectra[(m, z)]  (/ z if per_ion)
Each component (CBS 1..n) has a peak velocity u = L / t_peak, a FWHM, a fit window and
(kT, V0, A). V0 is never fitted on its own: it follows from (kT, u) through the peak
constraint V0_from_degen. The 'Fit' button scans kT on linspace(kT_min, kT_hi, kT_nscan)
against the data minus the other components, with A solved analytically (weights 1/data),
and repeats this n_iter_sub times. kT can also come from the FWHM first guess, a typed
value or the slider, and A from the peak-height guess, the hold rule or the slider: the
stored values do not say which (origin_of() classifies them).

Everything the later stages use (areas, fractions, fitted peak / FWHM, chi2) is a
deterministic function of the stored (kT, V0, A, window) and the checkpoint:
replay_fit_file() recomputes it and compares with the legacy fit and areas files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import physics as ph

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---- signals ---------------------------------------------------------------------------

def build_signals(ck: dict, mass_labels, per_ion: bool = True):
    """
    z curves and raw experimental points from a pipeline checkpoint, as in the fitter:
    only the (m, z) with m in mass_labels; the curve is the sum over those masses
    (divided by z for the per-ion signal); the exp. points are concatenated, sorted by
    time, sign-flipped (/ z if per ion). Returns (tof_s, curves {z: y}, exp {z: (t_us, y)}).
    """
    keep = None if mass_labels is None else {int(m) for m in mass_labels}

    def ok(k):
        return keep is None or int(k[0]) in keep

    spectra = {k: v for k, v in ck["corrected_spectra"].items() if ok(k)}
    points = {k: v for k, v in ck["corrected_exp_points"].items() if ok(k)}
    ptof = {k: v for k, v in ck["corrected_exp_tof"].items() if ok(k)}
    zs = sorted({z for (_, z) in spectra})
    curves = {z: sum(a for (m, zz), a in spectra.items() if zz == z) for z in zs}
    if per_ion:
        curves = {z: c / z for z, c in curves.items()}
    exp = {}
    for z in zs:
        t = [ptof[k] for k in points if k[1] == z]
        y = [points[k] for k in points if k[1] == z]
        if not t:
            continue
        t, y = np.concatenate(t), np.concatenate(y)
        o = np.argsort(t)
        yy = -y[o]
        exp[z] = (t[o] * 1e6, yy / z if per_ion else yy)
    return np.asarray(ck["corrected_tof"], float), curves, exp


# ---- model -----------------------------------------------------------------------------

@dataclass
class Component:
    kT: float | None = None
    V0: float | None = None
    A: float | None = None
    t_peak: float | None = None          # us (u = L / t_peak)
    fwhm: float | None = None            # picked FWHM, us
    win_left: float | None = None        # us; None -> default t_peak +/- hfac * FWHM
    win_right: float | None = None
    win_src: str = "default"
    t_left: float = np.nan               # clicked half-maximum edges, us
    t_right: float = np.nan
    extra: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return None not in (self.kT, self.V0, self.A, self.fwhm)


@dataclass
class CBSModel:
    tof: np.ndarray                      # s
    m_kg: float
    L_m: float
    e_C: float
    gamma: float = 5.0 / 3.0
    kT_min: float = 1.0
    kT_max: float = 100.0
    kT_guess_max: float = 1.0e4
    kT_nscan: int = 600
    win_hfac: float = 3.0
    noise_floor: float = 1e-12
    weight_floor_frac: float = 0.0
    n_iter_sub: int = 5
    per_ion: bool = True

    @classmethod
    def from_settings(cls, tof, m_kg, L_m, e_C, s: dict) -> CBSModel:
        return cls(tof=np.asarray(tof, float), m_kg=m_kg, L_m=L_m, e_C=e_C,
                   gamma=s.get("gamma", 5 / 3), kT_min=s.get("kT_min_eV", 1.0),
                   kT_max=s.get("kT_max_eV", 100.0),
                   kT_guess_max=s.get("kT_guess_max_eV", 1e4),
                   kT_nscan=int(s.get("kT_nscan", 600)), win_hfac=s.get("fit_win_hfac", 3.0),
                   noise_floor=s.get("noise_floor", 1e-12),
                   weight_floor_frac=s.get("weight_floor_frac", 0.0),
                   n_iter_sub=int(s.get("n_iter_sub", 5)), per_ion=s.get("per_ion", True))

    @property
    def tus(self):
        return self.tof * 1e6

    # physics (fitter literal)
    def shape(self, kT, V0, z, t=None):
        return ph.cbs_shape(self.tof if t is None else t, kT, V0, z, self.m_kg, self.L_m,
                            self.e_C, self.gamma)

    def V0_from_u(self, kT, u, z):
        """V0_from_degen; None where v_ac <= 0."""
        kT_J = kT * self.e_C
        v_th = np.sqrt(self.gamma * kT_J / self.m_kg)
        v_s = u - 5.0 * kT_J / (self.m_kg * u)
        v_ac = v_s - v_th
        if v_ac <= 0:
            return None
        return self.m_kg * v_ac ** 2 / (2.0 * self.e_C * z)

    def u_of(self, kT, V0, z):
        return float(ph.cbs_peak_velocity(kT, V0, z, self.m_kg, self.e_C, self.gamma))

    def half_edges(self, kT, V0, z, t=None):
        t = self.tof if t is None else t
        tL, tR = ph.half_maximum_edges(t * 1e6, self.shape(kT, V0, z, t))
        return tL, tR

    def fwhm(self, kT, V0, z, t=None):
        tL, tR = self.half_edges(kT, V0, z, t)
        return tR - tL

    def metrics(self, c: Component, z):
        """fit_metrics: peak position / height and half-maximum edges of the model."""
        y = c.A * self.shape(c.kT, c.V0, z)
        pk = int(np.argmax(y))
        tL, tR = self.half_edges(c.kT, c.V0, z)
        return {"t_peak": float(self.tus[pk]), "I_peak": float(y[pk]), "t_left": float(tL),
                "t_right": float(tR), "fwhm": float(tR - tL)}

    # windows and weights
    def mask(self, c: Component):
        if c.win_left is not None and c.win_right is not None:
            return (self.tus >= c.win_left) & (self.tus <= c.win_right)
        return np.abs(self.tus - c.t_peak) < self.win_hfac * c.fwhm

    def edges(self, c: Component):
        if c.win_left is not None and c.win_right is not None:
            return c.win_left, c.win_right
        hw = self.win_hfac * c.fwhm
        return max(c.t_peak - hw, float(self.tus.min())), min(c.t_peak + hw,
                                                                float(self.tus.max()))

    def denom(self, d):
        floor = self.noise_floor
        if self.weight_floor_frac > 0 and np.size(d):
            floor = max(floor, self.weight_floor_frac * float(np.max(d)))
        return np.maximum(d, floor)

    def analytic_A(self, F, data, mask):
        den_w = self.denom(data[mask])
        num = np.sum(data[mask] * F[mask] / den_w)
        den = np.sum(F[mask] ** 2 / den_w)
        return num / den if den > 0 else np.nan

    def chi2_residual(self, kT, u, z, resid, mask):
        V0 = self.V0_from_u(kT, u, z)
        if V0 is None or V0 <= 0 or mask.sum() < 5:
            return 1e30
        F = self.shape(kT, V0, z)
        A = self.analytic_A(F, resid, mask)
        if np.isnan(A) or A <= 0:
            return 1e30
        return float(np.sum((resid[mask] - A * F[mask]) ** 2 / self.denom(resid[mask])))

    def _scan(self, grid, u, z, resid, mask):
        """chi2_residual for every kT of the grid, with the shape evaluated only inside the
        window (same values element by element, ~30x faster than the full time grid)."""
        if mask.sum() < 5:
            return np.full(len(grid), 1e30)
        tm, rm = self.tof[mask], resid[mask]
        den = self.denom(rm)
        out = np.full(len(grid), 1e30)
        for j, k in enumerate(grid):
            V0 = self.V0_from_u(k, u, z)
            if V0 is None or V0 <= 0:
                continue
            F = self.shape(k, V0, z, tm)
            d = np.sum(F ** 2 / den)
            A = np.sum(rm * F / den) / d if d > 0 else np.nan
            if np.isnan(A) or A <= 0:
                continue
            out[j] = float(np.sum((rm - A * F) ** 2 / den))
        return out

    def residual(self, comps, i, curve, z):
        r = curve.copy()
        for j, c in enumerate(comps):
            if j != i and None not in (c.kT, c.V0, c.A):
                r -= c.A * self.shape(c.kT, c.V0, z)
        return r

    def chi2_component(self, comps, i, curve, z):
        c = comps[i]
        r = self.residual(comps, i, curve, z)
        m = self.mask(c)
        y = c.A * self.shape(c.kT, c.V0, z)
        x2 = float(np.sum((r[m] - y[m]) ** 2 / self.denom(r[m])))
        dof = int(m.sum()) - 2
        return x2, dof

    def chi2_total(self, comps, curve, z):
        act = [c for c in comps if c.active]
        if not act:
            return np.nan, 0
        m = np.zeros(self.tof.size, bool)
        y = np.zeros(self.tof.size)
        for c in act:
            m |= self.mask(c)
            y += c.A * self.shape(c.kT, c.V0, z)
        x2 = float(np.sum((curve[m] - y[m]) ** 2 / self.denom(curve[m])))
        return x2, int(m.sum()) - 2 * len(act)

    def areas(self, comps, curve, z):
        """compute_areas of the fitter (trapezoid, us)."""
        act = [c for c in comps if c.active]
        if not act:
            return None
        tus = self.tus
        full = float(_trapz(curve, tus))
        um = np.zeros(tus.size, bool)
        for c in act:
            um |= self.mask(c)
        union = float(_trapz(curve[um], tus[um])) if um.sum() > 1 else 0.0
        q = self.e_C * (1.0 if self.per_ion else z)
        rows, tot_full, tot_win = [], 0.0, 0.0
        for k, c in enumerate(act):
            m = self.mask(c)
            wl, wr = self.edges(c)
            y = c.A * self.shape(c.kT, c.V0, z)
            a_full = float(_trapz(y, tus))
            a_win = float(_trapz(y[m], tus[m]))
            a_dat = float(_trapz(curve[m], tus[m]))
            ft = self.metrics(c, z)
            tot_full += a_full
            tot_win += a_win
            rows.append({"z": z, "CBS": c.extra.get("cbs", k + 1), "t_peak_us": c.t_peak,
                         "FWHM_us": c.fwhm,
                         "win_left_us": wl, "win_right_us": wr, "Area_model": a_win,
                         "Area_data_i": a_dat, "Area_data_union": union,
                         "Area_data_full": full, "Area_model_full": a_full,
                         "Win_coverage": a_win / a_full if a_full > 0 else np.nan,
                         "t_peak_fit_us": ft["t_peak"], "FWHM_fit_us": ft["fwhm"],
                         "t_left_pick_us": c.t_left, "t_right_pick_us": c.t_right,
                         "t_left_fit_us": ft["t_left"], "t_right_fit_us": ft["t_right"],
                         "win_src": c.win_src, "Ions_model_win_cm2": a_win * 1e-6 / q,
                         "Ions_model_full_cm2": a_full * 1e-6 / q,
                         "Ions_data_win_cm2": a_dat * 1e-6 / q})
        for r in rows:
            r["Fraction"] = r["Area_model"] / tot_win if tot_win > 0 else np.nan
            r["Fraction_full"] = r["Area_model_full"] / tot_full if tot_full > 0 else np.nan
        return pd.DataFrame(rows)

    # fitting
    def scan_grid(self, kT_hi):
        return np.linspace(self.kT_min, max(kT_hi, self.kT_max), self.kT_nscan)

    def fit_component(self, comps, i, curve, z, kT_hi=None, n_iter=None):
        """
        The 'Fit' button: n_iter_sub times, scan kT on the grid against the data minus
        the other components (their current values), V0 from (kT, u), A analytic.
        Updates comps[i] in place; returns {kT, V0, A, chi2, grid, chi2_scan}.
        """
        c = comps[i]
        u = self.L_m / (c.t_peak * 1e-6)
        grid = self.scan_grid(kT_hi or self.kT_max)
        out = None
        for _ in range(n_iter or self.n_iter_sub):
            r = self.residual(comps, i, curve, z)
            m = self.mask(c)
            x2 = self._scan(grid, u, z, r, m)
            j = int(np.nanargmin(x2))
            kT = float(grid[j])
            V0 = self.V0_from_u(kT, u, z)
            if V0 is None or V0 <= 0:
                break
            A = self.analytic_A(self.shape(kT, V0, z), r, m)
            if np.isnan(A) or A <= 0:
                break
            c.kT, c.V0, c.A = kT, V0, A
            out = {"kT": kT, "V0": V0, "A": A, "chi2": float(x2[j]), "grid": grid,
                   "chi2_scan": x2}
        return out

    def kT_from_fwhm(self, fwhm_target, u, z):
        """find_kT_from_fwhm: log grid, FWHM on the coarse grid tof[::4]."""
        kT_arr = np.geomspace(self.kT_min, self.kT_guess_max, 600)
        t_c = self.tof[::4]
        fw = []
        for k in kT_arr:
            V0 = self.V0_from_u(k, u, z)
            fw.append(self.fwhm(k, V0, z, t_c) if V0 else np.nan)
        fw = np.array(fw)
        ok = ~np.isnan(fw)
        if not ok.any():
            return None, None
        diff = fw[ok] - fwhm_target
        kv = kT_arr[ok]
        sc = np.where(np.diff(np.sign(diff)))[0]
        if len(sc) == 0:
            kT = float(kv[int(np.argmin(np.abs(diff)))])
            return kT, self.V0_from_u(kT, u, z)
        i = sc[0]
        f = fw[ok]
        kT = kv[i] + (fwhm_target - f[i]) / (f[i + 1] - f[i]) * (kv[i + 1] - kv[i])
        return float(kT), self.V0_from_u(kT, u, z)


# ---- legacy replay ---------------------------------------------------------------------

def components_from_fit_file(fit) -> dict[int, list[Component]]:
    """Components per z from a parsed CBS_fit_results file (full-precision table values;
    t_peak from the text block, i.e. after t= / u= overrides)."""
    tab = fit.table
    out: dict[int, list[Component]] = {}
    for r in tab.sort_values(["z", "CBS"]).itertuples():
        blk = fit.component(int(r.z), int(r.CBS))
        t_peak = blk.picks.get("t_peak", r.t_peak_pick) if blk else r.t_peak_pick
        src = str(r.win_src)
        c = Component(kT=float(r.kT_eV), V0=float(r.V0_V), A=float(r.A), t_peak=float(t_peak),
                      fwhm=float(r.FWHM_pick), win_left=float(r.win_left),
                      win_right=float(r.win_right), win_src=src.replace("_", " "),
                      t_left=float(r.t_left_pick), t_right=float(r.t_right_pick),
                      extra={"t_peak_click": float(r.t_peak_pick), "row": r._asdict(),
                             "cbs": int(r.CBS),
                             "locked": bool(blk.window.get("locked", 0)) if blk else False})
        out.setdefault(int(r.z), []).append(c)
    return out


AREA_COLS = ["Area_model", "Fraction", "Area_data_i", "Area_data_union", "Area_data_full",
             "Area_model_full", "Fraction_full", "Win_coverage", "t_peak_fit_us",
             "FWHM_fit_us", "t_left_fit_us", "t_right_fit_us", "Ions_model_win_cm2",
             "Ions_model_full_cm2", "Ions_data_win_cm2"]


def origin_of(model: CBSModel, comps, i, curve, z) -> dict:
    """
    How were kT and A of component i most likely set (diagnostic, not exact):
      kT: 'fit'    - a local chi2 minimum of the scan within one grid step,
          'fwhm'   - the FWHM first guess for the picked FWHM,
          'manual' - neither (typed value / slider / older state);
      A:  'fit'    - equal to the analytic A for (kT, V0) on the current residual,
          'manual' - otherwise (peak-height guess, hold rule, slider, older residual).
    Also returns u from (kT, V0) vs u = L / t_peak and V0 from (kT, u).
    """
    c = comps[i]
    u_pick = model.L_m / (c.t_peak * 1e-6)
    u_kv = model.u_of(c.kT, c.V0, z)
    V0_u = model.V0_from_u(c.kT, u_pick, z)
    r = model.residual(comps, i, curve, z)
    m = model.mask(c)
    A_an = model.analytic_A(model.shape(c.kT, c.V0, z), r, m)
    a_org = "fit" if np.isfinite(A_an) and abs(A_an / c.A - 1) < 1e-4 else "manual"
    step = (max(model.kT_max, 1.5 * c.kT) - model.kT_min) / (model.kT_nscan - 1)
    ks = c.kT + step * np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    x2 = np.array([model.chi2_residual(k, u_pick, z, r, m) if k > 0 else 1e30 for k in ks])
    k_org = "fit" if x2[2] <= x2[0] and x2[2] <= x2[4] else "manual"
    if k_org != "fit":
        kT_g, _ = model.kT_from_fwhm(c.fwhm, u_pick, z)
        if kT_g is not None and abs(kT_g - c.kT) <= 5e-5 + 1e-6 * abs(c.kT):
            k_org = "fwhm"
    return {"kT_origin": k_org, "A_origin": a_org, "A_analytic": A_an,
            "u_pick_kms": u_pick / 1e3, "u_from_kT_V0_kms": u_kv / 1e3,
            "du_rel": u_kv / u_pick - 1,
            "V0_from_kT_u": np.nan if V0_u is None else V0_u}


# rounding of the stored parameters in the fit table: kT, V0 to 1e-4, A to 7 digits
ROUND_KT, ROUND_V0, ROUND_A_REL = 5e-5, 5e-5, 5e-7


def rounding_bounds(model: CBSModel, comps, i, curve, z) -> dict:
    """
    How much the stored rounding of (kT, V0, A) alone can move this component's model
    area (whole grid and in its window), fitted FWHM and peak height (relative, first
    order: sum of |d q / d p| * rounding of p).
    """
    c = comps[i]
    tus, m = model.tus, model.mask(c)

    def q(kT, V0, A):
        y = A * model.shape(kT, V0, z)
        return np.array([_trapz(y, tus), _trapz(y[m], tus[m]), model.fwhm(kT, V0, z),
                         y.max()])

    q0 = q(c.kT, c.V0, c.A)
    dk = q(c.kT + ROUND_KT, c.V0, c.A) - q0
    dv = q(c.kT, c.V0 + ROUND_V0, c.A) - q0      # V0 printed as 0.0000 lies in [0, 5e-5]
    da = q0 * ROUND_A_REL
    b = (np.abs(dk) + np.abs(dv) + np.abs(da)) / np.abs(q0)
    b[2] += 5e-7 / abs(q0[2])                      # FWHM printed to 1e-6 us
    return {"bound_area_full": b[0], "bound_area_win": b[1], "bound_fwhm": b[2],
            "bound_I_peak": b[3]}


def replay_fit_file(model: CBSModel, curves: dict, fit, areas=None,
                    comps_by_z: dict | None = None) -> dict:
    """
    Recompute the stored quantities of one CBS fit file from (kT, V0, A, window) and the
    checkpoint: fit metrics vs the fit table, chi2 vs the text blocks, areas vs the
    areas table, and the origin of each component. Returns {components, areas, totals}.
    """
    if comps_by_z is None:
        comps_by_z = components_from_fit_file(fit)
    comp_rows, area_tabs, tot_rows = [], [], []
    for z, comps in comps_by_z.items():
        curve = curves.get(z)
        if curve is None:
            for c in comps:
                comp_rows.append({"z": z, "CBS": c.extra["cbs"], "status": "no curve for z"})
            continue
        for k, c in enumerate(comps):
            row = c.extra["row"]
            ft = model.metrics(c, z)
            x2, dof = model.chi2_component(comps, k, curve, z)
            blk = fit.component(z, c.extra["cbs"])
            org = origin_of(model, comps, k, curve, z)
            bnd = rounding_bounds(model, comps, k, curve, z)
            comp_rows.append({
                "z": z, "CBS": c.extra["cbs"], "kT": c.kT, "V0": c.V0, "A": c.A,
                "t_peak_pick": c.t_peak, "t_peak_click": c.extra["t_peak_click"],
                "t_peak_fit_old": row["t_peak_fit"], "t_peak_fit_new": ft["t_peak"],
                "FWHM_fit_old": row["FWHM_fit"], "FWHM_fit_new": ft["fwhm"],
                "I_fit_old": row["I_peak_fit"], "I_fit_new": ft["I_peak"],
                "chi2_old": blk.chi2 if blk else np.nan, "chi2_new": x2, "dof_new": dof,
                **org, **bnd})
        x2t, doft = model.chi2_total(comps, curve, z)
        zt = fit.z_totals.get(z, {})
        tot_rows.append({"z": z, "chi2_old": zt.get("chi2", np.nan), "chi2_new": x2t,
                         "dof_old": zt.get("dof", np.nan), "dof_new": doft})
        a = model.areas(comps, curve, z)
        if a is not None:
            area_tabs.append(a)
    comp = pd.DataFrame(comp_rows)
    area_new = pd.concat(area_tabs, ignore_index=True) if area_tabs else pd.DataFrame()
    cmp_areas = pd.DataFrame()
    if areas is not None and not area_new.empty:
        old = areas.table.rename(columns=lambda s: s.strip())
        mrg = area_new.merge(old, on=["z", "CBS"], how="outer", suffixes=("_new", "_old"),
                             indicator=True)
        for col in AREA_COLS:
            if f"{col}_new" in mrg and f"{col}_old" in mrg:
                a_, b_ = mrg[f"{col}_new"].astype(float), mrg[f"{col}_old"].astype(float)
                mrg[f"d_{col}"] = (a_ - b_).abs() / b_.abs().where(b_ != 0, 1.0)
        cmp_areas = mrg
    return {"components": comp, "areas": area_new, "areas_cmp": cmp_areas,
            "totals": pd.DataFrame(tot_rows)}


def run_model(db, run, data_root, ck: dict | None = None, use_store: bool = True) -> dict:
    """
    The CBS setting of a run: model (mass, e, L from the constants profile and the
    fit header), per-z signals of the checkpoint (`ck` or the legacy one), and the
    components: the CBS fit store (Step 9) if there is one and use_store, else the legacy
    CBS_fit_results file. Returns {status, model, curves, exp, fit, areas, components, ...}.
    """
    from .audit import _find_cbs_file
    from .cbs_store import default_fits_root, read_store
    from .legacy import load_checkpoint, parse_cbs_areas, parse_cbs_fit_results

    res = run.folder(data_root) / "EIA-Results"
    store = read_store(default_fits_root(), run) if use_store and default_fits_root() else None
    fit_p = store.path if store is not None else \
        _find_cbs_file(res, "CBS_fit_results", run.fit_file_labels, run.target)
    area_p = None if store is not None else \
        _find_cbs_file(res, "CBS_areas", run.fit_file_labels, run.target)
    ck_p = res / "pipeline_checkpoint.npy"
    empty = fit_p is None and getattr(run, "is_new", False) and ck is not None
    if (fit_p is None and not empty) or (ck is None and not ck_p.is_file()):
        return {"status": "missing files", "files": {"fit": fit_p, "checkpoint": ck_p}}
    if empty:                 # new run, nothing fitted yet: no components (CBS fitter)
        import pandas as pd

        from .cbs_store import TABLE_COLS, StoreHeader
        from .legacy import CBSFitFile
        p0 = db.profile_for(run, "cbs")
        fit = CBSFitFile(path=None, header=StoreHeader(
            generated=None, L_EIA_m=run.eia.L_m, M_ION_kg=db.mass_kg(run.target, p0),
            signal="per_ion", e_C=db.e_C(p0)), z_totals={}, components=[],
            table=pd.DataFrame(columns=["z", "CBS", *TABLE_COLS]))
    else:
        fit = store if store is not None else parse_cbs_fit_results(fit_p)
    areas = parse_cbs_areas(area_p) if area_p else None
    if ck is None:
        ck = load_checkpoint(ck_p)
    else:
        ck_p = "<given>"
    prof = db.profile_for(run, "cbs")
    m_kg, e_C = db.mass_kg(run.target, prof), db.e_C(prof)
    L = fit.header.L_EIA_m or run.eia.L_m
    per_ion = True if fit.header.per_ion is None else fit.header.per_ion
    tof, curves, exp = build_signals(ck, run.mass_labels, per_ion)
    model = CBSModel.from_settings(tof, m_kg, L, e_C, db.pipeline["cbs"])
    comps = components_from_fit_file(fit)
    # the stored kT, V0 belong to the fit's constants (legacy profile; M_ION in the
    # header); the shape depends on kT e / M and V0 e / M only, so another mass unit or
    # charge rescales them and leaves times, FWHM and areas unchanged (I1)
    leg = run.legacy.profile_cbs if run.legacy else prof
    m_old = fit.header.M_ION_kg or db.mass_kg(run.target, leg)
    e_old = getattr(fit.header, "e_C", None) or db.e_C(leg)
    s = (m_kg / m_old) * (e_old / e_C)
    if abs(s - 1) > 1e-12:
        for cs in comps.values():
            for c in cs:
                if c.kT is not None:
                    c.kT *= s
                if c.V0 is not None:
                    c.V0 *= s
    return {"status": "ok", "model": model, "curves": curves, "exp": exp, "fit": fit,
            "areas": areas, "components": comps, "profile": prof, "param_scale": s,
            "per_ion": per_ion, "fit_path": fit_p, "areas_path": area_p, "ck_path": ck_p,
            "from_store": store is not None, "empty": empty}


def replay_run_cbs(db, run, data_root, ck: dict | None = None) -> dict:
    """
    Legacy replay of the CBS stage of one run: the run's CBS fit / areas files on a
    checkpoint - the legacy one in EIA-Results, or `ck` (e.g. the new calibration stage
    output, then 'data_area_scale' is the change of calibration since the fit). Criteria:
      data areas    equal to print precision (rel 1e-6), or all off by one common factor
                    (the checkpoint was regenerated after the fit: 'stale checkpoint');
      model areas   within twice the bound set by the rounding of the stored (kT, V0, A);
      FWHM_fit      same; t_peak_fit within 3 grid steps (argmax on a flat top).
    Returns {status, summary, components, areas_cmp, totals, files}.
    """
    rm = run_model(db, run, data_root, ck, use_store=False)
    if rm["status"] == "missing files":
        return rm
    model, curves, fit, areas = rm["model"], rm["curves"], rm["fit"], rm["areas"]
    prof, m_kg, L, fit_p, area_p, ck_p = (rm["profile"], model.m_kg, model.L_m, rm["fit_path"],
                                         rm["areas_path"], rm["ck_path"])
    out = replay_fit_file(model, curves, fit, areas, rm["components"])
    comp, a = out["components"], out["areas_cmp"]
    dt = float(np.median(np.diff(model.tus)))
    s = {"n_components": int(len(comp)), "profile": prof, "M_ION_kg": m_kg,
         "M_ION_header": fit.header.M_ION_kg, "L_EIA_m": L, "grid_step_us": dt,
         "t_peak_fit_max_steps": float(((comp.t_peak_fit_new - comp.t_peak_fit_old).abs()
                                         / dt).max()),
         "fwhm_over_bound": float(((comp.FWHM_fit_new / comp.FWHM_fit_old - 1).abs()
                                   / comp.bound_fwhm).max()),
         "kT_origin": comp.kT_origin.value_counts().to_dict(),
         "A_origin": comp.A_origin.value_counts().to_dict()}
    ok = s["t_peak_fit_max_steps"] <= 3 and s["fwhm_over_bound"] <= 2
    if not a.empty:
        a = a.merge(comp[["z", "CBS", "bound_area_full", "bound_area_win"]], on=["z", "CBS"],
                    how="left")
        out["areas_cmp"] = a
        both = a[a._merge == "both"]
        ratio = both.Area_data_i_new / both.Area_data_i_old
        scale = float(ratio.median())
        spread = float((ratio / scale - 1).abs().max())
        s.update({"areas_paired": int(len(both)), "areas_unpaired": int(len(a) - len(both)),
                  "data_area_scale": scale, "data_area_spread": spread,
                  "model_area_over_bound": float((both.d_Area_model_full
                                                  / both.bound_area_full).max()),
                  "model_win_area_over_bound": float((both.d_Area_model
                                                      / both.bound_area_win).max())})
        s["stale_checkpoint"] = bool(abs(scale - 1) > 1e-6 and spread < 1e-6)
        ok &= (len(both) == len(a) and s["model_area_over_bound"] <= 2
               and s["model_win_area_over_bound"] <= 2)
        data_ok = spread < 1e-6
        if not data_ok:      # spectra changed non-uniformly since the fit (e.g. a correction)
            rel = (ratio / scale - 1).abs()
            s["data_area_changed"] = int((rel > 1e-6).sum())
            s["data_area_max_change"] = spread
    status = "reproduced" if ok else "differs"
    if ok and s.get("data_area_changed"):
        status = (f"model reproduced; data areas changed since the fit "
                  f"({s['data_area_changed']} components, max {s['data_area_max_change']:.2%})")
    elif ok and s.get("stale_checkpoint"):
        status = f"reproduced (checkpoint data x {s['data_area_scale']:.6f} since the fit)"
    out.update(status=status, summary=s,
               files={"fit": str(fit_p), "areas": str(area_p) if area_p else None,
                      "checkpoint": str(ck_p)})
    return out
