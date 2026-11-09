"""Shared style: PRL rc parameters (from the legacy plot scripts) and colours."""
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
             "#CC79A7"]
EXTRA = ["#882255", "#44AA99", "#999933", "#332288", "#AA4499", "#117733", "#88CCEE",
         "#DDCC77", "#661100", "#6699CC", "#AA4466", "#4477AA"]
ENERGY_COLORS = {180: "#1f77b4", 370: "#2ca02c", 780: "#d62728"}   # tab:blue/green/red
ELEMENT_DASH = {"Al": "solid", "Nb": "dash"}
COMPONENT_COLORS = {"fast": "#d62728", "slow": "#1f77b4", "ultra_fast": "#9467bd",
                    "component_4": "#8c564b"}
COMPONENT_NAMES = {"fast": "Fast", "slow": "Slow", "ultra_fast": "Ultra-fast",
                   "component_4": "Component 4"}


def z_color(z: int) -> str:
    """Colour of charge state z (same in every figure): Okabe-Ito, then extras."""
    if 1 <= z <= len(OKABE_ITO):
        return OKABE_ITO[z - 1]
    return EXTRA[(z - len(OKABE_ITO) - 1) % len(EXTRA)]


def energy_color(mj) -> str:
    return ENERGY_COLORS.get(int(round(mj)) if mj is not None else None, "#7f7f7f")


PRL_RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9.5,
    "axes.linewidth": 0.7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "xtick.minor.size": 2, "ytick.minor.size": 2,
    "legend.fontsize": 6.5, "legend.frameon": False, "lines.markersize": 4.5,
    "lines.linewidth": 1.1, "axes.grid": False,
}


def prl_rc(font_scale: float = 1.0) -> dict:
    rc = dict(PRL_RC)
    for k in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
              "legend.fontsize"):
        rc[k] = rc[k] * font_scale
    return rc
