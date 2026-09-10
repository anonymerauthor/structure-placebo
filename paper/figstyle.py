"""Shared figure style for the manuscript.

IEEE requirements the settings below exist to satisfy:
  * column widths are fixed (3.5 in single, 7.16 in double), so figures are
    authored at final size and never rescaled in LaTeX;
  * Type 3 fonts are rejected, hence pdf.fonttype = 42 (TrueType);
  * text in figures must match the body font size after placement, so nothing
    smaller than 7 pt is used;
  * figures must remain readable in greyscale, so every colour-coded series is
    also distinguished by marker or linestyle.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

COL_SINGLE = 3.5      # inches, IEEE single column
COL_DOUBLE = 7.16     # inches, IEEE double column

# Okabe-Ito: colour-blind safe and well separated in greyscale.
PALETTE = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#7F7F7F",
}
ORDER = ["blue", "red", "green", "orange", "purple", "sky", "grey"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]


def apply():
    mpl.rcParams.update({
        "pdf.fonttype": 42,          # TrueType; Type 3 is rejected by IEEE
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def save(fig, name, outdir="paper/figures"):
    """Write both PDF (for LaTeX) and PNG (for quick inspection)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    return os.path.join(outdir, name + ".pdf")


def c(name):
    return PALETTE[name]
