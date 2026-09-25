"""Shared NMI style and reproducible static exports for the six-figure revision."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parents[1]
ROOT = next(p for p in RUN.parents if (p / "START_HERE.md").exists())
ASSETS = RUN / "assets"
DATA = RUN / "source_data"
A = "#B64342"
C = "#D98884"
W = "#2C737A"
ABSTAIN = "#E9EBEC"
EDGE = "#9AA0A6"
HUMAN = "#4B5C9B"
NEUTRAL = "#8A8F98"
TEXT = "#242424"
ORDER = ["A", "C", "W"]
COLORS = {"A": A, "C": C, "W": W}

def apply_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "lines.linewidth": 1.2,
        "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=10)
    ax.text(-.14, 1.045, letter, transform=ax.transAxes,
            fontweight="bold", fontsize=11, va="bottom", ha="left")

def export(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(ASSETS / f"{name}.{ext}", dpi=300)
    plt.close(fig)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_record(figure, panel_id, path, fields, unit, filters, notes=""):
    path = Path(path).resolve()
    return dict(figure=figure, panel=panel_id, source=str(path.relative_to(ROOT)),
                sha256=sha256(path), fields=fields, unit=unit, filters=filters, notes=notes)
