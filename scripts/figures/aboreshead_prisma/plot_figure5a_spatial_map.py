"""
Figure 5a – Spatial sample distribution map only
Training samples (filled circles) and test samples (crosses)
overlaid on SWIR–NIR–Red false-colour composite, colour-coded by class.
"""

import os, sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA")
from projutils.config import config

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure5a_spatial_map.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Unlabeled",
    "Alkali feldspar granites",
    "Alteration zones",
    "Biotite granites",
    "Gabbro",
    "Garnet-muscovite granites",
    "Gneiss",
    "Li-pegmatite granites",
    "Serpentinite",
    "Tonalite-granodiorite",
    "Wadi deposits",
    "Zinnwaldite-muscovite granites",
    "Ophiolitic metagabbro",
]

# 12 maximally distinct, high-saturation colours – no near-white or near-black.
# Each class has a unique hue region; black marker edges boost visibility on the FCC.
COLORS_NORM = [
    [0.86, 0.86, 0.86],  # 0  Unlabeled             – light gray
    [0.92, 0.10, 0.18],  # 1  Alkali feldspar gr.   – crimson red
    [1.00, 0.55, 0.00],  # 2  Alteration zones       – vivid orange
    [0.05, 0.62, 0.25],  # 3  Biotite granites       – emerald green
    [0.10, 0.35, 0.92],  # 4  Gabbro                – royal blue
    [0.65, 0.05, 0.85],  # 5  Garnet-musc. gr.      – vivid purple
    [0.00, 0.82, 0.88],  # 6  Gneiss                – bright cyan
    [1.00, 0.90, 0.00],  # 7  Li-pegmatite gr.       – vivid yellow
    [0.95, 0.15, 0.60],  # 8  Serpentinite           – hot pink / magenta
    [0.08, 0.72, 0.55],  # 9  Tonalite-granodiorite  – teal green
    [0.55, 0.28, 0.06],  # 10 Wadi deposits          – sienna brown
    [0.60, 0.88, 0.05],  # 11 Zinnwaldite-musc. gr. – lime yellow-green
    [0.28, 0.05, 0.72],  # 12 Ophiolitic metagabbro  – deep indigo
]

# ── wavelength → band index ────────────────────────────────────────────────────
vnir_all = np.linspace(400.0, 1010.0, 66)
swir_all = np.linspace(920.0, 2505.0, 173)
vnir_rm  = [i-1 for i in [1,2,3,4,5]]
swir_rm  = ([i-1 for i in range(3,6)] + [i-1 for i in range(40,57)] +
             [i-1 for i in range(86,112)] + [i-1 for i in range(152,172)])
SPECTRAL_WL = np.concatenate([vnir_all[[i for i in range(66) if i not in vnir_rm]],
                               swir_all[[i for i in range(173) if i not in swir_rm]]])

def wl_to_idx(nm):
    return int(np.argmin(np.abs(SPECTRAL_WL - nm)))

IDX_R = wl_to_idx(2200)  # SWIR clay
IDX_G = wl_to_idx(907)   # NIR
IDX_B = wl_to_idx(650)   # Red

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading …")
with rasterio.open(config.tifpath) as src:
    cube = src.read([IDX_R+1, IDX_G+1, IDX_B+1]).astype(np.float32)
    H, W = src.height, src.width

with rasterio.open(config.train_mask_path) as src:
    TR = src.read(1).astype(np.int16)
with rasterio.open(config.test_mask_path) as src:
    TE = src.read(1).astype(np.int16)

# ── FCC: 2–98th percentile stretch ────────────────────────────────────────────
fcc = np.zeros((H, W, 3), dtype=np.float32)
for i, band in enumerate(cube):
    valid = band[band > -9000.]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    fcc[:, :, i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 9))
ax.imshow(fcc, interpolation="bilinear", aspect="equal")

classes = list(range(1, 13))
for cls in classes:
    color = COLORS_NORM[cls]

    tr_rows, tr_cols = np.where(TR == cls)
    if len(tr_rows):
        ax.scatter(tr_cols, tr_rows, c=[color], s=14, marker="o",
                   linewidths=0.5, edgecolors="black", alpha=0.95, zorder=3)

    te_rows, te_cols = np.where(TE == cls)
    if len(te_rows):
        ax.scatter(te_cols, te_rows, c=[color], s=16, marker="+",
                   linewidths=1.0, alpha=0.80, zorder=2)

ax.set_xlim(0, W); ax.set_ylim(H, 0)
ax.set_xticks([]); ax.set_yticks([])

# ── FCC band label ────────────────────────────────────────────────────────────
ax.text(0.01, 0.01,
        "False-colour composite  R: 2200 nm · G: 907 nm · B: 650 nm",
        transform=ax.transAxes, fontsize=8, color="white",
        va="bottom", ha="left",
        bbox=dict(facecolor="black", alpha=0.45, pad=3, edgecolor="none"))

# ── marker-type legend (top-right) ────────────────────────────────────────────
marker_legend = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="dimgray",
           markeredgecolor="dimgray", markersize=7,
           label=f"Training  (n = {int(np.sum(TR > 0)):,})"),
    Line2D([0],[0], marker="+", color="dimgray", markersize=9,
           markeredgewidth=1.2,
           label=f"Test  (n = {int(np.sum(TE > 0)):,})"),
]
leg1 = ax.legend(handles=marker_legend, loc="upper right", fontsize=9,
                 framealpha=0.88, edgecolor="#aaaaaa")
ax.add_artist(leg1)

# ── class-colour legend (bottom, three rows) ──────────────────────────────────
class_handles = [
    mpatches.Patch(facecolor=COLORS_NORM[c],
                   edgecolor="black", linewidth=0.8,
                   label=CLASS_NAMES[c])
    for c in classes
]
ax.legend(handles=class_handles, loc="lower center",
          ncol=4, fontsize=8.0,
          framealpha=0.92, edgecolor="#555555",
          bbox_to_anchor=(0.5, -0.01),
          bbox_transform=ax.transAxes)

ax.set_title(
    "Figure 5a – Spatial Distribution of Training and Test Samples\n"
    "PRISMA Aboreshead · experiments_008",
    fontsize=12, fontweight="bold", pad=8,
)

plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
