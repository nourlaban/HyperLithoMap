"""
Figure 14 (Bir Gahelia) – Detail zoom-in comparison for three geologically
challenging zones.
Rows  : (a) Alteration zone halo, (b) Serpentinite lens, (c) Monzogranite /
        Tonalite-granodiorite contact
Cols  : PRISMA RGB | Ground Truth | RF | XGBoost | SpectralFormer | S²Mamba

Adapted from plot_zoom_comparison.py (Aboreshead) for the Bir Gahelia
(PRISMA_Gahelia_Clipped) dataset. Class names/colours are read directly from
the training shapefile (gahelia_class_scheme.py) — not hardcoded. Zone (a)
uses class 2 (Alteration zone), zone (b) uses class 8 (Serpentinites), and
zone (c) is the Monzogranite(7)/Tonalite-granodiorite(9) felsic-intrusion
contact, the closest analogous pair of granite-type units in this scene.
"""

import os, sys
import numpy as np
import rasterio
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA_Gahelia_Clipped")
from projutils.config import config

EXP_ID = "experiments_004"
config.experiment_path = os.path.join(config.output_dir, EXP_ID)
config.new_final_class_labels = os.path.join(config.output_dir, EXP_ID, "new__newfinal_class_labels.tif")
config.output_rf_tif_path = os.path.join(config.output_dir, EXP_ID, "rf_final_prediction_map.tif")
config.output_xgb_tif_path = os.path.join(config.output_dir, EXP_ID, "xgb_final_prediction_map.tif")
config.outputVIT_tif_path = os.path.join(config.output_dir, EXP_ID, "VIT_final_prediction_map.tif")
config.output_MAMBA_tif_path = os.path.join(config.output_dir, EXP_ID, "MAMBA_final_prediction_map.tif")

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure14_zoom_comparison.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata: read directly from the training shapefile (no hardcoding) ─
from projutils.class_scheme import load_class_scheme
CLASS_NAMES, SHORT_NAMES, COLORS_NORM, COLORS_RGB = load_class_scheme("PRISMA_Gahelia_Clipped")

CMAP = ListedColormap(COLORS_NORM)
NORM = BoundaryNorm(boundaries=np.arange(-0.5, 11.5, 1), ncolors=11)

def load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16)

GT    = load(config.new_final_class_labels)
RF    = load(config.output_rf_tif_path)
XGB   = load(config.output_xgb_tif_path)
VIT   = load(config.outputVIT_tif_path)
MAMBA = load(config.output_MAMBA_tif_path)

H, W = GT.shape

def best_window(mask, pad=55):
    labeled, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, n+1))
    best = int(np.argmax(sizes)) + 1
    rows, cols = np.where(labeled == best)
    r0, r1 = rows.min()-pad, rows.max()+pad
    c0, c1 = cols.min()-pad, cols.max()+pad
    return (max(0,r0), min(H,r1), max(0,c0), min(W,c1))

zone_a = best_window(GT == 2, pad=55)   # Alteration zone

zone_b = best_window(GT == 8, pad=55)   # Serpentinites

# Zone c: largest interface between Monzogranite (7) and Tonalite-granodiorite (9)
dil7 = ndimage.binary_dilation(GT == 7, iterations=3)
dil9 = ndimage.binary_dilation(GT == 9, iterations=3)
contact = dil7 & dil9 & ((GT == 7) | (GT == 9))
zone_c = best_window(contact, pad=65)
if zone_c is None:
    zone_c = best_window((GT==7)|(GT==9), pad=65)

ZONES = [zone_a, zone_b, zone_c]
ZONE_LABELS = [
    "(a) Alteration zone halo\n(structural corridor)",
    "(b) Serpentinites lens\n(ultramafic body)",
    "(c) Monzogranite / Tonalite-granodiorite\nfelsic intrusion contact",
]
FOCUS_CLASSES = [2, 8, (7, 9)]

# Bands selected as the highest inter-class separability within each atmospheric
# transmission window (PRISMA-instrument-wide, same selection as Aboreshead):
#   R: tif[88]  = 2283.5 nm – Mg-OH / serpentine
#   G: tif[99]  = 2061.4 nm – highest SWIR BetwnVar overall
#   B: tif[127] = 1554.8 nm – SWIR1 baseline
RGB_BANDS = (89, 100, 128)
NODATA    = -9999.0
with rasterio.open(config.tifpath) as src:
    RGB_RAW = src.read(list(RGB_BANDS)).astype(np.float32)

RGB_DISP = np.zeros_like(RGB_RAW)
for i, band in enumerate(RGB_RAW):
    valid = band[band > NODATA + 1]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    RGB_DISP[i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)
RGB_DISP = np.transpose(RGB_DISP, (1, 2, 0))

MAPS   = [GT, RF, XGB, VIT, MAMBA]
TITLES = ["Ground Truth\n(labeled polygons)", "Random Forest", "XGBoost",
          "SpectralFormer\n(ViT)", "S²Mamba"]
TITLE_COLORS = ["#333333", "#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

def highlight_ax(ax, color="#4CAF50", lw=2.8):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)

fig = plt.figure(figsize=(22, 11))

gs = fig.add_gridspec(
    3, 7,
    left=0.06, right=0.99,
    top=0.91,  bottom=0.14,
    hspace=0.10, wspace=0.04,
    width_ratios=[0.20, 1, 1, 1, 1, 1, 1],
)

for row, (zone, zlabel, focus) in enumerate(zip(ZONES, ZONE_LABELS, FOCUS_CLASSES)):
    r0, r1, c0, c1 = zone
    gt_crop = GT[r0:r1, c0:c1]
    gt_labeled = (gt_crop > 0).astype(float)

    focus_classes = [focus] if isinstance(focus, int) else list(focus)

    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.5, zlabel,
                ha="center", va="center", fontsize=9,
                fontweight="bold", linespacing=1.5,
                transform=ax_lbl.transAxes)

    ax_rgb = fig.add_subplot(gs[row, 1])
    rgb_crop = RGB_DISP[r0:r1, c0:c1, :]
    nodata_mask = RGB_RAW[0, r0:r1, c0:c1] <= NODATA + 1
    rgb_display = rgb_crop.copy()
    rgb_display[nodata_mask] = 0
    ax_rgb.imshow(rgb_display, interpolation="bilinear")
    for sp in ax_rgb.spines.values():
        sp.set_linewidth(0.5)
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
    if row == 0:
        ax_rgb.set_title("PRISMA SWIR FCC\nR:2284 / G:2061 / B:1555 nm",
                         fontsize=10.5, fontweight="bold",
                         color="#555555", pad=5)
    sh_rgb, sw_rgb = rgb_crop.shape[:2]
    ax_rgb.plot([5, 5+40], [sh_rgb-7, sh_rgb-7], color="white",
                linewidth=2.2, solid_capstyle="butt")
    ax_rgb.text(25, sh_rgb-12, "40 px", color="white", fontsize=6.5,
                ha="center", va="bottom", fontweight="bold")

    for col, (data, title, tcol) in enumerate(zip(MAPS, TITLES, TITLE_COLORS)):
        ax = fig.add_subplot(gs[row, col+2])
        crop = data[r0:r1, c0:c1]
        ax.imshow(crop, cmap=CMAP, norm=NORM, interpolation="nearest")

        if col == 0:
            for fc in focus_classes:
                fc_mask = (crop == fc).astype(float)
                if fc_mask.any():
                    ax.contour(fc_mask, levels=[0.5], colors=["white"],
                               linewidths=1.2, alpha=1.0)
        else:
            if gt_labeled.any():
                ax.contour(gt_labeled, levels=[0.5], colors=["white"],
                           linewidths=0.6, alpha=0.55, linestyles="dashed")

        if col == 4:
            highlight_ax(ax, color="#4CAF50", lw=3)
        else:
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

        ax.set_xticks([]); ax.set_yticks([])

        if row == 0:
            ax.set_title(title, fontsize=10.5, fontweight="bold",
                         color=tcol, pad=5)

        sb_len = 40
        sh, sw = crop.shape
        ax.plot([5, 5+sb_len], [sh-7, sh-7], color="white", linewidth=2.2,
                solid_capstyle="butt")
        ax.text(5+sb_len//2, sh-12, "40 px",
                color="white", fontsize=6.5, ha="center", va="bottom",
                fontweight="bold")

cax = fig.add_axes([0.07, 0.05, 0.91, 0.035])
sm  = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                   ticks=range(0, 11))
cb.set_ticklabels(CLASS_NAMES, fontsize=7.5)
cb.ax.tick_params(length=3, width=0.6)
plt.setp(cb.ax.get_xticklabels(), rotation=25, ha="right")
cax.set_title("Geological class", fontsize=8.5, pad=3)

legend_items = [
    mpatches.Patch(facecolor="none", edgecolor="white",
                   linewidth=1.2, label="Focus-class boundary (GT only)"),
    mpatches.Patch(facecolor="#4CAF50", edgecolor="#4CAF50",
                   linewidth=1, label="S²Mamba column"),
]
fig.legend(handles=legend_items, loc="lower right", fontsize=7.5,
           framealpha=0.85, ncol=2,
           bbox_to_anchor=(0.99, 0.002))

fig.suptitle(
    "Figure 14 – Zoom-in comparison of geologically challenging zones\n"
    "Bir Gahelia dataset · experiments_004",
    fontsize=12, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
