"""
Figure 14 – Detail zoom-in comparison for three geologically challenging zones.
Rows  : (a) Alteration zone halo, (b) Serpentinite lens, (c) Granite contact
Cols  : PRISMA RGB | Ground Truth | RF | XGBoost | SpectralFormer | S²Mamba
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
_cfg.reinit("PRISMA")
from projutils.config import config

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure14_zoom_comparison.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Unlabeled",                     # 0
    "Alkali feldspar granites",      # 1
    "Alteration zones",              # 2
    "Biotite granites",              # 3
    "Gabbro",                        # 4
    "Garnet-muscovite granites",     # 5
    "Gneiss",                        # 6
    "Li-pegmatite granites",         # 7
    "Serpentinite",                  # 8
    "Tonalite-granodiorite",         # 9
    "Wadi deposits",                 # 10
    "Zinnwaldite-muscovite granites",# 11
    "Ophiolitic metagabbro",         # 12
]

# RGB colours (0-255) — index = class value (0-12)
COLORS_RGB = [
    [220, 220, 220],  # 0  unlabeled           – light grey (not black)
    [83,  171,  72],  # 1  Alkali feldspar granites – mid-green
    [255, 140,   0],  # 2  Alteration zones         – vivid orange
    [66,  132,  91],  # 3  Biotite granites          – forest green
    [42,  100,  55],  # 4  Gabbro                   – dark green
    [160,  82,  45],  # 5  Garnet-muscovite granites – sienna
    [105, 188, 200],  # 6  Gneiss                   – sky blue
    [245, 245, 245],  # 7  Li-pegmatite granites     – near-white
    [160,  80, 200],  # 8  Serpentinite              – medium purple
    [218,  51,  44],  # 9  Tonalite-granodiorite     – red
    [119,  35,  36],  # 10 Wadi deposits             – dark red
    [55,  101, 166],  # 11 Zinnwaldite-musc.granites – blue
    [220, 210,  60],  # 12 Ophiolitic metagabbro     – yellow-green
]
COLORS_NORM = [[r/255, g/255, b/255] for r,g,b in COLORS_RGB]

CMAP = ListedColormap(COLORS_NORM)
NORM = BoundaryNorm(boundaries=np.arange(-0.5, 13.5, 1), ncolors=13)

# ── load maps ─────────────────────────────────────────────────────────────────
def load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16)

GT    = load(config.new_final_class_labels)
RF    = load(config.output_rf_tif_path)
XGB   = load(config.output_xgb_tif_path)
VIT   = load(config.outputVIT_tif_path)
MAMBA = load(config.output_MAMBA_tif_path)

H, W = GT.shape

# ── zone detection ────────────────────────────────────────────────────────────
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

zone_a = best_window(GT == 2, pad=55)   # Alteration zones

zone_b = best_window(GT == 8, pad=55)   # Serpentinite

# Zone c: largest interface between class 5 and class 1
dil1 = ndimage.binary_dilation(GT == 1, iterations=3)
dil5 = ndimage.binary_dilation(GT == 5, iterations=3)
contact = dil1 & dil5 & ((GT == 1) | (GT == 5))
zone_c = best_window(contact, pad=65)
if zone_c is None:
    zone_c = best_window((GT==1)|(GT==5), pad=65)

ZONES = [zone_a, zone_b, zone_c]
ZONE_LABELS = [
    "(a) Alteration zone halo\n(structural corridor)",
    "(b) Serpentinite lens\n(Ophiolitic metagabbro matrix)",
    "(c) Garnet-muscovite / Alkali-feldspar\ngranite contact",
]
FOCUS_CLASSES = [2, 8, (1, 5)]   # class(es) of interest per zone

# ── load PRISMA tif for SWIR false-colour composite ───────────────────────────
# Bands selected as the highest inter-class separability within each atmospheric
# transmission window (verified against actual HDF5 wavelengths; zero-wl
# artefact bands tif[166,167] excluded):
#   R: tif[88]  = 2283.5 nm (rasterio 89)  – Mg-OH / serpentine  [W1: 2222–2497 nm]
#   G: tif[99]  = 2061.4 nm (rasterio 100) – highest SWIR BetwnVar overall [W2: 1832–2078 nm]
#   B: tif[127] = 1554.8 nm (rasterio 128) – SWIR1 baseline       [W3: 1142–1565 nm]
RGB_BANDS = (89, 100, 128)   # 1-indexed rasterio bands
NODATA    = -9999.0
with rasterio.open(config.tifpath) as src:
    RGB_RAW = src.read(list(RGB_BANDS)).astype(np.float32)   # (3, H, W)

# per-band 2–98th percentile stretch using full scene valid pixels
RGB_DISP = np.zeros_like(RGB_RAW)
for i, band in enumerate(RGB_RAW):
    valid = band[band > NODATA + 1]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    RGB_DISP[i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)
RGB_DISP = np.transpose(RGB_DISP, (1, 2, 0))   # → (H, W, 3)

MAPS   = [GT, RF, XGB, VIT, MAMBA]
TITLES = ["Ground Truth\n(labeled polygons)", "Random Forest", "XGBoost",
          "SpectralFormer\n(ViT)", "S²Mamba"]
TITLE_COLORS = ["#333333", "#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

# ── helper: draw a thin border on the S²Mamba panel ──────────────────────────
def highlight_ax(ax, color="#4CAF50", lw=2.8):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 11))

# grid: 3 data rows, label col + RGB col + 5 classification cols
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
    # binary mask of any labeled pixel in this zone (for contour overlay)
    gt_labeled = (gt_crop > 0).astype(float)

    focus_classes = [focus] if isinstance(focus, int) else list(focus)

    # row label (col 0)
    ax_lbl = fig.add_subplot(gs[row, 0])
    ax_lbl.axis("off")
    ax_lbl.text(0.5, 0.5, zlabel,
                ha="center", va="center", fontsize=9,
                fontweight="bold", linespacing=1.5,
                transform=ax_lbl.transAxes)

    # RGB composite panel (col 1)
    ax_rgb = fig.add_subplot(gs[row, 1])
    rgb_crop = RGB_DISP[r0:r1, c0:c1, :]
    # mask nodata pixels (set to black)
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
    # scale bar on RGB panel
    sh_rgb, sw_rgb = rgb_crop.shape[:2]
    ax_rgb.plot([5, 5+40], [sh_rgb-7, sh_rgb-7], color="white",
                linewidth=2.2, solid_capstyle="butt")
    ax_rgb.text(25, sh_rgb-12, "40 px", color="white", fontsize=6.5,
                ha="center", va="bottom", fontweight="bold")

    for col, (data, title, tcol) in enumerate(zip(MAPS, TITLES, TITLE_COLORS)):
        ax = fig.add_subplot(gs[row, col+2])   # shifted: RGB is col 1
        crop = data[r0:r1, c0:c1]
        ax.imshow(crop, cmap=CMAP, norm=NORM, interpolation="nearest")

        # GT panel: highlight focus-class boundaries in white
        if col == 0:
            for fc in focus_classes:
                fc_mask = (crop == fc).astype(float)
                if fc_mask.any():
                    ax.contour(fc_mask, levels=[0.5], colors=["white"],
                               linewidths=1.2, alpha=1.0)
        else:
            # On model panels: overlay GT polygon boundary as thin white contour
            if gt_labeled.any():
                ax.contour(gt_labeled, levels=[0.5], colors=["white"],
                           linewidths=0.6, alpha=0.55, linestyles="dashed")

        # S²Mamba column gets a coloured border
        if col == 4:
            highlight_ax(ax, color="#4CAF50", lw=3)
        else:
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

        ax.set_xticks([]); ax.set_yticks([])

        if row == 0:
            ax.set_title(title, fontsize=10.5, fontweight="bold",
                         color=tcol, pad=5)

        # scale bar
        sb_len = 40
        sh, sw = crop.shape
        ax.plot([5, 5+sb_len], [sh-7, sh-7], color="white", linewidth=2.2,
                solid_capstyle="butt")
        ax.text(5+sb_len//2, sh-12, "40 px",
                color="white", fontsize=6.5, ha="center", va="bottom",
                fontweight="bold")

# ── shared colorbar ───────────────────────────────────────────────────────────
cax = fig.add_axes([0.07, 0.05, 0.91, 0.035])
sm  = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
sm.set_array([])
cb  = fig.colorbar(sm, cax=cax, orientation="horizontal",
                   ticks=range(0, 13))
cb.set_ticklabels(CLASS_NAMES, fontsize=7.5)
cb.ax.tick_params(length=3, width=0.6)
plt.setp(cb.ax.get_xticklabels(), rotation=25, ha="right")
cax.set_title("Geological class", fontsize=8.5, pad=3)

# ── legend for zone (a) annotation ───────────────────────────────────────────
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
    "PRISMA Aboreshead dataset · experiments_008",
    fontsize=12, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
