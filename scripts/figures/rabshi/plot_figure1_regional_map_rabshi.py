"""
Figure 1 (Rabshi) – Regional geological map of the Eastern Desert of Egypt
Left panel : Regional context map – Egypt boundary (local shapefile), Eastern
             Desert highlighted, PRISMA scene footprint polygon (from the
             stacked GeoTIFF's UTM bounds, reprojected to WGS84).
Right panel: PRISMA false-colour composite (850 / 650 / 550 nm as R/G/B) of
             the Rabshi scene with training samples colour-coded by class.

Adapted from plot_figure1_regional_map_gahelia.py for the Rabshi
(RabshiPRISMA) dataset. Two differences from the Gahelia script:
1. Class names/colours/scene path point at Rabshi's 13-class scheme
   (rabshi_class_scheme.py), and the input scene is an ENVI .dat cube
   (not an .he5 file), so there are no corner-attribute footprint
   coordinates to read — the footprint is instead derived from the
   stacked GeoTIFF's UTM-36N bounds, reprojected to WGS84.
2. R/G/B band indices are picked from the scene's own wavelength table
   (nearest to 850/650/550 nm) rather than reusing Gahelia's fixed indices,
   since band ordering/count differs between the two PRISMA products.
"""

import os, sys
import numpy as np
import h5py  # noqa: F401  (kept for parity with the Gahelia script; unused here)
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.warp import transform_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("RabshiPRISMA")
from projutils.config import config

EXP_ID = "experiments_003"
config.experiment_path = os.path.join(config.output_dir, EXP_ID)

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure1_regional_map.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata: read directly from the training shapefile (no hardcoding) ─
from projutils.class_scheme import load_class_scheme
CLASS_NAMES, SHORT_NAMES, COLORS_NORM, COLORS_RGB = load_class_scheme("RabshiPRISMA")

# ── PRISMA scene footprint from the stacked GeoTIFF's UTM bounds ──────────────
print("Reading PRISMA scene footprint …")
with rasterio.open(config.tifpath) as src:
    utm_bounds = src.bounds
    utm_crs = src.crs

fp_w, fp_s, fp_e, fp_n = transform_bounds(utm_crs, "EPSG:4326", *utm_bounds)
footprint_poly = box(fp_w, fp_s, fp_e, fp_n)
footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_poly], crs="EPSG:4326")
fp_cx = (fp_w + fp_e) / 2
fp_cy = (fp_s + fp_n) / 2

print(f"  Footprint: W({fp_w:.3f}°E) E({fp_e:.3f}°E) S({fp_s:.3f}°N) N({fp_n:.3f}°N)")

MAP_W, MAP_E = 29.5, 37.5
MAP_S, MAP_N = 21.5, 30.5

CITIES = [
    (31.235,  30.045, "Cairo",    "right",  "center"),
    (33.812,  26.841, "Hurghada", "left",   "center"),
    (32.649,  25.687, "Luxor",    "left",   "center"),
    (32.532,  23.971, "Aswan",    "left",   "center"),
    (33.938,  26.736, "Safaga",   "left",   "center"),
    (32.716,  26.163, "Qena",     "right",  "center"),
]

# ── load shapefile (Egypt boundary only — neighbours/rivers data unavailable) ──
EGY_SHP = config.boundary_shapefile_path
print("Loading Egypt boundary shapefile …")
egypt = gpd.read_file(EGY_SHP)

# ── load PRISMA tif for inset FCC ─────────────────────────────────────────────
print("Loading PRISMA tif …")

# Pick the bands nearest 850/650/550 nm from the scene's own wavelength table.
import re
HDR = os.path.join(config.input_data_dir, "PRISMA_Rabshi_clip.hdr")
with open(HDR) as f:
    hdr_txt = f.read()
_wl_block = re.search(r"wavelength = \{([^}]*)\}", hdr_txt).group(1)
WAVELENGTHS = [float(x) for x in _wl_block.replace("\n", "").split(",") if x.strip()]

def _nearest_band(target_nm):
    return min(range(len(WAVELENGTHS)), key=lambda i: abs(WAVELENGTHS[i] - target_nm)) + 1  # 1-indexed

BAND_R, BAND_G, BAND_B = _nearest_band(850), _nearest_band(650), _nearest_band(550)
print(f"  FCC bands (1-indexed): R={BAND_R} ({WAVELENGTHS[BAND_R-1]:.1f} nm), "
      f"G={BAND_G} ({WAVELENGTHS[BAND_G-1]:.1f} nm), B={BAND_B} ({WAVELENGTHS[BAND_B-1]:.1f} nm)")

with rasterio.open(config.tifpath) as src:
    fcc_raw = src.read([BAND_R, BAND_G, BAND_B]).astype(np.float32)
    H_tif, W_tif = src.height, src.width
    tif_bounds_utm = src.bounds

fcc = np.zeros((H_tif, W_tif, 3), dtype=np.float32)
for i, band in enumerate(fcc_raw):
    valid = band[band > -9000.]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    fcc[:, :, i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)

with rasterio.open(config.train_mask_path) as src:
    TR = src.read(1).astype(np.int16)

# right-panel axes are drawn in lon/lat, so reuse the reprojected footprint bounds
tif_bounds = type("B", (), dict(left=fp_w, right=fp_e, bottom=fp_s, top=fp_n))

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 9))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(1, 2,
                      left=0.03, right=0.97,
                      top=0.91,  bottom=0.10,
                      wspace=0.04,
                      width_ratios=[1.15, 1])

ax_map  = fig.add_subplot(gs[0])
ax_fcc  = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL – regional map
# ═══════════════════════════════════════════════════════════════════════════════
ax_map.set_facecolor("#d0e8f5")

egypt.plot(ax=ax_map, color="#f5e6c8", edgecolor="#555555",
           linewidth=0.9, zorder=2)

ed_poly = plt.Polygon([(32.5, 22), (36.5, 22), (36.5, 30), (32.5, 30)],
                       closed=True, facecolor="#f0c080", alpha=0.18,
                       edgecolor="none", zorder=2, linewidth=0)
ax_map.add_patch(ed_poly)

ax_map.text(34.0, 29.5, "Sinai", fontsize=7, color="#666666",
            ha="center", style="italic", zorder=6)
ax_map.text(36.5, 26.0, "Red Sea", fontsize=8.5, color="#1a6e9c",
            ha="center", va="center", style="italic",
            rotation=-28, zorder=5)
ax_map.text(31.8, 29.5, "Mediterranean\nSea", fontsize=7.5, color="#1a6e9c",
            ha="center", va="center", style="italic", zorder=5)
ax_map.text(30.5, 25.0, "Nile\nValley", fontsize=7.5, color="#333333",
            ha="center", va="center", style="italic",
            rotation=10, zorder=5)
ax_map.text(34.3, 24.0, "Eastern Desert\n(Arabian-Nubian Shield)",
            fontsize=8, color="#8B5E3C", ha="center", va="center",
            style="italic", fontweight="bold", zorder=5)

country_labels = [
    (30.0,  23.5, "Egypt",        12, "bold",   "#5a4020"),
    (33.5,  17.5, "Sudan",        10, "normal", "#666666"),
    (26.0,  27.0, "Libya",        10, "normal", "#666666"),
    (38.5,  24.0, "Saudi Arabia", 9,  "normal", "#666666"),
    (35.0,  31.5, "Israel/\nJordan", 8, "normal", "#666666"),
]
for lon, lat, name, fs, fw, fc in country_labels:
    if MAP_W <= lon <= MAP_E and MAP_S <= lat <= MAP_N:
        ax_map.text(lon, lat, name, fontsize=fs, fontweight=fw, color=fc,
                    ha="center", va="center", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

for lon, lat, name, ha, va in CITIES:
    if MAP_W <= lon <= MAP_E and MAP_S <= lat <= MAP_N:
        ax_map.plot(lon, lat, "^", color="#333333", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=7)
        ax_map.text(lon + (0.25 if ha == "left" else -0.25), lat, name,
                    fontsize=7.5, ha=ha, va=va, color="#222222", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])

footprint_gdf.plot(ax=ax_map, facecolor="#E5393520", edgecolor="#E53935",
                   linewidth=2.5, zorder=8)
ax_map.text(fp_e + 0.12, fp_cy,
            "PRISMA\nScene", fontsize=8, color="#E53935", ha="left", va="center",
            fontweight="bold", zorder=9,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

ax_map.set_xlim(MAP_W, MAP_E)
ax_map.set_ylim(MAP_S, MAP_N)
ax_map.set_aspect("equal")

ax_map.set_xlabel("Longitude (°E)", fontsize=10)
ax_map.set_ylabel("Latitude (°N)", fontsize=10)
ax_map.set_xticks(np.arange(30, 38, 2))
ax_map.set_yticks(np.arange(22, 31, 2))
ax_map.tick_params(labelsize=8.5)
ax_map.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa", zorder=0)
ax_map.spines[["top","right","bottom","left"]].set_linewidth(1.2)

ax_map.annotate("", xy=(MAP_W + 0.4, MAP_N - 0.4),
                xytext=(MAP_W + 0.4, MAP_N - 1.5),
                arrowprops=dict(arrowstyle="-|>", color="black",
                                lw=1.5, mutation_scale=14),
                zorder=10)
ax_map.text(MAP_W + 0.4, MAP_N - 0.2, "N", fontsize=10, ha="center",
            va="bottom", fontweight="bold", zorder=10)

sb_x0, sb_y0 = MAP_W + 0.5, MAP_S + 0.5
sb_len = 2.0
ax_map.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
            color="black", linewidth=3, solid_capstyle="butt", zorder=10)
ax_map.plot([sb_x0, sb_x0 + sb_len / 2], [sb_y0, sb_y0],
            color="white", linewidth=3, solid_capstyle="butt", zorder=11)
for tick_x in [sb_x0, sb_x0 + sb_len / 2, sb_x0 + sb_len]:
    ax_map.plot([tick_x, tick_x], [sb_y0 - 0.08, sb_y0 + 0.08],
                color="black", linewidth=1, zorder=12)
ax_map.text(sb_x0, sb_y0 - 0.22, "0", ha="center", fontsize=7, zorder=12)
ax_map.text(sb_x0 + sb_len / 2, sb_y0 - 0.22, "100", ha="center", fontsize=7, zorder=12)
ax_map.text(sb_x0 + sb_len, sb_y0 - 0.22, "200 km", ha="center", fontsize=7, zorder=12)

ax_map.set_title(
    "(a) Regional setting – Eastern Desert of Egypt\n"
    "Major physiographic provinces and study-area location",
    fontsize=11, fontweight="bold", pad=6,
)

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL – PRISMA FCC + training samples
# ═══════════════════════════════════════════════════════════════════════════════
ax_fcc.imshow(fcc, interpolation="bilinear", aspect="auto",
              extent=[tif_bounds.left, tif_bounds.right,
                      tif_bounds.bottom, tif_bounds.top],
              origin="upper")

classes = list(range(1, len(CLASS_NAMES)))
for cls in classes:
    color = COLORS_NORM[cls]
    tr_rows, tr_cols = np.where(TR == cls)
    if len(tr_rows):
        lons = tif_bounds.left + (tr_cols + 0.5) * (tif_bounds.right - tif_bounds.left) / W_tif
        lats = tif_bounds.top  - (tr_rows + 0.5) * (tif_bounds.top - tif_bounds.bottom) / H_tif
        ax_fcc.scatter(lons, lats, c=[color], s=4, marker="o",
                       linewidths=0, alpha=0.80, zorder=3)

ax_fcc.set_xlim(tif_bounds.left, tif_bounds.right)
ax_fcc.set_ylim(tif_bounds.bottom, tif_bounds.top)

ax_fcc.set_xlabel("Longitude (°E)", fontsize=10)
ax_fcc.set_ylabel("Latitude (°N)", fontsize=10)
lon_ticks = np.round(np.linspace(tif_bounds.left, tif_bounds.right, 5), 2)
lat_ticks = np.round(np.linspace(tif_bounds.bottom, tif_bounds.top, 4), 2)
ax_fcc.set_xticks(lon_ticks)
ax_fcc.set_yticks(lat_ticks)
ax_fcc.tick_params(labelsize=8)
ax_fcc.grid(True, linestyle="--", alpha=0.25, color="#cccccc", zorder=1)
ax_fcc.spines[["top","right","bottom","left"]].set_linewidth(1.2)

ax_fcc.text(0.01, 0.01,
            f"PRISMA FCC  R: {WAVELENGTHS[BAND_R-1]:.0f} nm · "
            f"G: {WAVELENGTHS[BAND_G-1]:.0f} nm · B: {WAVELENGTHS[BAND_B-1]:.0f} nm",
            transform=ax_fcc.transAxes, fontsize=7.5, color="white",
            va="bottom", ha="left",
            bbox=dict(facecolor="black", alpha=0.45, pad=3, edgecolor="none"))

ax_fcc.set_title(
    "(b) PRISMA false-colour composite – Rabshi scene\n"
    "Training samples colour-coded by lithological class",
    fontsize=11, fontweight="bold", pad=6,
)

class_handles = [
    mpatches.Patch(facecolor=COLORS_NORM[c], edgecolor="black",
                   linewidth=0.6, label=CLASS_NAMES[c])
    for c in classes
]
fig.legend(handles=class_handles,
           loc="lower center", ncol=5, fontsize=8,
           framealpha=0.92, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01),
           bbox_transform=fig.transFigure)

fig.suptitle(
    "Figure 1 – Regional Geological Setting and Study Area: "
    "PRISMA Rabshi, Eastern Desert, Egypt",
    fontsize=13, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
