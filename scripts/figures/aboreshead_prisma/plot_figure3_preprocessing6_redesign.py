"""
Figure 3 – PRISMA preprocessing: band spectra + RGB composites
(a) VNIR mean reflectance before/after bad-band removal (removed bands in grey)
(b) SWIR equivalent
(c) Three RGB composites stacked:
    SWIR FCC-1 (max inter-class separability) /
    SWIR FCC-2 (mineralogical targeting: Al-OH / Mg-OH / SWIR1) /
    MNF false colour
"""

import os, sys
import numpy as np
import h5py
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA")
from projutils.config import config

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure3_preprocessing6_redesign.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

HE5 = config.scene_path
NODATA = -9999.0

# ── actual wavelengths from HDF5 ──────────────────────────────────────────────
with h5py.File(HE5, "r") as f:
    vnir_wl_all = f.attrs["List_Cw_Vnir"][:]   # 66 bands, descending
    swir_wl_all = f.attrs["List_Cw_Swir"][:]   # 173 bands, descending

# config bad-band indices (0-indexed)
VNIR_RM = [0, 1, 2, 3, 4]
SWIR_RM = ([2, 3, 4] + list(range(39, 56)) + list(range(85, 111)) + list(range(151, 171)))

vnir_keep = [i for i in range(66)  if i not in VNIR_RM]   # 61
swir_keep = [i for i in range(173) if i not in SWIR_RM]   # 107

vnir_kept_wl = vnir_wl_all[vnir_keep]   # 956→407 nm
swir_kept_wl = swir_wl_all[swir_keep]   # 2497→0 nm (last 2 are 0-wl artefacts)

# Full wavelength array for tif bands 0-167 (same ordering as the tif)
SPECTRAL_WL = np.concatenate([vnir_kept_wl, swir_kept_wl])   # 168 values

def closest_band(target_nm):
    valid = SPECTRAL_WL > 0
    diffs = np.where(valid, np.abs(SPECTRAL_WL - target_nm), np.inf)
    idx = int(np.argmin(diffs))
    return idx, SPECTRAL_WL[idx]

# ── removed-band wavelength REGIONS (for grey shading) ───────────────────────
# VNIR: removed non-zero bands (967–977 nm)
vnir_rm_wl = vnir_wl_all[VNIR_RM]
vnir_rm_wl = vnir_rm_wl[vnir_rm_wl > 0]   # [967.0, 977.4] nm

# SWIR removed regions (contiguous groups):
swir_removed_groups = {
    "Top-edge noise\n(2470–2484 nm)":   (2470, 2484),
    "CO₂ absorption\n(2086–2215 nm)":   (2086, 2215),
    "H₂O absorption\n(1576–1822 nm)":   (1576, 1822),
    "VNIR–SWIR overlap\n(943–1131 nm)": (943,  1131),
}

# ── load processed tif ────────────────────────────────────────────────────────
print("Loading processed tif …")
with rasterio.open(config.tifpath) as src:
    cube = src.read().astype(np.float32)   # (178, H, W)
    H, W = src.height, src.width

spectral = cube[:168]    # first 168 bands = spectral
mnf      = cube[168:]    # last 10 = MNF

# mean reflectance per band from valid pixels
mean_ref = np.zeros(168)
for b in range(168):
    band = spectral[b]
    valid = band[band > NODATA + 1]
    mean_ref[b] = valid.mean() if len(valid) else np.nan

# ── FCC stretch helper ────────────────────────────────────────────────────────
def make_rgb(band_r, band_g, band_b, plo=2, phi=98):
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i, b in enumerate([band_r, band_g, band_b]):
        valid = b[b > NODATA + 1]
        lo, hi = np.percentile(valid, plo), np.percentile(valid, phi)
        rgb[:, :, i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb

# ── SWIR band validation and selection ────────────────────────────────────────
# Uses ground-truth labels to compute between-class variance per band, then
# picks the most discriminative band from each atmospheric transmission window.
# Zero-wavelength artefact bands (tif[166,167]) are excluded by the WL > 0 guard.
#
# Three SWIR windows (verified against actual HDF5 wavelengths):
#   W1: tif[ 61.. 96] → 2222–2497 nm  (36 bands)
#   W2: tif[ 97..125] → 1832–2078 nm  (29 bands, after CO₂ gap)
#   W3: tif[126..165] → 1142–1565 nm  (40 bands, after H₂O gap)
print("Selecting optimal SWIR bands …")
with rasterio.open(config.new_final_class_labels) as src:
    gt_labels = src.read(1).astype(np.int16)

gt_flat = gt_labels.ravel()
valid_classes = np.unique(gt_flat[gt_flat > 0])

SWIR_WINDOWS = {"W1": range(61, 97), "W2": range(97, 126), "W3": range(126, 166)}

def rank_swir_bands(band_indices):
    """Return list of (tif_idx, wl_nm, between_var) sorted by decreasing BetwnVar."""
    results = []
    for b in band_indices:
        wl = SPECTRAL_WL[b]
        if wl <= 0:                          # skip zero-wl artefact bands
            continue
        arr   = spectral[b].ravel()
        valid = arr > NODATA + 1
        vals  = arr[valid]
        if len(vals) < 100:
            continue
        cls_means = [vals[((gt_flat == c) & valid)].mean()
                     for c in valid_classes
                     if ((gt_flat == c) & valid).sum() > 5]
        bvar = np.std(cls_means) if len(cls_means) > 1 else 0.0
        results.append((b, wl, bvar))
    results.sort(key=lambda x: -x[2])
    return results

def closest_in_window(ranked_list, target_nm):
    """From an already-ranked list, return the entry whose wavelength is nearest target_nm."""
    return min(ranked_list, key=lambda x: abs(x[1] - target_nm))

ranked = {name: rank_swir_bands(rng) for name, rng in SWIR_WINDOWS.items()}

# ── FCC #1: maximum inter-class separability (best band per window) ────────────
b1_r, w1_r, bv1_r = ranked["W1"][0]   # → 2283.5 nm
b1_g, w1_g, bv1_g = ranked["W2"][0]   # → 2061.4 nm
b1_b, w1_b, bv1_b = ranked["W3"][0]   # → 1554.8 nm
fcc_swir1 = make_rgb(spectral[b1_r], spectral[b1_g], spectral[b1_b])
print(f"  FCC-1 (max separability):  R={w1_r:.0f} nm [tif{b1_r}, BVar={bv1_r:.0f}]"
      f"  G={w1_g:.0f} nm [tif{b1_g}, BVar={bv1_g:.0f}]"
      f"  B={w1_b:.0f} nm [tif{b1_b}, BVar={bv1_b:.0f}]")

# ── FCC #2: mineralogical targeting ──────────────────────────────────────────
# R → closest to 2222 nm in W1: Al-OH clay minerals (kaolinite, illite)
#          → alteration zones appear bright red
# G → closest to 2298 nm in W1: Mg-OH / serpentine feature
#          → serpentinite/gabbro differentiated from felsic units
# B → best W3 band farthest from FCC-1 B: shorter SWIR1 (iron-oxide sensitivity)
b2_r, w2_r, bv2_r = closest_in_window(ranked["W1"], 2222)   # Al-OH
b2_g, w2_g, bv2_g = closest_in_window(ranked["W1"], 2298)   # Mg-OH / serpentine
b2_b, w2_b, bv2_b = closest_in_window(ranked["W3"], 1274)   # SWIR1 short edge
fcc_swir2 = make_rgb(spectral[b2_r], spectral[b2_g], spectral[b2_b])
print(f"  FCC-2 (mineralogical):     R={w2_r:.0f} nm [tif{b2_r}, BVar={bv2_r:.0f}]"
      f"  G={w2_g:.0f} nm [tif{b2_g}, BVar={bv2_g:.0f}]"
      f"  B={w2_b:.0f} nm [tif{b2_b}, BVar={bv2_b:.0f}]")

# ── FCC #3: MNF false colour (MNF1 / MNF2 / MNF3) ────────────────────────────
def stretch_mnf(b, plo=2, phi=98):
    valid = b[b > NODATA + 1]
    lo, hi = np.percentile(valid, plo), np.percentile(valid, phi)
    out = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    out[b <= NODATA + 1] = 0
    return out

fcc_mnf = np.stack([stretch_mnf(mnf[0]),
                    stretch_mnf(mnf[1]),
                    stretch_mnf(mnf[2])], axis=2)

# ── figure layout ─────────────────────────────────────────────────────────────
# Figsize scaled up from the original (20, 11) while preserving the same
# width:height ratio, so the enlarged output is not stretched. The three
# right-column composites use aspect="equal" below (not "auto") so the
# satellite imagery keeps the raster's true pixel aspect ratio.
fig = plt.figure(figsize=(26, 14.3))
outer = gridspec.GridSpec(1, 2, left=0.05, right=0.98,
                          top=0.92, bottom=0.07,
                          wspace=0.28, width_ratios=[1, 1.05])

# left column: two spectral panels stacked
left_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                           hspace=0.42)
ax_a = fig.add_subplot(left_gs[0])
ax_b = fig.add_subplot(left_gs[1])

# right column: three image panels stacked
right_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1],
                                            hspace=0.06)
ax_c1 = fig.add_subplot(right_gs[0])
ax_c2 = fig.add_subplot(right_gs[1])
ax_c3 = fig.add_subplot(right_gs[2])

# ─ panel (a): VNIR spectrum (sorted ascending for display) ───────────────────
vnir_mask = (SPECTRAL_WL >= 400) & (SPECTRAL_WL <= 1010) & (SPECTRAL_WL > 0)
vnir_idx  = np.where(vnir_mask)[0]
sort_v    = np.argsort(SPECTRAL_WL[vnir_idx])
vnir_wl_plot   = SPECTRAL_WL[vnir_idx][sort_v]
vnir_ref_plot  = mean_ref[vnir_idx][sort_v]

# — grey shading for removed VNIR bands —
ax_a.axvspan(min(vnir_rm_wl) - 5, max(vnir_rm_wl) + 5,
             color="lightgrey", alpha=0.75, zorder=1,
             label="Removed bands\n(noise edge)")

ax_a.plot(vnir_wl_plot, vnir_ref_plot,
          color="#1565C0", linewidth=1.8, zorder=3, label="Kept bands (after removal)")

# mark removed wavelengths as vertical ticks
for wl in vnir_rm_wl:
    ax_a.axvline(wl, color="dimgray", linewidth=0.8, linestyle=":", alpha=0.7)
ax_a.text(max(vnir_rm_wl) + 8, ax_a.get_ylim()[1] if ax_a.get_ylim()[1] else 0.5,
          "Bands 4–5\nremoved", fontsize=7.5, color="dimgray", va="top")

ax_a.set_xlim(400, 1010)
ax_a.set_xlabel("Wavelength (nm)", fontsize=10)
ax_a.set_ylabel("Mean Reflectance (a.u.)", fontsize=10)
ax_a.set_title("(a) VNIR mean reflectance — bad-band removal\n"
               "(66 bands, 5 removed → 61 kept; 407–956 nm)",
               fontsize=10.5, fontweight="bold", pad=5)
ax_a.legend(fontsize=8, loc="upper left", framealpha=0.88)
ax_a.grid(True, alpha=0.22, linestyle="--")
ax_a.tick_params(labelsize=8.5)

# ─ panel (b): SWIR spectrum ───────────────────────────────────────────────────
swir_mask = (SPECTRAL_WL >= 900) & (SPECTRAL_WL <= 2510) & (SPECTRAL_WL > 0)
swir_idx  = np.where(swir_mask)[0]
sort_s    = np.argsort(SPECTRAL_WL[swir_idx])
swir_wl_plot   = SPECTRAL_WL[swir_idx][sort_s]
swir_ref_plot  = mean_ref[swir_idx][sort_s]

# — grey shading for each removed SWIR group —
shade_colors = ["#d4d4d4", "#c0c0c0", "#b0b0b0", "#a0a0a0"]
for (label, (lo, hi)), shade in zip(swir_removed_groups.items(), shade_colors):
    ax_b.axvspan(lo, hi, color=shade, alpha=0.72, zorder=1)
    mid = (lo + hi) / 2
    ax_b.text(mid, 0.97, label, transform=ax_b.get_xaxis_transform(),
              ha="center", va="top", fontsize=6.5, color="#444444",
              linespacing=1.3)

ax_b.plot(swir_wl_plot, swir_ref_plot,
          color="#C62828", linewidth=1.8, zorder=3, label="Kept bands (after removal)")

ax_b.set_xlim(900, 2510)
ax_b.set_xlabel("Wavelength (nm)", fontsize=10)
ax_b.set_ylabel("Mean Reflectance (a.u.)", fontsize=10)
ax_b.set_title("(b) SWIR mean reflectance — bad-band removal\n"
               "(173 bands, 66 removed → 107 kept; 1009–2497 nm)",
               fontsize=10.5, fontweight="bold", pad=5)
grey_patch = mpatches.Patch(color="#c0c0c0", alpha=0.72, label="Removed bands\n(atmospheric/noise)")
kept_line  = plt.Line2D([0],[0], color="#C62828", linewidth=1.8, label="Kept bands (after removal)")
ax_b.legend(handles=[grey_patch, kept_line], fontsize=8, loc="upper right", framealpha=0.88)
ax_b.grid(True, alpha=0.22, linestyle="--")
ax_b.tick_params(labelsize=8.5)

# ─ panel (c): three SWIR/MNF composites ──────────────────────────────────────
for ax, img, title in [
    (ax_c1, fcc_swir1,
     f"(c-top) SWIR FCC-1 – max separability"
     f"  R:{w1_r:.0f} / G:{w1_g:.0f} / B:{w1_b:.0f} nm"),
    (ax_c2, fcc_swir2,
     f"(c-mid) SWIR FCC-2 – Al-OH / Mg-OH / SWIR1"
     f"  R:{w2_r:.0f} / G:{w2_g:.0f} / B:{w2_b:.0f} nm"),
    (ax_c3, fcc_mnf,
     "(c-bot) MNF false colour  R:MNF₁ / G:MNF₂ / B:MNF₃"),
]:
    ax.imshow(img, interpolation="bilinear", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=4)

# ── suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 3 – PRISMA Preprocessing: Band Quality & RGB Composites · Aboreshead · experiments_008",
    fontsize=12, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
