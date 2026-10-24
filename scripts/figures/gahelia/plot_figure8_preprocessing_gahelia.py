"""
Figure 8 (Bir Gahelia) – Preprocessing validation
(a) Band 50 (~907 nm, NIR) vs Band 120 (~1639 nm, SWIR) scatterplot, coloured by class
(b) LDA 2-D projection of all 178 bands, coloured by class
(c) Per-class mean spectra (168 spectral bands) with ±1 std envelopes,
    annotated with key absorption features

Adapted from plot_figure8_preprocessing.py (Aboreshead) for the Bir Gahelia
(PRISMA_Gahelia_Clipped) dataset: 10 classes instead of 12.
"""

import os, sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA_Gahelia_Clipped")
from projutils.config import config

EXP_ID = "experiments_004"
config.experiment_path = os.path.join(config.output_dir, EXP_ID)
config.new_final_class_labels = os.path.join(config.output_dir, EXP_ID, "new__newfinal_class_labels.tif")

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure8_preprocessing_validation.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata: read directly from the training shapefile (no hardcoding) ─
from projutils.class_scheme import load_class_scheme
CLASS_NAMES, SHORT_NAMES, COLORS_NORM, COLORS_RGB = load_class_scheme("PRISMA_Gahelia_Clipped")

vnir_all = np.linspace(400.0, 1010.0, 66)
swir_all = np.linspace(920.0, 2505.0, 173)
vnir_remove = [i-1 for i in [1, 2, 3, 4, 5]]
swir_remove = ([i-1 for i in range(3, 6)] +
               [i-1 for i in range(40, 57)] +
               [i-1 for i in range(86, 112)] +
               [i-1 for i in range(152, 172)])
vnir_wl = vnir_all[[i for i in range(66) if i not in vnir_remove]]
swir_wl = swir_all[[i for i in range(173) if i not in swir_remove]]
SPECTRAL_WL = np.concatenate([vnir_wl, swir_wl])
N_SPECTRAL   = len(SPECTRAL_WL)

print("Loading hyperspectral cube …")
with rasterio.open(config.tifpath) as src:
    raw = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)

print("Loading ground-truth labels …")
with rasterio.open(config.new_final_class_labels) as src:
    gt = src.read(1).astype(np.int16)

labeled_mask = gt > 0
X_all = raw[labeled_mask]
y_all = gt[labeled_mask]

valid = np.all(X_all > -9000., axis=1)
X_all = X_all[valid]
y_all = y_all[valid]
print(f"  Labeled pixels: {len(y_all)}")

print("Normalising …")
flat = raw.reshape(-1, raw.shape[2])
flat_valid = flat[np.all(flat > -9000., axis=1)]
p2  = np.percentile(flat_valid, 2,  axis=0)
p98 = np.percentile(flat_valid, 98, axis=0)
denom = np.where(p98 > p2, p98 - p2, 1.0)
X_norm = np.clip((X_all - p2) / denom, 0.0, 1.0)

classes_present = sorted(np.unique(y_all))

print("Fitting LDA …")
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_norm)
lda     = LinearDiscriminantAnalysis(n_components=2)
X_lda   = lda.fit_transform(X_sc, y_all)
ld1_var = lda.explained_variance_ratio_[0] * 100
ld2_var = lda.explained_variance_ratio_[1] * 100
print(f"  LD1 {ld1_var:.1f}%   LD2 {ld2_var:.1f}%")

fig = plt.figure(figsize=(19, 6.5))
gs  = fig.add_gridspec(1, 3, left=0.05, right=0.98,
                        top=0.88, bottom=0.22,
                        wspace=0.32,
                        width_ratios=[1, 1, 1.35])

B1_IDX, B2_IDX = 49, 119
WL_B1 = SPECTRAL_WL[B1_IDX]
WL_B2 = SPECTRAL_WL[B2_IDX]

ax_a = fig.add_subplot(gs[0])
for cls in classes_present:
    idx   = y_all == cls
    color = COLORS_NORM[cls]
    ax_a.scatter(X_norm[idx, B1_IDX], X_norm[idx, B2_IDX],
                 c=[color], s=7, alpha=0.45, linewidths=0.25,
                 edgecolors="none")
ax_a.set_xlabel(f"Band 50 ({WL_B1:.0f} nm, NIR)", fontsize=10)
ax_a.set_ylabel(f"Band 120 ({WL_B2:.0f} nm, SWIR)", fontsize=10)
ax_a.set_title("(a) Band 50 vs Band 120\ninter-class separability",
               fontsize=11, fontweight="bold", pad=5)
ax_a.set_xlim(-0.03, 1.03); ax_a.set_ylim(-0.03, 1.03)
ax_a.grid(True, alpha=0.25, linestyle="--")
ax_a.tick_params(labelsize=8.5)

ax_b = fig.add_subplot(gs[1])
for cls in classes_present:
    idx   = y_all == cls
    color = COLORS_NORM[cls]
    ax_b.scatter(X_lda[idx, 0], X_lda[idx, 1],
                 c=[color], s=7, alpha=0.45, linewidths=0.25,
                 edgecolors="none")
ax_b.set_xlabel(f"LD1 ({ld1_var:.1f}% variance)", fontsize=10)
ax_b.set_ylabel(f"LD2 ({ld2_var:.1f}% variance)", fontsize=10)
ax_b.set_title("(b) LDA 2-D projection\n(178 bands → 2 discriminants)",
               fontsize=11, fontweight="bold", pad=5)
ax_b.grid(True, alpha=0.25, linestyle="--")
ax_b.tick_params(labelsize=8.5)

ax_c = fig.add_subplot(gs[2])
wl = SPECTRAL_WL
for cls in classes_present:
    idx    = y_all == cls
    X_spec = X_norm[idx, :N_SPECTRAL]
    mu     = X_spec.mean(axis=0)
    sigma  = X_spec.std(axis=0)
    color  = COLORS_NORM[cls]
    ax_c.plot(wl, mu, color=color, linewidth=1.4)
    ax_c.fill_between(wl, mu - sigma, mu + sigma,
                      color=color, alpha=0.09)

ax_c.set_ylim(bottom=0)
ymax = ax_c.get_ylim()[1]
absorption_features = [
    (1000,  "Fe²⁺\n~1000 nm"),
    (2200,  "Kaolinite\n~2200 nm"),
    (2302,  "Serpentine\n~2300 nm"),
]
for feat_wl, label in absorption_features:
    if wl.min() <= feat_wl <= wl.max():
        ax_c.axvline(feat_wl, color="#555555", linewidth=0.85,
                     linestyle="--", alpha=0.7)
        ax_c.text(feat_wl + 18, ymax * 0.97, label,
                  fontsize=7.2, color="#444444", va="top",
                  linespacing=1.3)

ax_c.set_xlabel("Wavelength (nm)", fontsize=10)
ax_c.set_ylabel("Normalised Reflectance", fontsize=10)
ax_c.set_title("(c) Per-class mean spectra ± 1 std\n(168 spectral bands, 446–2505 nm)",
               fontsize=11, fontweight="bold", pad=5)
ax_c.set_xlim(wl.min(), wl.max())
ax_c.grid(True, alpha=0.25, linestyle="--")
ax_c.tick_params(labelsize=8.5)

handles = [
    mpatches.Patch(facecolor=COLORS_NORM[cls], edgecolor=COLORS_NORM[cls],
                   linewidth=0.6, label=CLASS_NAMES[cls])
    for cls in classes_present
]
fig.legend(handles=handles,
           loc="lower center", ncol=5, fontsize=8.5,
           framealpha=0.92, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Figure 8 – Preprocessing Validation · Bir Gahelia · experiments_004",
    fontsize=13, fontweight="bold", y=0.97,
)

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
plt.close()
