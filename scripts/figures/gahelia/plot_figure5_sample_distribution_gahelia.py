"""
Figure 5 (Bir Gahelia) – Training and test sample distribution
(a) Spatial map: training (circles) and test (crosses) overlaid on SWIR–NIR–Red FCC
(b) Bar chart of sample counts per class (train / test)
(c) Per-class mean spectra ± 1 std from training pixels, offset vertically for clarity

Adapted from plot_figure5_sample_distribution.py (Aboreshead) for the Bir Gahelia
(PRISMA_Gahelia_Clipped) dataset. Class names and colours are read directly from
the training shapefile (gahelia_class_scheme.py) — not hardcoded.
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
_cfg.reinit("PRISMA_Gahelia_Clipped")
from projutils.config import config

EXP_ID = "experiments_004"
config.experiment_path = os.path.join(config.output_dir, EXP_ID)

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure5_sample_distribution.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── class metadata: read directly from the training shapefile (no hardcoding) ─
from projutils.class_scheme import load_class_scheme
CLASS_NAMES, SHORT_NAMES, COLORS_NORM, COLORS_RGB = load_class_scheme("PRISMA_Gahelia_Clipped")

# ── wavelength array (168 spectral bands) ─────────────────────────────────────
vnir_all = np.linspace(400.0, 1010.0, 66)
swir_all = np.linspace(920.0,  2505.0, 173)
vnir_rm  = [i-1 for i in [1,2,3,4,5]]
swir_rm  = ([i-1 for i in range(3,6)] + [i-1 for i in range(40,57)] +
             [i-1 for i in range(86,112)] + [i-1 for i in range(152,172)])
SPECTRAL_WL = np.concatenate([vnir_all[[i for i in range(66) if i not in vnir_rm]],
                               swir_all[[i for i in range(173) if i not in swir_rm]]])
N_SPEC = len(SPECTRAL_WL)

def wl_to_idx(target_nm):
    return int(np.argmin(np.abs(SPECTRAL_WL - target_nm)))

IDX_R = wl_to_idx(2200)
IDX_G = wl_to_idx(907)
IDX_B = wl_to_idx(650)

print("Loading data …")
with rasterio.open(config.tifpath) as src:
    cube = src.read([IDX_R+1, IDX_G+1, IDX_B+1]).astype(np.float32)
    H, W = src.height, src.width

with rasterio.open(config.train_mask_path) as src:
    TR = src.read(1).astype(np.int16)
with rasterio.open(config.test_mask_path) as src:
    TE = src.read(1).astype(np.int16)
with rasterio.open(config.tifpath) as src:
    cube_all = np.transpose(src.read(), (1, 2, 0)).astype(np.float32)

fcc = np.zeros((H, W, 3), dtype=np.float32)
for i, band in enumerate(cube):
    valid = band[band > -9000.]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    fcc[:, :, i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)

classes = list(range(1, 11))
tr_counts = [np.sum(TR == c) for c in classes]
te_counts = [np.sum(TE == c) for c in classes]

fig = plt.figure(figsize=(20, 7))
gs  = fig.add_gridspec(1, 3,
                        left=0.04, right=0.98,
                        top=0.88,  bottom=0.13,
                        wspace=0.30,
                        width_ratios=[1.35, 1, 1.2])

ax_a = fig.add_subplot(gs[0])
ax_a.imshow(fcc, interpolation="bilinear", aspect="equal")

for cls in classes:
    color = COLORS_NORM[cls]
    tr_rows, tr_cols = np.where(TR == cls)
    if len(tr_rows):
        ax_a.scatter(tr_cols, tr_rows, c=[color], s=4, marker="o",
                     linewidths=0, alpha=0.85, zorder=3)
    te_rows, te_cols = np.where(TE == cls)
    if len(te_rows):
        ax_a.scatter(te_cols, te_rows, c=[color], s=5, marker="x",
                     linewidths=0.7, alpha=0.65, zorder=2)

ax_a.set_xlim(0, W); ax_a.set_ylim(H, 0)
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.set_title("(a) Spatial sample distribution\n(SWIR–NIR–Red false-colour composite)",
               fontsize=11, fontweight="bold", pad=5)

legend_markers = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="dimgray",
           markersize=6, label="Training samples"),
    Line2D([0],[0], marker="x", color="dimgray",
           markersize=6, markeredgewidth=1.2, label="Test samples"),
]
ax_a.legend(handles=legend_markers, loc="upper right", fontsize=8,
            framealpha=0.85, edgecolor="#aaaaaa")

ax_b = fig.add_subplot(gs[1])
y_pos   = np.arange(len(classes))
bar_h   = 0.38
bars_tr = ax_b.barh(y_pos + bar_h/2, tr_counts, height=bar_h,
                    color="#2196F3", label="Training", alpha=0.88)
bars_te = ax_b.barh(y_pos - bar_h/2, te_counts, height=bar_h,
                    color="#F44336", label="Test",     alpha=0.88)

for bar, val in zip(bars_tr, tr_counts):
    ax_b.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
              str(val), va="center", fontsize=7.5, color="#1565C0")
for bar, val in zip(bars_te, te_counts):
    ax_b.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
              str(val), va="center", fontsize=7.5, color="#B71C1C")

ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([SHORT_NAMES[c] for c in classes], fontsize=8.5)
ax_b.set_xlabel("Number of samples", fontsize=10)
ax_b.set_title("(b) Sample counts per class\n(train vs. test)",
               fontsize=11, fontweight="bold", pad=5)
ax_b.legend(fontsize=9, loc="lower right")
ax_b.set_xlim(0, max(te_counts) * 1.22)
ax_b.grid(True, axis="x", alpha=0.25, linestyle="--")
ax_b.tick_params(axis="y", labelsize=8.5)
ax_b.invert_yaxis()

ax_c = fig.add_subplot(gs[2])
wl = SPECTRAL_WL
OFFSET = 0.22

flat_all = cube_all.reshape(-1, cube_all.shape[2])
valid_mask = np.all(flat_all > -9000., axis=1)
flat_valid = flat_all[valid_mask]
p2  = np.percentile(flat_valid, 2,  axis=0)
p98 = np.percentile(flat_valid, 98, axis=0)
denom = np.where(p98 > p2, p98 - p2, 1.0)

for i, cls in enumerate(classes):
    color  = COLORS_NORM[cls]
    offset = i * OFFSET

    tr_rows, tr_cols = np.where(TR == cls)
    if len(tr_rows) == 0:
        continue
    X_cls = cube_all[tr_rows, tr_cols, :N_SPEC].astype(float)
    X_cls = np.clip((X_cls - p2[:N_SPEC]) / denom[:N_SPEC], 0, 1)
    mu    = X_cls.mean(axis=0) + offset
    sigma = X_cls.std(axis=0)

    ax_c.plot(wl, mu, color=color, linewidth=1.3, zorder=3)
    ax_c.fill_between(wl, mu - sigma, mu + sigma,
                      color=color, alpha=0.12, zorder=2)
    ax_c.text(wl[-1] + 18, mu[-1], SHORT_NAMES[cls],
              va="center", fontsize=7.2, color=color,
              fontweight="bold")

abs_feats = [(1000, "Fe²⁺"), (2200, "Kaolinite"), (2302, "Mg-OH")]
for feat_wl, label in abs_feats:
    ax_c.axvline(feat_wl, color="#666666", linewidth=0.8,
                 linestyle="--", alpha=0.6)
    ax_c.text(feat_wl + 15, (len(classes)-0.5)*OFFSET + 0.07,
              label, fontsize=7, color="#555555", va="bottom")

ax_c.set_xlabel("Wavelength (nm)", fontsize=10)
ax_c.set_ylabel("Normalised Reflectance (offset for clarity)", fontsize=9.5)
ax_c.set_title("(c) Per-class training spectra ± 1 std\n(offset vertically for clarity)",
               fontsize=11, fontweight="bold", pad=5)
ax_c.set_xlim(wl[0], wl[-1] + 160)
ax_c.set_ylim(-0.05, (len(classes)) * OFFSET + 0.15)
ax_c.set_yticks([])
ax_c.grid(True, axis="x", alpha=0.2, linestyle="--")
ax_c.tick_params(axis="x", labelsize=8.5)

handles = [mpatches.Patch(facecolor=COLORS_NORM[c], edgecolor=COLORS_NORM[c],
                           linewidth=0.5, label=SHORT_NAMES[c])
           for c in classes]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
           framealpha=0.92, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(
    "Figure 5 – Training and Test Sample Distribution · Bir Gahelia · experiments_004",
    fontsize=13, fontweight="bold", y=0.97,
)

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
