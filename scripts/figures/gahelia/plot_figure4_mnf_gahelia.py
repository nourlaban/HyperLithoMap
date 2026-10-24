"""
Figure 4 (Bir Gahelia) – MNF transform validation
(a) MNF component 1 image – highest SNR, smooth geology-aligned structure
(b) MNF component 10 image – increasing noise
(c) Scree plot of all 168 MNF eigenvalues, elbow annotated at component 10

Adapted from plot_figure4_mnf.py (Aboreshead) for the Bir Gahelia
(PRISMA_Gahelia_Clipped) dataset. Fully config-driven — no region-specific changes
needed beyond the satellite name, experiment id override, and title.
"""

import os, sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA_Gahelia_Clipped")
from projutils.config import config

EXP_ID = "experiments_004"
config.experiment_path = os.path.join(config.output_dir, EXP_ID)

OUT_FILE = os.path.join(config.experiment_path, "figures", "figure4_mnf_abc.png")
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

N_MNF      = 10
N_SPEC     = 168
NODATA_VAL = -9999.0

vnir_all = np.linspace(400.0, 1010.0, 66)
swir_all = np.linspace(920.0, 2505.0, 173)
vnir_rm  = [i-1 for i in [1, 2, 3, 4, 5]]
swir_rm  = ([i-1 for i in range(3, 6)] + [i-1 for i in range(40, 57)] +
             [i-1 for i in range(86, 112)] + [i-1 for i in range(152, 172)])
SPECTRAL_WL = np.concatenate([
    vnir_all[[i for i in range(66) if i not in vnir_rm]],
    swir_all[[i for i in range(173) if i not in swir_rm]],
])

print("Loading cube …")
with rasterio.open(config.tifpath) as src:
    all_bands = src.read().astype(np.float32)
    H, W = src.height, src.width

spectral = all_bands[:N_SPEC]
mnf_imgs = all_bands[N_SPEC:N_SPEC + N_MNF]

data_hwb = np.transpose(spectral, (1, 2, 0))

print("Recomputing MNF transform …")
valid_2d = ~np.all(data_hwb == NODATA_VAL, axis=2)

diff = data_hwb[:, 1:, :] - data_hwb[:, :-1, :]
both_valid = valid_2d[:, 1:] & valid_2d[:, :-1]
valid_diff = diff[both_valid].astype(np.float64)
noise_cov  = np.cov(valid_diff.T) / 2.0

evals_n, evecs_n = np.linalg.eigh(noise_cov)
evals_n = np.clip(evals_n, 1e-10, None)
A = evecs_n @ np.diag(evals_n ** -0.5)

flat       = data_hwb.reshape(-1, N_SPEC).astype(np.float64)
valid_mask = valid_2d.ravel()
valid_data = flat[valid_mask]

rng = np.random.default_rng(42)
idx = rng.choice(len(valid_data), min(100_000, len(valid_data)), replace=False)
whitened_sample = valid_data[idx] @ A
sig_cov = np.cov(whitened_sample.T)

evals_s, evecs_s = np.linalg.eigh(sig_cov)
order = np.argsort(evals_s)[::-1]
evals_sorted = evals_s[order]
evecs_sorted  = evecs_s[:, order]

mnf_T = A @ evecs_sorted[:, :N_MNF]
mean_spec = valid_data.mean(axis=0)

def stretch(img, plo=2, phi=98, nodata=NODATA_VAL):
    valid = img[img > nodata - 1]
    lo, hi = np.percentile(valid, plo), np.percentile(valid, phi)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

fig = plt.figure(figsize=(19, 7))
gs  = fig.add_gridspec(
    1, 3,
    left=0.05, right=0.97,
    top=0.88,  bottom=0.10,
    wspace=0.32,
    width_ratios=[1, 1, 1.3],
)

ax_a = fig.add_subplot(gs[0, 0])
disp1 = stretch(mnf_imgs[0])
disp1[mnf_imgs[0] <= NODATA_VAL + 1] = np.nan
im1 = ax_a.imshow(disp1, cmap="gray", interpolation="bilinear",
                  vmin=0, vmax=1, aspect="equal")
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.set_title("(a) MNF Component 1\n(highest SNR — geology-aligned structure)",
               fontsize=11, fontweight="bold", pad=5)
divider = make_axes_locatable(ax_a)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im1, cax=cax, label="Normalised value")
cax.tick_params(labelsize=8)

ax_b = fig.add_subplot(gs[0, 1])
disp10 = stretch(mnf_imgs[9])
disp10[mnf_imgs[9] <= NODATA_VAL + 1] = np.nan
im2 = ax_b.imshow(disp10, cmap="gray", interpolation="bilinear",
                  vmin=0, vmax=1, aspect="equal")
ax_b.set_xticks([]); ax_b.set_yticks([])
ax_b.set_title("(b) MNF Component 10\n(lowest retained SNR — increasing noise)",
               fontsize=11, fontweight="bold", pad=5)
divider2 = make_axes_locatable(ax_b)
cax2 = divider2.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im2, cax=cax2, label="Normalised value")
cax2.tick_params(labelsize=8)

ax_c = fig.add_subplot(gs[0, 2])
n_show = 40
comp_x = np.arange(1, n_show + 1)
ax_c.plot(comp_x, evals_sorted[:n_show],
          color="#1565C0", linewidth=2, marker="o",
          markersize=4.5, markerfacecolor="white", markeredgewidth=1.4,
          zorder=3)
ax_c.scatter(comp_x[:N_MNF], evals_sorted[:N_MNF],
             color="#1565C0", s=35, zorder=4)

elbow_val = evals_sorted[N_MNF - 1]
ax_c.axvline(N_MNF, color="#F44336", linewidth=1.4,
             linestyle="--", alpha=0.8, zorder=2)
ax_c.annotate(f"Elbow at\ncomponent {N_MNF}\n(eigenvalue = {elbow_val:.2f})",
              xy=(N_MNF, elbow_val),
              xytext=(N_MNF + 4, elbow_val + (evals_sorted[0] - evals_sorted[N_MNF-1]) * 0.35),
              fontsize=8.5, color="#C62828",
              arrowprops=dict(arrowstyle="->", color="#C62828",
                              connectionstyle="arc3,rad=-0.2"),
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                        edgecolor="#F44336", alpha=0.9))

ax_c.axvspan(0.5, N_MNF + 0.5, alpha=0.08, color="#1565C0", zorder=1)
ax_c.text(N_MNF / 2, ax_c.get_ylim()[0] if ax_c.get_ylim()[0] > 0 else 0,
          "retained", ha="center", va="bottom", fontsize=8,
          color="#1565C0", style="italic")

ax_c.set_xlabel("MNF Component", fontsize=10)
ax_c.set_ylabel("Eigenvalue (signal-to-noise ratio)", fontsize=10)
ax_c.set_title("(c) MNF Scree Plot\n(first 40 of 168 components shown)",
               fontsize=11, fontweight="bold", pad=5)
ax_c.set_xlim(0.5, n_show + 0.5)
ax_c.grid(True, alpha=0.25, linestyle="--")
ax_c.tick_params(labelsize=9)
ax_c.xaxis.set_major_locator(mticker.MultipleLocator(5))

fig.suptitle(
    "Figure 4 – MNF Transform Validation · Bir Gahelia · experiments_004",
    fontsize=13, fontweight="bold", y=0.975,
)

plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.close()
