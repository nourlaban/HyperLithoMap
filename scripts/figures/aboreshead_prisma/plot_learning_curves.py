"""
Plot training learning curves for all four models (experiments_008, PRISMA Aboreshead).

- RF        : OA vs number of trees (staged_predict, 1–10 trees)
- XGBoost   : OA vs boosting round  (200 rounds, eval_set)
- ViT       : OA vs epoch           (loaded from saved history)
- S²Mamba   : OA vs epoch           (loaded from saved history)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import rasterio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from projutils.config import config as _cfg
_cfg.reinit("PRISMA")
from projutils.config import config

OUT_DIR  = os.path.join(config.save_model_path, "..", "figures")
OUT_FILE = os.path.join(OUT_DIR, "learning_curves.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette (consistent with paper) ────────────────────────────────────
C_RF    = "#2196F3"   # blue
C_XGB   = "#FF9800"   # orange
C_VIT   = "#9C27B0"   # purple
C_MAMBA = "#4CAF50"   # green

# ── helpers ───────────────────────────────────────────────────────────────────

def load_data():
    with rasterio.open(config.tifpath) as s: raw = np.transpose(s.read(), (1,2,0))
    with rasterio.open(config.train_mask_path) as s: TR = s.read(1)
    with rasterio.open(config.test_mask_path)  as s: TE = s.read(1)
    out = np.zeros_like(raw, dtype=float)
    for i in range(raw.shape[2]):
        b = raw[:,:,i]; v = b[b != -9999.]
        if len(v) and v.max() > v.min():
            out[:,:,i] = np.where(b!=-9999., (b-v.min())/(v.max()-v.min()), 0.)
    x_tr = out[TR>0]; y_tr = TR[TR>0]
    x_te = out[TE>0]; y_te = TE[TE>0]
    return x_tr, y_tr, x_te, y_te


def read_history(path):
    epochs, oas = [], []
    with open(path) as f:
        next(f)
        for line in f:
            if line.startswith("best"): break
            ep, oa = line.strip().split(",")
            epochs.append(int(ep)); oas.append(float(oa)*100)
    return np.array(epochs), np.array(oas)


# ── RF learning curve (staged_predict) ───────────────────────────────────────

def rf_curve(x_tr, y_tr, x_te, y_te):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    print("[RF] Computing per-tree learning curve …")
    rf = RandomForestClassifier(n_estimators=10, random_state=42, warm_start=True)
    oas = []
    for n in range(1, 11):
        rf.set_params(n_estimators=n)
        rf.fit(x_tr, y_tr)
        oas.append(accuracy_score(y_te, rf.predict(x_te)) * 100)
    return np.arange(1, 11), np.array(oas)


# ── XGBoost learning curve (iteration_range) ──────────────────────────────────

def xgb_curve(x_tr, y_tr, x_te, y_te):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    print("[XGB] Computing boosting round learning curve …")
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb.fit(x_tr, y_tr_enc, verbose=False)
    oas = []
    for n in range(1, 201):
        pred_enc = xgb.predict(x_te, iteration_range=(0, n))
        oas.append(accuracy_score(y_te, le.inverse_transform(pred_enc)) * 100)
    return np.arange(1, 201), np.array(oas)


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(rf_x, rf_y, xgb_x, xgb_y, vit_x, vit_y, mamba_x, mamba_y):

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Learning Curves — PRISMA Aboreshead (experiments_008)",
                 fontsize=14, fontweight="bold", y=1.01)

    panels = [
        (axes[0,0], rf_x,    rf_y,    C_RF,    "Random Forest",
         "Number of Trees", 1, 10),
        (axes[0,1], xgb_x,   xgb_y,   C_XGB,   "XGBoost",
         "Boosting Round", 1, 200),
        (axes[1,0], vit_x,   vit_y,   C_VIT,   "SpectralFormer (ViT)",
         "Epoch", 1, 200),
        (axes[1,1], mamba_x, mamba_y, C_MAMBA, "S²Mamba",
         "Epoch", 0, 199),
    ]

    for ax, x, y, col, title, xlabel, xmin, xmax in panels:
        ax.plot(x, y, color=col, linewidth=2, alpha=0.85)
        ax.fill_between(x, y, y.min(), alpha=0.08, color=col)

        best_idx = np.argmax(y)
        ax.scatter(x[best_idx], y[best_idx], color=col, s=80, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(f"Best: {y[best_idx]:.2f}%\n(iter {x[best_idx]})",
                    xy=(x[best_idx], y[best_idx]),
                    xytext=(8, -18), textcoords="offset points",
                    fontsize=8.5, color=col, fontweight="bold")

        # threshold lines
        for thresh, ls in [(90, ":"), (95, "--"), (98, "-.")]:
            if y.max() >= thresh:
                ax.axhline(thresh, color="gray", linewidth=0.7,
                           linestyle=ls, alpha=0.6)
                ax.text(xmax*0.97, thresh+0.3, f"{thresh}%",
                        ha="right", va="bottom", fontsize=7.5, color="gray")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(max(y.min()-2, 0), min(y.max()+2, 101))
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
    print(f"\nSaved: {OUT_FILE}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    x_tr, y_tr, x_te, y_te = load_data()

    rf_x,    rf_y    = rf_curve(x_tr, y_tr, x_te, y_te)
    xgb_x,   xgb_y   = xgb_curve(x_tr, y_tr, x_te, y_te)

    mamba_hist = os.path.join(config.save_model_path,
        "mamba_model_evaluation", "mamba_epoch_history.txt")
    vit_hist   = os.path.join(config.save_model_path,
        "VITransformer_model_evaluation", "vit_epoch_history.txt")

    mamba_x, mamba_y = read_history(mamba_hist)
    vit_x,   vit_y   = read_history(vit_hist)
    print(f"[ViT]   {len(vit_x)} checkpoints loaded")
    print(f"[Mamba] {len(mamba_x)} checkpoints loaded")

    plot(rf_x, rf_y, xgb_x, xgb_y, vit_x, vit_y, mamba_x, mamba_y)


if __name__ == "__main__":
    main()
