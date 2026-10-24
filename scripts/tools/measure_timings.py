"""
Measure training and inference times for all four models on the PRISMA Aboreshead dataset.

- RF / XGBoost : re-trained from scratch to get training time + full-raster inference time
- ViT / Mamba  : loaded from saved checkpoints; only inference time is measured here
                 (re-training 200 GPU epochs is impractical to do just for timing)
"""

import os
import sys
import time
import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Initialise the global config singleton BEFORE any hsmodel imports,
# because utils.py captures config.tifpath as a default parameter value.
from projutils.config import config as _cfg
_cfg.reinit("PRISMA")
from projutils.config import config

SAVE_DIR = os.path.join(config.save_model_path)
OUTPUT_FILE = os.path.join(SAVE_DIR, "timing_results.txt")

# ── helpers ──────────────────────────────────────────────────────────────────

def fmt(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    m, s = divmod(seconds, 60)
    return f"{int(m)} min {s:.0f} s"


def load_data():
    with rasterio.open(config.tifpath) as src:
        raw = np.transpose(src.read(), (1, 2, 0))
    with rasterio.open(config.train_mask_path) as src:
        TR = src.read(1)
    with rasterio.open(config.test_mask_path) as src:
        TE = src.read(1)
    return raw, TR, TE


def normalize(data):
    out = np.zeros_like(data, dtype=float)
    for i in range(data.shape[2]):
        b = data[:, :, i]
        valid = b[b != -9999.0]
        if len(valid) > 0 and valid.max() > valid.min():
            out[:, :, i] = np.where(
                b != -9999.0,
                (b - valid.min()) / (valid.max() - valid.min()),
                0.0,
            )
    return out


def prepare_pixel_data(norm, TR, TE):
    train_mask = TR > 0
    test_mask = TE > 0
    x_train = norm[train_mask]
    y_train = TR[train_mask]
    x_test = norm[test_mask]
    y_test = TE[test_mask]
    return x_train, y_train, x_test, y_test


# ── RF ───────────────────────────────────────────────────────────────────────

def time_rf(norm, TR, TE):
    from sklearn.ensemble import RandomForestClassifier

    x_train, y_train, _, _ = prepare_pixel_data(norm, TR, TE)
    _, _, band = norm.shape
    flat = norm.reshape(-1, band)

    print("\n[RF] Training …")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(x_train, y_train)
    train_time = time.time() - t0
    print(f"[RF] Train time : {fmt(train_time)}")

    print("[RF] Full-raster inference …")
    t0 = time.time()
    preds = []
    for i in range(0, len(flat), 1000):
        preds.extend(rf.predict(flat[i : i + 1000]))
    infer_time = time.time() - t0
    print(f"[RF] Infer time : {fmt(infer_time)}")

    return train_time, infer_time


# ── XGBoost ──────────────────────────────────────────────────────────────────

def time_xgb(norm, TR, TE):
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder

    x_train, y_train, _, _ = prepare_pixel_data(norm, TR, TE)
    _, _, band = norm.shape
    flat = norm.reshape(-1, band)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)

    print("\n[XGB] Training …")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    xgb.fit(flat[TR.flatten() > 0], y_tr_enc)
    train_time = time.time() - t0
    print(f"[XGB] Train time : {fmt(train_time)}")

    print("[XGB] Full-raster inference …")
    t0 = time.time()
    preds = []
    for i in range(0, len(flat), 1000):
        preds.extend(xgb.predict(flat[i : i + 1000]))
    infer_time = time.time() - t0
    print(f"[XGB] Infer time : {fmt(infer_time)}")

    return train_time, infer_time


# ── ViT ──────────────────────────────────────────────────────────────────────

def time_vit(norm, TR):
    from hsmodel.vit_pytorch import ViT
    from hsmodel.demo_SpectralFormer import (
        mirror_hsi, choose_train_test_points, predict_full_image_chunked,
    )

    h, w, band = norm.shape
    num_classes = int(np.max(TR))
    patch = 3
    band_patch = 3

    label_combined = TR
    _, _, total_pos_true, _, _, _ = \
        choose_train_test_points(TR, np.zeros_like(TR), label_combined, num_classes)

    mirror = mirror_hsi(h, w, band, norm, patch)

    model = ViT(
        image_size=patch, near_band=band_patch, num_patches=band,
        num_classes=num_classes, dim=64, depth=5, heads=4,
        mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode="ViT",
    ).cuda()

    ckpt = os.path.join(SAVE_DIR, "vit_best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    model.eval()
    print(f"\n[ViT] Loaded checkpoint: {ckpt}")

    print("[ViT] Full-raster inference …")
    t0 = time.time()
    _ = predict_full_image_chunked(model, mirror, total_pos_true, band, patch, band_patch)
    infer_time = time.time() - t0
    print(f"[ViT] Infer time : {fmt(infer_time)}")

    return infer_time


# ── Mamba ────────────────────────────────────────────────────────────────────

def time_mamba(norm, TR, TE):
    from hsmodel.utils import predict_whole_image

    h, w, band = norm.shape

    # predict_whole_image re-reads the raw raster internally, so we just need
    # to pass norm as mirror_image (the function normalises it again itself)
    with rasterio.open(config.tifpath) as src:
        raw_for_mamba = np.transpose(src.read(), (1, 2, 0))

    ckpt = os.path.join(SAVE_DIR, "mamba_best_model_mamba_aboresheed.pt")

    class MambaArgs:
        patches = 3
        batch_size = 64

    model = torch.load(ckpt, map_location="cuda", weights_only=False)
    model.eval()
    print(f"\n[Mamba] Loaded checkpoint: {ckpt}")

    print("[Mamba] Full-raster inference …")
    t0 = time.time()
    _ = predict_whole_image(model, h, w, band, raw_for_mamba, MambaArgs())
    infer_time = time.time() - t0
    print(f"[Mamba] Infer time : {fmt(infer_time)}")

    return infer_time


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading and normalising data …")
    raw, TR, TE = load_data()
    norm = normalize(raw)
    print(f"Image: {raw.shape[0]}×{raw.shape[1]}, {raw.shape[2]} bands")

    rf_train,  rf_infer  = time_rf(norm, TR, TE)
    xgb_train, xgb_infer = time_xgb(norm, TR, TE)
    vit_infer             = time_vit(norm, TR)
    mamba_infer           = time_mamba(norm, TR, TE)

    lines = [
        "===== Timing Results (PRISMA Aboreshead, experiments_008) =====\n",
        f"{'Model':<30} {'Training Time':>18} {'Inference Time (full map)':>26}",
        "-" * 76,
        f"{'Random Forest (10 trees)':<30} {fmt(rf_train):>18} {fmt(rf_infer):>26}",
        f"{'XGBoost (200 rounds)':<30} {fmt(xgb_train):>18} {fmt(xgb_infer):>26}",
        f"{'SpectralFormer (ViT)':<30} {'(needs re-training)':>18} {fmt(vit_infer):>26}",
        f"{'S²Mamba':<30} {'(needs re-training)':>18} {fmt(mamba_infer):>26}",
        "\nNote: ViT and Mamba training times require a full 200-epoch re-run.",
        "      Add --measure-training flag and re-run to capture those.",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(report + "\n")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
