"""
Time one training epoch for ViT and Mamba, then extrapolate to 200 epochs.
Does NOT overwrite saved models.
"""

import os, sys, time
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from projutils.config import config as _cfg
_cfg.reinit("PRISMA")
from projutils.config import config

EPOCHS = 200
PROBE  = 3   # epochs to time for averaging

# ── shared data loading ───────────────────────────────────────────────────────

def load_and_normalize():
    with rasterio.open(config.tifpath) as src:
        raw = np.transpose(src.read(), (1, 2, 0))
    with rasterio.open(config.train_mask_path) as src:
        TR = src.read(1)
    with rasterio.open(config.test_mask_path) as src:
        TE = src.read(1)

    out = np.zeros_like(raw, dtype=float)
    for i in range(raw.shape[2]):
        b = raw[:, :, i]
        valid = b[b != -9999.0]
        if len(valid) > 0 and valid.max() > valid.min():
            out[:, :, i] = np.where(b != -9999.0,
                                    (b - valid.min()) / (valid.max() - valid.min()), 0.0)
    return out, TR, TE

# ── ViT ──────────────────────────────────────────────────────────────────────

def time_vit_training(norm, TR, TE):
    from hsmodel.vit_pytorch import ViT
    from hsmodel.demo_SpectralFormer import (
        choose_train_test_points, mirror_hsi, prepare_data, create_labels,
        train_epoch, valid_epoch,
    )
    from hsmodel.utils import AvgrageMeter

    h, w, band = norm.shape
    num_classes = int(np.max(TR))
    patch, band_patch, batch_size, lr = 3, 3, 128, 5e-4

    total_pos_train, total_pos_test, total_pos_true, n_train, n_test, n_true = \
        choose_train_test_points(TR, TE, TR + TE, num_classes)
    mirror = mirror_hsi(h, w, band, norm, patch)
    x_train_band, x_test_band = prepare_data(mirror, band, total_pos_train, total_pos_test, patch, band_patch)
    y_train, y_test, _ = create_labels(n_train, n_test, n_true, num_classes)

    train_ds = Data.TensorDataset(
        torch.from_numpy(x_train_band.transpose(0, 2, 1)).float(),
        torch.from_numpy(y_train).long())
    test_ds  = Data.TensorDataset(
        torch.from_numpy(x_test_band.transpose(0, 2, 1)).float(),
        torch.from_numpy(y_test).long())
    train_loader = Data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = Data.DataLoader(test_ds,  batch_size=batch_size, shuffle=True)

    model = ViT(image_size=patch, near_band=band_patch, num_patches=band,
                num_classes=num_classes, dim=64, depth=5, heads=4,
                mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode="ViT").cuda()

    class_weights = (1.0 / torch.FloatTensor(n_train))
    class_weights = class_weights / class_weights.sum() * len(n_train)
    criterion  = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    print(f"\n[ViT] Timing {PROBE} training epochs …")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(PROBE):
        train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_epoch = elapsed / PROBE
    total_est = per_epoch * EPOCHS
    print(f"[ViT] {per_epoch:.2f} s/epoch  →  ~{total_est/60:.1f} min for {EPOCHS} epochs")
    return total_est

# ── Mamba ────────────────────────────────────────────────────────────────────

def time_mamba_training(norm, TR, TE):
    from hsmodel.s2mamba import S2Mamba
    from hsmodel.utils import (
        chooose_train_and_test_point, mirror_hsi,
        train_and_test_data, train_and_test_label, train_epoch, AvgrageMeter,
    )

    h, w, band = norm.shape
    label = TR + TE
    num_classes = int(np.max(TR))
    patch, batch_size, lr, wd = 3, 64, 5e-4, 5e-3

    total_pos_train, total_pos_test, _, n_train, n_test, n_true = \
        chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror = mirror_hsi(h, w, band, norm, patch)
    x_train, x_test = train_and_test_data(mirror, band, total_pos_train, total_pos_test, None, patch)
    y_train, y_test, _ = train_and_test_label(n_train, n_test, n_true, num_classes)

    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).float()
    y_train  = torch.from_numpy(y_train).long()
    train_loader = Data.DataLoader(Data.TensorDataset(x_train, y_train),
                                   batch_size=batch_size, shuffle=True)

    model = S2Mamba(in_chans=band, patch=patch, num_classes=num_classes,
                    depths=[1], dims=[64], drop_path_rate=0.1, attn_drop_rate=0.1).cuda()

    class_weights = (1.0 / torch.FloatTensor(n_train))
    class_weights = class_weights / class_weights.sum() * len(n_train)
    criterion  = nn.CrossEntropyLoss(weight=class_weights.cuda())
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    print(f"\n[Mamba] Timing {PROBE} training epochs …")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(PROBE):
        train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_epoch = elapsed / PROBE
    total_est = per_epoch * EPOCHS
    print(f"[Mamba] {per_epoch:.2f} s/epoch  →  ~{total_est/60:.1f} min for {EPOCHS} epochs")
    return total_est


def fmt(s):
    if s < 60:
        return f"~{s:.0f} s"
    return f"~{s/60:.1f} min"


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    cudnn.deterministic = True
    cudnn.benchmark = False

    print("Loading data …")
    norm, TR, TE = load_and_normalize()
    print(f"Image: {norm.shape[0]}×{norm.shape[1]}, {norm.shape[2]} bands  "
          f"| train px: {(TR>0).sum()}  test px: {(TE>0).sum()}")

    vit_train   = time_vit_training(norm, TR, TE)
    mamba_train = time_mamba_training(norm, TR, TE)

    print("\n" + "="*60)
    print(f"  ViT   estimated training time : {fmt(vit_train)}")
    print(f"  Mamba estimated training time : {fmt(mamba_train)}")
    print("="*60)

    out = os.path.join(config.save_model_path, "training_time_estimates.txt")
    os.makedirs(config.save_model_path, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"ViT   training time estimate ({EPOCHS} epochs): {fmt(vit_train)}\n")
        f.write(f"Mamba training time estimate ({EPOCHS} epochs): {fmt(mamba_train)}\n")
        f.write(f"(based on {PROBE} timed epochs each)\n")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
