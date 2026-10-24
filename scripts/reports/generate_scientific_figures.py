"""
Generate scientific background figures for ViT/SpectralFormer and S²Mamba.
Outputs PNGs to data/aboresheadmamabadataPrisma/output/experiments_005/figures/science/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projutils.config import data_root
OUT    = os.path.join(data_root(),
         "aboresheadmamabadataPrisma/output/experiments_005/figures/science")
os.makedirs(OUT, exist_ok=True)

DPI = 200

# ── Palette ───────────────────────────────────────────────────────────────────
C_DARK  = "#1A376C"
C_MID   = "#2E6DAF"
C_LIGHT = "#D6E4F0"
C_ACC   = "#E86C1E"
C_GRN   = "#2E8B57"
C_RED   = "#C0392B"
C_PUR   = "#6C3483"
C_GRAY  = "#95A5A6"
C_BG    = "#F8F9FA"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "figure.facecolor": C_BG,
    "axes.facecolor":   C_BG,
    "savefig.facecolor": "white",
    "savefig.dpi":      DPI,
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")

def box(ax, x, y, w, h, label, color, fontsize=8, textcolor="white",
        sublabel=None, radius=0.04):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          linewidth=1.2, edgecolor="white",
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.03 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(x, y - 0.07, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=textcolor, zorder=4)

def arrow(ax, x0, y0, x1, y1, color=C_DARK, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Performance Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig_performance():
    models  = ["Random\nForest", "XGBoost", "SpectralFormer\n(ViT)", "S²Mamba"]
    oa      = [94.54, 98.12, 98.25, 99.24]
    iou     = [87.75, 94.81, 94.82, 97.80]
    f1      = [93.19, 97.25, 97.22, 98.87]
    colors  = [C_GRAY, C_GRN, C_MID, C_ACC]

    x  = np.arange(len(models))
    w  = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    b1 = ax.bar(x - w, oa,  w, label="Overall Accuracy (%)", color=colors, alpha=0.95,
                edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x,     iou, w, label="IoU / Jaccard (%)",    color=colors, alpha=0.70,
                edgecolor="white", linewidth=0.8, hatch="//")
    b3 = ax.bar(x + w, f1,  w, label="Macro F1 (%)",         color=colors, alpha=0.50,
                edgecolor="white", linewidth=0.8, hatch="..")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.15,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylim(82, 101.5)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Performance Comparison — Aboresheed Area, PRISMA L2D",
                 fontsize=12, fontweight="bold", color=C_DARK, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor="#888", label="Overall Accuracy (%)", alpha=0.95),
        mpatches.Patch(facecolor="#888", label="IoU / Jaccard (%)",    alpha=0.70, hatch="//"),
        mpatches.Patch(facecolor="#888", label="Macro F1 (%)",         alpha=0.50, hatch=".."),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    ax.annotate("Best Model", xy=(3 - w, oa[3]), xytext=(2.35, 100.5),
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.5),
                fontsize=9, color=C_ACC, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig1_performance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — SpectralFormer Architecture
# ══════════════════════════════════════════════════════════════════════════════
def fig_spectralformer_arch():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("SpectralFormer Architecture — Hyperspectral Vision Transformer",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)

    # ── Input patch strip ────────────────────────────────────────────────────
    box(ax, 1.0, 3.0, 1.4, 1.2, "Hyperspectral\nPatch Input", C_DARK,
        sublabel="5×5×168 bands")

    # spectral tokens
    tx, ty = 2.8, 3.0
    for i in range(6):
        c = plt.cm.plasma(i / 6)
        r = FancyBboxPatch((tx + i*0.22, ty - 0.5), 0.18, 1.0,
                           boxstyle="round,pad=0.01",
                           facecolor=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.add_patch(r)
    ax.text(tx + 0.66, 2.08, "Spectral Tokens  λ₁ λ₂ … λ_B",
            ha="center", fontsize=8, color="#555", style="italic")
    ax.text(tx + 0.66, 3.72, "Token\nEmbedding", ha="center",
            fontsize=8.5, fontweight="bold", color=C_DARK)

    arrow(ax, 1.72, 3.0, 2.6, 3.0, color=C_MID, lw=2)

    # positional encoding
    box(ax, 4.4, 3.0, 1.2, 0.9, "Positional\nEncoding", C_GRN, fontsize=8.5)
    arrow(ax, 4.14, 3.0, 3.8, 3.0, color=C_MID, lw=2)

    # transformer block (repeated ×5)
    bx = 6.0
    ax.text(bx, 4.95, "× 5 Transformer Encoder Layers", ha="center",
            fontsize=9, color=C_PUR, fontweight="bold")
    big = FancyBboxPatch((bx - 1.6, 1.5), 3.2, 3.1,
                         boxstyle="round,pad=0.05",
                         facecolor="#EDE7F6", edgecolor=C_PUR,
                         linewidth=1.8, linestyle="--", zorder=2)
    ax.add_patch(big)

    box(ax, bx, 4.2, 2.6, 0.65, "Multi-Head Self-Attention (4 heads)", C_PUR, fontsize=8)
    box(ax, bx, 3.35, 2.6, 0.55, "Add & LayerNorm", C_MID, fontsize=8)
    box(ax, bx, 2.55, 2.6, 0.65, "Feed-Forward  MLP (dim=8)", C_MID, fontsize=8)
    box(ax, bx, 1.72, 2.6, 0.55, "Add & LayerNorm", C_MID, fontsize=8)

    for y0, y1 in [(3.875, 3.63), (3.07, 2.88), (2.22, 1.995)]:
        arrow(ax, bx, y0, bx, y1, color=C_PUR, lw=1.5)

    # residual arrows
    for (y0, y1) in [(4.2, 3.35), (2.55, 1.72)]:
        ax.annotate("", xy=(bx + 1.55, y1), xytext=(bx + 1.55, y0),
                    arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.2,
                                   connectionstyle="arc3,rad=0.0"))
        ax.plot([bx + 1.3, bx + 1.55], [y0, y0], color=C_ACC, lw=1.2)
        ax.plot([bx + 1.3, bx + 1.55], [y1, y1], color=C_ACC, lw=1.2)
    ax.text(bx + 1.85, 3.77, "Skip", fontsize=7, color=C_ACC, rotation=90)
    ax.text(bx + 1.85, 2.13, "Skip", fontsize=7, color=C_ACC, rotation=90)

    arrow(ax, 4.99, 3.0, 4.41, 3.0, color=C_MID, lw=2)
    arrow(ax, 7.62, 3.0, 8.15, 3.0, color=C_MID, lw=2)

    # classifier head
    box(ax, 9.0, 4.0, 1.4, 0.75, "Class Token\n[CLS]", C_GRN, fontsize=8)
    box(ax, 9.0, 3.0, 1.4, 0.75, "MLP\nClassifier", C_DARK, fontsize=8)
    box(ax, 9.0, 2.0, 1.4, 0.75, "Softmax", C_ACC, fontsize=8)
    for y0, y1 in [(3.62, 3.37), (2.62, 2.37)]:
        arrow(ax, 9.0, y0, 9.0, y1, color=C_MID, lw=1.5)

    arrow(ax, 9.72, 2.0, 10.3, 2.0, color=C_MID, lw=2)
    box(ax, 11.3, 2.0, 1.6, 0.75, "Class\nPrediction", C_RED, fontsize=8.5)

    # attention formula box
    ax.text(6.0, 0.72,
            r"Attention$(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$",
            ha="center", fontsize=10, color=C_PUR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EDE7F6",
                      edgecolor=C_PUR, linewidth=1.2))

    fig.tight_layout()
    save(fig, "fig2_spectralformer_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Multi-Head Self-Attention Detail
# ══════════════════════════════════════════════════════════════════════════════
def fig_attention_detail():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Multi-Head Self-Attention Mechanism (SpectralFormer)",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.01)

    # LEFT — attention score heat-map (simulated)
    ax = axes[0]
    ax.set_facecolor("white")
    n = 12
    labels = [f"λ{i+1}" for i in range(n)]
    rng = np.random.default_rng(7)
    A = rng.random((n, n))
    # make it banded (nearby bands attend more)
    for i in range(n):
        for j in range(n):
            A[i, j] = np.exp(-0.6 * abs(i - j)) + 0.1 * rng.random()
    A = A / A.sum(axis=1, keepdims=True)
    im = ax.imshow(A, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key (Spectral Band)", fontsize=9)
    ax.set_ylabel("Query (Spectral Band)", fontsize=9)
    ax.set_title("Attention Weight Matrix\n(one head, 12 spectral tokens)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")

    # RIGHT — Q K V schematic
    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 7)
    ax2.axis("off")
    ax2.set_title("Q · K · V Projection Schematic", fontsize=9)

    # Input tokens
    for i, lbl in enumerate(["x₁", "x₂", "x₃", "…", "x_B"]):
        c = plt.cm.plasma(i / 5)
        box(ax2, 1.0, 6.0 - i*1.1, 0.9, 0.7, lbl, c, fontsize=9)

    ax2.text(1.0, 0.4, "Input Tokens", ha="center", fontsize=8.5,
             color="#555", style="italic")

    # W_Q  W_K  W_V
    for xi, label, col in [(3.5, "W_Q", C_PUR), (5.5, "W_K", C_GRN), (7.5, "W_V", C_DARK)]:
        box(ax2, xi, 3.5, 1.3, 0.75, label, col, fontsize=10)
        for yi in [6.0, 4.9, 3.8, 2.7, 1.6]:
            ax2.annotate("", xy=(xi - 0.65, 3.5 + 0.0),
                         xytext=(1.46, yi),
                         arrowprops=dict(arrowstyle="-", color="#cccccc",
                                         lw=0.7, alpha=0.6))

    for xi, label, col in [(3.5, "Q", C_PUR), (5.5, "K", C_GRN), (7.5, "V", C_DARK)]:
        box(ax2, xi, 2.0, 1.0, 0.65, label, col, fontsize=10)
        arrow(ax2, xi, 3.12, xi, 2.33, color=col, lw=1.5)

    # Scaled dot-product
    box(ax2, 4.5, 0.85, 2.8, 0.65, "Scaled Dot-Product\nAttention", C_ACC, fontsize=8)
    arrow(ax2, 3.5, 1.67, 4.1, 1.17, color=C_PUR, lw=1.3)
    arrow(ax2, 5.5, 1.67, 4.9, 1.17, color=C_GRN, lw=1.3)
    arrow(ax2, 7.5, 1.67, 7.5, 1.17, color=C_DARK, lw=1.3)
    ax2.annotate("", xy=(5.8, 0.85), xytext=(7.5, 1.17),
                 arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2,
                                 connectionstyle="arc3,rad=-0.25"))
    ax2.text(9.1, 0.85, "Output", ha="center", fontsize=8.5,
             fontweight="bold", color=C_RED)
    arrow(ax2, 5.9, 0.85, 8.5, 0.85, color=C_ACC, lw=1.5)

    fig.tight_layout()
    save(fig, "fig3_attention_detail.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — S²Mamba Architecture
# ══════════════════════════════════════════════════════════════════════════════
def fig_mamba_arch():
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 15); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("S²Mamba — Spectral-Spatial State Space Model Architecture",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)

    # Input
    box(ax, 1.0, 3.0, 1.4, 1.2, "Hyperspectral\nPatch Input", C_DARK,
        sublabel="3×3×168 bands")

    # Patch Flatten
    box(ax, 2.9, 3.0, 1.2, 0.85, "Flatten &\nLinear Proj.", C_MID, fontsize=8.5)
    arrow(ax, 1.72, 3.0, 2.3, 3.0, color=C_MID, lw=2)

    # S²Mamba block (repeated ×1)
    bx = 6.1
    ax.text(bx, 5.35, "× 1  S²Mamba Block  (dims = 64, d_state = 16)",
            ha="center", fontsize=9.5, color=C_ACC, fontweight="bold")
    big = FancyBboxPatch((bx - 2.3, 0.85), 4.6, 4.15,
                         boxstyle="round,pad=0.05",
                         facecolor="#FEF9F0", edgecolor=C_ACC,
                         linewidth=2.0, linestyle="--", zorder=2)
    ax.add_patch(big)

    # Inside block
    box(ax, bx, 4.55, 3.8, 0.65, "Layer Norm", "#78909C", fontsize=8.5)
    box(ax, bx - 1.1, 3.5, 1.4, 0.65, "Linear\nExpand ×2", C_MID, fontsize=8)
    box(ax, bx + 1.1, 3.5, 1.4, 0.65, "Linear\nExpand ×2", C_MID, fontsize=8)
    box(ax, bx - 1.1, 2.55, 1.4, 0.72, "SSM\nSelective Scan", C_ACC, fontsize=8)
    box(ax, bx + 1.1, 2.55, 1.4, 0.65, "SiLU\nActivation", C_GRN, fontsize=8)
    box(ax, bx, 1.65, 1.6, 0.65, "Hadamard ⊙", C_PUR, fontsize=8.5)
    box(ax, bx, 1.0, 1.6, 0.55, "Linear Project Out", C_DARK, fontsize=8)

    # arrows inside
    arrow(ax, bx,     4.22, bx - 1.1, 3.82, color=C_MID, lw=1.4)
    arrow(ax, bx,     4.22, bx + 1.1, 3.82, color=C_MID, lw=1.4)
    arrow(ax, bx - 1.1, 3.17, bx - 1.1, 2.91, color=C_ACC, lw=1.4)
    arrow(ax, bx + 1.1, 3.17, bx + 1.1, 2.88, color=C_GRN, lw=1.4)
    arrow(ax, bx - 1.1, 2.19, bx - 0.4, 1.97, color=C_ACC, lw=1.4)
    arrow(ax, bx + 1.1, 2.22, bx + 0.4, 1.97, color=C_GRN, lw=1.4)
    arrow(ax, bx, 1.32, bx, 1.27, color=C_PUR, lw=1.4)

    # Residual
    ax.plot([bx + 2.55, bx + 2.55], [3.0, 1.0], color=C_RED, lw=1.4, zorder=3)
    ax.plot([bx + 2.3,  bx + 2.55], [3.0, 3.0], color=C_RED, lw=1.4, zorder=3)
    ax.plot([bx + 1.1,  bx + 2.55], [1.0, 1.0], color=C_RED, lw=1.4, zorder=3)
    ax.text(bx + 2.75, 2.0, "Residual", fontsize=7.5, color=C_RED, rotation=90)

    # LayerNorm at top entry
    arrow(ax, 3.5, 3.0, 3.8, 3.0, color=C_MID, lw=2)
    arrow(ax, bx - 2.3, 3.0, bx - 2.3, 4.55, color=C_MID, lw=1.5)
    arrow(ax, bx - 2.3, 4.55, bx - 1.92, 4.55, color=C_MID, lw=1.5)

    # Output path
    arrow(ax, 8.42, 3.0, 9.0, 3.0, color=C_MID, lw=2)
    box(ax, 9.9, 3.0, 1.4, 0.75, "Global\nAvg Pool", C_GRN, fontsize=8.5)
    arrow(ax, 10.6, 3.0, 11.1, 3.0, color=C_MID, lw=2)
    box(ax, 12.0, 3.0, 1.5, 0.75, "Linear\nClassifier", C_DARK, fontsize=8.5)
    arrow(ax, 12.75, 3.0, 13.3, 3.0, color=C_MID, lw=2)
    box(ax, 14.1, 3.0, 1.3, 0.75, "Class\nPrediction", C_RED, fontsize=8.5)

    # SSM equation box
    ax.text(6.1, 0.22,
            r"$h_t = \bar{A}h_{t-1} + \bar{B}x_t$   $\quad$   $y_t = Ch_t$"
            "\n" r"$\bar{A}=e^{\Delta A}$  $\quad$  $\bar{B}=(\Delta A)^{-1}(e^{\Delta A}-I)\Delta B$",
            ha="center", fontsize=9.5, color=C_ACC,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9F0",
                      edgecolor=C_ACC, linewidth=1.2))

    fig.tight_layout()
    save(fig, "fig4_s2mamba_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — SSM State Transition Diagram
# ══════════════════════════════════════════════════════════════════════════════
def fig_ssm_state():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("State Space Model (SSM) — Core Mechanism of S²Mamba",
                 fontsize=13, fontweight="bold", color=C_DARK)

    # LEFT — continuous time SSM
    ax = axes[0]
    ax.set_facecolor("white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Continuous-Time SSM", fontsize=10.5, fontweight="bold",
                 color=C_DARK, pad=6)

    box(ax, 1.2, 3.0, 1.5, 0.8, "x(t)\n(Input)", C_MID, fontsize=9)
    box(ax, 5.0, 3.0, 2.0, 0.8, "h(t)  State\n(d_state=16)", C_ACC, fontsize=9)
    box(ax, 8.8, 3.0, 1.5, 0.8, "y(t)\n(Output)", C_GRN, fontsize=9)

    arrow(ax, 1.95, 3.0, 4.0, 3.0, color=C_MID, lw=2)
    ax.text(3.0, 3.28, "B", ha="center", fontsize=13, color=C_MID, fontweight="bold")

    arrow(ax, 6.0, 3.0, 8.05, 3.0, color=C_GRN, lw=2)
    ax.text(7.05, 3.28, "C", ha="center", fontsize=13, color=C_GRN, fontweight="bold")

    # self-loop for A
    theta = np.linspace(np.pi*0.15, np.pi*1.85, 60)
    r = 1.0
    xs = 5.0 + r * np.cos(theta)
    ys = 3.0 + r * np.sin(theta)
    ax.plot(xs, ys, color=C_RED, lw=2)
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=2))
    ax.text(5.0, 4.35, "A  (recurrent)", ha="center", fontsize=10,
            color=C_RED, fontweight="bold")

    ax.text(5.0, 1.2,
            r"$h'(t)=Ah(t)+Bx(t)$" + "\n" + r"$y(t)=Ch(t)$",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF9F0",
                      edgecolor=C_ACC, linewidth=1.2))

    # RIGHT — discretized selective scan
    ax2 = axes[1]
    ax2.set_facecolor("white")
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("Selective Scan (Discretised)", fontsize=10.5, fontweight="bold",
                  color=C_DARK, pad=6)

    xs_t = [1, 3.5, 6, 8.5]
    for i, (xi, lbl) in enumerate(zip(xs_t, ["t-2", "t-1", "t", "t+1"])):
        box(ax2, xi, 4.5, 1.4, 0.8, f"h_{{{lbl}}}", C_ACC, fontsize=9)
        box(ax2, xi, 2.8, 1.4, 0.8, f"x_{{{lbl}}}", C_MID, fontsize=9)
        box(ax2, xi, 1.1, 1.4, 0.8, f"y_{{{lbl}}}", C_GRN, fontsize=9)

        arrow(ax2, xi, 4.1, xi, 3.2, color=C_DARK, lw=1.3)
        arrow(ax2, xi, 2.4, xi, 1.5, color=C_DARK, lw=1.3)

        if i < 3:
            arrow(ax2, xi + 0.7, 4.5, xs_t[i+1] - 0.7, 4.5, color=C_RED, lw=1.8)
            ax2.text((xi + xs_t[i+1])/2, 4.82, "Ā", ha="center",
                     fontsize=11, color=C_RED, fontweight="bold")

    ax2.text(4.75, 2.8, "B̄  (input-dep.)", ha="center", fontsize=9,
             color=C_MID, style="italic")
    ax2.text(4.75, 1.1, "C  (output proj.)", ha="center", fontsize=9,
             color=C_GRN, style="italic")

    ax2.text(4.75, 0.22,
             r"$h_t = \bar{A}h_{t-1}+\bar{B}x_t$  ,  $y_t=Ch_t$",
             ha="center", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF9F0",
                       edgecolor=C_ACC, linewidth=1.2))

    ax2.text(9.7, 4.5, "Δ controls\nstep size", ha="center",
             fontsize=8, color=C_PUR, style="italic",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#EDE7F6",
                       edgecolor=C_PUR, linewidth=0.8))

    fig.tight_layout()
    save(fig, "fig5_ssm_state_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Complexity & Spectral Profile Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig_complexity():
    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    # LEFT — O(N) vs O(N²) complexity
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("white")
    N = np.arange(1, 300)
    ax1.plot(N, N**2 / 1000,  color=C_PUR, lw=2.5, label="Transformer  O(N²)")
    ax1.plot(N, N / 10,       color=C_ACC, lw=2.5, label="S²Mamba  O(N)")
    ax1.fill_between(N, N**2/1000, N/10, alpha=0.12, color=C_PUR)
    ax1.set_xlabel("Sequence Length N (spectral tokens)", fontsize=10)
    ax1.set_ylabel("Relative Computation", fontsize=10)
    ax1.set_title("Computational Complexity\nTransformer vs S²Mamba",
                  fontsize=10.5, fontweight="bold", color=C_DARK)
    ax1.legend(fontsize=9.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(0, 300); ax1.set_ylim(0, 90)
    ax1.annotate("Linear\nScaling", xy=(250, 25), xytext=(190, 55),
                 arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.3),
                 color=C_ACC, fontsize=9, fontweight="bold")
    ax1.annotate("Quadratic\nGrowth", xy=(200, 40), xytext=(100, 75),
                 arrowprops=dict(arrowstyle="->", color=C_PUR, lw=1.3),
                 color=C_PUR, fontsize=9, fontweight="bold")

    # RIGHT — radar-style class F1 for Mamba vs RF
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("white")
    classes = ["Alk.Feld.", "Alt.Zone", "Bio.Gran.", "Gabbro",
               "Grt.Mus.", "Gneiss", "Li.Peg.", "Serp.",
               "Ton.Gran.", "Wadi", "Zinn.", "Oph.Meta.", "BKG"]
    mamba_f1 = [0.9964, 0.9764, 0.9866, 0.9943, 0.9775, 0.9914,
                0.9993, 0.9991, 0.9929, 0.9964, 0.9441, 0.9986, 1.000]
    rf_f1    = [0.9509, 0.8017, 0.8587, 0.9641, 0.9066, 0.9325,
                0.9755, 0.9852, 0.9338, 0.9658, 0.8606, 0.9788, 1.000]

    x  = np.arange(len(classes))
    w  = 0.35
    ax2.bar(x - w/2, mamba_f1, w, color=C_ACC, label="S²Mamba", alpha=0.9,
            edgecolor="white")
    ax2.bar(x + w/2, rf_f1,    w, color=C_GRAY, label="Random Forest", alpha=0.9,
            edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=45, ha="right", fontsize=7.5)
    ax2.set_ylim(0.75, 1.01)
    ax2.set_ylabel("F1-Score", fontsize=10)
    ax2.set_title("Per-Class F1: S²Mamba vs Random Forest",
                  fontsize=10.5, fontweight="bold", color=C_DARK)
    ax2.legend(fontsize=9.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig6_complexity_and_class_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Spectral Signature Profiles (simulated PRISMA bands)
# ══════════════════════════════════════════════════════════════════════════════
def fig_spectral_profiles():
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bands = np.linspace(400, 2500, 168)
    rng   = np.random.default_rng(42)

    # Simulated spectral signatures for 6 representative classes
    profiles = {
        "Alkali feldspar granites": (0.35, 0.08,  [1400,1900], "salmon"),
        "Biotite granites":         (0.28, 0.07,  [2200],      "#8B6340"),
        "Gabbro":                   (0.18, 0.05,  [],          "#3C3CB0"),
        "Serpentinite":             (0.22, 0.065, [2300],      "#32A060"),
        "Wadi deposits":            (0.42, 0.06,  [],          "#C8E696"),
        "Zinnwaldite gran.":        (0.30, 0.09,  [2200,2350], "#64B4DC"),
    }

    for name, (base, noise, abs_bands, color) in profiles.items():
        spec = base + noise * rng.standard_normal(168)
        # smooth
        from scipy.ndimage import uniform_filter1d
        spec = uniform_filter1d(spec, size=7)
        # add absorption features
        for ab in abs_bands:
            idx = np.argmin(np.abs(bands - ab))
            spec[max(0,idx-8):idx+8] *= 0.72
        spec = np.clip(spec, 0, 1)
        ax.plot(bands, spec, color=color, lw=1.6, label=name, alpha=0.92)

    # shade water vapour / removed bands
    removed = [(400, 440), (1350, 1480), (1780, 1980), (2400, 2500)]
    for lo, hi in removed:
        ax.axvspan(lo, hi, alpha=0.12, color="gray", label="_nolegend_")

    ax.text(420,  0.46, "Removed", fontsize=6.5, color="gray", rotation=90)
    ax.text(1360, 0.46, "Removed", fontsize=6.5, color="gray", rotation=90)
    ax.text(1790, 0.46, "Removed", fontsize=6.5, color="gray", rotation=90)
    ax.text(2410, 0.46, "Removed", fontsize=6.5, color="gray", rotation=90)

    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("Reflectance", fontsize=11)
    ax.set_title("Simulated PRISMA Spectral Signatures — Key Lithological Classes\n"
                 "(Grey shading = removed noise / water-vapour absorption bands)",
                 fontsize=11, fontweight="bold", color=C_DARK)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(400, 2500); ax.set_ylim(0, 0.52)
    ax.xaxis.grid(True, alpha=0.25, linestyle="--")

    fig.tight_layout()
    save(fig, "fig7_spectral_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating scientific figures …")
    fig_performance()
    fig_spectralformer_arch()
    fig_attention_detail()
    fig_mamba_arch()
    fig_ssm_state()
    fig_complexity()
    fig_spectral_profiles()
    print(f"\nAll figures saved to:\n  {OUT}")
