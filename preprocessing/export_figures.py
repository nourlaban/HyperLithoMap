"""
Export thematic classification TIF maps to publication-quality PNG figures.

Produces:
  - One figure per model (individual PNG)
  - One 2×2 comparison panel (comparison_panel.png)

All figures use the geological colour table from the .clr file and include
an accurate scale bar, north arrow, and class legend.
python preprocessing/export_figures.py --satellite PRISMA____
"""

import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_NODATA_COLOR = (1.00, 1.00, 1.00)   # white for unmapped pixels (class 0 / nodata)

# Human-readable model names keyed on TIF filename prefix
_MODEL_TITLES = {
    'rf':    'Random Forest',
    'xgb':   'XGBoost',
    'MAMBA': 'S²Mamba',
    'VIT':   'SpectralFormer (ViT)',
    'new':   'Reference Labels',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _class_names_from_shapefile(shapefile_path: str, class_name_field: str = 'Classname') -> dict:
    """
    Read class names from the training shapefile and return {class_id: name},
    using the same index assignment as the rasterisation step (create_code).
    """
    from preprocessing.convertshapelabelstoraster import create_code
    return {idx: name for name, idx in create_code(shapefile_path, class_name_field).items()}


def _build_colormap(class_names: dict) -> dict:
    """
    Generate a perceptually distinguishable RGB colormap for the given classes.

    Uses Kelly's 22 colors of maximum contrast for the first 20 geological
    classes, then falls back to evenly-spaced HSV hues for any extras.
    'Background' and 'Cloud' classes are always rendered in neutral grey.
    Returns {class_id: (r, g, b)} with float values 0–1.
    """
    # Kelly (1965) maximum-contrast palette — skipping white/black
    _KELLY_RGB = [
        (255, 179,   0),   # vivid yellow
        (  0,  83, 138),   # strong blue
        (193,   0,  32),   # vivid red
        (  0, 125,  52),   # vivid green
        (255, 104,   0),   # vivid orange
        ( 83,  55, 122),   # strong violet
        (244, 200,   0),   # vivid greenish-yellow
        (128,  62, 117),   # strong purple
        (147, 170,   0),   # vivid yellowish-green
        (179,  40,  81),   # strong purplish-red
        (166, 189, 215),   # very light blue
        (127,  24,  13),   # strong reddish-brown
        (255, 142,   0),   # strong orange-yellow
        (246, 118, 142),   # strong purplish-pink
        ( 89,  51,  21),   # deep yellowish-brown
        (241,  58,  19),   # vivid reddish-orange
        (206, 162,  98),   # grayish yellow
        ( 35,  44,  22),   # dark olive green
        (255, 122,  92),   # strong yellowish-pink
        (129, 112, 102),   # medium grey (non-background)
    ]

    _NEUTRAL_GREY = (0.82, 0.82, 0.82)
    _NEUTRAL_NAMES = {'background', 'cloud'}

    colormap: dict = {}
    kelly_idx = 0

    for cid in sorted(class_names.keys()):
        name = class_names[cid]
        if name.lower() in _NEUTRAL_NAMES:
            colormap[cid] = _NEUTRAL_GREY
        elif kelly_idx < len(_KELLY_RGB):
            r, g, b = _KELLY_RGB[kelly_idx]
            colormap[cid] = (r / 255.0, g / 255.0, b / 255.0)
            kelly_idx += 1
        else:
            # HSV fallback for very large class sets
            import colorsys
            hue = (kelly_idx * 0.618033988749895) % 1.0   # golden-ratio spacing
            colormap[cid] = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
            kelly_idx += 1

    return colormap


def _write_clr(colormap: dict, class_names: dict, out_path: str) -> None:
    """
    Write a QGIS-compatible .clr colour table file.

    Format per line:  class_id  R  G  B  255  Class Name
    """
    with open(out_path, 'w') as fh:
        for cid in sorted(colormap.keys()):
            r, g, b = colormap[cid]
            R, G, B = int(r * 255), int(g * 255), int(b * 255)
            label = class_names.get(cid, str(cid))
            fh.write(f'{cid} {R} {G} {B} 255 {label}\n')


def _parse_clr(clr_path: str) -> dict:
    """Parse a .clr colour table → {class_id: (r, g, b)} with floats 0–1."""
    colormap = {}
    with open(clr_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 4:
                cid  = int(parts[0])
                r, g, b = int(parts[1]) / 255.0, int(parts[2]) / 255.0, int(parts[3]) / 255.0
                colormap[cid] = (r, g, b)
    return colormap


def _resolve_model_title(filename: str) -> str:
    """Derive a human-readable model name from the TIF filename."""
    stem = Path(filename).stem
    for prefix, title in _MODEL_TITLES.items():
        if stem.startswith(prefix):
            return title
    return stem


def _downsample(data: np.ndarray, max_dim: int = 4000) -> np.ndarray:
    """Subsample a 2-D array so neither dimension exceeds max_dim."""
    h, w = data.shape
    sh = max(1, h // max_dim)
    sw = max(1, w // max_dim)
    return data[::sh, ::sw]


def _to_rgb(data: np.ndarray, colormap: dict) -> np.ndarray:
    """Map 2-D integer class array to H×W×3 RGB float array."""
    rgb = np.full((*data.shape, 3), _NODATA_COLOR, dtype=np.float32)
    for cid, color in colormap.items():
        rgb[data == cid] = color
    return rgb


def _try_read_oa(experiments_dir: str, model_prefix: str) -> str:
    """Try to read Overall Accuracy from saved evaluation .txt file."""
    prefix_map = {
        'rf':    'rf_model_evaluation/evaluation_results.txt',
        'xgb':   'xgb_model_evaluation/evaluation_results.txt',
        'MAMBA': 'mamba_model_evaluation/mamba_evaluation_results.txt',
        'VIT':   'VITransformer_model_evaluation/VITransformer_evaluation_results.txt',
    }
    rel = prefix_map.get(model_prefix)
    if rel is None:
        return ''
    full_path = os.path.join(experiments_dir, 'saved_models', rel)
    if not os.path.exists(full_path):
        return ''
    with open(full_path) as fh:
        for line in fh:
            if line.startswith('Accuracy:'):
                val = line.split(':')[1].strip()
                return f'OA = {float(val):.4f}'
    return ''


def _pixel_size_metres(src) -> float:
    """Return the pixel size in metres, converting from degrees if needed."""
    ps = abs(src.transform.a)
    if src.crs and src.crs.is_geographic:
        # 1 degree latitude ≈ 111 000 m; good enough for a scale bar
        ps = ps * 111_000
    return ps


def _auto_bar_km(pixel_size_m: float, image_width_px: int,
                 target_frac: float = 0.20) -> float:
    """Pick a round scale-bar length (km) that spans ~target_frac of the image."""
    total_m   = pixel_size_m * image_width_px
    ideal_km  = total_m * target_frac / 1_000
    nice_vals = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    for v in nice_vals:
        if v >= ideal_km:
            return v
    return nice_vals[-1]


def _add_scale_bar(ax, pixel_size_m: float, image_width_px: int,
                   bar_km: float = 0, y_frac: float = 0.04):
    """
    Draw a scale bar at the bottom-left of the axes.

    pixel_size_m  – pixel ground size in **metres** (already converted).
    image_width_px – original (full-resolution) raster width in pixels.
    bar_km        – desired bar length in km; 0 = auto-select.
    """
    if bar_km <= 0:
        bar_km = _auto_bar_km(pixel_size_m, image_width_px)

    bar_frac = (bar_km * 1_000) / (pixel_size_m * image_width_px)
    bar_frac = min(bar_frac, 0.35)   # never wider than 35% of axes

    x0, y0 = 0.04, y_frac
    x1     = x0 + bar_frac
    xmid   = (x0 + x1) / 2

    stroke = [pe.withStroke(linewidth=2, foreground='white')]

    bar = mpatches.FancyBboxPatch(
        (x0, y0), bar_frac, 0.012,
        boxstyle='square,pad=0',
        transform=ax.transAxes,
        facecolor='black', edgecolor='white', linewidth=0.5,
        clip_on=False, zorder=5,
    )
    ax.add_patch(bar)

    for x in (x0, x1):
        ax.plot([x, x], [y0, y0 + 0.018], transform=ax.transAxes,
                color='black', linewidth=1.0, clip_on=False, zorder=5)

    label = f'{bar_km:.0f} km' if bar_km >= 1 else f'{bar_km * 1000:.0f} m'
    kw = dict(transform=ax.transAxes, ha='center', va='bottom',
              fontsize=7, color='black', fontweight='bold',
              path_effects=stroke, clip_on=False, zorder=6)
    ax.text(x0,   y0 + 0.022, '0',    **kw)
    ax.text(xmid, y0 + 0.022, label,  **kw)


def _add_north_arrow(ax, x: float = 0.94, y_base: float = 0.08):
    """
    Draw a north arrow (upward with 'N' label) at the specified axes-fraction
    position (bottom-right corner by default).
    """
    stroke = [pe.withStroke(linewidth=2.5, foreground='white')]

    # Arrow shaft
    ax.annotate(
        '', xy=(x, y_base + 0.12), xytext=(x, y_base),
        xycoords='axes fraction',
        arrowprops=dict(
            arrowstyle='-|>', color='black', lw=1.5,
            mutation_scale=10,
        ),
        zorder=6,
    )
    # N label
    ax.text(x, y_base + 0.145, 'N',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='black',
            path_effects=stroke, zorder=6)


def _add_map_border(ax):
    """Draw a thin black border around the map axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor('#333333')


# ── Public API ────────────────────────────────────────────────────────────────

def export_thematic_maps_to_figures(
    experiments_dir: str,
    clr_path: 'str | None' = None,
    shapefile_path: 'str | None' = None,
    class_names: 'dict | None' = None,
    output_subdir: str = 'figures',
    bar_km: int = 5,
    dpi: int = 300,
    also_make_panel: bool = True,
    class_name_field: str = 'Classname',
):
    """
    Convert classified GeoTIFF maps to publication-quality PNG figures.

    Parameters
    ----------
    experiments_dir : str
        Path to the experiments folder, e.g.
        'data/RabshimamabadataPrisma/output/experiments_003'
    clr_path : str or None
        Path to the .clr colour table file.  When None the function falls back
        to a built-in 12-colour default palette.
    shapefile_path : str or None
        Path to the training shapefile.  Class names are read from its
        'Classname' column using the same sort order as the rasterisation
        step.  Takes precedence over class_names when both are supplied.
    class_names : dict or None
        {class_id: 'Class Name'}.  Used only when shapefile_path is None.
    output_subdir : str
        Sub-folder inside experiments_dir for PNG outputs (default 'figures').
    bar_km : int
        Scale bar length in kilometres (default 5).
    dpi : int
        Output resolution — 300 is standard for print journals (default 300).
    also_make_panel : bool
        If True, also write a multi-panel comparison figure (default True).

    Returns
    -------
    list[str]
        Paths of all PNG files written.
    """
    if shapefile_path and os.path.exists(shapefile_path):
        class_names = _class_names_from_shapefile(shapefile_path, class_name_field)
    elif class_names is None:
        raise ValueError(
            'Provide either shapefile_path (preferred) or class_names.'
        )

    figures_dir = os.path.join(experiments_dir, output_subdir)
    os.makedirs(figures_dir, exist_ok=True)

    if clr_path and os.path.exists(clr_path):
        colormap = _parse_clr(clr_path)
    else:
        colormap = _build_colormap(class_names)
        generated_clr = os.path.join(figures_dir, 'classification_colormap.clr')
        _write_clr(colormap, class_names, generated_clr)
        print(f'  Generated QGIS .clr: {generated_clr}')

    colormap_legend = dict(colormap)

    tif_files = sorted(glob.glob(os.path.join(experiments_dir, '*_prediction_map.tif')))
    if not tif_files:
        print(f'[export_figures] No *_prediction_map.tif files found in:\n  {experiments_dir}')
        return []

    # ── Legend handles (shared across all figures) ────────────────────────────
    legend_handles = []
    for cid in sorted(class_names.keys()):
        color = colormap_legend.get(cid, (0.5, 0.5, 0.5))
        edgecolor = '#555555' if max(color) > 0.85 else 'none'
        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor=edgecolor,
                           linewidth=0.4, label=class_names[cid])
        )

    saved_paths = []
    panel_items = []  # [(rgb, pixel_size_m, title, oa_str)]

    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.size':          8,
        'axes.titlesize':     11,
        'axes.titleweight':  'bold',
        'figure.facecolor':  'white',
        'savefig.facecolor': 'white',
    })

    for tif_path in tif_files:
        stem  = Path(tif_path).stem
        title = _resolve_model_title(os.path.basename(tif_path))

        # ── Determine model prefix for OA lookup ─────────────────────────────
        prefix = next((k for k in _MODEL_TITLES if stem.startswith(k)), None)
        oa_str = _try_read_oa(experiments_dir, prefix) if prefix else ''

        with rasterio.open(tif_path) as src:
            data         = src.read(1)
            pixel_size_m = _pixel_size_metres(src)
            h, w         = src.height, src.width

        rgb = _to_rgb(_downsample(data), colormap)
        panel_items.append((rgb, pixel_size_m, title, oa_str, w))

        # ── Individual figure ─────────────────────────────────────────────────
        # Derive figure size from the downsampled image aspect ratio so
        # matplotlib never tries to honour the original raster dimensions.
        img_h, img_w = rgb.shape[:2]
        map_h = min(6.5, 8.0 * img_h / max(img_w, 1))
        map_w = map_h * img_w / max(img_h, 1)
        map_w = min(map_w, 8.0)
        fig_w = map_w + 3.2   # room for legend column
        fig_h = max(map_h, 4.5)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        leg_ratio = int(3.2 / fig_w * 100)
        map_ratio = 100 - leg_ratio
        gs  = GridSpec(1, 2, figure=fig,
                       width_ratios=[map_ratio, leg_ratio],
                       left=0.02, right=0.98, top=0.92, bottom=0.04,
                       wspace=0.04)

        ax_map = fig.add_subplot(gs[0])
        ax_leg = fig.add_subplot(gs[1])

        # Map — aspect='auto' fills the axes without pixel-accurate scaling
        ax_map.imshow(rgb, aspect='auto', interpolation='nearest')
        ax_map.set_aspect('auto')
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        _add_map_border(ax_map)
        _add_scale_bar(ax_map, pixel_size_m, w, bar_km=bar_km)
        _add_north_arrow(ax_map)

        # Title + optional OA subtitle
        full_title = title
        if oa_str:
            full_title += f'\n{oa_str}'
        fig.suptitle(full_title, fontsize=12, fontweight='bold',
                     y=0.97, va='top')

        # Legend panel
        ax_leg.axis('off')
        leg = ax_leg.legend(
            handles=legend_handles,
            loc='center left',
            fontsize=8,
            title='Geological Units',
            title_fontsize=9,
            frameon=True,
            framealpha=1.0,
            edgecolor='#888888',
            handlelength=1.6,
            handleheight=1.6,
            borderpad=0.8,
            labelspacing=0.7,
        )
        leg.get_frame().set_linewidth(0.6)

        out_path = os.path.join(figures_dir, stem + '.png')
        fig.savefig(out_path, dpi=dpi)   # no bbox_inches='tight' — figsize is pre-computed
        plt.close(fig)
        print(f'  Saved: {out_path}')
        saved_paths.append(out_path)

    # ── Multi-panel comparison figure ─────────────────────────────────────────
    if also_make_panel and len(panel_items) >= 2:
        n      = len(panel_items)
        ncols  = 2
        nrows  = (n + 1) // 2

        fig = plt.figure(figsize=(12, nrows * 5.8 + 1.6), facecolor='white')

        # Grid: nrows × ncols map axes + one row for legend
        gs = GridSpec(
            nrows + 1, ncols,
            figure=fig,
            height_ratios=[5.0] * nrows + [1.0],
            left=0.03, right=0.97, top=0.94, bottom=0.01,
            hspace=0.14, wspace=0.05,
        )

        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

        for idx, (rgb, pixel_size_m, title, oa_str, w) in enumerate(panel_items):
            row, col = divmod(idx, ncols)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(rgb, aspect='auto', interpolation='nearest')
            ax.set_aspect('auto')
            ax.set_xticks([])
            ax.set_yticks([])
            _add_map_border(ax)
            _add_scale_bar(ax, pixel_size_m, w, bar_km=bar_km)
            _add_north_arrow(ax)

            subplot_title = f'{labels[idx]}  {title}'
            if oa_str:
                subplot_title += f'  —  {oa_str}'
            ax.set_title(subplot_title, fontsize=9, fontweight='bold', pad=4)

        # Hide any unused axes in the last row
        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            fig.add_subplot(gs[row, col]).set_visible(False)

        # Shared legend in the bottom row spanning all columns
        ax_leg = fig.add_subplot(gs[nrows, :])
        ax_leg.axis('off')
        ax_leg.legend(
            handles=legend_handles,
            loc='center',
            ncol=min(len(legend_handles), 5),
            fontsize=8.5,
            title='Geological Units',
            title_fontsize=9.5,
            frameon=True,
            framealpha=1.0,
            edgecolor='#888888',
            handlelength=1.6,
            handleheight=1.6,
            borderpad=0.8,
            columnspacing=1.2,
        ).get_frame().set_linewidth(0.6)

        fig.suptitle(
            'Hyperspectral Lithological Classification —  Study Area ',
            fontsize=13, fontweight='bold', y=0.975,
        )

        panel_path = os.path.join(figures_dir, 'comparison_panel.png')
        fig.savefig(panel_path, dpi=dpi)
        plt.close(fig)
        print(f'  Saved panel: {panel_path}')
        saved_paths.append(panel_path)

    print(f'\n[export_figures] Done — {len(saved_paths)} file(s) written to:\n  {figures_dir}')
    return saved_paths


# ── Confusion matrix panel ────────────────────────────────────────────────────

def export_confusion_matrix_panel(
    saved_models_dir: str,
    output_dir: str,
    dpi: int = 300,
) -> str:
    """
    Build a publication-quality 2×2 confusion matrix panel from evaluation
    results stored under saved_models_dir and write it to output_dir.

    Returns the path of the saved PNG.
    """
    import re

    _ABBREV = {
        'Alteration zone': 'ALT', 'MGr': 'MGr', 'Melange': 'MEL',
        'Mgb': 'MGB', 'Ms': 'Ms',  'Mv': 'Mv',  'SR': 'SR',
        'TG': 'TG',  'Ws': 'Ws',  'jms': 'jms', 'Background': 'BKG',
    }

    _MODEL_SPECS = [
        ('(a) Random Forest',    'rf_model_evaluation/evaluation_results.txt'),
        ('(b) XGBoost',          'xgb_model_evaluation/evaluation_results.txt'),
        ('(c) VI-Transformer',   'VITransformer_model_evaluation/VITransformer_evaluation_results.txt'),
        ('(d) S²Mamba',          'mamba_model_evaluation/mamba_evaluation_results.txt'),
    ]

    def _parse_results(path):
        txt = open(path).read()
        m_acc = re.search(r'Accuracy:\s*([\d.]+)', txt)
        m_iou = re.search(r'IoU.*?:\s*([\d.]+)', txt)
        acc = float(m_acc.group(1)) if m_acc else 0.0
        iou = float(m_iou.group(1)) if m_iou else 0.0
        classes  = re.findall(r'\d+\.?\d*:\s*(.+?)\s*\(\d+\)', txt)
        cm_lines = re.findall(r'^\s*\[+(.*?)\]+\s*$', txt, re.MULTILINE)
        cm = np.array([[int(x) for x in row.replace('[','').replace(']','').split()]
                       for row in cm_lines if row.strip()])
        return acc, iou, classes, cm

    # ── Collect data ──────────────────────────────────────────────────────────
    records = []
    for label, rel_path in _MODEL_SPECS:
        full = os.path.join(saved_models_dir, rel_path)
        if not os.path.exists(full):
            print(f'  [warn] not found: {full}')
            continue
        acc, iou, classes, cm = _parse_results(full)
        records.append((label, acc, iou, classes, cm))

    if not records:
        raise FileNotFoundError(f'No evaluation files found in {saved_models_dir}')

    # Use class names from the first model (all share the same classes)
    ref_classes = records[0][3]
    tick_labels = [_ABBREV.get(c, c[:4]) for c in ref_classes]
    n = len(tick_labels)

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':         8,
        'axes.titlesize':    9,
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
        'savefig.facecolor':'white',
    })

    ncols, nrows = 2, (len(records) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(14, 12),
                             gridspec_kw={'hspace': 0.42, 'wspace': 0.26})
    axes = axes.flatten()

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    vmin, vmax = 0, 100
    cmap = plt.colormaps['Blues']

    for idx, (label, acc, iou, classes, cm) in enumerate(records):
        ax = axes[idx]

        # Row-normalise → recall %
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm.astype(float) / row_sums * 100

        ax.imshow(cm_norm, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        # Minor gridlines between cells
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='#cccccc', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Tick labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7.5)
        ax.set_yticklabels(tick_labels, fontsize=7.5)
        ax.set_xlabel('Predicted class', fontsize=8, labelpad=4)
        ax.set_ylabel('True class',      fontsize=8, labelpad=4)

        # Remove outer spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Cell annotations
        for i in range(n):
            for j in range(n):
                v = cm_norm[i, j]
                if v < 1 and i != j:
                    continue
                is_diag  = (i == j)
                txt      = f'{v:.0f}'
                weight   = 'bold' if is_diag else 'normal'
                color    = 'white' if (is_diag and v >= 60) else 'black'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=6.5, fontweight=weight, color=color)

        title = f'{label}\nOA = {acc * 100:.1f} %  |  IoU = {iou * 100:.1f} %'
        ax.set_title(title, fontsize=9, fontweight='bold', pad=6)

    # Hide any spare axes
    for idx in range(len(records), len(axes)):
        axes[idx].set_visible(False)

    # Shared colorbar — attach to the right of the whole figure
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes((0.90, 0.15, 0.018, 0.70))
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label('Recall (%)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, 'confusion_matrix_panel.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')
    return out


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from projutils.config import Config, available_datasets

    parser = argparse.ArgumentParser(
        description='Export thematic classification maps to publication-quality PNG figures.'
    )
    parser.add_argument(
        '--dataset', '--satellite', dest='satellite', required=True,
        choices=available_datasets(),
        help='Dataset (from configs/datasets.yaml) to export figures for',
    )
    parser.add_argument('--experiment', default=None,
                        help='Experiment folder name (default: the dataset\'s configured experiment)')
    parser.add_argument('--data-root', default=None, help='Override the data root directory')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution in DPI (default: 300)')
    parser.add_argument('--bar-km', type=float, default=0,
                        help='Scale bar length in km (default: auto)')
    parser.add_argument('--confusion-panel', action='store_true',
                        help='Generate publication confusion matrix panel instead of map figures')
    args = parser.parse_args()

    cfg = Config(satellite_name=args.satellite, experiment=args.experiment, data_root=args.data_root)

    if args.confusion_panel:
        saved_models_dir = os.path.join(cfg.experiment_path, 'saved_models')
        figures_dir      = os.path.join(cfg.experiment_path, 'figures')
        export_confusion_matrix_panel(saved_models_dir, figures_dir, dpi=args.dpi)
    else:
        clr = getattr(cfg, 'clr_path', None)
        if clr is None or not os.path.exists(clr):
            clr = None
        export_thematic_maps_to_figures(
            experiments_dir=cfg.experiment_path,
            clr_path=clr,
            shapefile_path=cfg.shapefile_path,
            dpi=args.dpi,
            bar_km=args.bar_km,
            class_name_field=cfg.class_name_field,
        )
