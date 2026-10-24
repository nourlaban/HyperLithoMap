"""
Class names and display colours for any configured dataset, derived from its
training shapefile — no hardcoded class lists.

Index 0 is the unlabeled/background pixel value of the rasterised label maps.
Indices 1..N follow exactly the Classname -> integer mapping assigned by
preprocessing/convertshapelabelstoraster.create_code (alphabetical, with
'Cloud' and 'Background' last), so CLASS_NAMES[i] / COLORS_NORM[i] line up
with the class indices baked into the label rasters and trained models.

Colour precedence per class: the dataset's .clr table (what the pipeline's
coloured label rasters use) -> RED/GREEN/BLUE columns in the shapefile ->
a fixed geology palette by name -> neutral grey.

Usage:
    from projutils.class_scheme import load_class_scheme
    CLASS_NAMES, SHORT_NAMES, COLORS_NORM, COLORS_RGB = load_class_scheme("RabshiPRISMA")
"""

import os
from collections import namedtuple

import geopandas as gpd

from projutils.config import Config
from preprocessing.convertshapelabelstoraster import create_code

ClassScheme = namedtuple('ClassScheme', 'CLASS_NAMES SHORT_NAMES COLORS_NORM COLORS_RGB')

UNLABELED_COLOR = [0.86, 0.86, 0.86]

# Fixed palette (0-1 RGB) for the Abu Rusheid geology classes, matched to the
# original PRISMA / EnMAP Aboreshead figure scripts.
_PALETTE_BY_NAME = {
    "Alkali feldspar granites":        [0.92, 0.10, 0.18],
    "Alteration zones":                [1.00, 0.55, 0.00],
    "Biotite granites":                [0.05, 0.62, 0.25],
    "Gabbro":                          [0.10, 0.35, 0.92],
    "Garnet-muscovite granites":       [0.65, 0.05, 0.85],
    "Gneiss":                          [0.00, 0.82, 0.88],
    "Li-pegmatite granites":           [1.00, 0.90, 0.00],
    "Serpentinite":                    [0.95, 0.15, 0.60],
    "Tonalite-granodiorite":           [0.08, 0.72, 0.55],
    "Wadi deposits":                   [0.55, 0.28, 0.06],
    "Zinnwaldite-muscovite granites":  [0.60, 0.88, 0.05],
    "ophiolitic metagabbro":           [0.28, 0.05, 0.72],
    "Cloud":                           [0.75, 0.75, 0.80],
    "Background":                      [0.94, 0.94, 0.94],
}


def _clr_colors(clr_path):
    """{class index: [r, g, b]} (0-1) from a .clr file ('value R G B [A] [label]')."""
    colors = {}
    if not clr_path or not os.path.exists(clr_path):
        return colors
    with open(clr_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0].lstrip('-').isdigit():
                colors[int(parts[0])] = [int(parts[1]) / 255.0, int(parts[2]) / 255.0, int(parts[3]) / 255.0]
    return colors


def _shapefile_colors(gdf, name_field):
    """{class name: [r, g, b]} (0-1) from RED/GREEN/BLUE shapefile columns, when present."""
    if not {'RED', 'GREEN', 'BLUE'}.issubset(gdf.columns):
        return {}
    rgb = gdf[[name_field, 'RED', 'GREEN', 'BLUE']].dropna().drop_duplicates(subset=name_field)
    return {row[0]: [row[1] / 255.0, row[2] / 255.0, row[3] / 255.0]
            for row in rgb.itertuples(index=False)}


def load_class_scheme(dataset, use_clr=True, palette_first=False) -> ClassScheme:
    """
    Build the class scheme for a dataset name (or an existing Config).

    palette_first=True prefers the fixed geology palette over .clr/shapefile
    colours (used by the Aboreshead EnMAP figures).
    """
    cfg = dataset if isinstance(dataset, Config) else Config(dataset, create_dirs=False)
    name_field = cfg.class_name_field

    code_by_name = create_code(cfg.shapefile_path, name_field)
    name_by_code = {v: k for k, v in code_by_name.items()}
    names = [name_by_code[i] for i in range(1, len(name_by_code) + 1)]

    clr = _clr_colors(getattr(cfg, 'clr_path', None)) if use_clr else {}
    shp = _shapefile_colors(gpd.read_file(cfg.shapefile_path), name_field)

    colors = [UNLABELED_COLOR]
    for idx, name in enumerate(names, start=1):
        pal = _PALETTE_BY_NAME.get(name)
        if palette_first and pal:
            colors.append(pal)
        else:
            colors.append(clr.get(idx) or shp.get(name) or pal or [0.5, 0.5, 0.5])

    class_names = ["Unlabeled"] + names
    colors_rgb = [[round(c * 255) for c in rgb] for rgb in colors]
    return ClassScheme(class_names, class_names, colors, colors_rgb)
