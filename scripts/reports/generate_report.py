"""Generate the technical report as a formatted .docx file."""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projutils.config import data_root
DATA_ROOT = data_root()

EXP  = os.path.join(DATA_ROOT, "aboresheadmamabadataPrisma/output/experiments_005")
FIG  = os.path.join(EXP, "figures")
SCI  = os.path.join(FIG, "science")
MDL  = os.path.join(EXP, "saved_models")
OUT  = os.path.join(EXP, "Technical_Report_Aboresheed_PRISMA.docx")

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BLUE  = RGBColor(0x1A, 0x37, 0x6C)
MID_BLUE   = RGBColor(0x2E, 0x6D, 0xAF)
LIGHT_BLUE = RGBColor(0xD6, 0xE4, 0xF0)
ACCENT     = RGBColor(0xE8, 0x6C, 0x1E)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
DARK_GRAY  = RGBColor(0x40, 0x40, 0x40)

doc = Document()

# ── Page margins ─────────────────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin    = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── Helper: shade a table cell ───────────────────────────────────────────────
def shade_cell(cell, hex_color: str):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def set_cell_border(cell, **kwargs):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if side in kwargs:
            tag = OxmlElement(f"w:{side}")
            for k, v in kwargs[side].items():
                tag.set(qn(f"w:{k}"), v)
            tcBorders.append(tag)
    tcPr.append(tcBorders)

def cell_text(cell, text, bold=False, size=9, color=None, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    run = p.add_run(str(text))
    run.bold      = bold
    run.font.size = Pt(size)
    run.font.name = "Calibri"
    if color:
        run.font.color.rgb = color

def add_heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14 if level == 1 else 8)
    p.paragraph_format.space_after  = Pt(4)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.bold = True
    if level == 1:
        run.font.size  = Pt(14)
        run.font.color.rgb = DARK_BLUE
        # bottom border
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"),   "single")
        bottom.set(qn("w:sz"),    "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "1A376C")
        pBdr.append(bottom)
        pPr.append(pBdr)
    elif level == 2:
        run.font.size  = Pt(11)
        run.font.color.rgb = MID_BLUE
    else:
        run.font.size  = Pt(10)
        run.font.color.rgb = DARK_GRAY
    return p

def add_body(doc, text, size=10, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(space_after)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.color.rgb = DARK_GRAY
    return p

def add_bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    run = p.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.color.rgb = DARK_GRAY

def add_figure(doc, img_path, caption, width_cm=15):
    if not os.path.exists(img_path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run()
    run.add_picture(img_path, width=Cm(width_cm))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(0)
    cap.paragraph_format.space_after  = Pt(10)
    r = cap.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.name = "Calibri"
    r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

def styled_table(doc, headers, rows, col_widths=None):
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        shade_cell(cell, "1A376C")
        cell_text(cell, h, bold=True, size=9, color=WHITE,
                  align=WD_ALIGN_PARAGRAPH.CENTER)

    # Data rows
    for i, row in enumerate(rows):
        bg = "F2F2F2" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            shade_cell(cell, bg)
            align = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            cell_text(cell, val, size=9, align=align)

    if col_widths:
        for i, row_cells in enumerate(table.rows):
            for j, cell in enumerate(row_cells.cells):
                cell.width = Cm(col_widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return table

# ═══════════════════════════════════════════════════════════════════════════════
#  COVER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
cover_table = doc.add_table(rows=1, cols=1)
cover_table.alignment = WD_TABLE_ALIGNMENT.CENTER
cell = cover_table.cell(0, 0)
shade_cell(cell, "1A376C")
cell.width = Cm(16)

for text, size, italic in [
    ("TECHNICAL REPORT", 11, False),
    ("", 4, False),
    ("Hyperspectral Geological Mapping of the\nAboresheed Area", 20, False),
    ("Using PRISMA Satellite Data", 16, False),
    ("", 6, False),
    ("A Comparative Study of Machine Learning\nand Deep Learning Classifiers", 12, True),
    ("", 8, False),
    ("Experiment: experiments_005  ·  PRISMA L2D  ·  25 July 2020", 9, True),
    ("", 4, False),
    ("Prepared by: Nour Laban  |  May 2026", 10, False),
]:
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    run = p.add_run(text)
    run.font.name   = "Calibri"
    run.font.size   = Pt(size)
    run.font.color.rgb = WHITE
    run.bold   = (size >= 16 and not italic)
    run.italic = italic

doc.add_paragraph().paragraph_format.space_after = Pt(12)

# ═══════════════════════════════════════════════════════════════════════════════
#  1. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "1.  Executive Summary", 1)
add_body(doc,
    "This report presents a pixel-based hyperspectral image classification study applied to "
    "the Aboresheed geological area, Eastern Desert of Egypt, using Level 2D PRISMA satellite "
    "imagery acquired on 25 July 2020. Four classifiers were benchmarked: Random Forest (RF), "
    "XGBoost, SpectralFormer (Vision Transformer), and S²Mamba (Spectral-Spatial State Space "
    "Model). The S²Mamba model achieved the highest performance with an Overall Accuracy (OA) "
    "of 99.24 % and an IoU score of 97.80 % across 13 geological classes. Results confirm the "
    "superior capability of selective state space models for lithological discrimination from "
    "hyperspectral imagery."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  2. STUDY AREA & DATASET
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "2.  Study Area & Dataset", 1)

add_heading(doc, "2.1  Study Area", 2)
add_body(doc,
    "The study area is located in the Aboresheed region of the Eastern Desert of Egypt — "
    "a Precambrian basement terrain characterised by diverse igneous and metamorphic "
    "lithological units including alkaline granites, gabbros, serpentinites, gneisses, "
    "pegmatites, and Quaternary wadi deposits."
)

add_heading(doc, "2.2  PRISMA Satellite Data", 2)
styled_table(doc,
    ["Parameter", "Value"],
    [
        ["Sensor",                "PRISMA (PRecursore IperSpettrale della Missione Applicativa)"],
        ["Agency",                "Italian Space Agency (ASI)"],
        ["Processing Level",      "L2D Standard"],
        ["Scene ID",              "PRS_L2D_STD_20200725083506_20200725083510_0001.he5"],
        ["Acquisition Date",      "25 July 2020"],
        ["Data Format",           "HDF5 (.he5)"],
        ["Raw VNIR Bands",        "66"],
        ["Raw SWIR Bands",        "173"],
        ["Bands after Removal",   "61 VNIR + 107 SWIR = 168 usable bands"],
        ["CRS",                   "EPSG:4326"],
    ],
    col_widths=[5.5, 11]
)

add_heading(doc, "2.3  Ground Truth & Classes", 2)
add_body(doc,
    "Ground truth polygons were digitised in the shapefile PRISMA_18_2_2026.shp (collected "
    "18 February 2026). The attribute field Classcode encodes 13 lithological classes:"
)
styled_table(doc,
    ["ID", "Class Name", "Colour (RGB)"],
    [
        ["1",  "Alkali feldspar granites",       "(220, 80,  60)"],
        ["2",  "Alteration zones",               "(255, 165,  0)"],
        ["3",  "Biotite granites",               "(139, 90,  43)"],
        ["4",  "Gabbro",                         "( 60,  60, 180)"],
        ["5",  "Garnet-muscovite granites",      "(180, 100, 200)"],
        ["6",  "Gneiss",                         "(130, 130, 130)"],
        ["7",  "Li-pegmatite granites",          "(255, 220,  50)"],
        ["8",  "Serpentinite",                   "( 50, 160,  90)"],
        ["9",  "Tonalite-granodiorite",          "(210, 140,  70)"],
        ["10", "Wadi deposits",                  "(200, 230, 150)"],
        ["11", "Zinnwaldite-muscovite granites", "(100, 180, 220)"],
        ["12", "Ophiolitic metagabbro",          "( 30, 100, 140)"],
        ["13", "Background",                     "(240, 240, 240)"],
    ],
    col_widths=[1.5, 9, 4.5]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  3. METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "3.  Methodology", 1)

add_heading(doc, "3.1  Preprocessing Pipeline", 2)
steps = [
    "HDF5 → GeoTIFF Conversion: VNIR and SWIR cubes extracted, transposed to (H, W, B), and concatenated.",
    "Band Removal: 5 noisy VNIR bands and 66 SWIR water-vapour/noise bands removed → 168 clean bands.",
    "Nodata Masking: All-zero pixels flagged with nodata = −9999.0.",
    "MNF Transform: 10 most informative Minimum Noise Fraction components appended to the spectral stack.",
    "Label Rasterisation: Shapefile polygons burned to a pixel grid matching the PRISMA scene.",
    "Stratified Train/Test Split: 30 % training / 70 % test, up to 1,000 samples per class, random_state = 42.",
    "Per-Band Min-Max Normalisation: Each band independently scaled to [0, 1], nodata pixels excluded.",
    "Mirror Padding: Applied at image borders for patch-based deep learning extraction.",
]
for s in steps:
    add_bullet(doc, s)

add_heading(doc, "3.2  Model Hyperparameters", 2)

add_heading(doc, "Random Forest", 3)
styled_table(doc,
    ["Parameter", "Value"],
    [["n_estimators", "10"], ["random_state", "42"], ["Input", "Flattened pixel spectra (168 features)"]],
    col_widths=[6, 10]
)

add_heading(doc, "XGBoost", 3)
styled_table(doc,
    ["Parameter", "Value"],
    [
        ["n_estimators", "200"], ["max_depth", "6"], ["learning_rate", "0.1"],
        ["subsample", "0.8"], ["colsample_bytree", "0.8"],
        ["random_state", "42"], ["n_jobs", "−1 (all cores)"],
    ],
    col_widths=[6, 10]
)

add_heading(doc, "SpectralFormer (Vision Transformer)", 3)
styled_table(doc,
    ["Parameter", "Value"],
    [
        ["Patch size", "5 × 5 pixels"], ["Band patches", "3"], ["Hidden dim", "64"],
        ["Transformer depth", "5 layers"], ["Attention heads", "4"], ["MLP dim", "8"],
        ["Dropout", "0.1"], ["Epochs", "200"], ["Batch size", "128"],
        ["Optimizer", "Adam"], ["Learning rate", "5 × 10⁻⁴"],
        ["LR Scheduler", "StepLR (γ = 0.9)"], ["Loss", "CrossEntropyLoss (class-weighted)"],
    ],
    col_widths=[6, 10]
)

add_heading(doc, "S²Mamba (Spectral-Spatial State Space Model)", 3)
styled_table(doc,
    ["Parameter", "Value"],
    [
        ["Patch size", "3 × 3 pixels"], ["Embedding dim", "64"],
        ["Mamba layers", "[1]"], ["State dimension", "16"],
        ["Drop path rate", "0.1"], ["Epochs", "200"], ["Batch size", "64"],
        ["Optimizer", "AdamW"], ["Learning rate", "5 × 10⁻⁴"],
        ["Weight decay", "5 × 10⁻³"], ["LR Scheduler", "ExponentialLR (γ = 0.99)"],
        ["Loss", "CrossEntropyLoss (class-weighted)"],
    ],
    col_widths=[6, 10]
)

add_heading(doc, "3.3  Key Libraries", 2)
styled_table(doc,
    ["Library", "Role"],
    [
        ["rasterio",          "GeoTIFF I/O and georeferencing"],
        ["h5py",              "PRISMA HDF5 reading"],
        ["geopandas",         "Shapefile handling and reprojection"],
        ["numpy",             "Array and linear-algebra operations"],
        ["scikit-learn",      "RF classifier, metrics, normalisation"],
        ["xgboost",           "XGBoost gradient-boosted trees"],
        ["PyTorch (torch)",   "Deep learning model training and inference"],
        ["timm",              "Transformer building blocks (DropPath, trunc_normal_)"],
        ["einops",            "Tensor rearrangement for patch operations"],
        ["mamba_ssm",         "Mamba selective scan CUDA kernels"],
        ["matplotlib/seaborn","Confusion matrices and thematic map visualisation"],
        ["joblib",            "Model serialisation (.pkl)"],
    ],
    col_widths=[5, 11]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  4. RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4.  Results", 1)

add_heading(doc, "4.1  Overall Performance Summary", 2)
add_body(doc, "Evaluation set: 7,876 pixels across 13 classes (70 % stratified hold-out).")
styled_table(doc,
    ["Model", "Overall Accuracy (OA)", "IoU (Jaccard)", "Macro F1"],
    [
        ["S²Mamba (Best)",          "99.24 %", "97.80 %", "98.87 %"],
        ["SpectralFormer (ViT)",    "98.25 %", "94.82 %", "97.22 %"],
        ["XGBoost",                 "98.12 %", "94.81 %", "97.25 %"],
        ["Random Forest",           "94.54 %", "87.75 %", "93.19 %"],
    ],
    col_widths=[5.5, 4.5, 4, 3.5]
)

add_heading(doc, "4.2  Per-Class Results — S²Mamba", 2)
styled_table(doc,
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    [
        ["Alkali feldspar granites",       "0.9943", "0.9986", "0.9964", "700"],
        ["Alteration zones",              "0.9841", "0.9688", "0.9764", "128"],
        ["Biotite granites",              "0.9749", "0.9986", "0.9866", "700"],
        ["Gabbro",                        "0.9915", "0.9971", "0.9943", "700"],
        ["Garnet-muscovite granites",     "0.9956", "0.9600", "0.9775", "700"],
        ["Gneiss",                        "0.9986", "0.9843", "0.9914", "700"],
        ["Li-pegmatite granites",         "1.0000", "0.9986", "0.9993", "700"],
        ["Serpentinite",                  "0.9983", "1.0000", "0.9991", "577"],
        ["Tonalite-granodiorite",         "0.9929", "0.9929", "0.9929", "700"],
        ["Wadi deposits",                 "1.0000", "0.9929", "0.9964", "700"],
        ["Zinnwaldite-muscovite granites","0.9037", "0.9883", "0.9441", "171"],
        ["Ophiolitic metagabbro",         "0.9972", "1.0000", "0.9986", "700"],
        ["Background",                    "1.0000", "1.0000", "1.0000", "700"],
        ["Weighted Average",              "0.9926", "0.9924", "0.9924", "7876"],
    ],
    col_widths=[7, 2.5, 2.5, 2.5, 2]
)

add_heading(doc, "4.3  Per-Class Results — SpectralFormer (ViT)", 2)
styled_table(doc,
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    [
        ["Alkali feldspar granites",       "0.9985", "0.9786", "0.9885", "700"],
        ["Alteration zones",              "0.8601", "0.9609", "0.9077", "128"],
        ["Biotite granites",              "0.9970", "0.9486", "0.9722", "700"],
        ["Gabbro",                        "0.9721", "0.9971", "0.9845", "700"],
        ["Garnet-muscovite granites",     "0.9626", "0.9571", "0.9599", "700"],
        ["Gneiss",                        "0.9689", "0.9786", "0.9737", "700"],
        ["Li-pegmatite granites",         "1.0000", "0.9914", "0.9957", "700"],
        ["Serpentinite",                  "1.0000", "1.0000", "1.0000", "577"],
        ["Tonalite-granodiorite",         "0.9804", "1.0000", "0.9901", "700"],
        ["Wadi deposits",                 "0.9957", "0.9829", "0.9892", "700"],
        ["Zinnwaldite-muscovite granites","0.8404", "0.9240", "0.8802", "171"],
        ["Ophiolitic metagabbro",         "1.0000", "0.9943", "0.9971", "700"],
        ["Background",                    "1.0000", "1.0000", "1.0000", "700"],
        ["Weighted Average",              "0.9832", "0.9825", "0.9826", "7876"],
    ],
    col_widths=[7, 2.5, 2.5, 2.5, 2]
)

add_heading(doc, "4.4  Per-Class Results — XGBoost", 2)
styled_table(doc,
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    [
        ["Alkali feldspar granites",       "0.9872", "0.9929", "0.9900", "700"],
        ["Alteration zones",              "0.9187", "0.8828", "0.9004", "128"],
        ["Biotite granites",              "0.9599", "0.9571", "0.9585", "700"],
        ["Gabbro",                        "0.9734", "0.9929", "0.9830", "700"],
        ["Garnet-muscovite granites",     "0.9839", "0.9614", "0.9725", "700"],
        ["Gneiss",                        "0.9687", "0.9729", "0.9708", "700"],
        ["Li-pegmatite granites",         "0.9943", "1.0000", "0.9972", "700"],
        ["Serpentinite",                  "0.9948", "0.9913", "0.9931", "577"],
        ["Tonalite-granodiorite",         "0.9814", "0.9786", "0.9800", "700"],
        ["Wadi deposits",                 "0.9913", "0.9786", "0.9849", "700"],
        ["Zinnwaldite-muscovite granites","0.8989", "0.9357", "0.9169", "171"],
        ["Ophiolitic metagabbro",         "0.9929", "0.9986", "0.9957", "700"],
        ["Background",                    "1.0000", "1.0000", "1.0000", "700"],
        ["Weighted Average",              "0.9813", "0.9812", "0.9812", "7876"],
    ],
    col_widths=[7, 2.5, 2.5, 2.5, 2]
)

add_heading(doc, "4.5  Per-Class Results — Random Forest", 2)
styled_table(doc,
    ["Class", "Precision", "Recall", "F1-Score", "Support"],
    [
        ["Alkali feldspar granites",       "0.9339", "0.9686", "0.9509", "700"],
        ["Alteration zones",              "0.8942", "0.7266", "0.8017", "128"],
        ["Biotite granites",              "0.8411", "0.8771", "0.8587", "700"],
        ["Gabbro",                        "0.9501", "0.9786", "0.9641", "700"],
        ["Garnet-muscovite granites",     "0.9118", "0.9014", "0.9066", "700"],
        ["Gneiss",                        "0.9183", "0.9471", "0.9325", "700"],
        ["Li-pegmatite granites",         "0.9826", "0.9686", "0.9755", "700"],
        ["Serpentinite",                  "0.9895", "0.9809", "0.9852", "577"],
        ["Tonalite-granodiorite",         "0.9511", "0.9171", "0.9338", "700"],
        ["Wadi deposits",                 "0.9851", "0.9471", "0.9658", "700"],
        ["Zinnwaldite-muscovite granites","0.8931", "0.8304", "0.8606", "171"],
        ["Ophiolitic metagabbro",         "0.9705", "0.9871", "0.9788", "700"],
        ["Background",                    "1.0000", "1.0000", "1.0000", "700"],
        ["Weighted Average",              "0.9458", "0.9454", "0.9453", "7876"],
    ],
    col_widths=[7, 2.5, 2.5, 2.5, 2]
)

# ═══════════════════════════════════════════════════════════════════════════════
#  4.6  FIGURES — Thematic Classification Maps
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "4.6  Thematic Classification Maps", 2)
add_body(doc,
    "Figures 1–5 present the lithological classification maps generated by each model "
    "for the Aboresheed area, Aboresheed, Eastern Desert of Egypt. "
    "All maps are derived from PRISMA L2D imagery acquired on 25 July 2020 "
    "(Scene ID: PRS_L2D_STD_20200725083506_20200725083510_0001.he5). "
    "The colour legend follows the geology_classes.clr palette defined for 13 lithological classes."
)

add_figure(doc,
    os.path.join(FIG, "comparison_panel.png"),
    "Figure 1. Side-by-side comparison panel of all four classified geological maps "
    "(Random Forest, XGBoost, SpectralFormer, S²Mamba). "
    "Study area: Aboresheed, Eastern Desert, Egypt. "
    "Source: PRISMA L2D, 25 July 2020. "
    "Experiment: experiments_005.",
    width_cm=15.5
)

add_figure(doc,
    os.path.join(FIG, "MAMBA_final_prediction_map.png"),
    "Figure 2. S²Mamba (Spectral-Spatial State Space Model) geological classification map. "
    "OA = 99.24 %  |  IoU = 97.80 %  |  Macro F1 = 98.87 %. "
    "Patch size: 3×3 px · Embedding dim: 64 · Epochs: 200 · Optimiser: AdamW. "
    "Study area: Aboresheed, Eastern Desert, Egypt. Source: PRISMA L2D, 25 July 2020.",
    width_cm=15
)

add_figure(doc,
    os.path.join(FIG, "VIT_final_prediction_map.png"),
    "Figure 3. SpectralFormer (Vision Transformer) geological classification map. "
    "OA = 98.25 %  |  IoU = 94.82 %  |  Macro F1 = 97.22 %. "
    "Patch size: 5×5 px · Depth: 5 layers · Heads: 4 · Epochs: 200 · Optimiser: Adam. "
    "Study area: Aboresheed, Eastern Desert, Egypt. Source: PRISMA L2D, 25 July 2020.",
    width_cm=15
)

add_figure(doc,
    os.path.join(FIG, "xgb_final_prediction_map.png"),
    "Figure 4. XGBoost geological classification map. "
    "OA = 98.12 %  |  IoU = 94.81 %  |  Macro F1 = 97.25 %. "
    "n_estimators: 200 · max_depth: 6 · learning_rate: 0.1. "
    "Study area: Aboresheed, Eastern Desert, Egypt. Source: PRISMA L2D, 25 July 2020.",
    width_cm=15
)

add_figure(doc,
    os.path.join(FIG, "rf_final_prediction_map.png"),
    "Figure 5. Random Forest geological classification map. "
    "OA = 94.54 %  |  IoU = 87.75 %  |  Macro F1 = 93.19 %. "
    "n_estimators: 10 · random_state: 42. "
    "Study area: Aboresheed, Eastern Desert, Egypt. Source: PRISMA L2D, 25 July 2020.",
    width_cm=15
)

add_heading(doc, "4.7  Confusion Matrices", 2)
add_body(doc,
    "Row-normalised confusion matrices (recall %) are shown for each model in Figures 6–9. "
    "Diagonal values represent correct classification recall per class; "
    "off-diagonal values indicate inter-class confusion."
)
cm_files = [
    (os.path.join(MDL, "mamba_model_evaluation",         "mamba_confusion_matrix.png"),
     "Figure 6. S²Mamba row-normalised confusion matrix. "
     "OA = 99.24 %  |  IoU = 97.80 %  |  Macro F1 = 98.87 %. "
     "Aboresheed area, PRISMA L2D, 25 July 2020."),
    (os.path.join(MDL, "VITransformer_model_evaluation", "VITransformer_confusion_matrix.png"),
     "Figure 7. SpectralFormer (ViT) row-normalised confusion matrix. "
     "OA = 98.25 %  |  IoU = 94.82 %  |  Macro F1 = 97.22 %. "
     "Aboresheed area, PRISMA L2D, 25 July 2020."),
    (os.path.join(MDL, "xgb_model_evaluation",           "confusion_matrix.png"),
     "Figure 8. XGBoost row-normalised confusion matrix. "
     "OA = 98.12 %  |  IoU = 94.81 %  |  Macro F1 = 97.25 %. "
     "Aboresheed area, PRISMA L2D, 25 July 2020."),
    (os.path.join(MDL, "rf_model_evaluation",             "confusion_matrix.png"),
     "Figure 9. Random Forest row-normalised confusion matrix. "
     "OA = 94.54 %  |  IoU = 87.75 %  |  Macro F1 = 93.19 %. "
     "Aboresheed area, PRISMA L2D, 25 July 2020."),
]
for img, caption in cm_files:
    add_figure(doc, img, caption, width_cm=14)

# ═══════════════════════════════════════════════════════════════════════════════
#  5. SCIENTIFIC BACKGROUND
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "5.  Scientific Background", 1)
add_body(doc,
    "This section presents the theoretical foundations of the two deep learning architectures "
    "employed in this study: the SpectralFormer (Vision Transformer) and S²Mamba "
    "(Spectral-Spatial State Space Model). Figure S1 provides an overview of the performance "
    "of all four classifiers benchmarked in this work."
)

add_figure(doc,
    os.path.join(SCI, "fig1_performance_comparison.png"),
    "Figure S1. Overall Accuracy, IoU, and Macro F1 scores for all four classifiers. "
    "S²Mamba achieves the highest scores across all three metrics.",
    width_cm=15
)

add_figure(doc,
    os.path.join(SCI, "fig7_spectral_profiles.png"),
    "Figure S2. Simulated PRISMA spectral reflectance signatures for six representative "
    "lithological classes across the 400–2500 nm range (168 bands after removal of "
    "noise and water-vapour absorption regions, shown in grey).",
    width_cm=15
)

# ── 5.1 SpectralFormer ──────────────────────────────────────────────────────
add_heading(doc, "5.1  SpectralFormer — Vision Transformer for Hyperspectral Data", 2)
add_body(doc,
    "The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2021) and adapted for "
    "hyperspectral remote sensing by Hong et al. (2021) as SpectralFormer, re-formulates "
    "spectral classification as a sequence modelling problem. Instead of processing image "
    "patches spatially, SpectralFormer treats each spectral band group as a token, enabling "
    "the self-attention mechanism to capture long-range spectral dependencies across the "
    "entire wavelength range in a single forward pass."
)

add_heading(doc, "5.1.1  Token Embedding & Positional Encoding", 3)
add_body(doc,
    "The hyperspectral pixel vector of B bands is divided into overlapping spectral groups "
    "(band_patches = 3). Each group is linearly projected into a d-dimensional embedding "
    "(d = 64). A learnable [CLS] token is prepended to aggregate class information. "
    "Sinusoidal positional encodings are added to preserve the spectral ordering of tokens, "
    "since self-attention is permutation-invariant by design."
)

add_heading(doc, "5.1.2  Multi-Head Self-Attention", 3)
add_body(doc,
    "Each transformer layer applies Multi-Head Self-Attention (MHSA) with h = 4 heads. "
    "For each head, the input sequence X is projected into Query (Q), Key (K), and Value (V) "
    "matrices via learned weight matrices W_Q, W_K, W_V:"
)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after  = Pt(4)
r = p.add_run("Attention(Q, K, V)  =  softmax( QKᵀ / √dₖ ) · V")
r.font.name = "Courier New"; r.font.size = Pt(11); r.bold = True
r.font.color.rgb = RGBColor(0x6C, 0x34, 0x83)

add_body(doc,
    "where dₖ = d / h is the per-head key dimension. The scaling factor 1/√dₖ prevents "
    "vanishing gradients in the softmax when dₖ is large. The outputs of all heads are "
    "concatenated and projected back to dimension d. Five such transformer layers are stacked "
    "(depth = 5), each followed by a feed-forward MLP of dimension 8 and residual "
    "connections with LayerNorm (Pre-LN configuration)."
)

add_figure(doc,
    os.path.join(SCI, "fig2_spectralformer_architecture.png"),
    "Figure S3. SpectralFormer end-to-end architecture: spectral token embedding → "
    "positional encoding → 5× transformer encoder layers (MHSA + FFN + residuals) → "
    "MLP classification head. The attention formula is shown at the bottom.",
    width_cm=15.5
)

add_figure(doc,
    os.path.join(SCI, "fig3_attention_detail.png"),
    "Figure S4. Left: simulated attention weight matrix across 12 spectral tokens — "
    "warmer colours indicate stronger spectral band dependencies. "
    "Right: Q·K·V projection schematic showing how input tokens are transformed "
    "into queries, keys, and values before computing scaled dot-product attention.",
    width_cm=15
)

add_heading(doc, "5.1.3  Classification Head & Training", 3)
add_body(doc,
    "The [CLS] token representation from the final transformer layer is passed through a "
    "two-layer MLP classifier followed by a Softmax activation to produce class probabilities. "
    "Training used the Adam optimiser (lr = 5×10⁻⁴) with a StepLR scheduler (γ = 0.9), "
    "class-weighted CrossEntropyLoss, and dropout = 0.1 for regularisation over 200 epochs "
    "with batch size 128. Spatial context is captured through 5×5 pixel patch extraction "
    "with mirror-padding at image borders."
)

# ── 5.2 S²Mamba ─────────────────────────────────────────────────────────────
add_heading(doc, "5.2  S²Mamba — Spectral-Spatial State Space Model", 2)
add_body(doc,
    "S²Mamba (Spectral-Spatial Mamba) is a deep learning architecture built on the "
    "Structured State Space Model (SSM) framework, specifically the Mamba variant introduced "
    "by Gu & Dao (2023). Unlike Transformers, which compute global attention with O(N²) "
    "complexity in sequence length, Mamba achieves O(N) linear complexity through a "
    "hardware-aware selective scan operation, making it particularly efficient for long "
    "hyperspectral sequences."
)

add_heading(doc, "5.2.1  State Space Model Foundation", 3)
add_body(doc,
    "The SSM maps an input sequence x(t) ∈ ℝ to an output y(t) ∈ ℝ through a latent "
    "state vector h(t) ∈ ℝᴺ governed by a system of linear ODEs:"
)
for eq in [
    "h′(t) = A h(t) + B x(t)    [State update equation]",
    "y(t)  = C h(t)              [Output projection]",
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(eq)
    r.font.name = "Courier New"; r.font.size = Pt(10.5); r.bold = True
    r.font.color.rgb = RGBColor(0xE8, 0x6C, 0x1E)

add_body(doc,
    "where A ∈ ℝᴺˣᴺ is a learnable state transition matrix (initialised as HiPPO matrix "
    "for long-range memory), B ∈ ℝᴺˣ¹ is the input projection, and C ∈ ℝ¹ˣᴺ is the "
    "output projection. For discrete sequences (spectral bands), the continuous system "
    "is discretised via a learnable timescale parameter Δ (zero-order hold):"
)
for eq in [
    "Ā = exp(ΔA)",
    "B̄ = (ΔA)⁻¹ (exp(ΔA) − I) ΔB",
    "hₜ = Āhₜ₋₁ + B̄xₜ  ,   yₜ = Chₜ",
]:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    r = p.add_run(eq)
    r.font.name = "Courier New"; r.font.size = Pt(10.5); r.bold = True
    r.font.color.rgb = RGBColor(0xE8, 0x6C, 0x1E)

add_figure(doc,
    os.path.join(SCI, "fig5_ssm_state_diagram.png"),
    "Figure S5. Left: continuous-time SSM showing the recurrent state h(t) driven by "
    "input x(t) via matrices A, B, C. Right: discretised selective scan unrolled over "
    "spectral tokens t-2 … t+1, where Ā and B̄ are input-dependent (selective).",
    width_cm=15
)

add_heading(doc, "5.2.2  Selective Scan Mechanism", 3)
add_body(doc,
    "The key innovation of Mamba over prior SSMs is that the parameters B, C, and Δ are "
    "functions of the input x rather than fixed constants. This 'selectivity' allows the "
    "model to dynamically control which spectral information to retain or forget at each "
    "step — analogous to the gating in LSTMs, but implemented via a hardware-aware parallel "
    "scan algorithm that avoids materialising the full O(N²) recurrence matrix. "
    "The state dimension d_state = 16 is kept small relative to the model dimension (64) "
    "to balance expressivity and efficiency."
)

add_heading(doc, "5.2.3  Spectral-Spatial Integration", 3)
add_body(doc,
    "S²Mamba processes 3×3 spatial patches (9 pixels × 168 spectral bands) by flattening "
    "and projecting them into an embedding of dimension 64. The single S²Mamba block "
    "applies the selective SSM in a gated architecture: the input is split into two "
    "parallel branches — one passed through the SSM (spectral pathway) and one through a "
    "SiLU activation (spatial pathway). Their Hadamard product fuses spectral and spatial "
    "features before the output linear projection. A residual connection wraps the entire "
    "block to preserve gradient flow."
)

add_figure(doc,
    os.path.join(SCI, "fig4_s2mamba_architecture.png"),
    "Figure S6. S²Mamba block architecture: input patch → LayerNorm → dual-branch "
    "(SSM selective scan + SiLU activation) → Hadamard product → linear projection "
    "→ residual addition. The discretised SSM equations are shown at the bottom.",
    width_cm=15.5
)

add_heading(doc, "5.2.4  Training Configuration", 3)
add_body(doc,
    "S²Mamba was trained with the AdamW optimiser (lr = 5×10⁻⁴, weight_decay = 5×10⁻³) "
    "and an ExponentialLR scheduler (γ = 0.99) over 200 epochs, batch size 64. "
    "Class-weighted CrossEntropyLoss addressed the imbalance in class support "
    "(128–700 samples). Dropout = 0.1 and drop_path_rate = 0.1 were applied for "
    "regularisation. Model weights were checkpointed at the epoch achieving the lowest "
    "validation loss."
)

add_heading(doc, "5.3  Transformer vs State Space Model — Key Differences", 2)
styled_table(doc,
    ["Aspect", "SpectralFormer (ViT)", "S²Mamba"],
    [
        ["Complexity",        "O(N²) — quadratic in sequence length", "O(N) — linear in sequence length"],
        ["Global context",    "Full attention over all tokens",        "Selective recurrent scan"],
        ["Inductive bias",    "Permutation-invariant",                 "Ordered sequence (causal)"],
        ["Spatial patch",     "5×5 pixels",                           "3×3 pixels"],
        ["Key parameter",     "Attention weights (softmax)",           "State matrix A, selectivity Δ"],
        ["Memory",            "O(N) KV-cache",                        "O(d_state) fixed-size state"],
        ["OA (this study)",   "98.25 %",                              "99.24 % ✓"],
        ["IoU (this study)",  "94.82 %",                              "97.80 % ✓"],
    ],
    col_widths=[4, 6.5, 6.5]
)

add_figure(doc,
    os.path.join(SCI, "fig6_complexity_and_class_comparison.png"),
    "Figure S7. Left: computational complexity scaling — Transformer O(N²) vs S²Mamba O(N) "
    "as spectral sequence length increases. Right: per-class F1-score comparison between "
    "S²Mamba and Random Forest across all 13 lithological classes.",
    width_cm=15.5
)

# ═══════════════════════════════════════════════════════════════════════════════
#  6. DISCUSSION (renumbered)
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "6.  Discussion", 1)

add_heading(doc, "6.1  Model Comparison", 2)
add_body(doc,
    "S²Mamba achieved the best performance (OA = 99.24 %, IoU = 97.80 %), outperforming all "
    "other models by a clear margin. The selective state space mechanism efficiently captures "
    "long-range spectral dependencies and local 3×3 spatial context simultaneously — critical "
    "in hyperspectral geological mapping where subtle spectral gradients between similar "
    "lithologies govern class boundaries."
)
add_body(doc,
    "SpectralFormer (98.25 %) and XGBoost (98.12 %) performed nearly identically, which is "
    "noteworthy: a classical ensemble method almost matched a purpose-built hyperspectral "
    "transformer. This is attributable to the high spectral discriminability of the PRISMA "
    "sensor (168 clean bands) and high-quality field-validated training labels."
)
add_body(doc,
    "Random Forest (94.54 %) provided a solid baseline but showed expected limitations in "
    "spectrally similar classes such as Biotite granites (F1 = 0.8587) and Alteration zones "
    "(F1 = 0.8017), where shallow ensemble trees cannot resolve fine spectral gradients."
)

add_heading(doc, "6.2  Challenging Classes", 2)
add_body(doc,
    "Across all models the most difficult class was Zinnwaldite-muscovite granites (support = "
    "171 samples), showing the lowest F1 scores: 0.9441 (S²Mamba), 0.8802 (ViT), "
    "0.9169 (XGBoost), 0.8606 (RF). This is expected given spectral overlap with other "
    "granitic units and limited training support. Alteration zones also showed consistently "
    "reduced performance, reflecting the mixed spectral signature of hydrothermally altered "
    "rock."
)
add_body(doc,
    "Classes with perfect or near-perfect classification across all models included "
    "Li-pegmatite granites, Wadi deposits, Ophiolitic metagabbro, and Background — confirming "
    "strong spectral separability in the PRISMA wavelength range."
)

add_heading(doc, "6.3  PRISMA Sensor Suitability", 2)
add_body(doc,
    "The PRISMA sensor's contiguous VNIR+SWIR coverage (168 clean bands from ~400–2500 nm) "
    "proved highly effective for discriminating the granite rock types in the Aboresheed "
    "basement complex. Removal of 71 noise-affected and water-vapour absorption bands was "
    "critical to model stability. The MNF transform, appending 10 high-SNR components, "
    "further enhanced the spectral signal quality and contributed to improved classification "
    "performance in all models."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  6. CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "7.  Conclusions", 1)
conclusions = [
    "S²Mamba is the recommended model for operational lithological mapping with PRISMA data, "
    "achieving 99.24 % OA and 97.80 % IoU — a 4.70 percentage-point improvement over Random Forest.",
    "XGBoost is a strong classical alternative (98.12 % OA) with significantly lower "
    "computational cost than deep learning approaches, suitable for rapid mapping workflows.",
    "The HyperLithoMap framework successfully automates the full pipeline: HDF5 ingestion → band "
    "removal → MNF augmentation → label rasterisation → stratified sampling → model training "
    "→ evaluation output generation.",
    "Alteration zones and Zinnwaldite-muscovite granites remain the most challenging classes "
    "due to spectral similarity and limited training samples. Targeted field sampling for "
    "these units is recommended to improve future classification runs.",
]
for c in conclusions:
    add_bullet(doc, c)

# ═══════════════════════════════════════════════════════════════════════════════
#  7. REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
add_heading(doc, "8.  Reproducibility", 1)
styled_table(doc,
    ["Item", "Value"],
    [
        ["Code repository",    "this repository (main.py run-all --dataset PRISMA)"],
        ["Experiment folder",  "data/aboresheadmamabadataPrisma/output/experiments_005"],
        ["Saved models",       "experiments_005/saved_models/"],
        ["Random seed",        "42 (all models)"],
        ["Python environment", "myenv  (see requirements.txt)"],
        ["Entry point",        "python main.py run-all --satellite PRISMA --verbose"],
    ],
    col_widths=[5.5, 11]
)

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— End of Technical Report —")
r.italic = True
r.font.size = Pt(9)
r.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
r.font.name = "Calibri"

doc.save(OUT)
print(f"Saved: {OUT}")
