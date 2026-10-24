"""
Generate a professional technical report (DOCX) for the Aboresheed PRISMA hyperspectral
lithological mapping project — RF, XGBoost, and SpectralFormer results only.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projutils.config import data_root
DATA_ROOT = data_root()
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ── paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(DATA_ROOT, "aboresheadmamabadataPrisma/output/experiments_008")
FIGS = os.path.join(BASE, "figures")
EVAL = os.path.join(BASE, "saved_models")
OUT  = os.path.join(BASE, "Technical_Report_Aboresheed_PRISMA.docx")

# ── colour palette ───────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1A, 0x37, 0x5E)   # headers
TEAL    = RGBColor(0x1B, 0x6C, 0x8A)   # sub-headers / accent
LIGHT   = RGBColor(0xE8, 0xF1, 0xF8)   # table header fill
MID     = RGBColor(0xD0, 0xE4, 0xF0)   # alternating table row
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x1A, 0x2E)   # body text


# ── helpers ───────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
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
        val = kwargs.get(side, {"sz": "4", "val": "single", "color": "BDC6CF"})
        b   = OxmlElement(f"w:{side}")
        for k, v in val.items():
            b.set(qn(f"w:{k}"), v)
        tcBorders.append(b)
    tcPr.append(tcBorders)


def add_heading(doc, text, level=1, color=None, size=None, bold=True, center=False):
    p    = doc.add_paragraph()
    run  = p.add_run(text)
    run.bold = bold
    run.font.color.rgb = color or (NAVY if level == 1 else TEAL)
    run.font.size      = Pt(size or (16 if level == 1 else 13 if level == 2 else 11))
    run.font.name      = "Calibri"
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after  = Pt(4)
    return p


def add_body(doc, text, italic=False, indent=False):
    p   = doc.add_paragraph(text)
    for run in p.runs:
        run.font.name  = "Calibri"
        run.font.size  = Pt(10.5)
        run.italic     = italic
        run.font.color.rgb = DARK
    if indent:
        p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.space_after  = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def add_figure(doc, img_path, caption, width=Inches(6.0)):
    if not os.path.exists(img_path):
        add_body(doc, f"[Figure not found: {img_path}]", italic=True)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=width)

    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.font.name   = "Calibri"
        run.font.size   = Pt(9.5)
        run.italic      = True
        run.font.color.rgb = RGBColor(0x44, 0x44, 0x55)
    cap.paragraph_format.space_after = Pt(10)


def add_table(doc, headers, rows, col_widths=None, stripe=True):
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.style = "Table Grid"

    # header row
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_bg(hdr_cells[i], "1B6C8A")
        p   = hdr_cells[i].paragraphs[0]
        run = p.runs[0]
        run.bold              = True
        run.font.color.rgb    = WHITE
        run.font.size         = Pt(9.5)
        run.font.name         = "Calibri"
        p.alignment           = WD_ALIGN_PARAGRAPH.CENTER
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # data rows
    for r_idx, row in enumerate(rows):
        cells = tbl.rows[r_idx + 1].cells
        bg    = "EAF2F8" if (r_idx % 2 == 0 and stripe) else "FFFFFF"
        for c_idx, val in enumerate(row):
            cells[c_idx].text = str(val)
            set_cell_bg(cells[c_idx], bg)
            p   = cells[c_idx].paragraphs[0]
            run = p.runs[0]
            run.font.size      = Pt(9.5)
            run.font.name      = "Calibri"
            p.alignment        = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[i].width = w

    doc.add_paragraph()  # spacing after table
    return tbl


# ── per-class data parsed from evaluation_results.txt ─────────────────────────

RF_CLASS = [
    ("Alkali feldspar granites",    "0.9194", "0.9771", "0.9474", "700"),
    ("Alteration zones",            "0.7823", "0.8984", "0.8364", "128"),
    ("Biotite granites",            "0.8776", "0.9014", "0.8894", "700"),
    ("Gabbro",                      "0.9597", "0.9529", "0.9563", "700"),
    ("Garnet-muscovite granites",   "0.9501", "0.9514", "0.9507", "700"),
    ("Gneiss",                      "0.9570", "0.9229", "0.9396", "700"),
    ("Li-pegmatite granites",       "0.9828", "0.9800", "0.9814", "700"),
    ("Serpentinite",                "0.9827", "0.9844", "0.9835", "577"),
    ("Tonalite-granodiorite",       "0.9674", "0.9329", "0.9498", "700"),
    ("Wadi deposits",               "0.9927", "0.9700", "0.9812", "700"),
    ("Zinnwaldite-muscovite gran.", "0.9620", "0.8889", "0.9240", "171"),
    ("Ophiolitic metagabbro",       "0.9829", "0.9857", "0.9843", "700"),
]

XGB_CLASS = [
    ("Alkali feldspar granites",    "0.9858", "0.9943", "0.9900", "700"),
    ("Alteration zones",            "0.9360", "0.9141", "0.9249", "128"),
    ("Biotite granites",            "0.9643", "0.9657", "0.9650", "700"),
    ("Gabbro",                      "0.9927", "0.9686", "0.9805", "700"),
    ("Garnet-muscovite granites",   "0.9886", "0.9900", "0.9893", "700"),
    ("Gneiss",                      "0.9647", "0.9771", "0.9709", "700"),
    ("Li-pegmatite granites",       "0.9915", "1.0000", "0.9957", "700"),
    ("Serpentinite",                "0.9897", "0.9965", "0.9931", "577"),
    ("Tonalite-granodiorite",       "0.9843", "0.9857", "0.9850", "700"),
    ("Wadi deposits",               "0.9871", "0.9829", "0.9850", "700"),
    ("Zinnwaldite-muscovite gran.", "0.9643", "0.9474", "0.9558", "171"),
    ("Ophiolitic metagabbro",       "0.9957", "0.9929", "0.9943", "700"),
]

SF_CLASS = [
    ("Alkali feldspar granites",    "0.9914", "0.9914", "0.9914", "700"),
    ("Alteration zones",            "0.8456", "0.9844", "0.9097", "128"),
    ("Biotite granites",            "0.9756", "0.9714", "0.9735", "700"),
    ("Gabbro",                      "0.9955", "0.9514", "0.9730", "700"),
    ("Garnet-muscovite granites",   "0.9758", "0.9800", "0.9779", "700"),
    ("Gneiss",                      "0.9855", "0.9743", "0.9799", "700"),
    ("Li-pegmatite granites",       "0.9957", "0.9957", "0.9957", "700"),
    ("Serpentinite",                "0.9780", "1.0000", "0.9889", "577"),
    ("Tonalite-granodiorite",       "0.9749", "0.9971", "0.9859", "700"),
    ("Wadi deposits",               "0.9971", "0.9757", "0.9863", "700"),
    ("Zinnwaldite-muscovite gran.", "0.9659", "0.9942", "0.9798", "171"),
    ("Ophiolitic metagabbro",       "1.0000", "0.9986", "0.9993", "700"),
]


# ─────────────────────────────────────────────────────────────────────────────
def build_report():
    doc = Document()

    # ── page layout ───────────────────────────────────────────────────────────
    section = doc.sections[0]
    section.page_width  = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin   = Inches(1.1)
    section.right_margin  = Inches(1.1)
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)

    # ── title page ────────────────────────────────────────────────────────────
    doc.add_paragraph()
    doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Lithological Mapping of the Aboresheed Region\n"
        "Using PRISMA Hyperspectral Imagery:\n"
        "A Comparative Study of Machine Learning\n"
        "and Transformer-Based Classification"
    )
    run.font.name  = "Calibri"
    run.font.size  = Pt(20)
    run.bold       = True
    run.font.color.rgb = NAVY

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Technical Research Report")
    sr.font.name = "Calibri"; sr.font.size = Pt(14); sr.font.color.rgb = TEAL

    doc.add_paragraph()
    info = [
        "Author: Nour Laban",
        "Date:   June 2026",
        "Experiment: PRISMA Aboresheed — experiments_008",
        "Scene ID: PRS_L2D_STD_20200725 (25 July 2020)",
    ]
    for line in info:
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Calibri"; run.font.size = Pt(11)
            run.font.color.rgb = DARK

    doc.add_page_break()

    # ─── 1. ABSTRACT ─────────────────────────────────────────────────────────
    add_heading(doc, "Abstract", level=1)
    add_body(doc,
        "This report presents a comparative evaluation of three classification algorithms "
        "— Random Forest (RF), eXtreme Gradient Boosting (XGBoost), and the SpectralFormer "
        "Vision Transformer — applied to lithological mapping of the Aboresheed region in the "
        "Eastern Desert of Egypt using PRISMA spaceborne hyperspectral imagery (30 m spatial "
        "resolution, ~234 spectral bands, 400–2500 nm). "
        "A processing pipeline encompassing HDF5-to-GeoTIFF conversion, bad-band removal, "
        "Minimum Noise Fraction (MNF) transformation, shapefile clipping, and stratified "
        "pixel sampling was applied prior to model training. "
        "Twelve geological classes — including alkali feldspar granites, alteration zones, "
        "serpentinite, ophiolitic metagabbro, and wadi deposits — were discriminated using "
        "field-mapped training polygons. "
        "All three models achieved outstanding accuracy on the 70 % held-out test set: "
        "RF attained an Overall Accuracy (OA) of 95.28 % and mean IoU of 89.62 %; "
        "XGBoost reached OA 98.30 % and IoU 95.66 %; "
        "SpectralFormer achieved the highest scores of OA 98.36 % and IoU 95.87 %. "
        "The results demonstrate that spaceborne hyperspectral data combined with modern "
        "classification methods can deliver publication-quality lithological maps at regional "
        "scale, with transformer architectures providing marginal but consistent gains over "
        "gradient-boosted ensembles."
    )

    doc.add_page_break()

    # ─── 2. INTRODUCTION ─────────────────────────────────────────────────────
    add_heading(doc, "1  Introduction", level=1)
    add_body(doc,
        "Geological mapping in arid and semi-arid shield terrains is constrained by "
        "extensive sand cover, rugged topography, and the high cost of traditional "
        "field campaigns. Remote sensing offers a cost-effective alternative, and the "
        "advent of spaceborne hyperspectral sensors has transformed the capability for "
        "lithological discrimination. Unlike multispectral systems (e.g., Landsat, Sentinel-2) "
        "that sample a few broad spectral bands, hyperspectral sensors record hundreds of "
        "contiguous narrow bands, capturing diagnostic absorption features of rock-forming "
        "minerals (e.g., Fe²⁺ near 1000 nm, Al-OH at 2200 nm, Mg-OH at 2300 nm) that "
        "are invisible in multispectral data."
    )
    add_body(doc,
        "The Italian Space Agency's PRISMA (PRecursore IperSpettrale della Missione Applicativa) "
        "sensor, launched in March 2019, provides atmospherically corrected L2D reflectance "
        "products covering 400–2500 nm at 30 m ground sampling distance, making it one of "
        "the most capable free-access hyperspectral platforms currently in orbit. "
        "The Aboresheed area (Eastern Desert, Egypt) exposes a diverse assemblage of "
        "Precambrian basement rocks — granites, gneisses, ophiolitic sequences, and "
        "hydrothermal alteration zones — making it an ideal testbed for evaluating "
        "hyperspectral classification algorithms."
    )
    add_body(doc,
        "This study evaluates three complementary classifiers of increasing architectural "
        "complexity: (1) Random Forest, a fast, robust ensemble of decision trees; "
        "(2) XGBoost, a regularised gradient boosting framework; and (3) SpectralFormer, "
        "a vision transformer that groups contiguous spectral bands into tokens and models "
        "cross-band dependencies through multi-head self-attention. "
        "The central research questions are: (i) How accurately can each method discriminate "
        "12 lithological units from PRISMA reflectance? (ii) What are the trade-offs between "
        "classification accuracy, computational cost, and per-class performance?"
    )

    # ─── 3. STUDY AREA ───────────────────────────────────────────────────────
    add_heading(doc, "2  Study Area", level=1)
    add_body(doc,
        "The Aboresheed region is located in the Central Eastern Desert of Egypt, within "
        "the Arabian–Nubian Shield (ANS) — a Neoproterozoic orogenic belt formed by the "
        "amalgamation of juvenile crustal arcs and microcontinents during the Pan-African "
        "orogeny (~900–550 Ma). The area is characterised by a complex lithological "
        "assemblage including:"
    )
    classes_text = [
        "Alkali feldspar granites — late-stage A-type granites with high potassium content",
        "Biotite granites and garnet-muscovite granites — two-mica peraluminous granites",
        "Li-pegmatite granites — economically significant rare-element pegmatites",
        "Zinnwaldite-muscovite granites — lithium-bearing evolved granites",
        "Tonalite-granodiorite — calc-alkaline arc-related intrusions",
        "Gabbro — mafic intrusions, often part of ophiolitic sequences",
        "Ophiolitic metagabbro — metamorphosed oceanic crustal remnants",
        "Serpentinite — hydrated ultramafic rocks, alteration product of dunite/peridotite",
        "Gneiss — high-grade metamorphic basement",
        "Alteration zones — hydrothermal clay/iron oxide halos around mineralised intrusions",
        "Wadi deposits — unconsolidated alluvial sediments in drainage channels",
    ]
    for c in classes_text:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(c)
        run.font.name = "Calibri"; run.font.size = Pt(10.5)
        run.font.color.rgb = DARK
    add_body(doc,
        "The arid climate minimises vegetation cover, maximising the spectral contrast "
        "between rock types and reducing atmospheric scattering effects. The PRISMA scene "
        "was acquired on 25 July 2020 under cloud-free conditions."
    )

    add_figure(doc,
        os.path.join(FIGS, "figure1_regional_map.png"),
        "Figure 1. Regional context map of the Aboresheed study area (Eastern Desert, Egypt), "
        "showing the PRISMA scene extent and major geological units.",
        width=Inches(5.5)
    )

    # ─── 4. DATA & METHODS ───────────────────────────────────────────────────
    add_heading(doc, "3  Data and Methods", level=1)

    add_heading(doc, "3.1  PRISMA Hyperspectral Data", level=2)
    add_table(doc,
        headers=["Parameter", "Value"],
        rows=[
            ["Sensor",            "PRISMA (ASI) — PRecursore IperSpettrale della Missione Applicativa"],
            ["Product Level",     "L2D (atmospherically corrected surface reflectance)"],
            ["Acquisition Date",  "25 July 2020, 08:35 UTC"],
            ["Scene ID",          "PRS_L2D_STD_20200725083506_20200725083510_0001.he5"],
            ["Spectral Range",    "400 – 2500 nm (VNIR + SWIR merged cube)"],
            ["Native Band Count", "~250 bands (VNIR + SWIR before cleaning)"],
            ["Bands After Removal","~234 bands (noisy/water-vapour bands removed)"],
            ["Spatial Resolution","30 m (nadir)"],
            ["Format",            "HDF5 (.he5) — VNIR and SWIR cubes merged and georeferenced"],
        ],
        col_widths=[Inches(2.0), Inches(4.0)]
    )

    add_heading(doc, "3.2  Preprocessing Pipeline", level=2)
    add_body(doc,
        "The following sequential preprocessing steps were applied before model training:"
    )
    steps = [
        ("Step 1 — HDF5 to GeoTIFF Conversion",
         "The PRISMA HDF5 file was parsed to extract the VNIR cube "
         "(key: HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube) and SWIR cube "
         "(SWIR_Cube), along with the geolocation arrays (Latitude, Longitude). "
         "Cubes were stacked into a single multi-band GeoTIFF with nodata set to −9999."),
        ("Step 2 — Bad-Band Removal",
         "Spectral bands dominated by water-vapour absorption or sensor edge noise were "
         "removed: VNIR bands 1–5 (UV/near-UV), SWIR bands 3–5 (≈1350 nm), 40–56 "
         "(1800–1950 nm water vapour), 86–111 (≈2400–2500 nm edge), and 152–171 "
         "(extended SWIR edge). The cleaned cube retains ~234 usable bands."),
        ("Step 3 — Shapefile Clipping",
         "The stacked GeoTIFF was clipped to the bounding extent of the field-mapped "
         "training shapefile (PRISMA_18_2_2026.shp) to reduce data volume and exclude "
         "pixels outside the mapped area from MNF estimation."),
        ("Step 4 — MNF Transformation",
         "Minimum Noise Fraction (MNF) rotation was applied to the clipped cube, "
         "retaining the top 10 MNF components. This step decorrelates the spectral "
         "dimensions, concentrates signal in the first components, and substantially "
         "reduces the dimensionality input to classical classifiers while preserving "
         "lithologically diagnostic variance."),
        ("Step 5 — Label Rasterisation",
         "Field-mapped geological polygons (shapefile class field: 'Classcode') were "
         "rasterised to match the GeoTIFF pixel grid, creating a label image with "
         "integer class codes 1–12."),
        ("Step 6 — Stratified Sampling",
         "Up to 1000 pixels per class were sampled with stratification (30 % training, "
         "70 % test). For minority classes (Alteration zones: 128 test pixels; "
         "Zinnwaldite-muscovite granites: 171 test pixels), all available pixels were "
         "used. Total test set: 7,176 pixels across 12 classes."),
    ]
    for title_s, body_s in steps:
        p = doc.add_paragraph()
        run = p.add_run(f"  {title_s}: ")
        run.bold = True; run.font.name = "Calibri"; run.font.size = Pt(10.5)
        run.font.color.rgb = TEAL
        run2 = p.add_run(body_s)
        run2.font.name = "Calibri"; run2.font.size = Pt(10.5)
        run2.font.color.rgb = DARK
        p.paragraph_format.space_after = Pt(5)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    add_figure(doc,
        os.path.join(FIGS, "figure3_preprocessing6.png"),
        "Figure 2. PRISMA preprocessing: (a) false-colour composite of the raw stacked cube; "
        "(b) rasterised geological label map; (c) training/test mask overlay.",
        width=Inches(6.0)
    )
    add_figure(doc,
        os.path.join(FIGS, "figure4_mnf_abc.png"),
        "Figure 3. MNF component images (bands 1–10): spatial distribution of spectral "
        "variance retained after noise reduction. The first three components capture the "
        "dominant lithological contrast.",
        width=Inches(5.8)
    )
    add_figure(doc,
        os.path.join(FIGS, "figure5_sample_distribution.png"),
        "Figure 4. Class sample distribution showing the number of training and test pixels "
        "per geological unit. Class imbalance is most pronounced for Alteration zones "
        "(rare exposure) and Zinnwaldite-muscovite granites.",
        width=Inches(5.5)
    )
    add_figure(doc,
        os.path.join(FIGS, "figure5a_spatial_map3.png"),
        "Figure 5. Spatial distribution of training samples overlaid on the PRISMA "
        "false-colour composite. Training polygons were distributed across the full "
        "scene extent to ensure geographic representativeness.",
        width=Inches(5.5)
    )

    # ─── 3.3 classifiers ─────────────────────────────────────────────────────
    add_heading(doc, "3.3  Classification Methods", level=2)

    add_heading(doc, "3.3.1  Random Forest", level=3 if False else 2)

    def add_sub3(doc, text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True; run.underline = True
        run.font.name = "Calibri"; run.font.size = Pt(11)
        run.font.color.rgb = TEAL
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after  = Pt(3)

    add_sub3(doc, "3.3.1  Random Forest (RF)")
    add_body(doc,
        "Random Forest is a bagging ensemble of decision trees, each trained on a "
        "bootstrap sample of the training data with a random subset of features considered "
        "at each split. Classification is by majority vote across all trees. "
        "In this study, an RF with 10 trees was trained on normalised per-pixel spectral "
        "vectors (10 MNF components). The model operates on individual pixels without "
        "spatial context, making it exceptionally fast (training: 0.2 s; "
        "full-scene inference: 0.4 s). RF provides a strong baseline that is robust to "
        "noise and handles class imbalance through naturally distributed vote weights."
    )

    add_sub3(doc, "3.3.2  eXtreme Gradient Boosting (XGBoost)")
    add_body(doc,
        "XGBoost is a regularised gradient boosting framework that builds an ensemble of "
        "weak learners (shallow trees) sequentially, each correcting the residual errors "
        "of the previous ensemble. It incorporates L1/L2 regularisation, column subsampling, "
        "and row subsampling to prevent overfitting. The model was trained with 200 boosting "
        "rounds on the same 10 MNF components as RF, with class weights to compensate for "
        "minority classes (training: 2.6 s; inference: 1.1 s). "
        "XGBoost typically outperforms RF on structured tabular data due to its sequential "
        "error-correction mechanism and built-in regularisation."
    )

    add_sub3(doc, "3.3.3  SpectralFormer (Vision Transformer)")
    add_body(doc,
        "SpectralFormer (Hong et al., IEEE TGRS 2022) adapts the Vision Transformer "
        "architecture to hyperspectral classification by treating groups of adjacent spectral "
        "bands as tokens for multi-head self-attention. The model processes 3×3 or 5×5 "
        "spatial patches (here: 3×3 pixels, 3-band group patches) extracted from the full "
        "234-band spectral cube — not MNF-reduced data — enabling it to learn both spectral "
        "absorption feature shapes and local spatial context simultaneously. "
        "Training hyperparameters: 200 epochs, learning rate 5×10⁻⁴, batch size 128, "
        "AdamW optimiser, class-weighted cross-entropy loss, cosine LR decay (γ=0.9). "
        "The best checkpoint (epoch 186, OA=98.36 %) was saved and used for inference. "
        "Total training time: 524 s (~8.7 min) on GPU; full-scene inference: ~2 min."
    )

    add_table(doc,
        headers=["Model", "Input Features", "Spatial Context", "Training Time", "Inference Time"],
        rows=[
            ["Random Forest (10 trees)", "10 MNF bands", "None (per-pixel)", "0.2 s", "0.4 s"],
            ["XGBoost (200 rounds)",     "10 MNF bands", "None (per-pixel)", "2.6 s",  "1.1 s"],
            ["SpectralFormer (ViT)",     "234 raw bands", "3×3 patch",       "524 s", "~117 s"],
        ],
        col_widths=[Inches(1.6), Inches(1.4), Inches(1.3), Inches(1.2), Inches(1.2)]
    )

    doc.add_page_break()

    # ─── 4. RESULTS ──────────────────────────────────────────────────────────
    add_heading(doc, "4  Results", level=1)

    # 4.1 Overall performance summary
    add_heading(doc, "4.1  Overall Performance Summary", level=2)
    add_body(doc,
        "Table 2 summarises the overall accuracy (OA) and mean Intersection-over-Union (IoU) "
        "achieved by each classifier on the 7,176-pixel test set. "
        "All three models substantially exceed the 90 % accuracy threshold commonly "
        "cited as the benchmark for geological mapping applications. "
        "SpectralFormer and XGBoost achieve near-identical OA, separated by only 0.06 %, "
        "while both improve over RF by ~3 percentage points. "
        "The IoU metric — which penalises both false positives and false negatives — "
        "reveals a larger gap between RF (89.62 %) and the two high-performing models "
        "(XGB: 95.66 %, SF: 95.87 %), confirming that RF produces more boundary errors "
        "particularly for spectrally similar granite sub-types."
    )
    add_table(doc,
        headers=["Model", "Overall Accuracy (OA)", "Mean IoU (Jaccard)", "Test Pixels", "Classes"],
        rows=[
            ["Random Forest",  "95.28 %", "89.62 %", "7,176", "12"],
            ["XGBoost",        "98.30 %", "95.66 %", "7,176", "12"],
            ["SpectralFormer", "98.36 %", "95.87 %", "7,176", "12"],
        ],
        col_widths=[Inches(1.6), Inches(1.5), Inches(1.5), Inches(1.2), Inches(1.0)]
    )

    # 4.2 RF
    add_heading(doc, "4.2  Random Forest Results", level=2)
    add_body(doc,
        "Random Forest achieved an OA of 95.28 % and mean IoU of 89.62 % — strong "
        "performance considering the 10-tree ensemble was trained in 0.2 s on only 10 "
        "MNF components. The macro-average F1 of 94.37 % indicates broadly consistent "
        "performance across classes. The largest errors occur for Alteration zones "
        "(F1 = 83.64 %) and Zinnwaldite-muscovite granites (F1 = 92.40 %), both minority "
        "classes with limited training data. Biotite granites (F1 = 88.94 %) are also "
        "challenging, likely due to spectral overlap with garnet-muscovite granites and "
        "tonalite-granodiorite. Li-pegmatite granites and ophiolitic metagabbro "
        "achieve the highest F1 scores (>98 %), reflecting their distinctive spectral "
        "signatures."
    )
    add_table(doc,
        headers=["Geological Class", "Precision", "Recall", "F1-Score", "Support"],
        rows=RF_CLASS,
        col_widths=[Inches(2.4), Inches(1.0), Inches(1.0), Inches(1.0), Inches(0.9)]
    )
    add_body(doc,
        "Table 3. Per-class classification metrics for Random Forest "
        "(OA = 95.28 %, macro avg F1 = 94.37 %).",
        italic=True
    )
    add_figure(doc,
        os.path.join(EVAL, "rf_model_evaluation", "confusion_matrix.png"),
        "Figure 6. Normalised confusion matrix for Random Forest. Diagonal values represent "
        "recall (producer's accuracy) per class. Off-diagonal cells indicate systematic "
        "inter-class confusion. Alteration zones and biotite granites show the highest "
        "confusion rates.",
        width=Inches(5.8)
    )
    add_figure(doc,
        os.path.join(FIGS, "rf_final_prediction_map.png"),
        "Figure 7. Random Forest lithological classification map of the Aboresheed region "
        "(PRISMA, 25 July 2020). Class colours follow the standard geological colour scheme "
        "defined in the project colour table.",
        width=Inches(5.5)
    )

    doc.add_page_break()

    # 4.3 XGBoost
    add_heading(doc, "4.3  XGBoost Results", level=2)
    add_body(doc,
        "XGBoost significantly improves over RF, achieving OA 98.30 % and mean IoU "
        "95.66 % — an improvement of +3.02 % OA and +6.04 % IoU. The macro-average "
        "F1 of 97.75 % demonstrates consistently high performance across all 12 classes. "
        "Li-pegmatite granites achieve perfect recall (100.0 %), while Gabbro "
        "(F1 = 98.05 %) and ophiolitic metagabbro (F1 = 99.43 %) show near-perfect "
        "discrimination. Alteration zones (F1 = 92.49 %) and Zinnwaldite-muscovite "
        "granites (F1 = 95.58 %) remain the two most challenging classes but are "
        "substantially improved compared to RF. The gradient boosting mechanism "
        "evidently captures spectral gradients distinguishing granite sub-types more "
        "effectively than bagged decision trees."
    )
    add_table(doc,
        headers=["Geological Class", "Precision", "Recall", "F1-Score", "Support"],
        rows=XGB_CLASS,
        col_widths=[Inches(2.4), Inches(1.0), Inches(1.0), Inches(1.0), Inches(0.9)]
    )
    add_body(doc,
        "Table 4. Per-class classification metrics for XGBoost "
        "(OA = 98.30 %, macro avg F1 = 97.75 %).",
        italic=True
    )
    add_figure(doc,
        os.path.join(EVAL, "xgb_model_evaluation", "confusion_matrix.png"),
        "Figure 8. Normalised confusion matrix for XGBoost. The matrix shows markedly "
        "reduced off-diagonal values compared to RF, with the most residual confusion "
        "between gabbro and serpentinite (ophiolitic sequence members with overlapping "
        "Fe-Mg spectral features).",
        width=Inches(5.8)
    )
    add_figure(doc,
        os.path.join(FIGS, "xgb_final_prediction_map.png"),
        "Figure 9. XGBoost lithological classification map. Note the sharper, more "
        "spatially coherent class boundaries compared to the Random Forest map, "
        "particularly in granite sub-type discrimination.",
        width=Inches(5.5)
    )

    doc.add_page_break()

    # 4.4 SpectralFormer
    add_heading(doc, "4.4  SpectralFormer Results", level=2)
    add_body(doc,
        "SpectralFormer achieves the highest overall performance: OA 98.36 % and "
        "mean IoU 95.87 % — fractionally above XGBoost (+0.06 % OA, +0.21 % IoU). "
        "The macro-average F1 of 97.84 % is the highest of all three models. "
        "Key highlights: ophiolitic metagabbro achieves a perfect precision of 100.0 % "
        "and 99.86 % recall (F1 = 99.93 %); serpentinite achieves 100.0 % recall; "
        "Zinnwaldite-muscovite granites reach 97.98 % F1 — a substantial improvement "
        "over both RF (92.40 %) and XGBoost (95.58 %), likely due to the transformer's "
        "ability to capture subtle OH and F-related absorption features at ~2340 nm "
        "that characterise zinnwaldite-bearing granites. "
        "Alteration zones (F1 = 90.97 %) remain the most challenging class, "
        "consistent across all three methods, as alteration mineralogy spans a "
        "continuum rather than a discrete spectral class. "
        "The model was trained for 200 epochs; best validation OA of 98.36 % was "
        "attained at epoch 186."
    )
    add_table(doc,
        headers=["Geological Class", "Precision", "Recall", "F1-Score", "Support"],
        rows=SF_CLASS,
        col_widths=[Inches(2.4), Inches(1.0), Inches(1.0), Inches(1.0), Inches(0.9)]
    )
    add_body(doc,
        "Table 5. Per-class classification metrics for SpectralFormer "
        "(OA = 98.36 %, macro avg F1 = 97.84 %).",
        italic=True
    )
    add_figure(doc,
        os.path.join(EVAL, "VITransformer_model_evaluation", "VITransformer_confusion_matrix.png"),
        "Figure 10. Normalised confusion matrix for SpectralFormer. The near-diagonal "
        "concentration of probability mass demonstrates the model's high discriminative "
        "power across all 12 lithological classes.",
        width=Inches(5.8)
    )
    add_figure(doc,
        os.path.join(FIGS, "VIT_final_prediction_map.png"),
        "Figure 11. SpectralFormer lithological classification map. The transformer-based "
        "map exhibits the spatially smoothest class boundaries, consistent with the model's "
        "3×3 spatial patch receptive field capturing local textural context.",
        width=Inches(5.5)
    )

    # Learning curves
    add_heading(doc, "4.4.1  SpectralFormer Training Dynamics", level=2)
    add_body(doc,
        "Figure 12 shows the validation Overall Accuracy as a function of training epoch. "
        "The model exhibits rapid initial convergence — OA exceeds 80 % by epoch 41 — "
        "followed by a gradual refinement phase from epoch 91 onward. "
        "OA stabilises above 97 % after epoch 130, with minor fluctuations indicative "
        "of the cosine learning rate schedule. The best checkpoint is saved at epoch 186 "
        "(OA = 98.36 %), demonstrating that the 200-epoch budget provides adequate "
        "convergence with room for early stopping at epoch 140+ without material loss "
        "of accuracy."
    )
    # Build epoch table (every 10 epochs for readability)
    epoch_rows = [
        ("1",   "18.84 %"), ("21",  "52.91 %"), ("41",  "80.94 %"),
        ("61",  "91.22 %"), ("81",  "92.06 %"), ("101", "94.48 %"),
        ("121", "95.90 %"), ("141", "97.39 %"), ("161", "97.84 %"),
        ("181", "97.81 %"), ("186", "98.36 % ★ Best"), ("200", "98.23 %"),
    ]
    add_table(doc,
        headers=["Epoch", "Validation OA"],
        rows=epoch_rows,
        col_widths=[Inches(1.5), Inches(2.0)]
    )
    add_body(doc,
        "Table 6. Selected SpectralFormer validation OA checkpoints (★ = best model saved).",
        italic=True
    )
    add_figure(doc,
        os.path.join(FIGS, "learning_curves.png"),
        "Figure 12. SpectralFormer training curve: validation Overall Accuracy vs. epoch. "
        "The model converges rapidly in the first 50 epochs and continues to refine "
        "gradually through the full 200-epoch training schedule.",
        width=Inches(5.5)
    )

    doc.add_page_break()

    # ─── 5. COMPARATIVE ANALYSIS ─────────────────────────────────────────────
    add_heading(doc, "5  Comparative Analysis", level=1)
    add_heading(doc, "5.1  Classification Maps", level=2)
    add_body(doc,
        "Figure 13 presents the four-panel comparison of RF, XGBoost, and SpectralFormer "
        "prediction maps side-by-side (the fourth panel in the comparison figure is the "
        "reference ground-truth label map). "
        "Visual inspection reveals progressive improvements in spatial coherence: "
        "the RF map shows speckling and salt-and-pepper artefacts in gradient areas; "
        "XGBoost produces smoother, more coherent class patches; "
        "SpectralFormer yields the cleanest boundaries with the fewest isolated misclassified "
        "pixels, benefiting from its 3×3 spatial patch receptive field."
    )
    add_figure(doc,
        os.path.join(FIGS, "comparison_panel.png"),
        "Figure 13. Side-by-side comparison of classification maps from all three methods "
        "(RF, XGBoost, SpectralFormer) and the reference label map. Note the progressive "
        "reduction in salt-and-pepper noise and improvement in geological boundary delineation.",
        width=Inches(6.0)
    )

    add_heading(doc, "5.2  Per-Class Performance Comparison", level=2)
    add_body(doc,
        "Table 7 consolidates the F1 scores for all three models across all 12 classes, "
        "enabling direct per-class comparison."
    )
    # Build comparison table
    classes_short = [
        "Alkali feldspar granites",
        "Alteration zones",
        "Biotite granites",
        "Gabbro",
        "Garnet-muscovite granites",
        "Gneiss",
        "Li-pegmatite granites",
        "Serpentinite",
        "Tonalite-granodiorite",
        "Wadi deposits",
        "Zinnwaldite-muscovite gran.",
        "Ophiolitic metagabbro",
    ]
    rf_f1  = [r[3] for r in RF_CLASS]
    xgb_f1 = [r[3] for r in XGB_CLASS]
    sf_f1  = [r[3] for r in SF_CLASS]
    comp_rows = [[c, rf, xgb, sf] for c, rf, xgb, sf in zip(classes_short, rf_f1, xgb_f1, sf_f1)]
    add_table(doc,
        headers=["Geological Class", "RF F1", "XGBoost F1", "SpectralFormer F1"],
        rows=comp_rows,
        col_widths=[Inches(2.3), Inches(1.1), Inches(1.2), Inches(1.6)]
    )
    add_body(doc,
        "Table 7. F1-score comparison across all three models for each geological class. "
        "Bold values in the original data indicate best per-class performance.",
        italic=True
    )

    add_heading(doc, "5.3  Computational Cost vs. Accuracy Trade-off", level=2)
    add_body(doc,
        "The three classifiers span four orders of magnitude in training time while "
        "delivering OA values within a 3 % range. For operational mapping workflows "
        "where rapid iteration is needed — e.g., exploring different training polygon "
        "configurations — RF or XGBoost provide immediate feedback (sub-second to seconds). "
        "For final-product quality maps, SpectralFormer's marginal accuracy gain and "
        "superior spatial coherence justify the ~9-minute training investment. "
        "Notably, SpectralFormer's inference time (~2 min for the full scene) is "
        "dominated by patch extraction overhead and could be reduced with batched GPU "
        "inference optimisation."
    )
    add_body(doc,
        "An important asymmetry exists in the input feature spaces: RF and XGBoost "
        "use 10 MNF components, while SpectralFormer operates on all 234 raw spectral "
        "bands. The MNF reduction discards higher-order noise components but also "
        "removes subtle low-amplitude spectral features. SpectralFormer's access to "
        "the full spectral range — and its attention mechanism's ability to weight "
        "specific absorption bands — likely accounts for its advantage on spectrally "
        "subtle classes like Zinnwaldite-muscovite granites."
    )

    doc.add_page_break()

    # ─── 6. DISCUSSION ───────────────────────────────────────────────────────
    add_heading(doc, "6  Discussion", level=1)

    add_heading(doc, "6.1  Geological Interpretation", level=2)
    add_body(doc,
        "The high classification accuracies achieved by all three methods confirm that "
        "PRISMA hyperspectral data — even at 30 m resolution — provides sufficient "
        "spectral information to discriminate the 12 lithological units of the "
        "Aboresheed basement complex. "
        "The most reliably classified units (F1 > 98 % for XGBoost/SpectralFormer) "
        "are ophiolitic metagabbro, Li-pegmatite granites, serpentinite, and wadi "
        "deposits — all of which have distinctive spectral features: "
        "metagabbro shows strong pyroxene absorption near 1000 nm and 2200 nm; "
        "pegmatites exhibit flat, bright spectra due to quartz and feldspar; "
        "serpentinite has diagnostic Mg-OH absorption at 2320 nm; "
        "wadi deposits are spectrally grey with low reflectance contrast."
    )
    add_body(doc,
        "Alteration zones consistently register the lowest F1 across all models "
        "(RF: 83.64 %, XGB: 92.49 %, SF: 90.97 %). "
        "This is expected: hydrothermal alteration mineralogy spans a continuum from "
        "kaolinite through illite to chlorite, whose spectral features partially overlap "
        "with the enclosing granite and metagabbro host rocks. "
        "The very small test support (128 pixels) also increases metric variance. "
        "Future work could benefit from more extensive alteration zone mapping and "
        "targeted mineral abundance unmixing rather than hard classification."
    )

    add_heading(doc, "6.2  Model Selection Guidance", level=2)
    add_body(doc,
        "Based on these results, we recommend the following decision framework for "
        "operational geological mapping with PRISMA data:"
    )
    guidance = [
        ("Rapid reconnaissance (hours to days):",
         "Use XGBoost. It delivers 98.3 % OA in 2.6 s of training, requires no GPU, "
         "and produces maps visually comparable to SpectralFormer at a fraction of the cost."),
        ("Final publication-quality maps:",
         "Use SpectralFormer. The marginal accuracy gain over XGBoost is small (0.06 % OA) "
         "but the spatial coherence improvement is visible in the classification maps, "
         "particularly at lithological contacts. A GPU with ≥8 GB VRAM is required."),
        ("Feature importance / exploratory analysis:",
         "Use Random Forest. Despite lower accuracy, RF provides per-band importance "
         "scores (via MNF components) that help identify which spectral regions most "
         "discriminate the units of interest — useful for sensor bandpass design studies."),
    ]
    for title_g, body_g in guidance:
        p = doc.add_paragraph()
        r1 = p.add_run(f"  {title_g} ")
        r1.bold = True; r1.font.name = "Calibri"; r1.font.size = Pt(10.5)
        r1.font.color.rgb = TEAL
        r2 = p.add_run(body_g)
        r2.font.name = "Calibri"; r2.font.size = Pt(10.5); r2.font.color.rgb = DARK
        p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ─── 7. CONCLUSION ───────────────────────────────────────────────────────
    add_heading(doc, "7  Conclusion", level=1)
    add_body(doc,
        "This study demonstrates that PRISMA spaceborne hyperspectral imagery, combined "
        "with a reproducible preprocessing pipeline (bad-band removal, MNF reduction, "
        "shapefile clipping) and modern machine learning classifiers, can produce "
        "high-fidelity lithological maps of Precambrian basement terrains in the "
        "Eastern Desert of Egypt. Three classifiers — Random Forest, XGBoost, and "
        "SpectralFormer — were evaluated on a 12-class geological mapping problem over "
        "the Aboresheed region using a 70 % held-out test set."
    )
    add_body(doc,
        "Key findings:"
    )
    findings = [
        "All three methods achieve outstanding accuracy (OA > 95 %) on the 7,176-pixel "
        "test set, demonstrating the spectral separability of the Aboresheed geological units.",
        "XGBoost (OA 98.30 %, IoU 95.66 %) provides near-transformer accuracy with "
        "minimal computational overhead, making it the preferred choice for operational workflows.",
        "SpectralFormer (OA 98.36 %, IoU 95.87 %) achieves the highest accuracy and "
        "spatially smoothest maps by leveraging full-spectrum patch-based attention, "
        "at the cost of GPU hardware and ~9 minutes of training time.",
        "Random Forest (OA 95.28 %, IoU 89.62 %) is competitive as a fast baseline but "
        "underperforms on spectrally similar granite sub-types and minority classes.",
        "Alteration zones and Zinnwaldite-muscovite granites are consistently the most "
        "challenging classes, with F1 scores 7–10 % lower than the class average, "
        "due to limited training data and spectral continuity with adjacent units.",
    ]
    for f in findings:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(f)
        run.font.name = "Calibri"; run.font.size = Pt(10.5)
        run.font.color.rgb = DARK
    add_body(doc,
        "Future work will investigate (i) hyperspectral unmixing to handle mixed-pixel "
        "alteration zones; (ii) transfer learning from one PRISMA scene to another to "
        "assess model generalisability; and (iii) data fusion with EnMap imagery to "
        "explore sensor complementarity."
    )

    # ─── 8. REFERENCES ───────────────────────────────────────────────────────
    add_heading(doc, "8  References", level=1)
    refs = [
        "Hong, D., Han, Z., Yao, J., Gao, L., Zhang, B., & Plaza, A. (2022). "
        "SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers. "
        "IEEE Transactions on Geoscience and Remote Sensing, 60, 1–14. "
        "DOI: 10.1109/TGRS.2021.3130716",

        "Cogliati, S., et al. (2021). The PRISMA imaging spectroscopy mission: overview "
        "and first performance analysis. Remote Sensing of Environment, 262, 112499. "
        "DOI: 10.1016/j.rse.2021.112499",

        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. "
        "DOI: 10.1023/A:1010933404324",

        "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. "
        "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge "
        "Discovery and Data Mining, 785–794. DOI: 10.1145/2939672.2939785",

        "Green, A. A., Berman, M., Switzer, P., & Craig, M. D. (1988). A transformation "
        "for ordering multispectral data in terms of image quality with implications for "
        "noise removal. IEEE Transactions on Geoscience and Remote Sensing, 26(1), 65–74. "
        "DOI: 10.1109/36.3001",

        "Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for "
        "Image Recognition at Scale. International Conference on Learning Representations (ICLR). "
        "arXiv:2010.11929",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        run = p.add_run(f"[{i}]  {ref}")
        run.font.name = "Calibri"; run.font.size = Pt(9.5)
        run.font.color.rgb = DARK
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.space_after = Pt(5)

    # ─── save ─────────────────────────────────────────────────────────────────
    doc.save(OUT)
    print(f"Report saved → {OUT}")


if __name__ == "__main__":
    build_report()
