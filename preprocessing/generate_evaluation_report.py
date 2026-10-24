"""
Generate a data-driven evaluation report (DOCX) for a completed experiment.

Unlike the hand-authored paper scripts (generate_gahelia_paper.py,
generate_technical_report.py), this module parses the actual
evaluation_results.txt files written by each model's demo script at
training time, so the report's numbers always match whatever the most
recent pipeline run produced — no hardcoded metrics.
"""

import os
import re

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

_MODEL_SPECS = [
    ('Random Forest',         'rf_model_evaluation',            'evaluation_results.txt'),
    ('XGBoost',                'xgb_model_evaluation',           'evaluation_results.txt'),
    ('S2Mamba',                'mamba_model_evaluation',         'mamba_evaluation_results.txt'),
    ('SpectralFormer (ViT)',   'VITransformer_model_evaluation', 'VITransformer_evaluation_results.txt'),
]

_PER_CLASS_RE = re.compile(
    r'^\s*(?P<name>.+?)\s\((?P<id>\d+)\)\s+'
    r'(?P<p>[\d.]+)\s+(?P<r>[\d.]+)\s+(?P<f1>[\d.]+)\s+(?P<support>\d+)\s*$'
)
_AVG_RE = re.compile(
    r'^\s*(?P<kind>macro avg|weighted avg)\s+'
    r'(?P<p>[\d.]+)\s+(?P<r>[\d.]+)\s+(?P<f1>[\d.]+)\s+(?P<support>\d+)\s*$'
)
_HEADER_ACC_RE = re.compile(r'Accuracy:\s*([\d.]+)')
_HEADER_IOU_RE = re.compile(r'IoU.*?:\s*([\d.]+)')


def parse_evaluation_results(path):
    """Parse one model's evaluation_results.txt into a dict of metrics."""
    text = open(path).read()
    acc_m = _HEADER_ACC_RE.search(text)
    iou_m = _HEADER_IOU_RE.search(text)

    per_class, macro_avg, weighted_avg = [], None, None
    for line in text.splitlines():
        m = _PER_CLASS_RE.match(line)
        if m:
            per_class.append({
                'name': m['name'].strip(),
                'id': int(m['id']),
                'precision': float(m['p']),
                'recall': float(m['r']),
                'f1': float(m['f1']),
                'support': int(m['support']),
            })
            continue
        m = _AVG_RE.match(line)
        if m:
            row = {
                'precision': float(m['p']), 'recall': float(m['r']),
                'f1': float(m['f1']), 'support': int(m['support']),
            }
            if m['kind'] == 'macro avg':
                macro_avg = row
            else:
                weighted_avg = row

    per_class.sort(key=lambda c: c['id'])
    return {
        'accuracy': float(acc_m.group(1)) if acc_m else None,
        'iou': float(iou_m.group(1)) if iou_m else None,
        'per_class': per_class,
        'macro_avg': macro_avg,
        'weighted_avg': weighted_avg,
    }


def collect_results(save_model_path):
    """Parse every model's evaluation_results.txt found under save_model_path."""
    results = {}
    for label, subdir, filename in _MODEL_SPECS:
        path = os.path.join(save_model_path, subdir, filename)
        if os.path.exists(path):
            results[label] = parse_evaluation_results(path)
    return results


# ── docx style helpers ─────────────────────────────────────────────────────────

def _font(run, size=11, bold=False, italic=False, name="Calibri", color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def _heading(doc, text, level):
    p = doc.add_heading(text, level=level)
    sizes = {0: 20, 1: 15, 2: 13, 3: 11.5}
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(sizes.get(level, 11))
        run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    p.paragraph_format.space_before = Pt(14 if level <= 1 else 8)
    p.paragraph_format.space_after = Pt(6)
    return p


def _para(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    _font(r)
    return p


def _shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_fill)
    tcPr.append(shd)


def _table(doc, headers, rows, caption_text=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ci, h in enumerate(headers):
        cell = t.rows[0].cells[ci]
        cell.text = str(h)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        for run in cell.paragraphs[0].runs:
            _font(run, size=10, bold=True, color=(255, 255, 255))
        _shade(cell, "1F3864")
    for ri, row in enumerate(rows):
        fill = "EBF3FB" if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cell.paragraphs[0].runs:
                _font(run, size=10)
            _shade(cell, fill)
    if caption_text:
        p = doc.add_paragraph(caption_text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(14)
        for run in p.runs:
            _font(run, size=9.5, italic=True)
    return t


def _figure(doc, path, caption_text, width=Inches(6.0)):
    if not (path and os.path.exists(path)):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=width)
    cap = doc.add_paragraph(caption_text)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(14)
    for run in cap.runs:
        _font(run, size=9.5, italic=True)


# ── report builder ──────────────────────────────────────────────────────────────

def generate_report(config, experiment_path=None, save_model_path=None,
                     output_name=None):
    """
    Build a data-driven evaluation report DOCX from the evaluation_results.txt
    files of an experiment, plus the figures/comparison_panel.png and
    figures/confusion_matrix_panel.png if present.

    experiment_path / save_model_path default to config.experiment_path /
    config.save_model_path, but can be overridden to point at a specific
    past experiment directory without touching the live config.
    """
    experiment_path = experiment_path or config.experiment_path
    save_model_path = save_model_path or config.save_model_path
    region = getattr(config, 'region_label', config.satellite_name)

    results = collect_results(save_model_path)
    if not results:
        raise FileNotFoundError(f'No evaluation_results files found under {save_model_path}')

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(2.2)
        sec.bottom_margin = Cm(2.2)
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(2.5)

    title = doc.add_heading(f'Hyperspectral Classification Evaluation Report — {region}', level=0)
    for run in title.runs:
        run.font.name = "Calibri"
        run.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    sub = doc.add_paragraph(f'Scene: {getattr(config, "scene_id", "n/a")}   |   '
                             f'Satellite config: {config.satellite_name}   |   '
                             f'Experiment: {os.path.basename(experiment_path)}')
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        _font(run, size=10, italic=True, color=(0x55, 0x55, 0x55))

    # ── Methodology summary ──────────────────────────────────────────────────
    _heading(doc, '1. Pipeline Summary', 1)
    n_classes = len(next(iter(results.values()))['per_class'])
    sensor = getattr(config, 'sensor', 'satellite')
    source = {'prisma_he5': 'HDF5 (.he5)', 'envi': 'ENVI', 'geotiff': 'GeoTIFF'}.get(
        getattr(config, 'scene_format', ''), 'its native format')
    _para(doc,
        f'This report was generated automatically from the outputs of '
        f'`main.py run-all --dataset {config.satellite_name}`. The {sensor} scene '
        f'was converted from {source}, clipped to its training-shapefile extent, '
        f'augmented with {getattr(config, "num_mnf_bands", 0)} MNF bands, and split into '
        f'stratified train/test masks ({int(getattr(config, "samples", 0))} samples per class, '
        f'{getattr(config, "test_size", 0):.0%} test fraction) across {n_classes} geological classes. '
        f'All four models below ({", ".join(results.keys())}) were trained and evaluated on this identical split.')

    # ── Overall comparison ───────────────────────────────────────────────────
    _heading(doc, '2. Overall Model Comparison', 1)
    rows = []
    for label, r in results.items():
        macro_f1 = r['macro_avg']['f1'] if r['macro_avg'] else None
        weighted_f1 = r['weighted_avg']['f1'] if r['weighted_avg'] else None
        rows.append([
            label,
            f"{r['accuracy']:.4f}" if r['accuracy'] is not None else '—',
            f"{r['iou']:.4f}" if r['iou'] is not None else '—',
            f"{macro_f1:.4f}" if macro_f1 is not None else '—',
            f"{weighted_f1:.4f}" if weighted_f1 is not None else '—',
        ])
    rows.sort(key=lambda row: float(row[1]) if row[1] != '—' else -1, reverse=True)
    _table(doc, ['Model', 'Overall Accuracy', 'IoU (Jaccard)', 'Macro F1', 'Weighted F1'], rows,
           'Table 1. Overall accuracy, IoU, and F1 scores for all evaluated models, '
           'sorted best to worst by overall accuracy.')

    # ── Per-model per-class tables ───────────────────────────────────────────
    _heading(doc, '3. Per-Class Performance', 1)
    for label, r in results.items():
        _heading(doc, label, 2)
        class_rows = [
            [c['name'], c['support'], f"{c['precision']:.4f}", f"{c['recall']:.4f}", f"{c['f1']:.4f}"]
            for c in r['per_class']
        ]
        if r['macro_avg']:
            class_rows.append(['Macro avg', r['macro_avg']['support'],
                                f"{r['macro_avg']['precision']:.4f}",
                                f"{r['macro_avg']['recall']:.4f}",
                                f"{r['macro_avg']['f1']:.4f}"])
        if r['weighted_avg']:
            class_rows.append(['Weighted avg', r['weighted_avg']['support'],
                                f"{r['weighted_avg']['precision']:.4f}",
                                f"{r['weighted_avg']['recall']:.4f}",
                                f"{r['weighted_avg']['f1']:.4f}"])
        _table(doc, ['Class', 'Support', 'Precision', 'Recall', 'F1'], class_rows,
               f'Table — per-class precision/recall/F1 for {label}.')

    # ── Figures ───────────────────────────────────────────────────────────────
    figures_dir = os.path.join(experiment_path, 'figures')
    comparison_png = os.path.join(figures_dir, 'comparison_panel.png')
    confusion_png = os.path.join(figures_dir, 'confusion_matrix_panel.png')
    if os.path.exists(comparison_png) or os.path.exists(confusion_png):
        _heading(doc, '4. Figures', 1)
        _figure(doc, comparison_png,
                f'Figure 1. Full-scene prediction map comparison across all evaluated models — {region}.')
        _figure(doc, confusion_png,
                f'Figure 2. Row-normalised confusion matrices for all evaluated models — {region}.')

    output_name = output_name or f'Evaluation_Report_{region.replace(" ", "")}.docx'
    out_path = os.path.join(experiment_path, output_name)
    doc.save(out_path)
    print(f'Saved → {out_path}')
    return out_path


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from projutils.config import Config, available_datasets

    parser = argparse.ArgumentParser(description='Generate a data-driven evaluation report DOCX.')
    parser.add_argument('--dataset', '--satellite', dest='satellite', required=True,
                        choices=available_datasets())
    parser.add_argument('--experiment', default=None,
                        help='Experiment folder name (default: the dataset\'s configured experiment)')
    parser.add_argument('--experiment-path', default=None,
                        help='Override experiment directory with an explicit path.')
    parser.add_argument('--data-root', default=None, help='Override the data root directory')
    args = parser.parse_args()

    cfg = Config(satellite_name=args.satellite, experiment=args.experiment, data_root=args.data_root)
    exp_path = args.experiment_path
    save_path = os.path.join(exp_path, 'saved_models') if exp_path else None
    generate_report(cfg, experiment_path=exp_path, save_model_path=save_path)
