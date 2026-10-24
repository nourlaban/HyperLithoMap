"""
Apply all 16 recommended corrections to the paper as OOXML tracked changes
(w:del / w:ins) so Word shows exactly what was changed with author attribution.
"""

import copy, os, shutil, re, sys
from itertools import count
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from projutils.config import data_root
_EXP = os.path.join(data_root(), "aboresheadmamabadataPrisma", "output", "experiments_008")
SRC  = os.path.join(_EXP, "Research_Paper_Aboresheed_PRISMA.docx")
DST  = os.path.join(_EXP, "Research_Paper_Aboresheed_PRISMA_REVISED.docx")

AUTHOR = "Compatibility Review"
DATE   = "2026-06-02T00:00:00Z"
W      = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML    = "http://www.w3.org/XML/1998/namespace"

_id = count(1)

def nid():
    return str(next(_id))

# ── low-level XML helpers ──────────────────────────────────────────────────────

def _rpr(run_elem):
    """Return a deep-copy of the run's rPr, or None."""
    rpr = run_elem.find(f"{{{W}}}rPr")
    return copy.deepcopy(rpr) if rpr is not None else None

def _make_del(old_text, rpr=None):
    d = OxmlElement("w:del")
    d.set(qn("w:id"), nid())
    d.set(qn("w:author"), AUTHOR)
    d.set(qn("w:date"), DATE)
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(rpr)
    t = OxmlElement("w:delText")
    t.text = old_text
    t.set(f"{{{XML}}}space", "preserve")
    r.append(t)
    d.append(r)
    return d

def _make_ins(new_text, rpr=None):
    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), nid())
    ins.set(qn("w:author"), AUTHOR)
    ins.set(qn("w:date"), DATE)
    r = OxmlElement("w:r")
    if rpr is not None:
        r.append(rpr)
    t = OxmlElement("w:t")
    t.text = new_text
    t.set(f"{{{XML}}}space", "preserve")
    r.append(t)
    ins.append(r)
    return ins

# ── core replacement: works on any paragraph XML element ─────────────────────

def _para_full_text(p_xml):
    """Return (full_text, list_of_(run_elem, char_start, char_end))."""
    runs = []
    offset = 0
    for child in p_xml:
        if child.tag != f"{{{W}}}r":
            continue
        t_elem = child.find(f"{{{W}}}t")
        text = (t_elem.text or "") if t_elem is not None else ""
        runs.append((child, offset, offset + len(text)))
        offset += len(text)
    return offset, runs   # offset == total length


def track_replace(p_xml, old, new):
    """
    Replace the first occurrence of `old` in paragraph `p_xml` with
    tracked-change del(old)+ins(new).  Returns True if a replacement was made.
    Text may span multiple consecutive runs.
    """
    _, run_info = _para_full_text(p_xml)

    # Build full plain text
    full = "".join(
        (r.find(f"{{{W}}}t").text or "")
        if r.find(f"{{{W}}}t") is not None else ""
        for r, *_ in run_info
    )
    if old not in full:
        return False

    match_s = full.index(old)
    match_e = match_s + len(old)

    # Find which runs are fully or partially inside [match_s, match_e)
    involved = [(r, rs, re) for r, rs, re in run_info
                if rs < match_e and re > match_s]
    if not involved:
        return False

    first_run, first_rs, _ = involved[0]
    last_run,  _, last_re  = involved[-1]

    # Where inside the first / last run does the match start / end?
    cut_start = match_s - first_rs     # offset inside first run's text
    cut_end   = match_e - (last_re - len(
        (last_run.find(f"{{{W}}}t").text or "")
        if last_run.find(f"{{{W}}}t") is not None else ""
    ))                                 # offset inside last run's text

    # Grab rPr from first run for styling the del/ins
    rpr_del = _rpr(first_run)
    rpr_ins = copy.deepcopy(rpr_del)

    # Build replacement node list: [pre_run?] [del] [ins] [post_run?]
    new_nodes = []

    # ── pre-match fragment of first run ──────────────────────────────────────
    pre_text_first = (first_run.find(f"{{{W}}}t").text or "")[:cut_start]
    if pre_text_first:
        pre_r = copy.deepcopy(first_run)
        pre_t = pre_r.find(f"{{{W}}}t")
        pre_t.text = pre_text_first
        pre_t.set(f"{{{XML}}}space", "preserve")
        new_nodes.append(pre_r)

    # ── del element spanning all involved runs ────────────────────────────────
    del_elem = OxmlElement("w:del")
    del_elem.set(qn("w:id"), nid())
    del_elem.set(qn("w:author"), AUTHOR)
    del_elem.set(qn("w:date"), DATE)

    for run, rs, re in involved:
        t_elem = run.find(f"{{{W}}}t")
        run_text = (t_elem.text or "") if t_elem is not None else ""

        # Clip to [match_s, match_e) within this run
        local_s = max(match_s - rs, 0)
        local_e = min(match_e - rs, len(run_text))
        fragment = run_text[local_s:local_e]
        if not fragment:
            continue

        del_r = OxmlElement("w:r")
        rpr_copy = _rpr(run)
        if rpr_copy is not None:
            del_r.append(rpr_copy)
        del_t = OxmlElement("w:delText")
        del_t.text = fragment
        del_t.set(f"{{{XML}}}space", "preserve")
        del_r.append(del_t)
        del_elem.append(del_r)

    new_nodes.append(del_elem)

    # ── ins element ────────────────────────────────────────────────────────────
    new_nodes.append(_make_ins(new, rpr_ins))

    # ── post-match fragment of last run ───────────────────────────────────────
    last_t_elem = last_run.find(f"{{{W}}}t")
    last_text    = (last_t_elem.text or "") if last_t_elem is not None else ""
    # cut_end relative to last_re:
    local_end_in_last = match_e - (last_re - len(last_text))
    post_text_last = last_text[local_end_in_last:]
    if post_text_last:
        post_r = copy.deepcopy(last_run)
        post_t = post_r.find(f"{{{W}}}t")
        post_t.text = post_text_last
        post_t.set(f"{{{XML}}}space", "preserve")
        new_nodes.append(post_r)

    # ── splice into paragraph XML ──────────────────────────────────────────────
    # Insert position = index of first involved run in p_xml children
    children = list(p_xml)
    insert_idx = children.index(first_run)

    for run, *_ in involved:
        p_xml.remove(run)

    for i, node in enumerate(new_nodes):
        p_xml.insert(insert_idx + i, node)

    return True


def track_replace_in_doc(doc, old, new, limit=None):
    """
    Apply track_replace to every paragraph in the document body and in
    every table cell.  `limit` stops after that many replacements (None = all).
    Returns count of replacements made.
    """
    done = 0
    for para in doc.paragraphs:
        if track_replace(para._p, old, new):
            done += 1
            if limit and done >= limit:
                return done
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    if track_replace(para._p, old, new):
                        done += 1
                        if limit and done >= limit:
                            return done
    return done


def track_replace_cell(table, row_idx, col_idx, old, new):
    """Direct targeted replacement in a specific table cell."""
    cell = table.rows[row_idx].cells[col_idx]
    for para in cell.paragraphs:
        if track_replace(para._p, old, new):
            return True
    return False


# ── apply all corrections ─────────────────────────────────────────────────────

shutil.copy(SRC, DST)
doc = Document(DST)

applied = []

def log(tag, old, new, n):
    applied.append((tag, old, new, n))
    status = "✓" if n else "✗ NOT FOUND"
    print(f"  [{status}] {tag}: {repr(old)[:45]} → {repr(new)[:30]}  (n={n})")

print("\n=== Applying tracked changes ===\n")

# ─────────────────────────────────────────────────────────────────────────────
# D1-D4  ABSTRACT: headline metrics
# ─────────────────────────────────────────────────────────────────────────────
# OA 99.5 % appears multiple times → replace all
n = track_replace_in_doc(doc, "99.5%", "99.18%")
log("D1 OA abstract+body", "99.5%", "99.18%", n)

n = track_replace_in_doc(doc, "99.44%", "99.18%")
log("D16 OA convergence claim", "99.44%", "99.18%", n)

n = track_replace_in_doc(doc, "κ = 0.994", "κ = 0.991")
log("D2 kappa abstract", "κ = 0.994", "κ = 0.991", n)

n = track_replace_in_doc(doc, "κ = 0.9939", "κ = 0.9909")
log("D2 kappa conclusion", "κ = 0.9939", "κ = 0.9909", n)

n = track_replace_in_doc(doc, "mIoU = 98.4%", "mIoU = 97.62%")
log("D3 mIoU abstract", "mIoU = 98.4%", "mIoU = 97.62%", n)

n = track_replace_in_doc(doc, "mIoU = 98.45%", "mIoU = 97.62%")
log("D3 mIoU conclusion", "mIoU = 98.45%", "mIoU = 97.62%", n)

n = track_replace_in_doc(doc, "macro-average F1 = 0.992", "macro-average F1 = 0.988")
log("D4 macro-F1 abstract", "macro-average F1 = 0.992", "macro-average F1 = 0.988", n)

n = track_replace_in_doc(doc, "macro-average F1 = 0.992", "macro-average F1 = 0.988")
log("D4 macro-F1 (2nd pass)", "macro-average F1 = 0.992", "macro-average F1 = 0.988", n)

# ─────────────────────────────────────────────────────────────────────────────
# D1-D3  TABLE 4: S²Mamba row (table index 3, row 4)
# ─────────────────────────────────────────────────────────────────────────────
t4 = doc.tables[3]
r = track_replace_cell(t4, 4, 1, "99.5", "99.18")
log("D1 Table4 S²Mamba OA", "99.5", "99.18", int(r))

r = track_replace_cell(t4, 4, 2, "98.4", "97.62")
log("D3 Table4 S²Mamba mIoU", "98.4", "97.62", int(r))

r = track_replace_cell(t4, 4, 3, "0.994", "0.991")
log("D2 Table4 S²Mamba kappa", "0.994", "0.991", int(r))

# Add timing estimate footnote to training time cell
r = track_replace_cell(t4, 4, 4, "~2.4 min (GPU)", "~2.4 min (GPU)†")
log("D15 Table4 S²Mamba train-time footnote", "~2.4 min (GPU)", "~2.4 min (GPU)†", int(r))

r = track_replace_cell(t4, 3, 4, "~7 min (GPU)", "~6.8 min (GPU)†")
log("D15 Table4 ViT train-time footnote", "~7 min (GPU)", "~6.8 min (GPU)†", int(r))

# ─────────────────────────────────────────────────────────────────────────────
# D9-D10  TABLE 3: sample counts (table index 2)
# ─────────────────────────────────────────────────────────────────────────────
t3 = doc.tables[2]
# Alteration zones row (row 2): train 55→54, total 183→182
r = track_replace_cell(t3, 2, 1, "55", "54")
log("D10 Table3 Alteration train", "55", "54", int(r))

r = track_replace_cell(t3, 2, 3, "183", "182")
log("D10 Table3 Alteration total", "183", "182", int(r))

# Total row (row 13): 2,775→3,074 and 9,951→10,250
r = track_replace_cell(t3, 13, 1, "2,775", "3,074")
log("D9 Table3 total train", "2,775", "3,074", int(r))

r = track_replace_cell(t3, 13, 3, "9,951", "10,250")
log("D9 Table3 grand total", "9,951", "10,250", int(r))

# ─────────────────────────────────────────────────────────────────────────────
# D5-D8  TABLE 5: S²Mamba per-class (table index 4)
# ─────────────────────────────────────────────────────────────────────────────
t5 = doc.tables[4]

corrections_t5 = [
    # (row, col, old, new, label)
    # Alteration zones (row 2): P, R, F1
    (2, 1, "0.9685", "0.8951", "D5 Alteration P"),
    (2, 2, "0.9609", "1.0000", "D5 Alteration R"),
    (2, 3, "0.9647", "0.9446", "D5 Alteration F1"),
    # Biotite granites (row 3)
    (3, 1, "0.9857", "0.9706", "D5 Biotite P"),
    (3, 2, "0.9871", "0.9914", "D5 Biotite R"),
    (3, 3, "0.9864", "0.9809", "D5 Biotite F1"),
    # Gabbro (row 4)
    (4, 1, "0.9971", "0.9985", "D5 Gabbro P"),
    (4, 2, "0.9914", "0.9829", "D5 Gabbro R"),
    (4, 3, "0.9943", "0.9906", "D5 Gabbro F1"),
    # Garnet-muscovite (row 5)
    (5, 1, "0.9986", "1.0000", "D7 Garnet P"),
    (5, 2, "0.9929", "0.9786", "D7 Garnet R"),
    (5, 3, "0.9957", "0.9892", "D7 Garnet F1"),
    # Li-pegmatite (row 7)
    (7, 1, "0.9986", "0.9943", "D5 Li-pegm P"),
    # Serpentinite: already perfect → no change needed
    # Tonalite (row 9)
    (9, 3, "0.9957", "0.9964", "D5 Tonalite F1"),
    # Wadi deposits (row 10)
    (10, 3, "0.9957", "0.9935", "D5 Wadi F1"),
    # Zinnwaldite (row 11)
    (11, 1, "0.977",  "1.0000", "D6 Zinnw P"),
    (11, 2, "0.9942", "0.9532", "D6 Zinnw R"),
    (11, 3, "0.9855", "0.9760", "D6 Zinnw F1"),
    # Ophiolitic metagabbro (row 12)
    (12, 1, "0.9986", "1.0000", "D8 Ophi P"),
    (12, 3, "0.9993", "1.0000", "D8 Ophi F1"),
    # Macro avg (row 13)
    (13, 1, "0.9918", "0.9861", "D4 Macro P"),
    (13, 2, "0.9923", "0.9900", "D4 Macro R"),
    (13, 3, "0.992",  "0.9877", "D4 Macro F1"),
    # Weighted avg (row 14)
    (14, 1, "0.9946", "0.9921", "D4 Weighted P"),
    (14, 2, "0.9946", "0.9918", "D4 Weighted R"),
    (14, 3, "0.9946", "0.9918", "D4 Weighted F1"),
]
for row_i, col_i, old, new, label in corrections_t5:
    success = track_replace_cell(t5, row_i, col_i, old, new)
    log(label, old, new, int(success))

# ─────────────────────────────────────────────────────────────────────────────
# D11  MNF cumulative variance in §4.1 body text
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc, "82.7%", "65.9%")
log("D11 MNF cumulative variance", "82.7%", "65.9%", n)

# ─────────────────────────────────────────────────────────────────────────────
# D12  Clipped scene dimensions in §4.1
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc, "410 × 360 pixels", "585 × 781 pixels")
log("D12 clip pixel dims", "410 × 360 pixels", "585 × 781 pixels", n)

n = track_replace_in_doc(doc, "12.3 km × 10.8 km", "17.9 km × 23.7 km")
log("D12 clip km dims", "12.3 km × 10.8 km", "17.9 km × 23.7 km", n)

# ─────────────────────────────────────────────────────────────────────────────
# D13  Valid pixel count in §4.1
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc,
    "approximately 112,000 pixels (~76% of the clipped extent)",
    "456,885 pixels (100% of the clipped extent — no nodata)")
log("D13 valid pixel count", "~112,000 (~76%)", "456,885 (100%)", n)

n = track_replace_in_doc(doc, "~590,000", "456,300")
log("D13 noise pairs", "~590,000", "456,300", n)

# ─────────────────────────────────────────────────────────────────────────────
# D14  Cross-model gain claims §4.3.3
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc, "+23.2 pp F1", "+10.8 pp F1")
log("D14 Alteration gain", "+23.2 pp F1", "+10.8 pp F1", n)

n = track_replace_in_doc(doc, "+16.1 pp F1", "+5.1 pp F1")
log("D14 Gneiss gain", "+16.1 pp F1", "+5.1 pp F1", n)

# Also fix the summary sentence numbers
n = track_replace_in_doc(doc, "Alteration zones (+23.2 pp", "Alteration zones (+10.8 pp")
log("D14 Alteration gain sentence", "Alteration zones (+23.2 pp", "Alteration zones (+10.8 pp", n)

n = track_replace_in_doc(doc, "Gneiss (+16.1 pp", "Gneiss (+5.1 pp")
log("D14 Gneiss gain sentence", "Gneiss (+16.1 pp", "Gneiss (+5.1 pp", n)

# ─────────────────────────────────────────────────────────────────────────────
# D15  Training time caveat in §5.3
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc,
    "approximately 2.4 minutes of GPU training",
    "approximately 2.4 minutes of GPU training (extrapolated estimate from 3 timed epochs)")
log("D15 Mamba training time caveat §5.3",
    "approximately 2.4 minutes...", "...estimate from 3 timed epochs", n)

n = track_replace_in_doc(doc,
    "approximately 6.8 minutes to train",
    "approximately 6.8 minutes to train (extrapolated estimate from 3 timed epochs)")
log("D15 ViT training time caveat §5.3",
    "approximately 6.8 minutes to train", "+caveat", n)

# ─────────────────────────────────────────────────────────────────────────────
# D16  Best checkpoint / convergence OA
# ─────────────────────────────────────────────────────────────────────────────
n = track_replace_in_doc(doc,
    "best checkpoint at epoch 130 (OA = 99.44%)",
    "best checkpoint at epoch 130 (measured final OA = 99.18%; per-epoch log unavailable)")
log("D16 best checkpoint OA",
    "epoch 130 (OA = 99.44%)", "epoch 130 (OA = 99.18%; per-epoch log unavailable)", n)

# Also fix the body mention
n = track_replace_in_doc(doc, "OA = 99.44%", "OA = 99.18%")
log("D16 OA = 99.44% remaining", "OA = 99.44%", "OA = 99.18%", n)

# S²Mamba conclusion body
n = track_replace_in_doc(doc, "OA = 99.44%,", "OA = 99.18%,")
log("D1/D16 conclusion OA", "OA = 99.44%,", "OA = 99.18%,", n)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
doc.save(DST)
print(f"\n✓ Saved revised paper → {DST}")
print(f"  {len(applied)} replacement operations attempted")
print(f"  {sum(1 for *_, n in applied if n)} succeeded  |  {sum(1 for *_, n in applied if not n)} not found")
