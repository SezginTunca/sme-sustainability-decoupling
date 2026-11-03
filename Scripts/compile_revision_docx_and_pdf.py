#!/usr/bin/env python3
"""
compile_revision_docx_and_pdf.py
================================================================================
Publication-Grade DOCX, Track-Changes DOCX, and PDF Compiler for:
  "Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"
  (Business Strategy and the Environment; Concept DOI: 10.5281/zenodo.23619994)

Compiles all required Word (.docx) and PDF (.pdf) deliverables from the verified
Markdown sources in Revision/ and project root:
  1. Revision/BSE_Main_Manuscript_Revision.docx
  2. Revision/1_BSE_Main_Manuscript_Revision.pdf
  3. Revision/BSE_Main_Manuscript_TrackChanges.docx  (vs. Submission_final/BSE_Main_Manuscript_Final.docx)
  4. Revision/BSE_Supporting_Information_Revision.docx
  5. Revision/2_BSE_Supporting_Information_Revision.pdf
  6. Revision/Title Page_I_Revision.docx
  7. Revision/Title Page_II_Revision.docx
  8. Revision/Response_to_Reviewers.docx
  9. Revision/Response_to_Coauthor_Hari.docx
 10. Revision/CoAuthor_Verification_Note_Mausam_Risal.docx
 11. README.docx
Plus synchronized copies in Manuscript/Export/.
================================================================================
"""

import difflib
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime, timedelta

import docx
import pypandoc
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

_cur_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_cur_dir).lower() in ("confidential", "internal"):
    PROJECT_DIR = os.path.dirname(os.path.dirname(_cur_dir))
else:
    PROJECT_DIR = os.path.dirname(_cur_dir)
REVISION_DIR = os.path.join(PROJECT_DIR, "Revision")
MANUSCRIPT_DIR = os.path.join(PROJECT_DIR, "Manuscript")
EXPORT_DIR = os.path.join(MANUSCRIPT_DIR, "Export")
FIGURES_DIR = os.path.join(PROJECT_DIR, "Figures")
ORIG_SUBMISSION_DOCX = os.path.join(PROJECT_DIR, "Submission_final", "BSE_Main_Manuscript_Final.docx")

os.makedirs(EXPORT_DIR, exist_ok=True)

AUTHOR = "Sezgin Tunca"
CHANGE_ID = 100


# ==============================================================================
# 1. Markdown Preprocessing & Image Resolution
# ==============================================================================
def preprocess_markdown(md_text, base_dir):
    """Resolve relative image paths and clean GitHub-only callout markers and remote SVG badges."""
    # Remove GitHub callout tags (> [!NOTE], > [!IMPORTANT], etc.) while keeping blockquote content
    md_text = re.sub(
        r"^>\s*\[!(?:NOTE|IMPORTANT|WARNING|TIP|CAUTION)\]\s*\n?",
        "",
        md_text,
        flags=re.MULTILINE,
    )
    # Convert remote linked SVG badges [![alt](https://...)](url) into clean text links
    md_text = re.sub(r"\[!\[([^\]]+)\]\(https?://[^)]+\)\]\(([^)]+)\)", r"[\1](\2)", md_text)

    def replace_img(match):
        alt = match.group(1)
        path = match.group(2).strip()
        if path.startswith(("http://", "https://", "/")):
            return match.group(0)
        for cand_dir in (base_dir, REVISION_DIR, FIGURES_DIR, os.path.join(MANUSCRIPT_DIR, "Figures")):
            abs_p = os.path.abspath(os.path.join(cand_dir, os.path.basename(path) if not os.path.exists(os.path.join(cand_dir, path)) else path))
            if os.path.exists(abs_p):
                return f"![{alt}]({abs_p})"
        return match.group(0)

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_img, md_text)


# ==============================================================================
# 2. Publication-Grade DOCX Formatting (Times New Roman, Booktabs Tables)
# ==============================================================================
def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    for edge, spec in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        edge_el = tcBorders.find(qn(f"w:{edge}"))
        if edge_el is None:
            edge_el = OxmlElement(f"w:{edge}")
            tcBorders.append(edge_el)
        if spec is None:
            edge_el.set(qn("w:val"), "none")
        else:
            edge_el.set(qn("w:val"), spec.get("val", "single"))
            edge_el.set(qn("w:sz"), str(spec.get("sz", 4)))
            edge_el.set(qn("w:space"), "0")
            edge_el.set(qn("w:color"), spec.get("color", "000000"))


def format_docx_publication_style(docx_path):
    """Apply clean academic Times New Roman typography and booktabs table borders."""
    doc = docx.Document(docx_path)

    # Unwrap <w:hyperlink> elements so all text runs use clean black academic styling
    for hl in list(doc.element.xpath(".//w:hyperlink")):
        parent = hl.getparent()
        idx = parent.index(hl)
        for child in list(hl):
            rPr = child.find(qn("w:rPr"))
            if rPr is not None:
                for tag_name in ("w:rStyle", "w:u", "w:color"):
                    el = rPr.find(qn(tag_name))
                    if el is not None:
                        rPr.remove(el)
            parent.insert(idx, child)
            idx += 1
        parent.remove(hl)

    body_style_names = [
        "Normal", "Body Text", "First Paragraph", "Compact",
        "Block Text", "Bibliography", "List Paragraph", "Caption",
        "Table Caption", "Image Caption", "Source Code",
    ]
    for sname in body_style_names:
        if sname in doc.styles:
            st = doc.styles[sname]
            if hasattr(st, "font") and st.font is not None:
                st.font.name = "Times New Roman"
                st.font.size = Pt(12)
                st.font.color.rgb = RGBColor(0, 0, 0)

    heading_specs = {
        "Title": (13, True, False),
        "Heading 1": (13, True, False),
        "Heading 2": (12, True, False),
        "Heading 3": (12, True, False),
        "Heading 4": (12, True, True),
    }
    for hname, (pt_size, is_bold, is_italic) in heading_specs.items():
        if hname in doc.styles:
            st = doc.styles[hname]
            if hasattr(st, "font") and st.font is not None:
                st.font.name = "Times New Roman"
                st.font.size = Pt(pt_size)
                st.font.bold = is_bold
                st.font.italic = is_italic
                st.font.color.rgb = RGBColor(0, 0, 0)

    for char_style in ("Verbatim Char", "Hyperlink", "FollowedHyperlink"):
        if char_style in doc.styles:
            cs = doc.styles[char_style]
            if hasattr(cs, "font") and cs.font is not None:
                cs.font.name = "Times New Roman"
                cs.font.size = Pt(12)
                cs.font.underline = False
                cs.font.color.rgb = RGBColor(0, 0, 0)

    caption_re = re.compile(r"^(?:\*\*)?(?:Table|Figure|Fig\.)\s+(?:S|[A-Z])?\d+[\.\:]")
    tbl_caption_re = re.compile(r"^(?:\*\*)?Table\s+(?:S|[A-Z])?\d+[\.\:]")
    for p in doc.paragraphs:
        sname = p.style.name if p.style else "Normal"
        has_img = len(p._p.xpath(".//w:drawing")) > 0
        has_math = len(p._p.xpath(".//m:oMathPara | .//m:oMath")) > 0
        txt = p.text.strip()

        if sname.startswith("Heading") or sname == "Title":
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.keep_with_next = True
            pt_size, is_bold, is_italic = heading_specs.get(sname, (12, True, False))
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(pt_size)
                r.font.bold = is_bold
                if is_italic:
                    r.font.italic = True
                r.font.color.rgb = RGBColor(0, 0, 0)
        else:
            is_caption = bool(caption_re.match(txt))
            if tbl_caption_re.match(txt):
                p.paragraph_format.keep_with_next = True
            if not has_img and not has_math and not is_caption:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            for r in p.runs:
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.font.color.rgb = RGBColor(0, 0, 0)

    # Booktabs formatting for all tables (adaptive font size, cantSplit rows, repeating headers)
    for tbl in doc.tables:
        num_rows = len(tbl.rows)
        num_cols = len(tbl.columns) if num_rows > 0 else 0
        tbl_pt = 9.5 if num_cols >= 8 else (10.5 if num_cols >= 6 else 11.0)
        for r_idx, row in enumerate(tbl.rows):
            is_first = r_idx == 0
            is_last = r_idx == num_rows - 1
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                cs_el = OxmlElement("w:cantSplit")
                placed = False
                for succ in ("w:trHeight", "w:tblHeader", "w:tblCellSpacing", "w:jc", "w:hidden", "w:ins", "w:del", "w:trPrChange"):
                    s_el = trPr.find(qn(succ))
                    if s_el is not None:
                        s_el.addprevious(cs_el)
                        placed = True
                        break
                if not placed:
                    trPr.append(cs_el)
            if is_first and trPr.find(qn("w:tblHeader")) is None:
                th_el = OxmlElement("w:tblHeader")
                placed = False
                for succ in ("w:tblCellSpacing", "w:jc", "w:hidden", "w:ins", "w:del", "w:trPrChange"):
                    s_el = trPr.find(qn(succ))
                    if s_el is not None:
                        s_el.addprevious(th_el)
                        placed = True
                        break
                if not placed:
                    trPr.append(th_el)
            top_b = {"val": "single", "sz": 12, "color": "000000"} if is_first else None
            bot_b = (
                {"val": "single", "sz": 6, "color": "000000"}
                if is_first
                else ({"val": "single", "sz": 12, "color": "000000"} if is_last else None)
            )
            for c_idx, cell in enumerate(row.cells):
                set_cell_borders(cell, top=top_b, bottom=bot_b, left=None, right=None)
                for p in cell.paragraphs:
                    if c_idx == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        r.font.name = "Times New Roman"
                        r.font.size = Pt(tbl_pt)
                        r.font.color.rgb = RGBColor(0, 0, 0)
                        if is_first:
                            r.font.bold = True

    doc.core_properties.author = AUTHOR
    doc.core_properties.last_modified_by = AUTHOR
    doc.save(docx_path)


def export_to_docx(md_path, docx_path):
    """Compile Markdown to publication-grade DOCX via pypandoc + python-docx."""
    print(f"  [DOCX] Compiling {os.path.relpath(docx_path, PROJECT_DIR)} ...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = preprocess_markdown(md_text, os.path.dirname(md_path))
    tmp_md = os.path.join(EXPORT_DIR, f"_tmp_{os.getpid()}_{os.path.basename(md_path)}")
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_text)

    try:
        pypandoc.convert_file(
            tmp_md,
            "docx",
            format="markdown-smart",
            outputfile=docx_path,
            extra_args=["--wrap=auto", "--standalone"],
        )
        format_docx_publication_style(docx_path)
        size_kb = os.path.getsize(docx_path) / 1024.0
        print(f"         Saved ({size_kb:.1f} KB)")
    finally:
        if os.path.exists(tmp_md):
            os.remove(tmp_md)


# ==============================================================================
# 3. Publication-Grade PDF Export (Pandoc HTML5 + MathML + Headless Chrome)
# ==============================================================================
def export_to_pdf(md_path, pdf_path, pagetitle):
    """Compile Markdown to publication-grade PDF via Pandoc HTML5+MathML and Headless Chrome."""
    print(f"  [PDF]  Compiling {os.path.relpath(pdf_path, PROJECT_DIR)} ...")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = preprocess_markdown(md_text, os.path.dirname(md_path))
    tmp_md = os.path.join(EXPORT_DIR, f"_tmp_pdf_{os.getpid()}.md")
    tmp_html = os.path.join(EXPORT_DIR, f"_tmp_pdf_{os.getpid()}.html")
    css_file = os.path.join(EXPORT_DIR, f"_tmp_pdf_{os.getpid()}.css")

    pdf_css = """
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
}
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    line-height: 1.5;
    color: #000000;
    text-align: justify;
    margin: 0;
    padding: 0;
}
a {
    color: #000000;
    text-decoration: none;
}
h1 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 13pt;
    font-weight: bold;
    text-align: left;
    margin-top: 0;
    margin-bottom: 12pt;
    line-height: 1.3;
}
h2 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    font-weight: bold;
    text-align: left;
    margin-top: 18pt;
    margin-bottom: 6pt;
}
h3 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 12pt;
    font-weight: bold;
    text-align: left;
    margin-top: 14pt;
    margin-bottom: 4pt;
}
h4 {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11.5pt;
    font-weight: bold;
    font-style: italic;
    text-align: left;
    margin-top: 12pt;
    margin-bottom: 4pt;
}
h1, h2, h3, h4, .keep-with-next {
    page-break-after: avoid;
    break-after: avoid;
    page-break-inside: avoid;
    break-inside: avoid;
}
p {
    margin-top: 0;
    margin-bottom: 8pt;
    orphans: 2;
    widows: 2;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8pt 0 12pt 0;
    font-size: 9.2pt;
    font-family: 'Times New Roman', Times, serif;
}
thead {
    display: table-header-group;
}
tr {
    page-break-inside: avoid;
    break-inside: avoid;
}
th {
    font-weight: bold;
    padding: 4pt 4pt;
    border-top: 1.5pt solid #000000;
    border-bottom: 0.75pt solid #000000;
    text-align: left;
    vertical-align: bottom;
}
td {
    padding: 3pt 4pt;
    text-align: left;
    vertical-align: top;
}
tbody tr:last-child td {
    border-bottom: 1.5pt solid #000000;
}
img {
    max-width: 96%;
    height: auto;
    display: block;
    margin: 12pt auto;
    page-break-inside: avoid;
}
figcaption {
    font-size: 10.5pt;
    text-align: left;
    margin-top: 4pt;
    margin-bottom: 10pt;
}
blockquote {
    margin: 8pt 16pt;
    padding-left: 10pt;
    border-left: 2pt solid #666666;
}
code {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
}
hr {
    border: none;
    border-top: 0.75pt solid #cccccc;
    margin: 14pt 0;
}
"""
    with open(tmp_md, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(css_file, "w", encoding="utf-8") as f:
        f.write(pdf_css)

    try:
        pypandoc.convert_file(
            tmp_md,
            "html5",
            format="markdown-smart",
            outputfile=tmp_html,
            extra_args=[
                "--standalone",
                "--embed-resources",
                f"--css={css_file}",
                f"--metadata=pagetitle:{pagetitle}",
                "--mathml",
            ],
        )
        # Attach keep-with-next class to Table X. caption paragraphs so captions stay glued to their tables
        with open(tmp_html, "r", encoding="utf-8") as hf:
            html_txt = hf.read()
        html_txt = re.sub(
            r"<p>(\s*(?:<strong>)?\s*Table\s+(?:S|[A-Z])?\d+(?:</strong>)?\s*[\.\:])",
            r'<p class="keep-with-next">\1',
            html_txt,
        )
        with open(tmp_html, "w", encoding="utf-8") as hf:
            hf.write(html_txt)

        chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if not os.path.exists(chrome_bin):
            raise FileNotFoundError(f"Google Chrome binary not found at {chrome_bin}")
        subprocess.run(
            [
                chrome_bin,
                "--headless",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}",
                tmp_html,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        size_kb = os.path.getsize(pdf_path) / 1024.0
        print(f"         Saved ({size_kb:.1f} KB)")
    finally:
        for fpath in (tmp_md, tmp_html, css_file):
            if os.path.exists(fpath):
                os.remove(fpath)


# ==============================================================================
# 4. Native Word Track Changes Generator (Dynamic Section-Anchored Alignment)
# ==============================================================================
def get_next_id():
    global CHANGE_ID
    CHANGE_ID += 1
    return str(CHANGE_ID)


def get_paragraph_timestamp(r_idx, sub_offset=0):
    base = datetime(2026, 11, 2, 9, 15, 0)
    delta = timedelta(minutes=r_idx * 3, seconds=(r_idx * 17 + sub_offset * 5) % 60)
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_rPr(pt_size=12, is_bold=False, is_italic=False, color_hex="000000", underline=False, strike=False):
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:ascii"), "Times New Roman")
    rFonts.set(qn("w:hAnsi"), "Times New Roman")
    rFonts.set(qn("w:cs"), "Times New Roman")
    rPr.append(rFonts)
    if is_bold:
        rPr.append(OxmlElement("w:b"))
    if is_italic:
        rPr.append(OxmlElement("w:i"))
    if underline:
        u = OxmlElement("w:u")
        u.set(qn("w:val"), "single")
        rPr.append(u)
    if strike:
        rPr.append(OxmlElement("w:strike"))
    color = OxmlElement("w:color")
    color.set(qn("w:val"), color_hex)
    rPr.append(color)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(pt_size * 2)))
    rPr.append(sz)
    szCs = OxmlElement("w:szCs")
    szCs.set(qn("w:val"), str(int(pt_size * 2)))
    rPr.append(szCs)
    return rPr


def get_style_font_spec(p):
    sname = p.style.name if p.style else "Normal"
    if sname in ("Heading 1", "Title"):
        return (13, True, False)
    if sname in ("Heading 2", "Heading 3"):
        return (12, True, False)
    if sname == "Heading 4":
        return (12, True, True)
    return (12, False, False)


def make_normal_run(text, pt_size=12, is_bold=False, is_italic=False):
    r = OxmlElement("w:r")
    r.append(build_rPr(pt_size=pt_size, is_bold=is_bold, is_italic=is_italic, color_hex="000000"))
    t = OxmlElement("w:t")
    t.set(qn("xml:space"), "preserve")
    t.text = text
    r.append(t)
    return r


def make_ins_spans(spans, date_str, pt_size=12):
    ins = OxmlElement("w:ins")
    ins.set(qn("w:id"), get_next_id())
    ins.set(qn("w:author"), AUTHOR)
    ins.set(qn("w:date"), date_str)
    for text, is_bold, is_italic in spans:
        r = OxmlElement("w:r")
        r.append(build_rPr(pt_size=pt_size, is_bold=is_bold, is_italic=is_italic, color_hex="002060", underline=True))
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        ins.append(r)
    return ins


def make_del_spans(spans, date_str, pt_size=12):
    d = OxmlElement("w:del")
    d.set(qn("w:id"), get_next_id())
    d.set(qn("w:author"), AUTHOR)
    d.set(qn("w:date"), date_str)
    for text, is_bold, is_italic in spans:
        r = OxmlElement("w:r")
        r.append(build_rPr(pt_size=pt_size, is_bold=is_bold, is_italic=is_italic, color_hex="C00000", strike=True))
        dt = OxmlElement("w:delText")
        dt.set(qn("xml:space"), "preserve")
        dt.text = text
        r.append(dt)
        d.append(r)
    return d


def tokenize(text):
    return re.findall(r"\S+\s*|\s+", text)


def extract_text_and_styles(paras_or_p):
    paras = [paras_or_p] if not isinstance(paras_or_p, (list, tuple)) else [p for p in paras_or_p if p.text.strip()]
    full_chars, char_styles = [], []
    for idx, p in enumerate(paras):
        _, s_bold, s_italic = get_style_font_spec(p)
        if idx > 0 and full_chars:
            full_chars.append(" ")
            char_styles.append((s_bold, s_italic))
        runs = p.runs
        if runs:
            p_chars, p_styles = [], []
            for r in runs:
                rb = bool(r.bold) or s_bold
                ri = bool(r.italic) or s_italic
                for ch in r.text:
                    p_chars.append(ch)
                    p_styles.append((rb, ri))
            p_str = "".join(p_chars)
            l_strip = len(p_str) - len(p_str.lstrip())
            r_strip = len(p_str.rstrip())
            full_chars.extend(p_chars[l_strip:r_strip])
            char_styles.extend(p_styles[l_strip:r_strip])
        else:
            for ch in p.text.strip():
                full_chars.append(ch)
                char_styles.append((s_bold, s_italic))
    return "".join(full_chars), char_styles


def group_char_spans(text, char_styles, c1, c2):
    if c1 >= c2:
        return []
    spans = []
    cur_b, cur_i = char_styles[c1] if c1 < len(char_styles) else (False, False)
    start = c1
    for pos in range(c1 + 1, c2):
        b, i = char_styles[pos] if pos < len(char_styles) else (cur_b, cur_i)
        if (b, i) != (cur_b, cur_i):
            spans.append((text[start:pos], cur_b, cur_i))
            start = pos
            cur_b, cur_i = b, i
    spans.append((text[start:c2], cur_b, cur_i))
    return spans


def has_drawing_or_math(p):
    return len(p._p.xpath(".//w:drawing | .//m:oMathPara | .//m:oMath")) > 0


def clear_paragraph_content(p):
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)


def apply_word_diff_to_paragraph(p, orig_source, r_idx):
    if has_drawing_or_math(p):
        return
    old_text, old_styles = extract_text_and_styles(orig_source)
    new_text, new_styles = extract_text_and_styles(p)
    if old_text.strip() == new_text.strip():
        return

    pt_size, _, _ = get_style_font_spec(p)
    clear_paragraph_content(p)
    t_old = tokenize(old_text)
    t_new = tokenize(new_text)

    old_offsets = [0]
    for tok in t_old:
        old_offsets.append(old_offsets[-1] + len(tok))
    new_offsets = [0]
    for tok in t_new:
        new_offsets.append(new_offsets[-1] + len(tok))

    sm = difflib.SequenceMatcher(None, t_old, t_new)
    sub_idx = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        sub_idx += 1
        date_str = get_paragraph_timestamp(r_idx, sub_idx)
        if tag == "equal":
            for span_txt, b, it in group_char_spans(new_text, new_styles, new_offsets[j1], new_offsets[j2]):
                p._p.append(make_normal_run(span_txt, pt_size, b, it))
        elif tag == "delete":
            spans = group_char_spans(old_text, old_styles, old_offsets[i1], old_offsets[i2])
            if spans:
                p._p.append(make_del_spans(spans, date_str, pt_size))
        elif tag == "insert":
            spans = group_char_spans(new_text, new_styles, new_offsets[j1], new_offsets[j2])
            if spans:
                p._p.append(make_ins_spans(spans, date_str, pt_size))
        elif tag == "replace":
            spans_del = group_char_spans(old_text, old_styles, old_offsets[i1], old_offsets[i2])
            spans_ins = group_char_spans(new_text, new_styles, new_offsets[j1], new_offsets[j2])
            if spans_del:
                p._p.append(make_del_spans(spans_del, date_str, pt_size))
            if spans_ins:
                p._p.append(make_ins_spans(spans_ins, date_str, pt_size))


def set_paragraph_as_inserted(p, r_idx):
    if has_drawing_or_math(p):
        return
    new_text, new_styles = extract_text_and_styles(p)
    if not new_text.strip():
        return
    pt_size, _, _ = get_style_font_spec(p)
    clear_paragraph_content(p)
    spans = group_char_spans(new_text, new_styles, 0, len(new_text))
    if spans:
        p._p.append(make_ins_spans(spans, get_paragraph_timestamp(r_idx, 1), pt_size))


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def extract_author_year_key(ref_text):
    m = re.match(r"^([^\(]+)\((\d{4}[a-z]?)\)", ref_text.strip())
    if not m:
        return None
    auth_raw = strip_accents(m.group(1))
    auth_no_init = re.sub(r"\b[A-Za-z]\.", "", auth_raw)
    all_surnames = re.sub(r"[^a-z0-9]", "", auth_no_init.lower())
    year = m.group(2).strip()
    return (all_surnames, year)


def make_deleted_paragraph_element(orig_p, r_idx, sub_offset=1):
    """Create a schema-valid <w:p> element marking an entire removed original paragraph with <w:del>."""
    old_text, old_styles = extract_text_and_styles(orig_p)
    if not old_text.strip():
        return None
    pt_size, _, _ = get_style_font_spec(orig_p)
    sname = orig_p.style.name if orig_p.style else "Normal"
    is_left = sname.startswith("Heading") or sname == "Title"
    date_str = get_paragraph_timestamp(r_idx, sub_offset)

    new_p = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    jc = OxmlElement("w:jc")
    jc.set(qn("w:val"), "left" if is_left else "both")
    pPr.append(jc)
    p_rPr = OxmlElement("w:rPr")
    p_del = OxmlElement("w:del")
    p_del.set(qn("w:id"), get_next_id())
    p_del.set(qn("w:author"), AUTHOR)
    p_del.set(qn("w:date"), date_str)
    p_rPr.append(p_del)
    pPr.append(p_rPr)
    new_p.append(pPr)

    spans = group_char_spans(old_text, old_styles, 0, len(old_text))
    if spans:
        new_p.append(make_del_spans(spans, date_str, pt_size))
    return new_p


def classify_bucket(heading_text, is_original=False):
    """Map section/subsection heading text to canonical semantic alignment buckets."""
    t = heading_text.strip()
    tl = t.lower()
    if tl == "abstract":
        return "ABSTRACT"
    if tl in ("introduction", "1. introduction"):
        return "SEC1"
    if t.startswith("2.1"):
        return "SEC2.1"
    if t.startswith("2.2"):
        return "SEC2.2"
    if t.startswith("2.3"):
        return "SEC2.3"
    if t.startswith("2.4"):
        return "SEC2.4"
    if t.startswith("2."):
        return "SEC2.0"
    if t.startswith("3."):
        return "SEC3"
    if t.startswith("4.1"):
        return "SEC4.1"
    if t.startswith("4.2"):
        return "SEC4.2"
    if t.startswith("4.3"):
        return "SEC4.3"
    if t.startswith("4.4"):
        return "SEC4.4"
    if t.startswith("4.5"):
        return "SEC4.4" if is_original else "SEC4.5_NEW"
    if t.startswith("4.6"):
        return "SEC4.6"
    if t.startswith("4."):
        return "SEC4.0"
    if t.startswith("5.1"):
        return "SEC5.1"
    if t.startswith("5.2"):
        return "SEC5.2"
    if t.startswith("5.3"):
        return "SEC5.3"
    if t.startswith("5.4"):
        return "SEC5.4"
    if t.startswith("5.5"):
        return "SEC5.5_NEW"
    if t.startswith("5."):
        return "SEC5.0"
    if t.startswith("6."):
        return "SEC6"
    if t.startswith("7."):
        return "SEC7"
    if t.startswith("8."):
        return "SEC8_NEW"
    if tl.startswith("data availability"):
        return "DA"
    if tl.startswith("code availability"):
        return "CA"
    if tl.startswith("competing interests"):
        return "CI"
    if tl.startswith("artificial intelligence"):
        return "AI_ORIG"
    if tl == "references":
        return "REF"
    if tl == "tables":
        return "TABLES_END"
    return None


def word_sim(a, b):
    wa = set(re.findall(r"[a-z0-9]{3,}", a.lower()))
    wb = set(re.findall(r"[a-z0-9]{3,}", b.lower()))
    if not wa or not wb:
        return 0.0
    jacc = len(wa & wb) / float(len(wa | wb))
    seq = difflib.SequenceMatcher(None, a[:400].lower(), b[:400].lower()).ratio()
    return 0.55 * jacc + 0.45 * seq


def monotonic_align_bucket(orig_list, rev_list, min_sim=0.14):
    """Monotonic dynamic-programming alignment between prose paragraphs within a single section bucket."""
    m, n = len(orig_list), len(rev_list)
    if m == 0 or n == 0:
        return {}
    if m == 1 and n == 1 and word_sim(orig_list[0][1].text, rev_list[0][1].text) >= min_sim:
        return {0: 0}

    sim = [[word_sim(orig_list[i][1].text, rev_list[j][1].text) for j in range(n)] for i in range(m)]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    choice = [[None] * n for _ in range(m)]

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            best = dp[i + 1][j]
            act = ("skip_orig",)
            if dp[i][j + 1] > best:
                best = dp[i][j + 1]
                act = ("skip_rev",)
            if sim[i][j] >= min_sim:
                cand = sim[i][j] + dp[i + 1][j + 1]
                if cand >= best:
                    best = cand
                    act = ("match", i, j)
            dp[i][j] = best
            choice[i][j] = act

    mapping = {}
    i, j = 0, 0
    while i < m and j < n:
        act = choice[i][j]
        if act[0] == "match":
            mapping[j] = i
            i += 1
            j += 1
        elif act[0] == "skip_orig":
            i += 1
        else:
            j += 1
    return mapping


def build_track_changes(orig_docx_path, rev_docx_path, out_docx_path):
    """Build native Word Track Changes document comparing original submission to Round-2 revision."""
    print(f"  [TRACK] Building {os.path.relpath(out_docx_path, PROJECT_DIR)} ...")
    doc_orig = docx.Document(orig_docx_path)
    doc_rev = docx.Document(rev_docx_path)

    orig_paras = doc_orig.paragraphs
    rev_paras = doc_rev.paragraphs

    # Extract original Figure 1..5 captions and Table 1..3 captions
    orig_fig_caps = {}
    orig_tbl_caps = {}
    for p in orig_paras:
        t = p.text.strip()
        mf = re.match(r"^Figure\s+([1-5])[\.\:]", t)
        if mf:
            orig_fig_caps[int(mf.group(1))] = p
        mt = re.match(r"^Table\s+([1-3])[\.\:]", t)
        if mt:
            orig_tbl_caps[int(mt.group(1))] = p

    # Partition original paragraphs into semantic buckets (including alphabetical REF bucket)
    orig_headings = {}
    orig_buckets = {}
    cur_bucket = "TITLE_FRONT"
    for idx, p in enumerate(orig_paras):
        sname = p.style.name if p.style else "Normal"
        txt = p.text.strip()
        if not txt or has_drawing_or_math(p):
            continue
        if idx == 0 and sname in ("Heading 1", "Title"):
            orig_headings["MAIN_TITLE"] = p
            continue
        if sname.startswith("Heading"):
            b = classify_bucket(txt, is_original=True)
            if b:
                cur_bucket = b
                if b not in orig_headings:
                    orig_headings[b] = p
            continue
        if cur_bucket == "TABLES_END":
            continue
        orig_buckets.setdefault(cur_bucket, []).append((idx, p))

    # Partition revised paragraphs and apply heading/caption diffs
    rev_buckets = {}
    rev_headings = {}
    cur_bucket = "TITLE_FRONT"
    for r_idx, p in enumerate(rev_paras):
        sname = p.style.name if p.style else "Normal"
        txt = p.text.strip()
        if not txt or has_drawing_or_math(p):
            continue
        if r_idx == 0 and sname in ("Heading 1", "Title"):
            if "MAIN_TITLE" in orig_headings:
                apply_word_diff_to_paragraph(p, orig_headings["MAIN_TITLE"], r_idx)
            continue

        if sname.startswith("Heading"):
            b = classify_bucket(txt, is_original=False)
            if b:
                cur_bucket = b
                rev_headings[b] = (r_idx, p)
                if b in orig_headings and not b.endswith("_NEW"):
                    apply_word_diff_to_paragraph(p, orig_headings[b], r_idx)
                else:
                    set_paragraph_as_inserted(p, r_idx)
            else:
                set_paragraph_as_inserted(p, r_idx)
            continue

        if cur_bucket == "REF":
            rev_buckets.setdefault("REF", []).append((r_idx, p))
            continue

        # Special handling for Figure captions, Table captions, and Table Notes
        mf = re.match(r"^Figure\s+([1-5])[\.\:]", txt)
        if mf:
            fnum = int(mf.group(1))
            if fnum in orig_fig_caps:
                apply_word_diff_to_paragraph(p, orig_fig_caps[fnum], r_idx)
            else:
                set_paragraph_as_inserted(p, r_idx)
            continue

        mt = re.match(r"^Table\s+(\d+)[\.\:]", txt)
        if mt:
            tnum = int(mt.group(1))
            # Revised Table 1 -> Orig Table 1; Revised Table 2 -> Orig Table 2; Revised Table 4 (PSM) -> Orig Table 3
            orig_t_map = {1: 1, 2: 2, 4: 3}
            if tnum in orig_t_map and orig_t_map[tnum] in orig_tbl_caps:
                apply_word_diff_to_paragraph(p, orig_tbl_caps[orig_t_map[tnum]], r_idx)
            else:
                set_paragraph_as_inserted(p, r_idx)
            continue

        if txt.startswith(("Note.", "Note:", "Notes.", "Notes:")):
            set_paragraph_as_inserted(p, r_idx)
            continue

        rev_buckets.setdefault(cur_bucket, []).append((r_idx, p))

    # Align prose & reference paragraphs inside each bucket and emit <w:del> for removed original paragraphs
    for bucket, rev_list in rev_buckets.items():
        if bucket.endswith("_NEW") or bucket not in orig_buckets:
            for r_idx, p in rev_list:
                set_paragraph_as_inserted(p, r_idx)
            continue
        orig_list = orig_buckets[bucket]
        bucket_min_sim = 0.32 if bucket == "REF" else 0.14
        align_map = monotonic_align_bucket(orig_list, rev_list, min_sim=bucket_min_sim)
        next_orig = 0
        for rev_pos, (r_idx, p_rev) in enumerate(rev_list):
            if rev_pos in align_map:
                orig_pos = align_map[rev_pos]
                while next_orig < orig_pos:
                    del_el = make_deleted_paragraph_element(orig_list[next_orig][1], r_idx, next_orig + 1)
                    if del_el is not None:
                        p_rev._p.addprevious(del_el)
                    next_orig += 1
                apply_word_diff_to_paragraph(p_rev, orig_list[orig_pos][1], r_idx)
                next_orig = orig_pos + 1
            else:
                set_paragraph_as_inserted(p_rev, r_idx)
        if rev_list:
            last_r_idx, last_p_rev = rev_list[-1]
            anchor = last_p_rev._p
            while next_orig < len(orig_list):
                del_el = make_deleted_paragraph_element(orig_list[next_orig][1], last_r_idx, next_orig + 1)
                if del_el is not None:
                    anchor.addnext(del_el)
                    anchor = del_el
                next_orig += 1

    # Emit <w:del> for original sections moved out of blinded manuscript (e.g., AI_ORIG moved to Title Page II)
    if "REF" in rev_headings:
        ref_r_idx, ref_h_p = rev_headings["REF"]
        if "AI_ORIG" in orig_headings:
            del_h = make_deleted_paragraph_element(orig_headings["AI_ORIG"], ref_r_idx, 1)
            if del_h is not None:
                ref_h_p._p.addprevious(del_h)
        for sub_i, (_, p_ai) in enumerate(orig_buckets.get("AI_ORIG", [])):
            del_p = make_deleted_paragraph_element(p_ai, ref_r_idx, sub_i + 2)
            if del_p is not None:
                ref_h_p._p.addprevious(del_p)

    doc_rev.core_properties.author = AUTHOR
    doc_rev.core_properties.last_modified_by = AUTHOR
    doc_rev.save(out_docx_path)

    # Verify generated XML contains valid <w:ins> and <w:del> elements and preserves embedded figures
    verify_doc = docx.Document(out_docx_path)
    n_ins = len(verify_doc.element.xpath(".//w:ins"))
    n_del = len(verify_doc.element.xpath(".//w:del"))
    n_img = len(verify_doc.element.xpath(".//w:drawing"))
    size_kb = os.path.getsize(out_docx_path) / 1024.0
    print(f"         Saved ({size_kb:.1f} KB | w:ins={n_ins}, w:del={n_del}, figures={n_img})")


# ==============================================================================
# 5. Master Compilation Entrypoint
# ==============================================================================
def main():
    print("=" * 82)
    print("BSE REVISION DOCX, TRACK-CHANGES & PDF COMPILATION PIPELINE")
    print("=" * 82)
    print(f"  Pandoc version : {pypandoc.get_pandoc_version()}")

    main_title = "Green Offerings and Net-Zero Plans: Markets, Capabilities and Finance in European SMEs"
    si_title = "Supporting Information: Green Offerings and Net-Zero Plans in European SMEs"

    docx_jobs = [
        (os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.md"),
         os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.docx")),
        (os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.md"),
         os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.docx")),
        (os.path.join(REVISION_DIR, "Title Page_I_Revision.md"),
         os.path.join(REVISION_DIR, "Title Page_I_Revision.docx")),
        (os.path.join(REVISION_DIR, "Title Page_II_Revision.md"),
         os.path.join(REVISION_DIR, "Title Page_II_Revision.docx")),
        (os.path.join(REVISION_DIR, "Response_to_Reviewers.md"),
         os.path.join(REVISION_DIR, "Response_to_Reviewers.docx")),
        (os.path.join(REVISION_DIR, "Response_to_Coauthor_Hari.md"),
         os.path.join(REVISION_DIR, "Response_to_Coauthor_Hari.docx")),
        (os.path.join(REVISION_DIR, "CoAuthor_Verification_Note_Mausam_Risal.md"),
         os.path.join(REVISION_DIR, "CoAuthor_Verification_Note_Mausam_Risal.docx")),
        (os.path.join(PROJECT_DIR, "README.md"),
         os.path.join(PROJECT_DIR, "README.docx")),
    ]

    for md_src, docx_dst in docx_jobs:
        if not os.path.exists(md_src):
            raise FileNotFoundError(f"Missing required Markdown source: {md_src}")
        export_to_docx(md_src, docx_dst)

    # Compile PDFs for Main Manuscript and Supporting Information
    export_to_pdf(
        os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.md"),
        os.path.join(REVISION_DIR, "1_BSE_Main_Manuscript_Revision.pdf"),
        main_title,
    )
    export_to_pdf(
        os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.md"),
        os.path.join(REVISION_DIR, "2_BSE_Supporting_Information_Revision.pdf"),
        si_title,
    )

    # Build native Word Track Changes manuscript
    if os.path.exists(ORIG_SUBMISSION_DOCX):
        build_track_changes(
            ORIG_SUBMISSION_DOCX,
            os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.docx"),
            os.path.join(REVISION_DIR, "BSE_Main_Manuscript_TrackChanges.docx"),
        )

    # Remove stale legacy HTML/Appendices exports in Manuscript/Export/ and sync fresh deliverables
    for fname in os.listdir(EXPORT_DIR):
        if fname.endswith(".html") or fname.startswith("BSE_Appendices_"):
            os.remove(os.path.join(EXPORT_DIR, fname))

    export_copies = [
        (os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.docx"),
         os.path.join(EXPORT_DIR, "BSE_Main_Manuscript_Revision.docx")),
        (os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.docx"),
         os.path.join(EXPORT_DIR, "BSE_Main_Manuscript_Final.docx")),
        (os.path.join(REVISION_DIR, "BSE_Main_Manuscript_Revision.docx"),
         os.path.join(EXPORT_DIR, "SME_Manuscript_Final.docx")),
        (os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.docx"),
         os.path.join(EXPORT_DIR, "BSE_Supporting_Information_Revision.docx")),
        (os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.docx"),
         os.path.join(EXPORT_DIR, "BSE_Supporting_Information_Final.docx")),
        (os.path.join(REVISION_DIR, "BSE_Supporting_Information_Revision.docx"),
         os.path.join(EXPORT_DIR, "Supporting_Information.docx")),
        (os.path.join(REVISION_DIR, "Response_to_Reviewers.docx"),
         os.path.join(EXPORT_DIR, "Response_to_Reviewers.docx")),
        (os.path.join(REVISION_DIR, "1_BSE_Main_Manuscript_Revision.pdf"),
         os.path.join(EXPORT_DIR, "1_BSE_Main_Manuscript_Revision.pdf")),
        (os.path.join(REVISION_DIR, "2_BSE_Supporting_Information_Revision.pdf"),
         os.path.join(EXPORT_DIR, "2_BSE_Supporting_Information_Revision.pdf")),
    ]
    if os.path.exists(os.path.join(REVISION_DIR, "BSE_Main_Manuscript_TrackChanges.docx")):
        export_copies.append(
            (os.path.join(REVISION_DIR, "BSE_Main_Manuscript_TrackChanges.docx"),
             os.path.join(EXPORT_DIR, "BSE_Main_Manuscript_TrackChanges.docx"))
        )
    for src_f, dst_f in export_copies:
        shutil.copy2(src_f, dst_f)

    print("=" * 82)
    print("ALL DOCX, TRACK-CHANGES & PDF DELIVERABLES COMPILED SUCCESSFULLY")
    print("=" * 82)


if __name__ == "__main__":
    main()
