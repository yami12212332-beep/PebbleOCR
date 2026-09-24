"""
Hybrid document extraction: text AND tables, with accuracy that matches
the source.

Typed/digital PDF pages already contain a perfect, ground-truth text layer.
Rasterizing them into an image and re-reading that image with OCR throws
that perfect text away and replaces it with a guess -- and small, dense
body text is exactly where OCR guesses worst (this is what caused the
garbled paragraphs on the discharge summary).

So: for any PDF page that has a real text layer, this script reads text
and tables DIRECTLY from the PDF via PyMuPDF -- zero OCR, 100% accurate.
OCR (PPStructureV3) is only used as a fallback for pages with no text
layer at all (genuine scans/photos) and for plain image files.

Requires: uv add "paddleocr[doc-parser]==3.7.0" pymupdf

Usage:
    uv run python document_extract_hybrid.py path/to/file.pdf
    uv run python document_extract_hybrid.py path/to/image.png
    uv run python document_extract_hybrid.py path/to/folder/
"""

import sys
import html
from pathlib import Path

import pymupdf
from paddleocr import PPStructureV3

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# A page with fewer extractable characters than this is treated as a scan
# (no usable text layer) rather than a typed page.
MIN_TEXT_CHARS = 30

# ---- OCR fallback (image / scanned page) rendering, same rules as before ----
HEADING_LABELS = {"doc_title": "h1", "paragraph_title": "h2"}
CAPTION_LABELS = {"table_title", "figure_title", "chart_title"}
SKIP_LABELS = {"header", "footer", "page_number", "header_image", "footer_image"}
PLACEHOLDER_LABELS = {"figure": "[figure]", "image": "[image]", "chart": "[chart]", "seal": "[seal]"}


def is_page_number(content: str) -> bool:
    stripped = content.strip()
    return stripped.isdigit() and len(stripped) <= 4


def render_ocr_block(block: dict) -> str:
    label = block.get("block_label", "text")
    content = (block.get("block_content") or "").strip()
    if label in SKIP_LABELS or not content or is_page_number(content):
        return ""
    if label == "table":
        return f'<div class="table-wrap">{content}</div>'
    if label in PLACEHOLDER_LABELS:
        return f'<p class="placeholder">{PLACEHOLDER_LABELS[label]}</p>'
    if label in CAPTION_LABELS:
        return f'<p class="caption">{html.escape(content)}</p>'
    if label in HEADING_LABELS:
        tag = HEADING_LABELS[label]
        return f"<{tag}>{html.escape(content)}</{tag}>"
    if label == "formula":
        return f'<pre class="formula">{html.escape(content)}</pre>'
    return f"<p>{html.escape(content)}</p>"


def ocr_sort_key(block: dict):
    order = block.get("block_order")
    bbox = block.get("block_bbox")
    if bbox and len(bbox) >= 2:
        return (0, bbox[1], bbox[0])
    return (1, order is None, order or 0)


def render_page_via_ocr(pipeline: PPStructureV3, image_path: Path) -> str:
    output = pipeline.predict(str(image_path))
    rendered = []
    for res in output:
        data = res.json["res"]
        blocks = sorted(data.get("parsing_res_list", []), key=ocr_sort_key)
        rendered.extend(render_ocr_block(b) for b in blocks)
    rendered = [r for r in rendered if r]
    return "\n".join(rendered) if rendered else "<p class='muted'>No content detected.</p>"


# ---- Native extraction (typed PDF page) ----

def table_to_html(rows: list) -> str:
    trs = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(c or '')}</td>" for c in row)
        trs.append(f"<tr>{cells}</tr>")
    return f'<div class="table-wrap"><table>{"".join(trs)}</table></div>'


def bbox_overlaps(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def render_page_native(page: pymupdf.Page) -> str:
    tables = list(page.find_tables())
    table_bboxes = [tuple(t.bbox) for t in tables]

    items = []  # (top_y, html_string)

    for t in tables:
        rows = t.extract()
        items.append((t.bbox[1], table_to_html(rows)))

    text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    for x0, y0, x1, y1, text, *_ in text_blocks:
        text = text.strip()
        if not text:
            continue
        if any(bbox_overlaps((x0, y0, x1, y1), tb) for tb in table_bboxes):
            continue  # this text is inside a table we already extracted separately
        for para in text.split("\n\n"):
            para = para.strip().replace("\n", " ")
            if para:
                items.append((y0, f"<p>{html.escape(para)}</p>"))

    items.sort(key=lambda it: it[0])
    rendered = [html_str for _, html_str in items]
    return "\n".join(rendered) if rendered else "<p class='muted'>No content detected.</p>"


def process_file(pipeline: PPStructureV3, file_path: Path, output_dir: Path):
    print(f"Processing {file_path.name} ...")
    page_sections = []

    if file_path.suffix.lower() == ".pdf":
        doc = pymupdf.open(file_path)
        for page_index, page in enumerate(doc):
            text_layer = page.get_text().strip()
            if len(text_layer) >= MIN_TEXT_CHARS:
                print(f"  Page {page_index + 1}: typed (native extraction)")
                page_html = render_page_native(page)
            else:
                print(f"  Page {page_index + 1}: no text layer, falling back to OCR")
                tmp_img = output_dir / f"_tmp_{file_path.stem}_p{page_index + 1}.png"
                pix = page.get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72))
                pix.save(tmp_img)
                page_html = render_page_via_ocr(pipeline, tmp_img)
                tmp_img.unlink(missing_ok=True)

            page_sections.append(
                f'<section class="page"><h2 class="page-marker">Page {page_index + 1}</h2>{page_html}</section>'
            )
        doc.close()
    else:
        # plain image file -- always OCR, there's no text layer to read
        page_html = render_page_via_ocr(pipeline, file_path)
        page_sections.append(f'<section class="page">{page_html}</section>')

    body = "\n<hr>\n".join(page_sections)

    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(file_path.name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px; background: #fafafa; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ font-size: 1.2rem; color: #333; }}
  .page-marker {{ color: #999; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
  .caption {{ font-style: italic; color: #555; font-size: 0.9rem; }}
  .placeholder {{ color: #aaa; border: 1px dashed #ccc; padding: 0.5rem; text-align: center; }}
  .formula {{ background: #f0f0f0; padding: 0.5rem; overflow-x: auto; }}
  .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #ddd; padding: 0.5rem; margin: 1rem 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  table, th, td {{ border: 1px solid #ccc; }}
  td, th {{ padding: 6px 10px; font-size: 0.9rem; text-align: left; }}
  hr {{ border: none; border-top: 2px solid #ddd; margin: 2rem 0; }}
  .muted {{ color: #999; font-style: italic; }}
</style>
</head>
<body>
  <h1>{html.escape(file_path.name)}</h1>
  {body}
</body>
</html>"""

    out_path = output_dir / f"{file_path.stem}.html"
    out_path.write_text(doc_html, encoding="utf-8")
    print(f"  -> {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python document_extract_hybrid.py <file_or_folder>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in SUPPORTED_EXT]
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    if not files:
        print("No supported files found (.pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff)")
        sys.exit(1)

    pipeline = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

    for f in files:
        process_file(pipeline, f, output_dir)


if __name__ == "__main__":
    main()