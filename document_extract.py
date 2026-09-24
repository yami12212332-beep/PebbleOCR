"""
Unified document retriever: extracts both plain text/words AND tables from
images or PDFs (scanned or typed), in their original reading order, and
renders everything as a single HTML page per document.

Built on PPStructureV3, which already tags every detected block with a
block_label ("text", "paragraph_title", "table", etc.) and a block_order
(its position in the reading order) -- we just render each block according
to its label, in order. Table blocks arrive with real <table> HTML already
built in; everything else is plain text.

Requires: uv add "paddleocr[doc-parser]==3.7.0"

Usage:
    uv run python document_extract.py path/to/file.pdf
    uv run python document_extract.py path/to/image.png
    uv run python document_extract.py path/to/folder_of_docs/
"""

import sys
import html
from pathlib import Path

from paddleocr import PPStructureV3

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# How to render each block_label. Anything not listed falls back to a <p>.
HEADING_LABELS = {"doc_title": "h1", "paragraph_title": "h2"}
CAPTION_LABELS = {"table_title", "figure_title", "chart_title"}
SKIP_LABELS = {"header", "footer", "page_number", "header_image", "footer_image"}
PLACEHOLDER_LABELS = {"figure": "[figure]", "image": "[image]", "chart": "[chart]", "seal": "[seal]"}

# Sort blocks by vertical position on the page (top-to-bottom, then left-to-right)
# rather than trusting PP-StructureV3's own block_order. The model's reading-order
# prediction is tuned for complex multi-column layouts and can misplace a table
# relative to nearby text on simple single-column documents. Position-based sorting
# is more reliable for that common case; if you're processing true multi-column
# documents (magazines, papers with side-by-side columns) where reading order isn't
# simply top-to-bottom, set this to False to fall back to the model's block_order.
ORDER_BY_POSITION = True


def is_page_number(content: str) -> bool:
    """Heuristic: a short, purely numeric block by itself is almost always a
    page number the layout model mislabeled as plain text, not real content."""
    stripped = content.strip()
    return stripped.isdigit() and len(stripped) <= 4


def block_sort_key(block: dict):
    order = block.get("block_order")
    bbox = block.get("block_bbox")
    if ORDER_BY_POSITION and bbox and len(bbox) >= 2:
        # bbox is [x0, y0, x1, y1] -- sort by top edge, then left edge
        return (0, bbox[1], bbox[0])
    # fall back to the model's order (None sorts last)
    return (1, order is None, order or 0)


def render_block(block: dict) -> str:
    label = block.get("block_label", "text")
    content = (block.get("block_content") or "").strip()

    if label in SKIP_LABELS or not content or is_page_number(content):
        return ""

    if label == "table":
        # block_content is already a full <table>...</table> HTML string
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

    # default: plain text / paragraph / list / abstract / reference / footnote / etc.
    return f"<p>{html.escape(content)}</p>"


def process_file(pipeline: PPStructureV3, file_path: Path, output_dir: Path):
    print(f"Processing {file_path.name} ...")
    output = pipeline.predict(str(file_path))

    page_sections = []
    for page_index, res in enumerate(output):
        # keep raw json too, useful for debugging or downstream reuse
        res.save_to_json(save_path=str(output_dir / "raw"))

        data = res.json["res"]
        blocks = sorted(data.get("parsing_res_list", []), key=block_sort_key)
        rendered = [render_block(b) for b in blocks]
        rendered = [r for r in rendered if r]

        page_html = "\n".join(rendered) if rendered else "<p class='muted'>No content detected.</p>"
        page_sections.append(
            f'<section class="page"><h2 class="page-marker">Page {page_index + 1}</h2>{page_html}</section>'
        )

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
        print("Usage: uv run python document_extract.py <file_or_folder>")
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

    pipeline = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )

    for f in files:
        process_file(pipeline, f, output_dir)


if __name__ == "__main__":
    main()