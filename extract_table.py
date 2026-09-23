"""
Extract tables from images and PDFs (scanned or typed) and produce a
single HTML report per document, with every detected table rendered
as a real <table>.

Requires: uv add "paddleocr[doc-parser]==3.7.0"

Usage:
    uv run python extract_tables.py path/to/file.pdf
    uv run python extract_tables.py path/to/image.png
    uv run python extract_tables.py path/to/folder_of_docs/
"""

import sys
import html
from pathlib import Path

from paddleocr import PPStructureV3

SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_report(doc_name: str, pages: list[dict]) -> str:
    """pages: list of {'page_index': int, 'tables': [pred_html, ...]}"""
    sections = []
    total_tables = 0

    for page in pages:
        idx = page["page_index"]
        tables = page["tables"]
        if not tables:
            sections.append(f'<h2>Page {idx + 1}</h2><p class="muted">No tables detected.</p>')
            continue

        for t_i, table_html in enumerate(tables, start=1):
            total_tables += 1
            sections.append(
                f'<h2>Page {idx + 1} — Table {t_i}</h2>\n'
                f'<div class="table-wrap">{table_html}</div>'
            )

    body = "\n".join(sections) if sections else "<p>No pages processed.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extracted Tables - {html.escape(doc_name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
  h1 {{ font-size: 1.3rem; }}
  h2 {{ font-size: 1rem; margin-top: 2rem; color: #444; }}
  .muted {{ color: #888; font-style: italic; }}
  .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #ddd; padding: 0.5rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  table, th, td {{ border: 1px solid #ccc; }}
  td, th {{ padding: 6px 10px; font-size: 0.9rem; text-align: left; }}
</style>
</head>
<body>
  <h1>Extracted Tables: {html.escape(doc_name)}</h1>
  <p>{total_tables} table(s) found across {len(pages)} page(s).</p>
  {body}
</body>
</html>"""


def process_file(pipeline: PPStructureV3, file_path: Path, output_dir: Path):
    print(f"Processing {file_path.name} ...")
    output = pipeline.predict(str(file_path))

    pages = []
    for page_index, res in enumerate(output):
        data = res.json["res"]

        # Keep the raw json/markdown too -- useful for debugging or reuse
        res.save_to_json(save_path=str(output_dir / "raw"))
        res.save_to_markdown(save_path=str(output_dir / "raw"))

        table_htmls = [t["pred_html"] for t in data.get("table_res_list", []) if t.get("pred_html")]
        pages.append({"page_index": page_index, "tables": table_htmls})

    report_html = build_report(file_path.name, pages)
    out_path = output_dir / f"{file_path.stem}.html"
    out_path.write_text(report_html, encoding="utf-8")
    print(f"  -> {out_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python extract_tables.py <file_or_folder>")
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

    # use_doc_orientation_classify / use_doc_unwarping help with skewed scans;
    # turn them off if you know inputs are clean, flat scans/typed PDFs for speed.
    pipeline = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )

    for f in files:
        process_file(pipeline, f, output_dir)


if __name__ == "__main__":
    main()