"""
Document extraction using Docling: combines native PDF text extraction
(accurate, no OCR needed for typed PDFs) with TableFormer, IBM Research's
trained table-structure model -- proper merged-cell / complex-header
handling that PyMuPDF's heuristic find_tables() doesn't give you.

Falls back to OCR automatically (via its built-in pipeline) for any page
that turns out to be a scan with no text layer.

Requires:
    uv add docling
    (first run downloads ~1.2GB of models to ~/.cache/docling/models/)

Usage:
    uv run python document_extract_docling.py path/to/file.pdf
    uv run python document_extract_docling.py path/to/image.png
"""

import sys
from pathlib import Path

from docling.document_converter import DocumentConverter


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python document_extract_docling.py <file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {input_path.name} ...")
    converter = DocumentConverter()
    result = converter.convert(str(input_path))

    # Whole-document HTML: text and tables together, in reading order,
    # with real merged-cell structure preserved.
    doc_html = result.document.export_to_html()
    out_path = output_dir / f"{input_path.stem}.html"
    out_path.write_text(doc_html, encoding="utf-8")
    print(f"-> {out_path}")

    # Also dump each detected table individually (CSV + standalone HTML) --
    # handy for downstream processing (e.g. loading straight into pandas).
    for i, table in enumerate(result.document.tables, start=1):
        df = table.export_to_dataframe(doc=result.document)
        csv_path = output_dir / f"{input_path.stem}_table{i}.csv"
        df.to_csv(csv_path, index=False)
        print(f"   table {i}: {df.shape[0]} rows x {df.shape[1]} cols -> {csv_path}")


if __name__ == "__main__":
    main()
