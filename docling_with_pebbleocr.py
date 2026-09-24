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

import pymupdf
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling_pp_ocrv6 import PPOCRv6Options

# A page with fewer extractable characters than this is treated as a scan
# with no usable text layer -- same threshold used in document_extract_hybrid.py.
MIN_TEXT_CHARS = 30


def report_scanned_pages(pdf_path: Path):
    """
    Docling doesn't expose a public 'this page was OCR'd' flag on its output,
    so we check the same underlying signal ourselves (native text presence)
    and print it before conversion -- lets you see up front which pages will
    rely on OCR text (lower accuracy) vs native text (perfect accuracy).
    """
    doc = pymupdf.open(pdf_path)
    for page_index, page in enumerate(doc):
        text_layer = page.get_text().strip()
        if len(text_layer) >= MIN_TEXT_CHARS:
            print(f"  Page {page_index + 1}: typed -- native text layer found")
        else:
            print(f"  Page {page_index + 1}: SCANNED -- no/little text layer, Docling will OCR this page")
    doc.close()


def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()

    # Leave OCR enabled -- Docling auto-detects per page whether a real text
    # layer exists and only falls back to OCR where needed, so this is safe
    # for a mix of typed and scanned documents.
    pipeline_options.do_ocr = True

    # Swap the default OCR engine (EasyOCR) for PaddleOCR's own PP-OCRv6
    # detection + recognition models, run via a third-party Docling plugin
    # (docling-pp-ocrv6). This is a different runtime (ONNX/RapidOCR) but
    # the same trained weights as PaddleOCR -- so text accuracy here should
    # reflect PaddleOCR's recognition quality, not EasyOCR's.
    pipeline_options.allow_external_plugins = True  # required for any third-party OCR plugin
    pipeline_options.ocr_options = PPOCRv6Options(lang=["en"])

    # TableFormer has two modes: FAST (default) and ACCURATE. FAST is already
    # what you want for most documents -- ACCURATE is meaningfully slower and
    # only worth it for genuinely hard/ambiguous table layouts.
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

    # Use a GPU if you have one -- the layout + TableFormer models are the
    # slow part, and both benefit heavily from CUDA. AUTO will pick CUDA/MPS
    # automatically if available, otherwise falls back to CPU.
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.AUTO,
        num_threads=8,  # raise this toward your CPU's core count if running on CPU
    )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),}
    )


SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def process_file(converter: DocumentConverter, input_path: Path, output_dir: Path):
    print(f"Converting {input_path.name} ...")
    if input_path.suffix.lower() == ".pdf":
        report_scanned_pages(input_path)

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


def main():
    if len(sys.argv) != 2:
        print("Usage: uv run python document_extract_docling.py <file_or_folder>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        files = [p for p in sorted(input_path.iterdir()) if p.suffix.lower() in SUPPORTED_EXT]
        if not files:
            print(f"No supported files found in {input_path} (.pdf, .png, .jpg, .jpeg, .bmp, .tif, .tiff)")
            sys.exit(1)
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"Not found (or not a file/folder): {input_path}")
        sys.exit(1)

    converter = build_converter()  # built once, reused across every file
    for f in files:
        process_file(converter, f, output_dir)


if __name__ == "__main__":
    main()