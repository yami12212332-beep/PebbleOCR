"""
Extract full document layout (text, titles, tables, headers, footers)
from images and PDFs (scanned or typed) and produce a single HTML
report per document.

Requires:
    uv add "paddleocr[doc-parser]==3.7.0"

Usage:
    uv run python extract_table_para.py path/to/file.pdf
    uv run python extract_table_para.py path/to/image.png
    uv run python extract_table_para.py path/to/folder_of_docs/
"""

import html
from pathlib import Path
import sys

from paddleocr import PPStructureV3


SUPPORTED_EXT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def extract_block_text(block) -> str:
    """Safely extract text content from a block."""

    if isinstance(block, str):
        return block.strip()

    if not isinstance(block, dict):
        return ""

    # Direct text field
    text = block.get("text")

    if isinstance(text, str):
        return text.strip()

    # Some PaddleOCR results store text inside "res"
    res_items = block.get("res", [])

    lines = []

    if isinstance(res_items, list):
        for item in res_items:

            if isinstance(item, dict):
                item_text = item.get("text")

                if isinstance(item_text, str):
                    lines.append(item_text)

            elif isinstance(item, (list, tuple)) and len(item) > 1:
                text_info = item[1]

                if isinstance(text_info, (list, tuple)):
                    if len(text_info) > 0:
                        lines.append(str(text_info[0]))

                elif isinstance(text_info, str):
                    lines.append(text_info)

    return "\n".join(lines).strip()


def render_block_to_html(block) -> str:
    """Convert a PaddleOCR layout block into semantic HTML."""

    # ---------------------------------------------------------
    # Plain string block
    # ---------------------------------------------------------

    if isinstance(block, str):
        text = block.strip()

        if not text:
            return ""

        escaped_text = html.escape(text).replace("\n", "<br>")

        return f"<p>{escaped_text}</p>"

    # Ignore anything that isn't a dictionary
    if not isinstance(block, dict):
        return ""

    block_type = str(
        block.get("type", "text")
    ).lower()

    # ---------------------------------------------------------
    # Tables
    # ---------------------------------------------------------

    if block_type == "table":

        table_html = block.get("pred_html", "")

        if isinstance(table_html, str) and table_html.strip():
            return (
                '<div class="table-wrap">'
                f"{table_html}"
                "</div>"
            )

        return ""

    # ---------------------------------------------------------
    # Extract text
    # ---------------------------------------------------------

    text = extract_block_text(block)

    if not text:
        return ""

    escaped_text = html.escape(text).replace(
        "\n",
        "<br>",
    )

    # ---------------------------------------------------------
    # Titles
    # ---------------------------------------------------------

    if block_type in (
        "title",
        "header_title",
    ):
        return f"<h2>{escaped_text}</h2>"

    # ---------------------------------------------------------
    # Headers
    # ---------------------------------------------------------

    elif block_type in (
        "header",
        "user_header",
    ):
        return (
            '<div class="doc-header">'
            f"{escaped_text}"
            "</div>"
        )

    # ---------------------------------------------------------
    # Footers
    # ---------------------------------------------------------

    elif block_type in (
        "footer",
        "page_number",
    ):
        return (
            '<div class="doc-footer">'
            f"{escaped_text}"
            "</div>"
        )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    elif block_type == "figure":
        return (
            '<div class="doc-figure">'
            "<em>"
            f"[Figure/Chart]: {escaped_text}"
            "</em>"
            "</div>"
        )

    # ---------------------------------------------------------
    # References
    # ---------------------------------------------------------

    elif block_type == "reference":
        return (
            '<blockquote class="doc-reference">'
            f"{escaped_text}"
            "</blockquote>"
        )

    # ---------------------------------------------------------
    # Normal text
    # ---------------------------------------------------------

    else:
        return f"<p>{escaped_text}</p>"


def build_report(
    doc_name: str,
    pages: list[dict],
) -> str:
    """Build the complete HTML report."""

    sections = []

    for page in pages:

        idx = page["page_index"]
        blocks = page["blocks"]

        page_content = []

        for block in blocks:

            rendered_html = render_block_to_html(block)

            if rendered_html:
                page_content.append(rendered_html)

        # -----------------------------------------------------
        # Empty page
        # -----------------------------------------------------

        if not page_content:

            sections.append(
                '<section class="page">'
                f'<div class="page-badge">Page {idx + 1}</div>'
                '<p class="muted">'
                "No readable content detected."
                "</p>"
                "</section>"
            )

        # -----------------------------------------------------
        # Page with content
        # -----------------------------------------------------

        else:

            joined_content = "\n".join(
                page_content
            )

            sections.append(
                '<section class="page">\n'
                f'<div class="page-badge">Page {idx + 1}</div>\n'
                f"{joined_content}\n"
                "</section>"
            )

    # ---------------------------------------------------------
    # Body
    # ---------------------------------------------------------

    if sections:
        body = "\n".join(sections)
    else:
        body = "<p>No pages processed.</p>"

    escaped_doc_name = html.escape(doc_name)

    # ---------------------------------------------------------
    # Complete HTML document
    # ---------------------------------------------------------

    return f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="utf-8">

<title>
Document OCR - {escaped_doc_name}
</title>

<style>

body {{
    font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    margin:
        2rem auto;

    max-width:
        900px;

    background:
        #f4f5f7;

    color:
        #222;

    line-height:
        1.6;
}}

h1 {{
    font-size:
        1.5rem;

    color:
        #111;

    border-bottom:
        2px solid #ddd;

    padding-bottom:
        0.5rem;

    margin-bottom:
        1.5rem;
}}

h2 {{
    font-size:
        1.25rem;

    margin-top:
        1.5rem;

    color:
        #1f2937;
}}

.page {{
    background:
        #fff;

    border:
        1px solid #e5e7eb;

    border-radius:
        8px;

    padding:
        2.5rem;

    margin-bottom:
        2rem;

    box-shadow:
        0 1px 3px rgba(0, 0, 0, 0.05);

    position:
        relative;
}}

.page-badge {{
    position:
        absolute;

    top:
        1rem;

    right:
        1rem;

    background:
        #f3f4f6;

    color:
        #4b5563;

    font-size:
        0.75rem;

    padding:
        3px 10px;

    border-radius:
        9999px;

    font-weight:
        600;

    text-transform:
        uppercase;
}}

.muted {{
    color:
        #6b7280;

    font-style:
        italic;
}}

.table-wrap {{
    overflow-x:
        auto;

    background:
        #fff;

    border:
        1px solid #d1d5db;

    margin:
        1.25rem 0;

    padding:
        0.5rem;

    border-radius:
        6px;
}}

table {{
    border-collapse:
        collapse;

    width:
        100%;
}}

table,
th,
td {{
    border:
        1px solid #d1d5db;
}}

td,
th {{
    padding:
        8px 12px;

    font-size:
        0.9rem;

    text-align:
        left;
}}

th {{
    background-color:
        #f9fafb;

    font-weight:
        600;
}}

.doc-header,
.doc-footer {{
    font-size:
        0.85rem;

    color:
        #6b7280;

    font-style:
        italic;

    margin:
        0.5rem 0;
}}

.doc-figure {{
    background:
        #f9fafb;

    border:
        1px dashed #d1d5db;

    padding:
        0.75rem;

    font-size:
        0.85rem;

    margin:
        1rem 0;

    color:
        #4b5563;

    border-radius:
        4px;
}}

.doc-reference {{
    border-left:
        3px solid #3b82f6;

    margin:
        1rem 0;

    padding-left:
        1rem;

    color:
        #4b5563;
}}

p {{
    margin:
        0.75rem 0;

    color:
        #374151;
}}

</style>

</head>

<body>

<h1>
Document OCR:
{escaped_doc_name}
</h1>

{body}

</body>

</html>
"""


def extract_blocks_from_result(data) -> list:
    """
    Extract layout/table blocks from a PaddleOCR result.

    PaddleOCR versions can return slightly different structures,
    so this function handles several common formats.
    """

    blocks = []

    # ---------------------------------------------------------
    # Result is already a list
    # ---------------------------------------------------------

    if isinstance(data, list):
        return data

    # ---------------------------------------------------------
    # Result is a dictionary
    # ---------------------------------------------------------

    if not isinstance(data, dict):
        return blocks

    # ---------------------------------------------------------
    # Layout detection results
    # ---------------------------------------------------------

    layout_data = data.get(
        "layout_det_res"
    )

    if isinstance(layout_data, list):
        blocks.extend(layout_data)

    # ---------------------------------------------------------
    # Table results
    # ---------------------------------------------------------

    table_data = data.get(
        "table_res_list"
    )

    if isinstance(table_data, list):
        blocks.extend(table_data)

    # ---------------------------------------------------------
    # Other possible list-based results
    # ---------------------------------------------------------

    if not blocks:

        for value in data.values():

            if isinstance(value, list):
                blocks.extend(value)

    return blocks


def process_file(
    pipeline: PPStructureV3,
    file_path: Path,
    output_dir: Path,
):
    """Process one document."""

    print(
        f"Processing {file_path.name} ..."
    )

    try:

        output = pipeline.predict(
            str(file_path)
        )

    except Exception as exc:

        print(
            f"ERROR processing "
            f"{file_path.name}: {exc}"
        )

        return

    pages = []

    # ---------------------------------------------------------
    # Process each page/result
    # ---------------------------------------------------------

    for page_index, res in enumerate(output):

        try:

            # PaddleOCR 3.x result object
            json_data = res.json

            if callable(json_data):
                json_data = json_data()

            # -------------------------------------------------
            # Extract "res"
            # -------------------------------------------------

            if isinstance(json_data, dict):

                data = json_data.get(
                    "res",
                    json_data,
                )

            else:

                data = json_data

            # -------------------------------------------------
            # Extract blocks
            # -------------------------------------------------

            blocks = extract_blocks_from_result(
                data
            )

            # -------------------------------------------------
            # Save raw JSON/Markdown output
            # -------------------------------------------------

            raw_dir = output_dir / "raw"

            raw_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:

                res.save_to_json(
                    save_path=str(raw_dir)
                )

            except Exception as exc:

                print(
                    "  Warning: "
                    f"could not save JSON: {exc}"
                )

            try:

                res.save_to_markdown(
                    save_path=str(raw_dir)
                )

            except Exception as exc:

                print(
                    "  Warning: "
                    f"could not save Markdown: {exc}"
                )

            # -------------------------------------------------
            # Store page
            # -------------------------------------------------

            pages.append(
                {
                    "page_index": page_index,
                    "blocks": blocks,
                }
            )

        except Exception as exc:

            print(
                f"  Warning: "
                f"could not process page "
                f"{page_index + 1}: {exc}"
            )

            pages.append(
                {
                    "page_index": page_index,
                    "blocks": [],
                }
            )

    # ---------------------------------------------------------
    # Generate HTML
    # ---------------------------------------------------------

    report_html = build_report(
        file_path.name,
        pages,
    )

    out_path = (
        output_dir /
        f"{file_path.stem}.html"
    )

    out_path.write_text(
        report_html,
        encoding="utf-8",
    )

    print(
        f"  -> {out_path}"
    )


def main():
    """Command-line entry point."""

    # ---------------------------------------------------------
    # Validate arguments
    # ---------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "Usage: "
            "uv run python "
            "extract_table_para.py "
            "<file_or_folder>"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    output_dir = Path(
        "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Find input files
    # ---------------------------------------------------------

    if input_path.is_dir():

        files = [
            p
            for p in sorted(
                input_path.iterdir()
            )
            if (
                p.is_file()
                and p.suffix.lower()
                in SUPPORTED_EXT
            )
        ]

    elif input_path.is_file():

        if (
            input_path.suffix.lower()
            not in SUPPORTED_EXT
        ):

            print(
                "Unsupported file type: "
                f"{input_path.suffix}"
            )

            sys.exit(1)

        files = [input_path]

    else:

        print(
            f"Not found: {input_path}"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # No files
    # ---------------------------------------------------------

    if not files:

        print(
            "No supported files found "
            "(.pdf, .png, .jpg, .jpeg, "
            ".bmp, .tif, .tiff)"
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # Create PaddleOCR pipeline
    # ---------------------------------------------------------

    print(
        "Initializing PaddleOCR "
        "PPStructureV3..."
    )

    pipeline = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
    )

    print(
        f"Found {len(files)} document(s)."
    )

    # ---------------------------------------------------------
    # Process documents
    # ---------------------------------------------------------

    for file_path in files:

        process_file(
            pipeline,
            file_path,
            output_dir,
        )

    print()
    print(
        "Processing complete."
    )

    print(
        f"Output directory: "
        f"{output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()