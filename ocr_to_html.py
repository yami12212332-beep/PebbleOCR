"""
Convert a PaddleOCR result into a single, self-contained HTML report:
the source image with bounding boxes drawn over the detected text,
plus a table listing every recognized line and its confidence score.

Usage:
    from paddleocr import PaddleOCR
    from ocr_to_html import result_to_html

    ocr = PaddleOCR(lang="en", enable_mkldnn=False)
    results = ocr.predict("your_image.jpg")

    for res in results:
        result_to_html(res, save_dir="output")
"""

import base64
import html
from pathlib import Path


def result_to_html(res, save_dir="output"):
    """
    res: one item yielded by PaddleOCR().predict(...)
    save_dir: folder to write the .html report into
    """
    data = res.json["res"]  # underlying dict: input_path, rec_texts, rec_polys, rec_scores, ...

    img_path = Path(data["input_path"])
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{img_path.stem}.html"

    # Embed the image directly in the HTML (no external file dependency)
    img_bytes = img_path.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode()
    ext = img_path.suffix.lstrip(".").lower() or "png"
    if ext == "jpg":
        ext = "jpeg"

    # Get image dimensions to convert box coords -> percentages (keeps overlay
    # aligned regardless of how the browser scales the <img>)
    from PIL import Image
    with Image.open(img_path) as im:
        img_w, img_h = im.size

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    polys = data.get("rec_polys", data.get("dt_polys", []))

    overlay_divs = []
    table_rows = []
    for i, (poly, text, score) in enumerate(zip(polys, texts, scores), start=1):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        left, top = min(xs), min(ys)
        w, h = max(xs) - left, max(ys) - top

        safe_text = html.escape(str(text))
        overlay_divs.append(
            f'<div class="box" style="left:{left / img_w * 100:.3f}%; '
            f'top:{top / img_h * 100:.3f}%; width:{w / img_w * 100:.3f}%; '
            f'height:{h / img_h * 100:.3f}%;" title="{safe_text} ({score:.2f})"></div>'
        )
        table_rows.append(
            f"<tr><td>{i}</td><td>{safe_text}</td><td>{score:.3f}</td></tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OCR Result - {html.escape(img_path.name)}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
  h1 {{ font-size: 1.2rem; }}
  .wrap {{ position: relative; display: inline-block; max-width: 100%; }}
  .wrap img {{ display: block; max-width: 100%; height: auto; border: 1px solid #ddd; }}
  .box {{
    position: absolute;
    border: 2px solid #e53935;
    background: rgba(229, 57, 53, 0.08);
    box-sizing: border-box;
  }}
  table {{ border-collapse: collapse; margin-top: 1.5rem; width: 100%; max-width: 700px; background: #fff; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 12px; text-align: left; font-size: 0.9rem; }}
  th {{ background: #f0f0f0; }}
</style>
</head>
<body>
  <h1>OCR Result: {html.escape(img_path.name)}</h1>
  <div class="wrap">
    <img src="data:image/{ext};base64,{img_b64}" alt="{html.escape(img_path.name)}">
    {''.join(overlay_divs)}
  </div>
  <h2>Extracted text ({len(texts)} lines)</h2>
  <table>
    <tr><th>#</th><th>Text</th><th>Confidence</th></tr>
    {''.join(table_rows)}
  </table>
</body>
</html>"""

    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Saved HTML report to {out_path}")
    return out_path


if __name__ == "__main__":
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en", enable_mkldnn=False)
    results = ocr.predict("image.png")

    for res in results:
        result_to_html(res, save_dir="output")