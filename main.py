# main.py
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en")
result = ocr.predict("image.png")
for res in result:
    res.print()
    res.save_to_img("output")