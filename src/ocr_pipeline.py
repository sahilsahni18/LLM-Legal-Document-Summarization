# src/ocr_pipeline.py
import cv2
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from typing import List
import os

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    # Deskew
    coords = np.column_stack(np.where(denoised > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = denoised.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(denoised, M, (w, h),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def ocr_pdf(pdf_path: str, dpi: int = 300) -> str:
    """Convert each page of PDF to text via OCR."""
    pages = convert_from_path(pdf_path, dpi)
    full_text = []
    for i, page in enumerate(pages):
        img = np.array(page)
        proc = preprocess_image(img)
        text = pytesseract.image_to_string(proc, lang='eng')
        full_text.append(text)
    return "\n\n".join(full_text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Path to scanned PDF")
    args = parser.parse_args()
    out = ocr_pdf(args.pdf)
    print(out)
