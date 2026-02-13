#!/usr/bin/env python3
"""
Extract text from a PDF to a .txt file using PyMuPDF.
Usage: python scripts/extract_pdf_text.py [path/to/file.pdf]
Output: same path with .txt extension (e.g. kmae250140.txt)
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def extract_pdf_to_txt(pdf_path: Path) -> Path:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise SystemExit("PyMuPDF required: pip install pymupdf")

    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    full_text = "\n".join(text_parts)

    out_path = pdf_path.with_suffix(".txt")
    out_path.write_text(full_text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "kmae250140.pdf"
    if not pdf.is_absolute():
        pdf = BASE_DIR / pdf
    if not pdf.exists():
        print(f"File not found: {pdf}")
        sys.exit(1)
    out = extract_pdf_to_txt(pdf)
    print(f"Extracted {len(out.read_text(encoding='utf-8').split())} words -> {out}")
