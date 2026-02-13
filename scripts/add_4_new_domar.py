#!/usr/bin/env python3
"""
Phase A1 (low-risk): Copy 4 court decisions from Ansökningar and extract text to TXT.

- Copies 4 PDFs to Data/Domar/data/
- Extracts text with PyMuPDF and saves as TXT in court_decisions/
- Does NOT modify cleaned_court_texts.json or run 02.

After running this, run: python scripts/02_clean_court_texts.py
to regenerate cleaned_court_texts.json with 50 decisions (46 + 4 new).
"""

import re
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ANSOKAN_DIR = BASE_DIR / "Data" / "Ansökningar"
DOMAR_DATA = BASE_DIR / "Data" / "Domar" / "data"
COURT_DECISIONS_DIR = DOMAR_DATA / "processed" / "court_decisions"

# Case numbers of the 4 misfiled domar (from COMPLETE_DATA_INVENTORY)
NEW_DOMAR_CASES = ["M 483-22", "M 2694-22", "M 2695-22", "M 2796-24"]


def normalize_case(s: str) -> str:
    """Match 'M 483-22' or 'M483-22' etc."""
    m = re.search(r"[Mm]\s*(\d+)\s*-\s*(\d+)", s)
    return f"M {m.group(1)}-{m.group(2)}" if m else ""


def find_domar_pdfs() -> list[Path]:
    """Find PDFs in Ansökningar that are Dom and match our case numbers."""
    if not ANSOKAN_DIR.exists():
        return []
    found = []
    for pdf in ANSOKAN_DIR.glob("*.pdf"):
        name = pdf.name
        if "dom" not in name.lower() or "dagboksblad" in name.lower():
            continue
        case = normalize_case(name)
        if case in NEW_DOMAR_CASES:
            found.append(pdf)
    return sorted(found, key=lambda p: p.name)


def extract_text_pymupdf(pdf_path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text or ""
    except ImportError:
        raise SystemExit("PyMuPDF required: pip install pymupdf")
    except Exception as e:
        raise RuntimeError(f"Extract failed for {pdf_path.name}: {e}") from e


def main():
    print("Phase A1 (low-risk): Add 4 new court decisions from Ansökningar")
    print("=" * 60)

    if not ANSOKAN_DIR.exists():
        print(f"ERROR: {ANSOKAN_DIR} not found.")
        return

    pdfs = find_domar_pdfs()
    if len(pdfs) != 4:
        print(f"Found {len(pdfs)} of 4 expected PDFs in Ansökningar.")
        for p in pdfs:
            print(f"  - {p.name}")
        if len(pdfs) == 0:
            return

    DOMAR_DATA.mkdir(parents=True, exist_ok=True)
    COURT_DECISIONS_DIR.mkdir(parents=True, exist_ok=True)

    for pdf in pdfs:
        # Copy to Domar/data/
        dst_pdf = DOMAR_DATA / pdf.name
        shutil.copy2(pdf, dst_pdf)
        print(f"Copied: {pdf.name} -> {dst_pdf}")

        # Extract text to TXT
        text = extract_text_pymupdf(pdf)
        txt_name = pdf.stem + ".txt"
        txt_path = COURT_DECISIONS_DIR / txt_name
        txt_path.write_text(text, encoding="utf-8")
        n_words = len(text.split())
        print(f"  -> {txt_name} ({n_words:,} words)")

    print()
    print("Next step: run 02 to add the 4 to cleaned_court_texts.json:")
    print("  python scripts/02_clean_court_texts.py")
    print("Then add labels for the 4 new decisions to Data/processed/label_overrides.json")
    print("and run: python scripts/03_create_labeled_dataset.py")


if __name__ == "__main__":
    main()
