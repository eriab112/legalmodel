"""
02_clean_court_texts.py - Text Preprocessing Pipeline for NAP LegalBERT

Cleans 46 unique court decision texts:
1. Removes OCR artifacts (page headers, footers, addresses, page numbers)
2. Segments into sections (Domslut, Domskäl, Yrkanden, etc.)
3. Extracts key sections for BERT input (Domslut + Domskäl)
4. Normalizes Swedish characters and whitespace
5. Extracts metadata (case number, date, court, parties)
6. Saves as Data/processed/cleaned_court_texts.json

Usage:
    python scripts/02_clean_court_texts.py
"""

import json
import os
import re
import hashlib
from pathlib import Path
from collections import OrderedDict

# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("Data/Domar/data/processed/court_decisions")
OUTPUT_FILE = Path("Data/processed/cleaned_court_texts.json")

# Files to SKIP (duplicates and non-decisions identified in exploration)
SKIP_FILES = {
    # Duplicates (keep the more descriptive name)
    "dom1.txt",               # == NAP_M9349-24_2025-10-02.txt
    "dom2.txt",               # == MOD_M10196-24_2025-09-24.txt
    "MOD_M10258-23_2024-12-19.txt",  # == MOD_M-10258-23_2024-12-19.txt (keep hyphenated)
    "NAP_M16477-23_2025-03-10_v2.txt",  # v2 of NAP_M16477-23
    "NAP_M3426-24_2025-05-23_Karsbols_v2.txt",
    "NAP_M3426-24_2025-05-23_Karsbols_v3.txt",
    # Non-decisions
    "SvK_NAP_Slutrapport_2023.txt",
    "VM_Riktlinjer_Vattenkraft.txt",
}

# ============================================================
# OCR Artifact Removal
# ============================================================

def remove_ocr_artifacts(text: str) -> str:
    """Remove OCR artifacts: page headers, footers, addresses, page numbers."""

    # Remove page break markers: "--- Sida X ---"
    text = re.sub(r'---\s*Sida\s+\d+\s*---', '', text)

    # Remove page number lines: "Sid X (Y)" or "Sid X" at start of lines
    text = re.sub(r'^Sid\s+\d+\s*(\(\d+\))?\s*$', '', text, flags=re.MULTILINE)

    # Remove repeated court headers (Svea hovrätt pattern)
    text = re.sub(
        r'^SVEA HOVRÄTT\s*(DOM|PROTOKOLL)?\s*(M\s*\d+[\s-]*\d+)?\s*$',
        '', text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(
        r'^Mark\s*-?\s*och\s+miljö(över)?domstolen\s*$',
        '', text, flags=re.MULTILINE
    )

    # Remove tingsrätt repeated headers
    text = re.sub(
        r'^(VÄXJÖ|VÄNERSBORGS|ÖSTERSUNDS|NACKA|UMEÅ)\s+TINGSRÄTT\s+(DOM|BESLUT)\s+M\s*[\d-]+\s*$',
        '', text, flags=re.MULTILINE
    )
    text = re.sub(
        r'^Mark-\s*och\s+miljödomstolen\s*$',
        '', text, flags=re.MULTILINE
    )

    # Remove court internal codes (060303, 060404, etc.)
    text = re.sub(r'^0[56]\d{4}\s*$', '', text, flags=re.MULTILINE)

    # Remove Dok.Id lines
    text = re.sub(r'^Dok\.?Id\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove address blocks (Postadress/Besöksadress/Box/telefon)
    text = re.sub(
        r'Postadress\s+Besöksadress\s+Telefon.*?(?=\n[A-ZÅÄÖ]|\n\n)',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'^Box\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d{3}\s+\d{2}\s+\w+\s+Birger\s+Jarls.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^08-561\s+\d{3}\s+\d{2}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^måndag\s*–\s*fredag\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^0[89]:\d{2}\s*–?\s*\d{2}:\d{2}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^E-post:.*@dom\.se\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www\.\w+\.se\s*$', '', text, flags=re.MULTILINE)

    # Remove Dok.Id inline
    text = re.sub(r'Dok\.?Id\s*\d+', '', text)

    # Remove tingsrätt address blocks
    text = re.sub(
        r'Postadress\s+Besöksadress\s+Telefon\s+Telefax\s+Expeditionstid.*?(?=\n[A-ZÅÄÖ]|\n\n)',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'^Box\s+\d{2,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d{3}\s+\d{2}\s+\w+\s+E-post:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^www\.domstol\.se/.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d{4}-\d{3}\s+\d{3}\s*$', '', text, flags=re.MULTILINE)

    # Remove separator lines
    text = re.sub(r'^[_]{3,}\s*$', '', text, flags=re.MULTILINE)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: fix line breaks within words, collapse blank lines."""
    # Fix line-break artifacts within entity names (e.g. "Marbäcks\nkraftverk")
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # hyphenation

    # Collapse multiple blank lines to single
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove trailing whitespace per line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)

    # Remove leading blank lines
    text = text.strip()

    return text


# ============================================================
# Metadata Extraction
# ============================================================

def extract_metadata(text: str, filename: str) -> dict:
    """Extract case metadata from text and filename."""
    meta = {
        "filename": filename,
        "case_number": None,
        "date": None,
        "court": None,
        "court_level": None,
        "subject": None,
        "parties": {"klagande": [], "motpart": []},
        "word_count": len(text.split()),
    }

    # Case number from text: "Mål nr M XXXX-XX" or "M XXXX-XX"
    m = re.search(r'[Mm]ål\s*(?:nr)?\s*(M\s*[\d]+[\s-]+\d+)', text)
    if m:
        meta["case_number"] = re.sub(r'\s+', ' ', m.group(1)).strip()
    else:
        # Try from filename
        m = re.search(r'[Mm][-_]?(\d+)[-_](\d+)', filename)
        if m:
            meta["case_number"] = f"M {m.group(1)}-{m.group(2)}"

    # Date from text
    m = re.search(r'(20\d{2})\s*-\s*(\d{2})\s*-\s*(\d{2})', text[:500])
    if m:
        meta["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        # From filename
        m = re.search(r'(20\d{2})[-_](\d{2})[-_](\d{2})', filename)
        if m:
            meta["date"] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Court detection
    text_upper = text[:2000].upper()
    if "SVEA HOVRÄTT" in text_upper or "MILJÖÖVERDOMSTOLEN" in text_upper:
        meta["court"] = "Mark- och miljööverdomstolen (MÖD)"
        meta["court_level"] = "appeal"
    elif "VÄXJÖ TINGSRÄTT" in text_upper:
        meta["court"] = "Växjö tingsrätt, MMD"
        meta["court_level"] = "first_instance"
    elif "VÄNERSBORGS TINGSRÄTT" in text_upper:
        meta["court"] = "Vänersborgs tingsrätt, MMD"
        meta["court_level"] = "first_instance"
    elif "ÖSTERSUNDS TINGSRÄTT" in text_upper:
        meta["court"] = "Östersunds tingsrätt, MMD"
        meta["court_level"] = "first_instance"
    elif "NACKA TINGSRÄTT" in text_upper:
        meta["court"] = "Nacka tingsrätt, MMD"
        meta["court_level"] = "first_instance"
    elif "UMEÅ TINGSRÄTT" in text_upper:
        meta["court"] = "Umeå tingsrätt, MMD"
        meta["court_level"] = "first_instance"
    else:
        meta["court"] = "Okänd"
        meta["court_level"] = "unknown"

    # Subject (SAKEN section - usually one-two lines)
    m = re.search(r'SAKEN\s*\n(.+?)(?:\n_|(?:\n[A-ZÅÄÖ]{2,}))', text, re.DOTALL)
    if m:
        subject = m.group(1).strip()
        subject = re.sub(r'\s+', ' ', subject)
        meta["subject"] = subject[:300]

    # Document type from filename
    if "slutligt-beslut" in filename.lower():
        meta["doc_type"] = "beslut"
    elif "dom" in filename.lower() or "DOM" in text[:200]:
        meta["doc_type"] = "dom"
    elif "PROTOKOLL" in text[:500]:
        meta["doc_type"] = "protokoll"
    else:
        meta["doc_type"] = "dom"

    return meta


# ============================================================
# Section Segmentation
# ============================================================

# Section header patterns (ordered by typical appearance in documents)
SECTION_PATTERNS = [
    ("överklagat_avgörande", r'ÖVERKLAGAT\s+AVGÖRANDE'),
    ("parter", r'PARTER'),
    ("saken", r'SAKEN'),
    ("domslut", r'(?:MARK-\s*OCH\s+MILJÖ(?:ÖVER)?DOMSTOLENS\s+)?DOMSLUT'),
    ("yrkanden", r'YRKANDE[NR]?\s*(?:I\s+MARK-\s*OCH\s+MILJÖ(?:ÖVER)?DOMSTOLEN|M\.?\s*M\.?)?\s*'),
    ("utveckling_av_talan", r'UTVECKLING\s+AV\s+TALAN'),
    ("bakgrund", r'BAKGRUND(?:\s+OCH\s+TIDIGARE\s+BESLUT)?'),
    ("grunder", r'GRUNDER'),
    ("domskäl", r'(?:MARK-\s*OCH\s+MILJÖ(?:ÖVER)?DOMSTOLENS\s+)?DOMSKÄL'),
    ("samlad_redovisning", r'SAMLAD\s+REDOVISNING'),
    ("övrigt", r'ÖVRIGT'),
]


def segment_sections(text: str) -> dict:
    """Split cleaned text into labeled sections."""
    sections = OrderedDict()

    # Find all section header positions
    found = []
    for section_name, pattern in SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            found.append((m.start(), m.end(), section_name))

    if not found:
        # No sections found - return full text as 'full'
        return {"full": text.strip()}

    # Sort by position
    found.sort(key=lambda x: x[0])

    # Extract text between section headers
    for i, (start, header_end, name) in enumerate(found):
        if i + 1 < len(found):
            next_start = found[i + 1][0]
            section_text = text[header_end:next_start].strip()
        else:
            section_text = text[header_end:].strip()

        # Handle duplicates (same section appearing multiple times)
        if name in sections:
            sections[name] += "\n\n" + section_text
        else:
            sections[name] = section_text

    return dict(sections)


# ============================================================
# Key Text Extraction (for BERT input)
# ============================================================

def extract_key_text(sections: dict, max_chars: int = 50000) -> str:
    """Extract key text for BERT: Domslut + Domskäl (most informative sections).

    Falls back to full text if key sections not found.
    Priority: domslut > domskäl > yrkanden > utveckling_av_talan > full text
    """
    parts = []

    # Primary: Domslut (verdict)
    if "domslut" in sections:
        parts.append(f"DOMSLUT:\n{sections['domslut']}")

    # Primary: Domskäl (reasoning)
    if "domskäl" in sections:
        parts.append(f"DOMSKÄL:\n{sections['domskäl']}")

    # If neither found, use yrkanden + utveckling as fallback
    if not parts:
        for key in ["yrkanden", "utveckling_av_talan", "grunder", "bakgrund"]:
            if key in sections:
                parts.append(f"{key.upper()}:\n{sections[key]}")

    # Final fallback: use full text
    if not parts:
        if "full" in sections:
            parts.append(sections["full"])
        else:
            parts.append("\n\n".join(sections.values()))

    key_text = "\n\n".join(parts)

    # Truncate if too long
    if len(key_text) > max_chars:
        key_text = key_text[:max_chars] + "\n[TRUNCATED]"

    return key_text


# ============================================================
# Cost Extraction (for labeling assistance)
# ============================================================

def extract_costs(text: str) -> list:
    """Extract monetary amounts from text for labeling assistance."""
    costs = []

    # Pattern: X kr, X SEK, X MSEK, X milj. kr, X miljoner kronor
    patterns = [
        (r'([\d\s,.]+)\s*(?:miljoner?\s+kronor|milj\.?\s*kr)', 'MSEK'),
        (r'([\d\s,.]+)\s*MSEK', 'MSEK'),
        (r'([\d\s,.]+)\s*(?:kr|kronor|SEK)(?!\w)', 'SEK'),
    ]

    for pattern, unit in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            amount_str = m.group(1).strip().replace(' ', '').replace(',', '.')
            try:
                amount = float(amount_str)
                if unit == 'MSEK':
                    amount *= 1_000_000
                if amount >= 1000:  # Skip trivially small amounts
                    costs.append({
                        "amount_sek": amount,
                        "original": m.group(0).strip(),
                        "context": text[max(0, m.start()-50):m.end()+50].strip()
                    })
            except ValueError:
                continue

    # Deduplicate by amount
    seen = set()
    unique_costs = []
    for c in costs:
        if c["amount_sek"] not in seen:
            seen.add(c["amount_sek"])
            unique_costs.append(c)

    return sorted(unique_costs, key=lambda x: x["amount_sek"], reverse=True)


# ============================================================
# Environmental Measures Extraction
# ============================================================

MEASURE_KEYWORDS = {
    "fiskväg": ["fiskväg", "fisktrappa", "fiskpassage"],
    "faunapassage": ["faunapassage", "fauna-passage", "faunpassage"],
    "fiskvandring": ["fiskvandring", "vandringsfisk", "vandringsbenägen"],
    "omlöp": ["omlöp", "omlopps"],
    "minimitappning": ["minimitappning", "minimivattenföring", "minimiflöde"],
    "utskov": ["utskov"],
    "kontrollprogram": ["kontrollprogram", "kontroll-program"],
    "biotopvård": ["biotopvård", "biotopåtgärd"],
    "skyddsgaller": ["skyddsgaller", "galler för avledning"],
    "utrivning": ["utrivning", "rivning av damm"],
}


def extract_measures(text: str) -> list:
    """Extract environmental measures mentioned in text."""
    text_lower = text.lower()
    found = []
    for measure, keywords in MEASURE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                found.append(measure)
                break
    return found


# ============================================================
# Main Processing Pipeline
# ============================================================

def process_decision(filepath: Path) -> dict:
    """Process a single court decision file."""
    text = filepath.read_text(encoding="utf-8")

    # Step 1: Extract metadata before cleaning
    metadata = extract_metadata(text, filepath.name)

    # Step 2: Remove OCR artifacts
    cleaned = remove_ocr_artifacts(text)

    # Step 3: Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    # Step 4: Segment into sections
    sections = segment_sections(cleaned)

    # Step 5: Extract key text for BERT
    key_text = extract_key_text(sections)

    # Step 6: Extract costs and measures
    costs = extract_costs(cleaned)
    measures = extract_measures(cleaned)

    # Generate stable ID from case number or filename
    case_id = metadata.get("case_number", "").replace(" ", "").lower()
    if not case_id:
        case_id = filepath.stem

    # Compute text hash for dedup verification
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]

    return {
        "id": case_id,
        "filename": filepath.name,
        "text_full": cleaned,
        "sections": sections,
        "key_text": key_text,
        "metadata": metadata,
        "extracted_costs": costs,
        "extracted_measures": measures,
        "text_hash": text_hash,
        "stats": {
            "original_chars": len(text),
            "cleaned_chars": len(cleaned),
            "key_text_chars": len(key_text),
            "num_sections": len(sections),
            "section_names": list(sections.keys()),
        }
    }


def main():
    print("=" * 60)
    print("NAP LegalBERT - Text Preprocessing Pipeline")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    # Get all .txt files
    all_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"\nFound {len(all_files)} text files in {INPUT_DIR}")

    # Filter out skipped files
    files_to_process = [f for f in all_files if f.name not in SKIP_FILES]
    print(f"After removing {len(all_files) - len(files_to_process)} duplicates/non-decisions: {len(files_to_process)} files")

    # Process each file
    decisions = []
    errors = []

    for i, filepath in enumerate(files_to_process, 1):
        try:
            print(f"  [{i:2d}/{len(files_to_process)}] Processing {filepath.name}...", end=" ")
            result = process_decision(filepath)
            decisions.append(result)

            # Summary
            n_sections = result["stats"]["num_sections"]
            has_domslut = "domslut" in result["sections"]
            has_domskal = "domskäl" in result["sections"]
            print(f"OK ({n_sections} sections, domslut={'Y' if has_domslut else 'N'}, domskäl={'Y' if has_domskal else 'N'})")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"file": filepath.name, "error": str(e)})

    # Summary statistics
    print(f"\n{'=' * 60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Processed:  {len(decisions)} decisions")
    print(f"  Errors:     {len(errors)}")

    # Section coverage
    has_domslut = sum(1 for d in decisions if "domslut" in d["sections"])
    has_domskal = sum(1 for d in decisions if "domskäl" in d["sections"])
    has_yrkanden = sum(1 for d in decisions if "yrkanden" in d["sections"])
    has_bakgrund = sum(1 for d in decisions if "bakgrund" in d["sections"])

    print(f"\n  Section coverage:")
    print(f"    Domslut:   {has_domslut}/{len(decisions)} ({100*has_domslut/len(decisions):.0f}%)")
    print(f"    Domskäl:   {has_domskal}/{len(decisions)} ({100*has_domskal/len(decisions):.0f}%)")
    print(f"    Yrkanden:  {has_yrkanden}/{len(decisions)} ({100*has_yrkanden/len(decisions):.0f}%)")
    print(f"    Bakgrund:  {has_bakgrund}/{len(decisions)} ({100*has_bakgrund/len(decisions):.0f}%)")

    # Cost extraction
    with_costs = sum(1 for d in decisions if d["extracted_costs"])
    total_costs = sum(len(d["extracted_costs"]) for d in decisions)
    print(f"\n  Cost extraction:")
    print(f"    Decisions with costs: {with_costs}/{len(decisions)}")
    print(f"    Total cost entries:   {total_costs}")

    # Measures
    all_measures = set()
    for d in decisions:
        all_measures.update(d["extracted_measures"])
    with_measures = sum(1 for d in decisions if d["extracted_measures"])
    print(f"\n  Measures extraction:")
    print(f"    Decisions with measures: {with_measures}/{len(decisions)}")
    print(f"    Unique measure types:    {len(all_measures)}")
    print(f"    Types found: {', '.join(sorted(all_measures))}")

    # Key text stats
    key_lengths = [d["stats"]["key_text_chars"] for d in decisions]
    print(f"\n  Key text (for BERT):")
    print(f"    Mean length:  {sum(key_lengths)/len(key_lengths):,.0f} chars")
    print(f"    Min length:   {min(key_lengths):,} chars")
    print(f"    Max length:   {max(key_lengths):,} chars")

    # Court distribution
    courts = {}
    for d in decisions:
        court = d["metadata"].get("court", "Okänd")
        courts[court] = courts.get(court, 0) + 1
    print(f"\n  Court distribution:")
    for court, count in sorted(courts.items(), key=lambda x: -x[1]):
        print(f"    {court}: {count}")

    # Save output
    output = {
        "pipeline_version": "2.0",
        "processing_date": "2026-02-06",
        "source_dir": str(INPUT_DIR),
        "total_decisions": len(decisions),
        "skipped_files": sorted(SKIP_FILES),
        "errors": errors,
        "decisions": decisions,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Output saved to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.1f} MB")

    if errors:
        print(f"\n  ERRORS:")
        for e in errors:
            print(f"    {e['file']}: {e['error']}")


if __name__ == "__main__":
    main()
