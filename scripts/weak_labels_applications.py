#!/usr/bin/env python3
"""
Phase A4 (low-risk): Create weak labels for applications only.

Reads: Data/processed/ansokan_texts.json
Writes: Data/processed/weakly_labeled_applications.json (NEW only)

- Skips documents that are Dom (court decisions) - inferred from filename.
- Only genuine Ansökningar get weak labels.
- Confidence capped 0.2-0.4; clearly marked as proposal, not verdict.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ANSOKAN_TEXTS_PATH = BASE_DIR / "Data" / "processed" / "ansokan_texts.json"
OUTPUT_PATH = BASE_DIR / "Data" / "processed" / "weakly_labeled_applications.json"


def document_type_from_filename(filename: str) -> str:
    """Infer document type from filename (matches extract_all_pdfs categorize_ansokan)."""
    fn = filename.lower()
    if "dom " in fn or "dom_" in fn:
        return "Dom"
    if "ansökan" in fn or "aktbil 1" in fn:
        return "Ansökan"
    if "beslut" in fn:
        return "Beslut"
    if "dagboksblad" in fn:
        return "Dagboksblad"
    if "konsoliderad" in fn or "kompletterad" in fn:
        return "Komplettering"
    return "Okänd"


def weak_label_application(ansokan_text: str) -> dict:
    """
    Create WEAK labels for applications based on proposal severity.
    These are NOT court outcomes; confidence intentionally low (0.2-0.4).
    """
    text_lower = (ansokan_text or "").lower()
    confidence = 0.2
    total_cost = 0.0

    cost_matches = re.findall(r"(\d+(?:[,.]\d+)?)\s*(?:msek|miljoner)", text_lower)
    if cost_matches:
        total_cost = sum(float(c.replace(",", ".")) for c in cost_matches)
        if total_cost > 15:
            confidence += 0.1
        elif total_cost > 10:
            confidence += 0.05

    high_indicators = 0.0
    if total_cost > 15:
        high_indicators += 2
    elif total_cost > 10:
        high_indicators += 1
    if "fiskväg" in text_lower:
        high_indicators += 1
    if "ledskena" in text_lower or "guidance" in text_lower:
        high_indicators += 1
    if "minimitappning" in text_lower:
        high_indicators += 0.5

    low_indicators = 0
    if total_cost < 5 and total_cost > 0:
        low_indicators += 2
        confidence += 0.1
    timeline_m = re.search(r"(\d+)\s*år", text_lower)
    if timeline_m:
        try:
            years = int(timeline_m.group(1))
            if years > 5:
                low_indicators += 1
        except ValueError:
            pass

    if high_indicators >= 3:
        weak_label = "HIGH_RISK"
        confidence = min(0.4, confidence)
    elif low_indicators >= 2:
        weak_label = "LOW_RISK"
        confidence = min(0.35, confidence)
    else:
        weak_label = "MEDIUM_RISK"
        confidence = min(0.3, confidence)

    return {
        "weak_label": weak_label,
        "confidence": round(confidence, 2),
        "source": "application_weak_supervision",
        "high_indicators": high_indicators,
        "low_indicators": low_indicators,
        "warning": "WEAK LABEL - Proposal, not verdict!",
    }


def main():
    if not ANSOKAN_TEXTS_PATH.exists():
        print(f"Missing {ANSOKAN_TEXTS_PATH}. Run scripts/extract_all_pdfs.py first.")
        return

    with open(ANSOKAN_TEXTS_PATH, "r", encoding="utf-8") as f:
        ansokan_list = json.load(f)

    if not isinstance(ansokan_list, list):
        print("Expected ansokan_texts.json to be a list.")
        return

    weakly_labeled = []
    for item in ansokan_list:
        filename = item.get("filename", "")
        text = item.get("text", "")
        word_count = item.get("word_count", 0) or len((text or "").split())

        doc_type = document_type_from_filename(filename)
        if doc_type == "Dom":
            continue
        if doc_type != "Ansökan":
            continue
        if not text or len(text.strip()) < 100:
            continue

        wl = weak_label_application(text)
        weakly_labeled.append({
            "id": filename.replace(".pdf", ""),
            "filename": filename,
            "text": text,
            "word_count": word_count,
            **wl,
            "document_type": "application",
            "use_for_training": True,
            "weight": wl["confidence"],
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "source": str(ANSOKAN_TEXTS_PATH),
            "n_applications": len(weakly_labeled),
            "note": "Weak labels for semi-supervised learning only. Do not use as ground truth.",
        },
        "applications": weakly_labeled,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(a["weak_label"] for a in weakly_labeled)
    print(f"Wrote {len(weakly_labeled)} weak labels to {OUTPUT_PATH}")
    print(f"  HIGH_RISK: {dist.get('HIGH_RISK', 0)}, MEDIUM_RISK: {dist.get('MEDIUM_RISK', 0)}, LOW_RISK: {dist.get('LOW_RISK', 0)}")


if __name__ == "__main__":
    main()
