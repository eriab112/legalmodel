"""
Extract text from ALL PDFs in Ansökningar and Lagstiftningdiverse.
Creates JSON outputs with text, word counts, page counts, and metadata.
"""
import fitz
import json
import re
import os
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"C:\Users\SE2I7A\Desktop\legalmodel\Data")


def extract_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        full_text = ""
        for page in doc:
            text = page.get_text()
            pages.append(text)
            full_text += text + "\n"
        page_count = len(doc)
        doc.close()
        word_count = len(full_text.split())
        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "text": full_text,
            "word_count": word_count,
            "page_count": page_count,
            "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            "status": "success"
        }
    except Exception as e:
        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "text": None,
            "word_count": 0,
            "page_count": 0,
            "size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            "status": "failed",
            "error": str(e)
        }


def categorize_ansokan(filename):
    """Categorize Ansökningar file by document type."""
    fn = filename.lower()
    if "ansökan" in fn or "ansökan" in fn or "aktbil 1" in fn:
        return "Ansökan"
    elif "dom " in fn or "dom_" in fn:
        return "Dom"
    elif "beslut" in fn:
        return "Beslut"
    elif "dagboksblad" in fn:
        return "Dagboksblad"
    elif "konsoliderad" in fn or "kompletterad" in fn:
        return "Komplettering"
    else:
        return "Okänd"


def extract_case_number(filename):
    """Extract case number (M-XXXX-XX) from filename."""
    match = re.search(r'[MmPpFf]\s*(\d+)-(\d+)', filename)
    if match:
        return f"M {match.group(1)}-{match.group(2)}"
    return None


def analyze_content(text):
    """Analyze content for NAP-relevant keywords."""
    if not text:
        return {}
    text_lower = text.lower()
    return {
        "has_vattenkraft": bool(re.search(r'vattenkraft|kraftverk|kraftstation', text_lower)),
        "has_water_body": bool(re.search(r'vattendrag|vattenförekomst|viss', text_lower)),
        "has_damm": bool(re.search(r'\bdamm\b|dammbyggnad|regleringsdamm', text_lower)),
        "has_measures": bool(re.search(r'åtgärd|fiskvandring|fiskväg|biotopvård|omlöp|minimitappning', text_lower)),
        "has_costs": bool(re.search(r'\d+[\s,.]?\d*\s*(kr|msek|mkr|miljoner|kronor)', text_lower)),
        "has_environmental": bool(re.search(r'miljöbalk|miljötillstånd|miljödom|miljöprövning', text_lower)),
        "has_nap": bool(re.search(r'nationell plan|nationella planen|omprövning', text_lower)),
        "has_legal_refs": bool(re.search(r'\d+\s*kap\.\s*\d+\s*§|miljöbalken', text_lower)),
    }


def categorize_lagtiftning(filename):
    """Categorize Lagstiftningdiverse file by type."""
    fn = filename.lower()
    if any(kw in fn for kw in ["directive", "direktiv", "cis", "guidance", "eudirektiv"]):
        return "EU_Directive"
    elif any(kw in fn for kw in ["miljöbalk", "miljobalken", "sfs", "swe"]):
        return "Swedish_Law"
    elif any(kw in fn for kw in ["kvalitetskrav", "vattendistrikt", "föreskrift", "fs 2021", "22fs"]):
        return "Water_District_Regulation"
    elif any(kw in fn for kw in ["nationell plan", "nap", "regeringsbeslut"]):
        return "NAP_Government"
    elif any(kw in fn for kw in ["bilaga", "riktlinj", "vägledning", "vagledning", "villkor"]):
        return "Technical_Guideline"
    elif any(kw in fn for kw in ["miljökonsekvensbeskrivning", "åtgärdsprogram", "atgardsprogram"]):
        return "Environmental_Assessment"
    elif any(kw in fn for kw in ["rapport", "remiss", "samråd", "samradsunderlag"]):
        return "Report_Consultation"
    elif "vm-riktlinjer" in fn or "vm_riktlinjer" in fn:
        return "Technical_Guideline"
    else:
        return "Other"


def main():
    out_dir = ROOT / "processed"
    os.makedirs(out_dir, exist_ok=True)

    # ===== ANSÖKNINGAR =====
    print("=" * 70)
    print("PHASE 2.1: ANSÖKNINGAR (Applications)")
    print("=" * 70)

    ansokan_dir = ROOT / "Ansökningar"
    ansokan_pdfs = sorted(ansokan_dir.glob("*.pdf"))
    print(f"\nFound {len(ansokan_pdfs)} PDFs in Ansökningar\n")

    ansokan_results = []
    for i, pdf in enumerate(ansokan_pdfs, 1):
        print(f"  [{i:2d}/{len(ansokan_pdfs)}] {pdf.name[:60]}...", end=" ")
        result = extract_pdf(pdf)
        result["doc_type"] = categorize_ansokan(pdf.name)
        result["case_number"] = extract_case_number(pdf.name)
        result["content_analysis"] = analyze_content(result.get("text", ""))
        ansokan_results.append(result)

        if result["status"] == "success":
            print(f"OK ({result['word_count']:,} words, {result['page_count']} pages)")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")

    # Summary
    successful = [r for r in ansokan_results if r["status"] == "success"]
    print(f"\n--- ANSÖKNINGAR SUMMARY ---")
    print(f"Total: {len(ansokan_results)}, Success: {len(successful)}, Failed: {len(ansokan_results) - len(successful)}")
    print(f"Total words: {sum(r['word_count'] for r in successful):,}")
    print(f"Total pages: {sum(r['page_count'] for r in successful):,}")

    # By type
    type_counts = defaultdict(int)
    for r in ansokan_results:
        type_counts[r["doc_type"]] += 1
    print(f"\nBy document type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    # Case number matching
    dom_dir = ROOT / "Domar" / "data" / "processed" / "court_decisions"
    dom_files = [f.stem for f in dom_dir.glob("*.txt")]

    matches = []
    for r in ansokan_results:
        cn = r.get("case_number")
        if cn:
            cn_normalized = cn.replace(" ", "-").replace("M-", "m-").lower()
            for dom in dom_files:
                dom_lower = dom.lower()
                # Match case number digits
                cn_digits = re.sub(r'[^0-9]', '', cn)
                dom_digits_match = re.search(r'm-?(\d+)-(\d+)', dom_lower)
                if dom_digits_match:
                    dom_digits = dom_digits_match.group(1) + dom_digits_match.group(2)
                    cn_parts = re.search(r'(\d+)-(\d+)', cn)
                    if cn_parts and cn_parts.group(1) + cn_parts.group(2) == dom_digits:
                        matches.append({
                            "ansokan": r["filename"],
                            "dom": dom + ".txt",
                            "case_number": cn,
                            "match_confidence": "high"
                        })

    # Deduplicate matches
    seen = set()
    unique_matches = []
    for m in matches:
        key = (m["ansokan"], m["dom"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)
    matches = unique_matches

    print(f"\nCase number matches with court decisions: {len(matches)}")
    for m in matches:
        print(f"  {m['ansokan'][:50]} <-> {m['dom']}")

    # NAP relevance
    nap_relevant = sum(1 for r in successful if r["content_analysis"].get("has_vattenkraft") or r["content_analysis"].get("has_nap"))
    print(f"\nNAP-relevant ansökningar: {nap_relevant}/{len(successful)}")

    # Save results (without full text for the summary)
    ansokan_summary = []
    for r in ansokan_results:
        summary = {k: v for k, v in r.items() if k != "text"}
        summary["text_preview"] = (r.get("text") or "")[:500]
        ansokan_summary.append(summary)

    with open(out_dir / "ansokan_structure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(ansokan_summary, f, indent=2, ensure_ascii=False)

    # Save full texts separately
    ansokan_texts = [{"filename": r["filename"], "text": r.get("text", ""), "word_count": r["word_count"]} for r in ansokan_results if r["status"] == "success"]
    with open(out_dir / "ansokan_texts.json", "w", encoding="utf-8") as f:
        json.dump(ansokan_texts, f, indent=2, ensure_ascii=False)

    with open(out_dir / "ansokan_dom_matches.json", "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)

    # ===== LAGSTIFTNINGDIVERSE =====
    print("\n" + "=" * 70)
    print("PHASE 2.2: LAGSTIFTNINGDIVERSE (Legislation)")
    print("=" * 70)

    lag_dir = ROOT / "Lagstiftningdiverse"
    lag_files = sorted(list(lag_dir.glob("*.pdf")) + list(lag_dir.glob("*.txt")))
    print(f"\nFound {len(lag_files)} files in Lagstiftningdiverse\n")

    lag_results = []
    for i, f in enumerate(lag_files, 1):
        print(f"  [{i:2d}/{len(lag_files)}] {f.name[:60]}...", end=" ")
        if f.suffix.lower() == ".pdf":
            result = extract_pdf(f)
        else:
            try:
                text = f.read_text(encoding="utf-8")
                result = {
                    "filename": f.name,
                    "path": str(f),
                    "text": text,
                    "word_count": len(text.split()),
                    "page_count": 0,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "status": "success"
                }
            except Exception as e:
                result = {
                    "filename": f.name,
                    "path": str(f),
                    "text": None,
                    "word_count": 0,
                    "page_count": 0,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                    "status": "failed",
                    "error": str(e)
                }

        result["category"] = categorize_lagtiftning(f.name)
        result["content_analysis"] = analyze_content(result.get("text", ""))
        lag_results.append(result)

        if result["status"] == "success":
            print(f"OK ({result['word_count']:,} words)")
        else:
            print(f"FAILED: {result.get('error', 'unknown')}")

    successful_lag = [r for r in lag_results if r["status"] == "success"]
    print(f"\n--- LAGSTIFTNING SUMMARY ---")
    print(f"Total: {len(lag_results)}, Success: {len(successful_lag)}, Failed: {len(lag_results) - len(successful_lag)}")
    print(f"Total words: {sum(r['word_count'] for r in successful_lag):,}")

    # By category
    cat_stats = defaultdict(lambda: {"count": 0, "words": 0, "nap_relevant": 0})
    for r in lag_results:
        cat = r["category"]
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["words"] += r["word_count"]
        if r["content_analysis"].get("has_vattenkraft") or r["content_analysis"].get("has_nap"):
            cat_stats[cat]["nap_relevant"] += 1

    print(f"\nBy category:")
    for cat, stats in sorted(cat_stats.items()):
        print(f"  {cat:30s}: {stats['count']:2d} files, {stats['words']:>8,} words, {stats['nap_relevant']} NAP-relevant")

    # Training potential assessment
    pretraining_candidates = []
    context_augmentation = []
    reference_only = []

    for r in successful_lag:
        ca = r["content_analysis"]
        nap_score = sum([
            ca.get("has_vattenkraft", False),
            ca.get("has_nap", False),
            ca.get("has_measures", False),
            ca.get("has_environmental", False),
            ca.get("has_legal_refs", False),
        ])
        if nap_score >= 3:
            pretraining_candidates.append(r["filename"])
        elif nap_score >= 1:
            context_augmentation.append(r["filename"])
        else:
            reference_only.append(r["filename"])

    print(f"\nTraining potential:")
    print(f"  Domain pre-training candidates: {len(pretraining_candidates)}")
    print(f"  Context augmentation: {len(context_augmentation)}")
    print(f"  Reference only: {len(reference_only)}")

    # Save results
    lag_summary = []
    for r in lag_results:
        summary = {k: v for k, v in r.items() if k != "text"}
        summary["text_preview"] = (r.get("text") or "")[:500]
        lag_summary.append(summary)

    with open(out_dir / "lagtiftning_categorization.json", "w", encoding="utf-8") as f:
        json.dump(lag_summary, f, indent=2, ensure_ascii=False)

    lag_texts = [{"filename": r["filename"], "text": r.get("text", ""), "word_count": r["word_count"], "category": r["category"]}
                 for r in lag_results if r["status"] == "success"]
    with open(out_dir / "lagtiftning_texts.json", "w", encoding="utf-8") as f:
        json.dump(lag_texts, f, indent=2, ensure_ascii=False)

    usefulness = {
        "pretraining_candidates": pretraining_candidates,
        "context_augmentation": context_augmentation,
        "reference_only": reference_only
    }
    with open(out_dir / "lagtiftning_usefulness.json", "w", encoding="utf-8") as f:
        json.dump(usefulness, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("ALL OUTPUTS SAVED TO Data/processed/")
    print("=" * 70)


if __name__ == "__main__":
    main()
