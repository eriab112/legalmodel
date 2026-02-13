#!/usr/bin/env python3
"""
Phase A5 (low-risk): Build domain pre-training corpus.

Reads:
  - Data/processed/lagtiftning_texts.json (list of {filename, text, word_count, category})
  - Data/processed/ansokan_texts.json (list; only Ansökan by filename)
  - Data/processed/cleaned_court_texts.json (decisions[].text_full)

Writes: Data/processed/dapt_corpus.json (NEW only)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "Data" / "processed"
LAGTIFTNING_PATH = PROCESSED / "lagtiftning_texts.json"
ANSOKAN_PATH = PROCESSED / "ansokan_texts.json"
CLEANED_PATH = PROCESSED / "cleaned_court_texts.json"
OUTPUT_PATH = PROCESSED / "dapt_corpus.json"


def is_ansokan_filename(filename: str) -> bool:
    fn = (filename or "").lower()
    if "dom " in fn or "dom_" in fn or "dagboksblad" in fn or "beslut" in fn:
        return False
    return "ansökan" in fn or "aktbil 1" in fn


def main():
    corpus_documents = []
    total_words = 0

    # 1. Legislation
    if LAGTIFTNING_PATH.exists():
        with open(LAGTIFTNING_PATH, "r", encoding="utf-8") as f:
            lag_list = json.load(f)
        if isinstance(lag_list, list):
            for doc in lag_list:
                text = (doc.get("text") or "").strip()
                if not text:
                    continue
                corpus_documents.append({
                    "text": text,
                    "source": "legislation",
                    "category": doc.get("category", "unknown"),
                    "filename": doc.get("filename", ""),
                })
                total_words += len(text.split())
    n_leg = sum(1 for d in corpus_documents if d["source"] == "legislation")

    # 2. Applications (Ansökan only, no Dom)
    if ANSOKAN_PATH.exists():
        with open(ANSOKAN_PATH, "r", encoding="utf-8") as f:
            ans_list = json.load(f)
        if isinstance(ans_list, list):
            for doc in ans_list:
                if not is_ansokan_filename(doc.get("filename", "")):
                    continue
                text = (doc.get("text") or "").strip()
                if not text or len(text) < 200:
                    continue
                corpus_documents.append({
                    "text": text,
                    "source": "application",
                    "filename": doc.get("filename", ""),
                })
                total_words += len(text.split())
    n_app = sum(1 for d in corpus_documents if d["source"] == "application")

    # 3. Court decisions (text_full from cleaned_court_texts)
    if CLEANED_PATH.exists():
        with open(CLEANED_PATH, "r", encoding="utf-8") as f:
            cleaned = json.load(f)
        decisions = cleaned.get("decisions", [])
        for d in decisions:
            text = (d.get("text_full") or d.get("key_text") or "").strip()
            if not text:
                continue
            corpus_documents.append({
                "text": text,
                "source": "court_decision",
                "id": d.get("id", ""),
                "filename": d.get("filename", ""),
            })
            total_words += len(text.split())
    n_court = sum(1 for d in corpus_documents if d["source"] == "court_decision")

    metadata = {
        "total_documents": len(corpus_documents),
        "total_words": total_words,
        "sources": {
            "legislation": n_leg,
            "applications": n_app,
            "court_decisions": n_court,
        },
        "creation_date": "2026-02-12",
        "purpose": "domain_adaptive_pretraining",
    }

    out = {"metadata": metadata, "documents": corpus_documents}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"DAPT corpus written to {OUTPUT_PATH}")
    print(f"  Total documents: {len(corpus_documents)}")
    print(f"  Total words: {total_words:,}")
    print(f"  Legislation: {n_leg}, Applications: {n_app}, Court decisions: {n_court}")


if __name__ == "__main__":
    main()
