"""
Diagnostic script for the RAG search index.

Inspects the full_embeddings.pkl cache to report:
- Chunk counts by doc_type
- Document coverage for specific case IDs
- Retrieval quality via cosine similarity searches
- Data quality issues (short docs, duplicates, garbage chunks)
"""

import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup paths — must come before any HuggingFace imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "nap-legal-ai-advisor" / ".cache" / "full_embeddings.pkl"

# Add project source to sys.path for imports (needed for ssl_fix)
APP_DIR = PROJECT_ROOT / "nap-legal-ai-advisor"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Apply SSL fix before any network/model imports
import utils.ssl_fix  # noqa: E402, F401

import numpy as np


def load_cache():
    """Load the full embeddings cache."""
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache file not found at {CACHE_PATH}")
        sys.exit(1)

    print(f"Loading cache from {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1024 / 1024:.1f} MB)...")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    chunks = cache["chunks"]
    embeddings = cache["embeddings"]
    data_hash = cache.get("data_hash", "unknown")

    print(f"Cache data hash: {data_hash}")
    print(f"Embeddings shape: {embeddings.shape}")
    print()
    return chunks, embeddings


def diagnose_index_structure(chunks):
    """Report chunk counts by doc_type and per document."""
    print("=" * 70)
    print("1. SEARCH INDEX STRUCTURE")
    print("=" * 70)
    print(f"\nTotal number of chunks: {len(chunks)}")

    # Chunks per doc_type
    doc_type_counts = Counter(c.get("doc_type", "decision") for c in chunks)
    print("\nChunks per doc_type:")
    for dt, count in sorted(doc_type_counts.items()):
        print(f"  {dt}: {count} chunks")

    # Documents per doc_type
    docs_by_type = defaultdict(set)
    chunks_per_doc = defaultdict(int)
    for c in chunks:
        doc_type = c.get("doc_type", "decision")
        doc_id = c["decision_id"]
        docs_by_type[doc_type].add(doc_id)
        chunks_per_doc[doc_id] += 1

    print("\nDocuments per doc_type:")
    for dt in sorted(docs_by_type.keys()):
        doc_ids = docs_by_type[dt]
        chunk_counts = [chunks_per_doc[did] for did in doc_ids]
        avg = sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
        print(f"  {dt}: {len(doc_ids)} documents, avg {avg:.1f} chunks/doc")

    # Top 5 largest documents by chunk count
    sorted_docs = sorted(chunks_per_doc.items(), key=lambda x: -x[1])
    print("\nTop 5 largest documents by chunk count:")
    for doc_id, count in sorted_docs[:5]:
        # Find doc_type for this doc
        dt = next((c.get("doc_type", "decision") for c in chunks if c["decision_id"] == doc_id), "?")
        print(f"  {doc_id}: {count} chunks ({dt})")

    # Top 5 smallest documents by chunk count
    print("\nTop 5 smallest documents by chunk count:")
    for doc_id, count in sorted_docs[-5:]:
        dt = next((c.get("doc_type", "decision") for c in chunks if c["decision_id"] == doc_id), "?")
        print(f"  {doc_id}: {count} chunks ({dt})")

    return docs_by_type, chunks_per_doc


def check_specific_documents(chunks, chunks_per_doc):
    """Check coverage for specific decision IDs."""
    print("\n" + "=" * 70)
    print("2. SPECIFIC DOCUMENT COVERAGE")
    print("=" * 70)

    target_ids = ["m483-22", "m3753-22", "m605-24", "m2796-24", "m2694-22"]

    all_doc_ids = set(c["decision_id"] for c in chunks)

    for case_id in target_ids:
        count = chunks_per_doc.get(case_id, 0)
        if count > 0:
            print(f"  {case_id}: {count} chunks - FOUND")
        else:
            # Check for close matches (different formatting)
            close = [did for did in all_doc_ids if case_id.replace("-", "") in did.replace("-", "")]
            if close:
                print(f"  {case_id}: MISSING (but found similar: {close})")
            else:
                print(f"  {case_id}: MISSING - not in index")

    print()


def test_retrieval_quality(chunks, embeddings):
    """Test retrieval quality by running sample queries."""
    print("=" * 70)
    print("3. RETRIEVAL QUALITY TEST")
    print("=" * 70)

    # Load the sentence-transformers model
    print("\nLoading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    test_queries = [
        "M 483-22",
        "vad hände i m 483-22",
        "fiskväg Stora Mölla Pinnån",
        "miljöbalken vattenverksamhet",
        "vattendirektivet artikel 4.7",
        "domstolens bedömning fiskpassage",
    ]

    for query in test_queries:
        print(f"\n--- Query: \"{query}\" ---")
        query_emb = model.encode([query], normalize_embeddings=True)
        similarities = np.dot(embeddings, query_emb.T).flatten()

        # Get top 5 unique documents
        sorted_indices = np.argsort(similarities)[::-1]
        seen = set()
        results = []
        for idx in sorted_indices:
            doc_id = chunks[idx]["decision_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_type = chunks[idx].get("doc_type", "decision")
            sim = similarities[idx]
            title = chunks[idx].get("title", doc_id)
            text_preview = chunks[idx]["text"][:100].replace("\n", " ")
            results.append((doc_id, doc_type, sim, title, text_preview))
            if len(results) >= 5:
                break

        for doc_id, doc_type, sim, title, preview in results:
            print(f"  [{sim:.4f}] {doc_type:12s} | {doc_id:20s} | {preview}...")


def check_data_quality(chunks):
    """Check for various data quality issues."""
    print("\n" + "=" * 70)
    print("4. DATA QUALITY ISSUES")
    print("=" * 70)

    # Group chunks by document
    doc_texts = defaultdict(list)
    doc_types = {}
    doc_filenames = {}
    for c in chunks:
        doc_id = c["decision_id"]
        doc_texts[doc_id].append(c["text"])
        doc_types[doc_id] = c.get("doc_type", "decision")
        doc_filenames[doc_id] = c.get("filename", "") if "filename" in c else ""

    # Documents with < 100 chars total text
    print("\nDocuments with fewer than 100 characters of text:")
    short_docs = 0
    for doc_id, texts in doc_texts.items():
        total_text = " ".join(texts)
        if len(total_text) < 100:
            short_docs += 1
            print(f"  {doc_id} ({doc_types[doc_id]}): {len(total_text)} chars")
    if short_docs == 0:
        print("  None found")

    # Documents appearing in multiple doc_types
    print("\nDocuments appearing in multiple doc_types:")
    doc_type_map = defaultdict(set)
    for c in chunks:
        doc_type_map[c["decision_id"]].add(c.get("doc_type", "decision"))
    dupes = {did: types for did, types in doc_type_map.items() if len(types) > 1}
    if dupes:
        for did, types in dupes.items():
            print(f"  {did}: {types}")
    else:
        print("  None found")

    # Chunks that are mostly whitespace or OCR garbage
    print("\nChunks that are mostly whitespace or OCR garbage:")
    garbage_count = 0
    for c in chunks:
        text = c["text"]
        # Check if mostly whitespace
        non_ws = text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
        if len(non_ws) < len(text) * 0.3 and len(text) > 10:
            garbage_count += 1
            if garbage_count <= 5:
                doc_id = c["decision_id"]
                preview = text[:80].replace("\n", "\\n")
                print(f"  [{doc_id}] Mostly whitespace: \"{preview}\"")
        # Check for OCR garbage (high ratio of non-alpha characters)
        elif len(text) > 20:
            alpha_count = sum(1 for ch in text if ch.isalpha())
            if alpha_count < len(text) * 0.3:
                garbage_count += 1
                if garbage_count <= 5:
                    doc_id = c["decision_id"]
                    preview = text[:80].replace("\n", "\\n")
                    print(f"  [{doc_id}] Possible OCR garbage: \"{preview}\"")

    print(f"  Total suspect chunks: {garbage_count}")

    # Application documents: dagboksblad vs substantive
    print("\nApplication document analysis (dagboksblad vs substantive):")
    app_docs = {did: texts for did, texts in doc_texts.items() if doc_types[did] == "application"}
    dagbok_count = 0
    substantive_count = 0
    for doc_id, texts in app_docs.items():
        full_text = " ".join(texts).lower()
        fname = doc_filenames.get(doc_id, "").lower()
        if "dagbok" in full_text or "dagbok" in fname:
            dagbok_count += 1
        else:
            substantive_count += 1

    print(f"  Total application documents: {len(app_docs)}")
    print(f"  Dagboksblad (diary sheets): {dagbok_count}")
    print(f"  Substantive applications: {substantive_count}")

    # List application filenames for inspection
    if app_docs:
        print("\n  Application document details:")
        for doc_id in sorted(app_docs.keys()):
            fname = doc_filenames.get(doc_id, "N/A")
            total_chars = sum(len(t) for t in doc_texts[doc_id])
            chunk_count = len(doc_texts[doc_id])
            full_text = " ".join(doc_texts[doc_id]).lower()
            is_dagbok = "dagbok" in full_text or "dagbok" in fname.lower()
            marker = " [DAGBOK]" if is_dagbok else ""
            print(f"    {doc_id}: {chunk_count} chunks, {total_chars} chars, file={fname}{marker}")


def main():
    print("NAP Legal AI — Search Index Diagnostic")
    print("=" * 70)
    print()

    chunks, embeddings = load_cache()
    docs_by_type, chunks_per_doc = diagnose_index_structure(chunks)
    check_specific_documents(chunks, chunks_per_doc)
    test_retrieval_quality(chunks, embeddings)
    check_data_quality(chunks)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
