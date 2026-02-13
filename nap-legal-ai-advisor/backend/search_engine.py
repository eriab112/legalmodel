"""
Semantic search engine using sentence-transformers + numpy cosine similarity.

For 46 documents (~2000 chunks), numpy is fast enough (<10ms per query)
and eliminates external DB dependencies like ChromaDB/FAISS.
"""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
EMBEDDINGS_CACHE = CACHE_DIR / "embeddings.pkl"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class SearchResult:
    decision_id: str
    chunk_text: str
    chunk_index: int
    similarity: float
    filename: str
    label: Optional[str]
    metadata: Dict


class SemanticSearchEngine:
    """Semantic search over court decision chunks using sentence-transformers."""

    def __init__(self):
        self._model = None
        self._chunks: List[Dict] = []
        self._embeddings: Optional[np.ndarray] = None
        self._data_hash: str = ""

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks at word boundaries.

        Matches the chunking pattern from nap_model-main/tools/rag_engine.py.
        """
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space

            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 <= overlap:
                        overlap_words.insert(0, w)
                        overlap_length += len(w) + 1
                    else:
                        break

                current_chunk = overlap_words
                current_length = overlap_length
            else:
                pass  # continue building chunk

            current_chunk.append(word)
            current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def build_index(self, decisions):
        """Build search index from DecisionRecord list.

        Chunks key_text for each decision and computes embeddings.
        Uses cache if data hasn't changed.
        """
        # Compute hash of input data for cache validation
        hash_input = "".join(
            f"{d.id}:{len(d.key_text)}" for d in decisions
        )
        data_hash = hashlib.md5(hash_input.encode()).hexdigest()

        # Try loading from cache
        if self._try_load_cache(data_hash):
            return

        # Build chunks
        self._chunks = []
        for dec in decisions:
            text = dec.key_text if dec.key_text else dec.full_text[:5000]
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                self._chunks.append({
                    "text": chunk,
                    "chunk_index": i,
                    "decision_id": dec.id,
                    "filename": dec.filename,
                    "label": dec.label,
                    "metadata": dec.metadata,
                })

        # Encode all chunks
        texts = [c["text"] for c in self._chunks]
        self._embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        self._data_hash = data_hash

        # Save cache
        self._save_cache()

    def _try_load_cache(self, data_hash: str) -> bool:
        if not EMBEDDINGS_CACHE.exists():
            return False
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("data_hash") == data_hash:
                self._chunks = cache["chunks"]
                self._embeddings = cache["embeddings"]
                self._data_hash = data_hash
                return True
        except Exception:
            pass
        return False

    def _save_cache(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump({
                "data_hash": self._data_hash,
                "chunks": self._chunks,
                "embeddings": self._embeddings,
            }, f)

    def search(
        self,
        query: str,
        n_results: int = 5,
        label_filter: Optional[str] = None,
        court_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for relevant chunks using cosine similarity."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        )

        # Cosine similarity (embeddings are already L2-normalized)
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Apply filters
        mask = np.ones(len(self._chunks), dtype=bool)
        for i, chunk in enumerate(self._chunks):
            if label_filter and chunk["label"] != label_filter:
                mask[i] = False
            if court_filter and chunk["metadata"].get("court", "") != court_filter:
                mask[i] = False
            if date_from and chunk["metadata"].get("date", "") < date_from:
                mask[i] = False
            if date_to and chunk["metadata"].get("date", "") > date_to:
                mask[i] = False

        similarities[~mask] = -1

        # Get top-k, deduplicate by decision_id
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        seen_decisions = set()

        for idx in sorted_indices:
            if similarities[idx] <= 0:
                break
            chunk = self._chunks[idx]
            dec_id = chunk["decision_id"]

            if dec_id in seen_decisions:
                continue
            seen_decisions.add(dec_id)

            results.append(SearchResult(
                decision_id=dec_id,
                chunk_text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                similarity=float(similarities[idx]),
                filename=chunk["filename"],
                label=chunk["label"],
                metadata=chunk["metadata"],
            ))

            if len(results) >= n_results:
                break

        return results

    def find_similar_decisions(
        self, decision_id: str, n_results: int = 5
    ) -> List[SearchResult]:
        """Find decisions similar to a given one by averaging its chunk embeddings."""
        if self._embeddings is None:
            return []

        # Get all chunk indices for this decision
        indices = [
            i for i, c in enumerate(self._chunks)
            if c["decision_id"] == decision_id
        ]
        if not indices:
            return []

        # Average embedding of all chunks as query
        avg_embedding = np.mean(self._embeddings[indices], axis=0, keepdims=True)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        similarities = np.dot(self._embeddings, avg_embedding.T).flatten()

        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        seen = {decision_id}  # exclude the query decision itself

        for idx in sorted_indices:
            chunk = self._chunks[idx]
            dec_id = chunk["decision_id"]
            if dec_id in seen:
                continue
            seen.add(dec_id)

            results.append(SearchResult(
                decision_id=dec_id,
                chunk_text=chunk["text"],
                chunk_index=chunk["chunk_index"],
                similarity=float(similarities[idx]),
                filename=chunk["filename"],
                label=chunk["label"],
                metadata=chunk["metadata"],
            ))

            if len(results) >= n_results:
                break

        return results

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)


@st.cache_resource
def get_search_engine():
    return SemanticSearchEngine()
