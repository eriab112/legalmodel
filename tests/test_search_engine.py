"""Tests for backend.search_engine — text chunking, search, filtering."""

import numpy as np
import pytest

from backend.search_engine import SearchResult, SemanticSearchEngine


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(
            decision_id="m1-22",
            chunk_text="Some relevant text",
            chunk_index=0,
            similarity=0.85,
            filename="m1-22.txt",
            label="HIGH_RISK",
            metadata={"court": "Nacka TR"},
        )
        assert sr.decision_id == "m1-22"
        assert sr.similarity == 0.85
        assert sr.label == "HIGH_RISK"


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    @pytest.fixture
    def engine(self):
        return SemanticSearchEngine()

    def test_short_text_single_chunk(self, engine):
        text = "Hello world this is a test."
        chunks = engine.chunk_text(text, chunk_size=500, overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self, engine):
        # Build text longer than chunk_size
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)
        chunks = engine.chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_overlap_creates_shared_words(self, engine):
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = engine.chunk_text(text, chunk_size=50, overlap=20)
        # The last words of chunk N should appear at the start of chunk N+1
        if len(chunks) >= 2:
            last_words_0 = set(chunks[0].split()[-5:])
            first_words_1 = set(chunks[1].split()[:10])
            assert len(last_words_0 & first_words_1) > 0

    def test_empty_text(self, engine):
        chunks = engine.chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []

    def test_all_text_covered(self, engine):
        words = [f"token{i}" for i in range(50)]
        text = " ".join(words)
        chunks = engine.chunk_text(text, chunk_size=80, overlap=20)
        # First word in first chunk, last word in last chunk
        assert "token0" in chunks[0]
        assert "token49" in chunks[-1]


# ---------------------------------------------------------------------------
# search with pre-built embeddings (no model loading)
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.fixture
    def loaded_engine(self):
        """Engine with pre-built chunks and fake embeddings (no model needed)."""
        engine = SemanticSearchEngine()
        engine._chunks = [
            {"text": "fiskvandringsväg", "chunk_index": 0, "decision_id": "m1-22",
             "filename": "m1-22.txt", "label": "HIGH_RISK",
             "metadata": {"court": "Nacka TR", "date": "2024-01-10"}},
            {"text": "minimitappning", "chunk_index": 0, "decision_id": "m2-22",
             "filename": "m2-22.txt", "label": "MEDIUM_RISK",
             "metadata": {"court": "Växjö TR", "date": "2023-06-01"}},
            {"text": "kostnad och åtgärd", "chunk_index": 0, "decision_id": "m3-22",
             "filename": "m3-22.txt", "label": "LOW_RISK",
             "metadata": {"court": "Nacka TR", "date": "2024-03-20"}},
        ]
        # Create normalized embeddings (3 chunks, dim=4)
        emb = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ], dtype=np.float32)
        engine._embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return engine

    def test_search_returns_empty_when_no_index(self):
        engine = SemanticSearchEngine()
        # Mock model.encode to avoid loading the real model
        engine._model = type("FakeModel", (), {
            "encode": lambda self, texts, **kw: np.zeros((len(texts), 4))
        })()
        results = engine.search("test query")
        assert results == []

    def test_search_with_label_filter(self, loaded_engine):
        # Mock the model so .search() can encode the query
        loaded_engine._model = type("FakeModel", (), {
            "encode": lambda self, texts, **kw: np.array([[1.0, 0.0, 0.0, 0.0]])
        })()
        results = loaded_engine.search("test", n_results=10, label_filter="MEDIUM_RISK")
        # Only MEDIUM_RISK should appear
        assert all(r.label == "MEDIUM_RISK" for r in results)

    def test_search_with_court_filter(self, loaded_engine):
        loaded_engine._model = type("FakeModel", (), {
            "encode": lambda self, texts, **kw: np.array([[1.0, 0.0, 0.0, 0.0]])
        })()
        results = loaded_engine.search("test", n_results=10, court_filter="Växjö TR")
        assert all(r.metadata.get("court") == "Växjö TR" for r in results)

    def test_search_deduplicates_by_decision(self, loaded_engine):
        # Add second chunk for same decision
        loaded_engine._chunks.append({
            "text": "second chunk", "chunk_index": 1, "decision_id": "m1-22",
            "filename": "m1-22.txt", "label": "HIGH_RISK",
            "metadata": {"court": "Nacka TR", "date": "2024-01-10"},
        })
        loaded_engine._embeddings = np.vstack([
            loaded_engine._embeddings,
            np.array([[0.9, 0.1, 0.0, 0.0]])
        ])
        loaded_engine._embeddings = loaded_engine._embeddings / np.linalg.norm(
            loaded_engine._embeddings, axis=1, keepdims=True
        )

        loaded_engine._model = type("FakeModel", (), {
            "encode": lambda self, texts, **kw: np.array([[1.0, 0.0, 0.0, 0.0]])
        })()
        results = loaded_engine.search("test", n_results=10)
        decision_ids = [r.decision_id for r in results]
        assert len(decision_ids) == len(set(decision_ids))

    def test_search_respects_n_results(self, loaded_engine):
        loaded_engine._model = type("FakeModel", (), {
            "encode": lambda self, texts, **kw: np.array([[0.5, 0.5, 0.0, 0.0]])
        })()
        results = loaded_engine.search("test", n_results=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# find_similar_decisions
# ---------------------------------------------------------------------------

class TestFindSimilar:
    @pytest.fixture
    def engine(self):
        engine = SemanticSearchEngine()
        engine._chunks = [
            {"text": "a", "chunk_index": 0, "decision_id": "d1",
             "filename": "d1.txt", "label": "HIGH_RISK", "metadata": {}},
            {"text": "b", "chunk_index": 0, "decision_id": "d2",
             "filename": "d2.txt", "label": "LOW_RISK", "metadata": {}},
        ]
        emb = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
        engine._embeddings = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        return engine

    def test_excludes_query_decision(self, engine):
        results = engine.find_similar_decisions("d1", n_results=5)
        assert all(r.decision_id != "d1" for r in results)

    def test_returns_results(self, engine):
        results = engine.find_similar_decisions("d1", n_results=5)
        assert len(results) >= 1

    def test_no_results_for_unknown_id(self, engine):
        results = engine.find_similar_decisions("unknown", n_results=5)
        assert results == []

    def test_no_results_when_no_embeddings(self):
        engine = SemanticSearchEngine()
        results = engine.find_similar_decisions("d1")
        assert results == []


# ---------------------------------------------------------------------------
# total_chunks property
# ---------------------------------------------------------------------------

class TestTotalChunks:
    def test_empty(self):
        engine = SemanticSearchEngine()
        assert engine.total_chunks == 0

    def test_with_chunks(self):
        engine = SemanticSearchEngine()
        engine._chunks = [{"text": "a"}, {"text": "b"}]
        assert engine.total_chunks == 2
