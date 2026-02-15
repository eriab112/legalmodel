"""
Startup integration tests for NAP Legal AI Advisor.

These tests replicate the app's init_backend() flow step-by-step using real
paths and real data (when run from repo root). They are used to diagnose
why the dashboard may not start.

Run from repo root: pytest tests/test_startup_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure nap-legal-ai-advisor is on path (conftest does this, but we need it for path checks)
_APP_DIR = Path(__file__).resolve().parent.parent / "nap-legal-ai-advisor"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# Apply SSL workaround before any Hugging Face / sentence_transformers usage (same as app.py)
import utils.ssl_fix  # noqa: F401


# ---------------------------------------------------------------------------
# Path resolution (what the app sees)
# ---------------------------------------------------------------------------

def get_data_loader_base_dir():
    """Return BASE_DIR as resolved by utils.data_loader (repo root)."""
    from utils import data_loader
    return data_loader.BASE_DIR


def get_required_data_paths():
    """Paths that must exist for load_data() and load_knowledge_base() to succeed."""
    base = get_data_loader_base_dir()
    return {
        "DATA_DIR": base / "Data" / "processed",
        "labeled_dataset.json": base / "Data" / "processed" / "labeled_dataset.json",
        "cleaned_court_texts.json": base / "Data" / "processed" / "cleaned_court_texts.json",
        "linkage_table.json": base / "Data" / "processed" / "linkage_table.json",
        "lagtiftning_texts.json": base / "Data" / "processed" / "lagtiftning_texts.json",
        "ansokan_texts.json": base / "Data" / "processed" / "ansokan_texts.json",
    }


def get_risk_model_path():
    """Return MODEL_DIR as used by risk_predictor (optional for startup)."""
    from backend import risk_predictor
    return risk_predictor.MODEL_DIR


@pytest.fixture(scope="module")
def repo_paths():
    """Resolved paths and existence flags."""
    paths = get_required_data_paths()
    base = get_data_loader_base_dir()
    model_dir = get_risk_model_path()
    return {
        "base_dir": base,
        "data_dir_exists": paths["DATA_DIR"].exists(),
        "labeled_exists": paths["labeled_dataset.json"].exists(),
        "cleaned_exists": paths["cleaned_court_texts.json"].exists(),
        "linkage_exists": paths["linkage_table.json"].exists(),
        "legislation_exists": paths["lagtiftning_texts.json"].exists(),
        "application_exists": paths["ansokan_texts.json"].exists(),
        "model_dir_exists": model_dir.exists(),
        "model_dir": model_dir,
        "all_data_exist": all([
            paths["labeled_dataset.json"].exists(),
            paths["cleaned_court_texts.json"].exists(),
            paths["linkage_table.json"].exists(),
        ]),
    }


def _data_files_available():
    """True if required data files exist at app's BASE_DIR."""
    base = get_data_loader_base_dir()
    return (base / "Data" / "processed" / "labeled_dataset.json").exists()


# ---------------------------------------------------------------------------
# Unit-style: path and file existence
# ---------------------------------------------------------------------------

class TestPathResolution:
    """Verify where the app thinks data and models live."""

    def test_base_dir_is_resolved(self, repo_paths):
        """BASE_DIR (repo root) should be an absolute path."""
        base = repo_paths["base_dir"]
        assert base.is_absolute(), f"BASE_DIR should be absolute, got {base}"

    def test_base_dir_has_nap_legal_ai_advisor_child(self, repo_paths):
        """Repo root should contain nap-legal-ai-advisor folder."""
        base = repo_paths["base_dir"]
        app_dir = base / "nap-legal-ai-advisor"
        assert app_dir.is_dir(), (
            f"Expected repo root {base} to contain nap-legal-ai-advisor/. "
            f"Contents: {list(base.iterdir()) if base.exists() else 'base does not exist'}"
        )

    def test_data_dir_exists(self, repo_paths):
        """Data/processed must exist for the app to start."""
        assert repo_paths["data_dir_exists"], (
            f"Data/processed not found under {repo_paths['base_dir']}. "
            "Run the app from the folder that contains both nap-legal-ai-advisor/ and Data/."
        )

    def test_labeled_dataset_exists(self, repo_paths):
        """labeled_dataset.json must exist."""
        assert repo_paths["labeled_exists"], (
            f"Missing: {repo_paths['base_dir'] / 'Data' / 'processed' / 'labeled_dataset.json'}"
        )

    def test_cleaned_court_texts_exists(self, repo_paths):
        """cleaned_court_texts.json must exist."""
        assert repo_paths["cleaned_exists"], (
            f"Missing: {repo_paths['base_dir'] / 'Data' / 'processed' / 'cleaned_court_texts.json'}"
        )

    def test_linkage_table_exists(self, repo_paths):
        """linkage_table.json should exist (optional in code but usually present)."""
        assert repo_paths["linkage_exists"], (
            f"Missing: {repo_paths['base_dir'] / 'Data' / 'processed' / 'linkage_table.json'}"
        )


# ---------------------------------------------------------------------------
# Integration: load_data (requires real files)
# ---------------------------------------------------------------------------

class TestLoadData:
    """Test load_data() with real paths."""

    @pytest.mark.skipif(
        not Path(__file__).resolve().parent.parent.joinpath(
            "Data/processed/labeled_dataset.json"
        ).exists(),
        reason="Required data files not found at repo root (run from legalmodel/)",
    )
    def test_load_data_succeeds(self):
        """load_data() should complete without error."""
        from utils.data_loader import load_data
        data = load_data()
        assert data is not None
        decisions = data.get_all_decisions()
        assert len(decisions) > 0, "Expected at least one decision"
        dist = data.get_label_distribution()
        assert isinstance(dist, dict)


# ---------------------------------------------------------------------------
# Integration: load_knowledge_base (requires real files)
# ---------------------------------------------------------------------------

class TestLoadKnowledgeBase:
    """Test load_knowledge_base() with real paths."""

    @pytest.mark.skipif(
        not Path(__file__).resolve().parent.parent.joinpath(
            "Data/processed/cleaned_court_texts.json"
        ).exists(),
        reason="Required data files not found at repo root",
    )
    def test_load_knowledge_base_succeeds(self):
        """load_knowledge_base() should complete without error."""
        from utils.data_loader import load_knowledge_base
        kb = load_knowledge_base()
        assert kb is not None
        docs = kb.get_all_documents()
        assert len(docs) > 0, "Expected at least one document"
        stats = kb.get_corpus_stats()
        assert "total" in stats


# ---------------------------------------------------------------------------
# Integration: search engine and build_full_index (slow if cache miss)
# ---------------------------------------------------------------------------

class TestSearchEngineBuild:
    """Test search engine and build_full_index."""

    @pytest.mark.skipif(not _data_files_available(), reason="Required data files not at app BASE_DIR")
    @pytest.mark.slow
    def test_build_full_index_succeeds(self):
        """build_full_index(kb.get_all_documents()) should complete or use cache.
        Requires sentence-transformers model (download can fail with SSL in corporate proxy)."""
        from utils.data_loader import load_knowledge_base
        from backend.search_engine import get_search_engine

        kb = load_knowledge_base()
        search = get_search_engine()
        documents = kb.get_all_documents()
        assert len(documents) > 0
        search.build_full_index(documents)
        assert search._embeddings is not None
        assert len(search._chunks) > 0
        assert search.total_chunks == len(search._chunks)


# ---------------------------------------------------------------------------
# Integration: full startup sequence (no Streamlit UI)
# ---------------------------------------------------------------------------

class TestFullStartupSequence:
    """Replicate init_backend() steps without Streamlit to isolate failure."""

    @pytest.mark.skipif(not _data_files_available(), reason="Required data files not at app BASE_DIR")
    @pytest.mark.slow
    def test_full_startup_without_ui(self):
        """
        Run the same steps as init_backend(): load data, KB, search engine,
        build index, predictor, LLM engine, RAGSystem, handlers.
        Fails at the first step that raises.
        """
        from utils.data_loader import load_data, load_knowledge_base
        from backend.search_engine import get_search_engine
        from backend.risk_predictor import get_predictor
        from backend.llm_engine import get_llm_engine
        from backend.rag_system import RAGSystem
        from integration.chat_handler import ChatHandler
        from integration.search_handler import SearchHandler

        # Step 1: load data
        data = load_data()
        assert data is not None
        assert len(data.get_all_decisions()) > 0

        # Step 2: search engine and predictor (lazy, no heavy load yet)
        search = get_search_engine()
        predictor = get_predictor()
        assert search is not None
        assert predictor is not None

        # Step 3: knowledge base
        kb = load_knowledge_base()
        assert kb is not None
        assert len(kb.get_all_documents()) > 0

        # Step 4: build full index (may load sentence-transformers and encode)
        search.build_full_index(kb.get_all_documents())
        assert search._embeddings is not None
        assert search.total_chunks > 0

        # Step 5: LLM engine (may be None if no API key)
        llm_engine = get_llm_engine()
        # No assert - can be None

        # Step 6: wire up RAG and handlers
        rag = RAGSystem(data, search, predictor, llm_engine=llm_engine)
        chat_handler = ChatHandler(rag)
        search_handler = SearchHandler(data, search, predictor)
        assert rag is not None
        assert chat_handler is not None
        assert search_handler is not None


# ---------------------------------------------------------------------------
# Diagnostic report (always runs, no skip)
# ---------------------------------------------------------------------------

class TestStartupDiagnostics:
    """Emit a diagnostic report of paths and existence (no assertions)."""

    def test_report_resolved_paths(self, repo_paths, capsys):
        """Print where the app looks for data and whether files exist."""
        with capsys.disabled():
            # Print so it shows in pytest -v output
            print("\n--- NAP Legal AI Advisor startup diagnostics ---")
            print(f"  BASE_DIR (repo root): {repo_paths['base_dir']}")
            print(f"  DATA_DIR exists:     {repo_paths['data_dir_exists']}")
            print(f"  labeled_dataset:     {repo_paths['labeled_exists']}")
            print(f"  cleaned_court_texts:  {repo_paths['cleaned_exists']}")
            print(f"  linkage_table:       {repo_paths['linkage_exists']}")
            print(f"  legislation:         {repo_paths['legislation_exists']}")
            print(f"  application:         {repo_paths['application_exists']}")
            print(f"  Model dir:           {repo_paths['model_dir']}")
            print(f"  Model dir exists:    {repo_paths['model_dir_exists']}")
            print(f"  All required data:   {repo_paths['all_data_exist']}")
            print("--------------------------------------------------\n")
