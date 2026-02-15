"""Tests for backend.llm_engine — context formatting, engine initialization."""

import pytest
from unittest.mock import MagicMock
from backend.llm_engine import format_context, get_llm_engine


class TestFormatContext:
    def test_format_single_result(self):
        result = MagicMock()
        result.title = "M 1234-22 — Nacka TR"
        result.doc_type = "decision"
        result.decision_id = "m1234-22"
        result.chunk_text = "Domstolen beslutar om fiskvandringsväg."
        result.metadata = {"case_number": "M 1234-22"}
        result.filename = "m1234-22.txt"

        context, sources = format_context([result])
        assert "[1]" in context
        assert "M 1234-22" in context or "Nacka TR" in context
        assert len(sources) == 1
        assert sources[0]["index"] == 1

    def test_format_multiple_results(self):
        results = []
        for i in range(3):
            r = MagicMock()
            r.title = f"Doc {i}"
            r.doc_type = "legislation"
            r.decision_id = f"doc{i}"
            r.chunk_text = f"Text content {i}"
            r.metadata = {}
            r.filename = f"doc{i}.txt"
            results.append(r)

        context, sources = format_context(results)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context
        assert len(sources) == 3

    def test_format_empty_results(self):
        context, sources = format_context([])
        assert sources == []

    def test_doc_type_labels(self):
        for doc_type, expected_sv in [("decision", "beslut"), ("legislation", "lagstiftning"), ("application", "ansökan")]:
            r = MagicMock()
            r.title = "Test"
            r.doc_type = doc_type
            r.decision_id = "test"
            r.chunk_text = "text"
            r.metadata = {}
            r.filename = "test.txt"
            context, sources = format_context([r])
            assert expected_sv in context.lower() or expected_sv in sources[0].get("type_label", "").lower()


class TestGetLlmEngine:
    def test_returns_none_without_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        engine = get_llm_engine()
        assert engine is None
