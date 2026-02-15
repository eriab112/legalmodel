"""Tests for utils.data_loader.KnowledgeBase — document loading and corpus stats."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from utils.data_loader import KnowledgeBase, DocumentRecord


class TestDocumentRecord:
    def test_creation(self):
        doc = DocumentRecord(
            doc_id="test",
            doc_type="decision",
            filename="test.txt",
            title="Test Document",
            text="Full text here",
            metadata={"court": "Nacka TR"},
        )
        assert doc.doc_id == "test"
        assert doc.doc_type == "decision"
        assert doc.label is None  # default

    def test_with_label(self):
        doc = DocumentRecord(
            doc_id="test",
            doc_type="decision",
            filename="test.txt",
            title="Test",
            text="text",
            metadata={},
            label="HIGH_RISK",
        )
        assert doc.label == "HIGH_RISK"


class TestKnowledgeBaseCorpusStats:
    """Test corpus stats computation without loading real data."""

    def test_stats_counting(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._documents = [
            DocumentRecord("d1", "decision", "f1", "t1", "text", {}),
            DocumentRecord("d2", "decision", "f2", "t2", "text", {}),
            DocumentRecord("d3", "legislation", "f3", "t3", "text", {}),
            DocumentRecord("d4", "application", "f4", "t4", "text", {}),
        ]
        kb._by_id = {d.doc_id: d for d in kb._documents}

        stats = kb.get_corpus_stats()
        assert stats["decision"] == 2
        assert stats["legislation"] == 1
        assert stats["application"] == 1
        assert stats["total"] == 4

    def test_get_documents_by_type(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._documents = [
            DocumentRecord("d1", "decision", "f1", "t1", "text", {}),
            DocumentRecord("d2", "legislation", "f2", "t2", "text", {}),
        ]
        kb._by_id = {d.doc_id: d for d in kb._documents}

        legs = kb.get_documents_by_type("legislation")
        assert len(legs) == 1
        assert legs[0].doc_id == "d2"

    def test_get_document_by_id(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        kb._documents = [DocumentRecord("d1", "decision", "f1", "t1", "text", {})]
        kb._by_id = {"d1": kb._documents[0]}

        assert kb.get_document("d1") is not None
        assert kb.get_document("nonexistent") is None
