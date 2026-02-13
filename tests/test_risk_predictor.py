"""Tests for backend.risk_predictor — chunking, softmax, prediction result."""

import numpy as np
import pytest

from backend.risk_predictor import (
    CHUNK_SIZE,
    ID2LABEL,
    LABEL_MAP,
    NUM_LABELS,
    STRIDE,
    PredictionResult,
    _softmax,
)


# ---------------------------------------------------------------------------
# _softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_uniform_logits(self):
        result = _softmax(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)

    def test_dominant_class(self):
        result = _softmax(np.array([10.0, 0.0, 0.0]))
        assert result[0] > 0.99
        assert result[1] < 0.01
        assert result[2] < 0.01

    def test_sums_to_one(self):
        result = _softmax(np.array([-1.5, 2.3, 0.7]))
        assert abs(result.sum() - 1.0) < 1e-7

    def test_numerical_stability_large_values(self):
        result = _softmax(np.array([1000.0, 1000.0, 1000.0]))
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-7)

    def test_negative_logits(self):
        result = _softmax(np.array([-5.0, -5.0, 0.0]))
        assert result[2] > result[0]
        assert abs(result.sum() - 1.0) < 1e-7


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_label_map_keys(self):
        assert set(LABEL_MAP.keys()) == {"HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"}

    def test_id2label_roundtrip(self):
        for name, idx in LABEL_MAP.items():
            assert ID2LABEL[idx] == name

    def test_num_labels(self):
        assert NUM_LABELS == 3


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_chunk_size(self):
        assert CHUNK_SIZE == 512

    def test_stride(self):
        assert STRIDE == 256


# ---------------------------------------------------------------------------
# PredictionResult
# ---------------------------------------------------------------------------

class TestPredictionResult:
    def test_creation(self):
        pr = PredictionResult(
            predicted_label="HIGH_RISK",
            probabilities={"HIGH_RISK": 0.7, "MEDIUM_RISK": 0.2, "LOW_RISK": 0.1},
            confidence=0.7,
            num_chunks=3,
            chunk_predictions=[],
        )
        assert pr.predicted_label == "HIGH_RISK"
        assert pr.confidence == 0.7
        assert pr.num_chunks == 3
        assert pr.ground_truth is None

    def test_with_ground_truth(self):
        pr = PredictionResult(
            predicted_label="LOW_RISK",
            probabilities={"HIGH_RISK": 0.1, "MEDIUM_RISK": 0.1, "LOW_RISK": 0.8},
            confidence=0.8,
            num_chunks=1,
            chunk_predictions=[],
            ground_truth="LOW_RISK",
        )
        assert pr.ground_truth == "LOW_RISK"


# ---------------------------------------------------------------------------
# _create_chunks (requires mocking tokenizer — tested via LegalBERTPredictor)
# ---------------------------------------------------------------------------

class TestCreateChunks:
    """Test chunking by directly instantiating the predictor with a mock tokenizer."""

    @pytest.fixture
    def predictor(self):
        from backend.risk_predictor import LegalBERTPredictor

        p = LegalBERTPredictor()
        # Inject a fake tokenizer instead of loading the real model
        tok = type("FakeTok", (), {
            "cls_token_id": 101,
            "sep_token_id": 102,
            "pad_token_id": 0,
            "__call__": lambda self, text, **kw: {
                "input_ids": list(range(1, len(text.split()) + 1))
            },
        })()
        p._tokenizer = tok
        p._device = "cpu"
        return p

    def test_short_text_single_chunk(self, predictor):
        # Short text → single chunk with padding
        text = " ".join(["word"] * 100)
        chunks = predictor._create_chunks(text)
        assert len(chunks) == 1
        assert len(chunks[0]["input_ids"]) == CHUNK_SIZE
        assert chunks[0]["input_ids"][0] == 101  # [CLS]
        # [SEP] at position 101 (CLS + 100 tokens)
        assert chunks[0]["input_ids"][101] == 102
        # Remaining should be padding
        assert all(x == 0 for x in chunks[0]["input_ids"][102:])

    def test_long_text_multiple_chunks(self, predictor):
        # 1000 tokens → should need multiple chunks (max_content = 510)
        text = " ".join(["word"] * 1000)
        chunks = predictor._create_chunks(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk["input_ids"]) == CHUNK_SIZE
            assert len(chunk["attention_mask"]) == CHUNK_SIZE
            assert chunk["input_ids"][0] == 101  # [CLS]

    def test_attention_mask_matches_content(self, predictor):
        text = " ".join(["word"] * 50)
        chunks = predictor._create_chunks(text)
        ids = chunks[0]["input_ids"]
        mask = chunks[0]["attention_mask"]
        # Non-pad tokens should have mask=1
        for i, (tok, m) in enumerate(zip(ids, mask)):
            if tok != 0:
                assert m == 1, f"Token {tok} at position {i} should have mask=1"

    def test_empty_text(self, predictor):
        # Edge case: empty text
        chunks = predictor._create_chunks("")
        assert len(chunks) == 1  # should still produce a chunk with [CLS][SEP]
