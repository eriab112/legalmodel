"""Tests for backend.rag_system — intent routing, formatting, keyword matching."""

import pytest
from unittest.mock import MagicMock

from backend.rag_system import RAGSystem, RISK_LABELS_SV, RISK_EMOJI, _matches, _matches_word
from backend.risk_predictor import PredictionResult
from tests.conftest import _make_decision


# ---------------------------------------------------------------------------
# _matches / _matches_word helpers
# ---------------------------------------------------------------------------

class TestMatches:
    def test_matches_found(self):
        assert _matches("visa hog risk beslut", ["hog risk"])

    def test_matches_not_found(self):
        assert not _matches("hej hopp", ["hog risk"])

    def test_matches_multiple_keywords(self):
        assert _matches("visa statistik", ["statistik", "statistics"])

    def test_matches_empty_keywords(self):
        assert not _matches("hello", [])

    def test_matches_word_true(self):
        assert _matches_word("visa kr per beslut", "kr")

    def test_matches_word_false_substring(self):
        # "kr" inside "kronor" should NOT match as whole word
        assert not _matches_word("kronor totalt", "kr")

    def test_matches_word_at_end(self):
        assert _matches_word("kostnad 500 kr", "kr")


# ---------------------------------------------------------------------------
# RAGSystem setup
# ---------------------------------------------------------------------------

@pytest.fixture
def rag():
    data_loader = MagicMock()
    search_engine = MagicMock()
    predictor = MagicMock()

    decisions = [
        _make_decision(id="m1-22", label="HIGH_RISK",
                       metadata={"court": "Nacka TR", "date": "2024-01-10", "case_number": "M 1-22"},
                       scoring_details={"outcome_desc": "Tillstånd", "domslut_measures": ["Fiskväg"], "max_cost_sek": 500000}),
        _make_decision(id="m2-22", label="LOW_RISK",
                       metadata={"court": "Växjö TR", "date": "2023-06-01", "case_number": "M 2-22"},
                       scoring_details={"outcome_desc": "Avslag", "domslut_measures": ["Minimitappning"]}),
        _make_decision(id="m3-22", label="LOW_RISK",
                       metadata={"court": "Nacka TR", "date": "2024-03-20", "case_number": "M 3-22"},
                       scoring_details={"outcome_desc": "Tillstånd", "domslut_measures": []}),
    ]

    data_loader.get_decisions_by_label.side_effect = lambda label: [d for d in decisions if d.label == label]
    data_loader.get_labeled_decisions.return_value = [d for d in decisions if d.label]
    data_loader.get_all_decisions.return_value = decisions
    data_loader.get_label_distribution.return_value = {"HIGH_RISK": 1, "LOW_RISK": 2}
    data_loader.get_measure_frequency.return_value = {"Fiskväg": 2, "Minimitappning": 1}
    data_loader.get_courts.return_value = ["Nacka TR", "Växjö TR"]
    data_loader.get_date_range.return_value = ("2023-06-01", "2024-03-20")
    data_loader.get_decision.side_effect = lambda did: next((d for d in decisions if d.id == did), None)

    search_engine.search.return_value = []

    return RAGSystem(data_loader, search_engine, predictor)


# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

class TestIntentRouting:
    def test_high_risk_query(self, rag):
        response = rag.generate_response("Visa hog risk beslut")
        assert "Hög risk" in response or "HIGH_RISK" in response

    def test_low_risk_query(self, rag):
        response = rag.generate_response("Vilka har lag risk?")
        assert "Låg risk" in response or "LOW_RISK" in response

    def test_medium_risk_query_empty(self, rag):
        response = rag.generate_response("medel risk beslut")
        assert "Inga beslut" in response

    def test_risk_prediction_intent(self, rag):
        rag.predictor.predict_decision.return_value = PredictionResult(
            predicted_label="HIGH_RISK",
            probabilities={"HIGH_RISK": 0.9, "LOW_RISK": 0.1},
            confidence=0.9,
            num_chunks=5,
            chunk_predictions=[],
        )
        response = rag.generate_response("Analysera M 1-22")
        assert "Riskbedömning" in response or "HIGH_RISK" in response or "Hög risk" in response

    def test_distribution_query(self, rag):
        response = rag.generate_response("Visa riskfordelning")
        assert "Riskfördelning" in response

    def test_measures_query(self, rag):
        response = rag.generate_response("Vanligaste atgarder")
        assert "åtgärder" in response.lower() or "Fiskväg" in response

    def test_recent_query(self, rag):
        response = rag.generate_response("Visa senaste besluten")
        assert "Senaste" in response

    def test_statistics_query(self, rag):
        response = rag.generate_response("Visa statistik")
        assert "Statistik" in response

    def test_cost_query(self, rag):
        response = rag.generate_response("Vad kostar det i kronor?")
        assert "Kostnad" in response or "kr" in response

    def test_fallback_to_search(self, rag):
        rag.search.search.return_value = []
        response = rag.generate_response("Berätta om Ume älv")
        assert "Inga relevanta" in response or "Sökresultat" in response


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_risk_response_empty(self, rag):
        rag.data.get_decisions_by_label.side_effect = None
        rag.data.get_decisions_by_label.return_value = []
        response = rag._format_risk_response("HIGH_RISK")
        assert "Inga beslut" in response

    def test_distribution_format(self, rag):
        response = rag._format_distribution()
        assert "Totalt" in response
        assert "3" in response  # total still 3 decisions

    def test_measures_empty(self, rag):
        rag.data.get_measure_frequency.return_value = {}
        response = rag._format_measures()
        assert "Inga åtgärder" in response

    def test_comparison_needs_two_ids(self, rag):
        response = rag._handle_comparison("jamfor M 1234-22")
        assert "två målnummer" in response or "målnummer" in response

    def test_comparison_with_two_ids(self, rag):
        response = rag._handle_comparison("jamfor m1-22 med m2-22")
        assert "Jämförelse" in response or "Kunde inte" in response

    def test_cost_info_empty(self, rag):
        rag.data.get_labeled_decisions.return_value = []
        response = rag._format_cost_info()
        assert "Ingen kostnadsinformation" in response

    def test_search_response_no_results(self, rag):
        rag.search.search.return_value = []
        response = rag._format_search_response("test query")
        assert "Inga relevanta" in response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestRagConstants:
    def test_risk_labels_sv(self):
        assert "HIGH_RISK" in RISK_LABELS_SV
        assert "LOW_RISK" in RISK_LABELS_SV

    def test_risk_emoji(self):
        assert "HIGH_RISK" in RISK_EMOJI
        assert "LOW_RISK" in RISK_EMOJI
