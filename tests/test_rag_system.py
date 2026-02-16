"""Tests for backend.rag_system — intent routing, formatting, keyword matching."""

import pytest
from collections import Counter
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
                       metadata={"court": "Nacka TR", "date": "2024-01-10", "case_number": "M 1-22", "processing_time_days": 100, "application_outcome": "granted"},
                       scoring_details={"outcome_desc": "Tillstånd", "domslut_measures": ["Fiskväg"], "max_cost_sek": 500000}),
        _make_decision(id="m2-22", label="LOW_RISK",
                       metadata={"court": "Växjö TR", "date": "2023-06-01", "case_number": "M 2-22", "processing_time_days": 50, "application_outcome": "denied"},
                       scoring_details={"outcome_desc": "Avslag", "domslut_measures": ["Minimitappning"]}),
        _make_decision(id="m3-22", label="LOW_RISK",
                       metadata={"court": "Nacka TR", "date": "2024-03-20", "case_number": "M 3-22", "processing_time_days": 75, "application_outcome": "granted"},
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
    data_loader.get_outcomes_by_court.return_value = {
        "Nacka TR": Counter({"granted": 2, "denied": 1}),
        "Växjö TR": Counter({"denied": 1, "appeal_denied": 1}),
    }
    data_loader.get_outcome_distribution.return_value = Counter({"granted": 2, "denied": 2})
    data_loader.get_processing_times.return_value = [("M 1-22", 100), ("M 2-22", 50), ("M 3-22", 75)]

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

    def test_case_number_query_uses_direct_lookup(self, rag):
        """When user asks about a specific case number, use direct data lookup."""
        rag.llm = MagicMock()
        rag.llm.generate_response.return_value = "Svar om mål m1-22"
        rag.search.find_similar_decisions.return_value = []
        response = rag.generate_response("vad hände i m1-22")
        # Should use direct lookup, not fall through to search
        rag.data.get_decision.assert_called()

    def test_recent_decisions_with_court_filter(self, rag):
        response = rag.generate_response("senaste besluten vid växjö")
        assert "Växjö" in response

    def test_recent_decisions_with_count(self, rag):
        response = rag.generate_response("3 senaste besluten")
        # Should contain at most 3 decision lines (each line has **M x-22**)
        bullet_lines = [l for l in response.split("\n") if l.strip().startswith("- ") and "**M " in l]
        assert len(bullet_lines) <= 3

    def test_rankings_denial_rate(self, rag):
        response = rag.generate_response("vilken domstol nekar flest")
        assert "avslag" in response or "nekande" in response
        assert "Nacka" in response or "Växjö" in response


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
        rag.data.get_labeled_decisions.return_value = []
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
# New intent handlers (outcomes, processing times, power plants, watercourses)
# ---------------------------------------------------------------------------

class TestNewIntentHandlers:
    def test_outcome_query(self, rag):
        response = rag.generate_response("Visa utfallsfördelning")
        assert "Utfall" in response or "granted" in response

    def test_processing_time_query(self, rag):
        response = rag.generate_response("Hur lång handläggningstid?")
        assert "Handläggningstider" in response or "dagar" in response

    def test_power_plant_query(self, rag):
        rag.data.get_power_plants.return_value = [("M 1-22", "Stora Mölla")]
        response = rag.generate_response("Vilka kraftverk finns?")
        assert "Kraftverk" in response or "Stora Mölla" in response

    def test_power_plant_empty(self, rag):
        rag.data.get_power_plants.return_value = []
        response = rag._format_power_plants()
        assert "Ingen kraftverksinformation" in response

    def test_watercourse_query(self, rag):
        rag.data.get_watercourses.return_value = ["Ume älv", "Testeboån"]
        response = rag.generate_response("Vilka vattendrag?")
        assert "Vattendrag" in response or "Ume" in response

    def test_watercourse_empty(self, rag):
        rag.data.get_watercourses.return_value = []
        response = rag._format_watercourses()
        assert "Ingen vattendragsinformation" in response

    def test_processing_time_format(self, rag):
        response = rag._format_processing_times()
        assert "Handläggningstider" in response
        assert "dagar" in response

    def test_processing_time_empty(self, rag):
        rag.data.get_all_decisions.return_value = []
        response = rag._format_processing_times()
        assert "Ingen handläggningstidsdata" in response

    def test_outcome_format(self, rag):
        response = rag._format_outcomes()
        assert "Utfall" in response

    def test_outcome_empty(self, rag):
        rag.data.get_outcome_distribution.return_value = Counter()
        response = rag._format_outcomes()
        assert "Ingen utfallsdata" in response

    def test_rankings_measure(self, rag):
        response = rag._format_rankings("vanligaste åtgärd")
        assert "åtgärder" in response.lower() or "Ranking" in response

    def test_rankings_cost(self, rag):
        response = rag._format_rankings("dyraste kostnad")
        assert "Ranking" in response or "Dyraste" in response

    def test_cost_with_filters(self, rag):
        response = rag.generate_response("Vad kostar fiskväg i kronor vid Nacka?")
        assert "Kostnad" in response or "kr" in response

    def test_statistics_with_court_filter(self, rag):
        response = rag.generate_response("Visa statistik vid Nacka")
        assert "Statistik" in response and "Nacka" in response

    def test_recent_with_risk_filter(self, rag):
        response = rag.generate_response("senaste besluten med hög risk")
        assert "Senaste" in response or "Hög risk" in response


# ---------------------------------------------------------------------------
# Filter extraction helpers
# ---------------------------------------------------------------------------

class TestFilterExtraction:
    def test_extract_court_filter(self):
        from backend.rag_system import _extract_court_filter
        assert _extract_court_filter("beslut vid Nacka") == "Nacka"
        assert _extract_court_filter("Växjö domstol") == "Växjö"
        assert _extract_court_filter("inget relevant") is None

    def test_extract_count(self):
        from backend.rag_system import _extract_count
        assert _extract_count("3 senaste") == 3
        assert _extract_count("10 senaste") == 10
        assert _extract_count("senaste besluten") == 5  # default

    def test_extract_outcome_filter(self):
        from backend.rag_system import _extract_outcome_filter
        assert _extract_outcome_filter("nekade beslut") == "denied"
        assert _extract_outcome_filter("beviljat tillstånd") == "granted"
        assert _extract_outcome_filter("inget") is None

    def test_extract_measure_filter(self):
        from backend.rag_system import _extract_measure_filter
        assert _extract_measure_filter("fiskväg") == "fiskväg"
        assert _extract_measure_filter("minimitappning") == "minimitappning"
        assert _extract_measure_filter("inget") is None

    def test_extract_risk_filter(self):
        from backend.rag_system import _extract_risk_filter
        assert _extract_risk_filter("hög risk beslut") == "HIGH_RISK"
        assert _extract_risk_filter("låg risk") == "LOW_RISK"
        assert _extract_risk_filter("inget") is None

    def test_apply_filters_by_court(self):
        from backend.rag_system import _apply_filters
        from tests.conftest import _make_decision
        decisions = [
            _make_decision(id="m1", metadata={"court": "Nacka TR", "date": "2024-01-01", "case_number": "M 1"}),
            _make_decision(id="m2", metadata={"court": "Växjö TR", "date": "2024-01-01", "case_number": "M 2"}),
        ]
        filtered = _apply_filters(decisions, court_filter="Nacka")
        assert len(filtered) == 1
        assert filtered[0].id == "m1"

    def test_apply_filters_by_outcome(self):
        from backend.rag_system import _apply_filters
        from tests.conftest import _make_decision
        decisions = [
            _make_decision(id="m1", metadata={"court": "Nacka TR", "date": "2024-01-01", "case_number": "M 1", "application_outcome": "denied"}),
            _make_decision(id="m2", metadata={"court": "Växjö TR", "date": "2024-01-01", "case_number": "M 2", "application_outcome": "granted"}),
        ]
        denied = _apply_filters(decisions, outcome_filter="denied")
        assert len(denied) == 1
        assert denied[0].id == "m1"


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


# ---------------------------------------------------------------------------
# Advisory routing & custom risk assessment
# ---------------------------------------------------------------------------

@pytest.fixture
def rag_with_llm(rag):
    """RAG system with a mocked LLM engine attached."""
    llm = MagicMock()
    llm.generate_response.return_value = "LLM-genererat svar"
    rag.llm = llm
    rag.router = MagicMock()
    rag.router.route.return_value = "Router-genererat svar"
    return rag


class TestAdvisoryRouting:
    def test_advisory_query_goes_to_llm(self, rag_with_llm):
        """Advisory queries should NOT trigger template responses."""
        response = rag_with_llm.generate_response("vilka åtgärder är mest kostnadseffektiva?")
        # Should NOT be the static measures list (contains "utskov: " pattern)
        assert "utskov: " not in response

    def test_personal_query_goes_to_assessment(self, rag_with_llm):
        """Personal queries with 'jag har ett' should go to assessment (checked before advisory)."""
        mock_result = MagicMock()
        mock_result.decision_id = "m1-22"
        mock_result.metadata = {"court": "Nacka TR", "case_number": "M 1-22"}
        mock_result.similarity = 0.85
        rag_with_llm.search.search.return_value = [mock_result]
        rag_with_llm.data.get_watercourses.return_value = []
        response = rag_with_llm.generate_response("jag har ett kraftverk, vad ska jag göra?")
        # Should NOT be the power plant listing — should go to assessment handler
        assert "Kraftverk (" not in response

    def test_advisory_routes_to_router(self, rag_with_llm):
        """Advisory queries should be routed via the multi-agent router."""
        rag_with_llm.generate_response("vilka åtgärder är mest kostnadseffektiva?")
        rag_with_llm.router.route.assert_called_once()

    def test_analytical_query_goes_to_llm(self, rag_with_llm):
        """Questions asking for explanation should go to LLM."""
        response = rag_with_llm.generate_response("varför kräver domstolen fiskväg?")
        assert "Kraftverk (" not in response
        rag_with_llm.router.route.assert_called_once()

    def test_non_advisory_still_uses_templates(self, rag_with_llm):
        """Non-advisory queries should still use template responses."""
        response = rag_with_llm.generate_response("Visa hog risk beslut")
        assert "Hög risk" in response or "HIGH_RISK" in response
        rag_with_llm.router.route.assert_not_called()


class TestCustomRiskAssessment:
    def test_custom_assessment_detected(self, rag_with_llm):
        """Custom risk assessment queries should trigger the assessment handler."""
        # Set up search to return some results
        mock_result = MagicMock()
        mock_result.decision_id = "m1-22"
        mock_result.metadata = {"court": "Nacka TR", "case_number": "M 1-22"}
        mock_result.similarity = 0.85
        rag_with_llm.search.search.return_value = [mock_result]
        rag_with_llm.data.get_watercourses.return_value = []

        response = rag_with_llm.generate_response("Jag har ett medelstort vattenkraftverk i Gävle")
        # "Jag har ett" is an assessment signal → triggers _custom_risk_assessment
        assert "Kraftverk (" not in response
        # Assessment handler calls LLM
        rag_with_llm.llm.generate_response.assert_called_once()

    def test_assessment_query_calls_llm(self, rag_with_llm):
        """Assessment should call the LLM with specialized prompt."""
        mock_result = MagicMock()
        mock_result.decision_id = "m1-22"
        mock_result.metadata = {"court": "Nacka TR", "case_number": "M 1-22"}
        mock_result.similarity = 0.85
        rag_with_llm.search.search.return_value = [mock_result]
        rag_with_llm.data.get_watercourses.return_value = []

        # "mitt kraftverk" + "bedöm min risk" triggers assessment (checked before advisory)
        response = rag_with_llm.generate_response("Bedöm min risk för mitt kraftverk")
        # Assessment check runs first, catches "mitt kraftverk" → calls _custom_risk_assessment
        assert response is not None
        rag_with_llm.llm.generate_response.assert_called_once()

    def test_assessment_without_llm_falls_through(self, rag):
        """Without LLM, assessment queries should fall through to templates."""
        rag.data.get_power_plants.return_value = [("M 1-22", "Testverket")]
        response = rag.generate_response("mitt kraftverk behöver bedömning")
        # Without LLM, the advisory/assessment checks are skipped
        # and template matching takes over
        assert response is not None

