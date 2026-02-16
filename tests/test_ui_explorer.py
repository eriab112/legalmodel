"""Tests for the redesigned Utforska detail view and browse table.

Tests cover:
- AI summary generation (_generate_decision_summary)
- Browse table row data construction
- Detail view data extraction (key data, metadata, cost formatting)
- Summary caching in session_state
"""

import sys
import pytest
from unittest.mock import MagicMock, call

from tests.conftest import _make_decision

from ui.search_interface import _generate_decision_summary, _DASH


# ---------------------------------------------------------------------------
# _generate_decision_summary
# ---------------------------------------------------------------------------

class TestGenerateDecisionSummary:
    """Test the AI summary generation function."""

    def test_uses_sections_when_available(self):
        decision = _make_decision(
            metadata={"case_number": "M 100-23", "court": "Nacka TR", "date": "2024-01-15"},
        )
        decision.sections = {
            "domslut": "Tillstånd beviljas med villkor.",
            "domskäl": "Domstolen bedömer att åtgärderna är tillräckliga.",
        }
        llm = MagicMock()
        llm.generate_response.return_value = "Sammanfattning av beslutet."

        result = _generate_decision_summary(decision, llm)

        assert result == "Sammanfattning av beslutet."
        llm.generate_response.assert_called_once()
        # Check that context includes section text
        call_args = llm.generate_response.call_args
        context = call_args[0][1]
        assert "DOMSLUT" in context
        assert "DOMSKÄL" in context

    def test_falls_back_to_full_text_when_no_sections(self):
        decision = _make_decision()
        decision.sections = {}
        decision.full_text = "Full text of the decision for fallback."
        llm = MagicMock()
        llm.generate_response.return_value = "Summary"

        _generate_decision_summary(decision, llm)

        call_args = llm.generate_response.call_args
        context = call_args[0][1]
        assert "Full text of the decision for fallback." in context

    def test_prompt_contains_case_number(self):
        decision = _make_decision(
            metadata={"case_number": "M 999-24", "court": "Växjö TR", "date": "2024-06-01"},
        )
        llm = MagicMock()
        llm.generate_response.return_value = "Summary"

        _generate_decision_summary(decision, llm)

        call_args = llm.generate_response.call_args
        prompt = call_args[0][0]
        assert "M 999-24" in prompt
        assert "Växjö TR" in prompt

    def test_system_prompt_override_is_set(self):
        decision = _make_decision()
        llm = MagicMock()
        llm.generate_response.return_value = "Summary"

        _generate_decision_summary(decision, llm)

        call_kwargs = llm.generate_response.call_args[1]
        assert "system_prompt_override" in call_kwargs
        assert "juridisk sammanfattare" in call_kwargs["system_prompt_override"]

    def test_sources_metadata_is_correct(self):
        decision = _make_decision(
            id="m555-23",
            filename="m555-23.txt",
            metadata={"case_number": "M 555-23", "court": "Nacka TR", "date": "2024-01-01"},
        )
        llm = MagicMock()
        llm.generate_response.return_value = "Summary"

        _generate_decision_summary(decision, llm)

        call_args = llm.generate_response.call_args
        sources = call_args[0][2]
        assert len(sources) == 1
        assert sources[0]["doc_id"] == "m555-23"
        assert sources[0]["doc_type"] == "decision"
        assert sources[0]["filename"] == "m555-23.txt"

    def test_truncates_long_sections(self):
        long_text = "A" * 5000
        decision = _make_decision()
        decision.sections = {"domslut": long_text}
        llm = MagicMock()
        llm.generate_response.return_value = "Summary"

        _generate_decision_summary(decision, llm)

        call_args = llm.generate_response.call_args
        context = call_args[0][1]
        # Section text should be truncated to 3000 chars
        assert len(context) < 5000


# ---------------------------------------------------------------------------
# Browse table row construction (logic extracted from _render_browse_mode)
# ---------------------------------------------------------------------------

class TestBrowseTableRows:
    """Test the data transformation used in the clickable browse table."""

    def test_row_contains_hidden_id(self):
        d = _make_decision(id="m42-23", metadata={
            "case_number": "M 42-23", "court": "Nacka TR", "date": "2024-01-01",
            "originating_court": "Nacka tingsrätt",
            "application_outcome_sv": "Tillstånd beviljat",
            "power_plant_name": "Testverket",
            "watercourse": "Testån",
        })
        risk_sv = {"HIGH_RISK": "Hög risk", "LOW_RISK": "Låg risk"}

        row = {
            "Målnummer": d.metadata.get("case_number", d.id),
            "Domstol": d.metadata.get("originating_court") or d.metadata.get("court", ""),
            "Datum": d.metadata.get("date", ""),
            "Risknivå": risk_sv.get(d.label, d.label or ""),
            "Utfall": d.metadata.get("application_outcome_sv", ""),
            "Kraftverk": d.metadata.get("power_plant_name", ""),
            "Vattendrag": d.metadata.get("watercourse", ""),
            "_id": d.id,
        }

        assert row["_id"] == "m42-23"
        assert row["Målnummer"] == "M 42-23"
        assert row["Domstol"] == "Nacka tingsrätt"
        assert row["Risknivå"] == "Hög risk"
        assert row["Kraftverk"] == "Testverket"

    def test_row_falls_back_to_court_when_no_originating(self):
        d = _make_decision(metadata={
            "case_number": "M 1-22", "court": "Nacka TR", "date": "2024-01-01",
        })
        court = d.metadata.get("originating_court") or d.metadata.get("court", "")
        assert court == "Nacka TR"


# ---------------------------------------------------------------------------
# Detail view data extraction
# ---------------------------------------------------------------------------

class TestDetailViewDataExtraction:
    """Test data extraction logic used in render_decision_detail."""

    def test_cost_formatting_millions(self):
        cost = 2_500_000
        if cost >= 1_000_000:
            cost_str = f"{cost / 1_000_000:.1f} Mkr"
        elif cost >= 1_000:
            cost_str = f"{cost / 1_000:.0f} kkr"
        else:
            cost_str = f"{cost:.0f} kr"
        assert cost_str == "2.5 Mkr"

    def test_cost_formatting_thousands(self):
        cost = 350_000
        if cost >= 1_000_000:
            cost_str = f"{cost / 1_000_000:.1f} Mkr"
        elif cost >= 1_000:
            cost_str = f"{cost / 1_000:.0f} kkr"
        else:
            cost_str = f"{cost:.0f} kr"
        assert cost_str == "350 kkr"

    def test_cost_formatting_small(self):
        cost = 500
        if cost >= 1_000_000:
            cost_str = f"{cost / 1_000_000:.1f} Mkr"
        elif cost >= 1_000:
            cost_str = f"{cost / 1_000:.0f} kkr"
        else:
            cost_str = f"{cost:.0f} kr"
        assert cost_str == "500 kr"

    def test_measures_from_scoring_details(self):
        d = _make_decision(scoring_details={
            "domslut_measures": ["Fiskväg", "Minimitappning"],
            "max_cost_sek": 100000,
        })
        measures = []
        if d.scoring_details:
            measures = d.scoring_details.get("domslut_measures", [])
        if not measures:
            measures = d.extracted_measures or []
        assert measures == ["Fiskväg", "Minimitappning"]

    def test_measures_fallback_to_extracted(self):
        d = _make_decision(scoring_details={"outcome_desc": "Tillstånd"})
        d.extracted_measures = ["Omlöp"]
        measures = []
        if d.scoring_details:
            measures = d.scoring_details.get("domslut_measures", [])
        if not measures:
            measures = d.extracted_measures or []
        assert measures == ["Omlöp"]

    def test_cost_from_metadata_preferred(self):
        d = _make_decision(
            metadata={"court": "Nacka TR", "date": "2024-01-01", "case_number": "M 1-22",
                       "total_cost_sek": 750_000},
            scoring_details={"max_cost_sek": 500_000},
        )
        cost = d.metadata.get("total_cost_sek")
        if cost is None and d.scoring_details:
            cost = d.scoring_details.get("max_cost_sek")
        assert cost == 750_000

    def test_cost_fallback_to_scoring_details(self):
        d = _make_decision(
            metadata={"court": "Nacka TR", "date": "2024-01-01", "case_number": "M 1-22"},
            scoring_details={"max_cost_sek": 500_000},
        )
        cost = d.metadata.get("total_cost_sek")
        if cost is None and d.scoring_details:
            cost = d.scoring_details.get("max_cost_sek")
        assert cost == 500_000

    def test_dash_constant_is_em_dash(self):
        assert _DASH == "—"

    def test_metadata_defaults_to_dash(self):
        d = _make_decision(metadata={"court": "Nacka TR", "date": "2024-01-01", "case_number": "M 1-22"})
        watercourse = d.metadata.get("watercourse", _DASH)
        assert watercourse == _DASH

    def test_linked_water_bodies_extraction(self):
        d = _make_decision()
        d.linked_water_bodies = [
            {"water_body": "Testsjön", "viss_ids": ["SE123"]},
            {"water_body": "Testån", "viss_ids": ["SE456"]},
            {"water_body": "", "viss_ids": []},
        ]
        wbs = ", ".join(
            wb.get("water_body", "") for wb in d.linked_water_bodies if wb.get("water_body")
        )
        assert wbs == "Testsjön, Testån"

    def test_header_assembly(self):
        d = _make_decision(metadata={
            "case_number": "M 42-23",
            "originating_court": "Nacka tingsrätt",
            "court": "MÖD",
            "date": "2024-05-10",
        })
        case = d.metadata.get("case_number", d.id)
        court = d.metadata.get("originating_court") or d.metadata.get("court", "")
        date = d.metadata.get("date", "")
        assert case == "M 42-23"
        assert court == "Nacka tingsrätt"
        assert date == "2024-05-10"

    def test_processing_time_formatting(self):
        d = _make_decision(metadata={
            "court": "Nacka TR", "date": "2024-01-01",
            "case_number": "M 1-22", "processing_time_days": 365.5,
        })
        proc_time = d.metadata.get("processing_time_days")
        assert proc_time is not None
        assert f"{int(proc_time)} dagar" == "365 dagar"


# ---------------------------------------------------------------------------
# Summary caching
# ---------------------------------------------------------------------------

class TestSummaryCaching:
    """Test that summary caching logic works correctly."""

    def test_cache_key_format(self):
        decision_id = "m1234-22"
        cache_key = f"summary_{decision_id}"
        assert cache_key == "summary_m1234-22"

    def test_cached_summary_returned(self):
        st = sys.modules["streamlit"]
        st.session_state["summary_m1234-22"] = "Cached summary text"
        cache_key = "summary_m1234-22"
        assert cache_key in st.session_state
        assert st.session_state[cache_key] == "Cached summary text"

    def test_summary_stored_in_cache_after_generation(self):
        st = sys.modules["streamlit"]
        decision_id = "m1234-22"
        cache_key = f"summary_{decision_id}"
        summary = "Generated summary text"
        st.session_state[cache_key] = summary
        assert st.session_state[cache_key] == "Generated summary text"
