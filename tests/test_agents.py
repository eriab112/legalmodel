"""Tests for backend.agents — query classification, routing, agent responses."""

import pytest
from unittest.mock import MagicMock, patch

from backend.agents import (
    AgentDomain,
    AgentResponse,
    DomainAgent,
    MultiAgentRouter,
    classify_legislation_domain,
    classify_query,
)


class TestClassifyLegislationDomain:
    def test_eu_documents(self):
        assert classify_legislation_domain("CIS_Guidance_Article_4_7_FINAL.pdf") == AgentDomain.EU_LAW
        assert classify_legislation_domain("eudirektiv.pdf") == AgentDomain.EU_LAW
        assert classify_legislation_domain("GD 07 - Monitoring - Policy Summary.pdf") == AgentDomain.EU_LAW
        assert classify_legislation_domain("Guidance No 4 - heavily modified water bodies - HMWB (WG 2.2).pdf") == AgentDomain.EU_LAW

    def test_swedish_documents(self):
        assert classify_legislation_domain("miljobalken.pdf") == AgentDomain.SWEDISH_LAW
        assert classify_legislation_domain("NAP_full_text.txt") == AgentDomain.SWEDISH_LAW
        assert classify_legislation_domain("VM-riktlinjer-vattenkraft_atgarder-undantag.pdf") == AgentDomain.SWEDISH_LAW
        assert classify_legislation_domain("bilaga-5-5-losningar-uppstromsvandrande-fisk.pdf") == AgentDomain.SWEDISH_LAW


class TestClassifyQuery:
    def test_court_query(self):
        assert classify_query("Vad beslutade domstolen i M 3753-22?") == AgentDomain.COURT

    def test_swedish_law_query(self):
        assert classify_query("Vad säger miljöbalken om vattenverksamhet?") == AgentDomain.SWEDISH_LAW

    def test_eu_query(self):
        assert classify_query("Vad kräver vattendirektivet artikel 4.7?") == AgentDomain.EU_LAW

    def test_multi_domain_query(self):
        # Mentions both court and law
        result = classify_query("Hur tolkar domstolen miljöbalkens krav på fiskväg?")
        assert result == AgentDomain.MULTI

    def test_ambiguous_defaults_to_multi(self):
        result = classify_query("Berätta om vattenkraftens framtid")
        assert result == AgentDomain.MULTI

    def test_risk_signals_court(self):
        assert classify_query("Vilka beslut har hög risk?") == AgentDomain.COURT

    def test_technical_signals_swedish(self):
        assert classify_query("Hur designar man en fiskpassage?") == AgentDomain.SWEDISH_LAW

    def test_cis_signals_eu(self):
        result = classify_query("Vad säger CIS guidance om HMWB?")
        assert result == AgentDomain.EU_LAW

    def test_case_number_routes_to_court(self):
        assert classify_query("vad hände i m 483-22") == AgentDomain.COURT
        assert classify_query("berätta om M 3753-22") == AgentDomain.COURT
        assert classify_query("M 605-24 utfall") == AgentDomain.COURT


class TestDomainAgent:
    @pytest.fixture
    def agent(self):
        search = MagicMock()
        search.search.return_value = []
        llm = MagicMock()
        return DomainAgent(
            domain=AgentDomain.COURT,
            search_engine=search,
            llm_engine=llm,
            doc_type_filters=["decision"],
        )

    def test_retrieve_empty(self, agent):
        results = agent.retrieve("test query")
        assert results == []

    def test_generate_empty_results(self, agent):
        response = agent.generate("test query")
        assert isinstance(response, AgentResponse)
        assert response.sources_used == 0
        assert "Inga relevanta" in response.content

    def test_generate_calls_search_with_doc_type(self, agent):
        agent.generate("test")
        agent.search.search.assert_called_with(
            "test", n_results=8, doc_type_filter="decision"
        )


class TestMultiAgentRouter:
    @pytest.fixture
    def router(self):
        search = MagicMock()
        search.search.return_value = []
        llm = None  # Test without LLM
        return MultiAgentRouter(search, llm)

    def test_route_court_query(self, router):
        response = router.route("Vad beslutade domstolen?")
        assert "Domstolsagent" in response or "Inga relevanta" in response

    def test_route_swedish_law_query(self, router):
        response = router.route("Vad säger miljöbalken om vattenverksamhet?")
        assert "Svensk rättsagent" in response or "Inga relevanta" in response

    def test_route_eu_query(self, router):
        response = router.route("Vad kräver vattendirektivet?")
        assert "EU-agent" in response or "Inga relevanta" in response

    def test_route_multi_domain(self, router):
        response = router.route("Hur tolkar domstolen miljöbalkens krav?")
        assert "Syntes" in response or "Inga relevanta" in response

    def test_agent_response_dataclass(self):
        r = AgentResponse(
            domain=AgentDomain.COURT,
            content="Test content",
            sources_used=3,
            doc_types_searched=["decision"],
        )
        assert r.domain == AgentDomain.COURT
        assert r.sources_used == 3
