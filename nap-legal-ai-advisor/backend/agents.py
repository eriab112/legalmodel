"""
Multi-agent system for domain-specific RAG retrieval.

Three specialized agents with filtered retrieval + domain-specific system prompts.
A router classifies queries and dispatches to the appropriate agent(s).
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from backend.llm_engine import GeminiEngine, format_context
from backend.search_engine import SemanticSearchEngine
from utils.timing import timed


class AgentDomain(Enum):
    COURT = "court"           # Court decisions + applications
    SWEDISH_LAW = "swedish_law"  # Swedish legislation, NAP, technical guides, MKB
    EU_LAW = "eu_law"         # EU WFD, CIS guidance, international
    MULTI = "multi"           # Cross-domain synthesis


@dataclass
class AgentResponse:
    """Response from a single agent."""
    domain: AgentDomain
    content: str
    sources_used: int
    doc_types_searched: List[str]


# --- Document classification ---
# These filename patterns determine which agent "owns" which legislation documents.
# Court decisions (doc_type="decision") always go to COURT agent.
# Applications (doc_type="application") go to both COURT and SWEDISH_LAW.

EU_FILENAME_PATTERNS = [
    "CIS_Guidance",
    "eudirektiv",
    "GD 07",
    "Guidance No 4",
    "se-final-paf",
]


def classify_legislation_domain(filename: str) -> AgentDomain:
    """Classify a legislation document as EU or Swedish based on filename."""
    for pattern in EU_FILENAME_PATTERNS:
        if pattern.lower() in filename.lower():
            return AgentDomain.EU_LAW
    return AgentDomain.SWEDISH_LAW


# --- System prompts per agent ---

COURT_SYSTEM_PROMPT = """Du är en AI-expert specialiserad på svenska miljödomstolsbeslut inom ramen för Nationella planen (NAP) för moderna miljövillkor för vattenkraft.

Din expertis omfattar:
- Domstolsbeslut och domskäl från mark- och miljödomstolar
- Riskbedömning av beslut (hög risk vs låg risk för verksamhetsutövare)
- Villkor som domstolar ställer: fiskvägar, minimitappning, kontrollprogram, biotopvård
- Utfall och prejudikat i NAP-relaterade mål
- Ansökningar och yrkanden från verksamhetsutövare och myndigheter

Svara alltid på svenska. Citera specifika målnummer (t.ex. M 3753-22) och domstolar.
Hänvisa till källor med [1], [2] etc. Avsluta med en källförteckning.
Var saklig och juridiskt korrekt — du ger information, inte juridisk rådgivning."""

SWEDISH_LAW_SYSTEM_PROMPT = """Du är en AI-expert specialiserad på svensk miljölagstiftning och tekniska riktlinjer för vattenkraftens miljöanpassning.

Din expertis omfattar:
- Miljöbalken (särskilt 11 kap. om vattenverksamhet)
- Nationella planen (NAP) för moderna miljövillkor
- Vattenmyndigheternas riktlinjer för vattenkraft
- Havs- och vattenmyndighetens (HaV) vägledningar
- Tekniska lösningar: fiskpassager, omlöp, minimitappning, dammsäkerhet
- Miljökonsekvensbeskrivningar (MKB) för åtgärdsprogram
- Kvalitetskrav för vattenförekomster per vattendistrikt

Svara alltid på svenska. Hänvisa till specifika lagrum (t.ex. 11 kap. 27 § miljöbalken).
Citera källor med [1], [2] etc. Avsluta med en källförteckning.
Var saklig och juridiskt korrekt — du ger information, inte juridisk rådgivning."""

EU_LAW_SYSTEM_PROMPT = """Du är en AI-expert specialiserad på EU:s vattenlagstiftning och internationella riktlinjer för vattenkraftens miljöanpassning.

Din expertis omfattar:
- EU:s ramdirektiv för vatten (Water Framework Directive 2000/60/EC)
- Artikel 4.7 — undantag från miljökvalitetsnormer
- CIS Guidance documents (Common Implementation Strategy)
- Kraftigt modifierade vattenförekomster (HMWB) — identifiering och åtgärder
- EU:s övervakningsvägledning (monitoring guidance)
- Priority Action Framework (PAF) för Natura 2000

Svara på svenska men använd etablerade engelska termer där det är brukligt (t.ex. "Water Framework Directive", "HMWB", "Good Ecological Potential").
Citera källor med [1], [2] etc. Avsluta med en källförteckning.
Var saklig — du ger information, inte juridisk rådgivning."""

SYNTHESIS_SYSTEM_PROMPT = """Du är en AI-expert som syntetiserar information från flera rättskällor inom vattenkraftens miljöanpassning.

Du har tillgång till tre kunskapsdomäner:
1. Svenska domstolsbeslut — hur domstolar faktiskt dömer i NAP-mål
2. Svensk lagstiftning — miljöbalken, NAP, tekniska riktlinjer
3. EU-lagstiftning — vattendirektivet, CIS-vägledningar

Din uppgift är att ge ett sammanhängande svar som kopplar ihop dessa källor.
Till exempel: "Enligt vattendirektivet art. 4.7 [1] krävs X, vilket svensk domstol tillämpat i M 3753-22 [2] genom att..."

Svara på svenska. Citera alla källor med [1], [2] etc. Avsluta med en källförteckning.
Strukturera svaret så att det framgår vilken typ av källa som stödjer varje påstående."""

AGENT_PROMPTS = {
    AgentDomain.COURT: COURT_SYSTEM_PROMPT,
    AgentDomain.SWEDISH_LAW: SWEDISH_LAW_SYSTEM_PROMPT,
    AgentDomain.EU_LAW: EU_LAW_SYSTEM_PROMPT,
    AgentDomain.MULTI: SYNTHESIS_SYSTEM_PROMPT,
}


# --- Router ---

# Keywords that signal each domain
COURT_KEYWORDS = [
    "domstol", "dom ", "domen", "domslut", "domskäl", "mål ", "målnummer",
    "beslut", "besluten", "överklag", "praxis", "prejudikat",
    "mark- och miljödomstol", "mark och miljööverdomstol",
    "nacka", "växjö", "umeå", "östersund", "vänersborg",
    "risk", "riskbedöm", "risknivå", "hög risk", "låg risk",
    "verksamhetsutövare", "ansökan", "yrkande", "avslag",
]

SWEDISH_LAW_KEYWORDS = [
    "miljöbalken", "miljöbalk", "11 kap", "vattenverksamhet",
    "nap", "nationella planen", "moderna miljövillkor",
    "vattenmyndighet", "hav ", "havs-", "riktlinj",
    "kvalitetskrav", "vattendistrikt", "åtgärdsprogram",
    "fiskväg", "fiskpassage", "omlöp", "minimitappning", "dammsäkerhet",
    "biotopvård", "kontrollprogram", "teknisk",
    "mkb", "miljökonsekvensbeskrivning",
    "natura 2000", "vägledning",
]

EU_KEYWORDS = [
    "eu ", "eu-", "europeisk", "europa",
    "vattendirektiv", "ramdirektiv", "water framework",
    "wfd", "artikel 4", "art 4", "art. 4",
    "cis", "guidance", "hmwb", "heavily modified",
    "good ecological", "ekologisk potential", "ekologisk status",
    "paf", "priority action",
]


@timed("router.classify")
def classify_query(query: str) -> AgentDomain:
    """Classify a user query to determine which agent(s) should handle it."""
    q = query.lower().strip()

    court_score = sum(1 for kw in COURT_KEYWORDS if kw in q)
    swedish_score = sum(1 for kw in SWEDISH_LAW_KEYWORDS if kw in q)
    eu_score = sum(1 for kw in EU_KEYWORDS if kw in q)

    total = court_score + swedish_score + eu_score

    # Multi-domain: signals from 2+ domains
    if total > 0:
        domains_hit = sum(1 for s in [court_score, swedish_score, eu_score] if s > 0)
        if domains_hit >= 2:
            return AgentDomain.MULTI

    # Single domain: pick the strongest signal
    if court_score > swedish_score and court_score > eu_score:
        return AgentDomain.COURT
    if eu_score > swedish_score and eu_score > court_score:
        return AgentDomain.EU_LAW
    if swedish_score > 0:
        return AgentDomain.SWEDISH_LAW

    # No clear signal: default to MULTI (searches everything, gives best chance of finding relevant content)
    return AgentDomain.MULTI


# --- Agents ---

class DomainAgent:
    """A specialized RAG agent for a specific knowledge domain."""

    def __init__(
        self,
        domain: AgentDomain,
        search_engine: SemanticSearchEngine,
        llm_engine: Optional[GeminiEngine],
        doc_type_filters: List[str],
        filename_filter_fn=None,
    ):
        self.domain = domain
        self.search = search_engine
        self.llm = llm_engine
        self.doc_type_filters = doc_type_filters
        self.filename_filter_fn = filename_filter_fn  # optional: further filter within doc_type
        self.system_prompt = AGENT_PROMPTS[domain]

    @timed("agent.retrieve")
    def retrieve(self, query: str, n_results: int = 8) -> list:
        """Retrieve relevant documents from this agent's domain."""
        all_results = []
        for doc_type in self.doc_type_filters:
            results = self.search.search(
                query,
                n_results=n_results,
                doc_type_filter=doc_type,
            )
            if self.filename_filter_fn and doc_type == "legislation":
                results = [r for r in results if self.filename_filter_fn(r.filename)]
            all_results.extend(results)

        # Sort by similarity and take top n_results
        all_results.sort(key=lambda r: r.similarity, reverse=True)
        return all_results[:n_results]

    @timed("agent.generate")
    def generate(self, query: str, n_results: int = 8) -> AgentResponse:
        """Retrieve and generate a response for the given query."""
        results = self.retrieve(query, n_results=n_results)

        if not results:
            return AgentResponse(
                domain=self.domain,
                content=f"Inga relevanta dokument hittades i {self.domain.value}-domänen.",
                sources_used=0,
                doc_types_searched=self.doc_type_filters,
            )

        context, sources = format_context(results)

        if self.llm:
            content = self.llm.generate_response(
                query, context, sources,
                system_prompt_override=self.system_prompt,
            )
        else:
            # Fallback: just show search excerpts
            lines = [f"**Sökresultat ({len(results)} träffar):**\n"]
            for i, r in enumerate(results[:5]):
                title = getattr(r, 'title', r.decision_id)
                excerpt = r.chunk_text[:200] + "..." if len(r.chunk_text) > 200 else r.chunk_text
                lines.append(f"[{i+1}] **{title}**\n> {excerpt}\n")
            content = "\n".join(lines)

        return AgentResponse(
            domain=self.domain,
            content=content,
            sources_used=len(results),
            doc_types_searched=self.doc_type_filters,
        )


class MultiAgentRouter:
    """Routes queries to specialized agents and optionally synthesizes multi-domain responses."""

    def __init__(
        self,
        search_engine: SemanticSearchEngine,
        llm_engine: Optional[GeminiEngine],
    ):
        self.llm = llm_engine

        # Court decisions agent
        self.court_agent = DomainAgent(
            domain=AgentDomain.COURT,
            search_engine=search_engine,
            llm_engine=llm_engine,
            doc_type_filters=["decision", "application"],
        )

        # Swedish law agent (legislation that is NOT EU)
        self.swedish_law_agent = DomainAgent(
            domain=AgentDomain.SWEDISH_LAW,
            search_engine=search_engine,
            llm_engine=llm_engine,
            doc_type_filters=["legislation", "application"],
            filename_filter_fn=lambda fn: classify_legislation_domain(fn) == AgentDomain.SWEDISH_LAW,
        )

        # EU law agent (EU legislation only)
        self.eu_agent = DomainAgent(
            domain=AgentDomain.EU_LAW,
            search_engine=search_engine,
            llm_engine=llm_engine,
            doc_type_filters=["legislation"],
            filename_filter_fn=lambda fn: classify_legislation_domain(fn) == AgentDomain.EU_LAW,
        )

        self._agents = {
            AgentDomain.COURT: self.court_agent,
            AgentDomain.SWEDISH_LAW: self.swedish_law_agent,
            AgentDomain.EU_LAW: self.eu_agent,
        }

    @timed("router.route")
    def route(self, query: str) -> str:
        """Route a query to the appropriate agent(s) and return a response."""
        domain = classify_query(query)

        if domain == AgentDomain.MULTI:
            return self._synthesize(query)

        agent = self._agents[domain]
        response = agent.generate(query)

        # Add agent attribution footer
        domain_labels = {
            AgentDomain.COURT: "\U0001f3db\ufe0f Domstolsagent",
            AgentDomain.SWEDISH_LAW: "\U0001f4dc Svensk r\u00e4ttsagent",
            AgentDomain.EU_LAW: "\U0001f1ea\U0001f1fa EU-agent",
        }
        footer = f"\n\n---\n*{domain_labels[domain]} | {response.sources_used} k\u00e4llor anv\u00e4nda*"
        return response.content + footer

    @timed("router.synthesize")
    def _synthesize(self, query: str) -> str:
        """Query multiple agents and synthesize a cross-domain response."""
        # Retrieve from all three agents (fewer results per agent to stay within context)
        court_results = self.court_agent.retrieve(query, n_results=5)
        swedish_results = self.swedish_law_agent.retrieve(query, n_results=5)
        eu_results = self.eu_agent.retrieve(query, n_results=4)

        if not court_results and not swedish_results and not eu_results:
            return "Inga relevanta dokument hittades i n\u00e5gon kunskapsdom\u00e4n."

        # Build structured context with domain sections
        context_parts = []
        sources = []
        idx = 1

        for domain_tag, results in [
            ("DOMSTOLSBESLUT", court_results),
            ("SVENSK LAGSTIFTNING", swedish_results),
            ("EU-LAGSTIFTNING", eu_results),
        ]:
            if results:
                context_parts.append(f"\n=== {domain_tag} ===")
                for r in results:
                    title = getattr(r, 'title', r.decision_id)
                    doc_type_label = {
                        "decision": "beslut",
                        "legislation": "lagstiftning",
                        "application": "ans\u00f6kan",
                    }.get(getattr(r, 'doc_type', ''), 'dokument')
                    context_parts.append(f"\n[{idx}] {title} ({doc_type_label})\n{r.chunk_text}")
                    sources.append({
                        "index": idx,
                        "title": title,
                        "type_label": doc_type_label,
                        "domain": domain_tag,
                    })
                    idx += 1

        context = "\n".join(context_parts)

        if self.llm:
            content = self.llm.generate_response(
                query, context, sources,
                system_prompt_override=SYNTHESIS_SYSTEM_PROMPT,
            )
        else:
            # Fallback without LLM
            lines = [f"**Syntes fr\u00e5n {len(sources)} k\u00e4llor:**\n"]
            for s in sources[:10]:
                lines.append(f"[{s['index']}] **{s['title']}** ({s['domain']})")
            content = "\n".join(lines)

        # Footer showing which agents contributed
        agent_summary = []
        if court_results:
            agent_summary.append(f"\U0001f3db\ufe0f Domstol: {len(court_results)}")
        if swedish_results:
            agent_summary.append(f"\U0001f4dc Svensk r\u00e4tt: {len(swedish_results)}")
        if eu_results:
            agent_summary.append(f"\U0001f1ea\U0001f1fa EU: {len(eu_results)}")

        footer = f"\n\n---\n*\U0001f500 Syntes | {' | '.join(agent_summary)}*"
        return content + footer
