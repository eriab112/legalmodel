"""
RAG system combining semantic search + risk predictor + data_loader.

Provides retrieval-based responses using template formatting enriched
with metadata from the labeled dataset. Optional LLM (Gemini) for open-ended questions.
"""

import re
import statistics
from typing import Callable, Dict, List, Optional, Tuple

from backend.agents import MultiAgentRouter, COURT_SYSTEM_PROMPT
from backend.llm_engine import GeminiEngine, format_context, get_llm_engine
from utils.data_loader import DecisionRecord
from utils.timing import timed


# ---------------------------------------------------------------------------
# Query filter extraction (compound query support)
# ---------------------------------------------------------------------------

def _extract_court_filter(q: str) -> Optional[str]:
    """Extract court name from query if mentioned."""
    court_map = {
        "växjö": "Växjö",
        "nacka": "Nacka",
        "östersund": "Östersund",
        "umeå": "Umeå",
        "vänersborg": "Vänersborg",
        "möd": "MÖD",
    }
    q_lower = q.lower()
    for key, value in court_map.items():
        if key in q_lower:
            return value
    return None


def _extract_count(q: str) -> int:
    """Extract a count from query like '3 senaste' or 'top 5'."""
    match = re.search(r"(\d+)\s*(?:senaste|första|top|dyraste|billigaste)", q, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 20)
    return 5  # default


def _extract_outcome_filter(q: str) -> Optional[str]:
    """Extract outcome filter from query."""
    q_lower = q.lower()
    if any(w in q_lower for w in ["nekad", "avslag", "avslagen", "nekat", "nekande"]):
        return "denied"
    if any(w in q_lower for w in ["beviljat", "godkänd", "tillstånd beviljat"]):
        return "granted"
    if any(w in q_lower for w in ["återförvis", "remand"]):
        return "remanded"
    return None


def _extract_measure_filter(q: str) -> Optional[str]:
    """Extract measure type from query."""
    measures = [
        "fiskväg",
        "omlöp",
        "minimitappning",
        "biotopvård",
        "utskov",
        "utrivning",
        "faunapassage",
    ]
    q_lower = q.lower()
    for m in measures:
        if m in q_lower:
            return m
    return None


def _extract_risk_filter(q: str) -> Optional[str]:
    """Extract risk label filter from query."""
    q_lower = q.lower()
    if any(w in q_lower for w in ["hög risk", "hog risk", "high risk", "hoga risker"]):
        return "HIGH_RISK"
    if any(w in q_lower for w in ["låg risk", "lag risk", "low risk", "laga risker"]):
        return "LOW_RISK"
    if any(w in q_lower for w in ["medel risk", "medium risk"]):
        return "MEDIUM_RISK"
    return None


def _apply_filters(
    decisions: List[DecisionRecord],
    court_filter: Optional[str] = None,
    risk_filter: Optional[str] = None,
    outcome_filter: Optional[str] = None,
    measure_filter: Optional[str] = None,
) -> List[DecisionRecord]:
    """Filter decisions by court, risk label, outcome, and/or measure."""
    result = list(decisions)
    if court_filter:
        c = court_filter.lower()
        result = [
            d
            for d in result
            if (
                (d.metadata.get("court") or "").lower().find(c) >= 0
                or (d.metadata.get("originating_court") or "").lower().find(c) >= 0
            )
        ]
    if risk_filter:
        result = [d for d in result if d.label == risk_filter]
    if outcome_filter:
        if outcome_filter == "denied":
            result = [
                d
                for d in result
                if d.metadata.get("application_outcome") in ("denied", "appeal_denied")
            ]
        elif outcome_filter == "granted":
            result = [
                d
                for d in result
                if d.metadata.get("application_outcome")
                in ("granted", "granted_modified", "conditions_changed")
            ]
        elif outcome_filter == "remanded":
            result = [
                d for d in result if d.metadata.get("application_outcome") == "remanded"
            ]
    if measure_filter:
        m = measure_filter.lower()
        result = [
            d
            for d in result
            if any(
                m in (x or "").lower()
                for x in (
                    (d.scoring_details or {}).get("domslut_measures") or []
                )
                + (d.extracted_measures or [])
            )
        ]
    return result


RISK_LABELS_SV = {
    "HIGH_RISK": "Hög risk",
    "MEDIUM_RISK": "Medelrisk",
    "LOW_RISK": "Låg risk",
}

RISK_EMOJI = {
    "HIGH_RISK": "🔴",
    "MEDIUM_RISK": "🟡",
    "LOW_RISK": "🟢",
}


class RAGSystem:
    """Combines search, prediction, and data for template-based responses."""

    def __init__(self, data_loader, search_engine, predictor, llm_engine=None):
        self.data = data_loader
        self.search = search_engine
        self.predictor = predictor
        self.llm = llm_engine  # None means fallback to template responses

        # Multi-agent router for domain-specific RAG
        self.router = MultiAgentRouter(search_engine, llm_engine)

    @timed("rag.generate_response")
    def generate_response(self, query: str) -> str:
        """Classify intent and route to appropriate handler. Supports compound queries with filters."""
        q = query.lower().strip()

        # Check if user is asking about a specific case — direct lookup first
        case_pattern = r'm[\s-]?(\d+)[\s-](\d+)'
        case_matches = re.findall(case_pattern, q, re.IGNORECASE)
        if case_matches and not _matches(q, ["analysera", "predicera", "bedöm risk", "riskbedöm", "predict"]):
            case_id = f"m{case_matches[0][0]}-{case_matches[0][1]}"
            decision = self.data.get_decision(case_id)
            if decision and self.llm:
                return self._answer_about_decision(query, decision)

        # Detect query modifiers (apply to any intent)
        court_filter = _extract_court_filter(q)
        count_filter = _extract_count(q)
        outcome_filter = _extract_outcome_filter(q)
        measure_filter = _extract_measure_filter(q)
        risk_filter = _extract_risk_filter(q)

        # Custom risk assessment — user describes their plant situation (check FIRST, more specific)
        assessment_signals = [
            "mitt kraftverk", "min anläggning", "mitt vattenkraftverk",
            "vår anläggning", "vårt kraftverk",
            "jag har ett", "vi har ett", "vi äger",
            "bedöm min risk", "bedöm risk för min", "riskbedömning för min",
            "vad kan jag förvänta", "vad blir utfallet",
        ]
        is_assessment = any(signal in q for signal in assessment_signals)

        if is_assessment and self.llm:
            return self._custom_risk_assessment(query)

        # Advisory/analytical patterns — check SECOND (more general)
        advisory_signals = [
            "vad ska jag", "vad bör jag", "vad kan jag", "vad rekommenderar",
            "hur ska jag", "hur bör jag", "hur kan jag",
            "vilken typ", "vilken sorts", "vilket alternativ",
            "kostnadseffektiv", "effektivast", "bäst", "snabbast", "billigast",
            "rekommendera", "föreslå", "tipsa", "råd",
            "min deadline",
            "jag behöver", "jag vill", "jag planerar",
            "alternativ", "möjlighet", "strategi",
            "vad säger", "vad kräver", "vad innebär",
            "varför", "hur fungerar", "förklara",
            "jämför med", "skillnad mellan",
        ]
        is_advisory = any(signal in q for signal in advisory_signals)

        if is_advisory and self.llm:
            # Skip template intents entirely — route to LLM agents
            return self.router.route(query)

        # Score all intents; pick best (most keyword matches, then first in list for tie)
        def score(keywords: List[str], extra: bool = False) -> int:
            s = sum(1 for kw in keywords if kw in q)
            return s + (1 if extra else 0)

        # (score, order_index, handler) — order_index for tie-break
        intent_candidates: List[Tuple[int, int, Callable[[], str]]] = []
        order = 0

        intent_candidates.append((
            score(["analysera", "predicera", "bedöm risk", "riskbedöm", "predict"]),
            order, lambda: self._handle_risk_prediction(query)))
        order += 1
        intent_candidates.append((score(["hog risk", "hög risk", "high risk", "hoga risker"]), order, lambda: self._format_risk_response("HIGH_RISK")))
        order += 1
        intent_candidates.append((score(["lag risk", "låg risk", "low risk", "laga risker"]), order, lambda: self._format_risk_response("LOW_RISK")))
        order += 1
        intent_candidates.append((score(["medel risk", "medium risk"]), order, lambda: self._format_risk_response("MEDIUM_RISK")))
        order += 1
        intent_candidates.append((score(["riskfordelning", "riskfördelning", "fordelning", "distribution"]), order, lambda: self._format_distribution()))
        order += 1
        intent_candidates.append((score(["jamfor", "jämför", "compare", "versus", " vs "]), order, lambda: self._handle_comparison(query)))
        order += 1
        intent_candidates.append((score(["atgard", "åtgärd", "vanligaste", "measures"]), order, lambda: self._format_measures(court_filter=court_filter)))
        order += 1
        intent_candidates.append((
            score(["senaste", "nyaste", "latest", "recent"]),
            order,
            lambda: self._format_recent_decisions(
                query,
                court_filter=court_filter,
                count_filter=count_filter,
                outcome_filter=outcome_filter,
                measure_filter=measure_filter,
                risk_filter=risk_filter,
            ),
        ))
        order += 1
        intent_candidates.append((score(["statistik", "statistics", "overblick", "översikt"]), order, lambda: self._format_statistics(court_filter=court_filter)))
        order += 1
        intent_candidates.append((
            score(["kostnad", "cost", "kronor", "dyraste", "dyr", "billigaste"]) or (1 if _matches_word(q, "kr") else 0),
            order,
            lambda: self._format_cost_info(court_filter=court_filter, measure_filter=measure_filter),
        ))
        order += 1
        intent_candidates.append((score(["utfall", "outcome", "beviljat", "avslag", "nekad", "avslaget", "tillstånd beviljat"]), order, lambda: self._format_outcomes()))
        order += 1
        intent_candidates.append((score(["handläggningstid", "processing time", "hur lång tid"]), order, lambda: self._format_processing_times()))
        order += 1
        intent_candidates.append((score(["kraftverk", "anläggning", "power plant"]), order, lambda: self._format_power_plants()))
        order += 1
        intent_candidates.append((score(["vattendrag", "watercourse", "river"]) or (1 if " å " in q else 0), order, lambda: self._format_watercourses()))
        order += 1
        intent_candidates.append((score(["rankordna", "ranking", "flest", "oftast", "domstol", "court", "nekar", "avslår"]), order, lambda: self._format_rankings(q)))

        best = max(intent_candidates, key=lambda x: (x[0], -x[1]))  # max score, then smallest order
        if best[0] > 0:
            return best[2]()

        # Fallback: domain-specific agents or search
        if self.llm:
            return self.router.route(query)
        return self._format_search_response(query)

    def _format_risk_response(self, label: str) -> str:
        """Format a list of decisions with the given risk label."""
        decisions = self.data.get_decisions_by_label(label)
        if not decisions:
            return f"Inga beslut hittades med risknivå {RISK_LABELS_SV.get(label, label)}."

        emoji = RISK_EMOJI.get(label, "")
        header = f"### {emoji} Beslut med {RISK_LABELS_SV[label]} ({len(decisions)} st)\n\n"
        lines = []
        for d in sorted(decisions, key=lambda x: x.metadata.get("date", ""), reverse=True):
            case = d.metadata.get("case_number", d.id)
            court = d.metadata.get("originating_court") or d.metadata.get("court", "")
            date = d.metadata.get("date", "")
            outcome = ""
            if d.scoring_details:
                outcome = d.scoring_details.get("outcome_desc", "")
            measures = ""
            if d.scoring_details and d.scoring_details.get("domslut_measures"):
                measures = ", ".join(d.scoring_details["domslut_measures"])

            line = f"- **{case}** ({date}) - {court}"
            if outcome:
                line += f"\n  - Utfall: {outcome}"
            if measures:
                line += f"\n  - Åtgärder: {measures}"
            lines.append(line)

        return header + "\n".join(lines)

    def _format_distribution(self) -> str:
        """Format label distribution summary."""
        dist = self.data.get_label_distribution()
        total = sum(dist.values())
        lines = ["### Riskfördelning\n"]
        for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
            count = dist.get(label, 0)
            if count == 0:
                continue
            pct = (count / total * 100) if total > 0 else 0
            emoji = RISK_EMOJI.get(label, "")
            lines.append(f"- {emoji} **{RISK_LABELS_SV[label]}**: {count} beslut ({pct:.0f}%)")
        lines.append(f"\n**Totalt**: {total} klassificerade beslut")
        return "\n".join(lines)

    def _format_measures(self, court_filter: Optional[str] = None) -> str:
        """Format most common measures from court rulings, optionally filtered by court."""
        decisions = self.data.get_labeled_decisions()
        if court_filter:
            decisions = _apply_filters(decisions, court_filter=court_filter)
        freq: Dict[str, int] = {}
        for d in decisions:
            measures = []
            if d.scoring_details:
                measures = d.scoring_details.get("domslut_measures", [])
            if not measures:
                measures = d.extracted_measures or []
            for m in measures:
                freq[m] = freq.get(m, 0) + 1
        freq = dict(sorted(freq.items(), key=lambda x: -x[1]))
        _exclude = {"kontrollprogram", "skyddsgaller"}
        freq = {k: v for k, v in freq.items() if k not in _exclude}
        if not freq:
            return "Inga åtgärder hittades."
        header = "### Vanligaste åtgärder i domslut\n"
        if court_filter:
            header = f"### Vanligaste åtgärder i domslut (vid {court_filter})\n"
        lines = [header]
        for measure, count in list(freq.items())[:10]:
            lines.append(f"- **{measure}**: {count} beslut")
        return "\n".join(lines)

    def _format_recent_decisions(
        self,
        query: str = "",
        court_filter: Optional[str] = None,
        count_filter: int = 5,
        outcome_filter: Optional[str] = None,
        measure_filter: Optional[str] = None,
        risk_filter: Optional[str] = None,
    ) -> str:
        """Format the most recent labeled decisions with optional filters."""
        decisions = self.data.get_labeled_decisions()
        decisions = _apply_filters(
            decisions,
            court_filter=court_filter,
            risk_filter=risk_filter,
            outcome_filter=outcome_filter,
            measure_filter=measure_filter,
        )
        decisions = sorted(
            decisions,
            key=lambda d: d.metadata.get("date", ""),
            reverse=True,
        )[:count_filter]

        filters_applied = []
        if court_filter:
            filters_applied.append(f"domstol={court_filter}")
        if risk_filter:
            filters_applied.append(f"risk={RISK_LABELS_SV.get(risk_filter, risk_filter)}")
        if outcome_filter:
            filters_applied.append(f"utfall={outcome_filter}")
        if measure_filter:
            filters_applied.append(f"åtgärd={measure_filter}")
        filters_applied.append(f"antal={count_filter}")

        header = "### Senaste besluten\n"
        if filters_applied:
            header = f"### Senaste besluten (filter: {', '.join(filters_applied)})\n"

        lines = [header]
        if not decisions:
            lines.append("Inga beslut matchade filtren.")
            return "\n".join(lines)
        for d in decisions:
            case = d.metadata.get("case_number", d.id)
            date = d.metadata.get("date", "")
            court = d.metadata.get("originating_court") or d.metadata.get("court", "")
            emoji = RISK_EMOJI.get(d.label, "")
            lines.append(f"- {emoji} **{case}** ({date}) - {court} [{RISK_LABELS_SV.get(d.label, '')}]")
        return "\n".join(lines)

    def _handle_comparison(self, query: str) -> str:
        """Extract two case IDs from query and format a comparison table."""
        # Extract case IDs from query (e.g., "m3753-22" or "M 3753-22")
        pattern = r'm[\s-]?(\d+)[\s-](\d+)'
        matches = re.findall(pattern, query, re.IGNORECASE)
        if len(matches) < 2:
            return "Ange två målnummer för att jämföra, t.ex. 'Jämför M 3753-22 med M 605-24'."

        ids = [f"m{m[0]}-{m[1]}" for m in matches[:2]]
        decisions = [self.data.get_decision(did) for did in ids]

        if not all(decisions):
            missing = [ids[i] for i, d in enumerate(decisions) if d is None]
            return f"Kunde inte hitta beslut: {', '.join(missing)}"

        return self._format_comparison(decisions[0], decisions[1])

    def _answer_about_decision(self, query: str, decision: DecisionRecord) -> str:
        """Answer a question about a specific decision using its own text + similar decisions."""
        # Build context from the decision's own sections
        sections = decision.sections or {}
        context_parts = []
        case_num = decision.metadata.get("case_number", decision.id)
        court = decision.metadata.get("originating_court") or decision.metadata.get("court", "")
        date = decision.metadata.get("date", "")

        context_parts.append(f"[1] {case_num} — {court} ({date}) [beslut]")
        # Include key sections in priority order
        for section_name in ["domslut", "domskäl", "bakgrund", "saken"]:
            section_text = sections.get(section_name, "")
            if section_text:
                context_parts.append(f"\n--- {section_name.upper()} ---\n{section_text[:3000]}")
        # Fallback to full_text if no sections
        if not any(sections.get(s) for s in ["domslut", "domskäl", "bakgrund", "saken"]):
            context_parts.append(f"\n{decision.full_text[:5000]}")

        sources = [{"index": 1, "title": f"{case_num} — {court}", "doc_type": "decision",
                     "type_label": "beslut", "doc_id": decision.id, "filename": decision.filename}]

        # Also retrieve 3-5 similar decisions for comparison context
        similar_results = self.search.find_similar_decisions(decision.id, n_results=4)
        for i, r in enumerate(similar_results, start=2):
            title = getattr(r, "title", "") or r.decision_id
            context_parts.append(f"\n[{i}] {title} [beslut]\n{r.chunk_text[:500]}")
            sources.append({"index": i, "title": title, "doc_type": "decision",
                            "type_label": "beslut", "doc_id": r.decision_id, "filename": r.filename})

        context = "\n".join(context_parts)

        response = self.llm.generate_response(
            query, context, sources,
            system_prompt_override=COURT_SYSTEM_PROMPT,
        )

        footer = f"\n\n---\n*\U0001f3db\ufe0f Domstolsagent | {len(sources)} källor använda*"
        return response + footer

    # Swedish cities/regions → likely MMD jurisdiction
    _LOCATION_TO_COURT = {
        "stockholm": "Nacka", "nacka": "Nacka", "uppsala": "Nacka",
        "göteborg": "Vänersborg", "gothenburg": "Vänersborg",
        "vänersborg": "Vänersborg", "karlstad": "Vänersborg", "värmland": "Vänersborg",
        "växjö": "Växjö", "jönköping": "Växjö", "kalmar": "Växjö",
        "kronoberg": "Växjö", "skåne": "Växjö", "blekinge": "Växjö",
        "östersund": "Östersund", "jämtland": "Östersund", "dalarna": "Östersund",
        "gävle": "Östersund", "gävleborg": "Östersund", "hälsingland": "Östersund",
        "umeå": "Umeå", "norrland": "Umeå", "luleå": "Umeå",
        "norrbotten": "Umeå", "västerbotten": "Umeå",
    }

    _ASSESSMENT_SYSTEM_PROMPT = """Du är en AI-rådgivare specialiserad på riskbedömning för vattenkraftens miljöanpassning inom NAP.

En verksamhetsutövare beskriver sitt vattenkraftverk och sin situation. Du har fått:
1. Deras beskrivning av anläggningen
2. Statistik från liknande domstolsbeslut i kunskapsbasen
3. Utdrag från de mest relevanta besluten

Din uppgift:
- Bedöm vilken risknivå (hög/låg) verksamhetsutövaren troligen står inför
- Lista de åtgärder som domstolar typiskt kräver för liknande anläggningar
- Uppskatta ungefärliga kostnader baserat på jämförelsebesluten
- Ge konkreta rekommendationer för hur de bör förbereda sig
- Var tydlig med att detta är en indikation baserad på historiska beslut, inte juridisk rådgivning

Svara på svenska. Strukturera svaret med tydliga rubriker.
Citera specifika jämförelsebeslut med målnummer.
Avsluta med en sammanfattande riskindikation: 🔴 Hög risk / 🟢 Låg risk / 🟡 Osäker."""

    def _custom_risk_assessment(self, query: str) -> str:
        """Perform a custom risk assessment based on user's plant description."""
        q = query.lower()

        # --- Extract plant characteristics ---
        # Location → court jurisdiction
        detected_court = None
        for location, court in self._LOCATION_TO_COURT.items():
            if location in q:
                detected_court = court
                break

        # Size
        size = None
        if "litet" in q or "liten" in q or "lilla" in q:
            size = "litet"
        elif "medelstort" in q or "medel" in q:
            size = "medelstort"
        elif "stort" in q or "stora" in q:
            size = "stort"

        # Fish species
        fish_species = []
        for species in ["lax", "öring", "ål", "harr", "sik", "gädda", "abborre", "nejonöga"]:
            if species in q:
                fish_species.append(species)

        # Existing measures
        existing_measures = []
        for measure in ["fiskväg", "omlöp", "minimitappning", "biotopvård",
                        "utskov", "utrivning", "faunapassage", "skyddsgaller"]:
            if measure in q:
                existing_measures.append(measure)

        # Production
        production = None
        gwh_match = re.search(r'(\d+(?:[.,]\d+)?)\s*gwh', q)
        mw_match = re.search(r'(\d+(?:[.,]\d+)?)\s*mw', q)
        if gwh_match:
            production = f"{gwh_match.group(1)} GWh"
        elif mw_match:
            production = f"{mw_match.group(1)} MW"

        # Watercourse
        watercourse_names = self.data.get_watercourses() if hasattr(self.data, 'get_watercourses') else []
        detected_watercourse = None
        for wc in watercourse_names:
            if wc.lower() in q:
                detected_watercourse = wc
                break

        # --- Find similar cases ---
        results = self.search.search(query, n_results=8)

        # If a court was detected, boost results from that court
        if detected_court and results:
            def court_boost(r):
                r_court = r.metadata.get("originating_court") or r.metadata.get("court", "")
                return 0 if detected_court.lower() in r_court.lower() else 1
            results = sorted(results, key=lambda r: (court_boost(r), -r.similarity))

        # --- Compute statistics from similar cases ---
        similar_decisions = []
        for r in results:
            d = self.data.get_decision(r.decision_id)
            if d:
                similar_decisions.append(d)

        n_similar = len(similar_decisions)
        high_risk_count = sum(1 for d in similar_decisions if d.label == "HIGH_RISK")
        low_risk_count = sum(1 for d in similar_decisions if d.label == "LOW_RISK")

        # Most common measures
        measure_freq: Dict[str, int] = {}
        for d in similar_decisions:
            measures = (d.scoring_details or {}).get("domslut_measures", []) or d.extracted_measures or []
            for m in measures:
                measure_freq[m] = measure_freq.get(m, 0) + 1
        measure_freq = dict(sorted(measure_freq.items(), key=lambda x: -x[1]))

        # Average costs
        costs_found = []
        for d in similar_decisions:
            cost = d.metadata.get("total_cost_sek")
            if cost is None and d.scoring_details:
                cost = d.scoring_details.get("max_cost_sek")
            if cost is not None:
                costs_found.append(float(cost))
        avg_cost = sum(costs_found) / len(costs_found) if costs_found else None

        # Outcomes
        outcome_dist: Dict[str, int] = {}
        for d in similar_decisions:
            o = d.metadata.get("application_outcome")
            if o:
                outcome_dist[o] = outcome_dist.get(o, 0) + 1

        # Average processing time
        proc_times = []
        for d in similar_decisions:
            pt = d.metadata.get("processing_time_days")
            if pt is not None:
                proc_times.append(int(pt))
        avg_proc_time = sum(proc_times) / len(proc_times) if proc_times else None

        # --- Build context for Gemini ---
        context_parts = [f"Användarens beskrivning: {query}\n"]

        # Add statistics summary
        context_parts.append("--- STATISTIK FRÅN LIKNANDE BESLUT ---")
        if n_similar:
            context_parts.append(f"Antal liknande beslut: {n_similar}")
            context_parts.append(f"Hög risk: {high_risk_count}, Låg risk: {low_risk_count}")
            if measure_freq:
                top_measures = ", ".join(f"{m} ({c})" for m, c in list(measure_freq.items())[:5])
                context_parts.append(f"Vanligaste åtgärder: {top_measures}")
            if avg_cost:
                context_parts.append(f"Genomsnittlig kostnad: {avg_cost:,.0f} kr")
            if outcome_dist:
                outcome_str = ", ".join(f"{o}: {c}" for o, c in outcome_dist.items())
                context_parts.append(f"Utfall: {outcome_str}")
            if avg_proc_time:
                context_parts.append(f"Genomsnittlig handläggningstid: {avg_proc_time:.0f} dagar")
        if detected_court:
            context_parts.append(f"Trolig domstol: {detected_court}")
        if size:
            context_parts.append(f"Anläggningsstorlek: {size}")
        if fish_species:
            context_parts.append(f"Nämnda fiskarter: {', '.join(fish_species)}")
        if detected_watercourse:
            context_parts.append(f"Vattendrag: {detected_watercourse}")
        if production:
            context_parts.append(f"Produktion: {production}")

        # Add excerpts from the 3 most similar decisions
        context_parts.append("\n--- UTDRAG FRÅN LIKNANDE BESLUT ---")
        sources = []
        for i, d in enumerate(similar_decisions[:3], start=1):
            case_num = d.metadata.get("case_number", d.id)
            court = d.metadata.get("originating_court") or d.metadata.get("court", "")
            date = d.metadata.get("date", "")
            domslut = (d.sections or {}).get("domslut", "")[:1500]
            label_sv = RISK_LABELS_SV.get(d.label, d.label or "Ej klassificerad")
            measures = (d.scoring_details or {}).get("domslut_measures", [])
            cost = d.metadata.get("total_cost_sek") or (d.scoring_details or {}).get("max_cost_sek")

            context_parts.append(f"\n[{i}] {case_num} — {court} ({date}) — {label_sv}")
            if measures:
                context_parts.append(f"Åtgärder: {', '.join(measures)}")
            if cost:
                context_parts.append(f"Kostnad: {float(cost):,.0f} kr")
            if domslut:
                context_parts.append(f"Domslut: {domslut}")

            sources.append({
                "index": i, "title": f"{case_num} — {court}",
                "doc_type": "decision", "type_label": "beslut",
                "doc_id": d.id, "filename": d.filename,
            })

        context = "\n".join(context_parts)

        # --- Send to Gemini ---
        response = self.llm.generate_response(
            query, context, sources,
            system_prompt_override=self._ASSESSMENT_SYSTEM_PROMPT,
        )

        # --- Format output ---
        # Statistics summary box
        stats_lines = []
        if n_similar:
            high_pct = (high_risk_count / n_similar * 100) if n_similar else 0
            low_pct = (low_risk_count / n_similar * 100) if n_similar else 0
            stats_lines.append(f"\n---\n**Baserat på {n_similar} liknande beslut:**")
            stats_lines.append(f"- Risknivå: {high_pct:.0f}% hög risk, {low_pct:.0f}% låg risk")
            if measure_freq:
                measure_summary = ", ".join(
                    f"{m} ({c}/{n_similar})" for m, c in list(measure_freq.items())[:5]
                )
                stats_lines.append(f"- Vanligaste åtgärder: {measure_summary}")
            if avg_cost:
                stats_lines.append(f"- Genomsnittlig kostnad: {avg_cost:,.0f} kr")
            if avg_proc_time:
                stats_lines.append(f"- Genomsnittlig handläggningstid: {avg_proc_time:.0f} dagar")

        footer = f"\n\n---\n*\U0001f3af Riskbedömning | {n_similar} jämförelsebeslut*"
        return f"### \U0001f3af Riskbedömning för din anläggning\n\n{response}" + "\n".join(stats_lines) + footer

    def _format_comparison(self, d1: DecisionRecord, d2: DecisionRecord) -> str:
        """Format a side-by-side comparison of two decisions."""
        lines = ["### Jämförelse\n"]
        lines.append("| | **{}** | **{}** |".format(
            d1.metadata.get("case_number", d1.id),
            d2.metadata.get("case_number", d2.id),
        ))
        lines.append("|---|---|---|")
        lines.append("| Domstol | {} | {} |".format(
            d1.metadata.get("originating_court") or d1.metadata.get("court", ""),
            d2.metadata.get("originating_court") or d2.metadata.get("court", ""),
        ))
        lines.append("| Datum | {} | {} |".format(
            d1.metadata.get("date", ""), d2.metadata.get("date", ""),
        ))
        lines.append("| Risknivå | {} {} | {} {} |".format(
            RISK_EMOJI.get(d1.label, ""), RISK_LABELS_SV.get(d1.label, d1.label or "Ej klassificerad"),
            RISK_EMOJI.get(d2.label, ""), RISK_LABELS_SV.get(d2.label, d2.label or "Ej klassificerad"),
        ))

        m1 = d1.scoring_details.get("domslut_measures", []) if d1.scoring_details else []
        m2 = d2.scoring_details.get("domslut_measures", []) if d2.scoring_details else []
        lines.append("| Åtgärder | {} | {} |".format(
            ", ".join(m1) if m1 else "-",
            ", ".join(m2) if m2 else "-",
        ))

        o1 = d1.scoring_details.get("outcome_desc", "-") if d1.scoring_details else "-"
        o2 = d2.scoring_details.get("outcome_desc", "-") if d2.scoring_details else "-"
        lines.append("| Utfall | {} | {} |".format(o1, o2))

        return "\n".join(lines)

    def _format_statistics(self, court_filter: Optional[str] = None) -> str:
        """Format overall statistics, optionally filtered by court (e.g. 'vanligaste åtgärder vid växjö')."""
        all_decisions = self.data.get_all_decisions()
        if court_filter:
            all_decisions = _apply_filters(all_decisions, court_filter=court_filter)
        labeled = [d for d in all_decisions if d.label is not None]
        total_labeled = len(labeled)
        dist: Dict[str, int] = {}
        for d in labeled:
            dist[d.label] = dist.get(d.label, 0) + 1
        courts = sorted({(d.metadata.get("originating_court") or d.metadata.get("court") or "").split("(")[0].strip() for d in all_decisions if (d.metadata.get("originating_court") or d.metadata.get("court"))})
        dates = [d.metadata.get("date") for d in all_decisions if d.metadata.get("date")]
        date_min = min(dates) if dates else None
        date_max = max(dates) if dates else None
        measure_freq: Dict[str, int] = {}
        for d in labeled:
            for m in (d.scoring_details or {}).get("domslut_measures", []) or (d.extracted_measures or []):
                measure_freq[m] = measure_freq.get(m, 0) + 1

        header = "### Statistik\n"
        if court_filter:
            header = f"### Statistik (vid {court_filter})\n"
        lines = [
            header,
            f"- **Totalt antal beslut**: {len(all_decisions)} (varav {total_labeled} klassificerade)",
            f"- **Datumintervall**: {date_min or '-'} till {date_max or '-'}",
            f"- **Domstolar**: {len(courts)} st",
            f"- **Unika åtgärder**: {len(measure_freq)} typer",
        ]
        for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
            count = dist.get(label, 0)
            if count == 0:
                continue
            emoji = RISK_EMOJI.get(label, "")
            lines.append(f"- {emoji} {RISK_LABELS_SV[label]}: {count}")

        outcome_dist: Dict[str, int] = {}
        for d in all_decisions:
            o = d.metadata.get("application_outcome")
            if o:
                outcome_dist[o] = outcome_dist.get(o, 0) + 1
        if outcome_dist:
            lines.append("\n**Utfallsfördelning:**")
            for outcome, count in sorted(outcome_dist.items(), key=lambda x: -x[1]):
                lines.append(f"- {outcome}: {count}")

        return "\n".join(lines)

    def _format_cost_info(
        self,
        court_filter: Optional[str] = None,
        measure_filter: Optional[str] = None,
    ) -> str:
        """Format cost information using total_cost_sek (fallback to max_cost_sek), with optional filters."""
        decisions = self.data.get_labeled_decisions()
        decisions = _apply_filters(decisions, court_filter=court_filter, measure_filter=measure_filter)
        costs: List[Tuple[DecisionRecord, float]] = []
        for d in decisions:
            cost = d.metadata.get("total_cost_sek")
            if cost is None and d.scoring_details:
                cost = d.scoring_details.get("max_cost_sek")
            if cost is not None:
                costs.append((d, float(cost)))

        filters_applied = [f for f in [court_filter and f"domstol={court_filter}", measure_filter and f"åtgärd={measure_filter}"] if f]
        header = "### Kostnadsinformation\n"
        if filters_applied:
            header = f"### Kostnadsinformation (filter: {', '.join(filters_applied)})\n"
        lines = [header]
        if not costs:
            return header + "Ingen kostnadsinformation tillgänglig."

        costs.sort(key=lambda x: -x[1])
        for d, cost in costs[:10]:
            case = d.metadata.get("case_number", d.id)
            court = d.metadata.get("originating_court") or d.metadata.get("court", "")
            outcome = d.metadata.get("application_outcome_sv") or (d.scoring_details or {}).get("outcome_desc", "")
            emoji = RISK_EMOJI.get(d.label, "")
            lines.append(f"- {emoji} **{case}**: {cost:,.0f} kr — {court}" + (f" ({outcome})" if outcome else ""))
        return "\n".join(lines)

    def _format_outcomes(self) -> str:
        """Format application outcome distribution and per-court breakdown."""
        dist = self.data.get_outcome_distribution()
        by_court = self.data.get_outcomes_by_court()

        if not dist:
            return "Ingen utfallsdata tillgänglig. Kör extraktionsskriptet först."

        total = sum(dist.values())
        lines = ["### Utfallsfördelning\n"]
        for outcome, count in dist.most_common():
            pct = count / total * 100 if total else 0
            lines.append(f"- **{outcome}**: {count} beslut ({pct:.0f}%)")
        lines.append(f"\n**Totalt**: {total} beslut\n")

        lines.append("### Utfall per domstol\n")
        for court, court_dist in sorted(by_court.items()):
            parts = [f"{o}={c}" for o, c in court_dist.most_common()]
            lines.append(f"- **{court}**: {', '.join(parts)}")

        return "\n".join(lines)

    def _format_processing_times(self) -> str:
        """Format processing time statistics: overall avg+median, per-court breakdown, top 10 longest."""
        decisions = self.data.get_all_decisions()
        with_days: List[Tuple[DecisionRecord, int]] = []
        for d in decisions:
            days = d.metadata.get("processing_time_days")
            if days is not None:
                with_days.append((d, int(days)))
        if not with_days:
            return "Ingen handläggningstidsdata tillgänglig."

        days_list = [d for _, d in with_days]
        avg = sum(days_list) / len(days_list)
        med = statistics.median(days_list)
        lines = [
            "### Handläggningstider\n",
            f"- **Antal beslut med data**: {len(with_days)}",
            f"- **Genomsnitt**: {avg:.0f} dagar",
            f"- **Median**: {med:.0f} dagar",
            f"- **Minimum**: {min(days_list)} dagar",
            f"- **Maximum**: {max(days_list)} dagar\n",
        ]

        # Per-court breakdown
        by_court: Dict[str, List[int]] = {}
        for d, days in with_days:
            court = d.metadata.get("originating_court") or d.metadata.get("court", "Okänd")
            court_short = court.split("(")[0].strip() if court else "Okänd"
            by_court.setdefault(court_short, []).append(days)
        lines.append("**Per domstol:**\n")
        for court_name in sorted(by_court.keys()):
            lst = by_court[court_name]
            lines.append(
                f"- **{court_name}**: snitt {sum(lst) / len(lst):.0f} dagar, "
                f"min {min(lst)}, max {max(lst)}, antal {len(lst)}"
            )
        lines.append("\n**10 längsta fallen:**\n")
        by_case = [(d.metadata.get("case_number", d.id), days) for d, days in with_days]
        by_case.sort(key=lambda x: -x[1])
        for case, days in by_case[:10]:
            lines.append(f"- **{case}**: {days} dagar")
        return "\n".join(lines)

    def _format_power_plants(self) -> str:
        """Format list of power plants found in decisions."""
        plants = self.data.get_power_plants()
        if not plants:
            return "Ingen kraftverksinformation tillgänglig."

        lines = [f"### Kraftverk ({len(plants)} identifierade)\n"]
        for case, name in sorted(plants, key=lambda x: x[1]):
            lines.append(f"- **{name}** ({case})")
        return "\n".join(lines)

    def _format_watercourses(self) -> str:
        """Format list of watercourses found in decisions."""
        wcs = self.data.get_watercourses()
        if not wcs:
            return "Ingen vattendragsinformation tillgänglig."

        lines = [f"### Vattendrag ({len(wcs)} unika)\n"]
        for wc in wcs:
            lines.append(f"- {wc}")
        return "\n".join(lines)

    def _format_rankings(self, query: str) -> str:
        """Format rankings: denial-focused court ranking (count + rate), dyraste by total_cost_sek, or default outcome."""
        q = query.lower()
        lines = []

        if _matches(q, ["domstol", "court", "nekar", "avslår"]):
            by_court = self.data.get_outcomes_by_court()
            denied_outcomes = {"denied", "appeal_denied"}
            court_stats = []
            for court, dist in by_court.items():
                total = sum(dist.values())
                denials = sum(dist.get(o, 0) for o in denied_outcomes)
                rate = (denials / total * 100) if total else 0
                court_stats.append((court, total, denials, rate))
            court_stats.sort(key=lambda x: (-x[2], -x[3]))  # by denial count, then rate
            lines.append("### Ranking: Domstolar (avslag/nekande)\n")
            lines.append("Sorterat efter antal avslag, sedan andel avslag.\n")
            for court, total, denials, rate in court_stats:
                lines.append(f"- **{court}**: {denials} avslag av {total} beslut ({rate:.0f}% nekande)")
        elif _matches(q, ["åtgärd", "measure", "atgard"]):
            freq = self.data.get_measure_frequency()
            lines.append("### Ranking: Vanligaste åtgärder\n")
            for measure, count in list(freq.items())[:15]:
                lines.append(f"- **{measure}**: {count} beslut")
        elif _matches(q, ["kostnad", "dyr", "dyraste", "cost"]):
            decisions = self.data.get_all_decisions()
            cost_list: List[Tuple[DecisionRecord, float]] = []
            for d in decisions:
                cost = d.metadata.get("total_cost_sek")
                if cost is None and d.scoring_details:
                    cost = d.scoring_details.get("max_cost_sek")
                if cost is not None:
                    cost_list.append((d, float(cost)))
            cost_list.sort(key=lambda x: -x[1])
            lines.append("### Ranking: Dyraste beslut\n")
            for d, cost in cost_list[:10]:
                case = d.metadata.get("case_number", d.id)
                court = d.metadata.get("originating_court") or d.metadata.get("court", "")
                outcome = d.metadata.get("application_outcome_sv") or (d.scoring_details or {}).get("outcome_desc", "")
                lines.append(f"- **{case}**: {cost:,.0f} kr — {court}" + (f" ({outcome})" if outcome else ""))
        else:
            dist = self.data.get_outcome_distribution()
            lines.append("### Ranking: Utfall\n")
            for outcome, count in dist.most_common():
                lines.append(f"- **{outcome}**: {count} beslut")

        return "\n".join(lines) if lines else "Kunde inte avgöra vad som ska rankordnas."

    @timed("rag.risk_prediction")
    def _handle_risk_prediction(self, query: str) -> str:
        """Handle risk prediction requests in the chat."""
        # Check if a case number is mentioned
        pattern = r'm[\s-]?(\d+)[\s-](\d+)'
        matches = re.findall(pattern, query, re.IGNORECASE)

        if matches:
            # Predict for a specific case
            case_id = f"m{matches[0][0]}-{matches[0][1]}"
            decision = self.data.get_decision(case_id)
            if not decision:
                return f"Kunde inte hitta beslut med målnummer {case_id}."

            prediction = self.predictor.predict_decision(decision)
            case_num = decision.metadata.get("case_number", case_id)
            court = decision.metadata.get("originating_court") or decision.metadata.get("court", "")
            date = decision.metadata.get("date", "")

            emoji = RISK_EMOJI.get(prediction.predicted_label, "")
            label_sv = RISK_LABELS_SV.get(prediction.predicted_label, prediction.predicted_label)

            lines = [
                f"### {emoji} Riskbedömning: {case_num}",
                f"**Domstol**: {court} | **Datum**: {date}",
                f"",
                f"**LegalBERT-prediktion**: {label_sv}",
                f"**Konfidens**: {prediction.confidence:.1%}",
                f"**Analyserade chunks**: {prediction.num_chunks}",
            ]

            # Add probability breakdown
            lines.append("")
            lines.append("**Sannolikheter:**")
            for label_name, prob in prediction.probabilities.items():
                prob_emoji = RISK_EMOJI.get(label_name, "")
                prob_label = RISK_LABELS_SV.get(label_name, label_name)
                bar = "\u2588" * int(prob * 20) + "\u2591" * (20 - int(prob * 20))
                lines.append(f"- {prob_emoji} {prob_label}: {bar} {prob:.1%}")

            # If ground truth exists, show comparison
            if prediction.ground_truth:
                gt_sv = RISK_LABELS_SV.get(prediction.ground_truth, prediction.ground_truth)
                gt_emoji = RISK_EMOJI.get(prediction.ground_truth, "")
                match = "\u2705 Korrekt" if prediction.predicted_label == prediction.ground_truth else "\u26a0\ufe0f Avviker"
                lines.append(f"")
                lines.append(f"**Faktisk klassificering**: {gt_emoji} {gt_sv} ({match})")

            # If LLM is available, add explanation
            if self.llm:
                lines.append("")
                lines.append("---")
                # Get relevant context for explanation
                results = self.search.search(f"risk {case_num}", n_results=3)
                if results:
                    context, sources = format_context(results)
                    explanation_query = (
                        f"Förklara kortfattat varför domstolsbeslut {case_num} "
                        f"kan klassificeras som {label_sv}. "
                        f"Fokusera på de viktigaste faktorerna i beslutet."
                    )
                    explanation = self.llm.generate_response(explanation_query, context, sources)
                    lines.append(explanation)

            return "\n".join(lines)
        else:
            return (
                "Ange ett målnummer för att analysera risk, t.ex.:\n"
                "- *Analysera M 3753-22*\n"
                "- *Predicera risk för M 1849-22*\n\n"
                "Eller klistra in text direkt i **Utforska**-fliken för att analysera egen text."
            )

    @timed("rag.llm_response")
    def _generate_llm_response(self, query: str, n_results: int = 10) -> str:
        """Generate an LLM response with RAG context from the full knowledge base."""
        # Retrieve relevant chunks across all document types
        results = self.search.search(query, n_results=n_results)

        if not results:
            return "Jag hittade inga relevanta dokument i kunskapsbasen för din fråga. Försök att omformulera eller vara mer specifik."

        # Format context for the LLM
        context, sources = format_context(results)

        # Generate response
        response = self.llm.generate_response(query, context, sources)

        return response

    def _format_search_response(self, query: str) -> str:
        """Format search results as a fallback when LLM is unavailable."""
        results = self.search.search(query, n_results=5)
        if not results:
            return "Inga relevanta resultat hittades. Försök med andra sökord."

        lines = [f"### Sökresultat för: *{query}*\n"]
        for r in results:
            case = r.metadata.get("case_number", r.decision_id)
            date = r.metadata.get("date", "")
            court = r.metadata.get("originating_court") or r.metadata.get("court", "")
            emoji = RISK_EMOJI.get(r.label, "")
            sim = f"{r.similarity:.1%}"

            lines.append(f"**{emoji} {case}** ({date}) - {court}")
            lines.append(f"*Likhet: {sim}*")
            # Show excerpt (first 300 chars of matched chunk)
            excerpt = r.chunk_text[:300]
            if len(r.chunk_text) > 300:
                excerpt += "..."
            lines.append(f"> {excerpt}\n")

        return "\n".join(lines)


def _matches(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


def _matches_word(text: str, word: str) -> bool:
    """Match a keyword as a whole word only."""
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))
