"""
RAG system combining semantic search + risk predictor + data_loader.

Provides retrieval-based responses using template formatting enriched
with metadata from the labeled dataset. Optional LLM (Gemini) for open-ended questions.
"""

import re
from typing import Dict, List, Optional

from backend.agents import MultiAgentRouter
from backend.llm_engine import GeminiEngine, format_context, get_llm_engine
from utils.data_loader import DecisionRecord
from utils.timing import timed


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
        """Classify intent and route to appropriate handler."""
        q = query.lower().strip()

        # Intent detection via keywords
        if _matches(q, ["hog risk", "hög risk", "high risk", "hoga risker"]):
            return self._format_risk_response("HIGH_RISK")
        elif _matches(q, ["lag risk", "låg risk", "low risk", "laga risker"]):
            return self._format_risk_response("LOW_RISK")
        elif _matches(q, ["medel risk", "medium risk"]):
            return self._format_risk_response("MEDIUM_RISK")
        elif _matches(q, ["riskfordelning", "riskfördelning", "fordelning", "distribution"]):
            return self._format_distribution()
        elif _matches(q, ["atgard", "åtgärd", "vanligaste", "measures"]):
            return self._format_measures()
        elif _matches(q, ["senaste", "nyaste", "latest", "recent"]):
            return self._format_recent_decisions()
        elif _matches(q, ["jamfor", "jämför", "compare", "versus", " vs "]):
            return self._handle_comparison(q)
        elif _matches(q, ["statistik", "statistics", "overblick", "översikt"]):
            return self._format_statistics()
        elif _matches(q, ["kostnad", "cost", "kronor"]) or _matches_word(q, "kr"):
            return self._format_cost_info()
        elif _matches(q, ["analysera", "predicera", "bedöm risk", "riskbedöm", "predict"]):
            return self._handle_risk_prediction(query)
        else:
            # Route to domain-specific agents
            if self.llm:
                return self.router.route(query)
            else:
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
            court = d.metadata.get("court", "")
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

    def _format_measures(self) -> str:
        """Format most common measures from court rulings."""
        freq = self.data.get_measure_frequency()
        _exclude = {"kontrollprogram", "skyddsgaller"}
        freq = {k: v for k, v in freq.items() if k not in _exclude}
        if not freq:
            return "Inga åtgärder hittades."
        lines = ["### Vanligaste åtgärder i domslut\n"]
        for measure, count in list(freq.items())[:10]:
            lines.append(f"- **{measure}**: {count} beslut")
        return "\n".join(lines)

    def _format_recent_decisions(self) -> str:
        """Format the most recent labeled decisions."""
        decisions = sorted(
            self.data.get_labeled_decisions(),
            key=lambda d: d.metadata.get("date", ""),
            reverse=True,
        )[:5]
        lines = ["### Senaste besluten\n"]
        for d in decisions:
            case = d.metadata.get("case_number", d.id)
            date = d.metadata.get("date", "")
            court = d.metadata.get("court", "")
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

    def _format_comparison(self, d1: DecisionRecord, d2: DecisionRecord) -> str:
        """Format a side-by-side comparison of two decisions."""
        lines = ["### Jämförelse\n"]
        lines.append("| | **{}** | **{}** |".format(
            d1.metadata.get("case_number", d1.id),
            d2.metadata.get("case_number", d2.id),
        ))
        lines.append("|---|---|---|")
        lines.append("| Domstol | {} | {} |".format(
            d1.metadata.get("court", ""), d2.metadata.get("court", ""),
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

    def _format_statistics(self) -> str:
        """Format overall statistics for the knowledge base."""
        dist = self.data.get_label_distribution()
        total = sum(dist.values())
        all_decisions = self.data.get_all_decisions()
        courts = self.data.get_courts()
        date_min, date_max = self.data.get_date_range()
        measures = self.data.get_measure_frequency()

        lines = [
            "### Statistik\n",
            f"- **Totalt antal beslut**: {len(all_decisions)} (varav {total} klassificerade)",
            f"- **Datumintervall**: {date_min} till {date_max}",
            f"- **Domstolar**: {len(courts)} st",
            f"- **Unika åtgärder**: {len(measures)} typer",
        ]
        for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
            count = dist.get(label, 0)
            if count == 0:
                continue
            emoji = RISK_EMOJI.get(label, "")
            lines.append(f"- {emoji} {RISK_LABELS_SV[label]}: {count}")
        return "\n".join(lines)

    def _format_cost_info(self) -> str:
        """Format cost information from decisions with max_cost_sek."""
        lines = ["### Kostnadsinformation\n"]
        decisions = self.data.get_labeled_decisions()
        costs = []
        for d in decisions:
            if d.scoring_details and d.scoring_details.get("max_cost_sek"):
                costs.append((d, d.scoring_details["max_cost_sek"]))

        if not costs:
            return "Ingen kostnadsinformation tillgänglig."

        costs.sort(key=lambda x: -x[1])
        for d, cost in costs[:10]:
            case = d.metadata.get("case_number", d.id)
            emoji = RISK_EMOJI.get(d.label, "")
            lines.append(f"- {emoji} **{case}**: {cost:,.0f} kr")
        return "\n".join(lines)

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
            court = decision.metadata.get("court", "")
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
            court = r.metadata.get("court", "")
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
    import re
    return bool(re.search(r'\b' + re.escape(word) + r'\b', text))
