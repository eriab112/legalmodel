"""
LLM response generation using Google Gemini API for NAP Legal AI Advisor.
"""

import os
from typing import Any, Dict, Generator, List, Optional

import google.generativeai as genai

from utils.timing import timed

SYSTEM_PROMPT = """Du är en AI-expert på svensk miljörätt, specifikt inom området Nationella planen för moderna miljövillkor (NAP) för vattenkraft.

Du har tillgång till en kunskapsbas med domstolsbeslut, lagstiftning, riktlinjer och ansökningar relaterade till vattenkraft och miljöanpassning i Sverige.

Regler du ALLTID följer:
1. Svara baserat på de tillhandahållna källorna. Citera alltid vilka dokument du baserar ditt svar på med fotnoter i formatet [1], [2] etc.
2. Om informationen inte finns i de tillhandahållna källorna, säg det tydligt. Hitta inte på information.
3. Svara alltid på svenska om inte användaren uttryckligen ber om ett annat språk.
4. Var saklig och professionell. Du ger information, inte juridisk rådgivning.
5. När du refererar till specifika mål, använd målnumret (t.ex. M 1234-22).
6. Avsluta alltid ditt svar med en källförteckning som listar alla refererade dokument med nummer.

Källförteckningen ska ha formatet:
---
**Källor:**
[1] Dokumenttitel — typ (beslut/lagstiftning/ansökan)
[2] Dokumenttitel — typ
"""


def _doc_type_label(doc_type: str) -> str:
    """Map doc_type to Swedish label for display."""
    return {
        "decision": "beslut",
        "legislation": "lagstiftning",
        "application": "ansökan",
    }.get(doc_type, doc_type)


def format_context(
    search_results: List, knowledge_base=None
) -> tuple[str, List[Dict]]:
    """Format search results into context string and source list for the LLM.

    Returns:
        context_str: Formatted text with numbered sources
        sources: List of dicts with source metadata for footnotes
    """
    context_parts: List[str] = []
    sources: List[Dict] = []

    for i, r in enumerate(search_results, start=1):
        title = getattr(r, "title", "") or getattr(r, "filename", "")
        doc_type = getattr(r, "doc_type", "decision")
        chunk_text = getattr(r, "chunk_text", "")
        doc_id = getattr(r, "decision_id", "")

        type_label = _doc_type_label(doc_type)
        context_parts.append(f"[{i}] {title} [{type_label}]")
        context_parts.append(f"Text: {chunk_text}")
        context_parts.append("")

        sources.append({
            "index": i,
            "title": title,
            "doc_type": doc_type,
            "type_label": type_label,
            "doc_id": doc_id,
            "filename": getattr(r, "filename", ""),
        })

    context_str = "\n".join(context_parts).strip()
    return context_str, sources


class GeminiEngine:
    """LLM engine using Google Gemini for RAG response generation."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        # REST transport: API calls use HTTPS (patched by ssl_fix), not gRPC — works behind corporate proxy
        genai.configure(api_key=api_key, transport="rest")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self._chat_sessions: Dict[str, Any] = {}  # for conversation memory later
        print(f"GeminiEngine initialized with model: {self.model_name}")

    @timed("llm.generate_response")
    def generate_response(
        self, query: str, context: str, sources: List[Dict],
        system_prompt_override: Optional[str] = None,
    ) -> str:
        """Generate a response using Gemini with RAG context.

        Args:
            query: User's question
            context: Formatted context from format_context()
            sources: Source metadata list from format_context()
            system_prompt_override: If provided, use this system instruction instead of the default
        Returns:
            Generated response text
        """
        prompt = f"""Här är relevanta källor från kunskapsbasen:

{context}

---
Användarens fråga: {query}
"""
        try:
            if system_prompt_override:
                temp_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_prompt_override,
                )
                response = temp_model.generate_content(prompt)
            else:
                response = self._model.generate_content(prompt)
            if response.text is None:
                return "Tyvärr kunde jag inte generera ett svar just nu. Försök igen senare. (Inget svar från modellen)"
            return response.text
        except Exception as e:
            return f"Tyvärr kunde jag inte generera ett svar just nu. Försök igen senare. (Fel: {e})"

    def generate_streaming(
        self, query: str, context: str, sources: List[Dict]
    ) -> Generator[str, None, None]:
        """Stream response chunks from Gemini."""
        prompt = f"""Här är relevanta källor från kunskapsbasen:

{context}

---
Användarens fråga: {query}
"""
        try:
            response = self._model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"Tyvärr kunde jag inte generera ett svar just nu. Försök igen senare. (Fel: {e})"


def get_llm_engine() -> Optional[GeminiEngine]:
    """Get LLM engine, returns None if API key not configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        return GeminiEngine()
    except Exception as e:
        print(f"Failed to initialize GeminiEngine: {e}")
        return None
