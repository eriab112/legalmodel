"""
Chat handler for processing user messages.

Routes user input to appropriate RAG system handlers via keyword-based
intent detection. Provides quick action buttons for common queries.
"""

from typing import List, Dict

from integration.shared_context import SharedContext


# Quick action definitions (Swedish)
QUICK_ACTIONS = [
    {
        "label": "Högrisk-beslut",
        "query": "Vilka beslut har hög risk?",
        "icon": "🔴",
    },
    {
        "label": "Vanligaste åtgärder",
        "query": "Vilka är de vanligaste åtgärderna?",
        "icon": "🔧",
    },
    {
        "label": "Riskfördelning",
        "query": "Visa riskfördelningen",
        "icon": "📊",
    },
    {
        "label": "Senaste besluten",
        "query": "Visa de senaste besluten",
        "icon": "📅",
    },
]

WELCOME_MESSAGE = """Hej! Jag är NAP Legal AI Advisor \u2013 ett AI-system för analys av svenska miljödomstolsbeslut om vattenkraft.

Jag kan hjälpa dig med:
- **Riskanalys** \u2013 Visa beslut per risknivå (hög/medel/låg)
- **Sökning** \u2013 Sök i domstolsbeslut, lagstiftning och ansökningar
- **Jämförelser** \u2013 Jämför två beslut sida vid sida
- **Statistik** \u2013 Överblick över åtgärder, kostnader och utfall
- **Riskprediktion** \u2013 Analysera text med LegalBERT-modellen

Ställ en fråga nedan eller använd snabbknapparna!"""


class ChatHandler:
    """Handles chat messages and routes to RAG system."""

    def __init__(self, rag_system):
        self.rag = rag_system

    def process_message(self, user_message: str) -> str:
        """Process a user message and return a response."""
        SharedContext.add_message("user", user_message)
        response = self.rag.generate_response(user_message)
        SharedContext.add_message("assistant", response)
        return response

    def get_quick_actions(self) -> List[Dict]:
        return QUICK_ACTIONS

    def get_welcome_message(self) -> str:
        return WELCOME_MESSAGE

    def initialize_chat(self):
        """Add welcome message if chat is empty."""
        messages = SharedContext.get_messages()
        if not messages:
            SharedContext.add_message("assistant", self.get_welcome_message())
