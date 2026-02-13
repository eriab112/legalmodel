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
        "label": "Hog-risk beslut",
        "query": "Vilka beslut har hog risk?",
        "icon": "🔴",
    },
    {
        "label": "Vanligaste atgarder",
        "query": "Vilka ar de vanligaste atgarderna?",
        "icon": "🔧",
    },
    {
        "label": "Riskfordelning",
        "query": "Visa riskfordelningen",
        "icon": "📊",
    },
    {
        "label": "Senaste besluten",
        "query": "Visa de senaste besluten",
        "icon": "📅",
    },
]

WELCOME_MESSAGE = """Hej! Jag ar NAP Legal AI Advisor - ett AI-system for analys av svenska miljodomstolsbeslut om vattenkraft.

Jag kan hjalpa dig med:
- **Riskanalys** - Visa beslut per riskniva (hog/medel/lag)
- **Semantisk sokning** - Sok i 40 domstolsbeslut med naturligt sprak
- **Jamforelser** - Jamfor tva beslut sida vid sida
- **Statistik** - Overblick over atgarder, kostnader och utfall
- **Riskprediktion** - Analysera text med LegalBERT-modellen

Prova en snabbfragor nedan eller stall en egen fraga!"""


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
