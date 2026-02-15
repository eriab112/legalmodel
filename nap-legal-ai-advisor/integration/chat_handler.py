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
    {
        "label": "Analysera risk",
        "query": "Analysera M 3753-22",
        "icon": "🎯",
    },
]

WELCOME_MESSAGE = """Hej! Jag är NAP Legal AI Advisor \u2013 ett AI-drivet beslutsstöd för vattenkraftens miljöanpassning.

Jag har tre specialiserade kunskapsagenter:
- \U0001f3db\ufe0f **Domstolsagent** \u2013 Expert på 50 domstolsbeslut och ansökningar
- \U0001f4dc **Svensk rättsagent** \u2013 Expert på miljöbalken, NAP, tekniska riktlinjer
- \U0001f1ea\U0001f1fa **EU-agent** \u2013 Expert på vattendirektivet och CIS-vägledningar

Jag kan även:
- \U0001f3af **Analysera risk** \u2013 LegalBERT-prediktion för specifika beslut (t.ex. *Analysera M 3753-22*)
- \U0001f4ca **Jämföra och sammanställa** \u2013 Statistik, jämförelser och kostnadsanalyser

Ställ en fråga \u2013 jag väljer automatiskt rätt agent baserat på din fråga, eller kombinerar flera vid behov!"""


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
