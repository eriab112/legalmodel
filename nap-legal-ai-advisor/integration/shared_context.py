"""
Shared session state management for the NAP Legal AI Advisor.

Manages st.session_state for messages, mode, selected decision,
search results, prediction cache, and active filters.
"""

from typing import Dict, List, Optional

import streamlit as st


class SharedContext:
    """Manages all shared session state."""

    @staticmethod
    def initialize():
        """Initialize all session state keys with defaults."""
        defaults = {
            "messages": [],
            "current_mode": "chat",  # "chat" or "search"
            "selected_decision": None,
            "search_results": [],
            "search_query": "",
            "prediction_cache": {},  # decision_id -> PredictionResult
            "active_filters": {
                "label": None,
                "court": None,
                "date_from": None,
                "date_to": None,
            },
            "show_decision_detail": False,
            "initialized": True,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def add_message(role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    @staticmethod
    def get_messages() -> List[Dict]:
        return st.session_state.get("messages", [])

    @staticmethod
    def set_selected_decision(decision_id: Optional[str]):
        st.session_state.selected_decision = decision_id
        st.session_state.show_decision_detail = decision_id is not None

    @staticmethod
    def get_selected_decision() -> Optional[str]:
        return st.session_state.get("selected_decision")

    @staticmethod
    def switch_mode(mode: str):
        st.session_state.current_mode = mode

    @staticmethod
    def get_mode() -> str:
        return st.session_state.get("current_mode", "chat")

    @staticmethod
    def cache_prediction(decision_id: str, prediction):
        st.session_state.prediction_cache[decision_id] = prediction

    @staticmethod
    def get_cached_prediction(decision_id: str):
        return st.session_state.get("prediction_cache", {}).get(decision_id)

    @staticmethod
    def set_filters(**kwargs):
        filters = st.session_state.get("active_filters", {})
        filters.update(kwargs)
        st.session_state.active_filters = filters

    @staticmethod
    def get_filters() -> Dict:
        return st.session_state.get("active_filters", {})

    @staticmethod
    def clear_filters():
        st.session_state.active_filters = {
            "label": None,
            "court": None,
            "date_from": None,
            "date_to": None,
        }
