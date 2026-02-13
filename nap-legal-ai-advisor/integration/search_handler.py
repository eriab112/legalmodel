"""
Search handler for managing search queries and result display.

Orchestrates search execution, decision detail retrieval,
risk prediction, and filter management.
"""

from typing import Dict, List, Optional

import streamlit as st

from integration.shared_context import SharedContext


class SearchHandler:
    """Manages search queries, results, and decision details."""

    def __init__(self, data_loader, search_engine, predictor):
        self.data = data_loader
        self.search = search_engine
        self.predictor = predictor

    def execute_search(self, query: str, n_results: int = 10) -> List:
        """Execute semantic search with current filters."""
        filters = SharedContext.get_filters()
        results = self.search.search(
            query,
            n_results=n_results,
            label_filter=filters.get("label"),
            court_filter=filters.get("court"),
            date_from=filters.get("date_from"),
            date_to=filters.get("date_to"),
        )
        st.session_state.search_results = results
        st.session_state.search_query = query
        return results

    def get_decision_detail(self, decision_id: str):
        """Get full decision details for display."""
        return self.data.get_decision(decision_id)

    def get_risk_prediction(self, decision_id: str):
        """Get or compute risk prediction for a decision."""
        cached = SharedContext.get_cached_prediction(decision_id)
        if cached:
            return cached

        decision = self.data.get_decision(decision_id)
        if not decision:
            return None

        prediction = self.predictor.predict_decision(decision)
        SharedContext.cache_prediction(decision_id, prediction)
        return prediction

    def get_available_filters(self) -> Dict:
        """Get available filter options from the dataset."""
        labels = ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
        courts = self.data.get_courts()
        date_min, date_max = self.data.get_date_range()
        return {
            "labels": labels,
            "courts": courts,
            "date_min": date_min,
            "date_max": date_max,
        }

    def get_search_results(self) -> List:
        return st.session_state.get("search_results", [])

    def get_search_query(self) -> str:
        return st.session_state.get("search_query", "")
