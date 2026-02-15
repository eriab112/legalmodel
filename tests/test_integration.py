"""Tests for integration modules — SharedContext, ChatHandler, SearchHandler."""

import sys
import pytest
from unittest.mock import MagicMock

from integration.shared_context import SharedContext
from integration.chat_handler import ChatHandler, QUICK_ACTIONS, WELCOME_MESSAGE
from integration.search_handler import SearchHandler
from tests.conftest import _make_decision


# ---------------------------------------------------------------------------
# SharedContext
# ---------------------------------------------------------------------------

class TestSharedContext:
    def test_initialize_sets_defaults(self):
        SharedContext.initialize()
        st = sys.modules["streamlit"]
        assert st.session_state["messages"] == []
        assert st.session_state["current_mode"] == "chat"
        assert st.session_state["prediction_cache"] == {}
        assert st.session_state["initialized"] is True

    def test_initialize_does_not_overwrite(self):
        st = sys.modules["streamlit"]
        st.session_state["messages"] = [{"role": "user", "content": "hi"}]
        SharedContext.initialize()
        assert len(st.session_state["messages"]) == 1

    def test_add_and_get_messages(self):
        SharedContext.initialize()
        SharedContext.add_message("user", "hello")
        SharedContext.add_message("assistant", "hi there")
        messages = SharedContext.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "hi there"

    def test_switch_mode(self):
        SharedContext.initialize()
        SharedContext.switch_mode("search")
        assert SharedContext.get_mode() == "search"

    def test_selected_decision(self):
        SharedContext.initialize()
        SharedContext.set_selected_decision("m1-22")
        assert SharedContext.get_selected_decision() == "m1-22"
        st = sys.modules["streamlit"]
        assert st.session_state["show_decision_detail"] is True

        SharedContext.set_selected_decision(None)
        assert SharedContext.get_selected_decision() is None
        assert st.session_state["show_decision_detail"] is False

    def test_prediction_cache(self):
        SharedContext.initialize()
        pred = MagicMock()
        SharedContext.cache_prediction("m1-22", pred)
        assert SharedContext.get_cached_prediction("m1-22") is pred
        assert SharedContext.get_cached_prediction("unknown") is None

    def test_set_and_get_filters(self):
        SharedContext.initialize()
        SharedContext.set_filters(label="HIGH_RISK", court="Nacka TR")
        filters = SharedContext.get_filters()
        assert filters["label"] == "HIGH_RISK"
        assert filters["court"] == "Nacka TR"

    def test_clear_filters(self):
        SharedContext.initialize()
        SharedContext.set_filters(label="HIGH_RISK")
        SharedContext.clear_filters()
        filters = SharedContext.get_filters()
        assert filters["label"] is None
        assert filters["court"] is None


# ---------------------------------------------------------------------------
# ChatHandler
# ---------------------------------------------------------------------------

class TestChatHandler:
    @pytest.fixture
    def handler(self):
        SharedContext.initialize()
        rag = MagicMock()
        rag.generate_response.return_value = "Mock response"
        return ChatHandler(rag)

    def test_process_message_stores_in_context(self, handler):
        handler.process_message("hej")
        messages = SharedContext.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hej"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Mock response"

    def test_process_message_returns_response(self, handler):
        result = handler.process_message("test")
        assert result == "Mock response"

    def test_get_quick_actions(self, handler):
        actions = handler.get_quick_actions()
        assert len(actions) == 6
        assert all("label" in a and "query" in a for a in actions)

    def test_get_welcome_message(self, handler):
        msg = handler.get_welcome_message()
        assert "NAP Legal AI Advisor" in msg

    def test_initialize_chat_adds_welcome(self, handler):
        handler.initialize_chat()
        messages = SharedContext.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_initialize_chat_no_duplicate(self, handler):
        SharedContext.add_message("user", "already started")
        handler.initialize_chat()
        messages = SharedContext.get_messages()
        # Should not add welcome if messages already exist
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# SearchHandler
# ---------------------------------------------------------------------------

class TestSearchHandler:
    @pytest.fixture
    def handler(self):
        SharedContext.initialize()
        data_loader = MagicMock()
        search_engine = MagicMock()
        predictor = MagicMock()

        data_loader.get_courts.return_value = ["Nacka TR"]
        data_loader.get_date_range.return_value = ("2023-01-01", "2024-12-31")

        decision = _make_decision()
        data_loader.get_decision.return_value = decision
        predictor.predict_decision.return_value = MagicMock()

        search_engine.search.return_value = [MagicMock(decision_id="m1")]

        return SearchHandler(data_loader, search_engine, predictor)

    def test_execute_search_returns_results(self, handler):
        results = handler.execute_search("fiskväg", n_results=5)
        assert len(results) == 1
        handler.search.search.assert_called_once()

    def test_execute_search_stores_in_session(self, handler):
        handler.execute_search("test")
        st = sys.modules["streamlit"]
        assert st.session_state["search_results"] is not None
        assert st.session_state["search_query"] == "test"

    def test_execute_search_uses_filters(self, handler):
        SharedContext.set_filters(label="HIGH_RISK")
        handler.execute_search("test")
        call_kwargs = handler.search.search.call_args[1]
        assert call_kwargs["label_filter"] == "HIGH_RISK"

    def test_get_decision_detail(self, handler):
        result = handler.get_decision_detail("m1234-22")
        assert result is not None
        handler.data.get_decision.assert_called_with("m1234-22")

    def test_get_risk_prediction_caches(self, handler):
        # First call should compute
        pred1 = handler.get_risk_prediction("m1234-22")
        assert pred1 is not None

        # Cache the prediction
        SharedContext.cache_prediction("m1234-22", pred1)

        # Second call should use cache
        pred2 = handler.get_risk_prediction("m1234-22")
        assert pred2 is pred1

    def test_get_available_filters(self, handler):
        filters = handler.get_available_filters()
        assert "labels" in filters
        assert "courts" in filters
        assert "Nacka TR" in filters["courts"]
        assert filters["date_min"] == "2023-01-01"

    def test_get_search_results_empty_initial(self, handler):
        results = handler.get_search_results()
        assert results == []

    def test_get_search_query_empty_initial(self, handler):
        assert handler.get_search_query() == ""
