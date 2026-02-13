"""
NAP Legal AI Advisor - Main Streamlit Application

Integrated platform combining:
1. Conversational AI with retrieval-based Q&A
2. Semantic search over 46 Swedish court decisions
3. LegalBERT risk predictions (fine-tuned KB-BERT, fold_4)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# SSL workaround for corporate proxy - must be before any HF imports
import utils.ssl_fix  # noqa: F401

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="NAP Legal AI Advisor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from backend.rag_system import RAGSystem
from backend.risk_predictor import get_predictor
from backend.search_engine import get_search_engine
from integration.chat_handler import ChatHandler
from integration.search_handler import SearchHandler
from integration.shared_context import SharedContext
from ui.chat_interface import render_chat_mode
from ui.search_interface import render_search_mode
from ui.styles import CUSTOM_CSS, metric_card_html
from utils.data_loader import load_data, load_knowledge_base


def init_backend():
    """Initialize all backend components (cached)."""
    if "backend_ready" not in st.session_state:
        with st.spinner("Laddar NAP Legal AI Advisor..."):
            data = load_data()
            search = get_search_engine()
            predictor = get_predictor()

            # Build search index
            search.build_index(data.get_all_decisions())

            # Build expanded search index with all document types
            kb = load_knowledge_base()
            search.build_full_index(kb.get_all_documents())
            st.session_state.knowledge_base = kb

            # Store in session state
            st.session_state.data_loader = data
            st.session_state.search_engine = search
            st.session_state.predictor = predictor
            st.session_state.rag_system = RAGSystem(data, search, predictor)
            st.session_state.chat_handler = ChatHandler(st.session_state.rag_system)
            st.session_state.search_handler = SearchHandler(data, search, predictor)
            st.session_state.backend_ready = True

    return (
        st.session_state.chat_handler,
        st.session_state.search_handler,
        st.session_state.data_loader,
    )


def render_sidebar(data_loader):
    """Render sidebar with app info and metrics."""
    with st.sidebar:
        st.markdown("## NAP Legal AI Advisor")
        st.markdown("AI-driven analys av svenska miljodomstolsbeslut om vattenkraft.")
        st.markdown("---")

        # Key metrics
        dist = data_loader.get_label_distribution()
        total_labeled = sum(dist.values())
        total_all = len(data_loader.get_all_decisions())
        search_engine = st.session_state.get("search_engine")
        n_chunks = search_engine.total_chunks if search_engine else 0

        kb = st.session_state.get("knowledge_base")
        stats = kb.get_corpus_stats() if kb else {}
        total_docs = stats.get("total", total_all) if kb else total_all

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                metric_card_html(total_docs, "Dokument"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                metric_card_html(n_chunks, "Chunks"),
                unsafe_allow_html=True,
            )

        if kb:
            st.caption(
                f"Beslut: {stats.get('decision', 0)} | "
                f"Lagstiftning: {stats.get('legislation', 0)} | "
                f"Ansökningar: {stats.get('application', 0)}"
            )

        st.markdown("")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                metric_card_html(dist.get("HIGH_RISK", 0), "Hog"),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                metric_card_html(dist.get("MEDIUM_RISK", 0), "Medel"),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                metric_card_html(dist.get("LOW_RISK", 0), "Lag"),
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "**Modell**: KB-BERT fine-tuned (fold_4)  \n"
            "**Sokning**: MiniLM multilingual  \n"
            f"**Data**: {total_labeled} klassificerade av {total_all} beslut"
        )


def main():
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    SharedContext.initialize()
    chat_handler, search_handler, data_loader = init_backend()

    # Header
    st.markdown(
        '<h1 class="main-header">NAP Legal AI Advisor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "AI-driven analys av svenska miljodomstolsbeslut om vattenkraft"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    render_sidebar(data_loader)

    # Mode toggle
    tab_chat, tab_search = st.tabs(["💬 Chatt", "🔍 Sok"])

    with tab_chat:
        render_chat_mode(chat_handler)

    with tab_search:
        render_search_mode(search_handler)


if __name__ == "__main__":
    main()
