"""
NAP Legal AI Advisor - Main Streamlit Application

Integrated platform combining:
1. Conversational AI with retrieval-based Q&A
2. Semantic search over Swedish court decisions, legislation, and applications
3. LegalBERT risk predictions (DAPT + fine-tuned KB-BERT, binary classification)
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")  # also check project root

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

from backend.llm_engine import get_llm_engine
from backend.rag_system import RAGSystem
from backend.risk_predictor import get_predictor
from backend.search_engine import get_search_engine
from integration.chat_handler import ChatHandler
from integration.search_handler import SearchHandler
from integration.shared_context import SharedContext
from ui.chat_interface import render_chat_mode
from ui.search_interface import render_search_mode
from ui.overview_interface import render_overview
from ui.explorer_interface import render_explorer
from ui.styles import CUSTOM_CSS, metric_card_html
from utils.data_loader import load_data, load_knowledge_base


def init_backend():
    """Initialize all backend components (cached)."""
    if "backend_ready" not in st.session_state:
        try:
            with st.spinner("Laddar data och kunskapsbas..."):
                data = load_data()
                search = get_search_engine()
                predictor = get_predictor()
                kb = load_knowledge_base()
                st.session_state.knowledge_base = kb

            with st.spinner("Bygger sökindex (kan ta 30–60 s första gången)..."):
                search.build_full_index(kb.get_all_documents())

            with st.spinner("Startar motorer..."):
                llm_engine = get_llm_engine()
                if llm_engine:
                    print("Gemini LLM engine: ACTIVE")
                else:
                    print("Gemini LLM engine: INACTIVE (no API key)")
                st.session_state.llm_engine = llm_engine
                st.session_state.data_loader = data
                st.session_state.search_engine = search
                st.session_state.predictor = predictor
                st.session_state.rag_system = RAGSystem(data, search, predictor, llm_engine=llm_engine)
                st.session_state.chat_handler = ChatHandler(st.session_state.rag_system)
                st.session_state.search_handler = SearchHandler(data, search, predictor)
                st.session_state.backend_ready = True
        except Exception as e:
            import traceback
            st.error(f"**Startfel:** {e}")
            st.code(traceback.format_exc(), language="text")
            st.caption("Kör appen från projektets rotmapp (där mapparna Data/ och nap-legal-ai-advisor/ finns).")
            return None, None, None

    return (
        st.session_state.chat_handler,
        st.session_state.search_handler,
        st.session_state.data_loader,
    )


def render_sidebar(data_loader):
    """Render sidebar with app info and metrics."""
    with st.sidebar:
        st.markdown("## NAP Legal AI Advisor")
        st.markdown("AI-driven analys av svenska miljödomstolsbeslut om vattenkraft.")
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

        high_count = dist.get("HIGH_RISK", 0)
        med_count = dist.get("MEDIUM_RISK", 0)
        low_count = dist.get("LOW_RISK", 0)

        if med_count > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    metric_card_html(high_count, "Hög"),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    metric_card_html(med_count, "Medel"),
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    metric_card_html(low_count, "Låg"),
                    unsafe_allow_html=True,
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    metric_card_html(high_count, "Hög risk"),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    metric_card_html(low_count, "Låg risk"),
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown(
            "**Modell**: KB-BERT binary (DAPT + fine-tuned)  \n"
            "**Precision**: 80% accuracy, 100% HIGH recall  \n"
            "**Sökning**: MiniLM multilingual  \n"
            f"**Data**: {total_labeled} klassificerade av {total_all} beslut"
        )


def main():
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize
    SharedContext.initialize()
    chat_handler, search_handler, data_loader = init_backend()
    if chat_handler is None:
        return

    # Header
    st.markdown(
        '<h1 class="main-header">NAP Legal AI Advisor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "AI-drivet beslutsstöd för vattenkraftens miljöanpassning"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    render_sidebar(data_loader)

    # Three-section layout
    tab_overview, tab_explorer, tab_chat = st.tabs(["📊 Översikt", "🔍 Utforska", "💬 AI-assistent"])

    with tab_overview:
        kb = st.session_state.get("knowledge_base")
        render_overview(data_loader, knowledge_base=kb)

    with tab_explorer:
        render_explorer(search_handler)

    with tab_chat:
        render_chat_mode(chat_handler)


if __name__ == "__main__":
    main()
