"""
Document explorer for NAP Legal AI Advisor.
Browse, search, and filter all documents in the knowledge base.
"""

import streamlit as st
import pandas as pd

from integration.shared_context import SharedContext
from ui.styles import risk_badge_html, RISK_LABELS_SV, doc_type_badge_html
from ui.search_interface import render_decision_detail, render_prediction_detail


# Map Swedish filter labels to internal doc_type values
_DOC_TYPE_MAP = {
    "Alla": None,
    "Beslut": "decision",
    "Lagstiftning": "legislation",
    "Ansökningar": "application",
}

_RISK_FILTER_MAP = {
    "Alla": None,
    "Hög risk": "HIGH_RISK",
    "Medelrisk": "MEDIUM_RISK",
    "Låg risk": "LOW_RISK",
}


def render_explorer(search_handler):
    """Render the document explorer interface."""
    # Decision detail redirect
    if st.session_state.get("show_decision_detail") and SharedContext.get_selected_decision():
        render_decision_detail(search_handler)
        return

    # --- Search bar ---
    search_col, btn_col = st.columns([4, 1])
    with search_col:
        query = st.text_input(
            "Sök i alla dokument",
            value=search_handler.get_search_query(),
            placeholder="Sök i alla dokument...",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.button("Sök", type="primary", width="stretch", key="explorer_search_btn")

    # --- Filter row (inline) ---
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        risk_choice = st.selectbox(
            "Risknivå",
            list(_RISK_FILTER_MAP.keys()),
            key="explorer_risk_filter",
        )

    with f_col2:
        available = search_handler.get_available_filters()
        court_options = ["Alla"] + available["courts"]
        court_choice = st.selectbox("Domstol", court_options, key="explorer_court_filter")

    with f_col3:
        doc_type_choice = st.selectbox(
            "Dokumenttyp",
            list(_DOC_TYPE_MAP.keys()),
            key="explorer_doc_type_filter",
        )

    with f_col4:
        sort_choice = st.selectbox(
            "Sortera efter",
            ["Relevans", "Datum (nyast först)", "Datum (äldst först)"],
            key="explorer_sort",
        )

    # Apply filters to SharedContext
    label_filter = _RISK_FILTER_MAP.get(risk_choice)
    court_filter = None if court_choice == "Alla" else court_choice
    doc_type_filter = _DOC_TYPE_MAP.get(doc_type_choice)
    SharedContext.set_filters(label=label_filter, court=court_filter)

    # --- Execute search or show browse mode ---
    if search_clicked and query:
        with st.spinner("Söker..."):
            results = search_handler.execute_search(query, n_results=20)
    elif query and search_handler.get_search_query() == query:
        results = search_handler.get_search_results()
    else:
        results = None

    if results is not None:
        # Apply doc_type_filter client-side if the search didn't filter it
        if doc_type_filter:
            results = [r for r in results if getattr(r, "doc_type", "decision") == doc_type_filter]

        # Apply sort
        if sort_choice == "Datum (nyast först)":
            results = sorted(results, key=lambda r: r.metadata.get("date", ""), reverse=True)
        elif sort_choice == "Datum (äldst först)":
            results = sorted(results, key=lambda r: r.metadata.get("date", ""))

        if results:
            st.markdown(f"**{len(results)} resultat** för *{search_handler.get_search_query()}*")
            for result in results:
                _render_explorer_card(result, search_handler)
        else:
            st.info("Inga resultat hittades. Prova andra sökord.")
    else:
        # Browse mode — show all labeled decisions
        _render_browse_mode(search_handler)


def _render_explorer_card(result, search_handler):
    """Render a search result card, adapting to document type."""
    doc_type = getattr(result, "doc_type", "decision")
    sim_pct = f"{result.similarity:.1%}"

    excerpt = result.chunk_text[:300]
    if len(result.chunk_text) > 300:
        excerpt += "..."

    if doc_type == "decision":
        case = result.metadata.get("case_number", result.decision_id)
        date = result.metadata.get("date", "")
        court = result.metadata.get("court", "")
        badge = risk_badge_html(result.label) if result.label else ""

        st.markdown(
            f"""<div class="result-card">
            <div class="result-card-header">
                <span class="result-card-title">{case}</span>
                <span class="similarity-score">{sim_pct} likhet</span>
            </div>
            <div class="result-card-meta">{court} | {date} {badge}</div>
            <div class="result-card-excerpt">{excerpt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Visa detaljer", key=f"exp_detail_{result.decision_id}"):
                SharedContext.set_selected_decision(result.decision_id)
                st.rerun()
        with col2:
            if st.button("Analysera risk", key=f"exp_predict_{result.decision_id}"):
                SharedContext.set_selected_decision(result.decision_id)
                st.session_state[f"run_prediction_{result.decision_id}"] = True
                st.rerun()

    elif doc_type == "legislation":
        title = getattr(result, "title", "") or result.decision_id
        type_badge = doc_type_badge_html("legislation")

        st.markdown(
            f"""<div class="result-card">
            <div class="result-card-header">
                <span class="result-card-title">{title}</span>
                <span class="similarity-score">{sim_pct} likhet</span>
            </div>
            <div class="result-card-meta">{type_badge}</div>
            <div class="result-card-excerpt">{excerpt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("Visa text"):
            st.text(result.chunk_text)

    elif doc_type == "application":
        title = getattr(result, "title", "") or result.decision_id
        type_badge = doc_type_badge_html("application")

        st.markdown(
            f"""<div class="result-card">
            <div class="result-card-header">
                <span class="result-card-title">{title}</span>
                <span class="similarity-score">{sim_pct} likhet</span>
            </div>
            <div class="result-card-meta">{type_badge}</div>
            <div class="result-card-excerpt">{excerpt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

        with st.expander("Visa text"):
            st.text(result.chunk_text)

    else:
        # Generic fallback
        title = getattr(result, "title", "") or result.decision_id
        st.markdown(
            f"""<div class="result-card">
            <div class="result-card-header">
                <span class="result-card-title">{title}</span>
                <span class="similarity-score">{sim_pct} likhet</span>
            </div>
            <div class="result-card-excerpt">{excerpt}</div>
            </div>""",
            unsafe_allow_html=True,
        )


def _render_browse_mode(search_handler):
    """Show a browseable table of all labeled decisions."""
    st.markdown("### Alla klassificerade beslut")

    decisions = search_handler.data.get_labeled_decisions()
    decisions_sorted = sorted(
        decisions,
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )

    risk_sv = {
        "HIGH_RISK": "Hög risk",
        "MEDIUM_RISK": "Medelrisk",
        "LOW_RISK": "Låg risk",
    }

    rows = []
    for d in decisions_sorted:
        rows.append({
            "Målnummer": d.metadata.get("case_number", d.id),
            "Domstol": d.metadata.get("court", ""),
            "Datum": d.metadata.get("date", ""),
            "Risknivå": risk_sv.get(d.label, d.label or ""),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # Decision detail lookup
        st.markdown("---")
        case_input = st.text_input(
            "Ange målnummer för att visa detaljer (t.ex. m3753-22)",
            key="explorer_case_lookup",
        )
        if case_input:
            # Try to find the decision by case number
            match = None
            for d in decisions_sorted:
                cn = d.metadata.get("case_number", d.id)
                if cn.lower() == case_input.strip().lower() or d.id.lower() == case_input.strip().lower():
                    match = d
                    break
            if match:
                SharedContext.set_selected_decision(match.id)
                st.rerun()
            else:
                st.warning(f"Inget beslut hittades med målnummer '{case_input}'.")
    else:
        st.info("Inga klassificerade beslut tillgängliga.")
