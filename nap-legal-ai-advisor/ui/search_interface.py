"""
Search mode UI for NAP Legal AI Advisor.

Provides semantic search with result cards, filter sidebar,
decision detail view with risk prediction analysis.
"""

import streamlit as st

from integration.shared_context import SharedContext
from ui.styles import risk_badge_html, RISK_LABELS_SV, RISK_BADGE_CSS


def render_search_mode(search_handler):
    """Render the search mode interface."""
    # Check if we should show detail view
    if st.session_state.get("show_decision_detail") and SharedContext.get_selected_decision():
        render_decision_detail(search_handler)
        return

    # Search bar
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Sok i domstolsbeslut",
            value=search_handler.get_search_query(),
            placeholder="T.ex. faunapassage vid vattenkraftverk, minimitappning...",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("Sok", type="primary", use_container_width=True)

    # Filters sidebar
    render_filters_sidebar(search_handler)

    # Execute search
    if search_clicked and query:
        with st.spinner("Soker..."):
            results = search_handler.execute_search(query)
    else:
        results = search_handler.get_search_results()

    # Display results
    if results:
        st.markdown(f"**{len(results)} resultat** for *{search_handler.get_search_query()}*")
        for result in results:
            render_result_card(result, search_handler)
    elif search_handler.get_search_query():
        st.info("Inga resultat hittades. Prova andra sokord.")
    else:
        st.info("Anvand sokfaltet ovan for att soka i 40 domstolsbeslut om vattenkraft.")


def render_filters_sidebar(search_handler):
    """Render filter controls in the sidebar."""
    with st.sidebar:
        st.markdown("### Filter")

        available = search_handler.get_available_filters()

        # Risk level filter
        label_options = ["Alla"] + available["labels"]
        label_display = {
            "Alla": "Alla riskniva",
            "HIGH_RISK": "Hog risk",
            "MEDIUM_RISK": "Medel risk",
            "LOW_RISK": "Lag risk",
        }
        selected_label = st.selectbox(
            "Riskniva",
            label_options,
            format_func=lambda x: label_display.get(x, x),
        )
        label_filter = None if selected_label == "Alla" else selected_label

        # Court filter
        court_options = ["Alla"] + available["courts"]
        selected_court = st.selectbox("Domstol", court_options)
        court_filter = None if selected_court == "Alla" else selected_court

        # Apply filters
        SharedContext.set_filters(label=label_filter, court=court_filter)

        # Clear filters button
        if st.button("Rensa filter", use_container_width=True):
            SharedContext.clear_filters()
            st.rerun()


def render_result_card(result, search_handler):
    """Render a search result card."""
    case = result.metadata.get("case_number", result.decision_id)
    date = result.metadata.get("date", "")
    court = result.metadata.get("court", "")
    badge = risk_badge_html(result.label) if result.label else ""
    sim_pct = f"{result.similarity:.1%}"

    excerpt = result.chunk_text[:300]
    if len(result.chunk_text) > 300:
        excerpt += "..."

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
        if st.button("Visa detaljer", key=f"detail_{result.decision_id}"):
            SharedContext.set_selected_decision(result.decision_id)
            st.rerun()
    with col2:
        if st.button("Analysera risk", key=f"predict_{result.decision_id}"):
            SharedContext.set_selected_decision(result.decision_id)
            st.session_state[f"run_prediction_{result.decision_id}"] = True
            st.rerun()


def render_decision_detail(search_handler):
    """Render full decision detail view."""
    decision_id = SharedContext.get_selected_decision()
    decision = search_handler.get_decision_detail(decision_id)

    if not decision:
        st.error(f"Beslut '{decision_id}' hittades inte.")
        return

    # Back button
    if st.button("Tillbaka till sokresultat"):
        SharedContext.set_selected_decision(None)
        st.rerun()

    case = decision.metadata.get("case_number", decision.id)
    st.markdown(f"## {case}")

    # Two-column layout: metadata + prediction
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Metadata")
        st.markdown(f"**Domstol**: {decision.metadata.get('court', '')}")
        st.markdown(f"**Datum**: {decision.metadata.get('date', '')}")
        st.markdown(f"**Niva**: {decision.metadata.get('court_level', '')}")
        st.markdown(f"**Amne**: {decision.metadata.get('subject', '')}")

        if decision.label:
            badge = risk_badge_html(decision.label)
            st.markdown(f"**Klassificering**: {badge}", unsafe_allow_html=True)

        if decision.scoring_details:
            st.markdown("### Bedomningsdetaljer")
            sd = decision.scoring_details
            st.markdown(f"**Utfallstyp**: {sd.get('outcome_desc', '')}")
            if sd.get("domslut_measures"):
                st.markdown(f"**Domslut-atgarder**: {', '.join(sd['domslut_measures'])}")
            if sd.get("max_cost_sek"):
                st.markdown(f"**Max kostnad**: {sd['max_cost_sek']:,.0f} kr")

        if decision.linked_water_bodies:
            st.markdown("### Lankade vattenforekomster")
            for wb in decision.linked_water_bodies:
                viss = ", ".join(wb.get("viss_ids", []))
                st.markdown(f"- **{wb.get('water_body', '')}** ({viss})")

    with col2:
        st.markdown("### LegalBERT Riskprediktion")

        # Check if prediction should run automatically
        run_pred = st.session_state.get(f"run_prediction_{decision_id}", False)

        if run_pred or st.button("Kor prediktion", key=f"run_pred_{decision_id}"):
            # Clear the auto-run flag
            st.session_state.pop(f"run_prediction_{decision_id}", None)

            with st.spinner("Kor LegalBERT-inference..."):
                prediction = search_handler.get_risk_prediction(decision_id)

            if prediction:
                render_prediction_detail(prediction)
        else:
            cached = SharedContext.get_cached_prediction(decision_id)
            if cached:
                render_prediction_detail(cached)
            else:
                st.info("Klicka for att kora LegalBERT-prediktion pa detta beslut.")

    # Tabs for full text
    st.markdown("---")
    sections = decision.sections or {}
    section_names = list(sections.keys())

    if section_names:
        tab_labels = ["Nyckeltext"] + section_names + ["Fulltext"]
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            st.text_area("Nyckeltext (domslut + domskal)", decision.key_text[:5000], height=400, disabled=True)

        for i, name in enumerate(section_names):
            with tabs[i + 1]:
                content = sections[name]
                st.text_area(name, content[:5000] if content else "(tom)", height=400, disabled=True)

        with tabs[-1]:
            st.text_area("Fulltext", decision.full_text[:10000], height=400, disabled=True)
    else:
        st.text_area("Nyckeltext", decision.key_text[:5000], height=400, disabled=True)


def render_prediction_detail(prediction):
    """Render detailed prediction results with probability bars."""
    label_sv = RISK_LABELS_SV.get(prediction.predicted_label, prediction.predicted_label)
    badge = risk_badge_html(prediction.predicted_label)
    st.markdown(f"**Prediktion**: {badge}", unsafe_allow_html=True)
    st.markdown(f"**Konfidens**: {prediction.confidence:.1%}")
    st.markdown(f"**Antal chunks**: {prediction.num_chunks}")

    if prediction.ground_truth:
        gt_sv = RISK_LABELS_SV.get(prediction.ground_truth, prediction.ground_truth)
        match_icon = "Korrekt" if prediction.predicted_label == prediction.ground_truth else "Avviker"
        gt_badge = risk_badge_html(prediction.ground_truth)
        st.markdown(f"**Faktisk**: {gt_badge} ({match_icon})", unsafe_allow_html=True)

    # Probability bars
    st.markdown("#### Sannolikheter")
    for label_name in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
        prob = prediction.probabilities.get(label_name, 0)
        css_class = RISK_BADGE_CSS.get(label_name, "")
        label_sv = RISK_LABELS_SV.get(label_name, label_name)
        width = max(int(prob * 100), 1)
        st.markdown(
            f'{label_sv}: {prob:.1%} '
            f'<div class="prob-bar {css_class.replace("risk-", "prob-")}" '
            f'style="width: {width}%"></div>',
            unsafe_allow_html=True,
        )

    # Chunk-level breakdown (collapsible)
    if prediction.chunk_predictions:
        with st.expander(f"Chunk-prediktion ({prediction.num_chunks} chunks)"):
            for cp in prediction.chunk_predictions:
                pred_label = cp["predicted_label"]
                probs = cp["probabilities"]
                dominant_prob = max(probs.values())
                st.markdown(
                    f"Chunk {cp['chunk_index']}: **{RISK_LABELS_SV.get(pred_label, pred_label)}** "
                    f"({dominant_prob:.1%})"
                )
