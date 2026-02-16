"""
Search mode UI for NAP Legal AI Advisor.

Provides semantic search with result cards, filter sidebar,
decision detail view with AI summary, key data, risk prediction,
full text sections, and similar decisions.
"""

import streamlit as st

from integration.shared_context import SharedContext
from ui.styles import risk_badge_html, RISK_LABELS_SV, RISK_BADGE_CSS

# Constant used as default value in metadata lookups (em-dash)
_DASH = "—"


def _generate_decision_summary(decision, llm_engine):
    """Generate a structured summary of a decision using Gemini."""
    sections = decision.sections or {}
    case_num = decision.metadata.get("case_number", decision.id)
    court = decision.metadata.get("originating_court") or decision.metadata.get("court", "")

    # Build context from the decision's own text
    context_parts = []
    for section_name in ["domslut", "domskäl", "bakgrund", "saken"]:
        text = sections.get(section_name, "")
        if text:
            context_parts.append(f"--- {section_name.upper()} ---\n{text[:3000]}")
    if not context_parts:
        context_parts.append(decision.full_text[:6000])

    context = "\n\n".join(context_parts)

    summary_prompt = (
        f"Sammanfatta domstolsbeslut {case_num} ({court}) kortfattat och strukturerat.\n\n"
        "Använd EXAKT denna struktur:\n"
        "**Vad handlar målet om:** [1-2 meningar om vad ärendet gäller]\n\n"
        "**Domstolens beslut:** [1-2 meningar om vad domstolen beslutade]\n\n"
        "**Åtgärder som krävs:** [Punktlista med de viktigaste åtgärderna/villkoren]\n\n"
        "**Konsekvenser för verksamhetsutövaren:** [1-2 meningar om praktiska konsekvenser]\n\n"
        "Basera sammanfattningen ENBART på den tillhandahållna texten. Var kortfattad och konkret."
    )

    summary_system_prompt = (
        "Du är en juridisk sammanfattare specialiserad på svenska miljödomstolsbeslut. "
        "Din uppgift är att skapa korta, strukturerade sammanfattningar av domstolsbeslut. "
        "Svara ALLTID på svenska. Var koncis — max 200 ord total. Använd den exakta strukturen som efterfrågas."
    )

    sources = [{"index": 1, "title": f"{case_num} — {court}",
                "doc_type": "decision", "type_label": "beslut",
                "doc_id": decision.id, "filename": decision.filename}]

    response = llm_engine.generate_response(
        summary_prompt, context, sources,
        system_prompt_override=summary_system_prompt,
    )
    return response


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
            "Sök i domstolsbeslut",
            value=search_handler.get_search_query(),
            placeholder="T.ex. faunapassage vid vattenkraftverk, minimitappning...",
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button("Sök", type="primary", width="stretch")

    # Filters sidebar
    render_filters_sidebar(search_handler)

    # Execute search
    if search_clicked and query:
        with st.spinner("Söker..."):
            results = search_handler.execute_search(query)
    else:
        results = search_handler.get_search_results()

    # Display results
    if results:
        st.markdown(f"**{len(results)} resultat** för *{search_handler.get_search_query()}*")
        for result in results:
            render_result_card(result, search_handler)
    elif search_handler.get_search_query():
        st.info("Inga resultat hittades. Prova andra sökord.")
    else:
        st.info("Använd sökfältet ovan för att söka i alla dokument om vattenkraft.")


def render_filters_sidebar(search_handler):
    """Render filter controls in the sidebar."""
    with st.sidebar:
        st.markdown("### Filter")

        available = search_handler.get_available_filters()

        # Risk level filter
        label_options = ["Alla"] + available["labels"]
        label_display = {
            "Alla": "Alla risknivåer",
            "HIGH_RISK": "Hög risk",
            "MEDIUM_RISK": "Medelrisk",
            "LOW_RISK": "Låg risk",
        }
        selected_label = st.selectbox(
            "Risknivå",
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
        if st.button("Rensa filter", width="stretch"):
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
    """Render full decision detail view with progressive disclosure."""
    decision_id = SharedContext.get_selected_decision()
    decision = search_handler.get_decision_detail(decision_id)

    if not decision:
        st.error(f"Beslut '{decision_id}' hittades inte.")
        return

    # Back button
    if st.button("← Tillbaka"):
        SharedContext.set_selected_decision(None)
        st.rerun()

    case = decision.metadata.get("case_number", decision.id)
    court = decision.metadata.get("originating_court") or decision.metadata.get("court", "")
    date = decision.metadata.get("date", "")
    risk_label = decision.label

    # --- Header with badge ---
    badge = risk_badge_html(risk_label) if risk_label else ""
    st.markdown(f"## {case} {badge}", unsafe_allow_html=True)
    st.markdown(f"**{court}** · {date}")

    # --- AI-generated summary ---
    st.markdown("### Sammanfattning")

    cache_key = f"summary_{decision_id}"
    if cache_key in st.session_state:
        st.markdown(st.session_state[cache_key])
    else:
        llm_engine = st.session_state.get("llm_engine")
        if llm_engine:
            with st.spinner("Genererar sammanfattning..."):
                summary = _generate_decision_summary(decision, llm_engine)
                st.session_state[cache_key] = summary
            st.markdown(summary)
        else:
            st.info("AI-sammanfattning kräver Gemini API-nyckel.")
            fallback = decision.key_text[:1000] if decision.key_text else "Ingen nyckeltext tillgänglig."
            st.markdown(fallback)

    # --- Key data cards ---
    st.markdown("### Nyckeldata")
    data_cols = st.columns(4)

    with data_cols[0]:
        outcome_sv = decision.metadata.get("application_outcome_sv", "")
        st.metric("Utfall", outcome_sv or _DASH)

    with data_cols[1]:
        measures = []
        if decision.scoring_details:
            measures = decision.scoring_details.get("domslut_measures", [])
        if not measures:
            measures = decision.extracted_measures or []
        st.metric("Antal åtgärder", len(measures))
        if measures:
            st.caption(", ".join(measures[:5]))

    with data_cols[2]:
        cost = decision.metadata.get("total_cost_sek")
        if cost is None and decision.scoring_details:
            cost = decision.scoring_details.get("max_cost_sek")
        if cost is not None:
            if cost >= 1_000_000:
                cost_str = f"{cost / 1_000_000:.1f} Mkr"
            elif cost >= 1_000:
                cost_str = f"{cost / 1_000:.0f} kkr"
            else:
                cost_str = f"{cost:.0f} kr"
            st.metric("Uppskattad kostnad", cost_str)
        else:
            st.metric("Uppskattad kostnad", _DASH)

    with data_cols[3]:
        proc_time = decision.metadata.get("processing_time_days")
        if proc_time is not None:
            st.metric("Handläggningstid", f"{int(proc_time)} dagar")
        else:
            st.metric("Handläggningstid", _DASH)

    # --- Additional metadata ---
    with st.expander("📋 Ytterligare uppgifter", expanded=False):
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            watercourse = decision.metadata.get("watercourse", _DASH)
            st.markdown(f"**Vattendrag:** {watercourse}")
            plant = decision.metadata.get("power_plant_name", _DASH)
            st.markdown(f"**Kraftverk:** {plant}")
            operator = decision.metadata.get("operator_name", _DASH)
            st.markdown(f"**Operatör:** {operator}")
            court_level = decision.metadata.get("court_level", _DASH)
            st.markdown(f"**Instans:** {court_level}")
        with meta_col2:
            is_appeal = decision.metadata.get("is_appeal", False)
            appeal_text = "Ja" if is_appeal else "Nej"
            st.markdown(f"**Överklagande:** {appeal_text}")
            if is_appeal:
                orig = decision.metadata.get("originating_court", "")
                st.markdown(f"**Ursprungsdomstol:** {orig}")
            subject = decision.metadata.get("subject", _DASH)
            st.markdown(f"**Ämne:** {subject}")
            if decision.linked_water_bodies:
                wbs = ", ".join(
                    wb.get("water_body", "") for wb in decision.linked_water_bodies if wb.get("water_body")
                )
                st.markdown(f"**Vattenförekomster (VISS):** {wbs}")

    # --- Risk prediction ---
    with st.expander("🤖 LegalBERT Riskprediktion", expanded=False):
        run_pred = st.session_state.get(f"run_prediction_{decision_id}", False)

        if run_pred or st.button("Kör prediktion", key=f"run_pred_{decision_id}"):
            st.session_state.pop(f"run_prediction_{decision_id}", None)
            with st.spinner("Kör LegalBERT-inference..."):
                prediction = search_handler.get_risk_prediction(decision_id)
            if prediction:
                render_prediction_detail(prediction)
        else:
            cached = SharedContext.get_cached_prediction(decision_id)
            if cached:
                render_prediction_detail(cached)
            else:
                st.info("Klicka för att köra LegalBERT-prediktion på detta beslut.")

        # Show ground truth if available
        if decision.label:
            gt_badge = risk_badge_html(decision.label)
            st.markdown(f"**Faktisk klassificering:** {gt_badge}", unsafe_allow_html=True)

    # --- Full decision text ---
    with st.expander("📄 Fullständigt beslut", expanded=False):
        sections = decision.sections or {}
        section_names = list(sections.keys())

        if section_names:
            tab_labels = section_names + ["Fulltext"]
            tabs = st.tabs(tab_labels)

            for i, name in enumerate(section_names):
                with tabs[i]:
                    content = sections[name]
                    if content:
                        st.markdown(f"**{name.upper()}**")
                        st.text_area(
                            name, content[:8000],
                            height=400, disabled=True,
                            label_visibility="collapsed",
                            key=f"section_{decision_id}_{name}",
                        )
                    else:
                        st.info(f"Sektionen '{name}' är tom.")

            with tabs[-1]:
                st.text_area(
                    "Fulltext", decision.full_text[:15000],
                    height=500, disabled=True,
                    label_visibility="collapsed",
                    key=f"fulltext_{decision_id}",
                )
        else:
            st.text_area(
                "Fulltext", decision.full_text[:15000],
                height=500, disabled=True,
                label_visibility="collapsed",
                key=f"fulltext_only_{decision_id}",
            )

    # --- Similar decisions ---
    st.markdown("### Liknande beslut")
    st.caption("Baserat på semantisk likhet i beslutstexter.")

    search_engine = st.session_state.get("search_engine")
    if search_engine:
        similar_results = search_engine.find_similar_decisions(decision_id, n_results=5)
        if similar_results:
            for r in similar_results:
                sim_case = r.metadata.get("case_number", r.decision_id)
                sim_court = r.metadata.get("originating_court") or r.metadata.get("court", "")
                sim_date = r.metadata.get("date", "")
                sim_badge = risk_badge_html(r.label) if r.label else ""
                sim_pct = f"{r.similarity:.0%}"

                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"**{sim_case}** · {sim_court} · {sim_date} · {sim_pct} likhet {sim_badge}",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("Öppna", key=f"similar_{r.decision_id}"):
                        SharedContext.set_selected_decision(r.decision_id)
                        st.rerun()
        else:
            st.info("Inga liknande beslut hittades.")
    else:
        st.info("Sökmotor ej tillgänglig.")


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
    for label_name in prediction.probabilities:
        prob = prediction.probabilities[label_name]
        css_class = RISK_BADGE_CSS.get(label_name, "")
        label_sv = RISK_LABELS_SV.get(label_name, label_name)
        width = max(int(prob * 100), 1)
        bar_class = css_class.replace("risk-", "prob-")
        st.markdown(
            f'{label_sv}: {prob:.1%} '
            f'<div class="prob-bar {bar_class}" '
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
                pred_label_sv = RISK_LABELS_SV.get(pred_label, pred_label)
                st.markdown(
                    f"Chunk {cp['chunk_index']}: **{pred_label_sv}** "
                    f"({dominant_prob:.1%})"
                )
