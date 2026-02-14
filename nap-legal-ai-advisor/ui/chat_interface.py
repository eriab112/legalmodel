"""
Chat mode UI for NAP Legal AI Advisor.

Provides conversational Q&A with quick action buttons and
result cards for decision display.
"""

import streamlit as st

from integration.shared_context import SharedContext
from ui.styles import risk_badge_html, RISK_LABELS_SV


def render_chat_mode(chat_handler):
    """Render the chat mode interface."""
    chat_handler.initialize_chat()

    # Display chat history
    for message in SharedContext.get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Quick action buttons (only show when chat has just the welcome message)
    if len(SharedContext.get_messages()) <= 1:
        render_quick_actions(chat_handler)

    # Chat input
    if prompt := st.chat_input("Ställ en fråga om miljödomstolsbeslut..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyserar..."):
                response = chat_handler.process_message(prompt)
                st.markdown(response)


def render_quick_actions(chat_handler):
    """Render quick action buttons for common queries."""
    actions = chat_handler.get_quick_actions()
    cols = st.columns(len(actions))
    for col, action in zip(cols, actions):
        with col:
            if st.button(
                f"{action['icon']} {action['label']}",
                key=f"qa_{action['label']}",
                use_container_width=True,
            ):
                # Process quick action
                with st.chat_message("user"):
                    st.markdown(action["query"])
                with st.chat_message("assistant"):
                    with st.spinner("Analyserar..."):
                        response = chat_handler.process_message(action["query"])
                        st.markdown(response)
                st.rerun()


def render_decision_card(decision, prediction=None):
    """Render a compact decision card with risk badge."""
    case = decision.metadata.get("case_number", decision.id)
    date = decision.metadata.get("date", "")
    court = decision.metadata.get("court", "")

    badge = risk_badge_html(decision.label) if decision.label else ""
    st.markdown(
        f"""<div class="result-card">
        <div class="result-card-header">
            <span class="result-card-title">{case}</span>
            {badge}
        </div>
        <div class="result-card-meta">{court} | {date}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    if prediction:
        render_prediction_summary(prediction)


def render_prediction_summary(prediction):
    """Render a compact prediction summary."""
    label_sv = RISK_LABELS_SV.get(prediction.predicted_label, prediction.predicted_label)
    st.markdown(f"**LegalBERT prediktion**: {label_sv} (konfidens: {prediction.confidence:.1%})")

    if prediction.ground_truth:
        gt_sv = RISK_LABELS_SV.get(prediction.ground_truth, prediction.ground_truth)
        match = "Korrekt" if prediction.predicted_label == prediction.ground_truth else "Avviker"
        st.markdown(f"**Faktisk klassificering**: {gt_sv} ({match})")
