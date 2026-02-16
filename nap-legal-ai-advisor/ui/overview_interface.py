"""
Overview dashboard for NAP Legal AI Advisor.
Shows key metrics, charts, and system status at a glance.
"""

import statistics

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

from integration.shared_context import SharedContext


# Outcome color mapping (EN key -> color)
_OUTCOME_COLORS = {
    "granted": "#16A34A",
    "granted_modified": "#16A34A",
    "conditions_changed": "#3B82F6",
    "denied": "#DC2626",
    "appeal_denied": "#DC2626",
    "remanded": "#F59E0B",
    "overturned": "#F59E0B",
    "unclear": "#6B7280",
}

# Canonical display order for stacked bar legend
_OUTCOME_ORDER = [
    "granted",
    "granted_modified",
    "conditions_changed",
    "denied",
    "appeal_denied",
    "remanded",
    "overturned",
    "unclear",
]


def _outcome_color(outcome_key: str) -> str:
    return _OUTCOME_COLORS.get(outcome_key, "#6B7280")


def render_overview(data_loader, knowledge_base=None):
    """Render the overview dashboard."""

    # --- Row 1: Key metrics (5 cards) ---
    if knowledge_base:
        stats = knowledge_base.get_corpus_stats()
        total_docs = stats.get("total", 0)
    else:
        total_docs = len(data_loader.get_all_decisions())

    dist = data_loader.get_label_distribution()
    total_classified = sum(dist.values())
    courts = data_loader.get_courts()
    date_min, date_max = data_loader.get_date_range()

    if date_min and date_max:
        year_min = date_min[:4]
        year_max = date_max[:4]
        date_range_str = f"{year_min}\u2013{year_max}"
    else:
        date_range_str = "\u2014"

    proc_times = data_loader.get_processing_times()
    if proc_times:
        avg_days = int(statistics.mean(t[1] for t in proc_times))
        avg_proc_str = f"{avg_days} dagar"
    else:
        avg_proc_str = "\u2014"

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Dokument i kunskapsbas", total_docs)
    with col2:
        st.metric("Klassificerade beslut", total_classified)
    with col3:
        st.metric("Domstolar", len(courts))
    with col4:
        st.metric("Datumintervall", date_range_str)
    with col5:
        st.metric("Snitt handläggningstid", avg_proc_str)

    st.markdown("")

    # --- Row 2: Risk distribution pie + Outcome by court stacked bar ---
    chart_left, chart_right = st.columns(2)

    with chart_left:
        color_map = {"HIGH_RISK": "#DC2626", "MEDIUM_RISK": "#F59E0B", "LOW_RISK": "#16A34A"}
        label_sv_map = {"HIGH_RISK": "Hög risk", "MEDIUM_RISK": "Medelrisk", "LOW_RISK": "Låg risk"}
        chart_labels = []
        chart_values = []
        chart_colors = []
        for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
            count = dist.get(label, 0)
            if count > 0:
                chart_labels.append(label_sv_map[label])
                chart_values.append(count)
                chart_colors.append(color_map[label])

        fig = go.Figure(data=[go.Pie(
            labels=chart_labels,
            values=chart_values,
            hole=0.4,
            marker=dict(colors=chart_colors),
            textinfo="value+percent",
        )])
        fig.update_layout(
            title="Riskfördelning",
            showlegend=True,
            height=350,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, width="stretch")

    with chart_right:
        # Build EN->SV outcome label lookup
        all_decisions = data_loader.get_all_decisions()
        outcome_en_to_sv = {}
        for d in all_decisions:
            en = d.metadata.get("application_outcome")
            sv = d.metadata.get("application_outcome_sv")
            if en and sv:
                outcome_en_to_sv[en] = sv

        # Outcomes by court – stacked bar chart
        outcomes_by_court = data_loader.get_outcomes_by_court()
        if outcomes_by_court:
            all_outcomes_in_data = set()
            for ctr in outcomes_by_court.values():
                all_outcomes_in_data.update(ctr.keys())

            court_totals = {
                c: sum(ctr.values()) for c, ctr in outcomes_by_court.items()
            }
            sorted_court_names = sorted(
                court_totals, key=court_totals.get, reverse=True
            )

            fig = go.Figure()
            for outcome_key in _OUTCOME_ORDER:
                if outcome_key not in all_outcomes_in_data:
                    continue
                sv_label = outcome_en_to_sv.get(outcome_key, outcome_key)
                counts = [
                    outcomes_by_court[c].get(outcome_key, 0)
                    for c in sorted_court_names
                ]
                fig.add_trace(go.Bar(
                    name=sv_label,
                    x=sorted_court_names,
                    y=counts,
                    marker_color=_outcome_color(outcome_key),
                ))

            fig.update_layout(
                title="Utfall per domstol",
                barmode="stack",
                height=350,
                margin=dict(t=40, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Inga utfallsdata per domstol tillgängliga.")

    # --- Row 3: Model performance (compact single row) ---
    st.markdown("### Modellprestanda")
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.metric("Accuracy", "80%")
    with perf_col2:
        st.metric("Hög risk recall", "100%")
    with perf_col3:
        st.metric("F1 (macro)", "0.80")
    with perf_col4:
        st.metric("Träningsdata", "44 beslut")
    st.caption("DAPT + fine-tuned KB-BERT på svenska miljödomstolsbeslut. Binär klassificering (Hög/Låg risk).")

    st.markdown("")

    # --- Row 4: Corpus composition ---
    if knowledge_base:
        st.markdown("### Kunskapsbas")
        stats = knowledge_base.get_corpus_stats()
        kb_col1, kb_col2, kb_col3 = st.columns(3)
        with kb_col1:
            st.metric("Beslut", stats.get("decision", 0))
            st.caption("domstolsbeslut")
        with kb_col2:
            st.metric("Lagstiftning", stats.get("legislation", 0))
            st.caption("lagar och riktlinjer")
        with kb_col3:
            st.metric("Ansökningar", stats.get("application", 0))
            st.caption("tillståndsansökningar")

    # --- Row 5: Clickable decisions table ---
    st.markdown("")
    st.markdown("### Beslut")
    st.caption("Klicka på ett beslut för att se detaljer i Utforska-fliken")

    risk_sv = {"HIGH_RISK": "Hög risk", "LOW_RISK": "Låg risk"}

    all_sorted = sorted(
        data_loader.get_all_decisions(),
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )

    rows = []
    for d in all_sorted:
        court = d.metadata.get("originating_court") or d.metadata.get("court", "")
        rows.append({
            "Målnummer": d.metadata.get("case_number", d.id),
            "Domstol": court,
            "Datum": d.metadata.get("date", ""),
            "Risknivå": risk_sv.get(d.label, "") if d.label else "",
            "Utfall": d.metadata.get("application_outcome_sv", ""),
            "Kraftverk": d.metadata.get("power_plant_name", ""),
            "id": d.id,
        })

    if rows:
        df = pd.DataFrame(rows)
        event = st.dataframe(
            df.drop(columns=["id"]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="overview_decision_table",
        )

        if event and event.selection and event.selection.rows:
            selected_idx = event.selection.rows[0]
            selected_id = rows[selected_idx]["id"]
            SharedContext.set_selected_decision(selected_id)
            st.session_state["show_decision_detail"] = True
            st.info(
                f"Valt beslut: **{rows[selected_idx]['Målnummer']}** — "
                f"gå till **Utforska**-fliken för att se detaljer."
            )

        # Fallback: text input for case number lookup
        case_input = st.text_input(
            "Visa detaljer för målnummer:",
            placeholder="t.ex. m483-22",
            key="overview_case_lookup",
        )
        if case_input:
            match = data_loader.get_decision(case_input.strip().lower())
            if match:
                SharedContext.set_selected_decision(match.id)
                st.session_state["show_decision_detail"] = True
                st.info(
                    f"Valt: **{match.metadata.get('case_number', match.id)}** — "
                    f"gå till **Utforska**-fliken för att se detaljer."
                )
            else:
                st.warning(f"Inget beslut hittades: {case_input}")
