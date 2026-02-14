"""
Overview dashboard for NAP Legal AI Advisor.
Shows key metrics, charts, and system status at a glance.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_overview(data_loader, knowledge_base=None):
    """Render the overview dashboard."""

    # --- Top metrics row ---
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dokument i kunskapsbas", total_docs)
    with col2:
        st.metric("Klassificerade beslut", total_classified)
    with col3:
        st.metric("Domstolar", len(courts))
    with col4:
        st.metric("Datumintervall", date_range_str)

    st.markdown("")

    # --- Two-column chart row ---
    chart_left, chart_right = st.columns(2)

    with chart_left:
        fig = go.Figure(data=[go.Pie(
            labels=["Hög risk", "Medelrisk", "Låg risk"],
            values=[
                dist.get("HIGH_RISK", 0),
                dist.get("MEDIUM_RISK", 0),
                dist.get("LOW_RISK", 0),
            ],
            hole=0.4,
            marker=dict(colors=["#DC2626", "#F59E0B", "#16A34A"]),
            textinfo="value+percent",
        )])
        fig.update_layout(
            title="Riskfördelning",
            showlegend=True,
            height=350,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_right:
        decisions = data_loader.get_labeled_decisions()
        court_counts = {}
        for d in decisions:
            court = d.metadata.get("court", "Okänd")
            court_counts[court] = court_counts.get(court, 0) + 1

        sorted_courts = sorted(court_counts.items(), key=lambda x: x[1], reverse=True)
        court_names = [c[0] for c in sorted_courts]
        court_values = [c[1] for c in sorted_courts]

        fig = go.Figure(data=[go.Bar(
            x=court_values,
            y=court_names,
            orientation="h",
            marker_color="#1E3A8A",
        )])
        fig.update_layout(
            title="Beslut per domstol",
            height=350,
            margin=dict(t=40, b=20, l=20, r=120),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Corpus composition ---
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

    st.markdown("")

    # --- Recent decisions table ---
    st.markdown("### Senaste klassificerade beslut")
    labeled = data_loader.get_labeled_decisions()
    labeled_sorted = sorted(
        labeled,
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )[:10]

    risk_sv = {
        "HIGH_RISK": "Hög risk",
        "MEDIUM_RISK": "Medelrisk",
        "LOW_RISK": "Låg risk",
    }

    rows = []
    for d in labeled_sorted:
        rows.append({
            "Målnummer": d.metadata.get("case_number", d.id),
            "Domstol": d.metadata.get("court", ""),
            "Datum": d.metadata.get("date", ""),
            "Risknivå": risk_sv.get(d.label, d.label or ""),
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Inga klassificerade beslut tillgängliga.")

    st.markdown("")

    # --- Measures frequency ---
    st.markdown("### Vanligaste åtgärder")
    measure_freq = data_loader.get_measure_frequency()
    top_measures = dict(list(measure_freq.items())[:10])

    if top_measures:
        measure_names = list(top_measures.keys())
        measure_values = list(top_measures.values())

        fig = go.Figure(data=[go.Bar(
            x=measure_values,
            y=measure_names,
            orientation="h",
            marker_color="#0D9488",
        )])
        fig.update_layout(
            height=max(350, len(measure_names) * 35),
            margin=dict(t=20, b=20, l=20, r=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Inga åtgärder tillgängliga.")
