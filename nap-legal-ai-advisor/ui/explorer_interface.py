"""
Document explorer for NAP Legal AI Advisor.
Browse, search, and filter all documents in the knowledge base.
Includes interactive analysis tools moved from the overview dashboard.
"""

import statistics
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    "Låg risk": "LOW_RISK",
}

_OUTCOME_FILTER_MAP = {
    "Alla": None,
    "Tillstånd beviljat": "granted",
    "Villkor ändras": "conditions_changed",
    "Överklagande avslås": "appeal_denied",
    "Återförvisat": "remanded",
    "Ansökan avslås": "denied",
    "Upphävt": "overturned",
}

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


def _format_cost(value) -> str:
    """Format SEK cost: 'X.X Mkr' for >= 1M, 'X kkr' for >= 1000, else 'X kr'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} Mkr"
    if value >= 1_000:
        return f"{value / 1_000:.0f} kkr"
    return f"{value:.0f} kr"


def render_explorer(search_handler):
    """Render the document explorer interface."""
    # Decision detail redirect
    if st.session_state.get("show_decision_detail") and SharedContext.get_selected_decision():
        render_decision_detail(search_handler)
        return

    # --- Search bar (using st.form for reliable submit) ---
    with st.form("search_form"):
        search_col, btn_col = st.columns([4, 1])
        with search_col:
            query = st.text_input(
                "Sök i alla dokument",
                value=search_handler.get_search_query(),
                placeholder="Sök i alla dokument...",
                label_visibility="collapsed",
            )
        with btn_col:
            search_submitted = st.form_submit_button("Sök", type="primary")

    # --- Filter row (inline, 5 columns) ---
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)

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
        outcome_choice = st.selectbox(
            "Utfall",
            list(_OUTCOME_FILTER_MAP.keys()),
            key="explorer_outcome_filter",
        )

    with f_col5:
        sort_choice = st.selectbox(
            "Sortera efter",
            ["Relevans", "Datum (nyast först)", "Datum (äldst först)"],
            key="explorer_sort",
        )

    # Apply filters to SharedContext
    label_filter = _RISK_FILTER_MAP.get(risk_choice)
    court_filter = None if court_choice == "Alla" else court_choice
    doc_type_filter = _DOC_TYPE_MAP.get(doc_type_choice)
    outcome_filter = _OUTCOME_FILTER_MAP.get(outcome_choice)
    SharedContext.set_filters(label=label_filter, court=court_filter)

    # --- Execute search or show browse mode ---
    if search_submitted and query:
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

        # Apply outcome_filter client-side
        if outcome_filter:
            results = [r for r in results if r.metadata.get("application_outcome") == outcome_filter]

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
        _render_browse_mode(search_handler, outcome_filter)

    # --- Analysis tools section ---
    st.markdown("---")
    _render_analysis_tools(search_handler)


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
        court = result.metadata.get("originating_court") or result.metadata.get("court", "")
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


def _render_browse_mode(search_handler, outcome_filter=None):
    """Show a browseable table of all labeled decisions."""
    st.markdown("### Alla klassificerade beslut")

    decisions = search_handler.data.get_labeled_decisions()

    # Apply outcome filter
    if outcome_filter:
        decisions = [d for d in decisions if d.metadata.get("application_outcome") == outcome_filter]

    decisions_sorted = sorted(
        decisions,
        key=lambda d: d.metadata.get("date", ""),
        reverse=True,
    )

    risk_sv = {
        "HIGH_RISK": "Hög risk",
        "LOW_RISK": "Låg risk",
    }

    rows = []
    for d in decisions_sorted:
        rows.append({
            "Målnummer": d.metadata.get("case_number", d.id),
            "Domstol": d.metadata.get("originating_court") or d.metadata.get("court", ""),
            "Datum": d.metadata.get("date", ""),
            "Risknivå": risk_sv.get(d.label, d.label or ""),
            "Utfall": d.metadata.get("application_outcome_sv", ""),
            "Kraftverk": d.metadata.get("power_plant_name", ""),
            "Vattendrag": d.metadata.get("watercourse", ""),
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


def _render_analysis_tools(search_handler):
    """Render interactive analysis tools section."""
    data_loader = search_handler.data

    with st.expander("Analysverktyg", expanded=False):
        tool_tabs = st.tabs([
            "Jämför domstolar",
            "Filtrera och visualisera",
            "Tidsanalys",
            "Kostnadsanalys",
            "Åtgärdsanalys",
        ])

        all_decisions = data_loader.get_all_decisions()
        courts_list = data_loader.get_courts()

        # Build EN->SV outcome lookup
        outcome_en_to_sv = {}
        for d in all_decisions:
            en = d.metadata.get("application_outcome")
            sv = d.metadata.get("application_outcome_sv")
            if en and sv:
                outcome_en_to_sv[en] = sv
        sv_to_en = {v: k for k, v in outcome_en_to_sv.items()}

        # --- Tool 1: Jämför domstolar ---
        with tool_tabs[0]:
            _render_court_comparison(data_loader, all_decisions, courts_list, outcome_en_to_sv)

        # --- Tool 2: Filtrera och visualisera ---
        with tool_tabs[1]:
            _render_filter_viz(data_loader, all_decisions, courts_list, outcome_en_to_sv)

        # --- Tool 3: Tidsanalys ---
        with tool_tabs[2]:
            _render_time_analysis(data_loader, all_decisions)

        # --- Tool 4: Kostnadsanalys ---
        with tool_tabs[3]:
            _render_cost_analysis(data_loader, all_decisions)

        # --- Tool 5: Åtgärdsanalys ---
        with tool_tabs[4]:
            _render_measure_analysis(data_loader, all_decisions, courts_list)


def _render_court_comparison(data_loader, all_decisions, courts_list, outcome_en_to_sv):
    """Tool 1: Side-by-side court comparison."""
    st.markdown("#### Jämför domstolar")
    st.caption("Välj två domstolar för att jämföra utfall, handläggningstid och risknivå.")

    c1, c2 = st.columns(2)
    with c1:
        court_a = st.selectbox("Domstol A", courts_list, key="compare_court_a")
    with c2:
        court_b = st.selectbox(
            "Domstol B",
            [c for c in courts_list if c != court_a] if len(courts_list) > 1 else courts_list,
            key="compare_court_b",
        )

    if not court_a or not court_b:
        return

    def _decisions_for_court(court_name):
        results = []
        for d in all_decisions:
            c = d.metadata.get("originating_court") or d.metadata.get("court", "")
            if court_name in c:
                results.append(d)
        return results

    decs_a = _decisions_for_court(court_a)
    decs_b = _decisions_for_court(court_b)

    col_a, col_b = st.columns(2)

    for col, court_name, decs in [(col_a, court_a, decs_a), (col_b, court_b, decs_b)]:
        with col:
            st.markdown(f"**{court_name}** ({len(decs)} beslut)")

            # Outcome distribution
            outcome_counts = Counter()
            for d in decs:
                o = d.metadata.get("application_outcome")
                if o:
                    outcome_counts[o] += 1
            if outcome_counts:
                labels = [outcome_en_to_sv.get(k, k) for k in outcome_counts.keys()]
                values = list(outcome_counts.values())
                colors = [_outcome_color(k) for k in outcome_counts.keys()]
                fig = go.Figure(data=[go.Pie(
                    labels=labels, values=values,
                    hole=0.4, marker=dict(colors=colors),
                    textinfo="value+percent",
                )])
                fig.update_layout(
                    title="Utfall", height=280,
                    margin=dict(t=30, b=10, l=10, r=10),
                    showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Average processing time
            times = [d.metadata.get("processing_time_days") for d in decs
                     if d.metadata.get("processing_time_days") is not None]
            if times:
                st.metric("Snitt handläggningstid", f"{int(statistics.mean(times))} dagar")
            else:
                st.metric("Snitt handläggningstid", "\u2014")

            # Risk distribution
            risk_counts = Counter(d.label for d in decs if d.label)
            risk_sv = {"HIGH_RISK": "Hög risk", "LOW_RISK": "Låg risk"}
            if risk_counts:
                for label, count in risk_counts.most_common():
                    st.write(f"- {risk_sv.get(label, label)}: {count}")

            # Top measures
            measure_freq = Counter()
            for d in decs:
                measures = []
                if d.scoring_details:
                    measures = d.scoring_details.get("domslut_measures", [])
                if not measures:
                    measures = d.extracted_measures
                for m in measures:
                    measure_freq[m] += 1
            if measure_freq:
                st.markdown("**Vanligaste åtgärder:**")
                for m, cnt in measure_freq.most_common(5):
                    st.write(f"- {m} ({cnt})")


def _render_filter_viz(data_loader, all_decisions, courts_list, outcome_en_to_sv):
    """Tool 2: Filter and visualize."""
    st.markdown("#### Filtrera och visualisera")

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        fv_court = st.selectbox("Domstol", ["Alla"] + courts_list, key="fv_court")
    with fc2:
        fv_risk = st.selectbox("Risknivå", ["Alla", "Hög risk", "Låg risk"], key="fv_risk")
    with fc3:
        fv_outcome = st.selectbox("Utfall", list(_OUTCOME_FILTER_MAP.keys()), key="fv_outcome")
    with fc4:
        # Get all unique measures
        all_measures = set()
        for d in all_decisions:
            measures = []
            if d.scoring_details:
                measures = d.scoring_details.get("domslut_measures", [])
            if not measures:
                measures = d.extracted_measures
            all_measures.update(measures)
        fv_measure = st.selectbox("Åtgärd", ["Alla"] + sorted(all_measures), key="fv_measure")

    show_btn = st.button("Visa", key="fv_show", type="primary")

    if show_btn:
        filtered = list(all_decisions)

        if fv_court != "Alla":
            filtered = [d for d in filtered
                        if fv_court in (d.metadata.get("originating_court") or d.metadata.get("court", ""))]

        risk_map = {"Hög risk": "HIGH_RISK", "Låg risk": "LOW_RISK"}
        if fv_risk != "Alla":
            filtered = [d for d in filtered if d.label == risk_map.get(fv_risk)]

        outcome_val = _OUTCOME_FILTER_MAP.get(fv_outcome)
        if outcome_val:
            filtered = [d for d in filtered if d.metadata.get("application_outcome") == outcome_val]

        if fv_measure != "Alla":
            def _has_measure(d):
                measures = []
                if d.scoring_details:
                    measures = d.scoring_details.get("domslut_measures", [])
                if not measures:
                    measures = d.extracted_measures
                return fv_measure in measures
            filtered = [d for d in filtered if _has_measure(d)]

        total = len(all_decisions)
        n = len(filtered)
        pct = f"{n / total * 100:.1f}" if total > 0 else "0"
        st.markdown(f"**{n} beslut** ({pct}% av totalt {total})")

        if filtered:
            risk_sv = {"HIGH_RISK": "Hög risk", "LOW_RISK": "Låg risk"}
            rows = []
            for d in filtered:
                rows.append({
                    "Målnummer": d.metadata.get("case_number", d.id),
                    "Domstol": d.metadata.get("originating_court") or d.metadata.get("court", ""),
                    "Datum": d.metadata.get("date", ""),
                    "Risknivå": risk_sv.get(d.label, d.label or ""),
                    "Utfall": d.metadata.get("application_outcome_sv", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)

            # Bar chart of outcome distribution for filtered set
            outcome_counts = Counter()
            for d in filtered:
                o = d.metadata.get("application_outcome")
                if o:
                    outcome_counts[o] += 1
            if outcome_counts and len(outcome_counts) > 1:
                labels = [outcome_en_to_sv.get(k, k) for k in outcome_counts.keys()]
                values = list(outcome_counts.values())
                colors = [_outcome_color(k) for k in outcome_counts.keys()]
                fig = go.Figure(data=[go.Bar(
                    x=labels, y=values,
                    marker_color=colors,
                )])
                fig.update_layout(
                    title="Utfallsfördelning (filtrerat)",
                    height=300,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Inga beslut matchar filtren.")


def _render_time_analysis(data_loader, all_decisions):
    """Tool 3: Processing time analysis."""
    st.markdown("#### Tidsanalys")

    proc_times = data_loader.get_processing_times()
    if not proc_times:
        st.info("Inga handläggningstider tillgängliga.")
        return

    avg_days = int(statistics.mean(t[1] for t in proc_times))
    median_days = int(statistics.median(t[1] for t in proc_times))

    # Processing time bar chart (moved from overview)
    sorted_times = sorted(proc_times, key=lambda t: t[1], reverse=True)[:15]
    t_labels = [t[0] for t in sorted_times]
    t_values = [t[1] for t in sorted_times]

    # Look up court per case for coloring
    case_court = {}
    for d in all_decisions:
        cn = d.metadata.get("case_number", d.id)
        ct = d.metadata.get("originating_court") or d.metadata.get("court", "Okänd")
        ct = ct.split("(")[0].strip() if ct else "Okänd"
        case_court[cn] = ct

    unique_courts = sorted(set(case_court.get(lbl, "Okänd") for lbl in t_labels))
    court_palette = [
        "#1E3A8A", "#0D9488", "#7C3AED", "#DB2777",
        "#D97706", "#059669", "#4F46E5", "#BE185D",
    ]
    court_color_map = {
        c: court_palette[i % len(court_palette)]
        for i, c in enumerate(unique_courts)
    }
    t_colors = [court_color_map.get(case_court.get(lbl, "Okänd"), "#6B7280") for lbl in t_labels]

    fig = go.Figure(data=[go.Bar(
        x=t_values, y=t_labels,
        orientation="h", marker_color=t_colors,
    )])
    fig.update_layout(
        title="Handläggningstid (dagar)",
        height=max(380, len(t_labels) * 30),
        margin=dict(t=40, b=20, l=20, r=20),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="Dagar"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Snitt: {avg_days} dagar \u00b7 Median: {median_days} dagar "
        f"({len(proc_times)} beslut med tidsdata)"
    )

    # Timeline scatter plot: x=date, y=processing_time, color=risk
    st.markdown("##### Tidslinje")
    scatter_data = []
    for d in all_decisions:
        days = d.metadata.get("processing_time_days")
        date = d.metadata.get("date")
        if days is not None and date:
            scatter_data.append({
                "date": date,
                "days": days,
                "case": d.metadata.get("case_number", d.id),
                "risk": d.label or "unknown",
            })

    if scatter_data:
        risk_color_map = {
            "HIGH_RISK": "#DC2626",
            "LOW_RISK": "#16A34A",
            "unknown": "#6B7280",
        }
        risk_label_map = {
            "HIGH_RISK": "Hög risk",
            "LOW_RISK": "Låg risk",
            "unknown": "Oklassificerad",
        }

        fig = go.Figure()
        for risk_key in ["HIGH_RISK", "LOW_RISK", "unknown"]:
            subset = [s for s in scatter_data if s["risk"] == risk_key]
            if not subset:
                continue
            fig.add_trace(go.Scatter(
                x=[s["date"] for s in subset],
                y=[s["days"] for s in subset],
                mode="markers",
                name=risk_label_map.get(risk_key, risk_key),
                marker=dict(
                    color=risk_color_map.get(risk_key, "#6B7280"),
                    size=10,
                ),
                text=[s["case"] for s in subset],
                hovertemplate="%{text}<br>Datum: %{x}<br>Dagar: %{y}<extra></extra>",
            ))

        fig.update_layout(
            title="Handläggningstid per beslut",
            xaxis=dict(title="Datum"),
            yaxis=dict(title="Handläggningstid (dagar)"),
            height=400,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_cost_analysis(data_loader, all_decisions):
    """Tool 4: Cost analysis."""
    st.markdown("#### Kostnadsanalys")

    cost_data = []
    for d in all_decisions:
        cost = d.metadata.get("total_cost_sek")
        case = d.metadata.get("case_number", d.id)
        if cost is not None and cost > 0:
            cost_data.append({
                "case": case,
                "cost": cost,
                "risk": d.label or "unknown",
            })

    if not cost_data:
        st.info("Inga kostnadsdata tillgängliga.")
        return

    # Cost bar chart (moved from overview)
    sorted_costs = sorted(cost_data, key=lambda c: c["cost"], reverse=True)[:10]
    c_labels = [c["case"] for c in sorted_costs]
    c_values = [c["cost"] for c in sorted_costs]
    c_text = [_format_cost(v) for v in c_values]

    fig = go.Figure(data=[go.Bar(
        x=c_values, y=c_labels,
        orientation="h", marker_color="#7C3AED",
        text=c_text, textposition="outside",
    )])
    fig.update_layout(
        title="Uppskattad kostnad per beslut",
        height=max(380, len(c_labels) * 35),
        margin=dict(t=40, b=20, l=20, r=60),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title="SEK"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Baserat på kostnadsuppgifter i domstolstext")

    # Cost vs risk scatter plot
    st.markdown("##### Kostnad vs risknivå")
    risk_color_map = {
        "HIGH_RISK": "#DC2626",
        "LOW_RISK": "#16A34A",
        "unknown": "#6B7280",
    }
    risk_label_map = {
        "HIGH_RISK": "Hög risk",
        "LOW_RISK": "Låg risk",
        "unknown": "Oklassificerad",
    }

    fig = go.Figure()
    for risk_key in ["HIGH_RISK", "LOW_RISK", "unknown"]:
        subset = [c for c in cost_data if c["risk"] == risk_key]
        if not subset:
            continue
        fig.add_trace(go.Scatter(
            x=[s["case"] for s in subset],
            y=[s["cost"] for s in subset],
            mode="markers",
            name=risk_label_map.get(risk_key, risk_key),
            marker=dict(
                color=risk_color_map.get(risk_key, "#6B7280"),
                size=12,
            ),
            text=[_format_cost(s["cost"]) for s in subset],
            hovertemplate="%{x}<br>Kostnad: %{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Kostnad per beslut efter risknivå",
        xaxis=dict(title="Beslut"),
        yaxis=dict(title="Kostnad (SEK)"),
        height=400,
        margin=dict(t=40, b=20, l=20, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_measure_analysis(data_loader, all_decisions, courts_list):
    """Tool 5: Measure and watercourse analysis."""
    st.markdown("#### Åtgärdsanalys")

    # Optional court filter
    measure_court = st.selectbox(
        "Filtrera per domstol",
        ["Alla"] + courts_list,
        key="measure_court_filter",
    )

    filtered = all_decisions
    if measure_court != "Alla":
        filtered = [d for d in all_decisions
                    if measure_court in (d.metadata.get("originating_court") or d.metadata.get("court", ""))]

    # Measures frequency bar chart (moved from overview)
    measure_freq: Counter = Counter()
    _exclude = {"kontrollprogram", "skyddsgaller"}
    for d in filtered:
        measures = []
        if d.scoring_details:
            measures = d.scoring_details.get("domslut_measures", [])
        if not measures:
            measures = d.extracted_measures
        for m in measures:
            if m not in _exclude:
                measure_freq[m] += 1

    top_measures = measure_freq.most_common(10)

    if top_measures:
        measure_names = [m[0] for m in top_measures]
        measure_values = [m[1] for m in top_measures]

        fig = go.Figure(data=[go.Bar(
            x=measure_values, y=measure_names,
            orientation="h", marker_color="#0D9488",
        )])
        fig.update_layout(
            title="Vanligaste åtgärder",
            height=max(350, len(measure_names) * 35),
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Inga åtgärder tillgängliga.")

    # Watercourse distribution (moved from overview)
    st.markdown("##### Vattendrag")
    wc_counts: Counter = Counter()
    for d in filtered:
        wc = d.metadata.get("watercourse")
        if wc:
            wc_counts[wc] += 1

    multi_wc = {wc: cnt for wc, cnt in wc_counts.items() if cnt > 1}

    if multi_wc:
        watercourses = data_loader.get_watercourses()
        st.metric("Unika vattendrag", len(watercourses))
        st.caption(f"{sum(wc_counts.values())} beslut med vattendragsdata")

        sorted_wc = sorted(multi_wc.items(), key=lambda x: x[1], reverse=True)
        wc_labels = [w[0] for w in sorted_wc]
        wc_values = [w[1] for w in sorted_wc]

        fig = go.Figure(data=[go.Bar(
            x=wc_values, y=wc_labels,
            orientation="h", marker_color="#0284C7",
        )])
        fig.update_layout(
            title="Vattendrag med flera beslut",
            height=max(300, len(wc_labels) * 30),
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(dtick=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif wc_counts:
        st.write(f"{len(wc_counts)} unika vattendrag identifierade (alla med ett beslut vardera)")
    else:
        st.info("Inga vattendragsdata tillgängliga.")
