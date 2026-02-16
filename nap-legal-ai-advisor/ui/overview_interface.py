"""
Overview dashboard for NAP Legal AI Advisor.
Guided, pedagogical dashboard with dynamic visual builder.
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


def _compute_viz_data(data_loader, group_by, measure):
    """Compute visualization data based on user selections."""
    decisions = data_loader.get_all_decisions()

    # Build groups
    groups = {}  # group_label -> list of decisions

    if group_by == "Domstol":
        for d in decisions:
            court = d.metadata.get("originating_court") or d.metadata.get("court", "Okänd")
            court_short = court.split(",")[0].strip()
            groups.setdefault(court_short, []).append(d)

    elif group_by == "Utfall":
        for d in decisions:
            outcome = d.metadata.get("application_outcome_sv") or "Okänt"
            groups.setdefault(outcome, []).append(d)

    elif group_by == "Risknivå":
        risk_sv = {"HIGH_RISK": "Hög risk", "LOW_RISK": "Låg risk"}
        for d in decisions:
            if d.label:
                label = risk_sv.get(d.label, d.label)
                groups.setdefault(label, []).append(d)

    elif group_by == "Vattendrag":
        for d in decisions:
            wc = d.metadata.get("watercourse")
            if wc:
                groups.setdefault(wc, []).append(d)

    elif group_by == "Åtgärd":
        for d in decisions:
            measures_list = (
                (d.scoring_details or {}).get("domslut_measures", [])
                or d.extracted_measures
                or []
            )
            for m in measures_list:
                if m not in ("kontrollprogram", "skyddsgaller"):
                    groups.setdefault(m, []).append(d)

    elif group_by == "År":
        for d in decisions:
            date = d.metadata.get("date", "")
            if date and len(date) >= 4:
                year = date[:4]
                groups.setdefault(year, []).append(d)

    # Compute measure for each group
    result = {}

    if measure == "Antal beslut":
        result = {k: len(v) for k, v in groups.items()}

    elif measure == "Genomsnittlig kostnad (SEK)":
        for k, v in groups.items():
            costs = []
            for d in v:
                c = d.metadata.get("total_cost_sek")
                if c is None and d.scoring_details:
                    c = d.scoring_details.get("max_cost_sek")
                if c is not None:
                    costs.append(float(c))
            if costs:
                result[k] = sum(costs) / len(costs)

    elif measure == "Genomsnittlig handläggningstid (dagar)":
        for k, v in groups.items():
            times = []
            for d in v:
                t = d.metadata.get("processing_time_days")
                if t is not None:
                    times.append(int(t))
            if times:
                result[k] = sum(times) / len(times)

    # Sort by value descending
    result = dict(sorted(result.items(), key=lambda x: -x[1]))
    return result


def render_overview(data_loader, knowledge_base=None):
    """Render the overview dashboard."""

    # =====================================================================
    # Section 1: Welcome + User Guide
    # =====================================================================
    st.markdown("### Välkommen till NAP Legal AI Advisor")
    st.markdown(
        "Ett AI-drivet beslutsstöd för vattenkraftens miljöanpassning. "
        "Systemet analyserar **50 domstolsbeslut**, **37 lagar och riktlinjer**, "
        "och **26 tillståndsansökningar** för att hjälpa dig förstå risker, "
        "kostnader och vanliga åtgärder inom NAP-processen."
    )

    with st.expander("ℹ️ Användarguide — Hur fungerar systemet?", expanded=False):
        guide_tabs = st.tabs([
            "Vad betyder hög/låg risk?",
            "Tre AI-agenter",
            "Vad kan jag göra?",
            "Hur tolkar jag resultaten?",
        ])

        with guide_tabs[0]:
            st.markdown(
                "Riskklassificeringen mäter **risk för regulatorisk påverkan på "
                "verksamhetsutövaren** — i vilken grad ett domstolsbeslut ålägger nya "
                "miljökrav, verksamhetsbegränsningar och ekonomiska kostnader.\n\n"
                "- **HÖG RISK** — Domstolen ställer krav på omfattande och kostsamma "
                "miljöåtgärder, begränsar verksamheten eller ålägger betydande villkor.\n"
                "- **LÅG RISK** — Status quo bevaras i stort, mindre justeringar. "
                "Verksamhetsutövaren påverkas marginellt.\n\n"
                "Modellen har **100 % recall för hög risk** — den missar aldrig ett "
                "farligt beslut — och **80 % övergripande träffsäkerhet**."
            )

        with guide_tabs[1]:
            st.markdown(
                "Systemet har tre specialiserade AI-agenter som samarbetar:\n\n"
                "🏛️ **Domstolsagent** — Söker i 50 domstolsbeslut och ansökningar. "
                "Svarar på frågor om specifika mål, jämför beslut och identifierar mönster.\n\n"
                "📜 **Svensk rättsagent** — Söker i svensk miljölagstiftning "
                "(miljöbalken, NAP, HaV:s riktlinjer, tekniska vägledningar).\n\n"
                "🇪🇺 **EU-agent** — Söker i EU:s ramdirektiv för vatten, "
                "CIS-vägledningsdokument.\n\n"
                "Systemet dirigerar automatiskt din fråga till rätt agent, "
                "eller kombinerar flera vid behov."
            )

        with guide_tabs[2]:
            st.markdown(
                "**📊 Översikt** (du är här) — Se nyckeltal, utforska data visuellt, "
                "bygg egna diagram.\n\n"
                "**🔍 Utforska** — Sök i alla dokument, filtrera beslut, "
                "se detaljer och kör riskprediktioner.\n\n"
                "**💬 AI-assistent** — Ställ frågor på naturligt språk, "
                "beskriv ditt kraftverk för en riskbedömning, jämför beslut.\n\n"
                "---\n"
                "**Exempel på frågor till AI-assistenten:**\n"
                '- "Vad säger miljöbalken om fiskvägar?"\n'
                '- "Jag har ett medelstort vattenkraftverk utanför Gävle, bedöm min risk"\n'
                '- "Jämför M 3753-22 med M 605-24"\n'
                '- "Vilken domstol nekar flest ansökningar?"\n'
                '- "Senaste 3 besluten vid Växjö med fiskväg"'
            )

        with guide_tabs[3]:
            st.markdown(
                "Det här är ett **indikativt verktyg** baserat på historiska "
                "domstolsbeslut, inte juridisk rådgivning.\n\n"
                "Riskbedömningen bygger på mönstermatchning mot liknande ärenden. "
                'Om modellen säger "hög risk" kan du lita på det — modellen missar '
                'aldrig ett farligt beslut. Om den säger "låg risk" stämmer det i de '
                "flesta fall, men det finns en liten chans att risken är högre.\n\n"
                "**Konsultera alltid en jurist** för specifika beslut."
            )

    st.markdown("")

    # =====================================================================
    # Section 2: Kunskapsbasen — compact metrics row
    # =====================================================================
    if knowledge_base:
        stats = knowledge_base.get_corpus_stats()
        n_decisions = stats.get("decision", 0)
        n_legislation = stats.get("legislation", 0)
        n_applications = stats.get("application", 0)
    else:
        n_decisions = len(data_loader.get_all_decisions())
        n_legislation = 0
        n_applications = 0

    courts = data_loader.get_courts()
    date_min, date_max = data_loader.get_date_range()
    if date_min and date_max:
        date_range_str = f"{date_min[:4]}\u2013{date_max[:4]}"
    else:
        date_range_str = "\u2014"

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Beslut", n_decisions)
    with m2:
        st.metric("Lagstiftning", n_legislation)
    with m3:
        st.metric("Ansökningar", n_applications)
    with m4:
        st.metric("Domstolar", len(courts))
    with m5:
        st.metric("Period", date_range_str)

    st.markdown("")

    # =====================================================================
    # Section 3: Nyckelinsikter — two guided charts
    # =====================================================================
    st.markdown("### Nyckelinsikter")
    st.markdown(
        "Översikt av riskfördelning och utfall i domstolsbeslut om "
        "vattenkraftens miljöanpassning."
    )

    chart_left, chart_right = st.columns(2)

    # --- Left: Riskfördelning (donut) ---
    with chart_left:
        dist = data_loader.get_label_distribution()
        total_classified = sum(dist.values())
        high_count = dist.get("HIGH_RISK", 0)
        high_pct = round(100 * high_count / total_classified) if total_classified else 0

        st.markdown(
            f"**Riskfördelning** — Av {total_classified} klassificerade beslut bedöms "
            f"{high_pct} % som hög risk — domstolen ställer krav på omfattande och "
            f"kostsamma miljöåtgärder."
        )

        color_map = {
            "HIGH_RISK": "#DC2626",
            "MEDIUM_RISK": "#F59E0B",
            "LOW_RISK": "#16A34A",
        }
        label_sv_map = {
            "HIGH_RISK": "Hög risk",
            "MEDIUM_RISK": "Medelrisk",
            "LOW_RISK": "Låg risk",
        }
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
            showlegend=True,
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Right: Utfall per domstol (stacked bar) ---
    with chart_right:
        st.markdown(
            "**Utfall per domstol** — Fördelning av utfall (beviljat, avslaget, "
            "villkor ändrade, etc.) per mark- och miljödomstol. "
            "Baserat på ursprungsdomstol, inte överklagandeinstans."
        )

        all_decisions = data_loader.get_all_decisions()
        outcome_en_to_sv = {}
        for d in all_decisions:
            en = d.metadata.get("application_outcome")
            sv = d.metadata.get("application_outcome_sv")
            if en and sv:
                outcome_en_to_sv[en] = sv

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
                barmode="stack",
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Inga utfallsdata per domstol tillgängliga.")

    st.markdown("")

    # =====================================================================
    # Section 4: Bygg din egen analys — Dynamic Visual Builder
    # =====================================================================
    st.markdown("### 📊 Bygg din egen analys")
    st.markdown(
        "Välj dimensioner nedan för att skapa anpassade visualiseringar av datan."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        group_by = st.selectbox(
            "Gruppera efter",
            ["Domstol", "Utfall", "Risknivå", "Vattendrag", "Åtgärd", "År"],
            key="viz_group_by",
        )
    with col2:
        measure = st.selectbox(
            "Visa",
            [
                "Antal beslut",
                "Genomsnittlig kostnad (SEK)",
                "Genomsnittlig handläggningstid (dagar)",
            ],
            key="viz_measure",
        )
    with col3:
        chart_type = st.selectbox(
            "Diagramtyp",
            ["Stapeldiagram", "Cirkeldiagram", "Tabell"],
            key="viz_chart_type",
        )

    viz_data = _compute_viz_data(data_loader, group_by, measure)

    if not viz_data:
        st.info("Ingen data tillgänglig för denna kombination.")
    else:
        if chart_type == "Stapeldiagram":
            labels = list(viz_data.keys())
            values = list(viz_data.values())
            fig = go.Figure(data=[go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#1E3A8A",
            )])
            fig.update_layout(
                height=max(350, len(labels) * 30),
                margin=dict(t=20, b=40, l=20, r=20),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title=measure),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Cirkeldiagram":
            fig = go.Figure(data=[go.Pie(
                labels=list(viz_data.keys()),
                values=list(viz_data.values()),
                hole=0.4,
                textinfo="value+percent",
            )])
            fig.update_layout(
                height=400,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Tabell":
            df_viz = pd.DataFrame({
                group_by: list(viz_data.keys()),
                measure: [f"{v:,.0f}" for v in viz_data.values()],
            })
            st.dataframe(df_viz, use_container_width=True, hide_index=True)

        # Contextual caption
        n_groups = len(viz_data)
        if measure == "Antal beslut":
            total = sum(viz_data.values())
            st.caption(f"{n_groups} kategorier, {total:.0f} beslut totalt")
        elif "kostnad" in measure.lower():
            st.caption(
                f"{n_groups} kategorier med kostnadsdata. "
                "Baserat på uppgifter i domstolstext."
            )
        elif "handläggningstid" in measure.lower():
            st.caption(f"{n_groups} kategorier med tidsdata.")

    st.markdown("")

    # =====================================================================
    # Section 5: Alla beslut — clickable decisions table
    # =====================================================================
    st.markdown("### Alla beslut")
    st.caption(
        "Klicka på en rad för att öppna beslutet i Utforska-fliken, "
        "eller ange ett målnummer nedan."
    )

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

    st.markdown("")

    # =====================================================================
    # Section 6: Om AI-modellen
    # =====================================================================
    with st.expander("🤖 Om AI-modellen", expanded=False):
        st.markdown("""
**LegalBERT** är en specialanpassad språkmodell (KB-BERT) som tränats på svenska 
miljödomstolsbeslut för att klassificera risknivå.

**Hur bra är den?**
- ✅ Identifierar **alla högriskbeslut** korrekt (100% recall) — modellen missar aldrig ett farligt beslut
- 🎯 Övergripande träffsäkerhet: **80%** — 4 av 5 beslut klassificeras rätt
- ⚠️ Ibland flaggar den lågriskbeslut som hög risk (konservativ bias) — bättre att varna för mycket än för lite

**Vad betyder det i praktiken?**
Om modellen säger "hög risk" kan du lita på det. Om den säger "låg risk" stämmer det 
i de flesta fall, men det finns en liten chans att risken är högre. 
Modellen är designad som ett **screeningverktyg** — den hjälper dig prioritera vilka 
ärenden som behöver noggrannare granskning.

**Tekniska detaljer**: DAPT + fine-tuned KB-BERT, binär klassificering (Hög/Låg risk), 
tränad på 44 beslut med 80% accuracy och F1 (macro) 0.80.
""")
