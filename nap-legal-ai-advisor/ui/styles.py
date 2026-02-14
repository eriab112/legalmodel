"""
CSS styling for NAP Legal AI Advisor.

Matches existing dashboard: primary #1E3A8A, risk colors (RED/AMBER/GREEN).
"""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: white;
    }
    .risk-high { background-color: #DC2626; }
    .risk-medium { background-color: #F59E0B; color: #1a1a1a; }
    .risk-low { background-color: #16A34A; }

    /* Result cards */
    .result-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 0.75rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s;
    }
    .result-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .result-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .result-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .result-card-meta {
        font-size: 0.85rem;
        color: #6B7280;
    }
    .result-card-excerpt {
        font-size: 0.9rem;
        color: #374151;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* Similarity score */
    .similarity-score {
        font-size: 0.8rem;
        color: #0D9488;
        font-weight: 600;
    }

    /* Metric cards */
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0D9488;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
    }

    /* Quick action buttons */
    .quick-action-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem 0;
    }

    /* Mode toggle */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* Prediction breakdown */
    .prob-bar {
        height: 20px;
        border-radius: 4px;
        margin: 2px 0;
    }
    .prob-high { background-color: #DC2626; }
    .prob-medium { background-color: #F59E0B; }
    .prob-low { background-color: #16A34A; }

    /* Sidebar styling */
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
    .sidebar-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }

    /* Overview dashboard */
    .overview-stat {
        text-align: center;
        padding: 0.5rem;
    }

    /* Explorer cards for non-decision documents */
    .doc-type-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .doc-type-legislation {
        background-color: #DBEAFE;
        color: #1E40AF;
    }
    .doc-type-application {
        background-color: #FEF3C7;
        color: #92400E;
    }
    .doc-type-decision {
        background-color: #F3F4F6;
        color: #374151;
    }
</style>
"""

RISK_BADGE_CSS = {
    "HIGH_RISK": "risk-high",
    "MEDIUM_RISK": "risk-medium",
    "LOW_RISK": "risk-low",
}

RISK_LABELS_SV = {
    "HIGH_RISK": "Hög risk",
    "MEDIUM_RISK": "Medelrisk",
    "LOW_RISK": "Låg risk",
}


def risk_badge_html(label: str) -> str:
    css_class = RISK_BADGE_CSS.get(label, "")
    label_sv = RISK_LABELS_SV.get(label, label or "Ej klassificerad")
    return f'<span class="risk-badge {css_class}">{label_sv}</span>'


def metric_card_html(value, label: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def doc_type_badge_html(doc_type: str) -> str:
    labels = {
        "decision": "📄 Beslut",
        "legislation": "📜 Lagstiftning",
        "application": "📋 Ansökan",
    }
    css = {
        "decision": "doc-type-decision",
        "legislation": "doc-type-legislation",
        "application": "doc-type-application",
    }
    label = labels.get(doc_type, doc_type)
    css_class = css.get(doc_type, "")
    return f'<span class="doc-type-badge {css_class}">{label}</span>'
