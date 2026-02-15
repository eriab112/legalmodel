"""
Extract structured metadata from all 50 court decisions.

Reads Data/processed/cleaned_court_texts.json, extracts fields like
application_outcome, power_plant_name, watercourse, operator_name,
measures_ordered, total_cost_sek, and processing_time_days.  Then writes
them back into cleaned_court_texts.json and propagates outcome fields to
labeled_dataset_binary.json.
"""

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data" / "processed"
CLEANED_PATH = DATA_DIR / "cleaned_court_texts.json"
BINARY_LABELED_PATH = DATA_DIR / "labeled_dataset_binary.json"

# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

OUTCOME_LABELS_SV = {
    "granted": "Tillstånd beviljas",
    "granted_modified": "Tillstånd med ändrade villkor",
    "remanded": "Återförvisat",
    "denied": "Ansökan/yrkande avslås",
    "appeal_denied": "Överklagande avslås",
    "overturned": "Upphäver beslut",
    "conditions_changed": "Villkor ändras",
    "dismissed": "Avvisas",
    "unclear": "Oklart",
}


def classify_outcome(domslut: str, is_appeal: bool = False) -> str:
    """Classify the primary outcome of a decision from its domslut text."""
    if not domslut:
        return "unclear"

    text = domslut.lower()

    # Normalize OCR artifacts: hyphens/newlines within words
    text = re.sub(r'(?<=\w)\s*-\s*\n\s*', '', text)
    text = re.sub(r'(?<=\w)\s*\n\s*(?=\w)', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # ---- Priority 1: återförvisar (always the main outcome) ----
    if re.search(r'(återförvisar|undanröjer.{0,80}återförvisar)', text):
        return "remanded"
    if re.search(r'undanröjer.{0,100}(fortsatt behandling|fortsatt handläggning)', text):
        return "remanded"

    # ---- Priority 2: For MÖD appeal decisions ----
    # Check if MÖD explicitly dismisses the appeal
    appeal_denied_pattern = re.search(
        r'(mark-?\s*och\s*miljö.?verdomstolen|möd)\s+avslår\s+överklagande',
        text,
    )
    mod_avslår_överklagandet = re.search(r'avslår\s+överklagandet', text)

    # Check if MÖD modifies the lower court decision
    mod_ändrar = re.search(
        r'(mark-?\s*och\s*miljö.?verdomstolen|möd)\s+ändrar\s+mark',
        text,
    )
    just_ändrar = re.search(r'ändrar\s+(mark|den\s+överklagade)', text)

    # "avslår överklagandet" AND "ändrar" together — the ändrar is the main ruling
    if mod_avslår_överklagandet and (mod_ändrar or just_ändrar):
        # MÖD denies appeal but also modifies some conditions
        if _is_conditions_change(text):
            return "conditions_changed"
        return "granted_modified"

    # Pure appeal denial (no ändrar)
    if mod_avslår_överklagandet or appeal_denied_pattern:
        return "appeal_denied"

    # ---- Priority 3: upphäver (without återförvisar already caught above) ----
    if re.search(r'upphäver', text):
        # Check if they also grant new tillstånd
        if re.search(r'(lämnar|ger)\s+\w*\s*tillstånd', text):
            return "granted_modified"
        if re.search(r'fastställer', text):
            return "overturned"
        return "overturned"

    # ---- Priority 4: ändrar dom (MÖD modifies lower court) ----
    if mod_ändrar or just_ändrar:
        if re.search(r'(lämnar|ger)\s+\w*\s*tillstånd', text):
            return "granted_modified"
        if _is_conditions_change(text):
            return "conditions_changed"
        return "granted_modified"

    # ---- Priority 5: granted ----
    if re.search(r'(lämnar|ger)\s+\w*\s*tillstånd', text):
        # Make sure it's not also denied
        if re.search(r'avslår\s+(ansökan|yrkande)', text):
            # Both appear — check which is the main ruling
            # If "lämnar tillstånd" appears before "avslår", it's the main ruling
            grant_pos = re.search(r'(lämnar|ger)\s+\w*\s*tillstånd', text).start()
            deny_pos = re.search(r'avslår\s+(ansökan|yrkande)', text).start()
            if grant_pos < deny_pos:
                return "granted"
            else:
                return "denied"
        return "granted"

    # ---- Priority 6: fastställer (establishes/confirms) ----
    if re.search(r'fastställer', text):
        return "conditions_changed"

    # ---- Priority 7: godkänner (approves) ----
    if re.search(r'godkänner', text):
        return "granted"

    # ---- Priority 8: förenas med (combined with conditions) ----
    if re.search(r'förenas\s+med', text):
        return "conditions_changed"

    # ---- Priority 9: denied ----
    if re.search(r'avslår\s+(ansökan|yrkande)', text):
        return "denied"

    # ---- Priority 10: dismissed ----
    if re.search(r'avvisar', text):
        # Check if avvisar is only for a side issue
        # If there's a main ruling elsewhere, skip
        return "dismissed"

    # ---- Priority 11: förlänger (extends) ----
    if re.search(r'förlänger', text):
        return "granted_modified"

    return "unclear"


def _is_conditions_change(text: str) -> bool:
    """Check if the domslut text is primarily about changing conditions."""
    cond_patterns = [
        r'villkor\s+\d+\s+ska\s+ha\s+följande\s+lydelse',
        r'ändrar.{0,60}villkor',
        r'ersätts\s+med\s+följande',
        r'ska\s+ha\s+följande\s+lydelse',
        r'punkten\s+\d+\s+ska\s+ha',
    ]
    return any(re.search(p, text) for p in cond_patterns)


# ---------------------------------------------------------------------------
# Power plant name
# ---------------------------------------------------------------------------

def extract_power_plant(saken: str, text_full: str) -> str | None:
    """Extract the power plant or facility name from the saken section."""
    search_text = saken or ""
    if not search_text:
        search_text = (text_full or "")[:1500]

    # Normalize whitespace and OCR artifacts
    search_text = re.sub(r'\s+', ' ', search_text)

    patterns = [
        # "vid X vattenkraftverk" or "vid X kraftverk"
        r'(?:vid|avseende|för)\s+([\w\s\-åäöÅÄÖ]+?)\s+(?:vatten)?kraftverk',
        # "X vattenkraftverk" at start
        r'([\w\-åäöÅÄÖ]+(?:\s+[\w\-åäöÅÄÖ]+)?)\s+(?:vatten)?kraftverk',
        # Dam cases
        r'(?:vid|avseende)\s+([\w\s\-åäöÅÄÖ]+?)\s+damm',
        # "regleringsdammen i X"
        r'regleringsdammen\s+i\s+([\w\s\-åäöÅÄÖ]+?)(?:\s+m\.m|\s+i\s)',
        # "anläggningen X"
        r'anläggningen\s+([\w\s\-åäöÅÄÖ]+?)(?:\s+med|\s+till|\s+i\s|\s+ska)',
    ]

    for pattern in patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean up
            name = re.sub(r'^(vid|avseende|till|för|om)\s+', '', name, flags=re.IGNORECASE)
            name = name.strip(' \t\n-.')
            # Skip names that are too generic or garbage
            if len(name) < 2 or len(name) > 60:
                continue
            if name.lower() in ('mark', 'den', 'ett', 'de', 'en'):
                continue
            return name

    return None


# ---------------------------------------------------------------------------
# Watercourse extraction
# ---------------------------------------------------------------------------

KNOWN_WATERCOURSES = [
    "Pinnån", "Voxnan", "Testeboån", "Klarälven", "Rönne å",
    "Lagan", "Nissan", "Ätran", "Viskan", "Göta älv",
    "Dalälven", "Ume älv", "Ljusnan", "Ljungan", "Indalsälven",
    "Ångermanälven", "Faxälven", "Skellefte älv", "Pite älv",
    "Lule älv", "Torne älv", "Kalix älv", "Emån", "Helge å",
    "Mörrumsån", "Motala ström", "Nyköpingsån", "Eskilstunaån",
    "Arbogaån", "Svartån", "Norrström", "Fyrisån",
    "Jörleån", "Suseån", "Loån", "Grytån", "Gryckån",
    "Västerån", "Tyresån", "Örekilsälven", "Bäveån", "Orust",
    "Mieån", "Ronnebyån",
]


def extract_watercourse(saken: str, bakgrund: str, text_full: str) -> str | None:
    """Extract the watercourse (river/stream) name from the decision text."""
    search_texts = [saken or "", bakgrund or "", (text_full or "")[:2000]]

    for search_text in search_texts:
        if not search_text:
            continue
        search_text = re.sub(r'\s+', ' ', search_text)

        # Check known watercourses first
        for wc in KNOWN_WATERCOURSES:
            if wc.lower() in search_text.lower():
                return wc

        # Pattern: "i Xån", "i X älv", "i X å"
        patterns = [
            r'i\s+([\w\-åäöÅÄÖ]+(?:ån|älv|å\b))',
            r'i\s+([\w\-åäöÅÄÖ]+\s+å)\b',
            r'i\s+([\w\-åäöÅÄÖ]+\s+älv)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) >= 3 and name.lower() not in ('i', 'en', 'på'):
                    return name

    return None


# ---------------------------------------------------------------------------
# Operator name
# ---------------------------------------------------------------------------

def extract_operator(saken: str, parties: dict, text_full: str) -> str | None:
    """Extract the operator/applicant name from the decision."""
    # Check parties for klagande (appellant) or sökande (applicant)
    if isinstance(parties, dict):
        for key in ['sökande', 'klagande']:
            vals = parties.get(key, [])
            if vals and isinstance(vals, list):
                for v in vals:
                    if isinstance(v, str) and len(v) > 2:
                        return v.strip()

    search_text = (text_full or "")[:3000]
    search_text = re.sub(r'\s+', ' ', search_text)

    # Company patterns
    company_patterns = [
        r'([\w\s\-åäöÅÄÖ]+(?:AB|Kraft\s+AB|Energi\s+AB|HB|KB))',
        r'SÖKANDE[:\s]+([\w\s\-åäöÅÄÖ\.]+?)(?:\n|,|$)',
    ]
    for pattern in company_patterns:
        match = re.search(pattern, search_text)
        if match:
            name = match.group(1).strip()
            if 5 < len(name) < 80:
                return name

    return None


# ---------------------------------------------------------------------------
# Structured measures
# ---------------------------------------------------------------------------

MEASURE_TYPES = [
    "fiskväg", "omlöp", "minimitappning", "utskov", "biotopvård",
    "faunapassage", "utrivning", "intagsgaller", "kontrollprogram",
    "ålpassage",
]


def extract_structured_measures(domslut: str) -> list[dict]:
    """Extract structured measures from the domslut section."""
    if not domslut:
        return []

    text = re.sub(r'\s+', ' ', domslut.lower())
    measures = []
    seen_types = set()

    for mtype in MEASURE_TYPES:
        if mtype in text:
            if mtype in seen_types:
                continue
            seen_types.add(mtype)
            # Try to extract details nearby
            details = _extract_measure_details(text, mtype)
            measures.append({"type": mtype, "details": details})

    return measures


def _extract_measure_details(text: str, measure_type: str) -> str | None:
    """Extract details for a specific measure type from surrounding context."""
    # Find the measure mention and look at surrounding text
    idx = text.find(measure_type)
    if idx == -1:
        return None

    context = text[max(0, idx - 50):idx + 200]

    # Look for numeric details
    detail_patterns = [
        r'(\d[\d\s,\.]*\s*(?:l/s|m3/s|m²|m2|mm|meter|m\b))',
        r'(minst\s+\d[\d\s,\.]*\s*(?:l/s|m3/s|m\b))',
        r'(spaltvidd\s+\d[\d\s,\.]*\s*mm)',
        r'(\d[\d\s,\.]*\s*(?:kr|kronor|sek))',
    ]

    for pattern in detail_patterns:
        match = re.search(pattern, context)
        if match:
            return match.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Cost extraction
# ---------------------------------------------------------------------------

def extract_total_cost(extracted_costs: list, text_full: str) -> int | None:
    """Extract the maximum/total cost from existing data and text search."""
    costs = []

    # Existing extracted costs
    for c in (extracted_costs or []):
        amt = c.get("amount_sek") or c.get("cost_sek")
        if amt and isinstance(amt, (int, float)) and amt > 0:
            costs.append(int(amt))

    # Search text for additional cost mentions
    search_text = (text_full or "")[:15000]
    search_text = re.sub(r'\s+', ' ', search_text)

    # Pattern: "X MSEK" or "X Mkr" or "X miljoner kronor"
    for match in re.finditer(r'(\d+[\s,]*\d*)\s*(?:miljoner?\s+kronor|MSEK|Mkr)', search_text, re.IGNORECASE):
        raw = match.group(1).replace(' ', '').replace(',', '.')
        try:
            costs.append(int(float(raw) * 1_000_000))
        except ValueError:
            pass

    # Pattern: "X kr" or "X kronor" or "X SEK"
    for match in re.finditer(r'(\d[\d\s]*)\s*(?:kr|kronor|SEK)\b', search_text):
        raw = match.group(1).replace(' ', '')
        try:
            val = int(raw)
            if val > 1000:  # Skip trivially small amounts
                costs.append(val)
        except ValueError:
            pass

    if costs:
        return max(costs)
    return None


# ---------------------------------------------------------------------------
# Processing time
# ---------------------------------------------------------------------------

SWEDISH_MONTHS = {
    "januari": 1, "februari": 2, "mars": 3, "april": 4,
    "maj": 5, "juni": 6, "juli": 7, "augusti": 8,
    "september": 9, "oktober": 10, "november": 11, "december": 12,
}


def _parse_swedish_date(date_str: str) -> datetime | None:
    """Parse a Swedish date like '15 mars 2023' or '2023-03-15'."""
    # ISO format
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    # Swedish format: "15 mars 2023"
    m = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = int(m.group(3))
        month = SWEDISH_MONTHS.get(month_name)
        if month:
            try:
                return datetime(year, month, day)
            except ValueError:
                return None

    return None


def extract_processing_time(decision_date: str, is_appeal: bool, text_full: str) -> int | None:
    """Compute approximate processing time in days."""
    if not decision_date:
        return None

    dec_date = _parse_swedish_date(decision_date)
    if not dec_date:
        return None

    search_text = (text_full or "")[:10000]
    search_text = re.sub(r'\s+', ' ', search_text)

    if is_appeal:
        # Look for lower court decision date
        patterns = [
            r'mark-?\s*och\s*miljödomstolens\s+(?:dom|deldom)\s+(?:den\s+)?(\d{1,2}\s+\w+\s+\d{4})',
            r'mark-?\s*och\s*miljödomstolens\s+(?:dom|deldom)\s+(?:den\s+)?(\d{4}-\d{2}-\d{2})',
            r'(?:dom|deldom)\s+(?:den\s+)?(\d{1,2}\s+\w+\s+\d{4})\s+i\s+mål',
            r'(?:dom|deldom)\s+(?:den\s+)?(\d{4}-\d{2}-\d{2})\s+i\s+mål',
        ]
    else:
        # Look for filing date
        patterns = [
            r'(?:inkom|anhängiggjord|ansökan\s+(?:den|inkom))\s+(?:den\s+)?(\d{1,2}\s+\w+\s+\d{4})',
            r'(?:inkom|anhängiggjord|ansökan\s+(?:den|inkom))\s+(?:den\s+)?(\d{4}-\d{2}-\d{2})',
        ]

    for pattern in patterns:
        for match in re.finditer(pattern, search_text, re.IGNORECASE):
            earlier_date = _parse_swedish_date(match.group(1))
            if earlier_date and earlier_date < dec_date:
                days = (dec_date - earlier_date).days
                if 10 < days < 5000:  # Sanity check
                    return days

    return None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def run_extraction():
    """Run the full metadata extraction pipeline."""
    print(f"Loading data from {CLEANED_PATH}")
    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    decisions = data["decisions"]
    print(f"Found {len(decisions)} decisions\n")

    outcomes = Counter()
    outcomes_by_court = {}
    plants_found = []
    plants_missing = []
    watercourses_found = []
    watercourses_missing = []
    operators_found = []
    cost_data = []
    time_data = []

    for d in decisions:
        meta = d.get("metadata", {})
        sections = d.get("sections", {})
        domslut = sections.get("domslut", "")
        saken = sections.get("saken", "")
        bakgrund = sections.get("bakgrund", "")
        text_full = d.get("text_full", "")
        is_appeal = meta.get("is_appeal", False)
        case_number = meta.get("case_number", d["id"])
        court = meta.get("court", "")

        # 1. Application outcome
        outcome = classify_outcome(domslut, is_appeal)
        meta["application_outcome"] = outcome
        meta["application_outcome_sv"] = OUTCOME_LABELS_SV.get(outcome, "Oklart")
        outcomes[outcome] += 1

        # Track by court (simplified)
        court_short = court.split("(")[0].strip() if court else "Unknown"
        if "MÖD" in court or "överdomstolen" in court:
            court_short = "MÖD"
        outcomes_by_court.setdefault(court_short, Counter())[outcome] += 1

        # 2. Power plant name
        plant = extract_power_plant(saken, text_full)
        meta["power_plant_name"] = plant
        if plant:
            plants_found.append((case_number, plant))
        else:
            plants_missing.append(case_number)

        # 3. Watercourse
        wc = extract_watercourse(saken, bakgrund, text_full)
        meta["watercourse"] = wc
        if wc:
            watercourses_found.append((case_number, wc))
        else:
            watercourses_missing.append(case_number)

        # 4. Operator name
        parties = meta.get("parties", {})
        operator = extract_operator(saken, parties, text_full)
        meta["operator_name"] = operator
        if operator:
            operators_found.append((case_number, operator))

        # 5. Structured measures
        structured = extract_structured_measures(domslut)
        meta["measures_ordered"] = structured

        # 6. Total cost
        total_cost = extract_total_cost(d.get("extracted_costs", []), text_full)
        meta["total_cost_sek"] = total_cost
        if total_cost:
            cost_data.append((case_number, total_cost))

        # 7. Processing time
        proc_time = extract_processing_time(
            meta.get("date"), is_appeal, text_full
        )
        meta["processing_time_days"] = proc_time
        if proc_time:
            time_data.append((case_number, proc_time))

    # Write back to cleaned_court_texts.json
    print("Writing updated data back to cleaned_court_texts.json...")
    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Propagate to labeled_dataset_binary.json
    _propagate_to_labeled(decisions)

    # Print summary
    _print_summary(
        outcomes, outcomes_by_court, plants_found, plants_missing,
        watercourses_found, watercourses_missing, operators_found,
        cost_data, time_data, len(decisions),
    )


def _propagate_to_labeled(decisions: list):
    """Propagate application_outcome fields to labeled_dataset_binary.json."""
    if not BINARY_LABELED_PATH.exists():
        print("WARNING: labeled_dataset_binary.json not found, skipping propagation.")
        return

    # Build lookup from cleaned decisions
    outcome_by_id = {}
    for d in decisions:
        meta = d.get("metadata", {})
        outcome_by_id[d["id"]] = {
            "application_outcome": meta.get("application_outcome"),
            "application_outcome_sv": meta.get("application_outcome_sv"),
        }

    with open(BINARY_LABELED_PATH, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    updated = 0
    for split_name in ["train", "val", "test"]:
        for item in labeled_data.get("splits", {}).get(split_name, []):
            dec_id = item["id"]
            if dec_id in outcome_by_id:
                item["metadata"]["application_outcome"] = outcome_by_id[dec_id]["application_outcome"]
                item["metadata"]["application_outcome_sv"] = outcome_by_id[dec_id]["application_outcome_sv"]
                updated += 1

    with open(BINARY_LABELED_PATH, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)

    print(f"Propagated outcomes to {updated} labeled decisions in labeled_dataset_binary.json\n")


def _print_summary(
    outcomes, outcomes_by_court, plants_found, plants_missing,
    watercourses_found, watercourses_missing, operators_found,
    cost_data, time_data, total,
):
    """Print a comprehensive summary of the extraction."""
    print("=" * 60)
    print("=== Application Outcomes ===")
    print("=" * 60)
    for outcome in [
        "granted", "granted_modified", "remanded", "denied",
        "appeal_denied", "overturned", "conditions_changed",
        "dismissed", "unclear",
    ]:
        count = outcomes.get(outcome, 0)
        if count > 0:
            print(f"  {outcome:25s} {count:3d} decisions")
    print(f"  {'TOTAL':25s} {sum(outcomes.values()):3d}")

    print(f"\n{'=' * 60}")
    print("=== Outcomes by Court ===")
    print("=" * 60)
    for court, court_outcomes in sorted(outcomes_by_court.items()):
        parts = [f"{o}={c}" for o, c in court_outcomes.most_common()]
        print(f"  {court:30s} {', '.join(parts)}")

    print(f"\n{'=' * 60}")
    print(f"=== Power Plants ===")
    print("=" * 60)
    print(f"  Extracted: {len(plants_found)} of {total}")
    for case, name in plants_found:
        print(f"    {case:20s} -> {name}")
    print(f"  Missing: {len(plants_missing)}")
    for case in plants_missing:
        print(f"    {case}")

    print(f"\n{'=' * 60}")
    print(f"=== Watercourses ===")
    print("=" * 60)
    print(f"  Extracted: {len(watercourses_found)} of {total}")
    unique_wc = sorted(set(wc for _, wc in watercourses_found))
    for wc in unique_wc:
        print(f"    {wc}")
    print(f"  Missing: {len(watercourses_missing)}")
    for case in watercourses_missing:
        print(f"    {case}")

    print(f"\n{'=' * 60}")
    print(f"=== Operators ===")
    print("=" * 60)
    print(f"  Extracted: {len(operators_found)} of {total}")
    for case, name in operators_found:
        print(f"    {case:20s} -> {name}")

    print(f"\n{'=' * 60}")
    print(f"=== Costs ===")
    print("=" * 60)
    print(f"  Decisions with cost data: {len(cost_data)}")
    if cost_data:
        costs_sorted = sorted(cost_data, key=lambda x: -x[1])
        min_cost = min(c for _, c in cost_data)
        max_cost = max(c for _, c in cost_data)
        print(f"  Range: {min_cost:,.0f} - {max_cost:,.0f} SEK")
        print(f"  Top 5:")
        for case, cost in costs_sorted[:5]:
            print(f"    {case:20s} {cost:>15,.0f} SEK")

    print(f"\n{'=' * 60}")
    print(f"=== Processing Time ===")
    print("=" * 60)
    print(f"  Decisions with time data: {len(time_data)}")
    if time_data:
        times = [t for _, t in time_data]
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"  Average: {avg_time:.0f} days")
        print(f"  Range: {min_time} - {max_time} days")
        for case, days in sorted(time_data, key=lambda x: -x[1]):
            print(f"    {case:20s} {days:5d} days")


if __name__ == "__main__":
    run_extraction()
