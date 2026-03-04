"""
AI validation helper: generates a focused review document for flagged decisions.

Reads the cleaned court texts and labeled dataset, identifies decisions where
any validation FLAG is TRUE, and prints a structured review document with
current extractions and space for human corrections.

Output: Data/validation/flagged_decisions_review.txt
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_PATH = os.path.join(BASE_DIR, "Data", "processed", "cleaned_court_texts.json")
LABELED_PATH = os.path.join(BASE_DIR, "Data", "processed", "labeled_dataset_binary.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "validation")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "flagged_decisions_review.txt")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_map(labeled: dict) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for split in labeled.get("splits", {}).values():
        for entry in split:
            label_map[entry["id"]] = entry.get("label", "")
    return label_map


def build_scoring_map(labeled: dict) -> dict[str, dict]:
    scoring_map: dict[str, dict] = {}
    for split in labeled.get("splits", {}).values():
        for entry in split:
            if "scoring_details" in entry:
                scoring_map[entry["id"]] = entry["scoring_details"]
    return scoring_map


def cost_source_text(decision: dict) -> str:
    costs = decision.get("extracted_costs", [])
    if costs and isinstance(costs, list) and len(costs) > 0:
        return costs[0].get("context", "")
    return ""


def format_cost(amount) -> str:
    if amount is None:
        return "None"
    return f"{amount:,.0f} SEK"


def measures_list_str(measures) -> str:
    if not measures:
        return "[]"
    if isinstance(measures, list) and measures and isinstance(measures[0], dict):
        return "[" + ", ".join(m.get("type", "") for m in measures) + "]"
    return "[" + ", ".join(str(m) for m in measures) + "]"


def main():
    cleaned = load_json(CLEANED_PATH)
    labeled = load_json(LABELED_PATH)

    decisions = cleaned.get("decisions", [])
    label_map = build_label_map(labeled)
    scoring_map = build_scoring_map(labeled)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 80)
    lines.append("FLAGGED DECISIONS REVIEW")
    lines.append("Generated for manual validation of suspicious/incomplete extractions")
    lines.append("=" * 80)
    lines.append("")

    flagged_count = 0
    total_flags = 0

    # Sort by case_number
    decisions.sort(key=lambda d: d.get("metadata", {}).get("case_number", ""))

    for decision in decisions:
        meta = decision.get("metadata", {})
        sections = decision.get("sections", {})
        dec_id = decision.get("id", "")

        risk_label = label_map.get(dec_id, "")
        scoring = scoring_map.get(dec_id, {})

        case_number = meta.get("case_number", dec_id)
        application_outcome = meta.get("application_outcome", "")
        total_cost_sek = meta.get("total_cost_sek")
        measures_ord = meta.get("measures_ordered", [])
        domslut_meas = scoring.get("domslut_measures", [])
        extracted_meas = decision.get("extracted_measures", [])
        power_plant_name = meta.get("power_plant_name")
        operator_name = meta.get("operator_name")
        processing_time_days = meta.get("processing_time_days")

        # Compute flags
        flags = []
        if total_cost_sek is not None and total_cost_sek > 5_000_000:
            flags.append("cost_suspicious")
        if application_outcome == "unclear":
            flags.append("outcome_unclear")
        if not measures_ord and not domslut_meas and not extracted_meas:
            flags.append("no_measures")
        if total_cost_sek is None:
            flags.append("no_cost")
        if power_plant_name is None:
            flags.append("no_plant_name")
        if operator_name is None:
            flags.append("no_operator")
        if processing_time_days is None:
            flags.append("no_processing_time")
        if risk_label == "":
            flags.append("unlabeled")

        if not flags:
            continue

        flagged_count += 1
        total_flags += len(flags)

        # Build the review block
        domslut_text = sections.get("domslut", "")
        domslut_excerpt = domslut_text[:2000] if domslut_text else "(no domslut section)"

        cost_src = cost_source_text(decision)

        lines.append(f"=== {case_number} (FLAGS: {', '.join(flags)}) ===")
        lines.append("")
        lines.append("DOMSLUT (first 2000 chars):")
        lines.append(domslut_excerpt)
        lines.append("")
        lines.append("CURRENT EXTRACTION:")
        lines.append(f"  - outcome: {application_outcome}")
        lines.append(f"  - outcome_sv: {meta.get('application_outcome_sv', '')}")
        lines.append(f"  - cost: {format_cost(total_cost_sek)}" + (
            f' (source: "{cost_src[:120]}...")' if cost_src else ""
        ))
        lines.append(f"  - risk_label: {risk_label if risk_label else '(unlabeled)'}")
        lines.append(f"  - measures_ordered: {measures_list_str(measures_ord)}")
        lines.append(f"  - domslut_measures: {measures_list_str(domslut_meas)}")
        lines.append(f"  - extracted_measures: {measures_list_str(extracted_meas)}")
        lines.append(f"  - power_plant_name: {power_plant_name}")
        lines.append(f"  - operator_name: {operator_name}")
        lines.append(f"  - processing_time_days: {processing_time_days}")
        lines.append("")
        lines.append("SUGGESTED CORRECTIONS:")
        lines.append("[leave blank for human to fill]")
        lines.append("")
        lines.append("")

    # Summary header
    summary = [
        f"SUMMARY: {flagged_count} of {len(decisions)} decisions flagged "
        f"({total_flags} total flags)",
        "",
    ]

    # Insert summary at the top (after the header)
    lines.insert(5, summary[0])
    lines.insert(6, summary[1])

    output = "\n".join(lines)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Flagged decisions review saved to: {OUTPUT_PATH}")
    print(f"  Flagged decisions: {flagged_count} / {len(decisions)}")
    print(f"  Total flags: {total_flags}")


if __name__ == "__main__":
    main()
