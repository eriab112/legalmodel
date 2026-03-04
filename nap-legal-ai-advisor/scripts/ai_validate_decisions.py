#!/usr/bin/env python3
"""
Generate flagged decisions review file for manual validation.

Outputs: Data/validation/flagged_decisions_review.txt
For each decision where ANY FLAG is TRUE, output a structured review block
sorted by flag count descending.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Path resolution (same logic as generate_validation_report.py)
# ---------------------------------------------------------------------------

def resolve_data_root(cli_arg=None):
    """Resolve the Data directory root."""
    if cli_arg:
        p = os.path.abspath(cli_arg)
        if os.path.isdir(p):
            return p
        print(f"ERROR: --data-root '{cli_arg}' is not a directory")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    marker = os.path.join("Data", "processed", "cleaned_court_texts.json")

    candidates = [
        os.getcwd(),
        os.path.join(script_dir, ".."),
        os.path.join(script_dir, "..", ".."),
        os.path.join(script_dir, "..", "..", ".."),
    ]
    for base in candidates:
        check = os.path.join(base, marker)
        if os.path.isfile(check):
            return os.path.abspath(os.path.join(base, "Data"))

    print("ERROR: Could not locate Data/processed/cleaned_court_texts.json")
    print("Use --data-root to specify the Data directory explicitly.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_strommen_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def normalize_case_number(cn):
    if not cn:
        return ""
    return cn.strip().replace(" ", "").replace("-", "").upper()


def build_label_lookup(labeled_data):
    lookup = {}
    for split_name in ("train", "val", "test"):
        items = labeled_data.get("splits", {}).get(split_name, [])
        for item in items:
            lookup[item["id"]] = {
                "label": item["label"],
                "scoring_details": item.get("scoring_details", {}),
            }
    return lookup


def build_strommen_lookup(strommen_rows):
    lookup = {}
    cn_counts = {}

    for row in strommen_rows:
        for col in ("Målnummer", "Målnummer MMÖD"):
            cn_raw = (row.get(col) or "").strip()
            if not cn_raw:
                continue
            ncn = normalize_case_number(cn_raw)
            if not ncn:
                continue
            cn_counts[ncn] = cn_counts.get(ncn, 0) + 1
            if ncn not in lookup:
                lookup[ncn] = {"row": row, "plant_count": 1}

    for ncn in lookup:
        if ncn in cn_counts:
            lookup[ncn]["plant_count"] = cn_counts[ncn]

    return lookup


def check_outcome_alignment(our_outcome, str_beslut):
    if not str_beslut or str_beslut.strip() == "Ej Beslut":
        return True

    beslut = str_beslut.strip()
    outcome = (our_outcome or "").strip()

    if beslut == "Avvisning":
        return outcome in ("denied", "appeal_denied", "unclear")
    elif beslut == "Moderna miljövillkor":
        return outcome in ("granted", "granted_modified", "conditions_changed")
    elif beslut == "Avslag":
        return outcome in ("denied",)
    elif beslut.startswith("Återkallande"):
        return True
    elif beslut == "Avskrivet":
        return True

    if outcome == "unclear":
        return False

    return True


# ---------------------------------------------------------------------------
# Flag computation
# ---------------------------------------------------------------------------

FLAG_NAMES = [
    "FLAG_cost_over_5M",
    "FLAG_outcome_unclear",
    "FLAG_no_measures",
    "FLAG_no_cost",
    "FLAG_no_plant_name",
    "FLAG_no_operator",
    "FLAG_no_processing_time",
    "FLAG_unlabeled",
    "FLAG_plant_name_prefix",
    "FLAG_outcome_mismatch",
]

# Short names for display
FLAG_SHORT = {
    "FLAG_cost_over_5M": "cost_over_5M",
    "FLAG_outcome_unclear": "outcome_unclear",
    "FLAG_no_measures": "no_measures",
    "FLAG_no_cost": "no_cost",
    "FLAG_no_plant_name": "no_plant_name",
    "FLAG_no_operator": "no_operator",
    "FLAG_no_processing_time": "no_processing_time",
    "FLAG_unlabeled": "unlabeled",
    "FLAG_plant_name_prefix": "plant_name_prefix",
    "FLAG_outcome_mismatch": "outcome_mismatch",
}

# Swedish outcome display names
OUTCOME_SV = {
    "granted": "Tillstånd beviljas",
    "granted_modified": "Tillstånd beviljas med ändringar",
    "denied": "Ansökan/yrkande avslås",
    "appeal_denied": "Överklagande avslås",
    "conditions_changed": "Villkorsändring",
    "remanded": "Återförvisat",
    "overturned": "Upphäver beslut",
    "unclear": "Oklart",
    "withdrawn": "Återkallat",
    "partially_granted": "Delvis bifall",
}


def compute_flags(decision, meta, label_info, str_match):
    """Compute all flags for a decision. Return dict of flag_name -> bool."""
    total_cost = meta.get("total_cost_sek")
    plant = meta.get("power_plant_name")
    operator = meta.get("operator_name")
    outcome = meta.get("application_outcome", "")

    measures_ordered = meta.get("measures_ordered", []) or []
    measures_ordered_str = ", ".join(
        m.get("type", "") for m in measures_ordered if m.get("type")
    )

    scoring = label_info.get("scoring_details", {})
    domslut_measures = scoring.get("domslut_measures", []) or []
    domslut_str = ", ".join(domslut_measures)

    extracted_measures = decision.get("extracted_measures", []) or []
    extracted_str = ", ".join(extracted_measures)

    flags = {}
    flags["FLAG_cost_over_5M"] = total_cost is not None and total_cost > 5_000_000
    flags["FLAG_outcome_unclear"] = outcome == "unclear"
    flags["FLAG_no_measures"] = (
        not measures_ordered_str and not domslut_str and not extracted_str
    )
    flags["FLAG_no_cost"] = total_cost is None or total_cost == 0
    flags["FLAG_no_plant_name"] = (
        plant is None or str(plant).strip() == "" or str(plant) == "None"
    )
    flags["FLAG_no_operator"] = (
        operator is None or str(operator).strip() == "" or str(operator) == "None"
    )
    flags["FLAG_no_processing_time"] = meta.get("processing_time_days") is None
    flags["FLAG_unlabeled"] = label_info.get("label", "") == ""
    flags["FLAG_plant_name_prefix"] = "moderna miljövillkor" in str(plant or "").lower()

    if str_match:
        str_beslut = str_match["row"].get("Beslut", "")
        flags["FLAG_outcome_mismatch"] = not check_outcome_alignment(outcome, str_beslut)
    else:
        flags["FLAG_outcome_mismatch"] = False

    return flags


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_value(val, dash="—"):
    """Format a value for display, replacing None/empty with dash."""
    if val is None:
        return dash
    s = str(val).strip()
    if s == "" or s == "None" or s == "0":
        return dash
    return s


def generate_review_block(idx, total, decision, meta, label_info,
                          str_match, flags, active_flags):
    """Generate a text review block for one flagged decision."""
    lines = []

    case_number = meta.get("case_number", "???")
    flag_names = ", ".join(FLAG_SHORT.get(f, f) for f in active_flags)

    lines.append("")
    lines.append("\u2550" * 66)
    lines.append(
        f"[{idx}/{total}]  {case_number}  |  FLAGS: {flag_names}"
    )
    lines.append("\u2550" * 66)

    # Current extraction
    outcome = meta.get("application_outcome", "")
    outcome_sv = meta.get("application_outcome_sv", "")
    if not outcome_sv:
        outcome_sv = OUTCOME_SV.get(outcome, "")

    total_cost = meta.get("total_cost_sek")
    cost_str = format_value(total_cost)

    extracted_costs = decision.get("extracted_costs", []) or []
    cost_source = ""
    if extracted_costs:
        ctx = extracted_costs[0].get("context", "")
        cost_source = ctx[:120] if ctx else ""
    cost_source_str = format_value(cost_source if cost_source else None)

    plant = format_value(meta.get("power_plant_name"))

    measures_ordered = meta.get("measures_ordered", []) or []
    measures_str = ", ".join(
        m.get("type", "") for m in measures_ordered if m.get("type")
    )

    scoring = label_info.get("scoring_details", {})
    domslut_measures = scoring.get("domslut_measures", []) or []
    domslut_str = ", ".join(domslut_measures)

    extracted_measures = decision.get("extracted_measures", []) or []
    extracted_str = ", ".join(extracted_measures)

    risk_label = label_info.get("label", "")

    lines.append("")
    lines.append("CURRENT EXTRACTION:")
    lines.append(f"  Outcome: {format_value(outcome)}")
    lines.append(f"  Outcome (sv): {format_value(outcome_sv)}")
    lines.append(f"  Cost: {cost_str}")
    lines.append(f"  Cost source: {cost_source_str}")
    lines.append(f"  Plant: {plant}")
    lines.append(f"  Measures ordered: {format_value(measures_str if measures_str else None)}")
    lines.append(f"  Domslut measures: {format_value(domslut_str if domslut_str else None)}")
    lines.append(f"  Extracted measures: {format_value(extracted_str if extracted_str else None)}")
    lines.append(f"  Risk label: {format_value(risk_label if risk_label else None)}")
    lines.append(f"  Court: {format_value(meta.get('court'))}")
    lines.append(f"  Is appeal: {meta.get('is_appeal', False)}")

    # Strommen cross-reference
    lines.append("")
    if str_match:
        sr = str_match["row"]
        lines.append("STRÖMMEN CROSS-REFERENCE:")
        lines.append("  \u2713 Matched in Strömmen")
        lines.append(f"  Plant: {format_value(sr.get('Vattenkraftverk'))}")
        lines.append(f"  Beslut: {format_value(sr.get('Beslut'))}")
        lines.append(f"  Court: {format_value(sr.get('Domstol'))}")
        lines.append(
            f"  Size: {format_value((sr.get('Vattenkraftverk storlek') or '').strip())}"
        )
        lines.append(f"  Prövningsgrupp: {format_value(sr.get('Prövningsgrupp'))}")
        lines.append(f"  Län: {format_value(sr.get('Län'))}")
    else:
        lines.append(
            "Not found in Strömmen (likely MÖD appeal or historical case)"
        )

    # Domslut excerpt
    sections = decision.get("sections", {}) or {}
    domslut_text = (sections.get("domslut") or "").strip()

    lines.append("")
    lines.append("DOMSLUT (first 2000 chars):")
    lines.append("\u2500" * 35)
    if domslut_text:
        lines.append(domslut_text[:2000])
    else:
        lines.append("Section not available")
    lines.append("\u2500" * 35)

    # Corrections template
    lines.append("")
    lines.append("CORRECTIONS (fill in):")
    lines.append("  [ ] Outcome \u2192 ___________")
    lines.append("  [ ] Cost \u2192 ___________")
    lines.append("  [ ] Plant name \u2192 ___________")
    lines.append("  [ ] Measures \u2192 ___________")
    lines.append("  [ ] Risk label OK? Y/N \u2192 ___")
    lines.append("  [ ] Notes: ___________")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate flagged decisions review file"
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Path to the Data directory (default: auto-detect)",
    )
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    print(f"Data root: {data_root}")

    # Load cleaned court texts
    ct_path = os.path.join(data_root, "processed", "cleaned_court_texts.json")
    print(f"Loading {ct_path} ...")
    ct_data = load_json(ct_path)
    decisions = ct_data.get("decisions", [])
    print(f"  Loaded {len(decisions)} decisions")

    # Load labeled dataset
    lb_path = os.path.join(data_root, "processed", "labeled_dataset_binary.json")
    print(f"Loading {lb_path} ...")
    lb_data = load_json(lb_path)
    label_lookup = build_label_lookup(lb_data)
    print(f"  Built label lookup for {len(label_lookup)} decisions")

    # Load Strommen files
    str_dir = os.path.join(data_root, "Strommen")
    str_files = []
    for f in sorted(os.listdir(str_dir)):
        if f.lower().endswith(".csv"):
            str_files.append(os.path.join(str_dir, f))

    if len(str_files) < 2:
        print(f"ERROR: Expected 2 CSV files in {str_dir}, found {len(str_files)}")
        sys.exit(1)

    all_strommen = []
    for fp in str_files:
        print(f"Loading {os.path.basename(fp)} ...")
        rows = load_strommen_csv(fp)
        all_strommen.extend(rows)
        print(f"  Loaded {len(rows)} rows")

    strommen_lookup = build_strommen_lookup(all_strommen)
    print(f"  Built Strömmen lookup with {len(strommen_lookup)} unique case numbers")

    # Process decisions — compute flags
    print("\nComputing flags ...")
    flagged_entries = []

    for dec in decisions:
        meta = dec.get("metadata", {})
        dec_id = dec.get("id", "")
        label_info = label_lookup.get(dec_id, {})

        case_number = meta.get("case_number", "")
        ncn = normalize_case_number(case_number)
        str_match = strommen_lookup.get(ncn)

        flags = compute_flags(dec, meta, label_info, str_match)
        active_flags = [f for f in FLAG_NAMES if flags.get(f)]

        if active_flags:
            flagged_entries.append({
                "decision": dec,
                "meta": meta,
                "label_info": label_info,
                "str_match": str_match,
                "flags": flags,
                "active_flags": active_flags,
                "flag_count": len(active_flags),
            })

    # Sort by flag count descending
    flagged_entries.sort(key=lambda x: -x["flag_count"])

    total_flagged = len(flagged_entries)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate output
    output_lines = []
    output_lines.append("NAP Legal AI Advisor \u2014 Data Validation Review")
    output_lines.append(f"Generated: {now_str}")
    output_lines.append("Strömmen source: 2025-07-02")
    output_lines.append(f"Total decisions: {len(decisions)}")
    output_lines.append(f"Flagged for review: {total_flagged}")
    output_lines.append("\u2550" * 66)

    for idx, entry in enumerate(flagged_entries, 1):
        block = generate_review_block(
            idx,
            total_flagged,
            entry["decision"],
            entry["meta"],
            entry["label_info"],
            entry["str_match"],
            entry["flags"],
            entry["active_flags"],
        )
        output_lines.append(block)

    output_text = "\n".join(output_lines) + "\n"

    # Write output
    out_dir = os.path.join(data_root, "validation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flagged_decisions_review.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    # Summary
    print(f"\n{'='*60}")
    print(f"REVIEW FILE GENERATED: {out_path}")
    print(f"{'='*60}")
    print(f"  Total decisions:     {len(decisions)}")
    print(f"  Flagged for review:  {total_flagged}")
    print(f"  Not flagged:         {len(decisions) - total_flagged}")
    print()

    # Flag breakdown
    flag_totals = {}
    for entry in flagged_entries:
        for f in entry["active_flags"]:
            flag_totals[f] = flag_totals.get(f, 0) + 1

    print("  Flag breakdown:")
    for fn in FLAG_NAMES:
        cnt = flag_totals.get(fn, 0)
        if cnt > 0:
            print(f"    {FLAG_SHORT.get(fn, fn):25s} {cnt}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
