"""
03_create_labeled_dataset.py - Training Data Creation for NAP LegalBERT

Creates labeled dataset with HIGH/MEDIUM/LOW risk classification:
1. Loads cleaned court texts from 02_clean_court_texts.py output
2. Auto-suggests labels using keyword heuristics + cost analysis
3. Generates a human-readable review file for manual verification
4. Creates stratified train/val/test splits (70/15/15)
5. Outputs labeled_dataset.json ready for model training

Labeling criteria:
  HIGH_RISK:   Unfavorable to kraftverk operator, expensive measures (>10 MSEK),
               strict timelines, utrivning ordered, tillstånd återkallat
  LOW_RISK:    Favorable to operator, minimal measures (<5 MSEK), long timelines,
               föreläggande upphävt, operator wins appeal
  MEDIUM_RISK: Mixed outcome, partial measures, moderate costs

Usage:
    python scripts/03_create_labeled_dataset.py
    # Review Data/processed/label_review.txt
    # Edit Data/processed/label_overrides.json if needed
    # Re-run to apply overrides
"""

import json
import os
import re
import random
from pathlib import Path
from collections import Counter

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path("Data/processed/cleaned_court_texts.json")
OUTPUT_FILE = Path("Data/processed/labeled_dataset.json")
REVIEW_FILE = Path("Data/processed/label_review.txt")
OVERRIDES_FILE = Path("Data/processed/label_overrides.json")

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ============================================================
# Relevance Check (hydropower/water operations only)
# ============================================================

VATTENKRAFT_KEYWORDS = [
    "vattenkraft", "kraftverk", "vattenverksamhet", "damm",
    "miljöbalken", "fiskväg", "faunapassage", "minimitappning",
    "miljövillkor", "omprövning", "nationella planen", "nap",
    "regleringsdamm", "vattenhushållning", "vattendom",
]


def is_vattenkraft_relevant(decision: dict) -> bool:
    """Check if a decision is about hydropower/water operations."""
    subject = (decision.get("metadata", {}).get("subject", "") or "").lower()
    full_text = decision.get("text_full", "")[:3000].lower()

    # Check subject line for clear non-vattenkraft topics
    non_relevant = [
        "fastighetsreglering", "fastighetsbildning",
        "utsläppsrätter", "täktverksamhet", "bergtäkt",
        "solcell", "vindkraft",
    ]
    for term in non_relevant:
        if term in subject:
            return False

    # Check if any vattenkraft keywords appear
    for kw in VATTENKRAFT_KEYWORDS:
        if kw in subject or kw in full_text:
            return True

    return False


# ============================================================
# Domslut-Focused Risk Scoring
# ============================================================

def classify_domslut_outcome(domslut: str) -> tuple:
    """Classify outcome type from domslut text.

    Returns (outcome_type, description).
    Only analyzes the VERDICT section, not discussions.
    """
    d = domslut.lower()

    # 1. Operator wins: föreläggande/beslut overturned
    if re.search(r'upphäver\s+.{0,60}(länsstyrelsen|beslut|föreläggande)', d):
        return ("OPERATOR_WINS", "Upphäver myndighets beslut/föreläggande")

    # 2. Appeal rejected (need to determine who appealed - checked in main scoring)
    if "avslår överklagandet" in d:
        return ("APPEAL_REJECTED", "Överklagandet avslås")

    # 3. Case remanded
    if "återförvisar" in d or ("undanröjer" in d and "återförvis" in d):
        return ("REMANDED", "Målet återförvisas")

    # 4. Minor changes only (typically just rättegångskostnader)
    if re.search(r'ändrar.{0,30}dom\w*\s+endast', d):
        return ("MINOR_CHANGE", "Ändrar dom i mindre delar")

    # 5. Permit granted with conditions
    if re.search(r'(lämnar|meddelar).{0,30}tillstånd', d):
        return ("PERMIT_GRANTED", "Tillstånd meddelas (med villkor)")

    # 6. Conditions modified / omprövning
    if re.search(r'ändrar.{0,40}(villkor|dom)', d):
        return ("CONDITIONS_MODIFIED", "Villkor/dom ändras")

    # 7. Extension granted
    if "förlänger" in d:
        return ("EXTENSION_GRANTED", "Förlängning beviljas")

    # 8. Application/appeal rejected
    if re.search(r'avslår.{0,30}(ansök|yrkande)', d):
        return ("REJECTED", "Ansökan/yrkande avslås")

    # 9. Avvisning (procedural)
    if "avvisas" in d or "avvisar" in d:
        return ("DISMISSED", "Avvisning (processuellt)")

    return ("UNCLEAR", "Kan ej avgöras automatiskt")


def compute_risk_score(decision: dict) -> dict:
    """Compute risk score based on DOMSLUT outcome analysis.

    Strategy: Focus on the actual verdict (domslut), not the discussion.
    The domslut determines if the operator won/lost/got conditions imposed.
    Only use measures/costs as secondary signals.
    """
    sections = decision.get("sections", {})
    domslut_text = sections.get("domslut", "")
    domskal_text = sections.get("domskäl", "")
    key_text = decision.get("key_text", "")

    # Step 1: Classify domslut outcome
    outcome_type, outcome_desc = classify_domslut_outcome(domslut_text)

    # Step 2: Check relevance
    relevant = is_vattenkraft_relevant(decision)

    # Step 3: Extract measures and costs
    measures = decision.get("extracted_measures", [])
    costs = decision.get("extracted_costs", [])
    max_cost = max((c["amount_sek"] for c in costs), default=0)
    n_measures = len(measures)
    severe_measures = {"utrivning", "fiskväg", "faunapassage"}
    has_severe = bool(severe_measures & set(measures))

    # Step 4: Count measures ACTUALLY IMPOSED in domslut (not just discussed)
    domslut_lower = domslut_text.lower()
    domslut_measures = []
    for m in ["fiskväg", "faunapassage", "omlöp", "minimitappning",
              "utskov", "kontrollprogram", "biotopvård", "skyddsgaller"]:
        if m in domslut_lower:
            domslut_measures.append(m)
    utrivning_in_domslut = "utrivning" in domslut_lower or "rivas" in domslut_lower

    # Step 5: Score based on outcome type
    signals = []
    score = 0  # positive = HIGH risk, negative = LOW risk

    if outcome_type == "OPERATOR_WINS":
        score -= 6
        signals.append("-6 Operator wins (upphäver beslut)")

    elif outcome_type == "APPEAL_REJECTED":
        # Need context: if operator appealed, this is bad for them
        # If myndighet appealed, it's good for operator
        subject = (decision.get("metadata", {}).get("subject", "") or "").lower()
        klagande_section = sections.get("parter", "").lower()[:500]

        if "länsstyrelsen" in klagande_section[:200]:
            # Myndigheten överklagade → operator vinner
            score -= 3
            signals.append("-3 Myndighets överklagande avslås (operator vinner)")
        else:
            # Operator överklagade → operator förlorar
            score += 4
            signals.append("+4 Operators överklagande avslås")

    elif outcome_type == "REMANDED":
        score += 0  # Neutral - uncertain outcome
        signals.append("0 Målet återförvisas (osäkert utfall)")

    elif outcome_type == "MINOR_CHANGE":
        score -= 2  # Slightly favorable - mostly unchanged
        signals.append("-2 Mindre ändring (dom i övrigt oförändrad)")

    elif outcome_type == "PERMIT_GRANTED":
        # Permit granted, but with what conditions?
        if utrivning_in_domslut:
            score += 3
            signals.append("+3 Tillstånd med utrivning")
        elif len(domslut_measures) >= 4:
            score += 2
            signals.append(f"+2 Tillstånd med många villkor ({len(domslut_measures)} åtgärder)")
        else:
            score -= 1
            signals.append("-1 Tillstånd meddelas (standard villkor)")

    elif outcome_type == "CONDITIONS_MODIFIED":
        if utrivning_in_domslut:
            score += 4
            signals.append("+4 Villkor med utrivning")
        elif len(domslut_measures) >= 4:
            score += 2
            signals.append(f"+2 Omfattande villkorsändring ({len(domslut_measures)} åtgärder)")
        else:
            score += 1
            signals.append("+1 Villkor ändras (måttligt)")

    elif outcome_type == "EXTENSION_GRANTED":
        score -= 3
        signals.append("-3 Förlängning beviljas (gynnsamt)")

    elif outcome_type == "REJECTED":
        # What was rejected? If operator's request → bad. If myndighet → good.
        subject = (decision.get("metadata", {}).get("subject", "") or "").lower()
        if "förlängning" in subject or "ansökan" in subject:
            score += 2
            signals.append("+2 Ansökan/förlängning avslås")
        else:
            score += 1
            signals.append("+1 Yrkande avslås")

    elif outcome_type == "DISMISSED":
        score += 0
        signals.append("0 Avvisning (processuellt)")

    else:  # UNCLEAR or NO_DOMSLUT
        # Fall back to text analysis
        if not domslut_text:
            signals.append("0 Inget domslut hittat (processhandling?)")
        else:
            signals.append("0 Utfall oklart")

    # Step 6: Secondary cost adjustment (only if in domslut)
    cost_signal = ""
    if max_cost > 10_000_000:
        score += 2
        cost_signal = f"HIGH: {max_cost/1e6:.1f} MSEK"
        signals.append(f"+2 Höga kostnader ({max_cost/1e6:.1f} MSEK)")
    elif max_cost > 5_000_000:
        score += 1
        cost_signal = f"ELEVATED: {max_cost/1e6:.1f} MSEK"
    elif max_cost > 0:
        cost_signal = f"{max_cost/1e3:.0f} kSEK"

    # Step 7: Determine suggested label
    if score >= 3:
        suggested_label = "HIGH_RISK"
    elif score <= -2:
        suggested_label = "LOW_RISK"
    else:
        suggested_label = "MEDIUM_RISK"

    # Step 8: Confidence
    if abs(score) >= 5:
        confidence = 0.85
    elif abs(score) >= 3:
        confidence = 0.70
    elif abs(score) >= 1:
        confidence = 0.55
    else:
        confidence = 0.40

    if not relevant:
        confidence = max(0.20, confidence - 0.2)
        signals.append("IRRELEVANT: Ej vattenkraft-relaterat")

    return {
        "suggested_label": suggested_label,
        "confidence": round(confidence, 2),
        "net_score": score,
        "outcome_type": outcome_type,
        "outcome_desc": outcome_desc,
        "signals": signals,
        "cost_signal": cost_signal,
        "num_measures": n_measures,
        "domslut_measures": domslut_measures,
        "measures": measures,
        "max_cost_sek": max_cost,
        "is_relevant": relevant,
    }


# ============================================================
# Domslut Summary (for review)
# ============================================================

def get_domslut_summary(decision: dict, max_chars: int = 500) -> str:
    """Get a short summary of the domslut for human review."""
    domslut = decision.get("sections", {}).get("domslut", "")
    if not domslut:
        # Fallback to key_text first lines
        key_text = decision.get("key_text", "")
        lines = key_text.split("\n")[:5]
        domslut = "\n".join(lines)

    # Trim to max_chars
    if len(domslut) > max_chars:
        domslut = domslut[:max_chars] + "..."

    return domslut


# ============================================================
# Stratified Train/Val/Test Split
# ============================================================

def stratified_split(labeled_data: list, ratios: dict, seed: int) -> dict:
    """Create stratified train/val/test split maintaining class distribution."""
    random.seed(seed)

    # Group by label
    by_label = {}
    for item in labeled_data:
        label = item["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(item)

    splits = {"train": [], "val": [], "test": []}

    for label, items in by_label.items():
        random.shuffle(items)
        n = len(items)

        n_val = max(1, round(n * ratios["val"]))
        n_test = max(1, round(n * ratios["test"]))
        n_train = n - n_val - n_test

        # Ensure at least 1 in each split if possible
        if n < 3:
            # Very few samples: put all in train, note the issue
            splits["train"].extend(items)
            continue

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])

    # Shuffle within splits
    for split_name in splits:
        random.shuffle(splits[split_name])

    return splits


# ============================================================
# Review File Generation
# ============================================================

def generate_review_file(decisions_with_scores: list, output_path: Path):
    """Generate human-readable review file."""
    lines = []
    lines.append("=" * 80)
    lines.append("NAP LegalBERT - LABEL REVIEW FILE")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Review each decision's suggested label below.")
    lines.append("To override a label, create 'Data/processed/label_overrides.json' with:")
    lines.append('  {"m1275-22": "HIGH_RISK", "m5295-23": "LOW_RISK", ...}')
    lines.append("Then re-run this script to apply overrides.")
    lines.append("")

    # Summary
    label_counts = Counter(d["scoring"]["suggested_label"] for d in decisions_with_scores)
    lines.append(f"SUGGESTED DISTRIBUTION:")
    for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(decisions_with_scores) if decisions_with_scores else 0
        lines.append(f"  {label:12s}: {count:2d} ({pct:.0f}%)")
    lines.append("")

    # Relevance warnings
    irrelevant = [d for d in decisions_with_scores if not d["scoring"].get("is_relevant", True)]
    if irrelevant:
        lines.append(f"NOT RELEVANT ({len(irrelevant)} decisions are NOT about vattenkraft):")
        for d in irrelevant:
            subject = d.get("metadata", {}).get("subject", "N/A")[:60]
            lines.append(f"  - {d['id']}: {subject}")
        lines.append("")

    # Low confidence flags
    low_conf = [d for d in decisions_with_scores if d["scoring"]["confidence"] < 0.5]
    if low_conf:
        lines.append(f"LOW CONFIDENCE ({len(low_conf)} decisions need extra review):")
        for d in low_conf:
            lines.append(f"  - {d['id']} (conf={d['scoring']['confidence']:.2f})")
        lines.append("")

    lines.append("=" * 80)
    lines.append("")

    # Per-decision details
    for i, d in enumerate(decisions_with_scores, 1):
        scoring = d["scoring"]
        meta = d.get("metadata", {})

        relevance = "RELEVANT" if scoring.get("is_relevant", True) else "NOT RELEVANT (ej vattenkraft)"
        lines.append(f"--- [{i:2d}/{len(decisions_with_scores)}] {d['id']} ---")
        lines.append(f"  File:       {d.get('filename', 'N/A')}")
        lines.append(f"  Case:       {meta.get('case_number', 'N/A')}")
        lines.append(f"  Date:       {meta.get('date', 'N/A')}")
        lines.append(f"  Court:      {meta.get('court', 'N/A')}")
        lines.append(f"  Subject:    {meta.get('subject', 'N/A')}")
        lines.append(f"  Relevance:  {relevance}")
        lines.append(f"")
        lines.append(f"  SUGGESTED:  {scoring['suggested_label']}  (confidence: {scoring['confidence']:.2f})")
        lines.append(f"  Outcome:    {scoring.get('outcome_type', 'N/A')} - {scoring.get('outcome_desc', '')}")
        lines.append(f"  Net score:  {scoring['net_score']:+d}")
        lines.append(f"  Max cost:   {scoring['max_cost_sek']/1e6:.2f} MSEK" if scoring['max_cost_sek'] > 0 else "  Max cost:   N/A")
        lines.append(f"  Measures:   {', '.join(scoring['measures']) if scoring['measures'] else 'None'}")
        if scoring.get("domslut_measures"):
            lines.append(f"  In domslut: {', '.join(scoring['domslut_measures'])}")
        lines.append(f"")

        if scoring.get("signals"):
            lines.append(f"  Signals:")
            for sig in scoring["signals"]:
                lines.append(f"    {sig}")

        lines.append(f"")
        lines.append(f"  DOMSLUT (excerpt):")
        domslut = get_domslut_summary(d)
        for line in domslut.split("\n")[:8]:
            lines.append(f"    {line.strip()}")

        lines.append(f"")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("NAP LegalBERT - Training Data Creation")
    print("=" * 60)

    # Load cleaned texts
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        print("Run 02_clean_court_texts.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    decisions = data["decisions"]
    print(f"\nLoaded {len(decisions)} decisions from {INPUT_FILE}")

    # Load overrides if available
    overrides = {}
    if OVERRIDES_FILE.exists():
        with open(OVERRIDES_FILE, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        print(f"Loaded {len(overrides)} manual label overrides from {OVERRIDES_FILE}")

    # Score each decision
    print("\nScoring decisions...")
    scored_decisions = []
    for d in decisions:
        scoring = compute_risk_score(d)
        d["scoring"] = scoring
        scored_decisions.append(d)

    # Apply overrides
    override_count = 0
    for d in scored_decisions:
        decision_id = d["id"]
        if decision_id in overrides:
            old_label = d["scoring"]["suggested_label"]
            new_label = overrides[decision_id]
            d["scoring"]["suggested_label"] = new_label
            d["scoring"]["override"] = True
            d["scoring"]["original_suggestion"] = old_label
            override_count += 1
            print(f"  Override: {decision_id}: {old_label} -> {new_label}")

    if override_count:
        print(f"  Applied {override_count} overrides")

    # Filter out EXCLUDE decisions
    excluded = [d for d in scored_decisions if d["scoring"]["suggested_label"] == "EXCLUDE"]
    scored_decisions = [d for d in scored_decisions if d["scoring"]["suggested_label"] != "EXCLUDE"]

    if excluded:
        print(f"\n  Excluded {len(excluded)} decisions:")
        for d in excluded:
            print(f"    - {d['id']}: {d.get('metadata', {}).get('subject', 'N/A')[:60]}")

    # Print distribution
    label_counts = Counter(d["scoring"]["suggested_label"] for d in scored_decisions)
    print(f"\nLabel distribution:")
    for label in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]:
        count = label_counts.get(label, 0)
        pct = 100 * count / len(scored_decisions)
        bar = "#" * count
        print(f"  {label:12s}: {count:2d} ({pct:4.1f}%) {bar}")

    # Generate review file
    generate_review_file(scored_decisions, REVIEW_FILE)
    print(f"\nReview file saved to: {REVIEW_FILE}")

    # Create labeled dataset
    labeled_data = []
    for d in scored_decisions:
        labeled_data.append({
            "id": d["id"],
            "filename": d["filename"],
            "label": d["scoring"]["suggested_label"],
            "confidence": d["scoring"]["confidence"],
            "key_text": d["key_text"],
            "sections": d["sections"],
            "metadata": d["metadata"],
            "scoring_details": {
                "net_score": d["scoring"]["net_score"],
                "outcome_type": d["scoring"]["outcome_type"],
                "outcome_desc": d["scoring"]["outcome_desc"],
                "max_cost_sek": d["scoring"]["max_cost_sek"],
                "measures": d["scoring"]["measures"],
                "domslut_measures": d["scoring"].get("domslut_measures", []),
                "is_relevant": d["scoring"].get("is_relevant", True),
                "override": d["scoring"].get("override", False),
            }
        })

    # Stratified split
    splits = stratified_split(labeled_data, SPLIT_RATIOS, RANDOM_SEED)

    print(f"\nSplit distribution:")
    for split_name, items in splits.items():
        split_labels = Counter(item["label"] for item in items)
        label_str = ", ".join(f"{l}={c}" for l, c in sorted(split_labels.items()))
        print(f"  {split_name:5s}: {len(items):2d} ({label_str})")

    # Build output
    output = {
        "version": "1.0",
        "created": "2026-02-06",
        "total_decisions": len(labeled_data),
        "label_distribution": dict(label_counts),
        "split_ratios": SPLIT_RATIOS,
        "random_seed": RANDOM_SEED,
        "overrides_applied": override_count,
        "excluded_decisions": [d["id"] for d in excluded],
        "splits": {
            name: [
                {
                    "id": item["id"],
                    "filename": item["filename"],
                    "label": item["label"],
                    "confidence": item["confidence"],
                    "key_text": item["key_text"],
                    "metadata": item["metadata"],
                    "scoring_details": item["scoring_details"],
                }
                for item in items
            ]
            for name, items in splits.items()
        },
        "all_labeled": [
            {
                "id": item["id"],
                "filename": item["filename"],
                "label": item["label"],
                "confidence": item["confidence"],
                "scoring_details": item["scoring_details"],
                "is_relevant": item["scoring_details"].get("is_relevant", True),
            }
            for item in labeled_data
        ],
    }

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / (1024*1024):.1f} MB")

    # Low confidence warnings
    low_conf = [d for d in labeled_data if d["confidence"] < 0.5]
    if low_conf:
        print(f"\nWARNING: {len(low_conf)} decisions have low confidence (<0.5):")
        for d in low_conf:
            print(f"  - {d['id']}: {d['label']} (conf={d['confidence']:.2f})")
        print(f"\nPlease review these in {REVIEW_FILE}")
        print(f"Add overrides to {OVERRIDES_FILE} and re-run if needed.")

    print(f"\nNext steps:")
    print(f"  1. Review {REVIEW_FILE}")
    print(f"  2. Create {OVERRIDES_FILE} with any corrections")
    print(f"  3. Re-run this script to apply overrides")
    print(f"  4. Proceed to model fine-tuning (scripts/05_finetune_legalbert.py)")


if __name__ == "__main__":
    main()
