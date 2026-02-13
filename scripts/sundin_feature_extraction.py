#!/usr/bin/env python3
"""
Sundin-informed feature extraction (read-only on existing data).

Reads:
  - Data/processed/cleaned_court_texts.json
  - Data/processed/labeled_dataset.json

Writes (NEW file only, never overwrites existing pipeline data):
  - Data/processed/decision_features_sundin2026.json

Sundin et al. 2026 defines WHAT to look for; this script extracts
values (gap width, cost, timeline, etc.) so downstream code can
learn weights from the 40/44 labeled decisions. Safe to run anytime.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_PATH = BASE_DIR / "Data" / "processed" / "cleaned_court_texts.json"
LABELED_PATH = BASE_DIR / "Data" / "processed" / "labeled_dataset.json"
OUTPUT_PATH = BASE_DIR / "Data" / "processed" / "decision_features_sundin2026.json"

# Sundin-inspired feature categories (what to look for)
SUNDIN_FEATURE_CATEGORIES = {
    "downstream_passage": [
        "downstream_has_screen",
        "downstream_gap_mm",
        "downstream_angle_degrees",
        "downstream_bypass_ls",
    ],
    "upstream_passage": [
        "upstream_has_fishway",
        "upstream_type",
        "upstream_slope_pct",
        "upstream_discharge_ls",
        "upstream_has_eel_ramp",
    ],
    "flow_requirements": [
        "flow_min_ls",
        "flow_hydropeaking_banned",
        "flow_percent_mq",
    ],
    "monitoring": [
        "monitoring_required",
        "monitoring_functional",
    ],
    "burden": [
        "cost_msek",
        "timeline_years",
    ],
}


def _first_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    try:
        s = m.group(1).replace(",", ".").strip()
        return float(s)
    except (ValueError, IndexError):
        return None


def _first_int(pattern: str, text: str) -> int | None:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


def extract_sundin_features(decision: dict) -> dict:
    """
    Extract Sundin-inspired features from one decision.
    Returns a flat dict of feature name -> value (float, int, bool, str, or None).
    """
    text = decision.get("text_full") or decision.get("key_text") or ""
    sections = decision.get("sections") or {}
    domslut = sections.get("domslut", "").lower()
    text_lower = text.lower()

    features = {}

    # ----- Downstream passage -----
    features["downstream_has_screen"] = any(
        kw in text_lower
        for kw in ["ledskena", "guidance", "skyddsgaller", "avledningsgaller", "rack"]
    )
    features["downstream_gap_mm"] = _first_float(
        r"spalt(?:bredd|vidd|vidd).*?(\d+(?:[.,]\d+)?)\s*mm", text_lower
    ) or _first_float(r"(\d+(?:[.,]\d+)?)\s*mm.*?spalt", text_lower)
    features["downstream_angle_degrees"] = _first_float(
        r"vinkel.*?(\d+(?:[.,]\d+)?)\s*(?:grader|°)", text_lower
    ) or _first_float(r"lutning.*?(\d+(?:[.,]\d+)?)\s*(?:grader|°)", text_lower)
    features["downstream_bypass_ls"] = _first_int(
        r"omlöp.*?(\d+)\s*(?:l/s|liter/s)", text_lower
    ) or _first_int(r"(\d+)\s*l/s.*?omlöp", text_lower)

    # ----- Upstream passage -----
    features["upstream_has_fishway"] = any(
        kw in text_lower for kw in ["fiskväg", "fishway", "fisktrappa", "fiskpassage"]
    )
    if "naturlik" in text_lower or "naturliknande" in text_lower:
        features["upstream_type"] = "nature-like"
    elif "vertikal" in text_lower or "vertical" in text_lower:
        features["upstream_type"] = "vertical-slot"
    elif "ål" in text_lower and ("trappa" in text_lower or "ramp" in text_lower):
        features["upstream_type"] = "eel-ramp"
    elif features.get("upstream_has_fishway"):
        features["upstream_type"] = "undefined"
    else:
        features["upstream_type"] = None
    features["upstream_slope_pct"] = _first_float(
        r"lutning.*?(\d+(?:[.,]\d+)?)\s*%", text_lower
    )
    features["upstream_discharge_ls"] = _first_int(
        r"fiskväg.*?(\d+)\s*(?:l/s|liter)", text_lower
    ) or _first_int(r"(\d+)\s*l/s.*?fisk", text_lower)
    features["upstream_has_eel_ramp"] = "ålyngeltrappa" in text_lower or (
        "ål" in text_lower and "trappa" in text_lower
    )

    # ----- Flow -----
    features["flow_min_ls"] = _first_int(
        r"minimitappning.*?(\d+)\s*(?:l/s|liter)", text_lower
    ) or _first_int(r"(\d+)\s*l/s.*?minimi", text_lower)
    features["flow_hydropeaking_banned"] = (
        "korttidsreglering" in text_lower and "förbjud" in text_lower
    )
    features["flow_percent_mq"] = _first_float(r"(\d+(?:[.,]\d+)?)\s*%.*?mq", text_lower)

    # ----- Monitoring -----
    features["monitoring_required"] = (
        "övervakning" in text_lower or "uppföljning" in text_lower
    )
    features["monitoring_functional"] = (
        "funktionalitet" in text_lower and features["monitoring_required"]
    )

    # ----- Burden (cost & timeline) -----
    costs = decision.get("extracted_costs") or []
    cost_msek = None
    if costs:
        total_sek = sum(c.get("amount_sek", 0) for c in costs)
        if total_sek >= 1_000_000:
            cost_msek = round(total_sek / 1_000_000, 2)
    if cost_msek is None:
        for m in re.finditer(
            r"([\d\s,.]+)\s*(?:msek|miljoner?\s*kronor|milj\.?\s*kr)", text_lower
        ):
            try:
                val = float(m.group(1).replace(",", ".").replace(" ", ""))
                if 0.1 <= val <= 500:
                    cost_msek = round(val, 2)
                    break
            except ValueError:
                continue
    features["cost_msek"] = cost_msek

    timeline = _first_int(r"inom\s*(\d+)\s*år", text_lower)
    features["timeline_years"] = timeline

    return features


def build_id_to_label(labeled_data: dict) -> dict[str, str]:
    """Build mapping decision id -> label from splits."""
    out = {}
    for split_name in ("train", "val", "test"):
        for item in labeled_data.get("splits", {}).get(split_name, []):
            out[item["id"]] = item["label"]
    return out


def main():
    if not CLEANED_PATH.exists():
        print(f"Missing {CLEANED_PATH}. Run 02_clean_court_texts.py first.")
        return
    if not LABELED_PATH.exists():
        print(f"Missing {LABELED_PATH}. Run 03_create_labeled_dataset.py first.")
        return

    with open(CLEANED_PATH, "r", encoding="utf-8") as f:
        cleaned = json.load(f)
    with open(LABELED_PATH, "r", encoding="utf-8") as f:
        labeled = json.load(f)

    decisions = cleaned.get("decisions", [])
    id_to_label = build_id_to_label(labeled)

    results = []
    for d in decisions:
        dec_id = d.get("id")
        label = id_to_label.get(dec_id)
        if label is None:
            continue
        features = extract_sundin_features(d)
        results.append({
            "id": dec_id,
            "filename": d.get("filename", ""),
            "label": label,
            "features": features,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "metadata": {
            "source_cleaned": str(CLEANED_PATH),
            "source_labeled": str(LABELED_PATH),
            "n_decisions": len(results),
            "sundin_categories": SUNDIN_FEATURE_CATEGORIES,
        },
        "decisions": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(results)} decision feature rows to {OUTPUT_PATH}")
    print("Run: python scripts/sundin_validation.py  for clustering and RF importance.")


if __name__ == "__main__":
    main()
