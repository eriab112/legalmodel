#!/usr/bin/env python3
"""
Apply validated corrections from Gemini re-extraction and manual review
to produce clean, corrected court decision data files.

Usage:
    cd nap-legal-ai-advisor
    python scripts/apply_corrections.py --data-root ../Data
"""

import argparse
import io
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows to handle Swedish characters
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Constants ──────────────────────────────────────────────────────────────────

NON_NAP_CASES = {
    "M 8024-05",
    "M 503 -22",
    "M 7708 -22",
    "M 899-23",
    "M 3273-22",
    "M 2479-22",
    "M 4965-23",
    "M 4021-22",
    "M 6587-23",
    "M 2024-01",
}

NON_NAP_DESCRIPTIONS = {
    "M 8024-05": "Fastighetsreglering/strandskydd",
    "M 503 -22": "Marina/muddring",
    "M 7708 -22": "Utsläppsrätter",
    "M 899-23": "Industriellt miljötillstånd (NCC)",
    "M 3273-22": "Bergtäkt",
    "M 2479-22": "Solcellsanläggning",
    "M 4965-23": "Badbrygga",
    "M 4021-22": "Avloppsreningsverk",
    "M 6587-23": "Termisk förgasning/metallpulver",
    "M 2024-01": "Detaljplan/vägbyggnad",
}

MANUAL_OUTCOME_OVERRIDES = {
    "M 2693-22": ("dismissed", "Avvisad"),
    "M 391-99": ("dismissed", "Avvisad"),
}

OUTCOME_SV = {
    "granted": "Tillstånd beviljat",
    "granted_modified": "Tillstånd med ändringar",
    "conditions_changed": "Villkor ändrade",
    "denied": "Ansökan avslagen",
    "appeal_denied": "Överklagande avslaget",
    "remanded": "Återförvisat",
    "overturned": "Upphävt",
    "dismissed": "Avvisad",
    "withdrawn": "Återkallat",
}

GENERIC_PLANT_NAMES = {
    "vattenkraftsanläggning",
    "vattenkraftsanläggningen",
    "vattenkraftverket",
    "kraftverket",
}


def normalize_case_number(cn: str) -> str:
    """Normalize a case number by collapsing whitespace for comparison."""
    return re.sub(r"\s+", " ", cn.strip())


def is_non_nap(case_number: str) -> bool:
    """Check if a case number matches any of the non-NAP cases."""
    norm = normalize_case_number(case_number)
    for non_nap in NON_NAP_CASES:
        if normalize_case_number(non_nap) == norm:
            return True
    return False


def find_non_nap_original(case_number: str) -> str | None:
    """Return the original NON_NAP_CASES string matching this case_number."""
    norm = normalize_case_number(case_number)
    for non_nap in NON_NAP_CASES:
        if normalize_case_number(non_nap) == norm:
            return non_nap
    return None


def is_empty_or_none(val) -> bool:
    """Check if a value is None, empty string, or the string 'None'."""
    if val is None:
        return True
    if isinstance(val, str) and val.strip() in ("", "None"):
        return True
    return False


def is_generic_plant_name(name: str | None) -> bool:
    """Check if a plant name is too generic to be useful."""
    if name is None:
        return False
    lower = name.strip().lower()
    if lower in GENERIC_PLANT_NAMES:
        return True
    # Check for "X och Y" pattern
    if re.match(r"^[A-Z]\s+och\s+[A-Z]$", name.strip()):
        return True
    return False


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def build_correction_map(corrections: list) -> dict:
    """Build a map from normalized case_number to correction entry."""
    cmap = {}
    for c in corrections:
        norm = normalize_case_number(c["case_number"])
        cmap[norm] = c
    return cmap


# ── Step 0: Backups ───────────────────────────────────────────────────────────

def create_backups(data_root: Path) -> None:
    today = datetime.now().strftime("%Y%m%d")
    processed = data_root / "processed"

    for filename in ["cleaned_court_texts.json", "labeled_dataset_binary.json"]:
        src = processed / filename
        stem = filename.replace(".json", "")
        dst = processed / f"{stem}_backup_{today}.json"
        if dst.exists():
            print(f"  Backup already exists: {dst.name}")
        else:
            shutil.copy2(src, dst)
            print(f"  Created backup: {dst.name}")


# ── Step 1: Remove non-NAP decisions ─────────────────────────────────────────

def remove_non_nap(court_data: dict, labeled_data: dict) -> tuple[list[dict], int, int]:
    """Remove non-NAP decisions. Returns (removed_decisions, court_removed_count, labeled_removed_count)."""
    removed = []
    removed_ids = set()

    # Remove from court texts
    kept_decisions = []
    for d in court_data["decisions"]:
        cn = d["metadata"]["case_number"]
        if is_non_nap(cn):
            removed.append(d)
            removed_ids.add(d["id"])
        else:
            kept_decisions.append(d)

    court_removed = len(court_data["decisions"]) - len(kept_decisions)
    court_data["decisions"] = kept_decisions
    court_data["total_decisions"] = len(kept_decisions)

    # Remove from labeled dataset
    labeled_removed = 0
    for split_name in list(labeled_data["splits"].keys()):
        original = labeled_data["splits"][split_name]
        filtered = [item for item in original if item["id"] not in removed_ids]
        labeled_removed += len(original) - len(filtered)
        labeled_data["splits"][split_name] = filtered

    # Update labeled metadata
    total = sum(len(v) for v in labeled_data["splits"].values())
    labeled_data["total_decisions"] = total
    high = sum(1 for s in labeled_data["splits"].values() for item in s if item["label"] == "HIGH_RISK")
    low = sum(1 for s in labeled_data["splits"].values() for item in s if item["label"] == "LOW_RISK")
    labeled_data["label_distribution"] = {"HIGH_RISK": high, "LOW_RISK": low}
    labeled_data["split_sizes"] = {k: len(v) for k, v in labeled_data["splits"].items()}
    if total > 0:
        labeled_data["split_ratios"] = {k: len(v) / total for k, v in labeled_data["splits"].items()}

    return removed, court_removed, labeled_removed


# ── Step 2: Apply corrections ─────────────────────────────────────────────────

def apply_corrections(court_data: dict, labeled_data: dict, corrections: list) -> dict:
    """Apply Gemini corrections and manual overrides. Returns change log."""
    correction_map = build_correction_map(corrections)

    # Build labeled lookup: id -> (split_name, index)
    labeled_lookup = {}
    for split_name, items in labeled_data["splits"].items():
        for idx, item in enumerate(items):
            labeled_lookup[item["id"]] = (split_name, idx)

    changes = {}  # case_number -> list of (field, old, new, source)
    counters = {
        "outcomes": 0,
        "costs": 0,
        "plant_names": 0,
        "operators": 0,
        "watercourses": 0,
        "measures": 0,
    }

    for decision in court_data["decisions"]:
        cn = decision["metadata"]["case_number"]
        norm_cn = normalize_case_number(cn)
        corr = correction_map.get(norm_cn)
        if corr is None:
            continue

        decision_changes = []
        meta = decision["metadata"]
        gemini = corr.get("gemini_full_response", {})

        # 2a. Plant names
        if "power_plant_name" in corr["fields_changed"]:
            new_name = corr["fields_changed"]["power_plant_name"]["new"]
            old_name = meta.get("power_plant_name")
            if not is_empty_or_none(new_name) and not is_generic_plant_name(new_name):
                if old_name != new_name:
                    decision_changes.append(("plant_name", old_name, new_name, "gemini"))
                    meta["power_plant_name"] = new_name
                    counters["plant_names"] += 1

        # 2b. Outcomes
        original_cn = cn
        old_outcome = meta.get("application_outcome")
        old_outcome_sv = meta.get("application_outcome_sv")

        if original_cn in MANUAL_OUTCOME_OVERRIDES:
            new_outcome, new_outcome_sv = MANUAL_OUTCOME_OVERRIDES[original_cn]
            if old_outcome != new_outcome:
                decision_changes.append(("outcome", old_outcome, new_outcome, "manual"))
                meta["application_outcome"] = new_outcome
                meta["application_outcome_sv"] = new_outcome_sv
                counters["outcomes"] += 1
        else:
            gemini_outcome = gemini.get("application_outcome")
            if gemini_outcome is not None:
                gemini_outcome_sv = gemini.get("application_outcome_sv")
                if gemini_outcome_sv is None:
                    gemini_outcome_sv = OUTCOME_SV.get(gemini_outcome, gemini_outcome)
                if old_outcome != gemini_outcome:
                    decision_changes.append(("outcome", old_outcome, gemini_outcome, "gemini"))
                    meta["application_outcome"] = gemini_outcome
                    meta["application_outcome_sv"] = gemini_outcome_sv
                    counters["outcomes"] += 1

        # 2c. Costs
        if "total_cost_sek" in corr["fields_changed"]:
            gemini_cost = gemini.get("operator_cost_sek")
            gemini_cost_desc = gemini.get("cost_description")
            old_cost = meta.get("total_cost_sek")

            if gemini_cost is not None:
                if old_cost != gemini_cost:
                    decision_changes.append(("cost", old_cost, gemini_cost, "gemini"))
                    meta["total_cost_sek"] = gemini_cost
                    decision["extracted_costs"] = [
                        {
                            "amount_sek": gemini_cost,
                            "original": gemini_cost_desc,
                            "context": "Gemini re-extraction",
                        }
                    ]
                    counters["costs"] += 1
            else:
                if old_cost is not None:
                    decision_changes.append(("cost", old_cost, None, "gemini"))
                    meta["total_cost_sek"] = None
                    decision["extracted_costs"] = []
                    counters["costs"] += 1

        # 2d. Operator names
        gemini_operator = gemini.get("operator_name")
        current_operator = meta.get("operator_name")
        if not is_empty_or_none(gemini_operator) and is_empty_or_none(current_operator):
            decision_changes.append(("operator", current_operator, gemini_operator, "gemini"))
            meta["operator_name"] = gemini_operator
            counters["operators"] += 1

        # 2e. Watercourse
        gemini_watercourse = gemini.get("watercourse")
        current_watercourse = meta.get("watercourse")
        if not is_empty_or_none(gemini_watercourse) and is_empty_or_none(current_watercourse):
            decision_changes.append(("watercourse", current_watercourse, gemini_watercourse, "gemini"))
            meta["watercourse"] = gemini_watercourse
            counters["watercourses"] += 1

        # 2f. Measures
        gemini_measures = gemini.get("environmental_measures", [])
        if gemini_measures:
            current_measures = decision.get("extracted_measures", [])
            merged = sorted(set(current_measures) | set(gemini_measures))
            if merged != sorted(current_measures):
                added = sorted(set(gemini_measures) - set(current_measures))
                if added:
                    decision_changes.append(("measures", f"{len(current_measures)} items", f"{len(merged)} items (+{', '.join(added)})", "gemini"))
                    decision["extracted_measures"] = merged
                    counters["measures"] += 1

        # 2g. Update labeled_dataset_binary
        if decision_changes and decision["id"] in labeled_lookup:
            split_name, idx = labeled_lookup[decision["id"]]
            labeled_item = labeled_data["splits"][split_name][idx]

            # Sync metadata fields that exist in labeled
            labeled_meta = labeled_item.get("metadata", {})
            if "application_outcome" in labeled_meta or "application_outcome" in meta:
                labeled_meta["application_outcome"] = meta.get("application_outcome")
            if "application_outcome_sv" in labeled_meta or "application_outcome_sv" in meta:
                labeled_meta["application_outcome_sv"] = meta.get("application_outcome_sv")

            # Update scoring_details.outcome_type if it exists
            scoring = labeled_item.get("scoring_details", {})
            new_outcome = meta.get("application_outcome")
            if new_outcome and "outcome_type" in scoring:
                outcome_type_map = {
                    "granted": "GRANTED",
                    "granted_modified": "GRANTED_MODIFIED",
                    "conditions_changed": "CONDITIONS_CHANGED",
                    "denied": "DENIED",
                    "appeal_denied": "APPEAL_DENIED",
                    "remanded": "REMANDED",
                    "overturned": "OVERTURNED",
                    "dismissed": "DISMISSED",
                    "withdrawn": "WITHDRAWN",
                }
                mapped = outcome_type_map.get(new_outcome)
                if mapped:
                    scoring["outcome_type"] = mapped

        if decision_changes:
            changes[cn] = decision_changes

    return changes, counters


# ── Step 3: Generate diff report ──────────────────────────────────────────────

def generate_diff_report(
    removed_decisions: list[dict],
    changes: dict,
    counters: dict,
    decisions_before: int,
    decisions_after: int,
    output_path: Path,
) -> None:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("NAP Legal AI Advisor — Corrections Applied")
    lines.append(f"Generated: {now}")
    lines.append("=" * 60)
    lines.append("")

    # Removed decisions
    lines.append(f"DECISIONS REMOVED (non-NAP): {len(removed_decisions)}")
    for d in removed_decisions:
        cn = d["metadata"]["case_number"]
        orig_cn = find_non_nap_original(cn)
        desc = NON_NAP_DESCRIPTIONS.get(orig_cn or cn, "Unknown")
        lines.append(f"  {cn} — {desc}")
    lines.append("")
    lines.append(f"DECISIONS REMAINING: {decisions_after}")
    lines.append("")
    lines.append("=" * 60)
    lines.append("FIELD CORRECTIONS APPLIED:")
    lines.append("=" * 60)
    lines.append("")

    if not changes:
        lines.append("  (no field corrections applied)")
    else:
        for cn, field_changes in sorted(changes.items()):
            lines.append(f"{cn}:")
            for field, old_val, new_val, source in field_changes:
                lines.append(f"  {field}: {old_val} → {new_val} (source: {source})")
            lines.append("")

    lines.append("=" * 60)
    lines.append("SUMMARY:")
    lines.append(f"  Decisions before: {decisions_before}")
    lines.append(f"  Decisions after: {decisions_after}")
    lines.append(f"  Outcomes corrected: {counters['outcomes']}")
    lines.append(f"  Costs corrected: {counters['costs']}")
    lines.append(f"  Plant names corrected: {counters['plant_names']}")
    lines.append(f"  Operators added: {counters['operators']}")
    lines.append(f"  Watercourses added: {counters['watercourses']}")
    lines.append(f"  Measures merged: {counters['measures']}")
    lines.append("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved diff report: {output_path}")


# ── Step 4: Validate ──────────────────────────────────────────────────────────

def validate(court_data: dict, labeled_data: dict) -> bool:
    errors = []

    # All decisions have non-empty text_full
    for d in court_data["decisions"]:
        if not d.get("text_full"):
            errors.append(f"Decision {d['id']} has empty text_full")

    # All decisions have case_number
    for d in court_data["decisions"]:
        if not d.get("metadata", {}).get("case_number"):
            errors.append(f"Decision {d['id']} missing case_number")

    # No duplicate IDs
    ids = [d["id"] for d in court_data["decisions"]]
    if len(ids) != len(set(ids)):
        dupes = [x for x in ids if ids.count(x) > 1]
        errors.append(f"Duplicate decision IDs: {set(dupes)}")

    # Labeled splits don't reference removed decisions
    court_ids = set(ids)
    for split_name, items in labeled_data["splits"].items():
        for item in items:
            if item["id"] not in court_ids:
                errors.append(f"Labeled {split_name} references removed ID: {item['id']}")

    if errors:
        print("\n  VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("  Validation passed: all checks OK")
        return True


# ── Main ──────────────────────────────────────────────────────────────────────

def detect_data_root() -> Path:
    """Try to auto-detect data root relative to script or cwd."""
    candidates = [
        Path.cwd() / "Data",
        Path.cwd().parent / "Data",
        Path(__file__).resolve().parent.parent.parent / "Data",
        Path(__file__).resolve().parent.parent / "Data",
    ]
    for c in candidates:
        if (c / "processed" / "cleaned_court_texts.json").exists():
            return c
    return Path.cwd() / "Data"


def main():
    parser = argparse.ArgumentParser(description="Apply validated corrections to court decision dataset")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to Data/ directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing files")
    args = parser.parse_args()

    data_root = args.data_root or detect_data_root()
    data_root = data_root.resolve()

    court_path = data_root / "processed" / "cleaned_court_texts.json"
    labeled_path = data_root / "processed" / "labeled_dataset_binary.json"
    corrections_path = data_root / "validation" / "gemini_corrections.json"
    diff_report_path = data_root / "validation" / "corrections_diff_report.txt"

    print(f"Data root: {data_root}")
    for p in [court_path, labeled_path, corrections_path]:
        if not p.exists():
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)
    print()

    # Load data
    print("Loading data files...")
    court_data = load_json(court_path)
    labeled_data = load_json(labeled_path)
    corrections_data = load_json(corrections_path)
    decisions_before = len(court_data["decisions"])
    print(f"  Court texts: {decisions_before} decisions")
    print(f"  Labeled dataset: {labeled_data['total_decisions']} items across {len(labeled_data['splits'])} splits")
    print(f"  Gemini corrections: {len(corrections_data['corrections'])} entries")
    print()

    # Step 0: Create backups
    print("Step 0: Creating backups...")
    if not args.dry_run:
        create_backups(data_root)
    else:
        print("  (dry run — skipping backups)")
    print()

    # Step 1: Remove non-NAP decisions
    print("Step 1: Removing non-NAP decisions...")
    removed_decisions, court_removed, labeled_removed = remove_non_nap(court_data, labeled_data)
    decisions_after = len(court_data["decisions"])
    print(f"  Removed {court_removed} decisions from court texts ({decisions_before} → {decisions_after})")
    print(f"  Removed {labeled_removed} items from labeled dataset")
    for d in removed_decisions:
        cn = d["metadata"]["case_number"]
        orig = find_non_nap_original(cn)
        desc = NON_NAP_DESCRIPTIONS.get(orig or cn, "")
        print(f"    - {cn}: {desc}")
    print()

    # Step 2: Apply corrections
    print("Step 2: Applying metadata corrections...")
    changes, counters = apply_corrections(court_data, labeled_data, corrections_data["corrections"])
    print(f"  Outcomes corrected: {counters['outcomes']}")
    print(f"  Costs corrected: {counters['costs']}")
    print(f"  Plant names corrected: {counters['plant_names']}")
    print(f"  Operators added: {counters['operators']}")
    print(f"  Watercourses added: {counters['watercourses']}")
    print(f"  Measures merged: {counters['measures']}")
    print()

    # Step 3: Generate diff report
    print("Step 3: Generating diff report...")
    generate_diff_report(removed_decisions, changes, counters, decisions_before, decisions_after, diff_report_path)
    print()

    # Step 4: Validate and save
    print("Step 4: Validating...")
    valid = validate(court_data, labeled_data)
    print()

    if not valid:
        print("VALIDATION FAILED — not saving files.")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — no files written.")
        sys.exit(0)

    print(f"About to overwrite cleaned_court_texts.json ({decisions_before} → {decisions_after} decisions). Backups created. Proceeding...")
    save_json(court_data, court_path)

    labeled_total = sum(len(v) for v in labeled_data["splits"].values())
    print(f"About to overwrite labeled_dataset_binary.json ({labeled_total} items). Backups created. Proceeding...")
    save_json(labeled_data, labeled_path)
    print()

    # Final stats
    print("=" * 60)
    print("FINAL STATS:")
    print(f"  Decisions: {len(court_data['decisions'])}")
    for split_name, items in labeled_data["splits"].items():
        print(f"  Labeled {split_name}: {len(items)} items")

    # Field coverage
    total = len(court_data["decisions"])
    has_outcome = sum(1 for d in court_data["decisions"] if not is_empty_or_none(d["metadata"].get("application_outcome")))
    has_plant = sum(1 for d in court_data["decisions"] if not is_empty_or_none(d["metadata"].get("power_plant_name")))
    has_operator = sum(1 for d in court_data["decisions"] if not is_empty_or_none(d["metadata"].get("operator_name")))
    has_watercourse = sum(1 for d in court_data["decisions"] if not is_empty_or_none(d["metadata"].get("watercourse")))
    has_cost = sum(1 for d in court_data["decisions"] if d["metadata"].get("total_cost_sek") is not None)

    print(f"  Outcome coverage: {has_outcome}/{total} ({100*has_outcome/total:.0f}%)")
    print(f"  Plant name coverage: {has_plant}/{total} ({100*has_plant/total:.0f}%)")
    print(f"  Operator coverage: {has_operator}/{total} ({100*has_operator/total:.0f}%)")
    print(f"  Watercourse coverage: {has_watercourse}/{total} ({100*has_watercourse/total:.0f}%)")
    print(f"  Cost coverage: {has_cost}/{total} ({100*has_cost/total:.0f}%)")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
