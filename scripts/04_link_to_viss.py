"""
04_link_to_viss.py - Water Body Linkage for NAP LegalBERT

Links water bodies from court decisions to VISS IDs in the MCDA framework:
1. Tier 1: Direct VISS IDs from rich_court_database.json (5 known)
2. Tier 2: Fuzzy match via water body name similarity
3. Tier 3: MCDA database lookup via ecological_status entries
4. Output: linkage_table.json with confidence scores

Target: Link at least 15 of 18 water bodies from court decisions.

Usage:
    python scripts/04_link_to_viss.py
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

RICH_COURT_DB = Path("nap_model-main/data/rich_court_database.json")
MCDA_RANKINGS = Path("nap_model-main/data/mcda_rankings_full.json")
MASTER_DB = Path("nap_model-main/agent/nap_quantitative_data_v2.17.json")
CLEANED_TEXTS = Path("Data/processed/cleaned_court_texts.json")
OUTPUT_FILE = Path("Data/processed/linkage_table.json")


# ============================================================
# Data Loading
# ============================================================

def load_court_water_bodies(rich_db_path: Path) -> dict:
    """Load water body -> court decision mappings from rich_court_database.json."""
    with open(rich_db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    water_bodies = {}
    for wb_name, entries in db.get("water_bodies_index", {}).items():
        for entry in entries:
            if wb_name not in water_bodies:
                water_bodies[wb_name] = {
                    "name": wb_name,
                    "files": [],
                    "known_viss_ids": [],
                    "kommuner": [],
                    "kraftverk": [],
                }
            water_bodies[wb_name]["files"].append(entry.get("file"))
            water_bodies[wb_name]["known_viss_ids"].extend(entry.get("viss_ids", []))
            if entry.get("kommun"):
                water_bodies[wb_name]["kommuner"].append(entry["kommun"])
            water_bodies[wb_name]["kraftverk"].extend(entry.get("kraftverk", []))

    # Deduplicate
    for wb in water_bodies.values():
        wb["known_viss_ids"] = list(set(wb["known_viss_ids"]))
        wb["kommuner"] = list(set(wb["kommuner"]))
        wb["kraftverk"] = list(set(wb["kraftverk"]))
        wb["files"] = list(set(wb["files"]))

    return water_bodies


def extract_primary_name(compound_name: str) -> str:
    """Extract primary water body name from a compound VISS entry.

    Examples:
        'Rönne å: Bäljane å nr 1-Klövabäcken' → 'Rönne å'
        'Örekilsälven - Hajumsälvens inflöde...' → 'Örekilsälven'
        'Vessingeån (Mynningen-Prästabäcken)' → 'Vessingeån'
        'Vessingeån/Bölarpsån (X-Y)' → 'Vessingeån'
        'VEGE Å: Hasslarpsån-Humlebäcken' → 'VEGE Å'
    """
    name = compound_name.strip()
    # Strip parenthetical
    name = re.sub(r'\s*\(.*?\)\s*', '', name).strip()
    # Split on ":"
    if ':' in name:
        name = name.split(':')[0].strip()
    # Split on " - " (space-dash-space = segment separator)
    if ' - ' in name:
        name = name.split(' - ')[0].strip()
    # Split on "/" (alternative names)
    if '/' in name:
        name = name.split('/')[0].strip()
    return name


def load_viss_water_bodies(master_db_path: Path) -> dict:
    """Load VISS water body name -> ID mapping from master database."""
    with open(master_db_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    viss_map = {}  # name_lower -> {viss_id, name, ...}

    # Source 1: ecological_status water_bodies (~2045 entries, has municipality info)
    es_wbs = data.get("ecological_status", {}).get("water_bodies", [])
    for wb in es_wbs:
        viss_id = wb.get("viss_id", "")
        name = wb.get("name", "")
        if viss_id and name:
            viss_map[name.lower()] = {
                "viss_id": viss_id,
                "name": name,
                "basin": wb.get("basin", ""),
                "county": wb.get("county", ""),
                "municipalities": wb.get("municipalities", ""),
            }

    # Source 2: water_bodies aliases
    aliases = data.get("water_bodies", {}).get("aliases", {})
    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            if alias.lower() not in viss_map:
                for entry in es_wbs:
                    if canonical.lower() in entry.get("name", "").lower():
                        viss_map[alias.lower()] = {
                            "viss_id": entry["viss_id"],
                            "name": entry["name"],
                            "alias_of": canonical,
                        }
                        break

    # Source 3: py04_paverkan water_bodies (15,688 entries - full VISS coverage)
    # Index by BOTH the full compound name AND the extracted primary name
    py04_wbs = data.get("py04_paverkan", {}).get("water_bodies", [])
    py04_added = 0
    for wb in py04_wbs:
        wa_id = wb.get("wa_id", "")
        name = wb.get("name", "")
        if not wa_id or not name or name[0].isdigit():
            continue  # Skip coordinate-style names

        lan = wb.get("lan", "")

        # Add full compound name if not already present
        if name.lower() not in viss_map:
            viss_map[name.lower()] = {
                "viss_id": wa_id,
                "name": name,
                "county": lan,
                "municipalities": "",
                "source": "py04_paverkan",
            }
            py04_added += 1

        # Also index by primary name (before :, -, /, parens)
        primary = extract_primary_name(name)
        if primary and primary.lower() != name.lower() and primary.lower() not in viss_map:
            viss_map[primary.lower()] = {
                "viss_id": wa_id,
                "name": primary,
                "full_name": name,
                "county": lan,
                "municipalities": "",
                "source": "py04_paverkan_primary",
            }
            py04_added += 1

    print(f"(+{py04_added} from py04_paverkan) ", end="")

    return viss_map


def load_mcda_viss_ids(mcda_path: Path) -> set:
    """Load all VISS IDs that have MCDA rankings."""
    with open(mcda_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {r["water_body_id"] for r in data.get("rankings", [])}


# ============================================================
# Non-water-body blocklist (false NER extractions)
# ============================================================

FALSE_WATER_BODIES = {
    "utifrån",  # Swedish word meaning "based on", not a water body
}


# ============================================================
# Fuzzy Matching
# ============================================================

def normalize_name(name: str) -> str:
    """Normalize water body name for matching.

    Instead of stripping suffixes (which caused empty-string matches),
    normalizes variant spellings to a canonical form:
      "Rönne å" → "rönneå", "Rönneån" → "rönneå"
      "Vindelälven" → "vindelälv"
      "Söderköpingsån (X - Y)" → "söderköpingså"
    """
    name = name.lower().strip()
    name = re.sub(r'\s+', ' ', name)

    # Remove parenthetical qualifiers: "Söderköpingsån (X - Y)" → "Söderköpingsån"
    name = re.sub(r'\s*\(.*?\)\s*', '', name).strip()
    # Remove colon qualifiers: "HELGE Å: Lillån - X" → "helge å"
    name = re.sub(r'\s*:.*$', '', name).strip()
    # Remove " - " segment descriptions: "Örekilsälven - X till Y" → "örekilsälven"
    name = re.sub(r'\s+-\s+.*$', '', name).strip()
    # Note: plain "-" without spaces (e.g. "Muskån-Lillån") kept as-is

    # Merge separated suffixes: "Rönne å" → "rönneå", "Helge å" → "helgeå"
    name = re.sub(r'\s+(å|ån|älv|älven|sjö|sjön|bäck|bäcken)$', r'\1', name)

    # Normalize definite → indefinite form: ån→å, älven→älv, sjön→sjö, bäcken→bäck
    name = re.sub(r'ån$', 'å', name)
    name = re.sub(r'älven$', 'älv', name)
    name = re.sub(r'sjön$', 'sjö', name)
    name = re.sub(r'bäcken$', 'bäck', name)

    return name.strip()


def fuzzy_match_score(name1: str, name2: str) -> float:
    """Compute fuzzy match score between two water body names."""
    # Direct case-insensitive match
    if name1.lower().strip() == name2.lower().strip():
        return 1.0

    # Normalized match
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    # Guard: skip if either normalized to empty or very short (< 3 chars)
    if len(n1) < 3 or len(n2) < 3:
        return 0.0

    if n1 == n2:
        return 0.95

    # Containment: only if contained string is >= 4 chars
    # Score scales with length ratio to penalize very unequal matches
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if len(shorter) >= 4 and shorter in longer:
        ratio = len(shorter) / len(longer)
        return 0.75 + 0.10 * ratio  # Range: 0.75-0.85

    # Sequence matching
    ratio = SequenceMatcher(None, n1, n2).ratio()
    return ratio


def municipality_overlap(court_kommuner: list, viss_municipalities: str) -> bool:
    """Check if any court kommun matches VISS municipalities."""
    if not court_kommuner or not viss_municipalities:
        return False
    viss_lower = viss_municipalities.lower()
    for kommun in court_kommuner:
        k = kommun.lower().replace(" kommun", "").strip()
        if k and len(k) >= 3 and k in viss_lower:
            return True
    return False


def find_best_viss_match(court_wb_name: str, viss_map: dict,
                         court_kommuner: list = None,
                         threshold: float = 0.7) -> list:
    """Find best VISS matches for a court water body name.

    Uses municipality overlap to boost scores for common names
    (e.g. "Lillån" exists in many places).
    """
    matches = []

    for viss_name, viss_info in viss_map.items():
        score = fuzzy_match_score(court_wb_name, viss_info.get("name", viss_name))

        if score >= threshold:
            # Boost by 0.05 if municipality matches (helps disambiguate common names)
            kommun_match = False
            if court_kommuner and score < 1.0:
                viss_munis = viss_info.get("municipalities", "")
                if municipality_overlap(court_kommuner, viss_munis):
                    score = min(score + 0.05, 0.99)
                    kommun_match = True

            matches.append({
                "viss_id": viss_info["viss_id"],
                "viss_name": viss_info.get("name", viss_name),
                "match_score": round(score, 3),
                "match_type": "exact" if score >= 0.95 else "fuzzy",
                "kommun_match": kommun_match,
            })

    # Sort by score descending, then by name length ascending (prefer shorter/simpler names)
    matches.sort(key=lambda x: (-x["match_score"], len(x["viss_name"])))
    return matches[:5]  # Top 5


# ============================================================
# Main Linkage Pipeline
# ============================================================

def main():
    print("=" * 60)
    print("NAP LegalBERT - Water Body VISS Linkage")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    court_wbs = load_court_water_bodies(RICH_COURT_DB)
    print(f"  Court water bodies: {len(court_wbs)}")

    print("  Loading VISS database (large file)...", end=" ", flush=True)
    viss_map = load_viss_water_bodies(MASTER_DB)
    print(f"{len(viss_map)} entries")

    mcda_ids = load_mcda_viss_ids(MCDA_RANKINGS)
    print(f"  MCDA water bodies: {len(mcda_ids)}")

    # Process each court water body
    print(f"\nLinking {len(court_wbs)} water bodies...")

    linkages = []
    tier1_count = 0
    tier2_count = 0
    tier3_count = 0
    no_match = 0

    for wb_name, wb_info in sorted(court_wbs.items()):
        linkage = {
            "court_water_body": wb_name,
            "files": wb_info["files"],
            "kommuner": wb_info["kommuner"],
            "kraftverk": [k for k in wb_info["kraftverk"] if len(k) < 50],  # Filter noise
            "linked_viss_ids": [],
            "in_mcda": False,
            "match_tier": None,
            "confidence": 0.0,
            "needs_review": False,
        }

        # Skip known false water body names
        if wb_name.lower() in FALSE_WATER_BODIES:
            linkage["match_tier"] = "excluded"
            linkage["needs_review"] = False
            linkage["note"] = "Not a water body (false NER extraction)"
            print(f"  [SKIP]  {wb_name:20s} -> Excluded (not a water body)")
            linkages.append(linkage)
            continue

        # Tier 1: Known VISS IDs from rich_court_database
        known_ids = wb_info["known_viss_ids"]
        if known_ids:
            tier1_count += 1
            linkage["linked_viss_ids"] = known_ids
            linkage["match_tier"] = "tier1_known"
            linkage["confidence"] = 0.99
            linkage["in_mcda"] = any(vid in mcda_ids for vid in known_ids)
            print(f"  [TIER1] {wb_name:20s} -> {', '.join(known_ids)} (known, MCDA={'Y' if linkage['in_mcda'] else 'N'})")

        else:
            # Tier 2: Fuzzy match against VISS database
            matches = find_best_viss_match(wb_name, viss_map,
                                           court_kommuner=wb_info["kommuner"],
                                           threshold=0.7)

            if matches:
                best = matches[0]
                if best["match_score"] >= 0.85:
                    tier2_count += 1
                    linkage["linked_viss_ids"] = [best["viss_id"]]
                    linkage["match_tier"] = "tier2_fuzzy_high"
                    linkage["confidence"] = round(best["match_score"], 2)
                    linkage["in_mcda"] = best["viss_id"] in mcda_ids
                    linkage["fuzzy_matches"] = matches
                    print(f"  [TIER2] {wb_name:20s} -> {best['viss_id']} ({best['viss_name']}, score={best['match_score']:.2f}, MCDA={'Y' if linkage['in_mcda'] else 'N'})")

                elif best["match_score"] >= 0.7:
                    tier3_count += 1
                    linkage["linked_viss_ids"] = [best["viss_id"]]
                    linkage["match_tier"] = "tier3_fuzzy_low"
                    linkage["confidence"] = round(best["match_score"], 2)
                    linkage["in_mcda"] = best["viss_id"] in mcda_ids
                    linkage["fuzzy_matches"] = matches
                    linkage["needs_review"] = True
                    print(f"  [TIER3] {wb_name:20s} -> {best['viss_id']} ({best['viss_name']}, score={best['match_score']:.2f}) NEEDS REVIEW")

                else:
                    no_match += 1
                    linkage["match_tier"] = "no_match"
                    linkage["confidence"] = 0.0
                    linkage["needs_review"] = True
                    linkage["fuzzy_matches"] = matches
                    print(f"  [NONE]  {wb_name:20s} -> No match above threshold (best: {best['viss_name']}={best['match_score']:.2f})")
            else:
                no_match += 1
                linkage["match_tier"] = "no_match"
                linkage["needs_review"] = True
                print(f"  [NONE]  {wb_name:20s} -> No matches found")

        linkages.append(linkage)

    # Summary
    excluded = sum(1 for l in linkages if l["match_tier"] == "excluded")
    relevant = len(court_wbs) - excluded
    total_linked = tier1_count + tier2_count + tier3_count
    in_mcda = sum(1 for l in linkages if l["in_mcda"])
    needs_review = sum(1 for l in linkages if l["needs_review"])

    print(f"\n{'=' * 60}")
    print(f"LINKAGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total water bodies:  {len(court_wbs)} ({excluded} excluded as non-WB)")
    print(f"  Relevant:            {relevant}")
    print(f"  Linked:              {total_linked}/{relevant} ({100*total_linked/max(relevant,1):.0f}%)")
    print(f"    Tier 1 (known):    {tier1_count}")
    print(f"    Tier 2 (high conf): {tier2_count}")
    print(f"    Tier 3 (low conf):  {tier3_count}")
    print(f"    No match:          {no_match}")
    print(f"  In MCDA framework:   {in_mcda}/{relevant}")
    print(f"  Needs review:        {needs_review}")

    # Save output
    output = {
        "version": "1.1",
        "created": "2026-02-06",
        "total_court_water_bodies": len(court_wbs),
        "excluded_non_wb": excluded,
        "relevant_water_bodies": relevant,
        "total_linked": total_linked,
        "linkage_tiers": {
            "tier1_known": tier1_count,
            "tier2_fuzzy_high": tier2_count,
            "tier3_fuzzy_low": tier3_count,
            "no_match": no_match,
        },
        "in_mcda_count": in_mcda,
        "needs_review_count": needs_review,
        "linkages": linkages,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
