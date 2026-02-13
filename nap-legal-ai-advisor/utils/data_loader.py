"""
Data loading and merging for NAP Legal AI Advisor.

Merges labeled_dataset.json (40 labeled decisions) with
cleaned_court_texts.json (46 total decisions) and linkage_table.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Data" / "processed"

LABELED_PATH = DATA_DIR / "labeled_dataset.json"
CLEANED_PATH = DATA_DIR / "cleaned_court_texts.json"
LINKAGE_PATH = DATA_DIR / "linkage_table.json"


@dataclass
class DecisionRecord:
    id: str
    filename: str
    label: Optional[str]  # None for excluded decisions
    confidence: Optional[float]
    key_text: str
    full_text: str
    sections: Dict[str, str]
    metadata: Dict
    scoring_details: Optional[Dict]
    extracted_measures: List[str]
    extracted_costs: List[Dict]
    linked_water_bodies: List[Dict] = field(default_factory=list)
    split: Optional[str] = None  # train/val/test


class DataLoader:
    """Loads and merges all data sources into DecisionRecord objects."""

    def __init__(self):
        self._decisions: Dict[str, DecisionRecord] = {}
        self._label_distribution: Dict[str, int] = {}
        self._load_all()

    def _load_all(self):
        # Load cleaned texts (46 decisions - full text + sections)
        with open(CLEANED_PATH, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
        cleaned_by_id = {d["id"]: d for d in cleaned_data["decisions"]}

        # Load labeled dataset (40 decisions - labels + scoring)
        with open(LABELED_PATH, "r", encoding="utf-8") as f:
            labeled_data = json.load(f)

        self._label_distribution = labeled_data.get("label_distribution", {})

        # Build lookup: id -> (label_info, split)
        labeled_by_id = {}
        for split_name in ["train", "val", "test"]:
            for item in labeled_data.get("splits", {}).get(split_name, []):
                labeled_by_id[item["id"]] = (item, split_name)

        # Load linkage table
        linkage_by_file = {}
        if LINKAGE_PATH.exists():
            with open(LINKAGE_PATH, "r", encoding="utf-8") as f:
                linkage_data = json.load(f)
            for link in linkage_data.get("linkages", []):
                for fname in link.get("files", []):
                    linkage_by_file.setdefault(fname, []).append({
                        "water_body": link.get("court_water_body", ""),
                        "viss_ids": link.get("linked_viss_ids", []),
                        "match_tier": link.get("match_tier", ""),
                        "confidence": link.get("confidence", 0),
                        "in_mcda": link.get("in_mcda", False),
                    })

        # Merge all sources
        for dec_id, cleaned in cleaned_by_id.items():
            label_info, split = labeled_by_id.get(dec_id, (None, None))

            fname = cleaned.get("filename", "")
            linked_wbs = linkage_by_file.get(fname, [])

            if label_info:
                record = DecisionRecord(
                    id=dec_id,
                    filename=fname,
                    label=label_info.get("label"),
                    confidence=label_info.get("confidence"),
                    key_text=label_info.get("key_text", cleaned.get("key_text", "")),
                    full_text=cleaned.get("text_full", ""),
                    sections=cleaned.get("sections", {}),
                    metadata=label_info.get("metadata", cleaned.get("metadata", {})),
                    scoring_details=label_info.get("scoring_details"),
                    extracted_measures=cleaned.get("extracted_measures", []),
                    extracted_costs=cleaned.get("extracted_costs", []),
                    linked_water_bodies=linked_wbs,
                    split=split,
                )
            else:
                # Excluded or unlabeled decision
                record = DecisionRecord(
                    id=dec_id,
                    filename=fname,
                    label=None,
                    confidence=None,
                    key_text=cleaned.get("key_text", ""),
                    full_text=cleaned.get("text_full", ""),
                    sections=cleaned.get("sections", {}),
                    metadata=cleaned.get("metadata", {}),
                    scoring_details=None,
                    extracted_measures=cleaned.get("extracted_measures", []),
                    extracted_costs=cleaned.get("extracted_costs", []),
                    linked_water_bodies=linked_wbs,
                    split=None,
                )

            self._decisions[dec_id] = record

    def get_all_decisions(self) -> List[DecisionRecord]:
        return list(self._decisions.values())

    def get_decision(self, dec_id: str) -> Optional[DecisionRecord]:
        return self._decisions.get(dec_id)

    def get_labeled_decisions(self) -> List[DecisionRecord]:
        return [d for d in self._decisions.values() if d.label is not None]

    def get_decisions_by_label(self, label: str) -> List[DecisionRecord]:
        return [d for d in self._decisions.values() if d.label == label]

    def get_label_distribution(self) -> Dict[str, int]:
        return self._label_distribution

    def get_measure_frequency(self) -> Dict[str, int]:
        freq = {}
        for d in self.get_labeled_decisions():
            measures = []
            if d.scoring_details:
                measures = d.scoring_details.get("domslut_measures", [])
            if not measures:
                measures = d.extracted_measures
            for m in measures:
                freq[m] = freq.get(m, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    def get_courts(self) -> List[str]:
        courts = set()
        for d in self.get_labeled_decisions():
            court = d.metadata.get("court", "")
            if court:
                courts.add(court)
        return sorted(courts)

    def get_date_range(self):
        dates = []
        for d in self.get_labeled_decisions():
            date = d.metadata.get("date", "")
            if date:
                dates.append(date)
        if dates:
            return min(dates), max(dates)
        return None, None


@st.cache_data
def load_data() -> DataLoader:
    return DataLoader()
