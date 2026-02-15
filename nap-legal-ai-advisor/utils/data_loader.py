"""
Data loading and merging for NAP Legal AI Advisor.

Merges labeled_dataset_binary.json (44 labeled decisions) with
cleaned_court_texts.json (50 total decisions) and linkage_table.json.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Data" / "processed"

LABELED_PATH = DATA_DIR / "labeled_dataset.json"
BINARY_LABELED_PATH = DATA_DIR / "labeled_dataset_binary.json"
CLEANED_PATH = DATA_DIR / "cleaned_court_texts.json"
LINKAGE_PATH = DATA_DIR / "linkage_table.json"
LEGISLATION_PATH = DATA_DIR / "lagtiftning_texts.json"
APPLICATION_PATH = DATA_DIR / "ansokan_texts.json"


@dataclass
class DocumentRecord:
    """A document chunk-ready record for the unified search index."""
    doc_id: str
    doc_type: str  # "decision", "legislation", "application"
    filename: str
    title: str  # human-readable title
    text: str  # full text content for chunking
    metadata: Dict  # type-specific metadata
    label: Optional[str] = None  # only for decisions


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

    def _load_all(self) -> None:
        """Load and merge all data sources into DecisionRecord objects."""
        # Load cleaned texts (46 decisions - full text + sections)
        with open(CLEANED_PATH, "r", encoding="utf-8") as f:
            cleaned_data = json.load(f)
        cleaned_by_id = {d["id"]: d for d in cleaned_data["decisions"]}

        # Prefer binary labels if available
        labeled_path = BINARY_LABELED_PATH if BINARY_LABELED_PATH.exists() else LABELED_PATH
        with open(labeled_path, "r", encoding="utf-8") as f:
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
        """Return all decisions (labeled and unlabeled)."""
        return list(self._decisions.values())

    def get_decision(self, dec_id: str) -> Optional[DecisionRecord]:
        """Return a single decision by ID, or None if not found."""
        return self._decisions.get(dec_id)

    def get_labeled_decisions(self) -> List[DecisionRecord]:
        """Return only decisions that have a risk label."""
        return [d for d in self._decisions.values() if d.label is not None]

    def get_decisions_by_label(self, label: str) -> List[DecisionRecord]:
        """Return decisions matching the given risk label."""
        return [d for d in self._decisions.values() if d.label == label]

    def get_label_distribution(self) -> Dict[str, int]:
        """Return label distribution from the dataset metadata."""
        return self._label_distribution

    def get_measure_frequency(self) -> Dict[str, int]:
        """Return measure frequency counts sorted by frequency."""
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
        """Return sorted list of unique court names."""
        courts = set()
        for d in self.get_labeled_decisions():
            court = d.metadata.get("court", "")
            if court:
                courts.add(court)
        return sorted(courts)

    def get_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """Return (min_date, max_date) from labeled decisions."""
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


class KnowledgeBase:
    """Loads all document types (decisions, legislation, applications) for the RAG system."""

    def __init__(self):
        self._documents: List[DocumentRecord] = []
        self._by_id: Dict[str, DocumentRecord] = {}
        self._load_all()

    def _doc_id_from_filename(self, filename: str) -> str:
        """Derive doc_id from filename: strip extension, replace spaces with underscores, lowercase."""
        stem = Path(filename).stem
        return stem.replace(" ", "_").lower()

    def _load_all(self) -> None:
        """Load all document types into the knowledge base."""
        # Build label lookup — prefer binary labels if available
        labeled_path = BINARY_LABELED_PATH if BINARY_LABELED_PATH.exists() else LABELED_PATH
        label_by_id: Dict[str, str] = {}
        if labeled_path.exists():
            with open(labeled_path, "r", encoding="utf-8") as f:
                labeled_data = json.load(f)
            for split_name in ["train", "val", "test"]:
                for item in labeled_data.get("splits", {}).get(split_name, []):
                    if item.get("label"):
                        label_by_id[item["id"]] = item["label"]

        # Court decisions from cleaned_court_texts.json
        if CLEANED_PATH.exists():
            with open(CLEANED_PATH, "r", encoding="utf-8") as f:
                cleaned_data = json.load(f)
            for d in cleaned_data.get("decisions", []):
                doc_id = d.get("id", "")
                meta = d.get("metadata", {})
                case_number = meta.get("case_number", doc_id)
                court = meta.get("court", "")
                date = meta.get("date", "")
                parts = [case_number]
                if court:
                    parts.append(court)
                if date:
                    parts.append(f"({date})")
                title = " — ".join(parts) if parts else doc_id
                record = DocumentRecord(
                    doc_id=doc_id,
                    doc_type="decision",
                    filename=d.get("filename", ""),
                    title=title,
                    text=d.get("text_full", ""),
                    metadata={
                        "court": meta.get("court"),
                        "date": meta.get("date"),
                        "case_number": meta.get("case_number"),
                    },
                    label=label_by_id.get(doc_id),
                )
                self._documents.append(record)
                self._by_id[doc_id] = record

        n_decisions = len([r for r in self._documents if r.doc_type == "decision"])

        # Legislation from lagtiftning_texts.json
        if LEGISLATION_PATH.exists():
            with open(LEGISLATION_PATH, "r", encoding="utf-8") as f:
                legislation_data = json.load(f)
            seen_doc_ids = set(self._by_id.keys())
            for entry in legislation_data:
                filename = entry.get("filename", "")
                if filename == "CIS_Guidance_Article_4_7_FINAL (1).pdf":
                    continue
                text = entry.get("text", "")
                if len(text) <= 500:
                    continue
                doc_id = self._doc_id_from_filename(filename)
                if doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                title = Path(filename).stem
                record = DocumentRecord(
                    doc_id=doc_id,
                    doc_type="legislation",
                    filename=filename,
                    title=title,
                    text=text,
                    metadata={"source": "legislation", "filename": filename},
                )
                self._documents.append(record)
                self._by_id[doc_id] = record

        n_legislation = len([r for r in self._documents if r.doc_type == "legislation"])

        # Applications from ansokan_texts.json
        if APPLICATION_PATH.exists():
            with open(APPLICATION_PATH, "r", encoding="utf-8") as f:
                application_data = json.load(f)
            for entry in application_data:
                filename = entry.get("filename", "")
                text = entry.get("text", "")
                if len(text) <= 500:
                    continue
                doc_id = self._doc_id_from_filename(filename)
                if doc_id in self._by_id:
                    continue
                title = Path(filename).stem
                record = DocumentRecord(
                    doc_id=doc_id,
                    doc_type="application",
                    filename=filename,
                    title=title,
                    text=text,
                    metadata={"source": "application", "filename": filename},
                )
                self._documents.append(record)
                self._by_id[doc_id] = record

        n_applications = len([r for r in self._documents if r.doc_type == "application"])
        total = len(self._documents)
        print(
            f"KnowledgeBase loaded: {n_decisions} decisions, {n_legislation} legislation, "
            f"{n_applications} applications ({total} total)"
        )

    def get_all_documents(self) -> List[DocumentRecord]:
        """Return all documents in the knowledge base."""
        return list(self._documents)

    def get_documents_by_type(self, doc_type: str) -> List[DocumentRecord]:
        """Return documents filtered by type (decision, legislation, application)."""
        return [d for d in self._documents if d.doc_type == doc_type]

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Return a single document by ID, or None if not found."""
        return self._by_id.get(doc_id)

    def get_corpus_stats(self) -> Dict[str, int]:
        """Return document count per type and total."""
        counts = {}
        for d in self._documents:
            counts[d.doc_type] = counts.get(d.doc_type, 0) + 1
        counts["total"] = len(self._documents)
        return counts


@st.cache_data
def load_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
