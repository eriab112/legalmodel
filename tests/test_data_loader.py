"""Tests for utils.data_loader — DecisionRecord, DataLoader query methods."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from utils.data_loader import DecisionRecord, DataLoader


# ---------------------------------------------------------------------------
# DecisionRecord dataclass
# ---------------------------------------------------------------------------

class TestDecisionRecord:
    def test_creation(self, sample_decision):
        assert sample_decision.id == "m1234-22"
        assert sample_decision.label == "HIGH_RISK"
        assert sample_decision.confidence == 0.85
        assert isinstance(sample_decision.sections, dict)
        assert isinstance(sample_decision.extracted_measures, list)

    def test_optional_fields_default(self):
        d = DecisionRecord(
            id="test",
            filename="test.txt",
            label=None,
            confidence=None,
            key_text="",
            full_text="",
            sections={},
            metadata={},
            scoring_details=None,
            extracted_measures=[],
            extracted_costs=[],
        )
        assert d.linked_water_bodies == []
        assert d.split is None

    def test_with_linked_water_bodies(self):
        d = DecisionRecord(
            id="test",
            filename="test.txt",
            label="LOW_RISK",
            confidence=0.9,
            key_text="text",
            full_text="full",
            sections={},
            metadata={},
            scoring_details=None,
            extracted_measures=[],
            extracted_costs=[],
            linked_water_bodies=[{"water_body": "Ume älv"}],
            split="train",
        )
        assert len(d.linked_water_bodies) == 1
        assert d.split == "train"


# ---------------------------------------------------------------------------
# DataLoader — tested with mocked file I/O
# ---------------------------------------------------------------------------

def _make_cleaned_data(decisions):
    return json.dumps({
        "decisions": [
            {
                "id": d["id"],
                "filename": d.get("filename", f"{d['id']}.txt"),
                "text_full": d.get("text_full", "Full text"),
                "key_text": d.get("key_text", "Key text"),
                "sections": d.get("sections", {}),
                "metadata": d.get("metadata", {}),
                "extracted_measures": d.get("extracted_measures", []),
                "extracted_costs": d.get("extracted_costs", []),
            }
            for d in decisions
        ]
    })


def _make_labeled_data(splits, label_distribution=None):
    return json.dumps({
        "label_distribution": label_distribution or {},
        "splits": splits,
    })


def _create_loader(cleaned_decisions, splits, label_dist=None, linkages=None):
    """Create a DataLoader with mocked file reads."""
    cleaned_json = _make_cleaned_data(cleaned_decisions)
    labeled_json = _make_labeled_data(splits, label_dist)
    linkage_json = json.dumps({"linkages": linkages or []})

    files = {
        str(DataLoader.__init__.__code__): "unused",  # placeholder
    }

    def fake_open(path, *args, **kwargs):
        path_str = str(path)
        if "cleaned_court_texts" in path_str:
            return mock_open(read_data=cleaned_json)()
        elif "labeled_dataset" in path_str:
            return mock_open(read_data=labeled_json)()
        elif "linkage_table" in path_str:
            return mock_open(read_data=linkage_json)()
        raise FileNotFoundError(path_str)

    with patch("builtins.open", side_effect=fake_open):
        with patch.object(Path, "exists", return_value=True):
            loader = DataLoader()
    return loader


class TestDataLoader:
    @pytest.fixture
    def loader(self):
        decisions = [
            {"id": "m1", "metadata": {"court": "Nacka TR", "date": "2024-01-10"}},
            {"id": "m2", "metadata": {"court": "Växjö TR", "date": "2023-06-01"}},
            {"id": "m3", "metadata": {"court": "Nacka TR", "date": "2024-03-20"}},
            {"id": "m4", "metadata": {"court": "Umeå TR", "date": "2022-12-01"}},
        ]
        splits = {
            "train": [
                {"id": "m1", "label": "HIGH_RISK", "key_text": "text1",
                 "metadata": {"court": "Nacka TR", "date": "2024-01-10"},
                 "scoring_details": {"domslut_measures": ["Fiskväg"], "max_cost_sek": 100000}},
                {"id": "m2", "label": "MEDIUM_RISK", "key_text": "text2",
                 "metadata": {"court": "Växjö TR", "date": "2023-06-01"},
                 "scoring_details": {"domslut_measures": ["Minimitappning"]}},
            ],
            "val": [
                {"id": "m3", "label": "LOW_RISK", "key_text": "text3",
                 "metadata": {"court": "Nacka TR", "date": "2024-03-20"},
                 "scoring_details": {"domslut_measures": ["Fiskväg"]}},
            ],
            "test": [],
        }
        label_dist = {"HIGH_RISK": 1, "MEDIUM_RISK": 1, "LOW_RISK": 1}
        return _create_loader(decisions, splits, label_dist)

    def test_get_all_decisions(self, loader):
        all_dec = loader.get_all_decisions()
        assert len(all_dec) == 4

    def test_get_labeled_decisions(self, loader):
        labeled = loader.get_labeled_decisions()
        assert len(labeled) == 3
        assert all(d.label is not None for d in labeled)

    def test_get_decision_by_id(self, loader):
        d = loader.get_decision("m1")
        assert d is not None
        assert d.id == "m1"
        assert d.label == "HIGH_RISK"

    def test_get_decision_unknown_id(self, loader):
        assert loader.get_decision("nonexistent") is None

    def test_get_decisions_by_label(self, loader):
        high = loader.get_decisions_by_label("HIGH_RISK")
        assert len(high) == 1
        assert high[0].id == "m1"

    def test_unlabeled_decision(self, loader):
        d = loader.get_decision("m4")
        assert d is not None
        assert d.label is None
        assert d.split is None

    def test_get_label_distribution(self, loader):
        dist = loader.get_label_distribution()
        assert dist == {"HIGH_RISK": 1, "MEDIUM_RISK": 1, "LOW_RISK": 1}

    def test_get_courts(self, loader):
        courts = loader.get_courts()
        assert "Nacka TR" in courts
        assert "Växjö TR" in courts

    def test_get_date_range(self, loader):
        date_min, date_max = loader.get_date_range()
        assert date_min == "2023-06-01"
        assert date_max == "2024-03-20"

    def test_get_measure_frequency(self, loader):
        freq = loader.get_measure_frequency()
        assert freq["Fiskväg"] == 2
        assert freq["Minimitappning"] == 1

    def test_split_assignment(self, loader):
        d1 = loader.get_decision("m1")
        assert d1.split == "train"
        d3 = loader.get_decision("m3")
        assert d3.split == "val"
