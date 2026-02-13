"""
Shared fixtures for NAP Legal AI Advisor tests.

Mocks Streamlit session_state and provides sample data fixtures
so tests can run without a live Streamlit server, model files, or data files.
"""

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 1. Stub out streamlit *before* any project module is imported
#    Guard against double-execution (pytest loads conftest specially,
#    and `from tests.conftest import ...` loads it again as a module).
# ---------------------------------------------------------------------------


class SessionState(dict):
    """Dict that also supports attribute access, like st.session_state."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)


if "streamlit" not in sys.modules or not isinstance(sys.modules["streamlit"], types.ModuleType):
    _st_module = types.ModuleType("streamlit")
    _st_module.session_state = SessionState()
    _st_module.cache_resource = lambda f=None, **kw: f if f else (lambda fn: fn)
    _st_module.cache_data = lambda f=None, **kw: f if f else (lambda fn: fn)
    sys.modules["streamlit"] = _st_module
else:
    _st_module = sys.modules["streamlit"]

# ---------------------------------------------------------------------------
# 2. Add project source to sys.path so imports resolve
# ---------------------------------------------------------------------------

_APP_DIR = Path(__file__).resolve().parent.parent / "nap-legal-ai-advisor"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# ---------------------------------------------------------------------------
# 3. Import project types (after streamlit is stubbed)
# ---------------------------------------------------------------------------

from utils.data_loader import DecisionRecord  # noqa: E402


# ---------------------------------------------------------------------------
# 4. Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_session_state():
    """Clear the fake session_state before every test."""
    _st_module.session_state.clear()
    yield


def _make_decision(
    id: str = "m1234-22",
    filename: str = "m1234-22.txt",
    label: str = "HIGH_RISK",
    key_text: str = "Domstolen beslutar om fiskvandringsväg.",
    full_text: str = "Full text of the court decision goes here.",
    metadata: dict = None,
    scoring_details: dict = None,
) -> DecisionRecord:
    return DecisionRecord(
        id=id,
        filename=filename,
        label=label,
        confidence=0.85,
        key_text=key_text,
        full_text=full_text,
        sections={"domslut": "Tillstånd beviljas."},
        metadata=metadata or {"court": "Nacka TR", "date": "2024-01-15", "case_number": "M 1234-22"},
        scoring_details=scoring_details or {"outcome_desc": "Tillstånd", "domslut_measures": ["Fiskväg"], "max_cost_sek": 500000},
        extracted_measures=["Fiskväg"],
        extracted_costs=[{"amount": 500000, "currency": "SEK"}],
    )


@pytest.fixture
def sample_decision():
    return _make_decision()


@pytest.fixture
def sample_decisions():
    """Three decisions, one per risk level."""
    return [
        _make_decision(id="m1-22", label="HIGH_RISK", metadata={"court": "Nacka TR", "date": "2024-01-10", "case_number": "M 1-22"}),
        _make_decision(id="m2-22", label="MEDIUM_RISK", metadata={"court": "Växjö TR", "date": "2023-06-01", "case_number": "M 2-22"}),
        _make_decision(id="m3-22", label="LOW_RISK", metadata={"court": "Nacka TR", "date": "2024-03-20", "case_number": "M 3-22"}),
    ]
