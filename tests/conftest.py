import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "prime_forecast"))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("PF_SEARCH_BACKEND", "none")  # unit tests mock search
os.environ.setdefault("EXA_API_KEY", "test-key")
os.environ.setdefault("PF_LLM_FILTER", "0")  # heuristic-only in unit tests


@pytest.fixture
def sample_row() -> dict:
    return {
        "example_id": "pm-1",
        "market_id": "m1",
        "slug": "will-x-happen",
        "token_id": "123",
        "question": "Will X happen by June 2026?",
        "resolution_criteria": "Resolves YES if X happens before June 1, 2026.",
        "cutoff_date": "2026-05-01T00:00:00Z",
        "resolution_date": "2026-06-01T00:00:00Z",
        "outcome": 1,
        "price_at_cutoff": 0.437,
        "volume": 125000.0,
        "category": "ai_tech",
    }
