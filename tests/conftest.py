"""Share a single real model load across all opt-in integration tests."""

import pytest

from src.inference import FinBertAnalyzer


@pytest.fixture(scope="session")
def cached_analyzer():
    return FinBertAnalyzer(local_files_only=True)
