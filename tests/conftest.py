"""Share a single real model load across all opt-in integration tests."""

import pytest

from src.inference import FinBertAnalyzer


@pytest.fixture(scope="session")
def cached_analyzer():
    return FinBertAnalyzer(local_files_only=True)


@pytest.fixture
def pdf_text():
    """Read actual report contents independently with the installed PDF parser."""
    import pymupdf
    def read(path):
        with pymupdf.open(path) as document:
            return " ".join(" ".join(page.get_text(flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES).split()) for page in document)
    return read
