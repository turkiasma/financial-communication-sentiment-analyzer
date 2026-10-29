"""Local TXT ingestion and metadata validation, using original synthetic text."""

import hashlib
import json
from pathlib import Path

import pytest

from src.ingestion import IngestionError, load_document, validate_metadata

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_earnings.json"


@pytest.fixture
def metadata():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_valid_ingestion_and_provenance():
    document = load_document(FIXTURE)
    raw = FIXTURE.with_suffix(".txt").read_bytes()
    assert document.text == raw.decode("utf-8")
    assert document.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert document.metadata["fiscal_quarter"] == "Q1"
    assert document.metadata["publication_date"] == "2025-05-01"
    assert document.metadata["extra"]["synthetic"] is True


def test_relative_path_independent_of_working_directory(tmp_path, monkeypatch):
    absolute_fixture = FIXTURE.resolve()
    monkeypatch.chdir(tmp_path)
    assert load_document(absolute_fixture).metadata["document_id"] == "synthetic_q1_2025"


@pytest.mark.parametrize("field,value", [
    ("company", ""), ("ticker", None), ("document_id", "../unsafe"),
    ("fiscal_year", "2025"), ("fiscal_year", True), ("fiscal_year", 0),
    ("fiscal_quarter", "Q5"), ("publication_date", "2025-02-29"),
    ("publication_date", "20250501"), ("source_url", "not-a-url"),
    ("source_url", "ftp://example.com"), ("source_url", "https://a b.com"),
    ("source_url", "https://user:secret@example.com"), ("language", "fr"),
    ("file_path", "report.pdf"), ("document_type", " "), ("extra", []),
])
def test_invalid_metadata(metadata, field, value):
    metadata[field] = value
    with pytest.raises(IngestionError):
        validate_metadata(metadata)


def test_required_and_unknown_fields(metadata):
    del metadata["company"]
    with pytest.raises(IngestionError, match="company"):
        validate_metadata(metadata)
    metadata["company"] = "Fictional Company"
    metadata["fiscal_quater"] = "Q1"
    with pytest.raises(IngestionError, match="Unknown"):
        validate_metadata(metadata)


@pytest.mark.parametrize("data", [[], None, "metadata"])
def test_metadata_must_be_object(data):
    with pytest.raises(IngestionError, match="object"):
        validate_metadata(data)


def test_missing_metadata_file(tmp_path):
    with pytest.raises(IngestionError, match="Cannot read metadata"):
        load_document(tmp_path / "missing.json")


@pytest.mark.parametrize("payload", [b"not json", b"\xff", b"[]"])
def test_invalid_metadata_file(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(IngestionError):
        load_document(path)


def test_missing_text_file(tmp_path, metadata):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(IngestionError, match="Cannot read text document"):
        load_document(path)


@pytest.mark.parametrize("payload,error", [
    (b"", "empty"), (b" \t\r\n", "empty"), (b"\xef\xbb\xbf  ", "empty"),
    (b"\xff", "UTF-8"), (b"Revenue\x00rose", "control"),
])
def test_invalid_text(tmp_path, metadata, payload, error):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / metadata["file_path"]).write_bytes(payload)
    with pytest.raises(IngestionError, match=error):
        load_document(path)


def test_read_error_is_helpful(tmp_path, metadata):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / metadata["file_path"]).mkdir()
    with pytest.raises(IngestionError, match="Cannot read text document"):
        load_document(path)


def test_explicit_text_path():
    assert load_document(FIXTURE, text_path=FIXTURE.with_suffix(".txt"))
    with pytest.raises(IngestionError, match="does not match"):
        load_document(FIXTURE, text_path="unrelated.txt")


def test_crlf_and_bom_preserved(tmp_path, metadata):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    payload = b"\xef\xbb\xbfRevenue rose.\r\nCosts fell."
    (tmp_path / metadata["file_path"]).write_bytes(payload)
    assert load_document(path).text == payload.decode("utf-8")
