"""Validate JSON metadata and read a local UTF-8 financial document."""

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

REQUIRED_FIELDS = frozenset({
    "document_id", "company", "ticker", "fiscal_year", "fiscal_quarter",
    "document_type", "publication_date", "source_url", "language", "file_path",
})


class IngestionError(ValueError):
    """An invalid document, metadata file or text encoding."""


@dataclass(frozen=True)
class FinancialDocument:
    metadata: dict[str, Any]
    text: str
    source_sha256: str


def validate_metadata(data: Any) -> dict[str, Any]:
    """Validate supplied facts without inferring missing financial information.

    Extra provenance fields may be stored in the optional `extra` JSON object.
    Fiscal year/quarter describe the company's fiscal calendar, not publication.
    """
    if not isinstance(data, dict):
        raise IngestionError("Metadata must be a JSON object.")
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise IngestionError(f"Missing metadata fields: {', '.join(sorted(missing))}")
    unknown = data.keys() - REQUIRED_FIELDS - {"extra"}
    if unknown:
        raise IngestionError(f"Unknown metadata fields: {', '.join(sorted(unknown))}; use 'extra'.")
    for field in REQUIRED_FIELDS - {"fiscal_year"}:
        value = data[field]
        if not isinstance(value, str) or not value.strip():
            raise IngestionError(f"Metadata '{field}' must be a non-empty string.")
        if value != value.strip():
            raise IngestionError(f"Metadata '{field}' must not have surrounding whitespace.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", data["document_id"]):
        raise IngestionError("document_id must use letters, digits, underscores or hyphens.")
    year = data["fiscal_year"]
    if type(year) is not int or not 1 <= year <= 9999:
        raise IngestionError("fiscal_year must be an integer from 1 to 9999.")
    if data["fiscal_quarter"] not in {"Q1", "Q2", "Q3", "Q4"}:
        raise IngestionError("fiscal_quarter must be Q1, Q2, Q3 or Q4.")
    publication = data["publication_date"]
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", publication):
            raise ValueError
        date.fromisoformat(publication)
    except ValueError as exc:
        raise IngestionError("publication_date must be a valid YYYY-MM-DD date.") from exc
    try:
        url = urlsplit(data["source_url"])
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username or url.password or re.search(r"\s", data["source_url"])):
            raise ValueError
    except ValueError as exc:
        raise IngestionError("source_url must be an HTTP(S) URL without credentials.") from exc
    if data["language"] != "en":
        raise IngestionError("Only English documents (language='en') are supported.")
    if "\x00" in data["file_path"] or Path(data["file_path"]).suffix.lower() != ".txt":
        raise IngestionError("file_path must identify a local .txt file.")
    if "extra" in data and not isinstance(data["extra"], dict):
        raise IngestionError("extra must be a JSON object.")
    return dict(data)


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise IngestionError(f"Cannot read {description} '{path}': {exc.strerror}") from exc


def load_document(metadata_path: str | Path, *, text_path: str | Path | None = None) -> FinancialDocument:
    """Resolve file_path relative to its metadata JSON, never the working directory.

    An optional explicit text_path must resolve to the same file as metadata.
    Text is decoded without newline conversion so source character offsets remain
    meaningful. The SHA-256 identifies the exact original bytes.
    """
    metadata_path = Path(metadata_path)
    try:
        raw = json.loads(_read_bytes(metadata_path, "metadata").decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IngestionError(f"Invalid UTF-8 JSON metadata '{metadata_path}': {exc}") from exc
    metadata = validate_metadata(raw)
    source = Path(metadata["file_path"])
    if not source.is_absolute():
        source = metadata_path.parent / source
    if text_path is not None and Path(text_path).resolve() != source.resolve():
        raise IngestionError("--text does not match file_path in metadata (relative to the JSON file).")
    content = _read_bytes(source, "text document")
    try:
        text = content.decode("utf-8")
    except UnicodeError as exc:
        raise IngestionError(f"Text document '{source}' must be UTF-8: {exc}") from exc
    if not text.lstrip("\ufeff").strip():
        raise IngestionError(f"Text document '{source}' is empty or whitespace-only.")
    if any(ord(char) < 32 and char not in "\t\n\r\f" for char in text):
        raise IngestionError(f"Text document '{source}' contains unsupported control characters.")
    return FinancialDocument(metadata, text, hashlib.sha256(content).hexdigest())
