"""Session-owned upload/export adapter; no Streamlit or Gradio imports."""

import hashlib
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from src.document_processing import analyze_document, MAX_UPLOAD_BYTES
from src.reporting import AnalysisDownloads

RESULT_KEY = "finsentiment_result"
INPUT_KEY = "finsentiment_input"


def clear_result(state):
    state.pop(RESULT_KEY, None)


def synchronize_input(state, identity):
    """Invalidate results when the upload or optional metadata changes."""
    if state.get(INPUT_KEY) != identity:
        clear_result(state)
        state[INPUT_KEY] = identity


def upload_identity(filename, content, metadata):
    return (filename, hashlib.sha256(content).hexdigest(),
            tuple(sorted((key, str(value)) for key, value in metadata.items())))


def read_upload(upload):
    if upload is None:
        raise ValueError("Choose a TXT or text-based PDF document first.")
    # UploadedFile is BytesIO-like: inspect the buffer before copying it.
    content = upload.getbuffer()
    if content.nbytes > MAX_UPLOAD_BYTES:
        raise ValueError("Document exceeds the 5 MiB upload limit.")
    if not content.nbytes:
        raise ValueError("Uploaded document is empty.")
    return upload.name, content


def analyze_uploaded_bytes(filename, content, metadata=None, *, analyzer=None):
    """Use the existing pipeline/exporters, then immediately remove disk artifacts."""
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    if (not name or name in {".", ".."} or len(name.encode("utf-8")) > 240
            or any(ord(char) < 32 for char in name)):
        raise ValueError("Upload a document with a short, valid filename.")
    if Path(name).suffix.lower() not in {".txt", ".pdf"}:
        raise ValueError("Upload one UTF-8 .txt or text-based .pdf document.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Document exceeds the 5 MiB upload limit.")
    if not content:
        raise ValueError("Uploaded document is empty.")
    with TemporaryDirectory(prefix="finsentiment_streamlit_upload_") as directory:
        path = Path(directory) / name
        path.write_bytes(content)
        result = analyze_document(path, metadata, analyzer=analyzer)
        downloads = AnalysisDownloads(result)
        try:
            csv_bytes, pdf_bytes = downloads.csv.read_bytes(), downloads.report.read_bytes()
        finally:
            downloads.cleanup()
    # Retain only presentation data and two downloads in this user's session.
    evidence = [{"Predicted sentiment": label.title(),
                 "Confidence": round(row[f"prob_{label}"], 4), "Statement": row["text"]}
                for label in ("positive", "negative", "neutral") for row in result["evidence"][label]]
    return {"summary": result["summary"], "classified_units": result["classified_units"],
            "split_original_units": result["split_original_units"],
            "skipped_units": result["skipped_units"], "warnings": result["warnings"],
            "page_count": result["page_count"], "evidence": evidence,
            "csv_bytes": csv_bytes, "pdf_bytes": pdf_bytes}


def process_for_session(state, filename, content, metadata=None, *, analyzer=None):
    clear_result(state)  # Also clear when extraction, inference or export fails.
    result = analyze_uploaded_bytes(filename, content, metadata, analyzer=analyzer)
    state[RESULT_KEY] = result
    return result


def user_error(exc):
    """Return controlled validation messages without raw paths or tracebacks."""
    message = str(exc)
    if (isinstance(exc, ValueError) and len(message) <= 500
            and not re.search(r"(?:^|[\s'\"])(?:/|[A-Za-z]:\\)", message)):
        return message
    return "Analysis could not complete. Check the document, model availability and service resources."
