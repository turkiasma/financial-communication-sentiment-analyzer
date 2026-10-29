"""Model-independent validation of Phase 2 records and strict JSONL reading."""

import json
from pathlib import Path
import re
from typing import Any

from src.ingestion import REQUIRED_FIELDS, validate_metadata
from src.model_config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION

RECORD_FIELDS = REQUIRED_FIELDS | {
    "sentence_id", "sentence_index", "text", "original_text", "source_char_start",
    "source_char_end", "source_sha256", "token_count", "exceeds_token_limit",
    "max_tokens", "tokenizer_model", "tokenizer_revision", "long_sentence_policy",
    "preprocessing_version",
}
PREDICTION_FIELDS = {
    "predicted_sentiment", "prob_positive", "prob_negative", "prob_neutral",
    "model_name", "model_revision", "inference_device", "batch_inference_version", "skip_reason",
}



def validate_record(record: Any) -> None:
    """Check the Phase 2 contract without altering any supplied metadata."""
    if not isinstance(record, dict):
        raise ValueError("Each sentence record must be a JSON object.")
    missing = RECORD_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Missing sentence fields: {', '.join(sorted(missing))}")
    collisions = PREDICTION_FIELDS & record.keys()
    if collisions:
        raise ValueError(f"Input already contains prediction fields: {', '.join(sorted(collisions))}")
    validate_metadata({key: record[key] for key in REQUIRED_FIELDS | {"extra"} if key in record})
    for field in ("text", "original_text", "preprocessing_version"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")
    for field in ("sentence_index", "source_char_start", "source_char_end", "token_count", "max_tokens"):
        if type(record[field]) is not int or record[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer.")
    expected_id = f"{record['document_id']}_{record['sentence_index'] + 1:04d}"
    if record["sentence_id"] != expected_id:
        raise ValueError(f"sentence_id must match its document/index: {expected_id}")
    if (record["source_char_end"] - record["source_char_start"] != len(record["original_text"])):
        raise ValueError("Source offsets do not match original_text length.")
    if not isinstance(record["source_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"]):
        raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest.")
    if type(record["exceeds_token_limit"]) is not bool:
        raise ValueError("exceeds_token_limit must be a boolean.")
    if record["token_count"] < 2 or not 3 <= record["max_tokens"] <= MAX_INPUT_TOKENS:
        raise ValueError("Invalid token_count or max_tokens in sentence record.")
    if record["tokenizer_model"] != MODEL_ID or record["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Tokenizer identity differs from the pinned FinBERT snapshot; reprocess the TXT.")
    if not isinstance(record["long_sentence_policy"], str) or record["long_sentence_policy"] not in {"error", "keep"}:
        raise ValueError("Invalid Phase 2 long_sentence_policy.")
    if "segmentation_unit" in record and (not isinstance(record["segmentation_unit"], str) or not record["segmentation_unit"].strip()):
        raise ValueError("segmentation_unit must be a non-empty string.")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON constant: {value}")


def validate_records(records: list[dict[str, Any]]) -> None:
    seen = set()
    for index, record in enumerate(records):
        try:
            validate_record(record)
            if record["sentence_id"] in seen:
                raise ValueError(f"Duplicate sentence_id: {record['sentence_id']}")
            seen.add(record["sentence_id"])
        except ValueError as exc:
            raise ValueError(f"Record {index + 1}: {exc}") from exc


def read_jsonl(input_path: str | Path, *, allow_empty: bool = False) -> list[Any]:
    """Read strict UTF-8 JSONL without importing a tokenizer or model."""
    records = []
    path = Path(input_path)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                if not isinstance(record, dict):
                    raise ValueError("Each JSONL record must be a JSON object.")
            except ValueError as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
            records.append(record)
    if not records and not allow_empty:
        raise ValueError(f"Input '{path}' contains no sentence records.")
    return records


def read_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Read and validate Phase 2 sentence records, rejecting duplicate IDs."""
    records = read_jsonl(input_path)
    validate_records(records)
    return records
