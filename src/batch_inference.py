"""Validate Phase 2 JSONL and classify financial sentences in CPU batches."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

from src.ingestion import REQUIRED_FIELDS, validate_metadata
from src.inference import FinBertAnalyzer, SENTIMENT_LABELS
from src.model_config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION
from src.preprocessing import write_jsonl

BATCH_INFERENCE_VERSION = "1"
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


@dataclass
class BatchResult:
    predictions: list[dict[str, Any]]
    skipped: list[dict[str, Any]]

    @property
    def loaded_count(self) -> int:
        return len(self.predictions) + len(self.skipped)


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


def read_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSONL, ignore blank lines, reject malformed/duplicate records."""
    records = []
    path = Path(input_path)
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
                validate_record(record)
            except ValueError as exc:
                raise ValueError(f"{path}, line {line_number}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"Input '{path}' contains no sentence records.")
    validate_records(records)
    return records


def predict_records(
    records: list[dict[str, Any]],
    *,
    batch_size: int = 8,
    analyzer: FinBertAnalyzer | None = None,
    local_files_only: bool = True,
) -> BatchResult:
    """Preserve input order/metadata, skipping long sentences with explicit reasons.

    The supplied analyzer can be reused across documents. Otherwise one analyzer
    is loaded lazily for this invocation, only if there are eligible records.
    JSON records are held in memory; model input tensors are bounded by batch size.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    validate_records(records)
    eligible = []
    skipped = []
    for record in records:
        if record["exceeds_token_limit"] or record["token_count"] > record["max_tokens"]:
            skipped.append({**record, "skip_reason":
                "Phase 2 flagged an overlong sentence; revise/split the source and reprocess before inference."})
        else:
            eligible.append(record)
    if not eligible:
        return BatchResult([], skipped)
    if analyzer is None:
        analyzer = FinBertAnalyzer(local_files_only=local_files_only)
    if analyzer.device.type != "cpu":
        raise ValueError("Batch inference requires a CPU analyzer.")

    predictions = []
    for start in range(0, len(eligible), batch_size):
        compatible = []
        for record in eligible[start:start + batch_size]:
            actual_count = len(analyzer.tokenizer.encode(
                record["text"], add_special_tokens=True, truncation=False, verbose=False
            ))
            limit = min(record["max_tokens"], analyzer.model.config.max_position_embeddings)
            if actual_count > limit:
                skipped.append({**record, "skip_reason":
                    f"Actual token count {actual_count} exceeds {limit}; revise/split and reprocess. No truncation applied."})
                continue
            if actual_count != record["token_count"]:
                raise ValueError(
                    f"{record['sentence_id']}: stored token_count {record['token_count']} differs "
                    f"from actual count {actual_count}; reprocess the original TXT."
                )
            compatible.append(record)
        if not compatible:
            continue
        batch = analyzer.predict_batch([record["text"] for record in compatible])
        if len(batch) != len(compatible):
            raise ValueError("Analyzer returned a different number of predictions than input sentences.")
        for record, prediction in zip(compatible, batch):
            probabilities = prediction["probabilities"]
            if (prediction["sentence"] != record["text"] or set(probabilities) != SENTIMENT_LABELS
                    or prediction["label"] not in SENTIMENT_LABELS
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6)
                    or prediction["label"] != max(probabilities, key=probabilities.__getitem__)):
                raise ValueError("Analyzer returned invalid probabilities, labels or sentence ordering.")
            predictions.append({
                **record,
                "predicted_sentiment": prediction["label"],
                "prob_positive": probabilities["positive"],
                "prob_negative": probabilities["negative"],
                "prob_neutral": probabilities["neutral"],
                "model_name": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "inference_device": "cpu",
                "batch_inference_version": BATCH_INFERENCE_VERSION,
            })
    # Actual-length skips can arise after flagged skips; restore source order.
    positions = {record["sentence_id"]: i for i, record in enumerate(records)}
    skipped.sort(key=lambda record: positions[record["sentence_id"]])
    return BatchResult(predictions, skipped)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true", help="Require the existing cached checkpoint.")
    args = parser.parse_args(argv)
    skipped_path = Path(str(args.output) + ".skipped.jsonl")
    try:
        for destination in (args.output, skipped_path):
            if destination.exists():
                raise ValueError(f"Output already exists: {destination}; choose a new filename.")
        records = read_records(args.input)
        result = predict_records(records, batch_size=args.batch_size, local_files_only=args.local_files_only)
        write_jsonl(result.predictions, args.output)
        if result.skipped:
            write_jsonl(result.skipped, skipped_path)
            for record in result.skipped:
                print(f"Skipped {record['sentence_id']}: {record['skip_reason']}", file=sys.stderr)
            print(f"Skipped records and reasons: {skipped_path}", file=sys.stderr)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(
        f"Loaded: {result.loaded_count} | Classified: {len(result.predictions)} | "
        f"Skipped: {len(result.skipped)} | Model: {MODEL_ID} | Device: cpu\nOutput: {args.output}"
    )


if __name__ == "__main__":
    main()
