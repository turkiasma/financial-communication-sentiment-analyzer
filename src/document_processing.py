"""Optional-metadata upload adapter. Existing verified JSON contracts stay strict."""

import hashlib
import math
from pathlib import Path
import re
from threading import RLock

from src.analysis import sentiment_distribution
from src.ingestion import FinancialDocument
from src.inference import FinBertAnalyzer
from src.model_config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION
from src.pdf_extraction import extract_pdf, MAX_EXTRACTED_CHARS
from src.preprocessing import preprocess_document, _normalize_with_offsets, _document_sentences

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SEGMENTS = 2000
_MODEL_LOCK = RLock()
_ANALYZER = None


def get_analyzer():
    """Lazy, process-local CPU model; serialize tokenizer and inference access."""
    global _ANALYZER
    with _MODEL_LOCK:
        if _ANALYZER is None:
            _ANALYZER = FinBertAnalyzer(local_files_only=True)
        return _ANALYZER


def upload_metadata(metadata=None):
    supplied = dict(metadata or {})
    if supplied.keys() - {"company", "ticker", "fiscal_year", "fiscal_quarter"}:
        raise ValueError("Only company, ticker, fiscal year and fiscal quarter may be supplied.")
    result = {}
    for field in ("company", "ticker"):
        value = supplied.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be text.")
        value = value.strip() if value else None
        if value and (len(value) > 120 or any(ord(c) < 32 for c in value)):
            raise ValueError(f"{field} must be at most 120 characters without control characters.")
        result[field] = value or None
    year = supplied.get("fiscal_year")
    if year in (None, ""):
        result["fiscal_year"] = None
    else:
        if isinstance(year, bool) or not re.fullmatch(r"[0-9]{1,4}", str(year)) or not 1 <= int(year) <= 9999:
            raise ValueError("Fiscal year must be a whole number from 1 to 9999, or blank.")
        result["fiscal_year"] = int(year)
    quarter = supplied.get("fiscal_quarter")
    if quarter not in (None, "", "Q1", "Q2", "Q3", "Q4"):
        raise ValueError("Fiscal quarter must be Q1, Q2, Q3 or Q4, or blank.")
    result["fiscal_quarter"] = quarter or None
    return result


def token_safe_segments(record, tokenizer, max_tokens=MAX_INPUT_TOKENS):
    """Partition normalized text at word boundaries, with exact original offsets.

    No overlap or token decoding: financial figures/negations are never rewritten.
    A single unbreakable word exceeding the model limit is rejected explicitly.
    """
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))
    text = record["text"]
    normalized, mapping = _normalize_with_offsets(record["original_text"])
    if normalized != text:
        raise ValueError("Source mapping differs from normalized text.")
    if not 3 <= max_tokens <= MAX_INPUT_TOKENS:
        raise ValueError("Invalid model token limit.")
    boundaries = [m.end() for m in re.finditer(r"\S+\s*", text)]
    parts, start, boundary_index = [], 0, 0
    while start < len(text):
        if count(text[start:]) <= max_tokens:
            end = len(text)
        else:
            low, high, best = boundary_index, len(boundaries) - 1, None
            while low <= high:
                mid = (low + high) // 2
                if count(text[start:boundaries[mid]].rstrip()) <= max_tokens:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            if best is None:
                raise ValueError("A single unbreakable word exceeds the tokenizer limit; insert a meaningful word boundary in the source.")
            end = boundaries[best]
        # Keep a trailing standalone negator with its following word when possible.
        if end < len(text):
            final_word = re.search(r"\S+\s*$", text[start:end])
            if final_word and final_word.group().strip().casefold() in {"not", "no", "never", "without"} and final_word.start() > 0:
                end = start + final_word.start()
        stripped_end = end
        while stripped_end > start and text[stripped_end - 1].isspace():
            stripped_end -= 1
        source_start = record["source_char_start"] + mapping[start]
        source_end = record["source_char_start"] + mapping[stripped_end - 1] + 1
        piece = text[start:stripped_end]
        if count(piece) > max_tokens:
            raise ValueError("Token-safe splitting failed; no content was truncated.")
        parts.append({**record, "text": piece, "token_count": count(piece),
                      "original_text": record["original_text"][mapping[start]:mapping[stripped_end - 1] + 1],
                      "source_char_start": source_start, "source_char_end": source_end,
                      "parent_sentence_id": record["sentence_id"], "parent_sentence_index": record["sentence_index"],
                      "normalized_char_start": start, "normalized_char_end": stripped_end})
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
        while boundary_index < len(boundaries) and boundaries[boundary_index] <= start:
            boundary_index += 1
    for index, part in enumerate(parts):
        part.update(segment_index=index, segment_count=len(parts), split_from_long_unit=len(parts) > 1)
    assert " ".join(p["text"] for p in parts) == text
    return parts


def _process(path, metadata, analyzer, batch_size):
    path = Path(path)
    if path.suffix.lower() not in {".txt", ".pdf"}:
        raise ValueError("Upload one UTF-8 .txt or text-based .pdf document.")
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_UPLOAD_BYTES + 1)
    except OSError as exc:
        raise ValueError("Cannot read the uploaded document. Please upload it again.") from exc
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Document exceeds the 5 MiB upload limit.")
    if not content:
        raise ValueError("Uploaded document is empty.")
    facts = upload_metadata(metadata)
    if path.suffix.lower() == ".pdf":
        extraction = extract_pdf(content)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("TXT documents must use UTF-8 encoding.") from exc
        extraction = {"text": text, "page_spans": [], "omissions": [], "warnings": [], "page_count": None}
    text = extraction["text"]
    if not text.strip():
        raise ValueError("Document is empty or whitespace-only.")
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ValueError("Document exceeds the 500,000-character text limit.")
    if any(ord(c) < 32 and c not in "\t\n\r\f" for c in text):
        raise ValueError("Document contains unsupported control characters.")
    candidate_count = sum(1 for _ in _document_sentences(text))
    if candidate_count + len(extraction["omissions"]) > MAX_SEGMENTS:
        raise ValueError("Document exceeds the 2,000-segment limit; upload a smaller excerpt.")
    sha = hashlib.sha256(content).hexdigest()
    doc_id = "upload_" + sha[:20]
    # This adapter deliberately does not enter the strict verified-quarter schema.
    document = FinancialDocument({"document_id": doc_id, **facts}, text, sha)
    if analyzer is None:
        analyzer = get_analyzer()
    if analyzer.device.type != "cpu":
        raise ValueError("Document analysis requires a CPU classifier.")
    limit = min(MAX_INPUT_TOKENS, analyzer.model.config.max_position_embeddings)
    counter = lambda value: len(analyzer.tokenizer.encode(value, add_special_tokens=True, truncation=False, verbose=False))
    base = preprocess_document(document, token_counter=counter, long_sentence_policy="keep")
    segments = []
    for record in base:
        for part in token_safe_segments(record, analyzer.tokenizer, limit):
            index = len(segments)
            part.update(sentence_id=f"{doc_id}_{index + 1:04d}", sentence_index=index,
                        exceeds_token_limit=False, max_tokens=limit, segmentation_unit="token_safe_text_segment")
            part["source_pages"] = sorted({p["page"] for p in extraction["page_spans"]
                                          if p["start"] < part["source_char_end"] and p["end"] > part["source_char_start"]})
            segments.append(part)
            if len(segments) + candidate_count - len(base) + len(extraction["omissions"]) > MAX_SEGMENTS:
                raise ValueError("Document exceeds the 2,000-segment limit; upload a smaller excerpt.")
    skipped = [{"text": s, "source_char_start": start, "source_char_end": end,
                "skip_reason": "non_narrative_numeric_or_punctuation_unit", "source_pages": []}
               for s, start, end in _document_sentences(text) if not any(c.isalpha() for c in s)]
    skipped += [{"text": item["text"], "source_pages": [item["page"]], "skip_reason": item["skip_reason"]}
                for item in extraction["omissions"]]
    predictions = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start:start + batch_size]
        outputs = analyzer.predict_batch([r["text"] for r in batch])
        if len(outputs) != len(batch):
            raise ValueError("Classifier returned an incorrect prediction count.")
        for record, prediction in zip(batch, outputs):
            scores = prediction["probabilities"]
            if (prediction["sentence"] != record["text"] or set(scores) != {"positive", "negative", "neutral"}
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores.values())
                    or not math.isclose(sum(scores.values()), 1, abs_tol=1e-6)
                    or prediction["label"] not in scores or scores[prediction["label"]] != max(scores.values())):
                raise ValueError("Classifier returned invalid labels, probabilities or ordering.")
            predictions.append({**record, "predicted_sentiment": prediction["label"],
                                **{f"prob_{label}": p for label, p in scores.items()},
                                "model_name": MODEL_ID, "model_revision": MODEL_REVISION, "inference_device": "cpu"})
    summary = sentiment_distribution([r["predicted_sentiment"] for r in predictions])
    evidence = {label: sorted((r for r in predictions if r["predicted_sentiment"] == label),
                              key=lambda r: (-r[f"prob_{label}"], r["sentence_index"]))[:3]
                for label in ("positive", "negative", "neutral")}
    return {"document_id": doc_id, "document_name": path.name, "metadata": facts,
            "source_sha256": sha, "predictions": predictions, "skipped": skipped,
            "original_text_units": len(base), "split_original_units": sum(r["segment_count"] > 1 and r["segment_index"] == 0 for r in segments),
            "total_units": len(segments) + len(skipped), "classified_units": len(predictions),
            "skipped_units": len(skipped), "summary": summary, "evidence": evidence,
            "warnings": extraction["warnings"], "page_count": extraction["page_count"],
            "limitations": ["Percentages are predicted text-segment label distributions, not measured accuracy. Long sentences can contribute several segments.",
                            "English financial language only; sentence boundaries and PDF extraction are heuristic. Review the extracted content.",
                            "FinBERT confidence is uncalibrated; predictions can be wrong. No uploaded-document accuracy is established.",
                            "Sentiment describes document language, not financial health or investment value. This is not investment advice."]}


def analyze_document(path, metadata=None, *, analyzer=None, batch_size=8):
    """Analyze one local document independently of Gradio; persist no intermediate files."""
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Batch size must be an integer from 1 to 32.")
    with _MODEL_LOCK:
        return _process(path, metadata, analyzer, batch_size)
