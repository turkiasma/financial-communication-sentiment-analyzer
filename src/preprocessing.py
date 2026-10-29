"""Conservative sentence preprocessing with provenance; no sentiment inference."""

import argparse
from collections.abc import Callable, Iterator
import json
from pathlib import Path
import re
from typing import Any

from src.ingestion import FinancialDocument, load_document
from src.model_config import MAX_INPUT_TOKENS, MODEL_ID, MODEL_REVISION

PREPROCESSING_VERSION = "1"
TokenCounter = Callable[[str], int]
# Only formatting-only lines and unnumbered list prefixes are removed.
_ARTIFACT = re.compile(r"(?m)^[ \t\ufeff]*[-_=*]{3,}[ \t]*\r?$|^[ \t\ufeff]*[-*•·][ \t]+")
_END = re.compile(r'''[.!?]+["'”’\)\]]*(?=\s|$)''')
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "vs", "e.g", "i.e", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
_COMPANY_SUFFIXES = frozenset({"inc", "corp", "ltd", "co", "plc"})


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Map each normalized character back to its original character index."""
    removed = {i for match in _ARTIFACT.finditer(text) for i in range(*match.span())}
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if index in removed or char == "\ufeff":
            continue
        if char == "\u200b":
            char = " "
        if char.isspace():
            if not chars or chars[-1] == " ":
                continue
            char = " "
        chars.append(char)
        offsets.append(index)
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def normalize_text(text: str) -> str:
    """Collapse whitespace; retain numbers, punctuation, symbols and negation."""
    return _normalize_with_offsets(text)[0]


def sentence_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield sentence offsets in normalized text using a small English rule set.

    Decimal points have no following whitespace and are never boundaries. Common
    titles/months and dotted abbreviations are protected. Company suffixes before
    lowercase continuations are protected; an uppercase following word ends the
    sentence. This is a heuristic, not a general linguistic parser.
    """
    start = 0
    for match in _END.finditer(text):
        end = match.end()
        punctuation = match.group()
        if punctuation.startswith(".") and not punctuation.startswith(".."):
            prefix = text[:match.start()]
            word_match = re.search(r"([A-Za-z.]+)$", prefix)
            word = word_match.group(1).lower() if word_match else ""
            following = text[end:].lstrip()
            # Initials and dotted acronyms: U.S., U.K., U.S.A., J. Smith.
            acronym = re.fullmatch(r"(?:[a-z]\.)*[a-z]", word)
            if following and (word in _ABBREVIATIONS or acronym
                              or (word in _COMPANY_SUFFIXES and following[0].islower())):
                continue
        if text[start:end].strip():
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            yield left, end
        start = end
    if text[start:].strip():
        left = start + len(text[start:]) - len(text[start:].lstrip())
        yield left, len(text)


def load_token_counter() -> TokenCounter:
    """Load only the existing pinned tokenizer; never download or load weights."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True
        )
    except OSError as exc:
        raise ValueError("Pinned FinBERT tokenizer is not cached; complete Phase 1 first.") from exc

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))

    return count_tokens


def preprocess_document(
    document: FinancialDocument,
    *,
    token_counter: TokenCounter,
    long_sentence_policy: str = "error",
    max_tokens: int = MAX_INPUT_TOKENS,
) -> list[dict[str, Any]]:
    """Emit ordered records; reject overlong sentences or explicitly retain/flag.

    No sentence is truncated or split into context-free chunks. Output is fully
    prepared before the CLI opens a destination file, including on length errors.
    """
    if long_sentence_policy not in {"error", "keep"}:
        raise ValueError("long_sentence_policy must be 'error' or 'keep'.")
    if type(max_tokens) is not int or not 2 < max_tokens <= MAX_INPUT_TOKENS:
        raise ValueError(f"max_tokens must be an integer from 3 to {MAX_INPUT_TOKENS}.")
    normalized, offsets = _normalize_with_offsets(document.text)
    records = []
    for start, end in sentence_spans(normalized):
        sentence = normalized[start:end]
        # Discard punctuation-only and numeric-only/table fragments, not short words.
        if not any(char.isalpha() for char in sentence):
            continue
        index = len(records)
        count = token_counter(sentence)
        too_long = count > max_tokens
        sentence_id = f"{document.metadata['document_id']}_{index + 1:04d}"
        if too_long and long_sentence_policy == "error":
            raise ValueError(
                f"Sentence {sentence_id} has {count} tokens, exceeding {max_tokens}. "
                "No output written. Use --long-sentence-policy keep to retain and flag for review."
            )
        source_start, source_end = offsets[start], offsets[end - 1] + 1
        records.append({
            **document.metadata,
            "sentence_id": sentence_id,
            "sentence_index": index,
            "text": sentence,
            "original_text": document.text[source_start:source_end],
            "source_char_start": source_start,
            "source_char_end": source_end,
            "source_sha256": document.source_sha256,
            "token_count": count,
            "exceeds_token_limit": too_long,
            "max_tokens": max_tokens,
            "tokenizer_model": MODEL_ID,
            "tokenizer_revision": MODEL_REVISION,
            "long_sentence_policy": long_sentence_policy,
            "preprocessing_version": PREPROCESSING_VERSION,
        })
    if not records:
        raise ValueError("Document contains no usable text after preprocessing.")
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write UTF-8 JSONL, refusing to overwrite any existing file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True, help="UTF-8 metadata JSON.")
    parser.add_argument("--text", type=Path, help="Optional TXT path; must match metadata file_path.")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL output file.")
    parser.add_argument("--long-sentence-policy", choices=("error", "keep"), default="error")
    args = parser.parse_args(argv)
    try:
        document = load_document(args.metadata, text_path=args.text)
        counter = load_token_counter()
        records = preprocess_document(
            document, token_counter=counter, long_sentence_policy=args.long_sentence_policy
        )
        write_jsonl(records, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    flagged = sum(record["exceeds_token_limit"] for record in records)
    print(f"Wrote {len(records)} records to {args.output}; {flagged} exceed the token limit.")


if __name__ == "__main__":
    main()
