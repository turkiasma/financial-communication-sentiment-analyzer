"""Deterministic preprocessing tests; ordinary cases require no model or network."""

import json
import os
from pathlib import Path

import pytest

from src.ingestion import FinancialDocument, load_document
from src.model_config import MODEL_ID, MODEL_REVISION
from src.preprocessing import (
    load_token_counter, main, normalize_text, preprocess_document, sentence_spans, write_jsonl,
)

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_earnings.json"


@pytest.fixture
def document():
    return load_document(FIXTURE)


def fake_counter(text):
    """A deterministic stand-in; real cached WordPiece is tested separately."""
    return len(text.split()) + 2


def with_text(document, text):
    return FinancialDocument(document.metadata, text, "test-source-hash")


def split(text):
    normalized = normalize_text(text)
    return [normalized[start:end] for start, end in sentence_spans(normalized)]


def test_financial_segmentation():
    assert split("Revenue was $12.5 million, up 4.2%. Costs were €2.75 million.") == [
        "Revenue was $12.5 million, up 4.2%.", "Costs were €2.75 million."
    ]
    assert split("Dr. Smith met Ms. Jones on Jan. 30, 2025. They approved the report.") == [
        "Dr. Smith met Ms. Jones on Jan. 30, 2025.", "They approved the report."
    ]
    assert split("Example Inc. reported U.S. sales of £3.2 million. Profit rose.") == [
        "Example Inc. reported U.S. sales of £3.2 million.", "Profit rose."
    ]
    assert split('“Revenue rose.” Costs fell! Was cash stable? Yes, it was.') == [
        '“Revenue rose.”', 'Costs fell!', 'Was cash stable?', 'Yes, it was.'
    ]


def test_negation_and_symbols(document):
    text = "We did not lose $2.5m; no jobs were cut, and we never reduced R&D by 5%."
    records = preprocess_document(with_text(document, text), token_counter=fake_counter)
    assert len(records) == 1
    assert records[0]["text"] == text


def test_zero_width_space_preserves_word_boundaries(document):
    raw = "Costs did\u200bnot rise."
    record = preprocess_document(with_text(document, raw), token_counter=fake_counter)[0]
    assert record["text"] == "Costs did not rise."
    assert record["original_text"] == raw


def test_whitespace_artifacts_and_source_spans(document):
    raw = "\ufeff===\r\n• Revenue\t rose  by 4.2%.\r\n\r\nCosts did\nnot rise.\r\n---\r\n"
    records = preprocess_document(with_text(document, raw), token_counter=fake_counter)
    assert [record["text"] for record in records] == [
        "Revenue rose by 4.2%.", "Costs did not rise."
    ]
    assert records[0]["original_text"] == "Revenue\t rose  by 4.2%."
    assert records[1]["original_text"] == "Costs did\nnot rise."
    for record in records:
        start, end = record["source_char_start"], record["source_char_end"]
        assert raw[start:end] == record["original_text"]


def test_stable_ids_metadata_and_determinism(document):
    first = preprocess_document(document, token_counter=fake_counter)
    second = preprocess_document(document, token_counter=fake_counter)
    assert first == second
    assert len(first) == 5
    for index, record in enumerate(first):
        assert record["sentence_index"] == index
        assert record["sentence_id"] == f"synthetic_q1_2025_{index + 1:04d}"
        for key, value in document.metadata.items():
            assert record[key] == value
        assert record["source_sha256"] == document.source_sha256
        assert record["tokenizer_model"] == MODEL_ID
        assert record["tokenizer_revision"] == MODEL_REVISION
        assert not record["exceeds_token_limit"]
        assert not ({"label", "probabilities", "sentiment"} & record.keys())


def test_long_sentence_error_and_explicit_keep(document):
    sentence = "Revenue " + "continued " * 520 + "to grow."
    long_document = with_text(document, sentence)
    with pytest.raises(ValueError, match="exceeding 512"):
        preprocess_document(long_document, token_counter=fake_counter)
    records = preprocess_document(
        long_document, token_counter=fake_counter, long_sentence_policy="keep"
    )
    assert len(records) == 1
    assert records[0]["text"] == sentence
    assert records[0]["original_text"] == sentence
    assert records[0]["exceeds_token_limit"] is True
    assert records[0]["token_count"] == fake_counter(sentence)


def test_token_limit_boundary(document):
    sample = with_text(document, "Revenue rose.")
    assert not preprocess_document(sample, token_counter=lambda _: 512)[0]["exceeds_token_limit"]
    with pytest.raises(ValueError, match="513"):
        preprocess_document(sample, token_counter=lambda _: 513)


@pytest.mark.parametrize("text", ["... 123. 45.6%", "---\n===", " \n\t"])
def test_unusable_document(document, text):
    with pytest.raises(ValueError, match="no usable text"):
        preprocess_document(with_text(document, text), token_counter=fake_counter)


def test_unterminated_sentence_kept(document):
    records = preprocess_document(with_text(document, "No dividend was declared"), token_counter=fake_counter)
    assert records[0]["text"] == "No dividend was declared"


@pytest.mark.parametrize("options", [{"max_tokens": 513}, {"max_tokens": True},
                                         {"long_sentence_policy": "truncate"}])
def test_invalid_options(document, options):
    with pytest.raises(ValueError):
        preprocess_document(document, token_counter=fake_counter, **options)


def test_jsonl_round_trip_and_no_overwrite(document, tmp_path):
    records = preprocess_document(document, token_counter=fake_counter)
    path = tmp_path / "nested" / "sentences.jsonl"
    write_jsonl(records, path)
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == records
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_jsonl(records, path)
    assert path.read_bytes() == original


def test_cli(document, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.preprocessing.load_token_counter", lambda: fake_counter)
    path = tmp_path / "records.jsonl"
    main(["--metadata", str(FIXTURE), "--text", str(FIXTURE.with_suffix(".txt")),
          "--output", str(path)])
    assert "Wrote 5 records" in capsys.readouterr().out
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_cli_length_error_does_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.preprocessing.load_token_counter", lambda: lambda _: 513)
    path = tmp_path / "records.jsonl"
    with pytest.raises(SystemExit) as error:
        main(["--metadata", str(FIXTURE), "--output", str(path)])
    assert error.value.code == 2
    assert "exceeding 512" in capsys.readouterr().err
    assert not path.exists()


def test_cli_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--metadata", str(tmp_path / "missing.json"), "--output", str(tmp_path / "out.jsonl")])
    assert error.value.code == 2
    assert "Cannot read metadata" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_FINBERT_INTEGRATION") != "1", reason="Requires cached tokenizer.")
def test_cached_tokenizer_length_policy(document):
    counter = load_token_counter()
    assert counter("Revenue rose.") > 2  # Counts special tokens too.
    records = preprocess_document(document, token_counter=counter)
    assert len(records) == 5
    long_document = with_text(document, "Revenue " + "grew " * 520 + ".")
    with pytest.raises(ValueError, match="exceeding 512"):
        preprocess_document(long_document, token_counter=counter)
    record = preprocess_document(
        long_document, token_counter=counter, long_sentence_policy="keep"
    )[0]
    assert record["token_count"] == counter(record["text"])
    assert record["exceeds_token_limit"]
